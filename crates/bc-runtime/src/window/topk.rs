//! `row_number() OVER (PARTITION BY … ORDER BY …) <= k` without ordering the partitions.
//!
//! A `QUALIFY row_number() <= k` — "the three cheapest orders per customer", "the latest
//! reading per sensor" — is one of the shapes analytics is made of, and the plan Batcher builds
//! for it computes far more than it keeps. Kyber's `qualify_to_partition_topn` folds the bound
//! into `Window.rank_limit`, but that bound is applied as a **mask**: [`super::window_with`]
//! ranks every row, which means ordering every partition, and `filter_by_rank_limit` then throws
//! all but `k` of them away. Only `k = 1` escapes, by a rewrite onto `DISTINCT ON`.
//!
//! Ordering a partition to keep two of its rows is the wrong algorithm, and the competitors say
//! so with whole operators: Spark's `WindowGroupLimitExec` and Daft's
//! `window_partition_and_dynamic_frame`. This is that idea.
//!
//! ## Where it is applied, which is the whole difference
//!
//! **Inside [`super::window_serial`], which already runs once per hash bucket.** A first version
//! hooked it above `window_with` instead — at the top, over the whole batch — and that traded
//! the operator's parallelism for the better complexity: `O(n log k)` on one core against
//! `O(n log n)` on ninety-six, measured at **0.24-0.50x**, i.e. two to four times *slower*.
//! `window_with` hash-partitions rows so equal partition keys co-locate and then runs the serial
//! kernel per bucket across rayon; applied here the bounded selection inherits that instead of
//! replacing it, and each worker heaps only the partitions it owns.
//!
//! ## What it does
//!
//! One pass over the bucket's rows, holding a bounded max-heap of `k` entries per partition in a
//! single flat array — no per-partition allocation, which is what makes a
//! million-tiny-partition window expensive. A row is compared against its partition's current
//! worst and discarded when it cannot win, so the work is `O(n log k)` and for the common `k` of
//! 1-10 the `log k` is two or three comparisons.
//!
//! ## Why the answer is identical
//!
//! The ordering path sorts by `(partition keys, order keys, original row index)` — the trailing
//! index makes it a *total* order, which is what makes `row_number`'s choice among peers
//! deterministic and identical between the serial and per-bucket parallel paths. This module
//! compares exactly that key: the heap orders on `(packed order key, row index)`, so the `k` rows
//! it keeps are the `k` smallest under the same total order, ranked `1..=k` in the same sequence.
//!
//! Rows that do not survive are given rank `k + 1` rather than a null or a zero, because the
//! caller's mask is `rank <= k`: a zero would pass it and silently keep every row of every
//! partition. That is the one place this could go quietly wrong, so it is pinned by
//! `a_non_survivor_never_passes_the_mask`.
//!
//! ## Where it declines
//!
//! Anything the packed key does not cover: more than one order key, more than one partition key,
//! a non-numeric or nullable order key, or a `(groups x k)` heap that would be large next to the
//! rows it is selecting from. A decline costs one type check and returns `None`, and the caller
//! runs the ordering path exactly as before — a short-circuit, never a second definition of what
//! the window means.

use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array};
use arrow::compute::SortOptions;

use crate::error::RuntimeError;

/// Heap slots allowed per input row before the bounded selection stops being the cheap option.
///
/// `groups x k` is the state this holds. Past a small multiple of the rows it is selecting from,
/// nearly every row is a survivor and the ordering path the caller would run instead is the
/// honest answer.
const MAX_HEAP_SLOTS_PER_ROW: usize = 2;

/// The rank column for `row_number()` bounded to the top `k` of each partition, or `None` when
/// this path does not apply and the caller should rank the ordinary way.
///
/// Survivors carry `1..=k` in order; every other row carries `k + 1`, which the caller's
/// `rank <= k` mask drops. See the module docs for why the answer matches the ordering path.
pub(super) fn row_number_top_k(
    partition_keys: &[ArrayRef],
    order_keys: &[(ArrayRef, SortOptions)],
    num_rows: usize,
    k: usize,
) -> Result<Option<ArrayRef>, RuntimeError> {
    if k == 0 || num_rows == 0 || order_keys.len() != 1 || partition_keys.len() > 1 {
        return Ok(None);
    }
    let (ord_arr, opts) = &order_keys[0];
    let Some(mut ord) = super::pack_ordered_u64(ord_arr) else {
        return Ok(None);
    };
    // DESC inverts the order-preserving key, so an ascending selection yields descending order —
    // the same transform `try_ordered_partitions_packed` applies. `nulls_first` cannot matter:
    // `pack_ordered_u64` already refused a nullable column.
    if opts.descending {
        for x in &mut ord {
            *x = !*x;
        }
    }

    // Partition ids come from the shared grouper rather than a second packing, so this path
    // admits every partition-key type `GROUP BY` does (strings included) and agrees with the
    // no-ORDER-BY window branch on what a partition *is*, nulls included.
    let (group_ids, num_groups) = if partition_keys.is_empty() {
        (Vec::new(), 1usize)
    } else {
        let (ids, n, _) = crate::agg::assign_groups(partition_keys, num_rows)?;
        (ids, n)
    };
    let Some(slots) = num_groups.checked_mul(k) else {
        return Ok(None);
    };
    if slots > num_rows.saturating_mul(MAX_HEAP_SLOTS_PER_ROW) {
        return Ok(None);
    }

    // One flat `(order key, row)` array, `k` consecutive slots per partition, each region a
    // max-heap so the worst kept row is always at its front.
    let mut heap: Vec<(u64, u32)> = vec![(0, 0); slots];
    let mut len: Vec<u32> = vec![0; num_groups];
    for row in 0..num_rows {
        let g = if group_ids.is_empty() {
            0
        } else {
            group_ids[row] as usize
        };
        let base = g * k;
        let entry = (ord[row], row as u32);
        let n = len[g] as usize;
        if n < k {
            heap[base + n] = entry;
            sift_up(&mut heap[base..base + n + 1], n);
            len[g] = (n + 1) as u32;
        } else if entry < heap[base] {
            heap[base] = entry;
            sift_down(&mut heap[base..base + k]);
        }
    }

    // Everything not selected gets `k + 1` so the caller's `rank <= k` mask drops it.
    let mut out = vec![(k + 1) as i64; num_rows];
    for g in 0..num_groups {
        let n = len[g] as usize;
        let region = &mut heap[g * k..g * k + n];
        region.sort_unstable();
        for (rank0, &(_, row)) in region.iter().enumerate() {
            out[row as usize] = rank0 as i64 + 1;
        }
    }
    Ok(Some(Arc::new(Int64Array::from(out)) as ArrayRef))
}

/// Restore the max-heap property after appending at `idx`.
fn sift_up(region: &mut [(u64, u32)], mut idx: usize) {
    while idx > 0 {
        let parent = (idx - 1) / 2;
        if region[idx] <= region[parent] {
            break;
        }
        region.swap(idx, parent);
        idx = parent;
    }
}

/// Restore the max-heap property after replacing the root.
fn sift_down(region: &mut [(u64, u32)]) {
    let n = region.len();
    let mut idx = 0;
    loop {
        let (l, r) = (2 * idx + 1, 2 * idx + 2);
        let mut largest = idx;
        if l < n && region[l] > region[largest] {
            largest = l;
        }
        if r < n && region[r] > region[largest] {
            largest = r;
        }
        if largest == idx {
            return;
        }
        region.swap(idx, largest);
        idx = largest;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::window::{WindowCall, WindowFn};
    use arrow::array::{Array, Float64Array, Int64Array, StringArray};

    /// The ordering path's answer, masked the way the caller masks it: rank `1..=k` for the rows
    /// that survive `rank <= k`, `k + 1` for everyone else. Comparing against *this* rather than
    /// a hand-written expectation is the point — it is the path the selection replaces, and it
    /// shares no code with it.
    fn oracle(part: &[ArrayRef], ord: &[(ArrayRef, SortOptions)], n: usize, k: usize) -> Vec<i64> {
        let call = WindowCall {
            func: WindowFn::RowNumber,
            values: None,
            offset: 0,
            frame: None,
            alpha: None,
            half_life: None,
        };
        // `rank_limit: None` forces the ordering path even for a shape the bounded one covers.
        let cols =
            super::super::window_serial(part, ord, std::slice::from_ref(&call), n, None).unwrap();
        let ranks = cols[0].as_any().downcast_ref::<Int64Array>().unwrap();
        (0..n)
            .map(|i| {
                let r = ranks.value(i);
                if r <= k as i64 {
                    r
                } else {
                    k as i64 + 1
                }
            })
            .collect()
    }

    /// The whole operator with the bound threaded down, so the parallel bucketing is in play.
    fn via_window(
        part: &[ArrayRef],
        ord: &[(ArrayRef, SortOptions)],
        n: usize,
        k: usize,
        threshold: usize,
    ) -> Vec<i64> {
        let call = WindowCall {
            func: WindowFn::RowNumber,
            values: None,
            offset: 0,
            frame: None,
            alpha: None,
            half_life: None,
        };
        let cols = super::super::window_with(
            part,
            ord,
            std::slice::from_ref(&call),
            n,
            threshold,
            Some(k),
        )
        .unwrap();
        let a = cols[0].as_any().downcast_ref::<Int64Array>().unwrap();
        // Mask exactly as `bc_interp`'s `filter_by_rank_limit` does. The bounded path already
        // marks a non-survivor `k + 1`, and a *declining* shape falls through to the ordering
        // path and returns its full ranks — the mask is what makes the two comparable, and it
        // is the operator's real contract: the rows kept, and their ranks, must agree.
        (0..n)
            .map(|i| {
                let r = a.value(i);
                if r <= k as i64 {
                    r
                } else {
                    k as i64 + 1
                }
            })
            .collect()
    }

    fn asc() -> SortOptions {
        SortOptions {
            descending: false,
            nulls_first: false,
        }
    }
    fn desc() -> SortOptions {
        SortOptions {
            descending: true,
            nulls_first: false,
        }
    }

    /// Deterministic pseudo-random values, so a failure reproduces.
    fn spread(n: usize, m: i64, seed: u64) -> Vec<i64> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((x >> 33) as i64).rem_euclid(m)
            })
            .collect()
    }

    const N: usize = 5_000;

    /// The property that matters: **through the real operator**, at a threshold low enough that
    /// the parallel bucketing runs, every `k` and both directions agree with the ordering path.
    #[test]
    fn the_bounded_path_matches_the_ordering_path_through_the_parallel_operator() {
        let p: ArrayRef = Arc::new(Int64Array::from(spread(N, 300, 7)));
        let o: ArrayRef = Arc::new(Int64Array::from(spread(N, 1_000, 11)));
        for k in [1usize, 2, 3, 10] {
            for opts in [asc(), desc()] {
                let ord = vec![(Arc::clone(&o), opts)];
                let part = vec![Arc::clone(&p)];
                // threshold 1 forces the bucketed parallel path; usize::MAX forces serial.
                for threshold in [1usize, usize::MAX] {
                    assert_eq!(
                        via_window(&part, &ord, N, k, threshold),
                        oracle(&part, &ord, N, k),
                        "k={k} {opts:?} threshold={threshold}"
                    );
                }
            }
        }
    }

    /// Ties on the order key are where a bounded selection can silently disagree: the ordering
    /// path breaks them by original row index, so the heap must too.
    #[test]
    fn ties_are_broken_by_row_index_exactly_as_the_ordering_path_does() {
        let p: ArrayRef = Arc::new(Int64Array::from(spread(N, 40, 3)));
        let o: ArrayRef = Arc::new(Int64Array::from(spread(N, 4, 5)));
        for k in [1usize, 3] {
            let ord = vec![(Arc::clone(&o), asc())];
            let part = vec![Arc::clone(&p)];
            assert_eq!(
                via_window(&part, &ord, N, k, 1),
                oracle(&part, &ord, N, k),
                "k={k}"
            );
        }
    }

    /// A string partition key goes through the shared grouper, not the packed path.
    #[test]
    fn a_string_partition_key_agrees() {
        let vals = spread(N, 25, 13);
        let p: ArrayRef = Arc::new(StringArray::from(
            vals.iter().map(|v| format!("g{v:03}")).collect::<Vec<_>>(),
        ));
        let o: ArrayRef = Arc::new(Float64Array::from(
            spread(N, 900, 17)
                .into_iter()
                .map(|v| v as f64)
                .collect::<Vec<_>>(),
        ));
        let ord = vec![(o, asc())];
        let part = vec![p];
        assert_eq!(via_window(&part, &ord, N, 2, 1), oracle(&part, &ord, N, 2));
    }

    /// `OVER (ORDER BY x)` with no PARTITION BY is one global partition.
    #[test]
    fn an_unpartitioned_window_agrees() {
        let o: ArrayRef = Arc::new(Int64Array::from(spread(N, 5_000, 19)));
        let ord = vec![(o, desc())];
        assert_eq!(via_window(&[], &ord, N, 5, 1), oracle(&[], &ord, N, 5));
    }

    /// The failure this could have shipped: a non-survivor must never pass a `rank <= k` mask.
    /// A zero or a null would, and would silently keep every row of every partition.
    #[test]
    fn a_non_survivor_never_passes_the_mask() {
        let p: ArrayRef = Arc::new(Int64Array::from(spread(1_000, 10, 23)));
        let o: ArrayRef = Arc::new(Int64Array::from(spread(1_000, 500, 29)));
        let k = 2;
        let ranks = via_window(&[p], &[(o, asc())], 1_000, k, usize::MAX);
        assert_eq!(ranks.iter().filter(|&&r| r <= k as i64).count(), 10 * k);
        assert!(
            ranks.iter().all(|&r| r >= 1),
            "no row may carry 0 or a negative rank"
        );
    }

    /// The declines, so an unsupported shape reaches the ordering path rather than a wrong
    /// answer: a nullable order key, two order keys, two partition keys, `k = 0`, and a heap
    /// that would be large next to the rows it selects from.
    #[test]
    fn unsupported_shapes_decline() {
        let n = 100;
        let p: ArrayRef = Arc::new(Int64Array::from(spread(n, 5, 31)));
        let o: ArrayRef = Arc::new(Int64Array::from(spread(n, 50, 37)));
        let nullable: ArrayRef = Arc::new(Int64Array::from(
            (0..n)
                .map(|i| (i % 7 != 0).then_some(i as i64))
                .collect::<Vec<_>>(),
        ));
        let ord = vec![(Arc::clone(&o), asc())];
        let decl = |part: &[ArrayRef], ord: &[(ArrayRef, SortOptions)], k| {
            row_number_top_k(part, ord, n, k).unwrap().is_none()
        };
        assert!(decl(&[Arc::clone(&p)], &[(nullable, asc())], 2));
        assert!(decl(
            &[Arc::clone(&p)],
            &[(Arc::clone(&o), asc()), (Arc::clone(&o), asc())],
            2
        ));
        assert!(decl(&[Arc::clone(&p), Arc::clone(&p)], &ord, 2));
        assert!(decl(&[Arc::clone(&p)], &ord, 0));
        // 100 partitions of one row each, k = 3 would ask for 300 slots over 100 rows.
        let uniq: ArrayRef = Arc::new(Int64Array::from((0..n as i64).collect::<Vec<_>>()));
        assert!(decl(&[uniq], &ord, 3));
    }

    /// A declining shape must still produce the ordering path's answer through the operator.
    #[test]
    fn a_declining_shape_still_matches_the_ordering_path() {
        let p: ArrayRef = Arc::new(Int64Array::from(spread(N, 30, 41)));
        // Two order keys: outside the packed key, so the bounded path declines.
        let o1: ArrayRef = Arc::new(Int64Array::from(spread(N, 50, 43)));
        let o2: ArrayRef = Arc::new(Int64Array::from(spread(N, 7, 47)));
        let ord = vec![(o1, asc()), (o2, asc())];
        let part = vec![p];
        assert_eq!(via_window(&part, &ord, N, 3, 1), oracle(&part, &ord, N, 3));
    }
}
