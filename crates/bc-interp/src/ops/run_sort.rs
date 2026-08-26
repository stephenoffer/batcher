//! Natural-run detection for the fixed-width sort permutations.
//!
//! A sort's input is often already partly in order, and an engine that cannot see that pays
//! the same eight radix passes for `ORDER BY ts` over an append-only log as it does for a
//! random permutation. The shapes are ordinary rather than contrived: micro-batches appended
//! in arrival order, a lakehouse table whose files are each written sorted, a `UNION ALL` of
//! sorted sources, a re-sort by the key a scan is already clustered on, and the merge of the
//! per-partition results a distributed range sort produces.
//!
//! [`radix_sort::is_ordered`] already answers the *degenerate* version of that question — the
//! whole column in order, permutation is the identity — and it is all-or-nothing: one
//! out-of-place row and the full radix runs. This module answers the general one. It finds the
//! maximal ordered **runs**, sorts only what lies between them, and merges.
//!
//! ## Detection is nearly free on data that has no runs
//!
//! That is the property the whole module rests on, and it comes from DuckDB, which sorts with
//! `vergesort(begin, end, less, fallback = ska_sort)`
//! (`src/common/sort/sorted_run.cpp:269`, `third_party/vergesort/vergesort.h:181`). Vergesort
//! does not walk the input looking for runs — it **strides**. It jumps `n / log2(n)` positions
//! ahead, asks which way the pair there is ordered, and only then expands outward to the run's
//! true limits. A run shorter than the stride is never worth exploiting, so failing to find one
//! costs the two comparisons the expansion needs before it stops. Over the whole input that is
//! about `3 * log2(n)` comparisons — 60-odd on six million rows — against the ~48 M scattered
//! writes the radix it might replace performs.
//!
//! So the decline is free and the win is large, which is what makes this safe to try
//! unconditionally rather than gate on a plan-level `sorted_by` declaration. It *proves* the
//! ordering on the rows in hand instead of trusting a claim about them — the same reasoning
//! `bc_runtime::agg::group::runs` uses for grouping, and for the same reason: a declaration
//! that turns out to be false is a wrong answer, while a proof that finds nothing is a
//! rounding error.
//!
//! ## Where this departs from vergesort, and why it has to
//!
//! Vergesort feeds `std::sort`, which is not stable, so it reverses any **non-ascending** run.
//! Batcher's sort is stable by contract — ties resolve to input order, and the sequential
//! oracle, the parallel sample-sort and the external merge sort have to agree bit-for-bit on
//! them (`.claude/rules/rust-engine.md`). Reversing a run that contains two equal keys would
//! put the later row first. So a descending run is exploited only when it is **strictly**
//! descending, which makes its reversal tie-free and therefore stable. Ascending runs are
//! taken non-strictly, since they need no reversal.
//!
//! The merge is pairwise rather than a k-way heap: runs are merged two at a time in
//! `log2(k)` rounds, each round's merges independent and run in parallel. Two sorted streams
//! merge with sequential reads and a sequential write, which is the access pattern the radix
//! it replaces does not have.

use rayon::prelude::*;

/// Rows below which run detection is not attempted.
///
/// Not a cost argument — detection is `O(log n)` comparisons — but a value one. Below a few
/// thousand rows the radix is already cache-resident and finishes in microseconds, so there is
/// no win to collect, and the per-range sorts of the parallel sample-sort call this often
/// enough that an unprofitable branch is worth skipping outright.
const MIN_ROWS: usize = 1 << 12;

/// Runs above which merging costs more than radix-sorting from scratch.
///
/// The merge is `n * log2(k)` element moves; the radix is at most eight passes of `n` reads and
/// `n` scattered writes. At `k = 64` the merge does six sequential passes against the radix's
/// eight scattered ones, which is where the two stop being clearly separable — so this is the
/// point at which the answer stops being obvious rather than the point at which it inverts.
const MAX_RUNS: usize = 64;

/// A maximal ordered stretch of the index list, as offsets into it.
///
/// `reversed` marks a strictly-descending run, whose indices are emitted back-to-front.
/// `ordered` is false for a stretch that is not a run at all — it is sorted by the fallback
/// and then merged like any other stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Run {
    start: usize,
    end: usize,
    reversed: bool,
    ordered: bool,
}

/// The stable permutation of `idx` ordering it by `keys[i]`, or `None` when the input has too
/// little run structure for merging to beat `fallback`.
///
/// `keys` is indexed by the *values* in `idx`, and `descending` inverts the comparison exactly
/// as the radix does — so the run structure this looks for is the run structure of the
/// requested order, and a descending sort of ascending data finds one strictly-descending run
/// rather than none.
///
/// `fallback` sorts one stretch of `idx` and must produce the stable permutation of it; the
/// radix is the caller that supplies it. It is called on the stretches between runs, so a fully
/// unstructured input reaches it exactly once over the whole index list — which is what makes
/// declining cost nothing but the detection scan.
pub(super) fn run_aware_sort(
    idx: &[u32],
    keys: &[u64],
    descending: bool,
    fallback: impl Fn(Vec<u32>) -> Vec<u32> + Sync,
) -> Option<Vec<u32>> {
    let n = idx.len();
    if n < MIN_ROWS {
        return None;
    }
    // One ascending order to reason about: `descending` is folded into the key here, exactly as
    // `lsd_radix` folds it, so everything below compares `<=` and never re-asks the direction.
    let key_of = |row: u32| -> u64 {
        let k = keys[row as usize];
        if descending {
            !k
        } else {
            k
        }
    };
    let runs = detect_runs(n, &|pos: usize| key_of(idx[pos]))?;
    // A single ascending run covering everything is the identity. `is_ordered` normally settles
    // that before this is reached; it still holds here for a key with nulls, where that check
    // does not apply.
    if runs.len() == 1 && runs[0].ordered && !runs[0].reversed {
        return Some(idx.to_vec());
    }
    Some(merge_runs(idx, &key_of, &runs, &fallback))
}

/// The maximal ordered runs of `0..n`, or `None` when there are too many of them, or too few
/// rows inside them, for merging to pay.
fn detect_runs(n: usize, key: &impl Fn(usize) -> u64) -> Option<Vec<Run>> {
    // Vergesort's `unstable_limit`: a run shorter than this is not worth a merge stream, and it
    // is also the stride, so the two decisions stay consistent by construction.
    let min_run = (n / (usize::BITS - n.leading_zeros()) as usize).max(2);
    let mut runs: Vec<Run> = Vec::new();
    let mut unstable: Option<usize> = None;
    let mut begin = 0usize;
    let mut ordered_rows = 0usize;

    loop {
        // Too little left to hold a qualifying run; whatever remains is unstable.
        if n.saturating_sub(begin + 1) <= min_run {
            unstable.get_or_insert(begin);
            break;
        }
        // The stride. `probe + 1 < n` holds because of the guard above.
        let probe = begin + min_run;
        let ascending = key(probe) <= key(probe + 1);
        let mut lo = probe;
        let mut hi = probe + 1;
        if ascending {
            while lo > begin && key(lo - 1) <= key(lo) {
                lo -= 1;
            }
            while hi + 1 < n && key(hi) <= key(hi + 1) {
                hi += 1;
            }
        } else {
            // Strictly descending only — see the module header on why reversing a run with
            // ties would break stability.
            while lo > begin && key(lo - 1) > key(lo) {
                lo -= 1;
            }
            while hi + 1 < n && key(hi) > key(hi + 1) {
                hi += 1;
            }
        }
        if hi + 1 - lo >= min_run {
            if lo > begin {
                unstable.get_or_insert(begin);
            }
            if let Some(u) = unstable.take() {
                runs.push(Run {
                    start: u,
                    end: lo,
                    reversed: false,
                    ordered: false,
                });
            }
            ordered_rows += hi + 1 - lo;
            runs.push(Run {
                start: lo,
                end: hi + 1,
                reversed: !ascending,
                ordered: true,
            });
            begin = hi + 1;
            if begin >= n {
                break;
            }
        } else {
            unstable.get_or_insert(begin);
            begin = probe;
        }
        if runs.len() > MAX_RUNS {
            return None;
        }
    }
    if let Some(u) = unstable {
        runs.push(Run {
            start: u,
            end: n,
            reversed: false,
            ordered: false,
        });
    }
    if runs.len() > MAX_RUNS {
        return None;
    }
    // Merging pays for itself out of the rows it does *not* have to sort. When the ordered runs
    // cover less than half the input the fallback still sorts most of it and the merge is added
    // on top, so the whole thing becomes a pessimization — decline instead and let the caller
    // radix the lot in one pass.
    if ordered_rows * 2 < n {
        return None;
    }
    Some(runs)
}

/// Materialize each run as a sorted stream of row indices, then merge the streams pairwise.
///
/// Each round's merges are independent, so they run in parallel; `log2(k)` rounds reduce `k`
/// streams to one. Every merge reads two streams forward and writes one forward, which is the
/// access pattern that makes this cheaper than the scattered radix even when the run count is
/// high enough that the comparison counts are similar.
fn merge_runs(
    idx: &[u32],
    key_of: &(impl Fn(u32) -> u64 + Sync),
    runs: &[Run],
    fallback: &(impl Fn(Vec<u32>) -> Vec<u32> + Sync),
) -> Vec<u32> {
    let mut streams: Vec<Vec<u32>> = runs
        .par_iter()
        .map(|r| {
            let rows = &idx[r.start..r.end];
            if !r.ordered {
                fallback(rows.to_vec())
            } else if r.reversed {
                rows.iter().rev().copied().collect()
            } else {
                rows.to_vec()
            }
        })
        .collect();

    while streams.len() > 1 {
        streams = streams
            .par_chunks(2)
            .map(|pair| match pair {
                [a, b] => merge_two(a, b, key_of),
                [a] => a.clone(),
                _ => unreachable!("par_chunks(2) yields one or two"),
            })
            .collect();
    }
    streams.pop().unwrap_or_default()
}

/// Merge two ascending streams of row indices, preferring `a` on a tie.
///
/// `a` holds rows that all precede `b`'s in the input, so taking `a` first on equal keys is what
/// makes the merge stable — the same rule a two-way merge sort uses. It is the only place
/// stability is decided, because every stream reaching here is itself already stable: an
/// ascending run keeps input order, a strictly-descending one has no ties to keep, and an
/// unstable stretch was sorted by the stable fallback.
fn merge_two(a: &[u32], b: &[u32], key_of: &impl Fn(u32) -> u64) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        if key_of(a[i]) <= key_of(b[j]) {
            out.push(a[i]);
            i += 1;
        } else {
            out.push(b[j]);
            j += 1;
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stable permutation, computed the obvious way, as the oracle every case is held to.
    ///
    /// Deliberately *not* the radix: the claim is that run-merging produces the stable
    /// permutation, and pinning it against another optimized implementation would let a shared
    /// misunderstanding pass. A sort by `(key, position)` is the definition.
    fn oracle(keys: &[u64], idx: &[u32], descending: bool) -> Vec<u32> {
        let mut out = idx.to_vec();
        out.sort_by_key(|&row| {
            let k = keys[row as usize];
            (if descending { !k } else { k }, row)
        });
        out
    }

    /// A stable reference sorter standing in for the radix the real caller passes.
    fn stable_fallback(
        keys: &[u64],
        descending: bool,
    ) -> impl Fn(Vec<u32>) -> Vec<u32> + Sync + '_ {
        move |part: Vec<u32>| oracle(keys, &part, descending)
    }

    fn check(keys: &[u64], descending: bool) -> Option<Vec<u32>> {
        let idx: Vec<u32> = (0..keys.len() as u32).collect();
        let got = run_aware_sort(&idx, keys, descending, stable_fallback(keys, descending));
        if let Some(ref g) = got {
            assert_eq!(
                *g,
                oracle(keys, &idx, descending),
                "permutation differs from the oracle"
            );
        }
        got
    }

    /// A deterministic pseudo-random `u64` stream — no rand dependency in the data plane.
    fn noise(n: usize, seed: u64) -> Vec<u64> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x
            })
            .collect()
    }

    const N: usize = 1 << 16;

    #[test]
    fn a_fully_ascending_key_is_the_identity() {
        let keys: Vec<u64> = (0..N as u64).collect();
        assert_eq!(
            check(&keys, false).expect("runs found"),
            (0..N as u32).collect::<Vec<_>>()
        );
    }

    /// The reversal a strictly-descending run takes is the whole reason descending runs are
    /// admitted, so it is pinned separately from the general agreement check.
    #[test]
    fn a_strictly_descending_key_reverses() {
        let keys: Vec<u64> = (0..N as u64).rev().collect();
        assert_eq!(
            check(&keys, false).expect("runs found"),
            (0..N as u32).rev().collect::<Vec<_>>()
        );
    }

    /// The stability trap. A *non-strictly* descending run cannot be reversed, because the two
    /// rows sharing a key would come out later-first. Detection must not treat it as a run —
    /// and if it ever does, this is the test that catches it, because the oracle keeps input
    /// order on the tie and a reversal does not.
    #[test]
    fn a_descending_key_with_ties_still_sorts_stably() {
        let keys: Vec<u64> = (0..N as u64).rev().map(|v| v / 3).collect();
        check(&keys, false);
    }

    #[test]
    fn concatenated_sorted_runs_merge_to_the_stable_permutation() {
        for parts in [2usize, 3, 8, 16] {
            let per = N / parts;
            let mut keys = Vec::with_capacity(N);
            for p in 0..parts {
                let mut chunk = noise(per, 0x9E37 + p as u64);
                chunk.sort_unstable();
                keys.extend(chunk);
            }
            assert!(
                check(&keys, false).is_some(),
                "{parts} sorted runs should be exploited"
            );
            assert!(
                check(&keys, true).is_some(),
                "{parts} runs, descending request"
            );
        }
    }

    /// Sorted with late arrivals sprinkled in — the append-only-log shape. The runs between the
    /// out-of-order rows are still long, so this must be taken rather than declined.
    #[test]
    fn a_mostly_sorted_key_with_noise_is_still_exploited() {
        let mut keys: Vec<u64> = noise(N, 7);
        keys.sort_unstable();
        for (i, k) in keys.iter_mut().enumerate() {
            if i % 4096 == 17 {
                *k = 0;
            }
        }
        assert!(
            check(&keys, false).is_some(),
            "a mostly-sorted key should be exploited"
        );
    }

    /// The decline path, and the property that makes the whole module safe: unstructured input
    /// must be handed back rather than merged, so the caller's radix runs exactly as before.
    #[test]
    fn an_unstructured_key_declines() {
        assert!(
            check(&noise(N, 3), false).is_none(),
            "random input should decline"
        );
    }

    #[test]
    fn a_constant_key_is_one_run_and_the_identity() {
        let keys = vec![42u64; N];
        assert_eq!(
            check(&keys, false).expect("constant is a run"),
            (0..N as u32).collect::<Vec<_>>()
        );
        assert_eq!(
            check(&keys, true).expect("constant is a run"),
            (0..N as u32).collect::<Vec<_>>()
        );
    }

    /// Below the floor nothing is attempted, whatever the structure.
    #[test]
    fn a_short_input_declines_without_looking() {
        let keys: Vec<u64> = (0..(MIN_ROWS as u64 - 1)).collect();
        assert!(check(&keys, false).is_none());
    }

    /// A subset index list — the shape `radix_sort_indices` passes when the column has nulls.
    /// The runs are runs of the *live* rows, and the permutation must still be theirs.
    #[test]
    fn a_sparse_index_list_sorts_by_its_own_runs() {
        let mut keys: Vec<u64> = noise(2 * N, 11);
        let idx: Vec<u32> = (0..2 * N as u32).filter(|i| i % 2 == 0).collect();
        // Make the *live* rows ordered while the whole column is not.
        let mut live: Vec<u64> = idx.iter().map(|&i| keys[i as usize]).collect();
        live.sort_unstable();
        for (slot, v) in idx.iter().zip(live) {
            keys[*slot as usize] = v;
        }
        let got = run_aware_sort(&idx, &keys, false, stable_fallback(&keys, false))
            .expect("the live rows are ordered");
        assert_eq!(got, oracle(&keys, &idx, false));
    }

    /// Every run count from "one" to past the cap, ascending and descending, against the
    /// oracle — the agreement claim stated as a sweep rather than as three examples.
    #[test]
    fn the_permutation_matches_the_oracle_across_run_counts_and_directions() {
        for parts in [1usize, 2, 5, 17, 64, 65, 200] {
            let per = (N / parts).max(1);
            let mut keys = Vec::with_capacity(N);
            for p in 0..parts {
                let mut chunk = noise(per, 1 + p as u64);
                chunk.sort_unstable();
                keys.extend(chunk);
            }
            for descending in [false, true] {
                check(&keys, descending);
            }
        }
    }
}
