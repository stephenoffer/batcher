//! The flat hash join's probe, across cores, emitting exactly what the serial probe emits.
//!
//! [`super::JoinTable::build`] already shards its build across every core past one morsel. The
//! probe beside it did not: `build_probe_flat` walked every probe row on one thread, so any join
//! that reaches the flat table rather than the radix join — every `Right` and `Full` join, a build
//! under [`super::RADIX_MIN_BUILD_ROWS`], a key of three or more columns, a long string key, a
//! row-encoded key — spent its probe on a single core. The streaming executor's materialized
//! fallback runs exactly this join, and cannot switch to a parallel join that reorders its output,
//! because its contract is the oracle's row order: TPC-H sf1 `orders FULL JOIN lineitem` ran at 8%
//! CPU utilization for that reason.
//!
//! **The serial order is reproducible, so the probe can be split without changing it.** The
//! serial probe emits probe-major: probe row `l`'s pairs, in chain order, before row `l + 1`'s.
//! Contiguous probe ranges probed independently against the same table and concatenated in range
//! order are therefore the identical sequence. A `Right` or `Full` join then appends the build
//! rows nothing matched, ascending; "matched" is read back off the ranges' own right indices, the
//! same set of rows the serial loop marks as it goes, and the remainder follows in the same order.
//! The relation and its row order are both unchanged.

use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

use rayon::prelude::*;

use super::{IndexBuf, JoinIndices, JoinKeys, JoinTable, JoinType, NULL_INDEX};

/// Probe rows below which the serial probe is kept: the fan-out and the concatenation cost more
/// than a probe this small takes on one core.
const PARALLEL_PROBE_MIN_ROWS: usize = 1 << 16;

/// The contiguous probe ranges to run in parallel, or `None` to probe serially.
pub(super) fn probe_ranges(probe_rows: usize) -> Option<Vec<std::ops::Range<usize>>> {
    let threads = rayon::current_num_threads();
    if probe_rows < PARALLEL_PROBE_MIN_ROWS || threads < 2 {
        return None;
    }
    // At least a morsel a range: smaller ranges are scheduling overhead, not parallelism.
    let chunk = probe_rows
        .div_ceil(threads)
        .max(bc_arrow::DEFAULT_MORSEL_ROWS);
    Some(
        (0..probe_rows)
            .step_by(chunk)
            .map(|s| s..(s + chunk).min(probe_rows))
            .collect(),
    )
}

/// Probe `ranges` in parallel and assemble the serial probe's output — see the module note.
pub(super) fn probe_in_order<K: JoinKeys + Sync>(
    table: &JoinTable,
    keys: &K,
    ranges: &[std::ops::Range<usize>],
    left_null: &[bool],
    right_rows: usize,
    join_type: JoinType,
) -> JoinIndices {
    let pieces: Vec<(IndexBuf, IndexBuf)> = ranges
        .par_iter()
        .map(|r| {
            let mut left_out = IndexBuf::with_capacity(r.len());
            let mut right_out = IndexBuf::with_capacity(r.len());
            table.probe_range(
                keys,
                r.clone(),
                Some(left_null),
                join_type,
                &mut left_out,
                &mut right_out,
                None,
            );
            (left_out, right_out)
        })
        .collect();

    let unmatched = matches!(join_type, JoinType::Right | JoinType::Full)
        .then(|| unmatched_build_rows(&pieces, right_rows));

    // The unmatched remainder is one more piece after the ranges, so the concatenation below
    // places it last — where the serial probe appends it.
    let mut pieces = pieces;
    if let Some(rows) = unmatched.filter(|u| !u.is_empty()) {
        let mut left_out = IndexBuf::with_capacity(rows.len());
        let mut right_out = IndexBuf::with_capacity(rows.len());
        for r in rows {
            left_out.push_null();
            right_out.push(r);
        }
        pieces.push((left_out, right_out));
    }
    let (lefts, rights): (Vec<&IndexBuf>, Vec<&IndexBuf>) =
        pieces.iter().map(|(l, r)| (l, r)).unzip();
    JoinIndices::from_bufs(IndexBuf::concat(&lefts), IndexBuf::concat(&rights))
}

/// The build rows no range's output references, ascending. Null-key build rows match nothing, so
/// they are included, as the serial `right_matched` pass includes them.
fn unmatched_build_rows(pieces: &[(IndexBuf, IndexBuf)], right_rows: usize) -> Vec<u32> {
    let matched: Vec<AtomicBool> = (0..right_rows).map(|_| AtomicBool::new(false)).collect();
    pieces.par_iter().for_each(|(_, right)| {
        for &r in right.as_slice() {
            if r != NULL_INDEX {
                matched[r as usize].store(true, Relaxed);
            }
        }
    });
    matched
        .iter()
        .enumerate()
        .filter(|(_, m)| !m.load(Relaxed))
        .map(|(r, _)| r as u32)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Array, ArrayRef, Int64Array, StringArray, UInt32Array};

    use crate::join::{hash_join_indices, JoinIndices, JoinType};

    fn seq(idx: &JoinIndices) -> Vec<(Option<u32>, Option<u32>)> {
        let at = |a: &UInt32Array, i: usize| a.is_valid(i).then(|| a.value(i));
        (0..idx.left.len())
            .map(|i| (at(&idx.left, i), at(&idx.right, i)))
            .collect()
    }

    /// `finish` reuses its buffer rather than collecting `Option`s; the array it builds must be
    /// the one the collect built, bit for bit — including the `0` under every null slot, since a
    /// consumer is free to read a value before checking validity. `concat` must keep order and
    /// the NULL flag across its parts.
    #[test]
    fn finish_and_concat_build_the_arrays_the_serial_forms_built() {
        use crate::join::IndexBuf;
        let rows: Vec<Option<u32>> = (0..10_000u32)
            .map(|i| (i % 7 != 3).then_some(i * 3))
            .collect();
        let mut parts = [
            IndexBuf::default(),
            IndexBuf::default(),
            IndexBuf::default(),
        ];
        for (i, r) in rows.iter().enumerate() {
            let p = &mut parts[i * 3 / rows.len()];
            match r {
                Some(v) => p.push(*v),
                None => p.push_null(),
            }
        }
        let refs: Vec<&IndexBuf> = parts.iter().collect();
        let got = IndexBuf::concat(&refs).finish();
        let want = UInt32Array::from(rows);
        assert_eq!(got.values(), want.values());
        assert_eq!(got.nulls(), want.nulls());
        let mut clean = IndexBuf::default();
        clean.push(5);
        assert!(IndexBuf::concat(&[&clean]).finish().nulls().is_none());
    }

    fn in_pool(threads: usize, f: impl FnOnce() -> JoinIndices + Send) -> JoinIndices {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(f)
    }

    /// The parallel probe must emit the serial probe's rows **in the serial probe's order**, which
    /// is what the streaming executor's materialized join depends on — so the comparison is of
    /// the exact sequence, not a sorted multiset. One pool of one thread takes the serial path and
    /// one of eight the ranged one, over a probe well past the parallel floor, for every join type,
    /// with duplicates and nulls on both sides, unmatched rows on both sides, and both an `Int64`
    /// key under the radix threshold and a string key the radix join never takes.
    #[test]
    fn the_ranged_probe_emits_the_serial_probe_in_the_same_order() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let probe_rows = 3 * super::PARALLEL_PROBE_MIN_ROWS + 123;
        let left_i: Vec<Option<i64>> = (0..probe_rows)
            .map(|_| (next() % 29 != 0).then(|| (next() % 40_000) as i64))
            .collect();
        let right_i: Vec<Option<i64>> = (0..20_000)
            .map(|_| (next() % 31 != 0).then(|| (next() % 30_000) as i64))
            .collect();
        let as_str = |v: &[Option<i64>]| -> ArrayRef {
            Arc::new(StringArray::from(
                v.iter()
                    .map(|x| x.map(|k| format!("key-{k}")))
                    .collect::<Vec<_>>(),
            ))
        };
        let cases: Vec<(ArrayRef, ArrayRef)> = vec![
            (
                Arc::new(Int64Array::from(left_i.clone())),
                Arc::new(Int64Array::from(right_i.clone())),
            ),
            (as_str(&left_i), as_str(&right_i)),
        ];
        for (l, r) in cases {
            for jt in [
                JoinType::Inner,
                JoinType::Left,
                JoinType::Right,
                JoinType::Full,
                JoinType::Semi,
                JoinType::Anti,
            ] {
                let (lk, rk) = (vec![l.clone()], vec![r.clone()]);
                let serial = in_pool(1, || hash_join_indices(&lk, &rk, jt).unwrap());
                let ranged = in_pool(8, || hash_join_indices(&lk, &rk, jt).unwrap());
                assert_eq!(
                    seq(&ranged),
                    seq(&serial),
                    "{jt:?} over {:?}",
                    l.data_type()
                );
            }
        }
    }
}
