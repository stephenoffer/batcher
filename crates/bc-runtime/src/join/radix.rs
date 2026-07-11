//! Parallel radix partitioning — the scatter pass shared by both radix joins.
//!
//! `radix_join_scalar` (sequential) and `radix_join_scalar_parallel` (broadcast) both
//! begin by gathering each side's non-null rows into cache-sized partitions carrying the
//! key inline as `(key, abs_row)`. That scatter used to be a single serial loop over every
//! build *and* probe row — on a 60M-row TPC-H `lineitem` probe it is the whole join's
//! Amdahl bottleneck, and it left the "parallel" radix join with a fully sequential prefix
//! (measured: `hash_join` scaled only 12.4x across 48 workers, against 19.8x for the
//! group-by aggregate that has no such pass).
//!
//! This module partitions in three parallel phases — histogram, prefix-sum, scatter — which
//! is the textbook parallel radix partition:
//!
//! 1. each row-range chunk counts, independently, how many of its rows land in each
//!    partition;
//! 2. an exclusive prefix sum over `(chunk, partition)` reserves every chunk a disjoint
//!    slice of every partition's output;
//! 3. each chunk writes its rows into those reserved slices, in parallel.
//!
//! **The result is bit-identical to the serial scatter.** Because a chunk's slice of a
//! partition is offset by the counts of all *earlier* chunks, and each chunk walks its rows
//! in increasing index order, every partition ends up holding its rows in ascending
//! `abs_row` order — exactly what the serial `push` loop produced. That matters: the join's
//! output row order (and hence the `seq == par` oracle) depends on it.

use rayon::prelude::*;

/// Rows per chunk of the histogram/scatter passes. Large enough that the per-chunk
/// histogram allocation and rayon task overhead are amortized, small enough that a skewed
/// partition cannot starve the pool of work.
const CHUNK_ROWS: usize = 1 << 16;

/// A `*mut` wrapper asserting `Send`/`Sync` for the disjoint parallel writes in
/// [`partition_side`]. Sound only because each `(chunk, partition)` pair owns a distinct
/// output range; private to this module.
#[derive(Clone, Copy)]
struct SendMutPtr<T>(*mut T);
// SAFETY: see [`partition_side`] — writes through this pointer never alias across threads.
unsafe impl<T> Send for SendMutPtr<T> {}
unsafe impl<T> Sync for SendMutPtr<T> {}

/// Scatter one side's non-null rows into `parts` partitions, in parallel.
///
/// `key(i)` reads row `i`'s join key; `part_of(&k)` maps a key to its partition. `nulls[i]`
/// marks a row whose key is null (never joined, so never emitted). Returns one vector per
/// partition, each holding `(key, abs_row)` in ascending `abs_row` order.
pub(super) fn partition_side<O>(
    key: impl Fn(usize) -> O + Sync,
    nulls: &[bool],
    parts: usize,
    part_of: impl Fn(&O) -> usize + Sync,
) -> Vec<Vec<(O, u32)>>
where
    O: Copy + Send + Sync,
{
    let rows = nulls.len();
    if rows == 0 {
        return vec![Vec::new(); parts];
    }
    let ranges: Vec<std::ops::Range<usize>> = (0..rows)
        .step_by(CHUNK_ROWS)
        .map(|s| s..(s + CHUNK_ROWS).min(rows))
        .collect();

    // Phase 1: per-chunk histograms, computed independently.
    let hists: Vec<Vec<u32>> = ranges
        .par_iter()
        .map(|r| {
            let mut counts = vec![0u32; parts];
            for i in r.clone() {
                if !nulls[i] {
                    counts[part_of(&key(i))] += 1;
                }
            }
            counts
        })
        .collect();

    // Phase 2: exclusive prefix sum. `starts[c][p]` is where chunk `c` writes partition
    // `p`; summing chunks in order is what preserves the serial scatter's row order.
    let mut totals = vec![0usize; parts];
    let mut starts = vec![vec![0usize; parts]; ranges.len()];
    for p in 0..parts {
        let mut running = 0usize;
        for (c, hist) in hists.iter().enumerate() {
            starts[c][p] = running;
            running += hist[p] as usize;
        }
        totals[p] = running;
    }

    // Phase 3: each chunk writes into the slices it reserved. Capacities are exact, so no
    // allocation happens during the scatter (the serial `push` loop reallocated).
    let mut out: Vec<Vec<(O, u32)>> = totals.iter().map(|&n| Vec::with_capacity(n)).collect();
    let heads: Vec<SendMutPtr<(O, u32)>> =
        out.iter_mut().map(|v| SendMutPtr(v.as_mut_ptr())).collect();
    ranges
        .par_iter()
        .zip(starts.par_iter())
        .for_each(|(r, start)| {
            let mut cursor: Vec<usize> = start.clone();
            for i in r.clone() {
                if nulls[i] {
                    continue;
                }
                let k = key(i);
                let p = part_of(&k);
                // SAFETY: `cursor[p]` walks the range `starts[c][p] .. starts[c][p] +
                // hists[c][p]`, which phase 2 reserved for this chunk alone and which lies
                // within partition `p`'s `totals[p]` capacity. No two chunks — and no two
                // partitions — address the same slot, and every slot counted in phase 1 is
                // written exactly once.
                unsafe { heads[p].0.add(cursor[p]).write((k, i as u32)) };
                cursor[p] += 1;
            }
        });
    // SAFETY: phase 3 wrote exactly `totals[p]` elements into partition `p`.
    for (v, &n) in out.iter_mut().zip(&totals) {
        unsafe { v.set_len(n) };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parallel scatter reproduces the serial one exactly — same partitions, same
    /// order within each. The join's output order (and the `seq == par` oracle) rests on it.
    #[test]
    fn matches_the_serial_scatter_exactly() {
        let rows = 100_000;
        let parts = 8;
        let nulls: Vec<bool> = (0..rows).map(|i| i % 17 == 0).collect();
        let key = |i: usize| (i as i64) * 2654435761;
        let part_of = |k: &i64| (k.unsigned_abs() as usize) % parts;

        let mut expected: Vec<Vec<(i64, u32)>> = vec![Vec::new(); parts];
        for (i, &is_null) in nulls.iter().enumerate() {
            if !is_null {
                let k = key(i);
                expected[part_of(&k)].push((k, i as u32));
            }
        }
        assert_eq!(partition_side(key, &nulls, parts, part_of), expected);
    }

    /// Every non-null row lands in exactly one partition; null rows land nowhere.
    #[test]
    fn every_non_null_row_is_placed_once() {
        let nulls = [false, true, false, false];
        let got = partition_side(|i| i as i64, &nulls, 4, |k| (*k as usize) % 4);
        let mut rows: Vec<u32> = got.iter().flatten().map(|&(_, r)| r).collect();
        rows.sort_unstable();
        assert_eq!(rows, vec![0, 2, 3]);
    }

    /// An empty side yields the right number of empty partitions, not a panic.
    #[test]
    fn empty_side_yields_empty_partitions() {
        let got = partition_side(|i| i as i64, &[], 4, |k| (*k as usize) % 4);
        assert_eq!(got.len(), 4);
        assert!(got.iter().all(|p| p.is_empty()));
    }

    /// A chunk boundary must not disturb the ordering: rows straddling `CHUNK_ROWS` still
    /// come out ascending within their partition.
    #[test]
    fn ordering_survives_chunk_boundaries() {
        let rows = CHUNK_ROWS * 2 + 7;
        let nulls = vec![false; rows];
        let got = partition_side(|i| (i % 3) as i64, &nulls, 4, |k| *k as usize);
        for part in &got {
            assert!(part.windows(2).all(|w| w[0].1 < w[1].1), "not ascending");
        }
        assert_eq!(got.iter().map(|p| p.len()).sum::<usize>(), rows);
    }
}
