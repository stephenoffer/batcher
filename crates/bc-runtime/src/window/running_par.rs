//! Parallel prefix scan for *running* (frameless, ordered) window aggregates.
//!
//! A running aggregate over one ordered partition is a prefix scan: row `i` gets the fold of
//! rows `0..=i`. [`super::window_serial`] walks that partition on one thread, which is right
//! when a query has many partitions — [`super::parallel::window_parallel`] already puts each
//! partition on its own core, so the inner walk should stay serial.
//!
//! It is wrong when a query has *one* partition. A **global** window (no `PARTITION BY`) is
//! exactly that: `window_parallel` requires partition keys to bucket on and hands the whole
//! relation back to the serial kernel, so `SUM(x) OVER (ORDER BY t)` walked 8 M rows on a
//! single core no matter how many were available — measured flat at ~1.6x from 1 to 32
//! threads while every partitioned window scaled ~12x. The same happens to a *skewed*
//! partitioned window, where `window_parallel` deliberately bails to the serial kernel
//! because one bucket holds most of the rows.
//!
//! This module supplies the missing seam for both, in the standard two-pass form: cut the
//! partition into chunks, fold each chunk independently, exclusive-scan the chunk totals, then
//! re-walk each chunk seeded with its prefix. Both walks run across cores; only the scan of
//! `p` chunk totals is serial.
//!
//! # Why only some functions
//!
//! The second pass computes row `i` as `combine(prefix_of_its_chunk, fold_within_chunk)`
//! instead of folding every row from the partition's start, so the fold is **re-associated**.
//! That is invisible for exact arithmetic (integer `+`, `min`, `max`, counting) and visible
//! for floating-point `+`, where it changes the low bits. `bc-interp`'s parallel path must
//! compute *exactly* what the sequential oracle computes (`.claude/rules/rust-engine.md`), so
//! float `SUM`/`AVG` are deliberately **not** routed here and keep the serial walk. The
//! distributed global window (`dist/global_window/offsets.py`) does re-associate them, under
//! the float-reassociation tolerance stated for cross-partition results; that tolerance is a
//! property of a partition count the user chose, and silently extending it to a thread count
//! they did not would make the same query on the same machine answer differently.
//!
//! Everything here is therefore bit-identical to [`super::window_serial`], not merely close.

use rayon::prelude::*;

use crate::error::RuntimeError;

/// Below this many rows a partition is left to the serial walk: the chunking, the extra fold
/// pass and the scatter cost more than the walk they replace.
pub(crate) const MIN_ROWS_TO_SPLIT: usize = 1 << 16;

/// Whether `part` is worth splitting across `threads` cores.
pub(crate) fn worth_splitting(part_len: usize, threads: usize) -> bool {
    threads > 1 && part_len >= MIN_ROWS_TO_SPLIT
}

/// Cut positions `0..part_len` into at most `threads` ranges, each ending on a peer boundary.
///
/// A peer group (rows equal on the ORDER BY keys) must never straddle two chunks: every row of
/// it shares one emitted value, and a chunk that held only part of the group would emit a
/// value computed from a prefix that stops mid-group. Each cut is therefore pushed forward to
/// the next boundary, which is what makes the chunked walk agree with the serial one.
pub(crate) fn peer_chunks(
    part_len: usize,
    threads: usize,
    boundary: impl Fn(usize) -> bool,
) -> Vec<(usize, usize)> {
    let target = part_len.div_ceil(threads).max(1);
    let mut chunks = Vec::with_capacity(threads);
    let mut start = 0usize;
    while start < part_len {
        let mut end = (start + target).min(part_len);
        // Advance to the end of the peer group the cut landed inside.
        while end < part_len && !boundary(end - 1) {
            end += 1;
        }
        chunks.push((start, end));
        start = end;
    }
    chunks
}

/// Which positions of an ordered partition start a new peer group, computed once across cores.
///
/// Every consumer here needs this same predicate at least twice -- `peer_chunks` to place the
/// cuts, then the scan to close each group -- and a ranking fold needs it a third time to
/// decide whether the row begins a group. Recomputing it means re-running `same`, which for
/// the row-encoded order keys is two random reads into the encoded buffer plus a `memcmp`;
/// profiling a global `RANK()` put **37% of all cycles** in that one comparison. Materializing
/// the answer turns every later use into a byte load, and the one pass that does the
/// comparisons runs in parallel.
pub(crate) fn peer_group_starts(
    part: &[usize],
    same: impl Fn(usize, usize) -> bool + Sync,
) -> Vec<bool> {
    (0..part.len())
        .into_par_iter()
        .map(|pos| pos == 0 || !same(part[pos - 1], part[pos]))
        .collect()
}

/// The peer-*boundary* view of [`peer_group_starts`]: `pos` ends its group when it is the last
/// position or the next one starts a new group.
pub(crate) fn boundary_from_starts(starts: &[bool]) -> impl Fn(usize) -> bool + Sync + '_ {
    move |pos: usize| pos + 1 == starts.len() || starts[pos + 1]
}

/// Run a two-pass parallel prefix scan over one ordered partition.
///
/// `zero` must be an identity for `combine`, and `combine` must be associative and agree with
/// `fold`: folding a chunk from `zero` and then combining it onto the running prefix must give
/// the same accumulator as folding every row from the partition's start. That is what makes
/// the result identical to the serial walk rather than merely equivalent.
///
/// `emit` is called once per peer group with that group's accumulator, and its value is
/// written to every row of the group. Returns the per-row outputs in `part` order; the caller
/// scatters them to original row positions.
pub(crate) fn scan<A, T>(
    part_len: usize,
    chunks: &[(usize, usize)],
    zero: A,
    fold: impl Fn(A, usize) -> Result<A, RuntimeError> + Sync,
    combine: impl Fn(A, A) -> A + Sync,
    emit: impl Fn(A) -> Result<T, RuntimeError> + Sync,
    boundary: impl Fn(usize) -> bool + Sync,
) -> Result<Vec<T>, RuntimeError>
where
    A: Copy + Send + Sync,
    T: Copy + Send + Sync + Default,
{
    // Pass 1: each chunk's own total, folded from the identity, across cores.
    let totals: Vec<A> = chunks
        .par_iter()
        .map(|&(s, e)| (s..e).try_fold(zero, &fold))
        .collect::<Result<_, _>>()?;

    // The only serial step: exclusive scan of `chunks.len()` accumulators.
    let mut prefixes = Vec::with_capacity(chunks.len());
    let mut running = zero;
    for total in &totals {
        prefixes.push(running);
        running = combine(running, *total);
    }

    // Pass 2: re-walk each chunk seeded with the prefix of everything before it.
    let per_chunk: Vec<Vec<T>> = chunks
        .par_iter()
        .zip(prefixes)
        .map(|(&(s, e), prefix)| -> Result<Vec<T>, RuntimeError> {
            let mut vals = vec![T::default(); e - s];
            let mut acc = prefix;
            let mut group_start = s;
            for pos in s..e {
                acc = fold(acc, pos)?;
                if boundary(pos) {
                    let v = emit(acc)?;
                    vals[group_start - s..=pos - s].fill(v);
                    group_start = pos + 1;
                }
            }
            Ok(vals)
        })
        .collect::<Result<_, _>>()?;

    let mut out = Vec::with_capacity(part_len);
    for vals in per_chunk {
        out.extend_from_slice(&vals);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{ArrayRef, Float64Array, Int64Array};
    use arrow::compute::SortOptions;

    use super::{peer_chunks, worth_splitting, MIN_ROWS_TO_SPLIT};
    use crate::window::{window_serial, WindowCall, WindowFn};

    /// Enough rows to clear `MIN_ROWS_TO_SPLIT`, with deliberate ties on the ORDER BY key so
    /// peer groups straddle the chunk cuts the splitter picks, and nulls interleaved so the
    /// "no value yet" accumulator state crosses a boundary too.
    fn fixture(n: usize) -> (ArrayRef, ArrayRef, ArrayRef) {
        // Ties: each order key repeats 3x. Nulls: every 7th value.
        let order: Int64Array = (0..n).map(|i| (i / 3) as i64).collect();
        let ints: Int64Array = (0..n)
            .map(|i| (i % 7 != 0).then_some(((i * 2654435761) % 1000) as i64 - 500))
            .collect();
        let floats: Float64Array = (0..n)
            .map(|i| (i % 7 != 0).then_some(((i * 2654435761) % 1000) as f64 - 500.0))
            .collect();
        (Arc::new(order), Arc::new(ints), Arc::new(floats))
    }

    fn run(
        func: WindowFn,
        order: &ArrayRef,
        values: &ArrayRef,
        n: usize,
        threads: usize,
    ) -> ArrayRef {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let calls = vec![WindowCall {
            func,
            offset: 0,
            frame: None,
            alpha: None,
            half_life: None,
            ignore_nulls: false,
            values: Some(values.clone()),
        }];
        let ok = [(order.clone(), SortOptions::default())];
        pool.install(|| window_serial(&[], &ok, &calls, n, None).unwrap())
            .remove(0)
    }

    /// The whole contract of this module: on a single global partition big enough to split,
    /// the parallel prefix scan must produce **bit-identical** output to the one-thread walk.
    #[test]
    fn split_running_aggregates_are_bit_identical_to_the_serial_walk() {
        let n = MIN_ROWS_TO_SPLIT * 3 + 17; // not a multiple of any chunk size
        let (order, ints, floats) = fixture(n);
        assert!(
            worth_splitting(n, 8),
            "fixture must clear the split threshold"
        );

        for func in [WindowFn::Sum, WindowFn::Min, WindowFn::Max, WindowFn::Avg] {
            let serial = run(func, &order, &ints, n, 1);
            for threads in [2, 3, 8, 16] {
                assert_eq!(
                    &serial,
                    &run(func, &order, &ints, n, threads),
                    "i64 {func:?} diverged at {threads} threads"
                );
            }
        }
        // Float MIN/MAX re-associate exactly; float SUM/AVG stay serial and must also agree.
        for func in [WindowFn::Min, WindowFn::Max, WindowFn::Sum, WindowFn::Avg] {
            let serial = run(func, &order, &floats, n, 1);
            for threads in [2, 3, 8, 16] {
                assert_eq!(
                    &serial,
                    &run(func, &order, &floats, n, threads),
                    "f64 {func:?} diverged at {threads} threads"
                );
            }
        }
        // COUNT takes its own path (it counts validity, not values).
        let serial = run(WindowFn::Count, &order, &ints, n, 1);
        for threads in [2, 3, 8, 16] {
            assert_eq!(
                &serial,
                &run(WindowFn::Count, &order, &ints, n, threads),
                "COUNT diverged at {threads} threads"
            );
        }
    }

    /// The ranking family is positional rather than value-folding, so it takes a different
    /// route to the same requirement: `ROW_NUMBER` scatters through atomics, `RANK` recovers
    /// each peer group's start position, and `DENSE_RANK` prefix-sums distinct groups. The
    /// fixture's 3-wide ties are what make a wrong chunk boundary visible here.
    #[test]
    fn split_ranking_functions_are_bit_identical_to_the_serial_walk() {
        let n = MIN_ROWS_TO_SPLIT * 3 + 17;
        let (order, ints, _) = fixture(n);
        for func in [WindowFn::RowNumber, WindowFn::Rank, WindowFn::DenseRank] {
            let serial = run(func, &order, &ints, n, 1);
            for threads in [2, 3, 8, 16] {
                assert_eq!(
                    &serial,
                    &run(func, &order, &ints, n, threads),
                    "{func:?} diverged at {threads} threads"
                );
            }
        }
    }

    /// Every order key distinct (no ties at all) puts a peer boundary at every position, which
    /// is the opposite extreme from the 3-wide ties above and the case where `rank`'s
    /// "carry the group start" accumulator is reseeded on every single row.
    #[test]
    fn all_distinct_order_keys_rank_identically_when_split() {
        let n = MIN_ROWS_TO_SPLIT * 2 + 5;
        let order: ArrayRef = Arc::new((0..n).map(|i| i as i64).collect::<Int64Array>());
        let vals: ArrayRef = Arc::new((0..n).map(|i| Some(i as i64)).collect::<Int64Array>());
        for func in [
            WindowFn::RowNumber,
            WindowFn::Rank,
            WindowFn::DenseRank,
            WindowFn::Sum,
            WindowFn::Count,
        ] {
            assert_eq!(
                &run(func, &order, &vals, n, 1),
                &run(func, &order, &vals, n, 16),
                "{func:?} diverged with all-distinct order keys"
            );
        }
    }

    /// One single peer group spanning the whole partition: every cut is pushed to the end, so
    /// `peer_chunks` must degenerate to one chunk rather than emit a cut mid-group.
    #[test]
    fn one_giant_peer_group_degenerates_to_a_single_chunk() {
        let n = MIN_ROWS_TO_SPLIT * 2;
        let order: ArrayRef = Arc::new((0..n).map(|_| 7i64).collect::<Int64Array>());
        let vals: ArrayRef = Arc::new((0..n).map(|i| Some(i as i64)).collect::<Int64Array>());
        for func in [
            WindowFn::RowNumber,
            WindowFn::Rank,
            WindowFn::DenseRank,
            WindowFn::Sum,
        ] {
            assert_eq!(
                &run(func, &order, &vals, n, 1),
                &run(func, &order, &vals, n, 16),
                "{func:?} diverged on a single whole-partition peer group"
            );
        }
    }

    /// A cut may never land inside a peer group, or the group's rows would be emitted from
    /// different prefixes. Checks the property directly rather than through a window result.
    #[test]
    fn peer_chunks_never_cut_a_peer_group() {
        // Peer groups of 4: boundary at every 4th position.
        let boundary = |pos: usize| pos % 4 == 3;
        for threads in [1, 2, 3, 7, 64] {
            let chunks = peer_chunks(1000, threads, boundary);
            assert_eq!(chunks.first().unwrap().0, 0);
            assert_eq!(chunks.last().unwrap().1, 1000);
            for w in chunks.windows(2) {
                assert_eq!(w[0].1, w[1].0, "chunks must tile the range with no gap");
            }
            for &(_, end) in &chunks {
                assert!(
                    end == 1000 || boundary(end - 1),
                    "chunk ended at {end}, inside a peer group"
                );
            }
        }
    }

    /// An all-null column keeps the accumulator empty across every chunk, so the emitted
    /// value stays null rather than becoming a zero from the identity element.
    #[test]
    fn an_all_null_column_stays_null_across_chunks() {
        let n = MIN_ROWS_TO_SPLIT * 2;
        let order: ArrayRef = Arc::new((0..n).map(|i| i as i64).collect::<Int64Array>());
        let nulls: ArrayRef = Arc::new((0..n).map(|_| None::<i64>).collect::<Int64Array>());
        for func in [WindowFn::Sum, WindowFn::Min, WindowFn::Max] {
            assert_eq!(
                &run(func, &order, &nulls, n, 1),
                &run(func, &order, &nulls, n, 8)
            );
        }
    }
}
