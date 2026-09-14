//! A range join whose right side is a handful of rows: scan, don't sort.
//!
//! Both general strategies in this module pay `O(n log n)` on the *left* side — IEJoin sorts
//! the union on two axes, and the band sorts each left bound — because both answer "which
//! right rows match this left row" by searching a sorted order. That is the right trade when
//! the two sides are comparable in size. It is the wrong one when the right side is a bucket
//! table: `l_discount >= b.lo AND l_discount < b.hi` against six price bands sorted six
//! million left rows twice to answer six comparisons a row, and measured 329 ms against
//! DuckDB's 89 ms.
//!
//! With a tiny right side the whole join is `|R|` vectorized comparisons over the left key
//! column. No sort, no universe, no mark array, and every pass is an Arrow kernel over a
//! contiguous slice.
//!
//! **This also covers a shape the band path cannot see.** [`super::band::bounds`] detects two
//! conditions bounding one *right* key (`L.a <= R.y AND R.y <= L.b`). The mirror of it — one
//! left key bounded by two right columns, which is what a bucket table is — is not a band by
//! that definition and fell through to IEJoin. It is an interval-stabbing query, whose
//! matches are not a contiguous slice of anything when the intervals overlap, so the general
//! answer really is harder; at this size it does not need to be answered generally.
//!
//! The gate is on *size only*, never on the shape of the predicate, so this path serves one
//! inequality and two, every operator combination, and every join type. It computes exactly
//! what the sorted paths compute — `tests` holds it against them pair for pair.

use arrow::array::{Array, ArrayRef, BooleanArray, Scalar};
use arrow::compute::kernels::cmp;
use rayon::prelude::*;

use super::{Out, RangeOp, SWEEP_MAX_WORKERS};
use crate::error::RuntimeError;

/// Right rows up to which scanning beats sorting.
///
/// The scan costs `|R|` comparisons a left row against the sorted paths' `log2 |L|` search
/// steps plus the sort itself, so the crossover sits well above this. It is set low because
/// the two costs are not symmetric: a scan that should have been a sort grows linearly in
/// `|R|`, while a sort that should have been a scan loses a constant factor. Overshooting the
/// bucket-table case buys little and risks a lot.
const MAX_RIGHT_ROWS: usize = 32;

/// Left rows below which the sorted paths are fast enough that the choice does not matter.
const MIN_LEFT_ROWS: usize = 16_384;

/// Left rows one worker takes: enough that a kernel call amortizes, small enough to spread.
const CHUNK: usize = 65_536;

/// Whether this join's shape is one the scan answers better than a sort.
pub(super) fn worth_it(n_left: usize, n_right: usize) -> bool {
    n_right <= MAX_RIGHT_ROWS && n_left >= MIN_LEFT_ROWS
}

/// The match mask of one right row against a slice of the left key column.
///
/// `left OP right`, so the scalar is always the right-hand operand and the operator is used
/// as written. A null on either side yields a null bit, which the caller reads as no match —
/// consistent with the sorted paths, which exclude null keys before they start.
fn mask(
    left: &ArrayRef,
    right: &ArrayRef,
    row: usize,
    op: RangeOp,
) -> Result<BooleanArray, RuntimeError> {
    let scalar = Scalar::new(right.slice(row, 1));
    let out = match op {
        RangeOp::Lt => cmp::lt(left, &scalar),
        RangeOp::Le => cmp::lt_eq(left, &scalar),
        RangeOp::Gt => cmp::gt(left, &scalar),
        RangeOp::Ge => cmp::gt_eq(left, &scalar),
    };
    out.map_err(|e| RuntimeError::UnsupportedRangeJoin {
        reason: format!("comparison against a scalar right key failed: {e}"),
    })
}

/// Emit every pair for the left rows `lmap[..]` covering `[base, base + span)`.
fn scan_chunk(
    left_keys: &[ArrayRef],
    right_keys: &[ArrayRef],
    ops: &[RangeOp],
    (base, span): (usize, usize),
    (chunk, rmap): (&[u32], &[u32]),
    out: &mut Out,
) -> Result<(), RuntimeError> {
    let sliced: Vec<ArrayRef> = left_keys.iter().map(|k| k.slice(base, span)).collect();
    let mut masks: Vec<BooleanArray> = Vec::with_capacity(rmap.len());
    for &r in rmap {
        let row = r as usize;
        let mut m = mask(&sliced[0], &right_keys[0], row, ops[0])?;
        if ops.len() == 2 {
            let second = mask(&sliced[1], &right_keys[1], row, ops[1])?;
            m = arrow::compute::kernels::boolean::and(&m, &second).map_err(|e| {
                RuntimeError::UnsupportedRangeJoin {
                    reason: format!("combining the two conditions failed: {e}"),
                }
            })?;
        }
        masks.push(m);
    }

    let all = out.needs_all_matches();
    for &l in chunk {
        let i = l as usize - base;
        let mut matched = false;
        for (m, &r) in masks.iter().zip(rmap) {
            if m.is_valid(i) && m.value(i) {
                matched = true;
                out.pair(l, r);
                if !all {
                    break;
                }
            }
        }
        out.finish_left(l, matched);
    }
    Ok(())
}

/// Run the join by scanning each right row against the left key column.
///
/// `lmap` and `rmap` are the non-excluded rows of each side, ascending, exactly as the sorted
/// paths take them.
pub(super) fn run(
    left_keys: &[ArrayRef],
    right_keys: &[ArrayRef],
    ops: &[RangeOp],
    lmap: &[u32],
    rmap: &[u32],
    out: &mut Out,
) -> Result<(), RuntimeError> {
    // Chunks are cut on *left row* boundaries rather than on positions in `lmap`, so each
    // worker's slice of the key column is contiguous and the kernel reads it as one run. An
    // excluded row inside the span costs a compared value nobody reads, which is cheaper than
    // the gather that avoiding it would need.
    let n_left = left_keys[0].len();
    let bounds: Vec<(usize, usize)> = (0..n_left)
        .step_by(CHUNK)
        .map(|base| (base, CHUNK.min(n_left - base)))
        .collect();
    // `lmap` is ascending, so each chunk's rows are a contiguous run of it.
    let mut cuts: Vec<&[u32]> = Vec::with_capacity(bounds.len());
    let mut rest = lmap;
    for &(base, span) in &bounds {
        let end = rest.partition_point(|&l| (l as usize) < base + span);
        let (here, after) = rest.split_at(end);
        cuts.push(here);
        rest = after;
    }

    let workers = rayon::current_num_threads()
        .min(bounds.len())
        .min(SWEEP_MAX_WORKERS);
    if workers < 2 {
        for (&b, &chunk) in bounds.iter().zip(&cuts) {
            scan_chunk(left_keys, right_keys, ops, b, (chunk, rmap), out)?;
        }
        return Ok(());
    }

    let parts: Vec<Result<Out, RuntimeError>> = bounds
        .par_iter()
        .zip(cuts)
        .map(|(&b, chunk)| {
            let mut o = out.sibling(chunk.len());
            scan_chunk(left_keys, right_keys, ops, b, (chunk, rmap), &mut o).map(|()| o)
        })
        .collect();
    // Absorbed in chunk order, so the parallel result is byte-identical to the sequential one.
    for part in parts {
        out.absorb(part?);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int64Array};

    use super::*;
    use crate::join::range::two_conditions;
    use crate::join::{JoinIndices, JoinType};

    /// The pairs a strategy produces, as a sorted relation — the comparison is of the join's
    /// *answer*, which is an unordered set of index pairs.
    fn pairs(idx: &JoinIndices) -> Vec<(Option<u32>, Option<u32>)> {
        let at = |a: &arrow::array::UInt32Array, i: usize| a.is_valid(i).then(|| a.value(i));
        let mut v: Vec<(Option<u32>, Option<u32>)> = (0..idx.left.len())
            .map(|i| (at(&idx.left, i), at(&idx.right, i)))
            .collect();
        v.sort_unstable();
        v
    }

    fn indices(
        left: &[ArrayRef],
        right: &[ArrayRef],
        ops: &[RangeOp],
        join_type: JoinType,
        scan: bool,
    ) -> JoinIndices {
        let (nl, nr) = (left[0].len(), right[0].len());
        let lmap: Vec<u32> = (0..nl as u32).collect();
        let rmap: Vec<u32> = (0..nr as u32).collect();
        let mut out = Out::new(join_type, nr, nl);
        if scan {
            run(left, right, ops, &lmap, &rmap, &mut out).unwrap();
        } else {
            two_conditions(left, right, ops, &lmap, &rmap, &mut out).unwrap();
        }
        out.into_indices(&vec![false; nr])
    }

    fn sample(n: usize, seed: u64, span: i64) -> ArrayRef {
        let mut s = seed | 1;
        Arc::new(Int64Array::from_iter_values((0..n).map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % span as u64) as i64
        })))
    }

    /// The whole correctness argument: on the same data, the scan and IEJoin are the same
    /// relation — for every operator pair and every join type.
    #[test]
    fn the_scan_and_iejoin_agree_pair_for_pair() {
        let left = vec![sample(600, 7, 50), sample(600, 99, 50)];
        let right = vec![sample(9, 31, 50), sample(9, 5, 50)];
        let ops = [RangeOp::Lt, RangeOp::Le, RangeOp::Gt, RangeOp::Ge];
        let types = [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
            JoinType::Semi,
            JoinType::Anti,
        ];
        for a in ops {
            for b in ops {
                for jt in types {
                    let scanned = indices(&left, &right, &[a, b], jt, true);
                    let sorted = indices(&left, &right, &[a, b], jt, false);
                    assert_eq!(
                        pairs(&scanned),
                        pairs(&sorted),
                        "{a:?}/{b:?} {jt:?} disagree"
                    );
                }
            }
        }
    }

    /// A bucket table: two right columns bounding one left key, which is the shape
    /// `band::bounds` declines and the shape this path exists for.
    #[test]
    fn a_bucket_table_assigns_every_left_row_to_its_interval() {
        let lo: ArrayRef = Arc::new(Float64Array::from(vec![0.0, 0.2, 0.4, 0.6, 0.8]));
        let hi: ArrayRef = Arc::new(Float64Array::from(vec![0.2, 0.4, 0.6, 0.8, 1.0]));
        let x: ArrayRef = Arc::new(Float64Array::from_iter_values(
            (0..1000).map(|i| f64::from(i) / 1000.0),
        ));
        let idx = indices(
            &[x.clone(), x],
            &[lo, hi],
            &[RangeOp::Ge, RangeOp::Lt],
            JoinType::Inner,
            true,
        );
        assert_eq!(
            idx.left.len(),
            1000,
            "every row falls in exactly one bucket"
        );
        for i in 0..idx.left.len() {
            let (l, r) = (idx.left.value(i), idx.right.value(i));
            assert_eq!(r, l / 200, "row {l} landed in bucket {r}");
        }
    }

    #[test]
    fn the_gate_fires_on_a_bucket_table_and_not_on_two_large_sides() {
        assert!(worth_it(6_000_000, 6));
        assert!(!worth_it(6_000_000, MAX_RIGHT_ROWS + 1));
        assert!(!worth_it(MIN_LEFT_ROWS - 1, 6));
    }

    /// Chunking is what makes the scan parallel, and a left row must be emitted by exactly
    /// one chunk. This spans several chunk boundaries with a right side of one interval.
    #[test]
    fn rows_are_emitted_once_across_chunk_boundaries() {
        let n = CHUNK * 2 + 7;
        let x: ArrayRef = Arc::new(Int64Array::from_iter_values(0..n as i64));
        let lo: ArrayRef = Arc::new(Int64Array::from(vec![0_i64]));
        let hi: ArrayRef = Arc::new(Int64Array::from(vec![n as i64]));
        let idx = indices(
            &[x.clone(), x],
            &[lo, hi],
            &[RangeOp::Ge, RangeOp::Lt],
            JoinType::Inner,
            true,
        );
        assert_eq!(idx.left.len(), n);
        let mut seen: Vec<u32> = idx.left.values().to_vec();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), n, "a row was emitted twice or not at all");
    }
}
