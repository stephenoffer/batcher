//! The band join: two inequalities that bound **one** right key from both sides.
//!
//! `L.a <= R.y AND R.y <= L.b` is not a general two-inequality join, and paying IEJoin for
//! it is what made this operator lose to DuckDB above a million rows. IEJoin is general: it
//! assumes the two conditions constrain *different* right keys, so a left row's matches are
//! an arbitrary subset of the right side and have to be read out of a bit array. When both
//! conditions read the same right key the matches are a **contiguous slice** of that key
//! sorted once — and everything IEJoin needs in order to be general disappears with it:
//!
//! | | IEJoin | band |
//! |---|---|---|
//! | sorts | the `2n` union, twice (one per axis) | the right side once, each left bound once |
//! | per left row | scan the axis-1 mark suffix | a cursor step, amortized |
//! | extra state | mark bitmap + two rank tables + `order2` | two `u32` bound arrays |
//!
//! The per-row cost is a *cursor step* rather than a binary search because both bounds are
//! monotone in the left key: see [`bounds_by_merge`], which is where the measured win is.
//! The first version of this module did search per row, and at five million rows a side that
//! was 11.5 s of an 11.8 s join — 23 random probes into a 40 MB array, five million times.
//!
//! The band shape is not a corner case. Interval containment, temporal overlap, IP-range and
//! price-band lookups, and `BETWEEN` against a computed pair are all this, and they are most
//! of what an inequality join is used for in practice. The general shape still routes to
//! IEJoin, unchanged.
//!
//! **Detection is conservative and fails safe.** It requires the two conditions to reference
//! the literally-same right array (`Arc::ptr_eq`, which holds because `columns_by_name`
//! hands out `Arc` clones of one column) and to face opposite ways. Anything else — two
//! different right columns, two conditions facing the same way, a float column that
//! canonicalization rebuilt — simply misses the band path and runs IEJoin. A miss costs
//! speed; it cannot cost correctness.

use std::cmp::Ordering;
use std::sync::Arc;

use arrow::array::ArrayRef;
use rayon::prelude::*;

use super::keys::AxisKeys;
use super::{Out, RangeOp, SWEEP_MAX_WORKERS};
use crate::error::RuntimeError;

/// Left rows per worker below which splitting the searches is not worth the fan-out.
const PARALLEL_MIN_PER_WORKER: usize = 4_096;

/// Element floor above which the sorted-right gather is worth handing to rayon.
///
/// Matches `keys::PARALLEL_MAP_MIN`: the same trade, guarding a linear pass over an array the
/// size of the right side, and it must not fire on the small joins this operator also serves.
const PARALLEL_GATHER_MIN: usize = 32_768;

/// The condition indices bounding the shared right key from below and from above, or `None`
/// when this pair of conditions is not a band.
///
/// Orientation is `left OP right`, so `<`/`<=` put the left key beneath the right one and
/// therefore bound the right key from *below*; `>`/`>=` bound it from above. A band needs
/// exactly one of each, over one shared right key.
pub(super) fn bounds(right_keys: &[ArrayRef], ops: &[RangeOp]) -> Option<(usize, usize)> {
    if !Arc::ptr_eq(&right_keys[0], &right_keys[1]) {
        return None;
    }
    match (ops[0].lower_bounds_right(), ops[1].lower_bounds_right()) {
        (true, false) => Some((0, 1)),
        (false, true) => Some((1, 0)),
        // Both conditions face the same way, so one of them is redundant rather than a
        // bound; that is a degenerate shape IEJoin handles without a special case.
        _ => None,
    }
}

/// Which side of the band a merge pass is resolving.
#[derive(Clone, Copy)]
enum Side {
    /// The first right position that clears the left row's lower bound.
    Lower,
    /// The first right position that exceeds the left row's upper bound.
    Upper,
}

/// The right side as both merges see it: sorted once, and — on the fast axis — laid out
/// contiguously so a cursor walk reads sequentially.
struct Right<'a> {
    /// Right universe entries in ascending key order.
    order: &'a [u32],
    /// `order`'s keys, materialized in that order. `None` on the encoded (variable-width)
    /// axis, where the generic comparison path runs instead.
    sorted: Option<&'a [u64]>,
    nl: usize,
    lmap: &'a [u32],
    rmap: &'a [u32],
}

impl Right<'_> {
    /// Every left row's bound into `order`, by one merge instead of a binary search each.
    ///
    /// Both bounds are **monotone in the left key**: a larger lower bound can only move the
    /// start right, and a larger upper bound can only move the end right. So walking the
    /// left rows in key order lets one cursor over `order` serve all of them — `O(L + R)`
    /// sequential reads in place of `O(L log R)` random ones. That is the whole win; the
    /// binary-search version of this was 97% of the band join's time at five million rows.
    ///
    /// Returned indexed by left universe entry, not by sorted position, so the emission loop
    /// stays in `lmap` order and the output is unchanged.
    fn bounds_by_merge(&self, keys: &AxisKeys, strict: bool, side: Side) -> Vec<u32> {
        let (nl, lmap, rmap) = (self.nl, self.lmap, self.rmap);
        let mut at = vec![0u32; nl];
        let sorted_left = keys.sorted_left(nl, lmap, rmap);
        // The four cases are two comparisons, with strictness deciding whether the
        // equal-key group is skipped or kept:
        //
        //   lower, non-strict (`a <= y`): stop at the first `y >= a` -> skip while `y < a`
        //   lower, strict     (`a <  y`): stop at the first `y >  a` -> skip while `y <= a`
        //   upper, non-strict (`b >= y`): stop at the first `y >  b` -> skip while `y <= b`
        //   upper, strict     (`b >  y`): stop at the first `y >= b` -> skip while `y <  b`
        //
        // so `Lower`-strict and `Upper`-non-strict share a test, as do the other two.
        let skip_equal = matches!((side, strict), (Side::Lower, true) | (Side::Upper, false));

        // Fast path: both sides contiguous ascending `u64`, so the merge is two sequential
        // scans. The generic path below reads `keys[order[p]]`, and although `p` advances in
        // order, `order[p]` does not — every cursor step is a random probe into a 40 MB
        // array at five million rows, which was most of what the merge cost.
        if let (Some(rk), Some(all)) = (self.sorted, keys.fast()) {
            let mut p = 0usize;
            for &e in &sorted_left {
                let lk = all[e as usize];
                while p < rk.len() && (rk[p] < lk || (skip_equal && rk[p] == lk)) {
                    p += 1;
                }
                at[e as usize] = p as u32;
            }
            return at;
        }

        let mut p = 0usize;
        for &e in &sorted_left {
            while p < self.order.len() {
                let ord = keys.cmp(self.order[p], e, nl, lmap, rmap);
                let skip = if skip_equal {
                    ord != Ordering::Greater
                } else {
                    ord == Ordering::Less
                };
                if !skip {
                    break;
                }
                p += 1;
            }
            at[e as usize] = p as u32;
        }
        at
    }
}

/// Join by the band `lower <= right key <= upper`, appending index pairs to `out`.
pub(super) fn run(
    left_keys: &[ArrayRef],
    right_keys: &[ArrayRef],
    ops: &[RangeOp],
    (lower, upper): (usize, usize),
    lmap: &[u32],
    rmap: &[u32],
    out: &mut Out,
) -> Result<(), RuntimeError> {
    let nl = lmap.len();
    let n = nl + rmap.len();

    // Both universes are built **ascending**, deliberately not with each op's own
    // `axis1_descending()`. The two conditions face opposite ways, so letting each pick its
    // own sense would sort the shared right key two different ways and the two bounds would
    // then index two different orders — the searches would be individually correct and the
    // slice between them meaningless.
    let k_lo = AxisKeys::build(&left_keys[lower], &right_keys[lower], false, lmap, rmap)?;
    let k_hi = AxisKeys::build(&left_keys[upper], &right_keys[upper], false, lmap, rmap)?;
    // Either universe orders the right side identically (same column, same sense), so one
    // sorted order serves both bounds.
    let order = k_lo.sorted_right(n, nl, lmap, rmap);

    // The right keys laid out in sorted order, once, so each merge scans a contiguous array
    // instead of chasing `order[p]` into the universe. Both merges can share it: the two
    // universes encode the *same* right column with the same sense, so their right halves
    // are identical `u64`s. `None` on the encoded (variable-width) axis, where the generic
    // comparison path runs instead.
    let right_sorted: Option<Vec<u64>> = k_lo.fast().map(|all| {
        // One gather over the sorted right side, so it fans out the way the merges below do.
        // rayon's indexed `collect` writes each element at the index it was read from, so this
        // is the sequential gather's output element for element.
        if order.len() >= PARALLEL_GATHER_MIN {
            order.par_iter().map(|&r| all[r as usize]).collect()
        } else {
            order.iter().map(|&r| all[r as usize]).collect()
        }
    });
    let right = Right {
        order: &order,
        sorted: right_sorted.as_deref(),
        nl,
        lmap,
        rmap,
    };

    // The two merges are independent and each is a sort plus a linear walk, so they run
    // concurrently rather than one after the other.
    let (start, end) = rayon::join(
        || right.bounds_by_merge(&k_lo, ops[lower].strict(), Side::Lower),
        || right.bounds_by_merge(&k_hi, ops[upper].strict(), Side::Upper),
    );

    let all = out.needs_all_matches();
    let emit = |e: usize, l: u32, o: &mut Out| {
        let (s, t) = (start[e] as usize, end[e] as usize);
        // An empty band gives `t <= s`; slicing on that would panic, and the row is simply
        // unmatched.
        let matched = s < t;
        if all && matched {
            for &r in &order[s..t] {
                o.pair(l, rmap[r as usize - nl]);
            }
        }
        o.finish_left(l, matched);
    };

    let workers = rayon::current_num_threads()
        .min(nl / PARALLEL_MIN_PER_WORKER)
        .min(SWEEP_MAX_WORKERS);
    if workers < 2 {
        for (i, &l) in lmap.iter().enumerate() {
            emit(i, l, out);
        }
        return Ok(());
    }

    // Left rows are independent — a row's matches are a function of the right side alone —
    // so the slices need no shared state at all, unlike the IEJoin sweep's mark rebuild.
    // Folded back in slice order, so the output is identical to the sequential loop's.
    let per = nl.div_ceil(workers);
    let parts: Vec<Out> = lmap
        .par_chunks(per)
        .enumerate()
        .map(|(chunk, slice)| {
            let mut o = out.sibling(slice.len());
            let base = chunk * per;
            for (i, &l) in slice.iter().enumerate() {
                emit(base + i, l, &mut o);
            }
            o
        })
        .collect();
    for part in parts {
        out.absorb(part);
    }
    Ok(())
}
