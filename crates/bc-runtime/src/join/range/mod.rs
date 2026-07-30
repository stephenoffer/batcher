//! Range (inequality) join: `L.x op R.y`, optionally with a second inequality.
//!
//! Every inequality, interval-containment and band join used to lower to a *materialized*
//! cartesian product with the predicate as a filter above it, so the intermediate was
//! `|L| x |R|` rows however few survived. This module is the replacement, and nothing
//! quadratic is materialized on the way to the answer.
//!
//! Cost, stated precisely because "IEJoin is `O(n log n)`" is the wrong reading: `O(n log n)`
//! for the sorts, `O(k)` for the `k` emitted pairs, plus a term for *skipping* the unset bits
//! of the mark array — each left row scans the axis-1 suffix from its own bound to the end.
//! [`MarkSet`] derives its level count from the universe size, and each level multiplies the
//! span one word read dismisses by 64, so walking an *empty* suffix costs one read per level
//! (four at ten million entries) rather than one per word. DuckDB removes the term outright by
//! pruning block *pairs*; this bounds it instead.
//!
//! **That term is no longer what the operator spends its time on, and the difference matters
//! because the two have opposite fixes.** A phase study split the non-sort remainder into the
//! sweep and the seven passes that build the sweep's inputs, and at five million rows a side
//! the passes were 781 ms against a single-threaded sweep of 638 ms — while the sweep already
//! fanned out to rayon and the passes did not. They are pure gathers and scatters over the
//! universe, so they parallelize completely; three of them re-read `order2` to produce three
//! arrays indexed identically, so they also fuse. Doing both took the whole operator, on the
//! shape `report_range_join_phases` measures, from **107 ms to 91 ms at 500,000 rows a side,
//! 558 ms to 386 ms at 2,000,000, and 1.9 s to 1.3 s at 5,000,000** (best of three each way,
//! same box, load average ~13). Block-pair pruning would not have moved any of it: the
//! suffix walk it removes was already the smaller half.
//!
//! Three algorithms, picked by the *shape* of the condition rather than only its arity:
//!
//! - **One inequality** — sort the right side once; for each left row the matching right
//!   rows are a *contiguous suffix* found by binary search, so the pairs are emitted
//!   directly with no per-pair predicate evaluation.
//! - **Two inequalities bounding one shared right key** (a *band*: `L.a <= R.y AND R.y <=
//!   L.b`) — see [`band`]. The matches are a contiguous *slice* of one sorted array, and
//!   both of its bounds are monotone in the left key, so neither the union sort nor the
//!   mark array is needed. This is the common real-world shape (interval containment,
//!   temporal overlap, `BETWEEN` against a computed pair) and it is 1.7-2.2x the general
//!   path from 500K to 5M rows a side.
//! - **Two inequalities, general** — IEJoin (Khayyat et al., *Lightning Fast and Space
//!   Efficient Inequality Joins*, VLDB 2015), the algorithm DuckDB's `PhysicalIEJoin`
//!   implements. Sort the union of both sides on each axis, sweep the second axis marking
//!   right rows in a bit array indexed by first-axis rank, and read each left row's matches
//!   off as the set bits in a suffix of that array. The mark-scan term described above is
//!   this path's, and only this path's — the band does not pay it at all.
//!
//! Both produce the same [`JoinIndices`](super::JoinIndices) relation the hash join does,
//! so every join type falls out of the same index-pair shape and the caller's gather is
//! unchanged.
//!
//! Parallelism, and why it is sound: a left row's matches are a function of the whole right
//! side and of nothing else about the left, so the left rows can be split arbitrarily. Both
//! axis sorts fan out to rayon, and the sweep splits the left rows into contiguous slices of
//! axis-2 order — each worker rebuilds the mark array for its slice's start with one binary
//! search and one prefix pass, then sweeps only its own rows. Slices are folded back in slice
//! order, so the output is *identical* to the sequential sweep's, not merely equivalent as an
//! unordered relation (`the_parallel_paths_agree_with_an_analytic_answer` pins that against a
//! single-threaded rayon pool).
//!
//! The same decomposition is what would make the operator **distributable**; that is not yet
//! wired, and nothing here carries a single-node assumption that would block it.
//!
//! # Null and float semantics
//!
//! SQL three-valued logic: a comparison with NULL is UNKNOWN, so a NULL-keyed row never
//! matches. Those rows are excluded from the sweep and rejoined with the unmatched rows.
//!
//! Floats follow the engine's one float-identity contract (`bc_arrow::float_ident`): a
//! **total** order in which every NaN is equal and ranks above every number, and `-0.0`
//! compares equal to `0.0`. That is what `ORDER BY`, `=` and the comparison kernels
//! already do here, so it is what this join must do — an IEEE reading, under which a
//! comparison with NaN is always false, would silently return fewer rows than the
//! cross-product-plus-filter plan this replaces. `canonicalize_float_keys` folds both
//! zeros and every NaN bit pattern to one representative, after which arrow's row
//! encoding *is* that total order.

use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};

use arrow::array::{Array, ArrayRef};
use rayon::prelude::*;

use super::{null_mask, IndexBuf, JoinIndices, JoinType};
use crate::error::RuntimeError;

mod band;
mod keys;
mod marks;

use keys::{supported_key_type, AxisKeys};
use marks::MarkSet;

/// One inequality in a range join's condition, oriented `left_key OP right_key`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeOp {
    Lt,
    Le,
    Gt,
    Ge,
}

impl RangeOp {
    /// `<`/`>` exclude equal keys; `<=`/`>=` admit them. This is the only place the
    /// distinction is needed — it selects an upper vs a lower bound in the searches.
    fn strict(self) -> bool {
        matches!(self, RangeOp::Lt | RangeOp::Gt)
    }

    /// Sort sense for the **first** axis: descending for `>`/`>=`, ascending for `<`/`<=`.
    ///
    /// Under this encoding the right rows satisfying the condition against a given left
    /// key are exactly a *suffix* of the sorted order, for all four operators — which is
    /// what lets one binary search serve every case.
    fn axis1_descending(self) -> bool {
        matches!(self, RangeOp::Gt | RangeOp::Ge)
    }

    /// Whether this operator bounds the **right** key from below.
    ///
    /// Orientation is `left OP right`, so `<`/`<=` place the left key beneath the right one
    /// — a lower bound on the right key — and `>`/`>=` place it above. [`band`] uses this to
    /// tell a two-sided band from two conditions facing the same way.
    fn lower_bounds_right(self) -> bool {
        matches!(self, RangeOp::Lt | RangeOp::Le)
    }

    /// Sort sense for the **second** axis, the opposite of [`Self::axis1_descending`].
    ///
    /// The mirrored sense is what makes the satisfying set a *prefix* on this axis, so a
    /// single monotone cursor can mark it as the sweep advances.
    fn axis2_descending(self) -> bool {
        matches!(self, RangeOp::Lt | RangeOp::Le)
    }
}

/// `mark_at` sentinel for an `order2` position holding a *left* row, which is never marked.
///
/// A real bit index cannot reach it: indices are axis-1 positions in a universe already
/// bounded by `u32`, so a relation of `u32::MAX` rows would have overflowed the join first.
const NOT_A_RIGHT_ROW: u32 = u32::MAX;

/// Universe size below which the sweep stays sequential.
///
/// Each worker rebuilds the mark array for its slice's starting point, which costs a linear
/// pass over the `order2` prefix it needs. That is cheap against a sweep worth splitting and
/// pure overhead against one that is not.
const PARALLEL_SWEEP_MIN_ROWS: usize = 65_536;

/// Ceiling on sweep workers, independent of how many cores there are.
///
/// Each worker allocates its own mark bitmap — `n` bits — so the peak is `workers x n / 8`
/// bytes. On a 96-core box joining two million rows a side that is 48 MB of bitmap for a
/// join whose inputs are 32 MB. Capping at 16 costs nothing measurable (a 2,000,000-row join
/// moved by less than run-to-run noise) and holds the peak to 8 MB.
const SWEEP_MAX_WORKERS: usize = 16;

/// Left rows per worker below which splitting is not worth the per-slice mark rebuild.
const PARALLEL_SWEEP_MIN_PER_WORKER: usize = 4_096;

/// Chunk length for the parallel setup passes that build the sweep's `u32` inputs.
///
/// These passes gather randomly over the whole universe, so the chunk length does not affect
/// locality and is set purely for scheduling granularity: large enough that rayon's per-task
/// overhead disappears against the work, small enough that 96 cores all get a share of a
/// universe of a few hundred thousand. One morsel's worth is both.
const SETUP_CHUNK: usize = 16_384;

/// Accumulates the index pairs, applying the join type's emission rules once per left row.
struct Out {
    left: IndexBuf,
    right: IndexBuf,
    join_type: JoinType,
    /// Right rows that matched at least once — only tracked for `Right`/`Full`, where the
    /// unmatched ones still have to be emitted.
    right_matched: Vec<bool>,
}

impl Out {
    fn new(join_type: JoinType, n_right: usize, hint: usize) -> Self {
        let track = matches!(join_type, JoinType::Right | JoinType::Full);
        Self {
            left: IndexBuf::with_capacity(hint),
            right: IndexBuf::with_capacity(hint),
            join_type,
            right_matched: if track {
                vec![false; n_right]
            } else {
                Vec::new()
            },
        }
    }

    /// An empty `Out` with this one's rules, for a worker sweeping a slice of the left rows.
    fn sibling(&self, hint: usize) -> Self {
        Self::new(self.join_type, self.right_matched.len(), hint)
    }

    /// Fold a worker's `Out` back in. Called in slice order, which is the same order the
    /// sequential sweep emits in — so the parallel result is byte-identical, not merely
    /// equivalent as an unordered relation.
    fn absorb(&mut self, other: Self) {
        for (mine, theirs) in self.right_matched.iter_mut().zip(&other.right_matched) {
            *mine |= *theirs;
        }
        self.left.extend(other.left);
        self.right.extend(other.right);
    }

    /// Whether the caller must keep enumerating a left row's matches after the first one.
    /// Semi and anti only need existence, so they stop at one.
    fn needs_all_matches(&self) -> bool {
        !matches!(self.join_type, JoinType::Semi | JoinType::Anti)
    }

    fn pair(&mut self, l: u32, r: u32) {
        if matches!(
            self.join_type,
            JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full
        ) {
            self.left.push(l);
            self.right.push(r);
        }
        if !self.right_matched.is_empty() {
            self.right_matched[r as usize] = true;
        }
    }

    /// Close out one left row once its matches are known.
    fn finish_left(&mut self, l: u32, matched: bool) {
        match self.join_type {
            JoinType::Semi if matched => {
                self.left.push(l);
                self.right.push_null();
            }
            JoinType::Anti if !matched => {
                self.left.push(l);
                self.right.push_null();
            }
            JoinType::Left | JoinType::Full if !matched => {
                self.left.push(l);
                self.right.push_null();
            }
            _ => {}
        }
    }

    /// Left rows dropped before the sweep (NULL or NaN key) match nothing by definition.
    fn excluded_left(&mut self, l: u32) {
        self.finish_left(l, false);
    }

    fn into_indices(mut self, right_excluded: &[bool]) -> JoinIndices {
        if matches!(self.join_type, JoinType::Right | JoinType::Full) {
            // An excluded right row (NULL or NaN key) is unmatched by construction, so it
            // is emitted here once — not once here and once by the exclusion loop, which
            // would duplicate it.
            for (r, &matched) in self.right_matched.iter().enumerate() {
                if !matched || right_excluded[r] {
                    debug_assert!(!(matched && right_excluded[r]));
                    self.left.push_null();
                    self.right.push(r as u32);
                }
            }
        }
        JoinIndices::from_bufs(self.left, self.right)
    }
}

/// Range join on one or two inequalities.
///
/// `left_keys[i] ops[i] right_keys[i]` is condition `i`; one or two conditions are
/// supported, and both sides of a condition must share a data type (the planner
/// establishes that — a mismatch is an invariant violation, not a fallback).
///
/// Produces the same [`JoinIndices`](super::JoinIndices) relation the hash join does, for
/// every join type. Output order is unspecified, as it is for every other join here.
pub fn range_join_indices(
    left_keys: &[ArrayRef],
    right_keys: &[ArrayRef],
    ops: &[RangeOp],
    join_type: JoinType,
) -> Result<JoinIndices, RuntimeError> {
    if ops.is_empty()
        || ops.len() > 2
        || left_keys.len() != ops.len()
        || right_keys.len() != ops.len()
    {
        return Err(RuntimeError::UnsupportedRangeJoin {
            reason: format!(
                "expected 1 or 2 conditions with matching key arity, got {} ops / {} left / {} right",
                ops.len(),
                left_keys.len(),
                right_keys.len()
            ),
        });
    }
    // Decode dictionary keys *before* the type check below, not after it. The two sides of a
    // join are reached by different operator chains, so one can arrive dictionary-encoded and
    // the other decoded; comparing physical types then reads that as "key types differ" and
    // declines a join it is perfectly able to run. Declining fails safe rather than wrong — the
    // caller gets an error, not a bad answer — but it is still a query that stops working, so
    // the encodings are reconciled first and the check then compares the types that matter.
    // Same argument as `keys::decode_dict_keys`, which the hash join uses for the same reason.
    let l_dec = crate::keys::decode_dict_keys(left_keys);
    let r_dec = crate::keys::decode_dict_keys(right_keys);
    let left_keys: &[ArrayRef] = l_dec.as_deref().unwrap_or(left_keys);
    let right_keys: &[ArrayRef] = r_dec.as_deref().unwrap_or(right_keys);

    for (l, r) in left_keys.iter().zip(right_keys) {
        if l.data_type() != r.data_type() {
            return Err(RuntimeError::UnsupportedRangeJoin {
                reason: format!(
                    "key types differ across the condition: {} vs {}",
                    l.data_type(),
                    r.data_type()
                ),
            });
        }
        if !supported_key_type(l.data_type()) {
            return Err(RuntimeError::UnsupportedRangeJoin {
                reason: format!("key type {} has no usable total order", l.data_type()),
            });
        }
    }

    let n_left = left_keys[0].len();
    let n_right = right_keys[0].len();

    // `-0.0` and `0.0` compare equal under IEEE but encode to different bytes, so fold them
    // exactly as every other key path here does. NaN is handled by exclusion, below.
    let l_canon = crate::keys::canonicalize_float_keys(left_keys);
    let r_canon = crate::keys::canonicalize_float_keys(right_keys);
    let left_keys: &[ArrayRef] = l_canon.as_deref().unwrap_or(left_keys);
    let right_keys: &[ArrayRef] = r_canon.as_deref().unwrap_or(right_keys);

    let left_excluded = null_mask(left_keys, n_left);
    let right_excluded = null_mask(right_keys, n_right);
    let lmap: Vec<u32> = (0..n_left as u32)
        .filter(|&i| !left_excluded[i as usize])
        .collect();
    let rmap: Vec<u32> = (0..n_right as u32)
        .filter(|&i| !right_excluded[i as usize])
        .collect();

    let mut out = Out::new(join_type, n_right, lmap.len().max(rmap.len()));
    for (i, &ex) in left_excluded.iter().enumerate() {
        if ex {
            out.excluded_left(i as u32);
        }
    }

    if lmap.is_empty() || rmap.is_empty() {
        for &l in &lmap {
            out.finish_left(l, false);
        }
        return Ok(out.into_indices(&right_excluded));
    }

    if ops.len() == 1 {
        single_condition(left_keys, right_keys, ops[0], &lmap, &rmap, &mut out)?;
    } else if let Some(sides) = band::bounds(right_keys, ops) {
        // Both conditions bound ONE right key, so the matches are a contiguous slice of it
        // sorted once — no union sort, no second axis, no mark array. See `band`.
        band::run(left_keys, right_keys, ops, sides, &lmap, &rmap, &mut out)?;
    } else {
        two_conditions(left_keys, right_keys, ops, &lmap, &rmap, &mut out)?;
    }
    Ok(out.into_indices(&right_excluded))
}

/// One inequality: the matches for a left row are a contiguous suffix of the sorted right
/// side, so they are emitted directly — no bit array, no per-pair comparison.
fn single_condition(
    left_keys: &[ArrayRef],
    right_keys: &[ArrayRef],
    op: RangeOp,
    lmap: &[u32],
    rmap: &[u32],
    out: &mut Out,
) -> Result<(), RuntimeError> {
    use std::cmp::Ordering;

    let nl = lmap.len();
    let n = nl + rmap.len();
    let keys = AxisKeys::build(
        &left_keys[0],
        &right_keys[0],
        op.axis1_descending(),
        lmap,
        rmap,
    )?;
    let order = keys.sorted_right(n, nl, lmap, rmap);

    let strict = op.strict();
    let all = out.needs_all_matches();
    for (i, &l) in lmap.iter().enumerate() {
        let e = i as u32;
        // Satisfying right rows are a suffix; find where it starts. Strict excludes the
        // equal-key group, non-strict includes it.
        let lo = if strict {
            order.partition_point(|&r| keys.cmp(r, e, nl, lmap, rmap) != Ordering::Greater)
        } else {
            order.partition_point(|&r| keys.cmp(r, e, nl, lmap, rmap) == Ordering::Less)
        };
        let matched = lo < order.len();
        if all {
            for &r in &order[lo..] {
                out.pair(l, rmap[r as usize - nl]);
            }
        }
        out.finish_left(l, matched);
    }
    Ok(())
}

/// Two inequalities: IEJoin.
///
/// Both sides are unioned into one universe and sorted on each axis. On axis 1 the right
/// rows satisfying condition 1 against a left key are a suffix; on axis 2 (sorted in the
/// mirrored sense) the rows satisfying condition 2 are a prefix, and that prefix only
/// *grows* as the sweep advances — so one monotone cursor marks them, and each left row's
/// answer is the set bits of the axis-1 bit array from its suffix bound onward.
fn two_conditions(
    left_keys: &[ArrayRef],
    right_keys: &[ArrayRef],
    ops: &[RangeOp],
    lmap: &[u32],
    rmap: &[u32],
    out: &mut Out,
) -> Result<(), RuntimeError> {
    let nl = lmap.len();
    let n = nl + rmap.len();

    let k1 = AxisKeys::build(
        &left_keys[0],
        &right_keys[0],
        ops[0].axis1_descending(),
        lmap,
        rmap,
    )?;
    let k2 = AxisKeys::build(
        &left_keys[1],
        &right_keys[1],
        ops[1].axis2_descending(),
        lmap,
        rmap,
    )?;

    let (order1, drank1) = k1.sorted_order_and_ranks(n, nl, lmap, rmap);
    // Two ranks on axis 1, and they are different things. `pos1` is the entry's *position*,
    // which is what the mark bitmap is indexed by and what the binary search returns.
    // `drank1` is the *dense* rank, where equal keys share a value, which is what makes a
    // `u32` compare mean the same thing as a key compare.
    //
    // This is the inverse of `order1`, so every entry is written exactly once and no two
    // writes target the same slot. The disjointness is a property of `order1` being a
    // permutation, which the compiler cannot see, so the slot type carries it instead: a
    // `Relaxed` store compiles to the same move a `u32` write does, and the ordering that
    // makes the stores visible to the reads below comes from rayon's join, not from the
    // atomic. It was the largest setup pass at 181 ms for five million rows a side, entirely
    // because it ran on one core.
    let pos1: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
    order1.par_iter().enumerate().for_each(|(i, &e)| {
        pos1[e as usize].store(i as u32, AtomicOrdering::Relaxed);
    });
    // Where each axis-1 dense rank first appears in `order1`. Because the ranks are dense
    // and `order1` is sorted by them, one reverse pass gives the whole table — and it turns
    // every left row's axis-1 bound from a binary search into an array read.
    //
    // That is not a micro-optimization at scale: the search probes a `u32` array the size of
    // the universe, so at five million rows a side it was ~23 random reads into 40 MB, five
    // million times. The phase study put the sweep at 900 ms of a 1.6 s join, and this is
    // most of it.
    let sorted_drank1: Vec<u32> = order1.par_iter().map(|&e| drank1[e as usize]).collect();
    let max_rank = sorted_drank1.last().copied().unwrap_or(0) as usize;
    let mut first_at = vec![n as u32; max_rank + 2];
    for (i, &r) in sorted_drank1.iter().enumerate().rev() {
        first_at[r as usize] = i as u32;
    }

    let (order2, drank2) = k2.sorted_order_and_ranks(n, nl, lmap, rmap);
    // The sweep walks `order2` twice per step — once for the entry, once to look up its
    // axis-2 rank, and once more to find a right entry's axis-1 bit. Precomputing all three in
    // `order2` order turns those gathers into sequential reads of flat arrays, which is
    // what the marking loop (the largest phase after the sorts) actually spends its time on.
    //
    // The three are built in **one** parallel pass rather than three sequential ones, and that
    // is the difference between 341 ms and 26 ms at five million rows a side. Each was a
    // separate `order2.iter().map(...).collect()`, so `order2` — 40 MB at that size — was
    // streamed three times to produce three arrays indexed identically, on one core of a
    // box with 96. Fusing them reads it once; chunking hands the gathers to rayon.
    //
    // `par_chunks`/`par_chunks_mut` are zipped at a common chunk length, so chunk `c` of every
    // output lines up with chunk `c` of `order2` and each element lands at the index it had
    // before. That is what keeps this a pure reindexing: the arrays are byte-for-byte what the
    // sequential passes produced, so no sweep behaviour and no join result can move.
    //
    // The scattered reads inside a chunk (`drank2[e]`, `pos1[e]`, `first_at[..]`) are the cost
    // that parallelizes; they are random over the whole universe, so there is nothing to gain
    // from a smaller chunk and the chunk length is set for scheduling granularity alone.
    let bump = usize::from(ops[0].strict());
    let mut drank2_seq = vec![0u32; n];
    let mut lo_seq = vec![0u32; n];
    let mut mark_at = vec![0u32; n];
    order2
        .par_chunks(SETUP_CHUNK)
        .zip(drank2_seq.par_chunks_mut(SETUP_CHUNK))
        .zip(lo_seq.par_chunks_mut(SETUP_CHUNK))
        .zip(mark_at.par_chunks_mut(SETUP_CHUNK))
        .for_each(|(((entries, d2), lo), mark)| {
            for (j, &e) in entries.iter().enumerate() {
                let eu = e as usize;
                d2[j] = drank2[eu];
                // A left row carries an axis-1 suffix bound and is never marked; a right row
                // carries a mark bit and no bound. A strict condition starts at the next
                // rank's first position, a non-strict one at this rank's.
                if eu < nl {
                    lo[j] = first_at[drank1[eu] as usize + bump];
                    mark[j] = NOT_A_RIGHT_ROW;
                } else {
                    lo[j] = 0;
                    mark[j] = pos1[eu].load(AtomicOrdering::Relaxed);
                }
            }
        });

    let sweep = Sweep {
        nl,
        n,
        strict2: ops[1].strict(),
        all: out.needs_all_matches(),
        order1: &order1,
        order2: &order2,
        lo_seq: &lo_seq,
        drank2_seq: &drank2_seq,
        mark_at: &mark_at,
        lmap,
        rmap,
    };

    // Which `order2` positions hold left rows. Only these do any work; the rest are marked
    // by the cursor. Materializing them is what makes the slices below evenly sized.
    // rayon's `collect` into a `Vec` preserves iteration order, so this is the same ascending
    // slice of `order2` positions the sequential filter produced — which the sweep relies on
    // (`Sweep::run` documents that `at` must be contiguous and ascending).
    let left_at: Vec<u32> = (0..n as u32)
        .into_par_iter()
        .filter(|&i| (order2[i as usize] as usize) < nl)
        .collect();

    // Cap the worker count by how much work there is, rather than slicing 96 ways for
    // 40,000 rows and paying 96 mark rebuilds to save one pass.
    let workers = rayon::current_num_threads()
        .min(left_at.len() / PARALLEL_SWEEP_MIN_PER_WORKER)
        .min(SWEEP_MAX_WORKERS);
    if workers < 2 || n < PARALLEL_SWEEP_MIN_ROWS {
        sweep.run(&left_at, out);
        return Ok(());
    }
    let per = left_at.len().div_ceil(workers);
    // Each worker builds the marked set as of its own slice's first left row — a binary
    // search plus a sequential pass over the `mark_at` prefix it needs. That is `O(n)` of
    // *work* per worker, but it happens on all of them at once, so it costs `O(n)` of wall
    // clock once.
    //
    // Two asymptotically cheaper schemes were tried and both lost, which is why this stays:
    //
    // - Walk the cursor once sequentially and snapshot the bitmap at each boundary — `O(n)`
    //   instead of `O(workers x n)`. **20% slower** at two million rows: the snapshots are
    //   serialized bitmap copies where the rebuilds are not.
    // - Build each segment's marks in parallel and combine them by a prefix union — the
    //   marks are then set once in total. **~10% slower**, because each segment needs a
    //   bitmap sized to the whole universe, and zeroing `workers + 1` of those costs more
    //   than the sequential rescan it replaces.
    //
    // The rescan reads a flat `u32` array in order and the branch predicts; that is hard to
    // beat with anything that allocates.
    //
    // Slices are contiguous in `order2` order and folded back in slice order, so the output
    // is identical to the sequential sweep's — not merely equivalent.
    let parts: Vec<Out> = left_at
        .par_chunks(per)
        .map(|slice| {
            let mut o = out.sibling(slice.len());
            sweep.run(slice, &mut o);
            o
        })
        .collect();
    for part in parts {
        out.absorb(part);
    }
    Ok(())
}

/// Everything the axis-2 sweep reads. Bundled so a worker can be handed a slice of the left
/// rows and nothing else.
///
/// Every field here is a `u32` array. That is the point: after [`dense_ranks`] the sweep
/// never touches an encoded key again, so its inner loop is integer compares over flat
/// slices rather than `arrow::row::Row` derivations and `memcmp`s.
struct Sweep<'a> {
    nl: usize,
    n: usize,
    strict2: bool,
    all: bool,
    order1: &'a [u32],
    order2: &'a [u32],
    /// `order2` position -> that left row's axis-1 suffix bound (unused for a right row).
    lo_seq: &'a [u32],
    /// `order2` position -> that entry's dense axis-2 rank.
    drank2_seq: &'a [u32],
    /// `order2` position -> the mark bit for that entry, or [`NOT_A_RIGHT_ROW`].
    mark_at: &'a [u32],
    lmap: &'a [u32],
    rmap: &'a [u32],
}

impl Sweep<'_> {
    /// Answer every left row at the `order2` positions in `at`, appending to `out`.
    ///
    /// `at` must be a contiguous, ascending slice of the left rows' `order2` positions.
    fn run(&self, at: &[u32], out: &mut Out) {
        let Some(&first) = at.first() else {
            return;
        };
        // The marked set as of this slice's first left row: every right row whose axis-2 key
        // already satisfies condition 2 against it. The bound is monotone in the sweep, so
        // one binary search finds it and one pass sets the bits.
        let start = self.drank2_seq[first as usize];
        let cursor = if self.strict2 {
            self.drank2_seq.partition_point(|&r| r < start)
        } else {
            self.drank2_seq.partition_point(|&r| r <= start)
        };
        let mut marks = MarkSet::new(self.n);
        for &bit in &self.mark_at[..cursor] {
            if bit != NOT_A_RIGHT_ROW {
                marks.set(bit as usize);
            }
        }
        self.run_from(at, cursor, marks, out);
    }

    /// [`Self::run`], from a marking cursor and mark set the caller already has.
    fn run_from(&self, at: &[u32], mut cursor: usize, mut marks: MarkSet, out: &mut Out) {
        for &idx in at {
            let e = self.order2[idx as usize];
            // Grow the marked prefix to every right row satisfying condition 2 against this
            // left key. The bound is monotone, so the cursor never rewinds and the whole
            // marking phase is one linear pass over the slice's span of `order2`.
            let ke2 = self.drank2_seq[idx as usize];
            while cursor < self.n {
                let kf2 = self.drank2_seq[cursor];
                let satisfies = if self.strict2 { kf2 < ke2 } else { kf2 <= ke2 };
                if !satisfies {
                    break;
                }
                let bit = self.mark_at[cursor];
                if bit != NOT_A_RIGHT_ROW {
                    marks.set(bit as usize);
                }
                cursor += 1;
            }

            // Condition 1 admits an axis-1 suffix; its start was precomputed.
            let lo = self.lo_seq[idx as usize] as usize;

            let mut matched = false;
            let mut scan = lo;
            while let Some(bit) = marks.next_set(scan) {
                matched = true;
                if !self.all {
                    break;
                }
                let f = self.order1[bit] as usize;
                out.pair(self.lmap[e as usize], self.rmap[f - self.nl]);
                scan = bit + 1;
            }
            out.finish_left(self.lmap[e as usize], matched);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int64Array, StringArray};

    use super::*;

    /// Deterministic xorshift, so the fuzz cases are reproducible without a dev-dep.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    fn cmp_op(op: RangeOp, a: i64, b: i64) -> bool {
        match op {
            RangeOp::Lt => a < b,
            RangeOp::Le => a <= b,
            RangeOp::Gt => a > b,
            RangeOp::Ge => a >= b,
        }
    }

    /// The oracle: the materialized cross product plus the predicate, which is exactly the
    /// plan this operator replaces.
    fn brute(
        l: &[Option<i64>],
        r: &[Option<i64>],
        ops: &[RangeOp],
        l2: Option<&[Option<i64>]>,
        r2: Option<&[Option<i64>]>,
        join_type: JoinType,
    ) -> Vec<(Option<u32>, Option<u32>)> {
        let mut pairs: Vec<(Option<u32>, Option<u32>)> = Vec::new();
        let mut right_matched = vec![false; r.len()];
        for (i, lv) in l.iter().enumerate() {
            let mut matched = false;
            for (j, rv) in r.iter().enumerate() {
                let ok = match (lv, rv) {
                    (Some(a), Some(b)) => {
                        let first = cmp_op(ops[0], *a, *b);
                        let second = match (l2, r2) {
                            (Some(l2), Some(r2)) => match (l2[i], r2[j]) {
                                (Some(a2), Some(b2)) => cmp_op(ops[1], a2, b2),
                                _ => false,
                            },
                            _ => true,
                        };
                        first && second
                    }
                    _ => false,
                };
                if ok {
                    matched = true;
                    right_matched[j] = true;
                    if matches!(
                        join_type,
                        JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full
                    ) {
                        pairs.push((Some(i as u32), Some(j as u32)));
                    }
                }
            }
            match join_type {
                JoinType::Semi if matched => pairs.push((Some(i as u32), None)),
                JoinType::Anti if !matched => pairs.push((Some(i as u32), None)),
                JoinType::Left | JoinType::Full if !matched => pairs.push((Some(i as u32), None)),
                _ => {}
            }
        }
        if matches!(join_type, JoinType::Right | JoinType::Full) {
            for (j, &m) in right_matched.iter().enumerate() {
                if !m {
                    pairs.push((None, Some(j as u32)));
                }
            }
        }
        pairs.sort_unstable();
        pairs
    }

    fn actual(idx: &JoinIndices) -> Vec<(Option<u32>, Option<u32>)> {
        let mut v: Vec<(Option<u32>, Option<u32>)> = (0..idx.left.len())
            .map(|i| {
                (
                    idx.left.is_valid(i).then(|| idx.left.value(i)),
                    idx.right.is_valid(i).then(|| idx.right.value(i)),
                )
            })
            .collect();
        v.sort_unstable();
        v
    }

    fn arr(v: &[Option<i64>]) -> ArrayRef {
        Arc::new(Int64Array::from(v.to_vec()))
    }

    const OPS: [RangeOp; 4] = [RangeOp::Lt, RangeOp::Le, RangeOp::Gt, RangeOp::Ge];
    const TYPES: [JoinType; 6] = [
        JoinType::Inner,
        JoinType::Left,
        JoinType::Right,
        JoinType::Full,
        JoinType::Semi,
        JoinType::Anti,
    ];

    #[test]
    fn one_inequality_matches_the_cross_product_oracle() {
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        for trial in 0..40 {
            let nl = 1 + (rng.next() % 30) as usize;
            let nr = 1 + (rng.next() % 30) as usize;
            let l: Vec<Option<i64>> = (0..nl)
                .map(|_| {
                    let v = rng.next();
                    (v % 7 != 0).then(|| (v % 20) as i64 - 10)
                })
                .collect();
            let r: Vec<Option<i64>> = (0..nr)
                .map(|_| {
                    let v = rng.next();
                    (v % 9 != 0).then(|| (v % 20) as i64 - 10)
                })
                .collect();
            for op in OPS {
                for jt in TYPES {
                    let got =
                        range_join_indices(&[arr(&l)], &[arr(&r)], &[op], jt).expect("join runs");
                    assert_eq!(
                        actual(&got),
                        brute(&l, &r, &[op], None, None, jt),
                        "trial {trial} op {op:?} join {jt:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn two_inequalities_match_the_cross_product_oracle() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for trial in 0..40 {
            let nl = 1 + (rng.next() % 25) as usize;
            let nr = 1 + (rng.next() % 25) as usize;
            let mk = |rng: &mut Rng, n: usize, nullmod: u64| -> Vec<Option<i64>> {
                (0..n)
                    .map(|_| {
                        let v = rng.next();
                        (v % nullmod != 0).then(|| (v % 14) as i64 - 7)
                    })
                    .collect()
            };
            let la = mk(&mut rng, nl, 8);
            let lb = mk(&mut rng, nl, 11);
            let ra = mk(&mut rng, nr, 9);
            let rb = mk(&mut rng, nr, 13);
            for op1 in OPS {
                for op2 in OPS {
                    for jt in TYPES {
                        let got = range_join_indices(
                            &[arr(&la), arr(&lb)],
                            &[arr(&ra), arr(&rb)],
                            &[op1, op2],
                            jt,
                        )
                        .expect("join runs");
                        assert_eq!(
                            actual(&got),
                            brute(&la, &ra, &[op1, op2], Some(&lb), Some(&rb), jt),
                            "trial {trial} ops {op1:?}/{op2:?} join {jt:?}"
                        );
                    }
                }
            }
        }
    }

    /// The band shape — `L.a OP R.y` and `L.b OP R.y` over the **same** right array — against
    /// the cross-product oracle, across every operator pair, join type, and null pattern.
    ///
    /// The pre-existing two-inequality test cannot reach this path: it builds two *different*
    /// right arrays, so `band::bounds` declines and IEJoin runs. Passing one `ArrayRef` twice
    /// is what a real `r.y BETWEEN l.a AND l.b` lowers to, because `columns_by_name` hands out
    /// `Arc` clones of one column.
    #[test]
    fn a_band_over_one_right_key_matches_the_cross_product_oracle() {
        let mut rng = Rng(0x243F_6A88_85A3_08D3);
        for trial in 0..40 {
            let nl = 1 + (rng.next() % 25) as usize;
            let nr = 1 + (rng.next() % 25) as usize;
            let mk = |rng: &mut Rng, n: usize, nullmod: u64| -> Vec<Option<i64>> {
                (0..n)
                    .map(|_| {
                        let v = rng.next();
                        (v % nullmod != 0).then(|| (v % 14) as i64 - 7)
                    })
                    .collect()
            };
            let la = mk(&mut rng, nl, 8);
            let lb = mk(&mut rng, nl, 11);
            let ry = mk(&mut rng, nr, 9);
            // ONE array, cloned — this is what makes `Arc::ptr_eq` hold and the band fire.
            let ry_arr = arr(&ry);
            for op1 in OPS {
                for op2 in OPS {
                    for jt in TYPES {
                        let got = range_join_indices(
                            &[arr(&la), arr(&lb)],
                            &[ry_arr.clone(), ry_arr.clone()],
                            &[op1, op2],
                            jt,
                        )
                        .expect("join runs");
                        assert_eq!(
                            actual(&got),
                            brute(&la, &ry, &[op1, op2], Some(&lb), Some(&ry), jt),
                            "trial {trial} ops {op1:?}/{op2:?} join {jt:?}"
                        );
                    }
                }
            }
        }
    }

    /// The band path and IEJoin must agree **on the same input**, not merely each against the
    /// oracle. Handing the identical data as one shared array (band) and as two equal-valued
    /// arrays (IEJoin) is the only way to compare the two algorithms directly, and it is what
    /// would catch a band that is self-consistently wrong.
    #[test]
    fn the_band_path_agrees_with_iejoin_on_the_same_data() {
        let mut rng = Rng(0xB7E1_5162_8AED_2A6A);
        for _ in 0..25 {
            let (nl, nr) = (
                1 + (rng.next() % 30) as usize,
                1 + (rng.next() % 30) as usize,
            );
            let mk = |rng: &mut Rng, n: usize| -> Vec<Option<i64>> {
                (0..n)
                    .map(|_| {
                        let v = rng.next();
                        (v % 7 != 0).then(|| (v % 20) as i64 - 10)
                    })
                    .collect()
            };
            let la = mk(&mut rng, nl);
            let lb = mk(&mut rng, nl);
            let ry = mk(&mut rng, nr);
            let shared = arr(&ry);
            for op1 in OPS {
                for op2 in OPS {
                    for jt in TYPES {
                        // Same Arc twice -> band (when the ops face opposite ways).
                        let band = range_join_indices(
                            &[arr(&la), arr(&lb)],
                            &[shared.clone(), shared.clone()],
                            &[op1, op2],
                            jt,
                        )
                        .expect("band runs");
                        // Two separately-built arrays of the same values -> IEJoin.
                        let ie = range_join_indices(
                            &[arr(&la), arr(&lb)],
                            &[arr(&ry), arr(&ry)],
                            &[op1, op2],
                            jt,
                        )
                        .expect("iejoin runs");
                        assert_eq!(
                            actual(&band),
                            actual(&ie),
                            "band vs IEJoin disagree: ops {op1:?}/{op2:?} join {jt:?}"
                        );
                    }
                }
            }
        }
    }

    /// Detection must decline everything it cannot prove is a band, because a false positive
    /// would slice an order the second bound does not index.
    #[test]
    fn band_detection_declines_what_is_not_a_band() {
        let y = arr(&[Some(1i64), Some(2), Some(3)]);
        let z = arr(&[Some(1i64), Some(2), Some(3)]);
        // Opposite-facing ops over one shared key: this IS a band.
        assert_eq!(
            band::bounds(&[y.clone(), y.clone()], &[RangeOp::Le, RangeOp::Ge]),
            Some((0, 1))
        );
        assert_eq!(
            band::bounds(&[y.clone(), y.clone()], &[RangeOp::Gt, RangeOp::Lt]),
            Some((1, 0))
        );
        // Same values, different arrays: not provably one key, so decline.
        assert_eq!(
            band::bounds(&[y.clone(), z], &[RangeOp::Le, RangeOp::Ge]),
            None
        );
        // One shared key but both conditions face the same way: not a band.
        assert_eq!(
            band::bounds(&[y.clone(), y.clone()], &[RangeOp::Le, RangeOp::Lt]),
            None
        );
        assert_eq!(
            band::bounds(&[y.clone(), y.clone()], &[RangeOp::Ge, RangeOp::Gt]),
            None
        );
    }

    /// Floats reach the band through `canonicalize_float_keys`, which rebuilds the arrays —
    /// so the shared-`Arc` test can fail on them and IEJoin runs instead. Either way the
    /// answer must obey the engine's total float order, so pin the answer, not the route.
    #[test]
    fn a_float_band_follows_the_engines_total_order() {
        let la = vec![Some(-0.0f64), Some(1.0), Some(f64::NAN)];
        let lb = vec![Some(2.0f64), Some(3.0), Some(f64::NAN)];
        let ry = vec![Some(0.0f64), Some(2.0), Some(f64::NAN)];
        let f = |v: &[Option<f64>]| -> ArrayRef {
            std::sync::Arc::new(arrow::array::Float64Array::from(v.to_vec())) as ArrayRef
        };
        let shared = f(&ry);
        let got = range_join_indices(
            &[f(&la), f(&lb)],
            &[shared.clone(), shared.clone()],
            &[RangeOp::Le, RangeOp::Ge],
            JoinType::Inner,
        )
        .expect("join runs");
        // -0.0 <= 0.0 <= 2.0 and 1.0 <= 2.0 <= 3.0; NaN equals NaN under the total order,
        // so the NaN row's band [NaN, NaN] contains the NaN right row.
        let mut pairs = actual(&got);
        pairs.sort_unstable();
        assert_eq!(
            pairs,
            vec![
                (Some(0), Some(0)),
                (Some(0), Some(1)),
                (Some(1), Some(1)),
                (Some(2), Some(2)),
            ]
        );
    }

    #[test]
    fn interval_containment_is_output_sensitive() {
        // The shape the ceiling was measured on: points against [lo, hi) intervals. What
        // this pins is the *answer*, not the speed — that a 4,000 x 4,000 join returning a
        // few thousand rows runs at all inside a unit test is the point.
        let n = 4_000usize;
        let points: Vec<Option<i64>> = (0..n).map(|i| Some((i as i64 * 7919) % 40_000)).collect();
        let lo: Vec<Option<i64>> = (0..n).map(|i| Some((i as i64 * 13) % 40_000)).collect();
        let hi: Vec<Option<i64>> = lo.iter().map(|v| Some(v.unwrap() + 100)).collect();

        // pt.x >= iv.lo AND pt.x < iv.hi
        let got = range_join_indices(
            &[arr(&points), arr(&points)],
            &[arr(&lo), arr(&hi)],
            &[RangeOp::Ge, RangeOp::Lt],
            JoinType::Inner,
        )
        .expect("join runs");

        let mut expect = 0usize;
        for p in points.iter().flatten() {
            for (l, h) in lo.iter().flatten().zip(hi.iter().flatten()) {
                if p >= l && p < h {
                    expect += 1;
                }
            }
        }
        assert_eq!(got.left.len(), expect);
        assert!(expect > 0, "the fixture must actually match something");
    }

    #[test]
    fn floats_follow_the_engines_total_order() {
        // The contract is `bc_arrow::float_ident`: every NaN equal and ranked above every
        // number, `-0.0` equal to `0.0`. Not IEEE — under IEEE every pair below involving
        // a NaN would drop out, which is exactly the divergence from the
        // cross-product-plus-filter plan this test exists to catch.
        let l: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(f64::NAN),
            Some(-0.0),
            Some(1.0),
        ]));
        let r: ArrayRef = Arc::new(Float64Array::from(vec![Some(0.0), Some(f64::NAN)]));
        let got = range_join_indices(&[l], &[r], &[RangeOp::Le], JoinType::Inner).expect("runs");
        assert_eq!(
            actual(&got),
            vec![
                (Some(0), Some(1)), // NaN <= NaN
                (Some(1), Some(0)), // -0.0 <= 0.0
                (Some(1), Some(1)), // -0.0 <= NaN
                (Some(2), Some(1)), //  1.0 <= NaN
            ]
        );
    }

    #[test]
    fn strings_compare_in_byte_order() {
        let l: ArrayRef = Arc::new(StringArray::from(vec!["apple", "pear", "zebra"]));
        let r: ArrayRef = Arc::new(StringArray::from(vec!["banana", "quince"]));
        let got = range_join_indices(&[l], &[r], &[RangeOp::Lt], JoinType::Inner).expect("runs");
        assert_eq!(
            actual(&got),
            vec![(Some(0), Some(0)), (Some(0), Some(1)), (Some(1), Some(1))]
        );
    }

    #[test]
    fn an_empty_side_yields_no_pairs_but_keeps_outer_rows() {
        let l = vec![Some(1i64), None, Some(3)];
        let empty: Vec<Option<i64>> = Vec::new();
        let got = range_join_indices(&[arr(&l)], &[arr(&empty)], &[RangeOp::Lt], JoinType::Left)
            .expect("runs");
        assert_eq!(
            actual(&got),
            vec![(Some(0), None), (Some(1), None), (Some(2), None)]
        );
        let got = range_join_indices(&[arr(&l)], &[arr(&empty)], &[RangeOp::Lt], JoinType::Inner)
            .expect("runs");
        assert_eq!(got.left.len(), 0);
    }

    #[test]
    fn a_mismatched_key_type_is_declined_rather_than_guessed() {
        let l: ArrayRef = Arc::new(Int64Array::from(vec![1i64]));
        let r: ArrayRef = Arc::new(Float64Array::from(vec![1.0f64]));
        assert!(range_join_indices(&[l], &[r], &[RangeOp::Lt], JoinType::Inner).is_err());
    }

    #[test]
    fn the_parallel_paths_agree_with_an_analytic_answer() {
        // Above `PARALLEL_SORT_MIN_ROWS` the axis sorts fan out, and above
        // `PARALLEL_SWEEP_MIN_ROWS` so does the sweep, so this is the only test that runs
        // either. The expected answer is computed in closed form rather than by brute
        // force: at this size the cross-product oracle would be 10^10 comparisons.
        const N: usize = 40_000; // universe = 80,000 entries, past both thresholds
        const WIDTH: i64 = 5;
        let points: Vec<Option<i64>> = (0..N as i64).map(Some).collect();
        let lo: Vec<Option<i64>> = (0..N as i64).map(Some).collect();
        let hi: Vec<Option<i64>> = (0..N as i64).map(|v| Some(v + WIDTH)).collect();

        // pt.x >= iv.lo AND pt.x < iv.hi, with x = i and [lo, hi) = [j, j + WIDTH):
        // point i matches every j in (i - WIDTH, i], i.e. min(WIDTH, i + 1) of them.
        let expected: usize = (0..N as i64).map(|i| (WIDTH.min(i + 1)) as usize).sum();

        let got = range_join_indices(
            &[arr(&points), arr(&points)],
            &[arr(&lo), arr(&hi)],
            &[RangeOp::Ge, RangeOp::Lt],
            JoinType::Inner,
        )
        .expect("join runs");
        assert_eq!(got.left.len(), expected);

        // And the pairs themselves, not just the count.
        for i in 0..got.left.len() {
            let (p, iv) = (got.left.value(i) as i64, got.right.value(i) as i64);
            assert!(
                p >= iv && p < iv + WIDTH,
                "pair ({p}, {iv}) does not satisfy"
            );
        }

        // The parallel sweep must produce the *same rows in the same order* as the
        // sequential one, not merely the same relation — its slices are contiguous in
        // axis-2 order and folded back in slice order precisely so that holds. A
        // single-threaded rayon pool takes the sequential branch, which is what makes the
        // two comparable at all.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("single-thread pool");
        let sequential = pool
            .install(|| {
                range_join_indices(
                    &[arr(&points), arr(&points)],
                    &[arr(&lo), arr(&hi)],
                    &[RangeOp::Ge, RangeOp::Lt],
                    JoinType::Inner,
                )
            })
            .expect("join runs");
        assert_eq!(got.left, sequential.left);
        assert_eq!(got.right, sequential.right);
    }

    /// The band's parallel cursor walk must equal the sequential one row for row.
    ///
    /// Nothing else covers it, and the gap was structural rather than an oversight:
    /// `a_band_over_one_right_key_matches_the_cross_product_oracle` runs at 25 rows a side, and
    /// `the_parallel_paths_agree_with_an_analytic_answer` hands the two conditions *different*
    /// right arrays, so `Arc::ptr_eq` fails and it routes to IEJoin. The band's parallel branch
    /// does not engage below `2 * PARALLEL_MIN_PER_WORKER` left rows, so both of those run the
    /// sequential merge and a broken seek would pass every band test in the file.
    ///
    /// The walk is split by seeking each chunk's starting cursor with a binary search, which is
    /// exact only because the walk is monotone over a sorted array. **Heavy key duplication is
    /// the point of this data**: `skip_equal` decides whether an equal-key run is skipped or
    /// kept, so a chunk boundary landing inside such a run is precisely where a wrong seek
    /// would surface. At ~250 rows per distinct key, boundaries land inside runs constantly.
    ///
    /// All four strictness combinations are bands (`Lt`/`Le` bound the right key from below,
    /// `Gt`/`Ge` from above), and each is checked against a single-threaded rayon pool, which
    /// takes the sequential branch and is what makes the two comparable.
    ///
    /// This test was mutation-checked rather than assumed to have teeth, and the result is worth
    /// knowing: replacing the seek with `partition_point(|y| y <= lk)` fails it on the first
    /// combination, while `partition_point(|y| y < lk)` **passes** — because the walk only
    /// advances, so an undershooting seek is corrected by it and is merely slower. Only an
    /// overshoot is a wrong answer, which is the property the comment on the seek records.
    #[test]
    fn the_parallel_band_merge_agrees_with_the_sequential_one() {
        const N: usize = 50_000;
        const SPAN: u64 = 200;

        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let ry: Vec<Option<i64>> = (0..N).map(|_| Some((rng.next() % SPAN) as i64)).collect();
        let la: Vec<Option<i64>> = (0..N).map(|_| Some((rng.next() % SPAN) as i64)).collect();
        let lb: Vec<Option<i64>> = la.iter().map(|v| Some(v.unwrap() + 3)).collect();
        // ONE array, cloned, so `Arc::ptr_eq` holds and the band path fires.
        let ry_arr = arr(&ry);

        for (op_lo, op_hi) in [
            (RangeOp::Le, RangeOp::Ge),
            (RangeOp::Lt, RangeOp::Ge),
            (RangeOp::Le, RangeOp::Gt),
            (RangeOp::Lt, RangeOp::Gt),
        ] {
            let run = || {
                range_join_indices(
                    &[arr(&la), arr(&lb)],
                    &[ry_arr.clone(), ry_arr.clone()],
                    &[op_lo, op_hi],
                    JoinType::Inner,
                )
                .expect("join runs")
            };

            let parallel = run();
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("single-thread pool");
            let sequential = pool.install(run);

            assert_eq!(
                parallel.left, sequential.left,
                "{op_lo:?}/{op_hi:?}: left indices differ between the parallel and \
                 sequential band merge"
            );
            assert_eq!(
                parallel.right, sequential.right,
                "{op_lo:?}/{op_hi:?}: right indices differ between the parallel and \
                 sequential band merge"
            );
            // A shape that matched nothing would make the equality above vacuous.
            assert!(
                !parallel.left.is_empty(),
                "{op_lo:?}/{op_hi:?}: the band matched no rows, so this proves nothing"
            );
        }
    }

    #[test]
    fn the_mark_set_finds_every_bit_across_summary_blocks() {
        // Every level boundary: a word (64), and then each successive summary's span
        // (4,096, 262,144, 16,777,216). A skip that overshoots or stalls shows up at exactly
        // these points, and the span is chosen so the bitmap has four levels.
        let bits = 20_000_000;
        let mut m = MarkSet::new(bits);
        let set: Vec<usize> = vec![
            0,
            1,
            63,
            64,
            4_095,
            4_096,
            262_143,
            262_144,
            262_145,
            16_777_215,
            16_777_216,
            16_777_217,
            bits - 1,
        ];
        for &b in &set {
            m.set(b);
        }
        let mut found = Vec::new();
        let mut from = 0;
        while let Some(b) = m.next_set(from) {
            found.push(b);
            from = b + 1;
        }
        assert_eq!(found, set);
        assert_eq!(m.next_set(bits), None);
        assert_eq!(m.next_set(16_777_218), Some(bits - 1));
        // A wholly empty set must terminate, not walk.
        assert_eq!(MarkSet::new(bits).next_set(0), None);
    }

    /// Where a two-condition range join's time actually goes, by phase.
    ///
    /// Committed rather than remembered, per this repo's convention for timing studies. Run:
    ///
    /// ```text
    /// cargo test --release -p bc-runtime --lib report_range_join_phases -- --ignored --nocapture
    /// ```
    ///
    /// It exists because a plausible hypothesis about the bottleneck (the mark-array scan)
    /// was measured and found to be wrong: adding a third summary level, which cuts that
    /// term 64x, moved a 2,000,000-row join by 1%. The sorts were 70% of it, which is what
    /// [`AxisKeys`] and [`dense_ranks`] were then written against.
    #[test]
    #[ignore = "timing study, not an assertion"]
    fn report_range_join_phases() {
        use std::time::Instant;

        for n in [500_000usize, 2_000_000, 5_000_000] {
            let mut rng = Rng(0x51ED_270B_6989_1A0D);
            let points: Vec<Option<i64>> = (0..n)
                .map(|_| Some((rng.next() % (10 * n as u64)) as i64))
                .collect();
            let lo: Vec<Option<i64>> = (0..n)
                .map(|_| Some((rng.next() % (10 * n as u64)) as i64))
                .collect();
            let hi: Vec<Option<i64>> = lo.iter().map(|v| Some(v.unwrap() + 10)).collect();
            let (lk, rk) = (vec![arr(&points), arr(&points)], vec![arr(&lo), arr(&hi)]);
            let ops = [RangeOp::Ge, RangeOp::Lt];

            let lmap: Vec<u32> = (0..n as u32).collect();
            let rmap: Vec<u32> = (0..n as u32).collect();
            let (nl, un) = (n, 2 * n);

            let t = Instant::now();
            let k1 =
                AxisKeys::build(&lk[0], &rk[0], ops[0].axis1_descending(), &lmap, &rmap).unwrap();
            let k2 =
                AxisKeys::build(&lk[1], &rk[1], ops[1].axis2_descending(), &lmap, &rmap).unwrap();
            let encode = t.elapsed();

            let t = Instant::now();
            let (_o1, _r1) = k1.sorted_order_and_ranks(un, nl, &lmap, &rmap);
            let sort1 = t.elapsed();

            let t = Instant::now();
            let (_o2, _r2) = k2.sorted_order_and_ranks(un, nl, &lmap, &rmap);
            let sort2 = t.elapsed();

            let t = Instant::now();
            let total = range_join_indices(&lk, &rk, &ops, JoinType::Inner).unwrap();
            let whole = t.elapsed();

            // The same join with intervals that match nothing: everything except emitting
            // and buffering the pairs. `whole - empty` is what the output costs.
            let empty_hi: Vec<Option<i64>> = lo.iter().map(|v| Some(v.unwrap())).collect();
            let rk_empty = vec![arr(&lo), arr(&empty_hi)];
            let t = Instant::now();
            let none = range_join_indices(&lk, &rk_empty, &ops, JoinType::Inner).unwrap();
            let no_out = t.elapsed();
            assert_eq!(none.left.len(), 0);

            println!(
                "n={n:>9}  keys={:>7.1?}  sort1={:>7.1?}  sort2={:>7.1?}  \
                 no_output={:>7.1?}  whole={:>7.1?}  out_rows={}",
                encode,
                sort1,
                sort2,
                no_out,
                whole,
                total.left.len()
            );
        }
    }

    /// What the band path is worth against the IEJoin it replaces, on identical data.
    ///
    /// ```text
    /// cargo test -p bc-runtime --release --lib report_band_vs_iejoin -- --ignored --nocapture
    /// ```
    ///
    /// Same rows, same predicate, same answer — the only difference is whether the two
    /// conditions are handed the right key as one shared `ArrayRef` (band) or as two
    /// equal-valued arrays (IEJoin). Equality of the two answers is asserted here too, so a
    /// speed number can never be reported for a path that stopped agreeing.
    /// Where the band path's own time goes: encode, sorts, merges, emission.
    ///
    /// ```text
    /// cargo test -p bc-runtime --release --lib report_band_phases -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "timing study, not an assertion"]
    fn report_band_phases() {
        use super::keys::AxisKeys;
        use std::time::Instant;

        for n in [2_000_000usize, 5_000_000] {
            let mut rng = Rng(0x51ED_270B_6989_1A0D);
            let span = 50 * n as u64;
            let lo: Vec<Option<i64>> = (0..n).map(|_| Some((rng.next() % span) as i64)).collect();
            let hi: Vec<Option<i64>> = lo.iter().map(|v| Some(v.unwrap() + 20)).collect();
            let ry: Vec<Option<i64>> = (0..n).map(|_| Some((rng.next() % span) as i64)).collect();
            let (la, lb, shared) = (arr(&lo), arr(&hi), arr(&ry));
            let lmap: Vec<u32> = (0..n as u32).collect();
            let rmap: Vec<u32> = (0..n as u32).collect();
            let (nl, un) = (n, 2 * n);

            let t = Instant::now();
            let k_lo = AxisKeys::build(&la, &shared, false, &lmap, &rmap).unwrap();
            let _k_hi = AxisKeys::build(&lb, &shared, false, &lmap, &rmap).unwrap();
            let encode = t.elapsed();

            let t = Instant::now();
            let order = k_lo.sorted_right(un, nl, &lmap, &rmap);
            let sort_r = t.elapsed();

            let t = Instant::now();
            let sl = k_lo.sorted_left(nl, &lmap, &rmap);
            let sort_l = t.elapsed();

            let t = Instant::now();
            let mut at = vec![0u32; nl];
            let mut p = 0usize;
            for &e in &sl {
                while p < order.len()
                    && k_lo.cmp(order[p], e, nl, &lmap, &rmap) == std::cmp::Ordering::Less
                {
                    p += 1;
                }
                at[e as usize] = p as u32;
            }
            let merge = t.elapsed();

            let t = Instant::now();
            let whole = range_join_indices(
                &[la.clone(), lb.clone()],
                &[shared.clone(), shared.clone()],
                &[RangeOp::Le, RangeOp::Ge],
                JoinType::Inner,
            )
            .unwrap();
            let total = t.elapsed();

            println!(
                "n={n:>9} encode={:>7.1?} sort_right={:>7.1?} sort_left={:>7.1?} \
                 merge1={:>7.1?} whole={:>7.1?} pairs={}",
                encode,
                sort_r,
                sort_l,
                merge,
                total,
                whole.left.len()
            );
        }
    }

    #[test]
    #[ignore = "timing study, not an assertion"]
    fn report_band_vs_iejoin() {
        use std::time::Instant;

        for n in [500_000usize, 2_000_000, 5_000_000] {
            let mut rng = Rng(0x51ED_270B_6989_1A0D);
            // Keys spread far wider than the band, so the answer stays sparse and the
            // measurement is the algorithm rather than the gather.
            let span = 50 * n as u64;
            let lo: Vec<Option<i64>> = (0..n).map(|_| Some((rng.next() % span) as i64)).collect();
            let hi: Vec<Option<i64>> = lo.iter().map(|v| Some(v.unwrap() + 20)).collect();
            let ry: Vec<Option<i64>> = (0..n).map(|_| Some((rng.next() % span) as i64)).collect();
            let ops = [RangeOp::Le, RangeOp::Ge];

            let shared = arr(&ry);
            let t = Instant::now();
            let band = range_join_indices(
                &[arr(&lo), arr(&hi)],
                &[shared.clone(), shared.clone()],
                &ops,
                JoinType::Inner,
            )
            .unwrap();
            let band_ms = t.elapsed();

            let t = Instant::now();
            let ie = range_join_indices(
                &[arr(&lo), arr(&hi)],
                &[arr(&ry), arr(&ry)],
                &ops,
                JoinType::Inner,
            )
            .unwrap();
            let ie_ms = t.elapsed();

            assert_eq!(
                actual(&band),
                actual(&ie),
                "band and IEJoin disagree at n={n}; the timings below would be meaningless"
            );
            println!(
                "n={n:>9}  band={:>8.1?}  iejoin={:>8.1?}  speedup={:>5.2}x  out_rows={}",
                band_ms,
                ie_ms,
                ie_ms.as_secs_f64() / band_ms.as_secs_f64(),
                band.left.len()
            );
        }
    }

    #[test]
    fn the_mark_set_matches_a_naive_bitmap() {
        // The levels are a skip structure, so the only thing that can go wrong is skipping
        // past a bit. Fuzzed against the obvious implementation over a span wide enough that
        // every level is exercised.
        let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D);
        for trial in 0..4 {
            let bits = 300_000 + (rng.next() % 20_000_000) as usize;
            let count = 1 + (rng.next() % 200) as usize;
            let mut m = MarkSet::new(bits);
            let mut naive = vec![false; bits];
            for _ in 0..count {
                let b = (rng.next() as usize) % bits;
                m.set(b);
                naive[b] = true;
            }
            let expect: Vec<usize> = (0..bits).filter(|&i| naive[i]).collect();
            let mut found = Vec::new();
            let mut from = 0;
            while let Some(b) = m.next_set(from) {
                found.push(b);
                from = b + 1;
            }
            assert_eq!(found, expect, "trial {trial} over {bits} bits");
        }
    }
}
