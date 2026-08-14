//! Group assignment for a key that arrives in sorted order — runs instead of a hash table.
//!
//! When rows are ordered by the group key, equal keys are adjacent, so the group a row belongs
//! to is decided by comparing it with its predecessor. No hash, no probe, no group table: the
//! assignment is a scan over runs of equal adjacent values. That is the shape a lakehouse table
//! with a declared sort key produces, and the shape any aggregate above a `Sort`, a sort-merge
//! join, or a time-ordered ingest sees.
//!
//! # Why this verifies rather than trusts
//!
//! Polars ships the same specialization (`nodes/sorted_group_by.rs`) and reaches it from a
//! *flag* on the column — a promise made by the planner or by the user. A promise is exactly
//! what this module refuses to take. Sortedness would otherwise come from `RelStats.sorted_by`,
//! which for a lakehouse table is a declaration in table metadata that nothing enforces on
//! write; believing it when it is false does not produce a slow answer, it produces a **wrong**
//! one, silently, because one key split across two non-adjacent runs is emitted as two groups.
//!
//! So the property is *established*, on every aggregate, whether or not anything declared it:
//!
//! 1. Adjacent rows are compared with arrow's vectorized kernels, **in chunks**, and the scan
//!    stops at the first chunk containing a violation. Unordered input is rejected after one
//!    chunk, which is what makes this affordable to attempt unconditionally.
//! 2. The same pass records where the key *changes*, which — given monotonicity — is exactly
//!    where one group ends and the next begins.
//!
//! The chunking is the whole reason step 1 is cheap, and it was not free to learn: a sampled
//! prefix gate looks equivalent and is not. A key cycling `0,1,…,99,0,1,…` is ordered across
//! any short prefix and ordered across most sampled pairs, so a gate passes it and pays for a
//! full detection pass that then fails — measured at **4x slower** than simply hashing. An
//! exact scan that exits at the first violating chunk cannot be fooled that way, and costs the
//! same on the input it rejects as the sample did.
//!
//! # What it is worth, measured end to end rather than at this function
//!
//! Read this before quoting a number from here. At the level of one `partial()` call over 6M
//! rows this path is worth **6.0x** on an all-distinct sorted integer key, 1.56x on a sorted
//! string key and 1.40x on a sorted composite one. **Almost none of that survives to the
//! query.** An A/B of two builds over the same data measured 1.0-1.2x on those same shapes,
//! and at full parallelism the difference sat inside the noise band.
//!
//! The reason is that the engine morselizes: it never makes the 6M-row call the microbenchmark
//! made. At 16,384 rows the hash paths are already good and this is worth 1.1-1.2x, which is a
//! few percent of a query that also scans, accumulates, combines and finalizes. The honest
//! claim is a **small win on sorted input and free on everything else** — rejection measured
//! 1.00x at 6M rows and 0.97x at morsel size, which is what makes it safe to attempt
//! unconditionally, and which is the property worth defending here rather than the 6x.
//!
//! This is the second time this document's mechanism-level probe disagreed with the query (see
//! `competitor_technique_review.md`, 10g). A ratio measured at the function is a fact about the
//! function.
//!
//! # What the caller gets
//!
//! The same `(group_ids, num_groups, group_columns)` triple every other path in
//! [`super::assign`] returns, with group ids in first-seen row order — so this is a pure
//! short-circuit under [`super::assign_groups`], and every accumulator, `combine` and
//! `finalize` downstream is untouched. Sortedness changes how a row finds its group, never
//! what the group computes.

use arrow::array::{Array, ArrayRef, BooleanArray, BooleanBufferBuilder};
use arrow::buffer::BooleanBuffer;
use arrow::compute::kernels::boolean::{and, not, or};
use arrow::compute::kernels::cmp;
use arrow::datatypes::DataType;

use crate::error::RuntimeError;

/// What every group-assignment path returns: each row's dense group id, the group count, and
/// the distinct key columns in first-seen order.
type Assignment = (Vec<u32>, usize, Vec<ArrayRef>);

/// Rows compared in the *first* vectorized step, and so the most work a rejected key can cost.
///
/// Deliberately far below a morsel. Rejection is the common case and it is paid per batch, so
/// a first step of one morsel would charge the streaming path for scanning half a 16,384-row
/// batch before declining — measured at 0.81x on a cyclic key, where the same scan at 6M rows
/// amortizes to nothing and reads 1.00x. Shrinking the first step to 1,024 took that to 0.95x
/// and to 256 took it to 0.97x, with every other shape unchanged.
const FIRST_CHUNK: usize = 256;

/// The largest step, once a key has earned it.
///
/// The step doubles after each clean chunk, so rejection stays priced at [`FIRST_CHUNK`] while
/// input that keeps verifying reaches a step whose per-kernel dispatch is amortized away.
///
/// The cap is what matters and it is **not** "as large as possible": the step's temporaries are
/// six boolean arrays, and holding them in cache is worth more than saving dispatches. Growing
/// to 1M rows measured *slower* than a flat 8,192 on every sorted shape (1.40x against 1.46x on
/// a composite key, 32.4 ms against 30.0 ms on a low-cardinality integer one), consistently
/// across three runs. 8,192 is the largest step that stayed cache-resident here.
const MAX_CHUNK: usize = 8_192;

/// Which key columns this module will compare.
///
/// Nested types are out because arrow compares them with a per-row dynamic comparator, which is
/// the cost this path exists to avoid. A dictionary is out for a correctness reason instead: its
/// codes carry the sort order only when the dictionary is ordered, which nothing here can
/// establish, and [`super::assign`] already groups dictionary codes without hashing values.
/// Keys containing nulls are out because a null compares as null under `lt`, which this reads as
/// unordered — declining is a missed optimization, never a wrong answer.
fn comparable(a: &ArrayRef) -> bool {
    !a.data_type().is_nested()
        && !matches!(a.data_type(), DataType::Dictionary(..) | DataType::Null)
        && a.null_count() == 0
}

/// Is every bit set?
fn all(b: &BooleanArray) -> bool {
    b.values().count_set_bits() == b.len()
}

/// Is no bit set?
fn none(b: &BooleanArray) -> bool {
    b.values().count_set_bits() == 0
}

/// Compare rows `off..off+len` against their successors, lexicographically across all key
/// columns, returning `(strictly_less, equal)` for each pair.
///
/// Built by folding the columns front to back: a pair is *less* at the first column where it
/// differs, so column `j` contributes only where every column before it was equal.
fn compare_adjacent(
    keys: &[ArrayRef],
    off: usize,
    len: usize,
) -> Result<(BooleanArray, BooleanArray), RuntimeError> {
    let mut prefix_eq: Option<BooleanArray> = None;
    let mut less: Option<BooleanArray> = None;
    for key in keys {
        let a = key.slice(off, len);
        let b = key.slice(off + 1, len);
        let eq_j = cmp::not_distinct(&a, &b)?;
        let lt_j = cmp::lt(&a, &b)?;
        let term = match &prefix_eq {
            Some(p) => and(p, &lt_j)?,
            None => lt_j,
        };
        less = Some(match less {
            Some(acc) => or(&acc, &term)?,
            None => term,
        });
        prefix_eq = Some(match prefix_eq {
            Some(p) => and(&p, &eq_j)?,
            None => eq_j,
        });
    }
    // `keys` is non-empty at every call site, so both accumulators are set.
    Ok((
        less.expect("a key column"),
        prefix_eq.expect("a key column"),
    ))
}

/// Where the key changes, when the key is verifiably monotonic.
///
/// Bit `i` of the result is set when row `i + 1` starts a new group. `None` means the ordering
/// does not hold, and the caller must fall back to a path that does not depend on it — this
/// never returns `Some` for input whose runs are not its groups.
fn key_boundaries(
    keys: &[ArrayRef],
    num_rows: usize,
) -> Result<Option<BooleanBuffer>, RuntimeError> {
    if keys.is_empty() || num_rows < 2 || !keys.iter().all(comparable) {
        return Ok(None);
    }

    // Compare on the same canonical form the hash paths group on, so a float key cannot make
    // the two disagree: arrow orders `-0.0` below `0.0` and NaN above everything, while SQL
    // groups both zeros together and all NaNs together. Canonicalizing first is what makes the
    // runs found here the groups `assign_groups` would have built.
    let canon = crate::keys::canonicalize_float_keys(keys);
    let keys: &[ArrayRef] = canon.as_deref().unwrap_or(keys);

    let pairs = num_rows - 1;
    let mut bounds = BooleanBufferBuilder::new(pairs);
    // Direction is discovered, not requested: the first chunk holding a strict comparison fixes
    // it and every later chunk must agree. Descending input clusters equal keys exactly as well
    // as ascending, and `ORDER BY x DESC` is as common as its opposite.
    let mut descending: Option<bool> = None;

    let mut off = 0;
    let mut step = FIRST_CHUNK;
    while off < pairs {
        let len = step.min(pairs - off);
        let (less, equal) = compare_adjacent(keys, off, len)?;

        // A pair is out of order if it is neither equal nor ordered the way the rest are. With
        // the direction still open, a chunk that is ascending-clean settles it as ascending;
        // one that is descending-clean settles it as descending; one that is both (every pair
        // equal) leaves it open.
        let strictly_less = and(&less, &not(&equal)?)?;
        let ascending_ok = all(&or(&less, &equal)?);
        let descending_ok = none(&strictly_less);
        match descending {
            Some(true) if !descending_ok => return Ok(None),
            Some(false) if !ascending_ok => return Ok(None),
            None => {
                if ascending_ok {
                    // Leave the direction open while nothing has compared strictly, so a long
                    // opening run of equal keys does not lock in a direction it never saw.
                    if !descending_ok {
                        descending = Some(false);
                    }
                } else if descending_ok {
                    descending = Some(true);
                } else {
                    return Ok(None);
                }
            }
            _ => {}
        }

        bounds.append_buffer(&not(&equal)?.values().clone());
        off += len;
        step = (step * 2).min(MAX_CHUNK);
    }

    Ok(Some(bounds.finish()))
}

/// [`super::assign_groups`] for a verifiably sorted key: group ids from runs, no hash table.
///
/// `None` when the key is not verifiably sorted, leaving the caller on its hash paths.
pub(crate) fn assign_groups_runs(
    group_keys: &[ArrayRef],
    num_rows: usize,
) -> Result<Option<Assignment>, RuntimeError> {
    let Some(bounds) = key_boundaries(group_keys, num_rows)? else {
        return Ok(None);
    };

    // Every row of run `g` is group `g`, and runs are produced in row order, so group ids come
    // out in the first-seen order every other path in `assign` promises. Walking the set bits
    // (which arrow does a word at a time) fills each run's ids with one `fill` rather than a
    // per-row write.
    let mut group_ids = vec![0u32; num_rows];
    let mut reps: Vec<u32> = Vec::new();
    let mut start = 0usize;
    for boundary in bounds.set_indices() {
        reps.push(start as u32);
        group_ids[start..=boundary].fill(reps.len() as u32 - 1);
        start = boundary + 1;
    }
    reps.push(start as u32);
    group_ids[start..num_rows].fill(reps.len() as u32 - 1);

    let num_groups = reps.len();
    let group_columns = super::assign::group_columns(group_keys, reps, num_rows)?;
    Ok(Some((group_ids, num_groups, group_columns)))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int64Array, StringArray};

    use super::*;

    fn i64s(v: &[i64]) -> ArrayRef {
        Arc::new(Int64Array::from(v.to_vec())) as ArrayRef
    }

    fn strs(v: &[&str]) -> ArrayRef {
        Arc::new(StringArray::from(v.to_vec())) as ArrayRef
    }

    /// The whole point: on sorted input the run assignment is the hash assignment.
    #[test]
    fn matches_the_hash_oracle_on_sorted_input() {
        let keys = vec![i64s(&[1, 1, 2, 2, 2, 5, 9, 9])];
        let (ids, n, cols) = assign_groups_runs(&keys, 8).unwrap().unwrap();
        let (want_ids, want_n, want_cols) = super::super::assign_groups(&keys, 8).unwrap();
        assert_eq!(ids, want_ids);
        assert_eq!(n, want_n);
        assert_eq!(cols[0].as_ref(), want_cols[0].as_ref());
        assert_eq!(ids, vec![0, 0, 1, 1, 1, 2, 3, 3]);
    }

    #[test]
    fn descending_input_clusters_just_as_well() {
        let keys = vec![i64s(&[9, 9, 5, 2, 2, 1])];
        let (ids, n, _) = assign_groups_runs(&keys, 6).unwrap().unwrap();
        assert_eq!(ids, vec![0, 0, 1, 2, 2, 3]);
        assert_eq!(n, 4);
    }

    /// The safety property. Unsorted input must decline, not answer.
    #[test]
    fn declines_unsorted_input() {
        let keys = vec![i64s(&[1, 2, 1, 2])];
        assert!(assign_groups_runs(&keys, 4).unwrap().is_none());
    }

    /// The failure this module exists to prevent: a key that recurs after an intervening
    /// group. Runs would say three groups; the truth is two.
    #[test]
    fn declines_a_key_that_recurs_in_a_later_run() {
        let keys = vec![i64s(&[1, 1, 2, 2, 1, 1])];
        assert!(assign_groups_runs(&keys, 6).unwrap().is_none());
    }

    /// A violation past the first chunk must still be caught: the chunking is an early exit,
    /// not a limit on what gets checked.
    #[test]
    fn a_violation_past_the_first_chunk_is_still_caught() {
        let mut v: Vec<i64> = (0..FIRST_CHUNK as i64 + 500).collect();
        v.push(3);
        v.push(9);
        let n = v.len();
        assert!(assign_groups_runs(&[i64s(&v)], n).unwrap().is_none());
    }

    #[test]
    fn one_direction_only() {
        // Up then down: clusters fine, but is not monotonic, and 1 recurs.
        let keys = vec![i64s(&[1, 2, 3, 2, 1])];
        assert!(assign_groups_runs(&keys, 5).unwrap().is_none());
    }

    #[test]
    fn a_single_run_is_one_group() {
        let keys = vec![i64s(&[4, 4, 4, 4])];
        let (ids, n, cols) = assign_groups_runs(&keys, 4).unwrap().unwrap();
        assert_eq!(ids, vec![0, 0, 0, 0]);
        assert_eq!(n, 1);
        assert_eq!(cols[0].len(), 1);
    }

    #[test]
    fn every_row_its_own_group() {
        let keys = vec![i64s(&[1, 2, 3, 4])];
        let (ids, n, cols) = assign_groups_runs(&keys, 4).unwrap().unwrap();
        assert_eq!(ids, vec![0, 1, 2, 3]);
        assert_eq!(n, 4);
        assert_eq!(cols[0].as_ref(), i64s(&[1, 2, 3, 4]).as_ref());
    }

    #[test]
    fn string_keys() {
        let keys = vec![strs(&["a", "a", "b", "c", "c"])];
        let (ids, n, cols) = assign_groups_runs(&keys, 5).unwrap().unwrap();
        assert_eq!(ids, vec![0, 0, 1, 2, 2]);
        assert_eq!(n, 3);
        let (want_ids, _, want_cols) = super::super::assign_groups(&keys, 5).unwrap();
        assert_eq!(ids, want_ids);
        assert_eq!(cols[0].as_ref(), want_cols[0].as_ref());
    }

    /// A composite key is ordered lexicographically, so the minor key may descend within a
    /// major run without the whole key ceasing to be sorted.
    #[test]
    fn composite_keys_compare_lexicographically() {
        let keys = vec![i64s(&[1, 1, 1, 2, 2]), i64s(&[1, 1, 2, 1, 3])];
        let (ids, n, _) = assign_groups_runs(&keys, 5).unwrap().unwrap();
        assert_eq!(ids, vec![0, 0, 1, 2, 3]);
        assert_eq!(n, 4);
        let (want_ids, want_n, _) = super::super::assign_groups(&keys, 5).unwrap();
        assert_eq!((ids, n), (want_ids, want_n));
    }

    #[test]
    fn declines_a_composite_key_out_of_lexicographic_order() {
        // Sorted on the second column, not on the pair.
        let keys = vec![i64s(&[2, 1, 2, 1]), i64s(&[1, 2, 3, 4])];
        assert!(assign_groups_runs(&keys, 4).unwrap().is_none());
    }

    /// A key containing nulls declines, deliberately. A null compares as null under `lt`,
    /// which reads as unordered, and resolving that would mean deciding nulls-first against
    /// nulls-last on top of the ascending/descending question the scan already answers. The
    /// cost of declining is a missed optimization on one shape; the cost of guessing wrong
    /// would be a wrong group count. The hash paths below serve it exactly as before.
    #[test]
    fn a_key_with_nulls_declines() {
        let keys: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![
            None,
            None,
            Some(1),
            Some(1),
        ]))];
        assert!(assign_groups_runs(&keys, 4).unwrap().is_none());
        // And the hash path still groups the two nulls together, which is what `GROUP BY` means.
        let (_, want_n, _) = super::super::assign_groups(&keys, 4).unwrap();
        assert_eq!(want_n, 2);
    }

    /// `-0.0` and `0.0` are one group under SQL, and arrow orders them apart. Canonicalizing
    /// before comparing is what keeps this path's answer equal to the hash path's.
    #[test]
    fn negative_zero_groups_with_zero() {
        let keys: Vec<ArrayRef> = vec![Arc::new(Float64Array::from(vec![-0.0, 0.0, 1.0]))];
        let (ids, n, _) = assign_groups_runs(&keys, 3).unwrap().unwrap();
        let (want_ids, want_n, _) = super::super::assign_groups(&keys, 3).unwrap();
        assert_eq!((ids.clone(), n), (want_ids, want_n));
        assert_eq!(ids, vec![0, 0, 1]);
    }

    /// Every NaN is one group under SQL. Canonicalization also makes them compare equal, so a
    /// run of NaNs stays a run.
    #[test]
    fn nans_are_one_group() {
        let keys: Vec<ArrayRef> = vec![Arc::new(Float64Array::from(vec![1.0, f64::NAN, f64::NAN]))];
        let (ids, n, _) = assign_groups_runs(&keys, 3).unwrap().unwrap();
        let (want_ids, want_n, _) = super::super::assign_groups(&keys, 3).unwrap();
        assert_eq!((ids, n), (want_ids, want_n));
        assert_eq!(want_n, 2);
    }

    #[test]
    fn declines_types_it_cannot_cheaply_order() {
        // A dictionary key: the codes carry no order this module can establish.
        let values = strs(&["a", "b"]);
        let dict: ArrayRef = Arc::new(
            arrow::array::DictionaryArray::<arrow::datatypes::Int32Type>::try_new(
                arrow::array::Int32Array::from(vec![0, 0, 1]),
                values,
            )
            .unwrap(),
        );
        assert!(assign_groups_runs(&[dict], 3).unwrap().is_none());
    }

    #[test]
    fn too_short_to_bother() {
        assert!(assign_groups_runs(&[i64s(&[1])], 1).unwrap().is_none());
        assert!(assign_groups_runs(&[], 0).unwrap().is_none());
    }
}
