//! Value-frequency state for the aggregates that only ever ask "how often?".
//!
//! `mode` and `top_k` (DuckDB spells it `approx_top_k`) both reduce a group to its most
//! frequent values, and both used to carry **every value of the group** as their partial state
//! — the list [`median_state`](super::median::median_state) builds for the quantile family —
//! then count it at finalize. That is the right state for a quantile, which needs the values
//! themselves, and the wrong one for a counter: the answer is bounded by the group's
//! *cardinality*, and the state was bounded by its *row count*.
//!
//! The gap that opens between those two is not subtle. Ten million rows in one group over fifty
//! distinct values, asking for the top three:
//!
//! | | time | peak RSS |
//! |---|---|---|
//! | `top_k(3)`, list state | 532.8 ms | +854 MB |
//! | `mode()`, list state | 4,230.9 ms | +6 MB |
//! | DuckDB `approx_top_k(v, 3)` | 23.3 ms | +0 MB |
//! | `GROUP BY v` + `count()` — the same counting, spelled as a group-by | 10.8 ms | +0 MB |
//!
//! The last row is the one that indicts the old state: Batcher's own hash group-by does exactly
//! this counting, over the same ten million rows, in 10.8 ms. The aggregate was 22.9x slower
//! than DuckDB and allocated 854 MB to return three values, because it kept all ten million to
//! do it.
//!
//! ## What the state is instead
//!
//! Two parallel `List` columns per group: the **distinct** values, and their counts. The state
//! is `O(distinct)` rather than `O(rows)`, which is the bound the answer always had. It stays
//! **exact** — nothing here is sketched or sampled, despite the `approx_` the SQL name carries
//! for DuckDB compatibility — and it stays mergeable, because counts add: `combine` re-groups
//! the partials' `(value, count)` pairs and sums. That keeps
//! `combine_finalize(partition(partial(pₖ)))` equal to the single-node reduction, so the
//! parallel, spilling and distributed paths inherit the speedup without a second implementation.
//!
//! ## Three invariants the old path kept and this one must keep
//!
//! * **Nulls do not count.** Excluded when the state is built, as `median_state` excluded them.
//! * **Float identity folds before counting.** `-0.0`/`0.0` and every NaN bit pattern are one
//!   value, and the representative handed back is the canonical one. Counting *raw* bytes would
//!   split `mode([-0.0, -0.0, 0.0])` into a spurious 2-versus-1 and answer `-0.0`, where DuckDB
//!   and every other Batcher path answer `0.0`. Arrow's row format is not canonical for floats,
//!   so this cannot be left to the encoder.
//! * **Ties break to the smaller value.** This is what makes the winner a function of the
//!   group's contents rather than of the order its partitions happened to arrive in — the
//!   property without which none of this would be a legal mergeable aggregate.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Int64Array, UInt32Array};
use arrow::datatypes::Int64Type;
use arrow::row::{Row, RowConverter, Rows, SortField};

use super::distinct::bucket_values_into_list;
use crate::error::RuntimeError;

/// Distinct values and counts per group, as the pair of parallel `List` columns the state is.
///
/// `elem_groups` is the owning group of each surviving element, so both lists are bucketed by
/// the same deterministic permutation and stay index-paired.
fn emit(
    values: &ArrayRef,
    keep: Vec<u32>,
    counts: Vec<i64>,
    elem_groups: Vec<i64>,
    num_groups: usize,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    let groups = Int64Array::from(elem_groups);
    let taken = arrow::compute::take(values.as_ref(), &UInt32Array::from(keep), None)?;
    let counts: ArrayRef = Arc::new(Int64Array::from(counts));
    Ok(vec![
        bucket_values_into_list(&groups, &taken, num_groups)?,
        bucket_values_into_list(&groups, &counts, num_groups)?,
    ])
}

/// Count `(group, encoded value)` pairs, returning the first-seen element index and the tally.
///
/// `weight` supplies each element's contribution, which is `1` when counting raw values at
/// `partial` and the carried count when re-counting partial states at `combine`.
fn tally<'a>(
    rows: &'a Rows,
    len: usize,
    num_groups: usize,
    group_of: impl Fn(usize) -> usize,
    skip: impl Fn(usize) -> bool,
    weight: impl Fn(usize) -> i64,
) -> (Vec<u32>, Vec<i64>, Vec<i64>) {
    // Keyed by the encoded `Row` itself rather than by its bytes: `Row::as_ref` borrows the
    // temporary the accessor returns, while the `Row` borrows `rows`, which outlives the loop.
    let mut maps: Vec<ahash::AHashMap<Row<'a>, usize>> =
        (0..num_groups).map(|_| ahash::AHashMap::new()).collect();
    let (mut keep, mut counts, mut elem_groups) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..len {
        if skip(i) {
            continue;
        }
        let g = group_of(i);
        let key = rows.row(i);
        match maps[g].get(&key) {
            Some(&slot) => counts[slot] += weight(i),
            None => {
                maps[g].insert(key, keep.len());
                keep.push(i as u32);
                counts.push(weight(i));
                elem_groups.push(g as i64);
            }
        }
    }
    (keep, counts, elem_groups)
}

/// Partial state: each group's distinct (canonicalized, non-null) values and their counts.
pub(crate) fn counted_state(
    values: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    // Fold float identity BEFORE counting, and keep the folded column as the one the
    // representative is taken from — see the module docs for what raw bytes do to `mode`.
    let canon = crate::keys::canonicalize_float_keys(std::slice::from_ref(values));
    let vals: &ArrayRef = canon.as_ref().map_or(values, |c| &c[0]);
    let converter = RowConverter::new(vec![SortField::new(vals.data_type().clone())])?;
    let rows = converter.convert_columns(std::slice::from_ref(vals))?;

    let nulls = values.nulls();
    let (keep, counts, elem_groups) = tally(
        &rows,
        group_ids.len(),
        num_groups,
        |i| group_ids[i] as usize,
        |i| nulls.is_some_and(|n| n.is_null(i)),
        |_| 1,
    );
    emit(vals, keep, counts, elem_groups, num_groups)
}

/// Merge partial counted states: re-group the `(value, count)` pairs and sum the counts.
pub(crate) fn merge_counted(
    state: &[ArrayRef],
    group_ids: &[u32],
    num_groups: usize,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    let vlist = state[0].as_list::<i32>();
    let cchild = state[1]
        .as_list::<i32>()
        .values()
        .as_primitive::<Int64Type>();
    let vchild = vlist.values();
    // Values already passed through `canonicalize_float_keys` at `partial`, so the encoded
    // bytes are canonical here and re-folding would be redundant work on every merge.
    let converter = RowConverter::new(vec![SortField::new(vchild.data_type().clone())])?;
    let rows = converter.convert_columns(std::slice::from_ref(vchild))?;

    // Element -> the output group of the state row that carries it.
    let offsets = vlist.value_offsets();
    let mut owner: Vec<u32> = vec![0; vchild.len()];
    for row in 0..vlist.len() {
        let span = &mut owner[offsets[row] as usize..offsets[row + 1] as usize];
        span.fill(group_ids[row]);
    }

    let (keep, counts, elem_groups) = tally(
        &rows,
        vchild.len(),
        num_groups,
        |e| owner[e] as usize,
        |_| false,
        |e| cchild.value(e),
    );
    emit(vchild, keep, counts, elem_groups, num_groups)
}

/// Per group, its element indices ordered by descending count then ascending value.
fn ranked(state: &[ArrayRef]) -> Result<(Vec<Vec<u32>>, ArrayRef), RuntimeError> {
    let vlist = state[0].as_list::<i32>();
    let cchild = state[1]
        .as_list::<i32>()
        .values()
        .as_primitive::<Int64Type>();
    let vchild = vlist.values().clone();
    let converter = RowConverter::new(vec![SortField::new(vchild.data_type().clone())])?;
    let rows = converter.convert_columns(std::slice::from_ref(&vchild))?;
    let offsets = vlist.value_offsets();

    let mut out: Vec<Vec<u32>> = Vec::with_capacity(vlist.len());
    for row in 0..vlist.len() {
        let (lo, hi) = (offsets[row] as u32, offsets[row + 1] as u32);
        let mut idx: Vec<u32> = (lo..hi).collect();
        idx.sort_unstable_by(|&a, &b| {
            cchild
                .value(b as usize)
                .cmp(&cchild.value(a as usize))
                .then_with(|| rows.row(a as usize).cmp(&rows.row(b as usize)))
        });
        out.push(idx);
    }
    Ok((out, vchild))
}

/// `top_k`: each group's `k` most frequent values, most frequent first, as a `List`.
pub(crate) fn finalize_top_k(state: &[ArrayRef], k: usize) -> Result<ArrayRef, RuntimeError> {
    let (ranked_idx, vchild) = ranked(state)?;
    let mut keep: Vec<u32> = Vec::new();
    let mut elem_groups: Vec<i64> = Vec::new();
    for (g, idx) in ranked_idx.iter().enumerate() {
        for &e in idx.iter().take(k) {
            keep.push(e);
            elem_groups.push(g as i64);
        }
    }
    let num_groups = ranked_idx.len();
    let groups = Int64Array::from(elem_groups);
    let taken = arrow::compute::take(vchild.as_ref(), &UInt32Array::from(keep), None)?;
    bucket_values_into_list(&groups, &taken, num_groups)
}

/// `mode`: each group's most frequent value, ties to the smallest; an empty group is NULL.
pub(crate) fn finalize_mode(state: &[ArrayRef]) -> Result<ArrayRef, RuntimeError> {
    let (ranked_idx, vchild) = ranked(state)?;
    let winners: Vec<Option<u32>> = ranked_idx.iter().map(|i| i.first().copied()).collect();
    Ok(arrow::compute::take(
        vchild.as_ref(),
        &UInt32Array::from(winners),
        None,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array};
    use arrow::datatypes::{DataType, Float64Type, Int64Type};

    fn state(values: Vec<Option<i64>>, groups: &[u32], n: usize) -> Vec<ArrayRef> {
        let v: ArrayRef = Arc::new(Int64Array::from(values));
        counted_state(&v, groups, n).unwrap()
    }

    #[test]
    fn mode_picks_most_frequent_tiebreak_smallest() {
        // group 0: [5,5,7,5] -> 5 (freq 3); group 1: [3,9,9,3] -> tie(3,9) -> 3 (smallest).
        let values: ArrayRef = Arc::new(Int64Array::from(vec![5, 5, 7, 5, 3, 9, 9, 3]));
        let st = counted_state(&values, &[0u32, 0, 0, 0, 1, 1, 1, 1], 2).unwrap();
        let modes = finalize_mode(&st).unwrap();
        let m = modes.as_primitive::<Int64Type>();
        assert_eq!(m.value(0), 5);
        assert_eq!(m.value(1), 3); // tie broken to the smaller value -> deterministic
    }

    #[test]
    fn mode_empty_group_is_null() {
        let st = state(vec![Some(5), None], &[0u32, 1], 2);
        let modes = finalize_mode(&st).unwrap();
        assert!(modes.is_valid(0) && !modes.is_valid(1));
    }

    /// `mode` over a Float64 group must fold `-0.0`/`0.0` (and every NaN) into one value --
    /// the same distinct identity `GROUP BY`/`count(distinct)`/DuckDB use. With a
    /// non-canonical row encoding `[-0.0, -0.0, 0.0]` splits into a 2-vs-1 frequency race and
    /// returns `-0.0`; the canonical mode is `0.0` (frequency 3). Counting made this sharper
    /// than it was under the list state: the fold now decides a *count*, not just a winner.
    #[test]
    fn mode_folds_signed_zero_and_nan() {
        let values: ArrayRef = Arc::new(Float64Array::from(vec![-0.0, -0.0, 0.0]));
        let st = counted_state(&values, &[0u32, 0, 0], 1).unwrap();
        let modes = finalize_mode(&st).unwrap();
        // Canonical `+0.0` (bits 0), NOT `-0.0` (bits 0x8000...): the two zeros are one value.
        assert_eq!(
            modes.as_primitive::<Float64Type>().value(0).to_bits(),
            0.0f64.to_bits(),
            "mode must fold -0.0 into +0.0"
        );

        // Two differing NaN bit patterns are one value -> mode is that NaN, not a tie.
        let nans: ArrayRef = Arc::new(Float64Array::from(vec![
            f64::NAN,
            f64::from_bits(0x7ff8_0000_0000_0001),
            1.0,
        ]));
        let st = counted_state(&nans, &[0u32, 0, 0], 1).unwrap();
        let out = finalize_mode(&st).unwrap();
        assert!(
            out.as_primitive::<Float64Type>().value(0).is_nan(),
            "the two NaNs collapse to frequency 2 and win over the single 1.0"
        );
    }

    /// The state is the point: it must hold one entry per DISTINCT value, not per row.
    #[test]
    fn state_is_bounded_by_cardinality_not_row_count() {
        let values: ArrayRef = Arc::new(Int64Array::from(
            (0..1_000).map(|i| i % 4).collect::<Vec<i64>>(),
        ));
        let st = counted_state(&values, &vec![0u32; 1_000], 1).unwrap();
        assert_eq!(
            st[0].as_list::<i32>().value(0).len(),
            4,
            "1,000 rows over 4 distinct values must not keep 1,000 elements"
        );
        // ...and the counts must still account for every row.
        let counts = st[1].as_list::<i32>().value(0);
        let total: i64 = counts.as_primitive::<Int64Type>().values().iter().sum();
        assert_eq!(total, 1_000);
    }

    #[test]
    fn top_k_is_most_frequent_first_and_truncates() {
        // 7 appears 4x, 5 3x, 9 2x, 1 1x.
        let mut v: Vec<i64> = Vec::new();
        v.extend([7, 7, 7, 7, 5, 5, 5, 9, 9, 1]);
        let values: ArrayRef = Arc::new(Int64Array::from(v.clone()));
        let st = counted_state(&values, &vec![0u32; v.len()], 1).unwrap();
        let out = finalize_top_k(&st, 3).unwrap();
        let got = out.as_list::<i32>().value(0);
        assert_eq!(
            got.as_primitive::<Int64Type>().values(),
            &[7, 5, 9],
            "most frequent first, truncated to k"
        );
        // k beyond the cardinality yields every distinct value, not padding.
        let all = finalize_top_k(&st, 99).unwrap();
        assert_eq!(all.as_list::<i32>().value(0).len(), 4);
    }

    /// The mergeable invariant, which is what lets this run on many cores and many machines:
    /// splitting the rows into partitions, taking a partial of each, combining and finalizing
    /// must equal the single-node answer.
    #[test]
    fn combine_finalize_of_partitions_equals_single_node() {
        let all: Vec<i64> = (0..300).map(|i| (i * 7) % 11).collect();
        let groups: Vec<u32> = (0..300).map(|i| (i % 3) as u32).collect();

        let single = finalize_top_k(
            &counted_state(
                &(Arc::new(Int64Array::from(all.clone())) as ArrayRef),
                &groups,
                3,
            )
            .unwrap(),
            4,
        )
        .unwrap();

        // Three partitions of unequal size, each producing a partial.
        let mut vals: Vec<ArrayRef> = Vec::new();
        let mut cnts: Vec<ArrayRef> = Vec::new();
        let mut owner: Vec<u32> = Vec::new();
        for (lo, hi) in [(0usize, 37usize), (37, 200), (200, 300)] {
            let part: ArrayRef = Arc::new(Int64Array::from(all[lo..hi].to_vec()));
            let st = counted_state(&part, &groups[lo..hi], 3).unwrap();
            vals.push(st[0].clone());
            cnts.push(st[1].clone());
            owner.extend([0u32, 1, 2]); // each partial carries all three groups, in order
        }
        let cat = |a: &[ArrayRef]| {
            arrow::compute::concat(&a.iter().map(|x| x.as_ref()).collect::<Vec<_>>()).unwrap()
        };
        let merged = merge_counted(&[cat(&vals), cat(&cnts)], &owner, 3).unwrap();
        let distributed = finalize_top_k(&merged, 4).unwrap();

        assert_eq!(
            format!("{single:?}"),
            format!("{distributed:?}"),
            "combine_finalize(partition(partial)) must equal the single-node reduction"
        );
    }

    /// Held against a deliberately naive counter over the raw values -- the reduction the old
    /// list state performed at finalize. Randomized over shapes with heavy duplication, nulls
    /// and several groups, which is where a counted state can go wrong and a hand-picked case
    /// will not notice.
    #[test]
    fn agrees_with_a_naive_count_over_random_shapes() {
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for trial in 0..200 {
            let n = 1 + (next() % 400) as usize;
            let cardinality = 1 + (next() % 12) as i64;
            let num_groups = 1 + (next() % 4) as usize;
            let vals: Vec<Option<i64>> = (0..n)
                .map(|_| {
                    let r = next();
                    // ~1 in 8 null, so empty groups occur too.
                    if r % 8 == 0 {
                        None
                    } else {
                        Some((r / 8) as i64 % cardinality)
                    }
                })
                .collect();
            let groups: Vec<u32> = (0..n)
                .map(|_| (next() % num_groups as u64) as u32)
                .collect();

            let st = state(vals.clone(), &groups, num_groups);
            let modes = finalize_mode(&st).unwrap();

            for g in 0..num_groups {
                let mut tally: std::collections::BTreeMap<i64, i64> = Default::default();
                for (v, &gg) in vals.iter().zip(&groups) {
                    if gg as usize == g {
                        if let Some(v) = v {
                            *tally.entry(*v).or_insert(0) += 1;
                        }
                    }
                }
                // BTreeMap iterates ascending, so `max_by_key` on the count alone would take
                // the LAST maximum; fold explicitly to keep the smallest value on a tie.
                let want = tally
                    .iter()
                    .fold(None::<(i64, i64)>, |best, (&v, &c)| match best {
                        Some((_, bc)) if bc >= c => best,
                        _ => Some((v, c)),
                    });
                match want {
                    None => assert!(!modes.is_valid(g), "trial {trial} group {g}: expected null"),
                    Some((v, _)) => assert_eq!(
                        modes.as_primitive::<Int64Type>().value(g),
                        v,
                        "trial {trial} group {g}"
                    ),
                }
            }
        }
    }

    /// The counted state row-encodes at `partial`, which the list state it replaced only did
    /// at `finalize`. So the degenerate column types reach the encoder a stage earlier, and
    /// the ones that used to be someone else's problem are now this module's: a `Null`-typed
    /// column (what an empty or untyped source yields) and a zero-row input must not raise.
    ///
    /// A `NullArray` carries no null *buffer* -- `nulls()` is `None` -- so its elements are
    /// counted rather than skipped. That is exactly what `median_state` did with the same
    /// `match values.nulls()`, and the output type is `Null` either way, so the observable
    /// answer is unchanged. Asserted as "does not error" rather than as a null result,
    /// because the latter would be pinning an expectation the engine never had.
    #[test]
    fn null_typed_and_empty_inputs_do_not_error() {
        use arrow::array::NullArray;
        let v: ArrayRef = Arc::new(NullArray::new(3));
        let st = counted_state(&v, &[0u32, 0, 1], 2).unwrap();
        let modes = finalize_mode(&st).unwrap();
        assert_eq!(modes.len(), 2);
        assert_eq!(modes.data_type(), &DataType::Null);
        // Zero rows, zero groups.
        let v: ArrayRef = Arc::new(Int64Array::from(Vec::<i64>::new()));
        let st = counted_state(&v, &[], 0).unwrap();
        assert_eq!(finalize_mode(&st).unwrap().len(), 0);
        assert_eq!(finalize_top_k(&st, 3).unwrap().len(), 0);
    }
}
