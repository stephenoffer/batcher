//! MEDIAN / continuous-quantile — exact, mergeable via a per-group value list
//! (no dedup, unlike COUNT(DISTINCT)).

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Float64Builder, Int64Array, UInt32Array};
use arrow::compute::take;
use arrow::datatypes::{DataType, Float64Type, Int64Type};
use arrow::row::{RowConverter, SortField};

use super::bucket_values_into_list;
use crate::error::RuntimeError;

/// Partial state for MEDIAN: each group's non-null values as one `List` column.
pub(crate) fn median_state(
    values: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
) -> Result<ArrayRef, RuntimeError> {
    // Bounded by the row count (the kept subset never exceeds it) — pre-size to skip
    // the geometric reallocations these two parallel Vecs would otherwise churn through.
    let mut keep: Vec<u32> = Vec::with_capacity(group_ids.len());
    let mut kept_groups: Vec<i64> = Vec::with_capacity(group_ids.len());
    for (i, &g) in group_ids.iter().enumerate() {
        if values.is_valid(i) {
            keep.push(i as u32);
            kept_groups.push(g as i64);
        }
    }
    let kept_values = take(values.as_ref(), &UInt32Array::from(keep), None)?;
    bucket_values_into_list(&Int64Array::from(kept_groups), &kept_values, num_groups)
}

/// Partial state for `array_agg`/`list_agg`: each group's values as one `List` column,
/// **keeping NULL elements** and their arrival order within the partial.
///
/// Unlike [`median_state`], which filters nulls (correct for a median, which ignores
/// them), SQL `array_agg(x)` collects every value including NULLs — `array_agg` over
/// `[3, NULL, 1]` is `[3, NULL, 1]`, not `[3, 1]`. Reusing `median_state` here dropped
/// them. The merge (`merge_median`) and finalize (the list itself) already preserve
/// nulls, so collecting them at `partial` is the whole fix.
pub(crate) fn listagg_state(
    values: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
) -> Result<ArrayRef, RuntimeError> {
    let groups: Vec<i64> = group_ids.iter().map(|&g| g as i64).collect();
    bucket_values_into_list(&Int64Array::from(groups), values, num_groups)
}

/// Finalize `array_agg`/`list_agg`: a **non-null empty** per-group list means that group
/// saw zero input rows, and DuckDB yields NULL there — not `[]`.
///
/// This is only reachable for a global aggregate over empty input (`array_agg(x)` /
/// `string_agg(x, sep)` over zero rows): a real `GROUP BY` group exists because ≥1 row
/// mapped to it, so its list always has ≥1 element (nulls are kept, so even an all-null
/// group is `[null, …]`, not `[]`). Without this, `array_agg` over an empty relation
/// returned `[]` and `string_agg` returned `""`, both of which DuckDB reports as NULL.
///
/// Fast path: if no row is a non-null empty list (the overwhelmingly common case) the
/// state is returned unchanged (a zero-copy `Arc` bump).
pub(crate) fn finalize_list_agg(state: &ArrayRef) -> Result<ArrayRef, RuntimeError> {
    use arrow::array::ListArray;
    use arrow::buffer::NullBuffer;

    let list = state.as_list::<i32>();
    let offsets = list.value_offsets();
    // A zero-length row is either already null or an empty-input group; both must be NULL.
    let has_empty = (0..list.len()).any(|r| offsets[r + 1] == offsets[r] && !list.is_null(r));
    if !has_empty {
        return Ok(state.clone());
    }
    // Validity = "the list has at least one element". Empty rows (existing-null or
    // empty-input) flip to null; offsets/values are untouched (an empty row spans no child).
    let valid: NullBuffer = (0..list.len())
        .map(|r| offsets[r + 1] > offsets[r])
        .collect();
    let field = match list.data_type() {
        DataType::List(f) => f.clone(),
        other => {
            return Err(RuntimeError::UnsupportedAggregate {
                func: "list_agg".to_string(),
                dtype: other.to_string(),
            })
        }
    };
    let rebuilt = ListArray::try_new(
        field,
        list.offsets().clone(),
        list.values().clone(),
        Some(valid),
    )?;
    Ok(Arc::new(rebuilt))
}

/// Merge per-group value lists across partitions (flatten to `(group, value)`,
/// re-bucket — no dedup, unlike COUNT(DISTINCT)).
pub(crate) fn merge_median(
    state: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
) -> Result<ArrayRef, RuntimeError> {
    let list = state.as_list::<i32>();
    let offsets = list.value_offsets();
    let child = list.values();
    // The concatenated child holds every partial list's values in list-row order, and the list
    // offsets are contiguous (0, len0, len0+len1, …), so flattening in row order visits child
    // element `e` exactly at output position `e`. The old code therefore built an identity index
    // `0..child.len()` and `take`d the child through it — a full copy of every value (~60M on a
    // low-cardinality MEDIAN/QUANTILE) for no reordering. Skip it: expand only the per-element
    // group id and hand the child straight to the bucketer, which does the one real gather.
    let contiguous = offsets.first().copied().unwrap_or(0) == 0
        && offsets.last().map(|&o| o as usize) == Some(child.len());
    let mut elem_groups: Vec<i64> = Vec::with_capacity(child.len());
    for row in 0..list.len() {
        let n = (offsets[row + 1] - offsets[row]) as usize;
        let g = group_ids[row] as i64;
        elem_groups.extend(std::iter::repeat_n(g, n));
    }
    if contiguous {
        return bucket_values_into_list(&Int64Array::from(elem_groups), child, num_groups);
    }
    // Defensive fallback (a sliced/offset child): gather through the explicit element indices.
    let elem_idx: Vec<u32> = (0..list.len())
        .flat_map(|row| offsets[row] as u32..offsets[row + 1] as u32)
        .collect();
    let values = take(child.as_ref(), &UInt32Array::from(elem_idx), None)?;
    bucket_values_into_list(&Int64Array::from(elem_groups), &values, num_groups)
}

/// Median per group: the middle value (averaging the two middle for an even count).
/// Always yields Float64; empty groups → null. (Median is the `q=0.5` quantile.)
pub(crate) fn finalize_median(state: &ArrayRef) -> Result<ArrayRef, RuntimeError> {
    finalize_select(state, "median", quickselect_median)
}

/// Continuous quantile per group at `q` in [0,1] (`percentile_cont`): linearly
/// interpolate at position `q·(n-1)`. Always yields Float64; empty groups → null.
pub(crate) fn finalize_quantile(state: &ArrayRef, q: f64) -> Result<ArrayRef, RuntimeError> {
    finalize_select(state, "quantile", move |v| quickselect_quantile(v, q))
}

/// Shared finalize for median/quantile: each group's value list is independent, so the
/// per-group selection runs across cores. The selection itself is **quickselect**
/// (`select_nth_unstable_by`, O(n) average) instead of a full O(n log n) sort — median
/// and quantile need only the value(s) at a fixed rank, not the whole order. Identical
/// result to sorting then indexing (quickselect places the k-th smallest at index k with
/// all lesser elements before it). Always Float64; an empty group → null.
fn finalize_select(
    state: &ArrayRef,
    func: &str,
    pick: impl Fn(&mut [f64]) -> f64 + Sync,
) -> Result<ArrayRef, RuntimeError> {
    use rayon::prelude::*;

    let list = state.as_list::<i32>();
    let results: Vec<Option<f64>> = (0..list.len())
        .into_par_iter()
        .map(|row| -> Result<Option<f64>, RuntimeError> {
            let mut v = group_values_f64(&list.value(row), func)?;
            Ok((!v.is_empty()).then(|| pick(&mut v)))
        })
        .collect::<Result<_, _>>()?;

    let mut out = Float64Builder::with_capacity(list.len());
    for r in results {
        match r {
            Some(x) => out.append_value(x),
            None => out.append_null(),
        }
    }
    Ok(Arc::new(out.finish()))
}

/// One group's non-null values as `f64` (Int64 widened). The list state is Int64 or
/// Float64 element lists; any other element type is an unsupported aggregate.
fn group_values_f64(vals: &ArrayRef, func: &str) -> Result<Vec<f64>, RuntimeError> {
    match vals.data_type() {
        DataType::Int64 => {
            let a = vals.as_primitive::<Int64Type>();
            Ok((0..a.len())
                .filter(|&i| a.is_valid(i))
                .map(|i| a.value(i) as f64)
                .collect())
        }
        DataType::Float64 => {
            let a = vals.as_primitive::<Float64Type>();
            Ok((0..a.len())
                .filter(|&i| a.is_valid(i))
                .map(|i| a.value(i))
                .collect())
        }
        other => Err(RuntimeError::UnsupportedAggregate {
            func: func.to_string(),
            dtype: other.to_string(),
        }),
    }
}

/// Median of `v` via quickselect — partition so the n/2-th smallest sits at index n/2
/// (`total_cmp`, so NaN orders deterministically). For an even count the lower-middle is
/// the max of the now-lesser partition, exactly the sorted `v[n/2-1]`.
fn quickselect_median(v: &mut [f64]) -> f64 {
    use crate::keys::float_total_cmp;
    let n = v.len();
    let (lo, mid, _) = v.select_nth_unstable_by(n / 2, |a, b| float_total_cmp(*a, *b));
    if n % 2 == 1 {
        *mid
    } else {
        let lower = lo.iter().copied().fold(f64::NEG_INFINITY, |a, b| {
            if float_total_cmp(a, b).is_lt() {
                b
            } else {
                a
            }
        });
        (lower + *mid) / 2.0
    }
}

/// Continuous quantile of `v` at `q` via quickselect on the bracketing ranks: select the
/// `floor(q·(n-1))`-th smallest; the next rank (when `q` falls between two) is the min of
/// the resulting greater partition. Matches the sort-then-interpolate result.
fn quickselect_quantile(v: &mut [f64], q: f64) -> f64 {
    use crate::keys::float_total_cmp;
    let n = v.len();
    let pos = q.clamp(0.0, 1.0) * (n - 1) as f64;
    let lo_i = pos.floor() as usize;
    let frac = pos - lo_i as f64;
    let (_, lo_ref, greater) = v.select_nth_unstable_by(lo_i, |a, b| float_total_cmp(*a, *b));
    let lo_val = *lo_ref;
    let hi_val = if frac == 0.0 || greater.is_empty() {
        lo_val
    } else {
        greater.iter().copied().fold(f64::INFINITY, |a, b| {
            if float_total_cmp(b, a).is_lt() {
                b
            } else {
                a
            }
        })
    };
    lo_val + (hi_val - lo_val) * frac
}

/// Mode per group: the most frequent value in each group's list (same list state
/// as MEDIAN, so it is type-general). Ties are broken by the **smallest** value, so
/// the result is deterministic and partition-independent regardless of merge order.
/// The output preserves the input element type; empty groups → null.
pub(crate) fn finalize_mode(state: &ArrayRef) -> Result<ArrayRef, RuntimeError> {
    let list = state.as_list::<i32>();
    // Canonicalize float leaves BEFORE grouping so `-0.0`/`0.0` (and every NaN bit pattern)
    // count as one value — the same distinct identity `GROUP BY`, `count(distinct)`, and
    // DuckDB use. Arrow's row format is NOT canonical for floats, so without this
    // `mode([-0.0, -0.0, 0.0])` returned `-0.0` (a spurious 2-vs-1 frequency split) instead
    // of `0.0`, and multi-NaN groups fractured. `take`ing the winner from the canonical
    // column also returns the canonical representative (`0.0`, one quiet NaN) DuckDB does.
    let child_ref = list.values();
    let canon = crate::keys::canonicalize_float_keys(std::slice::from_ref(child_ref));
    let child: &ArrayRef = canon.as_ref().map_or(child_ref, |c| &c[0]);
    // Encode every value once into arrow's order-preserving row format, so values
    // of any type can be compared/grouped (and ties broken by the smallest value).
    let converter = RowConverter::new(vec![SortField::new(child.data_type().clone())])?;
    let rows = converter.convert_columns(std::slice::from_ref(child))?;
    let offsets = list.value_offsets();

    let mut winners: Vec<Option<u32>> = Vec::with_capacity(list.len());
    for row in 0..list.len() {
        let (start, end) = (offsets[row] as usize, offsets[row + 1] as usize);
        if start == end {
            winners.push(None);
            continue;
        }
        // Sort the group's element indices by value, then the longest run of equal
        // values is the mode; scanning with a strict `>` keeps the *first* (smallest)
        // run on a frequency tie.
        let mut idxs: Vec<u32> = (start as u32..end as u32).collect();
        idxs.sort_by(|&a, &b| rows.row(a as usize).cmp(&rows.row(b as usize)));
        let (mut best_idx, mut best_len) = (idxs[0], 1usize);
        let (mut run_start, mut run_len) = (0usize, 1usize);
        for j in 1..idxs.len() {
            if rows.row(idxs[j] as usize) == rows.row(idxs[j - 1] as usize) {
                run_len += 1;
            } else {
                if run_len > best_len {
                    best_len = run_len;
                    best_idx = idxs[run_start];
                }
                run_start = j;
                run_len = 1;
            }
        }
        if run_len > best_len {
            best_idx = idxs[run_start];
        }
        winners.push(Some(best_idx));
    }
    Ok(take(child.as_ref(), &UInt32Array::from(winners), None)?)
}

/// `histogram` finalize: turn each group's value list into a `Map<value, count>`
/// (DuckDB `histogram`). Keys are the distinct values **sorted ascending** (via the
/// order-preserving row format, so any value type works); values are their counts.
pub(crate) fn finalize_histogram(state: &ArrayRef) -> Result<ArrayRef, RuntimeError> {
    use arrow::array::{MapArray, StructArray};
    use arrow::buffer::OffsetBuffer;
    use arrow::datatypes::{Field, Fields};

    let list = state.as_list::<i32>();
    // Canonicalize float leaves BEFORE grouping so `-0.0`/`0.0` (and every NaN bit pattern)
    // form ONE histogram key with the summed count — matching `GROUP BY`, `count(distinct)`,
    // and DuckDB. Arrow's row format is not canonical for floats, so without this
    // `histogram([0.0, -0.0])` produced two keys of count 1 instead of `{0.0: 2}`. The map
    // keys are `take`n from the canonical column, so they read back as the canonical value.
    let child_ref = list.values();
    let canon = crate::keys::canonicalize_float_keys(std::slice::from_ref(child_ref));
    let child: &ArrayRef = canon.as_ref().map_or(child_ref, |c| &c[0]);
    let converter = RowConverter::new(vec![SortField::new(child.data_type().clone())])?;
    let rows = converter.convert_columns(std::slice::from_ref(child))?;
    let offsets = list.value_offsets();

    // Distinct-run entries are bounded above by the total element count — pre-size to it.
    let mut key_idx: Vec<u32> = Vec::with_capacity(child.len());
    let mut counts: Vec<i64> = Vec::with_capacity(child.len());
    let mut map_offsets: Vec<i32> = Vec::with_capacity(list.len() + 1);
    // A group with no values (all-null) yields a NULL map, not an empty one (DuckDB).
    let mut valid: Vec<bool> = Vec::with_capacity(list.len());
    map_offsets.push(0);
    for row in 0..list.len() {
        let (start, end) = (offsets[row] as usize, offsets[row + 1] as usize);
        valid.push(start < end);
        if start < end {
            // Sort the group's element indices by value; equal values form a run,
            // and each run is one (key, count) entry of the histogram map.
            let mut idxs: Vec<u32> = (start as u32..end as u32).collect();
            idxs.sort_by(|&a, &b| rows.row(a as usize).cmp(&rows.row(b as usize)));
            let mut run_start = 0usize;
            for j in 1..=idxs.len() {
                let breaks = j == idxs.len()
                    || rows.row(idxs[j] as usize) != rows.row(idxs[run_start] as usize);
                if breaks {
                    key_idx.push(idxs[run_start]);
                    counts.push((j - run_start) as i64);
                    run_start = j;
                }
            }
        }
        map_offsets.push(key_idx.len() as i32);
    }

    let keys = take(child.as_ref(), &UInt32Array::from(key_idx), None)?;
    let vals: ArrayRef = Arc::new(Int64Array::from(counts));
    let key_field = Arc::new(Field::new("key", keys.data_type().clone(), false));
    let val_field = Arc::new(Field::new("value", DataType::Int64, true));
    let struct_fields = Fields::from(vec![key_field, val_field]);
    let entries = StructArray::new(struct_fields.clone(), vec![keys, vals], None);
    let entries_field = Arc::new(Field::new(
        "entries",
        DataType::Struct(struct_fields),
        false,
    ));
    let map = MapArray::try_new(
        entries_field,
        OffsetBuffer::new(map_offsets.into()),
        entries,
        Some(arrow::buffer::NullBuffer::from(valid)),
        false,
    )?;
    Ok(Arc::new(map))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;

    #[test]
    fn mode_picks_most_frequent_tiebreak_smallest() {
        use arrow::array::Int64Array;
        // group 0: [5,5,7,5] → 5 (freq 3); group 1: [3,9,9,3] → tie(3,9) → 3 (smallest).
        let values: ArrayRef = Arc::new(Int64Array::from(vec![5, 5, 7, 5, 3, 9, 9, 3]));
        let group_ids = [0u32, 0, 0, 0, 1, 1, 1, 1];
        let state = median_state(&values, &group_ids, 2).unwrap();
        let modes = finalize_mode(&state).unwrap();
        let m = modes.as_primitive::<Int64Type>();
        assert_eq!(m.value(0), 5);
        assert_eq!(m.value(1), 3); // tie broken to the smaller value → deterministic
    }

    #[test]
    fn mode_empty_group_is_null() {
        use arrow::array::Int64Array;
        let values: ArrayRef = Arc::new(Int64Array::from(vec![Some(5), None]));
        let state = median_state(&values, &[0u32, 1], 2).unwrap();
        let modes = finalize_mode(&state).unwrap();
        assert!(modes.is_valid(0) && !modes.is_valid(1));
    }

    /// `mode` over a Float64 group must fold `-0.0`/`0.0` (and every NaN) into one value —
    /// the same distinct identity `GROUP BY`/`count(distinct)`/DuckDB use. Before the fix the
    /// non-canonical row encoding split `[-0.0, -0.0, 0.0]` into a 2-vs-1 frequency race and
    /// returned `-0.0`; the canonical mode is `0.0` (frequency 3).
    #[test]
    fn mode_folds_signed_zero_and_nan() {
        let values: ArrayRef = Arc::new(Float64Array::from(vec![-0.0, -0.0, 0.0]));
        let state = median_state(&values, &[0u32, 0, 0], 1).unwrap();
        let modes = finalize_mode(&state).unwrap();
        let m = modes.as_primitive::<Float64Type>();
        // Canonical `+0.0` (bits 0), NOT `-0.0` (bits 0x8000…): the two zeros are one value.
        assert_eq!(
            m.value(0).to_bits(),
            0.0f64.to_bits(),
            "mode must fold -0.0 into +0.0"
        );

        // Two differing NaN bit patterns are one value → mode is that (canonical) NaN, not a tie.
        let nans: ArrayRef = Arc::new(Float64Array::from(vec![
            f64::NAN,
            f64::from_bits(0x7ff8_0000_0000_0001),
            1.0,
        ]));
        let st = median_state(&nans, &[0u32, 0, 0], 1).unwrap();
        let out = finalize_mode(&st).unwrap();
        assert!(
            out.as_primitive::<Float64Type>().value(0).is_nan(),
            "the two NaNs collapse to frequency 2 and win over the single 1.0"
        );
    }

    /// `histogram` over a Float64 group must fold `-0.0`/`0.0` (and every NaN) into ONE key
    /// with the summed count. Before the fix `[0.0, -0.0]` produced two keys of count 1
    /// instead of `{0.0: 2}`.
    #[test]
    fn histogram_folds_signed_zero_and_nan() {
        use arrow::array::MapArray;
        let values: ArrayRef = Arc::new(Float64Array::from(vec![0.0, -0.0]));
        let state = median_state(&values, &[0u32, 0], 1).unwrap();
        let out = finalize_histogram(&state).unwrap();
        let m = out.as_any().downcast_ref::<MapArray>().unwrap();
        assert_eq!(m.value_length(0), 1, "-0.0 and 0.0 are ONE histogram key");
        let counts = m.value(0);
        let counts = counts.column(1).as_primitive::<Int64Type>();
        assert_eq!(counts.value(0), 2, "the folded zero key has count 2");

        // Two NaN bit patterns → one key, count 2.
        let nans: ArrayRef = Arc::new(Float64Array::from(vec![
            f64::NAN,
            f64::from_bits(0x7ff8_0000_0000_0001),
        ]));
        let st = median_state(&nans, &[0u32, 0], 1).unwrap();
        let o = finalize_histogram(&st).unwrap();
        let mm = o.as_any().downcast_ref::<MapArray>().unwrap();
        assert_eq!(mm.value_length(0), 1, "all NaN is ONE histogram key");
        let c = mm.value(0);
        assert_eq!(c.column(1).as_primitive::<Int64Type>().value(0), 2);
    }

    #[test]
    fn histogram_counts_and_null_group() {
        use arrow::array::{Int64Array, MapArray};
        // group 0: [1,1,2] → {1:2, 2:1}; group 1: [None] → null map.
        let values: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), Some(1), Some(2), None]));
        let state = median_state(&values, &[0u32, 0, 0, 1], 2).unwrap();
        let out = finalize_histogram(&state).unwrap();
        let m = out.as_any().downcast_ref::<MapArray>().unwrap();
        assert!(m.is_valid(0));
        assert_eq!(m.value_length(0), 2); // two distinct keys
        let counts = m.value(0);
        let counts = counts.column(1).as_primitive::<Int64Type>();
        assert_eq!(counts.value(0), 2); // key 1 → count 2 (sorted ascending)
        assert_eq!(counts.value(1), 1); // key 2 → count 1
        assert!(m.is_null(1)); // all-null group → NULL map
    }

    #[test]
    fn listagg_keeps_nulls_and_order_and_merges() {
        use arrow::array::ListArray;
        // group 0: [3, NULL, 1, 3]  → array_agg keeps the NULL and arrival order.
        // group 1: [NULL]           → a single NULL element (not an empty list).
        let values: ArrayRef = Arc::new(Int64Array::from(vec![
            Some(3),
            None,
            Some(1),
            Some(3),
            None,
        ]));
        let group_ids = [0u32, 0, 0, 0, 1];
        let state = listagg_state(&values, &group_ids, 2).unwrap();
        let list = state.as_any().downcast_ref::<ListArray>().unwrap();

        let row0 = list.value(0);
        let g0 = row0.as_primitive::<Int64Type>();
        let got0: Vec<Option<i64>> = (0..g0.len())
            .map(|i| g0.is_valid(i).then(|| g0.value(i)))
            .collect();
        assert_eq!(got0, vec![Some(3), None, Some(1), Some(3)]);

        let row1 = list.value(1);
        let g1 = row1.as_primitive::<Int64Type>();
        assert_eq!(g1.len(), 1);
        assert!(g1.is_null(0)); // a lone NULL is preserved, not dropped to an empty list

        // median_state, by contrast, filters the nulls — the two states must differ.
        let med = median_state(&values, &group_ids, 2).unwrap();
        let med_list = med.as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(med_list.value(0).len(), 3); // [3,1,3] — the NULL is gone
        assert_eq!(med_list.value(1).len(), 0); // all-null group → empty list
    }

    #[test]
    fn finalize_list_agg_empty_group_becomes_null() {
        use arrow::array::ListArray;
        // group 0: one row (value 7) → [7]; group 1: ZERO rows → empty list.
        // The empty list is only reachable when a group saw no input rows (here, a group
        // that never received a value, mirroring a global aggregate over empty input).
        let values: ArrayRef = Arc::new(Int64Array::from(vec![Some(7)]));
        let group_ids = [0u32];
        let state = listagg_state(&values, &group_ids, 2).unwrap();
        let raw = state.as_any().downcast_ref::<ListArray>().unwrap();
        // Before finalize: the empty group is a non-null empty list.
        assert!(raw.is_valid(1));
        assert_eq!(raw.value_length(1), 0);

        let out = finalize_list_agg(&state).unwrap();
        let list = out.as_any().downcast_ref::<ListArray>().unwrap();
        assert!(list.is_valid(0));
        assert_eq!(list.value_length(0), 1); // [7] kept
        assert!(list.is_null(1)); // empty group → NULL (DuckDB), not []

        // A non-empty group (even all-null) is never touched: [null] stays [null], not null.
        let vals2: ArrayRef = Arc::new(Int64Array::from(vec![None::<i64>]));
        let st2 = listagg_state(&vals2, &[0u32], 1).unwrap();
        let out2 = finalize_list_agg(&st2).unwrap();
        let l2 = out2.as_any().downcast_ref::<ListArray>().unwrap();
        assert!(l2.is_valid(0)); // [null] is a non-null one-element list
        assert_eq!(l2.value_length(0), 1);
    }

    #[test]
    fn quickselect_matches_sorted_oracle() {
        // quickselect median/quantile must equal sorting then indexing, for odd/even
        // counts, negatives, and duplicates — across many random vectors and quantiles.
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut rng = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for trial in 0..400 {
            let n = 1 + (rng() as usize % 257);
            let mut v: Vec<f64> = (0..n).map(|_| (rng() % 1000) as f64 - 500.0).collect();
            let mut sorted = v.clone();
            sorted.sort_by(f64::total_cmp);
            // median oracle
            let om = if n % 2 == 1 {
                sorted[n / 2]
            } else {
                (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
            };
            assert_eq!(
                super::quickselect_median(&mut v.clone()),
                om,
                "median trial {trial} n={n}"
            );
            // quantile oracle at a few q
            for &q in &[0.0, 0.1, 0.25, 0.5, 0.9, 1.0] {
                let pos = q * (n - 1) as f64;
                let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
                let oq = sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64);
                assert_eq!(
                    super::quickselect_quantile(&mut v, q),
                    oq,
                    "quantile q={q} trial {trial} n={n}"
                );
            }
        }
    }

    #[test]
    fn median_and_quantile_with_nan_do_not_panic() {
        // A NaN in the value list previously panicked via partial_cmp(..).unwrap().
        let values: ArrayRef = Arc::new(Float64Array::from(vec![1.0, f64::NAN, 3.0, 2.0]));
        let group_ids = [0u32, 0, 0, 0];
        let state = median_state(&values, &group_ids, 1).unwrap();
        let med = finalize_median(&state).unwrap();
        assert_eq!(med.len(), 1);
        let q = finalize_quantile(&state, 0.9).unwrap();
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn median_ranks_negative_nan_greatest_like_group_by() {
        // A *negative* NaN (what `0.0/0.0` yields on x86) must rank as the engine's
        // greatest float, not below -inf. With the old `f64::total_cmp` it sat at the
        // bottom and shifted the selected rank: median of [1, 2, 3, -NaN] wrongly
        // became 2.0 (mid of a list that put -NaN first). The engine's total order puts
        // every NaN last, so the sorted order is [1, 2, 3, NaN] and the lower-middle of
        // four is (2+3)/2 = 2.5 — matching DuckDB and `MAX`.
        let neg_nan = f64::from_bits(0xfff8_0000_0000_0000);
        assert!(neg_nan.is_nan() && neg_nan.is_sign_negative());
        let mut v = vec![1.0, 2.0, 3.0, neg_nan];
        assert_eq!(super::quickselect_median(&mut v), 2.5);
        // The 3rd-of-4 quantile (q=2/3) brackets ranks 2 and 3 → value 3.0, never the NaN.
        let mut v2 = vec![1.0, 2.0, 3.0, neg_nan];
        assert_eq!(super::quickselect_quantile(&mut v2, 2.0 / 3.0), 3.0);
    }
}
