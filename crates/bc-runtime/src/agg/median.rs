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
    let mut keep: Vec<u32> = Vec::new();
    let mut kept_groups: Vec<i64> = Vec::new();
    for (i, &g) in group_ids.iter().enumerate() {
        if values.is_valid(i) {
            keep.push(i as u32);
            kept_groups.push(g as i64);
        }
    }
    let kept_values = take(values.as_ref(), &UInt32Array::from(keep), None)?;
    bucket_values_into_list(&Int64Array::from(kept_groups), &kept_values, num_groups)
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
    let mut elem_idx: Vec<u32> = Vec::new();
    let mut elem_groups: Vec<i64> = Vec::new();
    for row in 0..list.len() {
        let (start, end) = (offsets[row] as usize, offsets[row + 1] as usize);
        let g = group_ids[row] as i64;
        for e in start..end {
            elem_idx.push(e as u32);
            elem_groups.push(g);
        }
    }
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
    let n = v.len();
    let (lo, mid, _) = v.select_nth_unstable_by(n / 2, f64::total_cmp);
    if n % 2 == 1 {
        *mid
    } else {
        let lower = lo.iter().copied().fold(f64::NEG_INFINITY, |a, b| {
            if a.total_cmp(&b).is_lt() {
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
    let n = v.len();
    let pos = q.clamp(0.0, 1.0) * (n - 1) as f64;
    let lo_i = pos.floor() as usize;
    let frac = pos - lo_i as f64;
    let (_, lo_ref, greater) = v.select_nth_unstable_by(lo_i, f64::total_cmp);
    let lo_val = *lo_ref;
    let hi_val = if frac == 0.0 || greater.is_empty() {
        lo_val
    } else {
        greater.iter().copied().fold(
            f64::INFINITY,
            |a, b| {
                if b.total_cmp(&a).is_lt() {
                    b
                } else {
                    a
                }
            },
        )
    };
    lo_val + (hi_val - lo_val) * frac
}

/// Mode per group: the most frequent value in each group's list (same list state
/// as MEDIAN, so it is type-general). Ties are broken by the **smallest** value, so
/// the result is deterministic and partition-independent regardless of merge order.
/// The output preserves the input element type; empty groups → null.
pub(crate) fn finalize_mode(state: &ArrayRef) -> Result<ArrayRef, RuntimeError> {
    let list = state.as_list::<i32>();
    let child = list.values();
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
    let child = list.values();
    let converter = RowConverter::new(vec![SortField::new(child.data_type().clone())])?;
    let rows = converter.convert_columns(std::slice::from_ref(child))?;
    let offsets = list.value_offsets();

    let mut key_idx: Vec<u32> = Vec::new();
    let mut counts: Vec<i64> = Vec::new();
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
}
