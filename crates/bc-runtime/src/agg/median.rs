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

/// Median per group: sort the value list and take the middle (averaging the two
/// middle values for an even count). Always yields Float64; empty groups → null.
pub(crate) fn finalize_median(state: &ArrayRef) -> Result<ArrayRef, RuntimeError> {
    let list = state.as_list::<i32>();
    let mut out = Float64Builder::with_capacity(list.len());
    for row in 0..list.len() {
        let vals = list.value(row);
        let mut v: Vec<f64> = match vals.data_type() {
            DataType::Int64 => {
                let a = vals.as_primitive::<Int64Type>();
                (0..a.len())
                    .filter(|&i| a.is_valid(i))
                    .map(|i| a.value(i) as f64)
                    .collect()
            }
            DataType::Float64 => {
                let a = vals.as_primitive::<Float64Type>();
                (0..a.len())
                    .filter(|&i| a.is_valid(i))
                    .map(|i| a.value(i))
                    .collect()
            }
            other => {
                return Err(RuntimeError::UnsupportedAggregate {
                    func: "median".to_string(),
                    dtype: other.to_string(),
                })
            }
        };
        if v.is_empty() {
            out.append_null();
            continue;
        }
        // `total_cmp` is a total order over f64, so a NaN value sorts deterministically
        // instead of panicking the way `partial_cmp(..).unwrap()` would on a NaN input.
        v.sort_by(f64::total_cmp);
        let n = v.len();
        let m = if n % 2 == 1 {
            v[n / 2]
        } else {
            (v[n / 2 - 1] + v[n / 2]) / 2.0
        };
        out.append_value(m);
    }
    Ok(Arc::new(out.finish()))
}

/// Continuous quantile per group at `q` in [0,1] (`percentile_cont`): sort the
/// non-null values and linearly interpolate at position `q·(n-1)`. Always yields
/// Float64; empty groups → null. (Median is the q=0.5 special case.)
pub(crate) fn finalize_quantile(state: &ArrayRef, q: f64) -> Result<ArrayRef, RuntimeError> {
    let list = state.as_list::<i32>();
    let mut out = Float64Builder::with_capacity(list.len());
    for row in 0..list.len() {
        let vals = list.value(row);
        let mut v: Vec<f64> = match vals.data_type() {
            DataType::Int64 => {
                let a = vals.as_primitive::<Int64Type>();
                (0..a.len())
                    .filter(|&i| a.is_valid(i))
                    .map(|i| a.value(i) as f64)
                    .collect()
            }
            DataType::Float64 => {
                let a = vals.as_primitive::<Float64Type>();
                (0..a.len())
                    .filter(|&i| a.is_valid(i))
                    .map(|i| a.value(i))
                    .collect()
            }
            other => {
                return Err(RuntimeError::UnsupportedAggregate {
                    func: "quantile".to_string(),
                    dtype: other.to_string(),
                })
            }
        };
        if v.is_empty() {
            out.append_null();
            continue;
        }
        // `total_cmp` is a total order over f64, so a NaN value sorts deterministically
        // instead of panicking the way `partial_cmp(..).unwrap()` would on a NaN input.
        v.sort_by(f64::total_cmp);
        let pos = q.clamp(0.0, 1.0) * (v.len() - 1) as f64;
        let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
        out.append_value(v[lo] + (v[hi] - v[lo]) * (pos - lo as f64));
    }
    Ok(Arc::new(out.finish()))
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
