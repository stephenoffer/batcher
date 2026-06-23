//! COUNT(DISTINCT) — exact, mergeable via a per-group value list — plus the
//! `bucket_values_into_list` helper shared with the median path.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Int64Array, ListArray, UInt32Array};
use arrow::buffer::OffsetBuffer;
use arrow::compute::take;
use arrow::datatypes::{Field, Int64Type};

use super::assign_groups;
use crate::error::RuntimeError;

/// Partial state for COUNT(DISTINCT): each group's distinct non-null values as one
/// `List` column (row `g` = group `g`). Nulls are excluded (SQL semantics). The
/// dedup reuses `assign_groups` on `(group, value)` pairs — no bespoke set code.
pub(crate) fn distinct_state(
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
    let group_col: ArrayRef = Arc::new(Int64Array::from(kept_groups));
    distinct_pairs_to_list(group_col, kept_values, num_groups)
}

/// Merge per-group distinct lists across partitions: flatten to `(group, value)`
/// pairs, dedup, re-bucket. `combine` has already concatenated the list columns.
pub(crate) fn merge_distinct(
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
    let group_col: ArrayRef = Arc::new(Int64Array::from(elem_groups));
    distinct_pairs_to_list(group_col, values, num_groups)
}

/// Dedup `(group, value)` pairs and bucket the distinct values into a per-group
/// `List` column — the shared core of the distinct partial and merge steps.
fn distinct_pairs_to_list(
    groups: ArrayRef,
    values: ArrayRef,
    num_groups: usize,
) -> Result<ArrayRef, RuntimeError> {
    let n = values.len();
    let (_ids, _n_pairs, pair_cols) = assign_groups(&[groups, values], n)?;
    let distinct_groups = pair_cols[0].as_primitive::<Int64Type>();
    bucket_values_into_list(distinct_groups, &pair_cols[1], num_groups)
}

/// Bucket `values` into a `List` column by their `group_ids` (each in
/// `0..num_groups`), preserving stable order within each group.
pub(crate) fn bucket_values_into_list(
    group_ids: &Int64Array,
    values: &ArrayRef,
    num_groups: usize,
) -> Result<ArrayRef, RuntimeError> {
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); num_groups];
    for i in 0..group_ids.len() {
        buckets[group_ids.value(i) as usize].push(i as u32);
    }
    let mut order: Vec<u32> = Vec::with_capacity(values.len());
    let mut offsets: Vec<i32> = Vec::with_capacity(num_groups + 1);
    offsets.push(0);
    for bucket in &buckets {
        order.extend_from_slice(bucket);
        offsets.push(order.len() as i32);
    }
    let ordered = take(values.as_ref(), &UInt32Array::from(order), None)?;
    let field = Arc::new(Field::new("item", values.data_type().clone(), true));
    let list = ListArray::try_new(field, OffsetBuffer::new(offsets.into()), ordered, None)?;
    Ok(Arc::new(list))
}

/// Distinct count per group = the length of its distinct-value list.
pub(crate) fn finalize_count_distinct(state: &ArrayRef) -> ArrayRef {
    let list = state.as_list::<i32>();
    let offsets = list.value_offsets();
    let counts: Vec<i64> = (0..list.len())
        .map(|i| (offsets[i + 1] - offsets[i]) as i64)
        .collect();
    Arc::new(Int64Array::from(counts))
}
