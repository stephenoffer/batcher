//! Whole-partition window aggregates (`SUM`/`AVG`/`MIN`/`MAX`/`COUNT` with no ORDER BY
//! and no frame): one value per partition, broadcast to every row of that partition.
//!
//! Computed via **dense group ids** — reduce each group in one linear pass over the
//! rows, then broadcast its value back by index. This is exactly a group-by aggregate
//! followed by a scatter, and it avoids the per-partition index lists and the scattered
//! gather they force. The order within a partition never affects a whole-partition
//! aggregate, so the result is identical to ordering first.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Float64Type, Int64Type};

use crate::error::RuntimeError;
use crate::window::{require, WindowFn};

/// Whole-partition aggregate from per-partition row-index lists (the slow window path,
/// reached only when a no-ORDER-BY aggregate is mixed with non-aggregate functions).
/// Flattens the lists to dense group ids, then defers to [`broadcast_partition_aggregate`].
pub(crate) fn partition_aggregate(
    func: WindowFn,
    partitions: &[Vec<usize>],
    values: Option<&ArrayRef>,
    num_rows: usize,
) -> Result<ArrayRef, RuntimeError> {
    let mut group_ids = vec![0u32; num_rows];
    for (g, part) in partitions.iter().enumerate() {
        for &i in part {
            group_ids[i] = g as u32;
        }
    }
    broadcast_partition_aggregate(func, &group_ids, partitions.len(), values)
}

/// Whole-partition aggregate via dense group ids: reduce each group in one linear pass
/// over the rows, then broadcast its value to every member row.
pub(crate) fn broadcast_partition_aggregate(
    func: WindowFn,
    group_ids: &[u32],
    num_groups: usize,
    values: Option<&ArrayRef>,
) -> Result<ArrayRef, RuntimeError> {
    if func == WindowFn::Count {
        let values = require(values, func)?;
        let mut counts = vec![0i64; num_groups];
        for (i, &g) in group_ids.iter().enumerate() {
            if values.is_valid(i) {
                counts[g as usize] += 1;
            }
        }
        let out: Vec<i64> = group_ids.iter().map(|&g| counts[g as usize]).collect();
        return Ok(Arc::new(Int64Array::from(out)));
    }
    let values = require(values, func)?;
    match values.data_type() {
        DataType::Int64 => grouped_i64(func, group_ids, num_groups, values),
        DataType::Float64 => grouped_f64(func, group_ids, num_groups, values),
        DataType::Utf8 if matches!(func, WindowFn::Min | WindowFn::Max) => {
            grouped_str_minmax(func, group_ids, num_groups, values)
        }
        other => Err(RuntimeError::UnsupportedWindow {
            func: func.name().to_string(),
            dtype: other.to_string(),
        }),
    }
}

fn grouped_i64(
    func: WindowFn,
    group_ids: &[u32],
    num_groups: usize,
    values: &ArrayRef,
) -> Result<ArrayRef, RuntimeError> {
    let arr = values.as_primitive::<Int64Type>();
    if func == WindowFn::Avg {
        let (mut sum, mut cnt) = (vec![0f64; num_groups], vec![0i64; num_groups]);
        for (i, &g) in group_ids.iter().enumerate() {
            if arr.is_valid(i) {
                sum[g as usize] += arr.value(i) as f64;
                cnt[g as usize] += 1;
            }
        }
        let grp: Vec<Option<f64>> = (0..num_groups)
            .map(|g| (cnt[g] > 0).then(|| sum[g] / cnt[g] as f64))
            .collect();
        let out: Vec<Option<f64>> = group_ids.iter().map(|&g| grp[g as usize]).collect();
        return Ok(Arc::new(Float64Array::from(out)));
    }
    let mut acc: Vec<Option<i64>> = vec![None; num_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        if arr.is_valid(i) {
            let v = arr.value(i);
            let slot = &mut acc[g as usize];
            *slot = Some(match (func, *slot) {
                (_, None) => v,
                (WindowFn::Sum, Some(a)) => a + v,
                (WindowFn::Min, Some(a)) => a.min(v),
                (WindowFn::Max, Some(a)) => a.max(v),
                (_, Some(a)) => a,
            });
        }
    }
    let out: Vec<Option<i64>> = group_ids.iter().map(|&g| acc[g as usize]).collect();
    Ok(Arc::new(Int64Array::from(out)))
}

fn grouped_f64(
    func: WindowFn,
    group_ids: &[u32],
    num_groups: usize,
    values: &ArrayRef,
) -> Result<ArrayRef, RuntimeError> {
    let arr = values.as_primitive::<Float64Type>();
    let (mut acc, mut cnt): (Vec<Option<f64>>, Vec<i64>) =
        (vec![None; num_groups], vec![0i64; num_groups]);
    for (i, &g) in group_ids.iter().enumerate() {
        if arr.is_valid(i) {
            let v = arr.value(i);
            let g = g as usize;
            cnt[g] += 1;
            acc[g] = Some(match (func, acc[g]) {
                (_, None) => v,
                (WindowFn::Sum | WindowFn::Avg, Some(a)) => a + v,
                (WindowFn::Min, Some(a)) => a.min(v),
                (WindowFn::Max, Some(a)) => a.max(v),
                (_, Some(a)) => a,
            });
        }
    }
    let grp: Vec<Option<f64>> = (0..num_groups)
        .map(|g| match func {
            WindowFn::Avg => acc[g].map(|s| s / cnt[g] as f64),
            _ => acc[g],
        })
        .collect();
    let out: Vec<Option<f64>> = group_ids.iter().map(|&g| grp[g as usize]).collect();
    Ok(Arc::new(Float64Array::from(out)))
}

fn grouped_str_minmax(
    func: WindowFn,
    group_ids: &[u32],
    num_groups: usize,
    values: &ArrayRef,
) -> Result<ArrayRef, RuntimeError> {
    let arr = values.as_any().downcast_ref::<StringArray>().expect("utf8");
    let is_min = func == WindowFn::Min;
    let mut acc: Vec<Option<&str>> = vec![None; num_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        if arr.is_valid(i) {
            let v = arr.value(i);
            let slot = &mut acc[g as usize];
            *slot = Some(match *slot {
                None => v,
                Some(a) if (is_min && v < a) || (!is_min && v > a) => v,
                Some(a) => a,
            });
        }
    }
    let grp: Vec<Option<String>> = acc.iter().map(|o| o.map(|s| s.to_string())).collect();
    let out: Vec<Option<String>> = group_ids.iter().map(|&g| grp[g as usize].clone()).collect();
    Ok(Arc::new(StringArray::from(out)))
}
