//! Variance / standard-deviation / mean finalizers and their shared
//! (sum, sum_of_squares, count) partial-state producer.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Float64Array, Float64Builder, Int64Array};
use arrow::datatypes::{DataType, Float64Type, Int64Type};

use super::AggFunc;
use crate::error::RuntimeError;

/// One-pass (sum, sum_of_squares, count) per group, read as f64 (so integer
/// inputs don't overflow when squared).
pub(crate) fn var_state(
    values: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
    func: AggFunc,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    let mut sum = vec![0f64; num_groups];
    let mut sumsq = vec![0f64; num_groups];
    let mut count = vec![0i64; num_groups];

    let mut update = |g: usize, v: f64| {
        sum[g] += v;
        sumsq[g] += v * v;
        count[g] += 1;
    };
    match values.data_type() {
        DataType::Int64 => {
            let a = values.as_primitive::<Int64Type>();
            for (i, &g) in group_ids.iter().enumerate() {
                if a.is_valid(i) {
                    update(g as usize, a.value(i) as f64);
                }
            }
        }
        DataType::Float64 => {
            let a = values.as_primitive::<Float64Type>();
            for (i, &g) in group_ids.iter().enumerate() {
                if a.is_valid(i) {
                    update(g as usize, a.value(i));
                }
            }
        }
        other => {
            return Err(RuntimeError::UnsupportedAggregate {
                func: func.name().to_string(),
                dtype: other.to_string(),
            })
        }
    }
    Ok(vec![
        Arc::new(Float64Array::from(sum)),
        Arc::new(Float64Array::from(sumsq)),
        Arc::new(Int64Array::from(count)),
    ])
}

pub(crate) fn count_non_null(values: &ArrayRef, group_ids: &[u32], num_groups: usize) -> ArrayRef {
    let mut counts = vec![0i64; num_groups];
    if values.null_count() == 0 {
        // No-null fast path: every row counts, so skip the per-row validity bitmap
        // check entirely (the dominant COUNT(col)/AVG path, e.g. TPC-H Q1).
        for &g in group_ids {
            counts[g as usize] += 1;
        }
    } else {
        for (i, &g) in group_ids.iter().enumerate() {
            if values.is_valid(i) {
                counts[g as usize] += 1;
            }
        }
    }
    Arc::new(Int64Array::from(counts))
}

/// Finalize sample variance (or its sqrt for stddev) from (sum, sumsq, count).
/// `var = (Σx² − (Σx)²/n) / (n − 1)`; null when `n < 2`.
pub(crate) fn finalize_var(
    sum: &ArrayRef,
    sumsq: &ArrayRef,
    count: &ArrayRef,
    stddev: bool,
) -> Result<ArrayRef, RuntimeError> {
    let sum = sum.as_primitive::<Float64Type>();
    let sumsq = sumsq.as_primitive::<Float64Type>();
    let count = count.as_primitive::<Int64Type>();
    let mut b = Float64Builder::with_capacity(count.len());
    for i in 0..count.len() {
        let n = count.value(i);
        if n < 2 {
            b.append_null();
            continue;
        }
        let s = sum.value(i);
        let ss = sumsq.value(i);
        let var = (ss - s * s / n as f64) / (n - 1) as f64;
        let var = var.max(0.0); // guard tiny negatives from float cancellation
        b.append_value(if stddev { var.sqrt() } else { var });
    }
    Ok(Arc::new(b.finish()))
}

/// Finalize `mean = sum / count`, always producing Float64.
pub(crate) fn finalize_mean(sum: &ArrayRef, count: &ArrayRef) -> Result<ArrayRef, RuntimeError> {
    let counts = count.as_primitive::<Int64Type>();
    let mut b = Float64Builder::with_capacity(counts.len());
    match sum.data_type() {
        DataType::Int64 => {
            let sums = sum.as_primitive::<Int64Type>();
            for i in 0..counts.len() {
                push_mean(
                    &mut b,
                    sums.is_valid(i).then(|| sums.value(i) as f64),
                    counts.value(i),
                );
            }
        }
        DataType::Float64 => {
            let sums = sum.as_primitive::<Float64Type>();
            for i in 0..counts.len() {
                push_mean(
                    &mut b,
                    sums.is_valid(i).then(|| sums.value(i)),
                    counts.value(i),
                );
            }
        }
        other => {
            return Err(RuntimeError::UnsupportedAggregate {
                func: "mean".to_string(),
                dtype: other.to_string(),
            })
        }
    }
    Ok(Arc::new(b.finish()))
}

fn push_mean(b: &mut Float64Builder, sum: Option<f64>, count: i64) {
    match (sum, count) {
        (Some(s), c) if c > 0 => b.append_value(s / c as f64),
        _ => b.append_null(),
    }
}
