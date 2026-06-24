//! Two-input covariance/correlation and single-input skewness/kurtosis.
//!
//! All use a *sum-of-powers* partial state so `combine` is plain column-wise
//! summation (associative + commutative) — the same trick `var` uses, which is
//! what makes these mergeable single-node and distributed.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Float64Array, Float64Builder, Int64Array};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Float64Type, Int64Type};

use super::AggFunc;
use crate::error::RuntimeError;

/// Per-group covariance/correlation state, 6 columns:
/// `[n, Σx, Σy, Σxy, Σx², Σy²]` (n is Int64, the rest Float64). A pair counts only
/// when both `x` and `y` are non-null. `covar_*` use the first four columns;
/// `corr` uses all six. All columns merge across partitions by summing.
pub(crate) fn covar_state(
    x: &ArrayRef,
    y: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    let xf = cast(x, &DataType::Float64)?;
    let yf = cast(y, &DataType::Float64)?;
    let xa = xf.as_primitive::<Float64Type>();
    let ya = yf.as_primitive::<Float64Type>();
    let (mut n, mut sx, mut sy) = (
        vec![0i64; num_groups],
        vec![0f64; num_groups],
        vec![0f64; num_groups],
    );
    let (mut sxy, mut sxx, mut syy) = (
        vec![0f64; num_groups],
        vec![0f64; num_groups],
        vec![0f64; num_groups],
    );
    for (i, &g) in group_ids.iter().enumerate() {
        if xa.is_valid(i) && ya.is_valid(i) {
            let (g, vx, vy) = (g as usize, xa.value(i), ya.value(i));
            n[g] += 1;
            sx[g] += vx;
            sy[g] += vy;
            sxy[g] += vx * vy;
            sxx[g] += vx * vx;
            syy[g] += vy * vy;
        }
    }
    Ok(vec![
        Arc::new(Int64Array::from(n)),
        Arc::new(Float64Array::from(sx)),
        Arc::new(Float64Array::from(sy)),
        Arc::new(Float64Array::from(sxy)),
        Arc::new(Float64Array::from(sxx)),
        Arc::new(Float64Array::from(syy)),
    ])
}

/// Per-group moment state for skewness/kurtosis, 5 columns:
/// `[n, Σx, Σx², Σx³, Σx⁴]` (n is Int64, the rest Float64). Null-skipping; merges by
/// summing each column.
pub(crate) fn moment_state(
    values: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
    func: AggFunc,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    let f = cast(values, &DataType::Float64).map_err(|_| RuntimeError::UnsupportedAggregate {
        func: func.name().to_string(),
        dtype: values.data_type().to_string(),
    })?;
    let a = f.as_primitive::<Float64Type>();
    let mut n = vec![0i64; num_groups];
    let (mut s1, mut s2) = (vec![0f64; num_groups], vec![0f64; num_groups]);
    let (mut s3, mut s4) = (vec![0f64; num_groups], vec![0f64; num_groups]);
    for (i, &g) in group_ids.iter().enumerate() {
        if a.is_valid(i) {
            let (g, v) = (g as usize, a.value(i));
            let (v2, v3, v4) = (v * v, v * v * v, v * v * v * v);
            n[g] += 1;
            s1[g] += v;
            s2[g] += v2;
            s3[g] += v3;
            s4[g] += v4;
        }
    }
    Ok(vec![
        Arc::new(Int64Array::from(n)),
        Arc::new(Float64Array::from(s1)),
        Arc::new(Float64Array::from(s2)),
        Arc::new(Float64Array::from(s3)),
        Arc::new(Float64Array::from(s4)),
    ])
}

/// `covar_pop = Σxy/n − (Σx/n)(Σy/n)` (n ≥ 1) or `covar_samp = (Σxy − Σx·Σy/n)/(n−1)`
/// (n ≥ 2). Null when the count is too small.
pub(crate) fn finalize_covar(state: &[ArrayRef], sample: bool) -> Result<ArrayRef, RuntimeError> {
    let n = state[0].as_primitive::<Int64Type>();
    let sx = state[1].as_primitive::<Float64Type>();
    let sy = state[2].as_primitive::<Float64Type>();
    let sxy = state[3].as_primitive::<Float64Type>();
    let mut b = Float64Builder::with_capacity(n.len());
    for i in 0..n.len() {
        let cnt = n.value(i);
        let cov = sxy.value(i) - sx.value(i) * sy.value(i) / cnt as f64;
        if sample {
            if cnt < 2 {
                b.append_null();
            } else {
                b.append_value(cov / (cnt - 1) as f64);
            }
        } else if cnt < 1 {
            b.append_null();
        } else {
            b.append_value(cov / cnt as f64);
        }
    }
    Ok(Arc::new(b.finish()))
}

/// `corr = (n·Σxy − Σx·Σy) / sqrt((n·Σx² − Σx²)(n·Σy² − Σy²))`. Null when n < 2 or
/// either variable has zero variance (a flat column has no correlation).
pub(crate) fn finalize_corr(state: &[ArrayRef]) -> Result<ArrayRef, RuntimeError> {
    let n = state[0].as_primitive::<Int64Type>();
    let sx = state[1].as_primitive::<Float64Type>();
    let sy = state[2].as_primitive::<Float64Type>();
    let sxy = state[3].as_primitive::<Float64Type>();
    let sxx = state[4].as_primitive::<Float64Type>();
    let syy = state[5].as_primitive::<Float64Type>();
    let mut b = Float64Builder::with_capacity(n.len());
    for i in 0..n.len() {
        let cnt = n.value(i) as f64;
        if n.value(i) < 2 {
            b.append_null();
            continue;
        }
        let cov = cnt * sxy.value(i) - sx.value(i) * sy.value(i);
        let vx = cnt * sxx.value(i) - sx.value(i) * sx.value(i);
        let vy = cnt * syy.value(i) - sy.value(i) * sy.value(i);
        let denom = (vx * vy).max(0.0).sqrt();
        if denom == 0.0 {
            b.append_null();
        } else {
            b.append_value(cov / denom);
        }
    }
    Ok(Arc::new(b.finish()))
}

/// Sample skewness (adjusted Fisher–Pearson, matching DuckDB):
/// `g1·√(n(n−1))/(n−2)` where `g1 = m3 / m2^1.5` and `mk` are the central moments.
/// Null when n < 3 or the variance is zero.
pub(crate) fn finalize_skewness(state: &[ArrayRef]) -> Result<ArrayRef, RuntimeError> {
    moment_finalize(state, |n, m2, m3, _m4| {
        if n < 3.0 || m2 <= 0.0 {
            return None;
        }
        let g1 = m3 / m2.powf(1.5);
        Some(g1 * (n * (n - 1.0)).sqrt() / (n - 2.0))
    })
}

/// Sample excess kurtosis (matching DuckDB): the bias-corrected fourth standardized
/// moment, `0` for a normal distribution. Null when n < 4 or the variance is zero.
pub(crate) fn finalize_kurtosis(state: &[ArrayRef]) -> Result<ArrayRef, RuntimeError> {
    moment_finalize(state, |n, m2, _m3, m4| {
        if n < 4.0 || m2 <= 0.0 {
            return None;
        }
        let g2 = m4 / (m2 * m2);
        let term = (n - 1.0) / ((n - 2.0) * (n - 3.0));
        Some(term * ((n + 1.0) * g2 - 3.0 * (n - 1.0)))
    })
}

/// Shared finalize for the moment aggregates: derive the population central moments
/// `m2/m3/m4` (per element, dividing by n) from the sum-of-powers state and apply
/// `f(n, m2, m3, m4)` (which returns `None` for the null cases).
fn moment_finalize(
    state: &[ArrayRef],
    f: impl Fn(f64, f64, f64, f64) -> Option<f64>,
) -> Result<ArrayRef, RuntimeError> {
    let n = state[0].as_primitive::<Int64Type>();
    let s1 = state[1].as_primitive::<Float64Type>();
    let s2 = state[2].as_primitive::<Float64Type>();
    let s3 = state[3].as_primitive::<Float64Type>();
    let s4 = state[4].as_primitive::<Float64Type>();
    let mut b = Float64Builder::with_capacity(n.len());
    for i in 0..n.len() {
        let cnt = n.value(i) as f64;
        if cnt < 1.0 {
            b.append_null();
            continue;
        }
        let mean = s1.value(i) / cnt;
        // Population central moments via the binomial expansion of Σ(x−μ)^k.
        let m2 = s2.value(i) / cnt - mean * mean;
        let m3 = s3.value(i) / cnt - 3.0 * mean * s2.value(i) / cnt + 2.0 * mean.powi(3);
        let m4 = s4.value(i) / cnt - 4.0 * mean * s3.value(i) / cnt
            + 6.0 * mean * mean * s2.value(i) / cnt
            - 3.0 * mean.powi(4);
        match f(cnt, m2, m3, m4) {
            Some(v) => b.append_value(v),
            None => b.append_null(),
        }
    }
    Ok(Arc::new(b.finish()))
}
