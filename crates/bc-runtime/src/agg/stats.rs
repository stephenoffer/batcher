//! Two-input covariance/correlation and single-input skewness/kurtosis.
//!
//! All use a **central-moment** partial state (co-moment for covar/corr, the first
//! four central moments for skew/kurt) accumulated in one pass, and merged with the
//! parallel (Chan/Terriberry) update formulas — associative + commutative, so they
//! distribute single-node and distributed.
//!
//! The earlier state was a *sum-of-powers* (`Σx`, `Σx²`, `Σxy`, …) merged by plain
//! column-wise summation. That is mergeable, but the finalize then recovered the
//! moment as `Σx² − (Σx)²/n` (and `Σxy − Σx·Σy/n`) — a subtraction of two nearly
//! equal large numbers that **catastrophically cancels** when the mean dwarfs the
//! spread: `covar_pop([1e9+1,1e9+2,…],[…])` came back as exactly `0` (true `2`), and
//! `corr` as `NULL`. The central-moment form never forms that difference, so it is
//! accurate at large offsets — the same fix `var` took (Welford / B9).

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Float64Array, Float64Builder, Int64Array};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Float64Type, Int64Type};

use super::AggFunc;
use crate::agg::var::NeumaierSum;
use crate::error::RuntimeError;

/// Per-group covariance/correlation state, 6 columns:
/// `[n, mean_x, mean_y, C2, M2x, M2y]` (n is Int64, the rest Float64), where
/// `C2 = Σ(x−x̄)(y−ȳ)` is the co-moment and `M2x`/`M2y` are the centered squared-
/// deviation sums of `x`/`y`. A pair counts only when both `x` and `y` are non-null.
/// `covar_*` use `[n, C2]`; `corr` uses all six. The state merges via Chan's parallel
/// mean-difference correction ([`merge_covar`]), not summation.
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
    // Split on nullability once, so the two passes below run over raw value slices with no
    // per-row validity lookup. Both arms are `covar_pairs` monomorphized on a pair predicate,
    // so the arithmetic is written once and the null-free arm is the same code with the test
    // folded to a constant. `corr` reads six accumulators over two passes, which makes the
    // per-row overhead this removes proportionally larger than on a scalar aggregate: the
    // H2O `groupby` q9 (`corr(v1, v2)` over 10 M null-free rows) is the shape it serves.
    let pairs = if xa.logical_null_count() == 0 && ya.logical_null_count() == 0 {
        covar_pairs(xa.values(), ya.values(), group_ids, num_groups, |_| true)
    } else {
        covar_pairs(xa.values(), ya.values(), group_ids, num_groups, |i| {
            xa.is_valid(i) && ya.is_valid(i)
        })
    };
    Ok(pairs)
}

/// The two centred passes of [`covar_state`], over raw values with `keep` deciding a pair.
///
/// Two-pass within this partition: pass 1 accumulates the exact per-group sums, pass 2 the
/// centered products around the resulting mean. A single mean rounding followed by
/// well-conditioned centered sums is far more accurate than a streaming (Welford) co-moment
/// at a large offset — it matches DuckDB exactly for a single-partition aggregate (the common
/// small-query path) — and the state still merges via Chan's parallel formula
/// ([`merge_covar`]) across partitions/morsels.
///
/// `xs`/`ys` are the arrays' value buffers, so a slot `keep` rejects may hold anything; it is
/// read only where `keep` says a pair exists, which is exactly where the validity bitmaps say
/// a value does.
fn covar_pairs<F: Fn(usize) -> bool>(
    xs: &[f64],
    ys: &[f64],
    group_ids: &[u32],
    num_groups: usize,
    keep: F,
) -> Vec<ArrayRef> {
    let mut n = vec![0i64; num_groups];
    let mut sum_x = vec![NeumaierSum::default(); num_groups];
    let mut sum_y = vec![NeumaierSum::default(); num_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        if keep(i) {
            let g = g as usize;
            n[g] += 1;
            sum_x[g].add(xs[i]);
            sum_y[g].add(ys[i]);
        }
    }
    let mut mean_x = vec![0f64; num_groups];
    let mut mean_y = vec![0f64; num_groups];
    for g in 0..num_groups {
        if n[g] > 0 {
            let nn = n[g] as f64;
            mean_x[g] = sum_x[g].total() / nn;
            mean_y[g] = sum_y[g].total() / nn;
        }
    }
    let mut c2 = vec![0f64; num_groups];
    let mut m2x = vec![0f64; num_groups];
    let mut m2y = vec![0f64; num_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        if keep(i) {
            let g = g as usize;
            let dx = xs[i] - mean_x[g];
            let dy = ys[i] - mean_y[g];
            c2[g] += dx * dy;
            m2x[g] += dx * dx;
            m2y[g] += dy * dy;
        }
    }
    vec![
        Arc::new(Int64Array::from(n)),
        Arc::new(Float64Array::from(mean_x)),
        Arc::new(Float64Array::from(mean_y)),
        Arc::new(Float64Array::from(c2)),
        Arc::new(Float64Array::from(m2x)),
        Arc::new(Float64Array::from(m2y)),
    ]
}

/// Merge partial covar/corr states by group with Chan's parallel formula — the
/// mergeable combine for [`covar_state`]. Associative + commutative, so partials
/// merge in any order and single-node == distributed.
pub(crate) fn merge_covar(
    state: &[ArrayRef],
    group_ids: &[u32],
    num_groups: usize,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    let n_in = state[0].as_primitive::<Int64Type>();
    let mx_in = state[1].as_primitive::<Float64Type>();
    let my_in = state[2].as_primitive::<Float64Type>();
    let c2_in = state[3].as_primitive::<Float64Type>();
    let m2x_in = state[4].as_primitive::<Float64Type>();
    let m2y_in = state[5].as_primitive::<Float64Type>();

    let mut n = vec![0i64; num_groups];
    let mut mean_x = vec![0f64; num_groups];
    let mut mean_y = vec![0f64; num_groups];
    let mut c2 = vec![0f64; num_groups];
    let mut m2x = vec![0f64; num_groups];
    let mut m2y = vec![0f64; num_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        let nb = n_in.value(i);
        if nb == 0 {
            continue;
        }
        let g = g as usize;
        let na = n[g];
        if na == 0 {
            // First partial for this group: copy it exactly. Folding it through the
            // mean-difference formula instead is algebraically the same but rounds three
            // times on the way, so a single-partial group would not equal its own state.
            n[g] = nb;
            mean_x[g] = mx_in.value(i);
            mean_y[g] = my_in.value(i);
            c2[g] = c2_in.value(i);
            m2x[g] = m2x_in.value(i);
            m2y[g] = m2y_in.value(i);
            continue;
        }
        let ntot = na + nb;
        let (naf, nbf, nf) = (na as f64, nb as f64, ntot as f64);
        let dx = mx_in.value(i) - mean_x[g];
        let dy = my_in.value(i) - mean_y[g];
        mean_x[g] += dx * nbf / nf;
        mean_y[g] += dy * nbf / nf;
        c2[g] += c2_in.value(i) + dx * dy * naf * nbf / nf;
        m2x[g] += m2x_in.value(i) + dx * dx * naf * nbf / nf;
        m2y[g] += m2y_in.value(i) + dy * dy * naf * nbf / nf;
        n[g] = ntot;
    }
    Ok(vec![
        Arc::new(Int64Array::from(n)),
        Arc::new(Float64Array::from(mean_x)),
        Arc::new(Float64Array::from(mean_y)),
        Arc::new(Float64Array::from(c2)),
        Arc::new(Float64Array::from(m2x)),
        Arc::new(Float64Array::from(m2y)),
    ])
}

/// Per-group central-moment state for skewness/kurtosis, 5 columns:
/// `[n, mean, M2, M3, M4]` (n is Int64, the rest Float64), where `Mk = Σ(x−x̄)^k`.
/// Null-skipping; merges via Terriberry's parallel formula ([`merge_moments`]).
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
    // Two-pass within this partition (exact mean, then centered power sums): far more
    // accurate at a large offset than a streaming higher-moment update, matching DuckDB
    // exactly for a single-partition aggregate. The state still merges across partitions
    // via Terriberry's parallel formula ([`merge_moments`]).
    let mut n = vec![0i64; num_groups];
    let mut sums = vec![NeumaierSum::default(); num_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        if a.is_valid(i) {
            let g = g as usize;
            n[g] += 1;
            sums[g].add(a.value(i));
        }
    }
    let mut mean = vec![0f64; num_groups];
    for g in 0..num_groups {
        if n[g] > 0 {
            mean[g] = sums[g].total() / n[g] as f64;
        }
    }
    let mut m2 = vec![0f64; num_groups];
    let mut m3 = vec![0f64; num_groups];
    let mut m4 = vec![0f64; num_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        if a.is_valid(i) {
            let g = g as usize;
            let d = a.value(i) - mean[g];
            let d2 = d * d;
            m2[g] += d2;
            m3[g] += d2 * d;
            m4[g] += d2 * d2;
        }
    }
    Ok(vec![
        Arc::new(Int64Array::from(n)),
        Arc::new(Float64Array::from(mean)),
        Arc::new(Float64Array::from(m2)),
        Arc::new(Float64Array::from(m3)),
        Arc::new(Float64Array::from(m4)),
    ])
}

/// Merge partial `[n, mean, M2, M3, M4]` states by group with Terriberry's parallel
/// higher-moment formula — the mergeable combine for [`moment_state`]. Associative +
/// commutative, so partials merge in any order and single-node == distributed.
pub(crate) fn merge_moments(
    state: &[ArrayRef],
    group_ids: &[u32],
    num_groups: usize,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    let n_in = state[0].as_primitive::<Int64Type>();
    let mean_in = state[1].as_primitive::<Float64Type>();
    let m2_in = state[2].as_primitive::<Float64Type>();
    let m3_in = state[3].as_primitive::<Float64Type>();
    let m4_in = state[4].as_primitive::<Float64Type>();

    let mut n = vec![0i64; num_groups];
    let mut mean = vec![0f64; num_groups];
    let mut m2 = vec![0f64; num_groups];
    let mut m3 = vec![0f64; num_groups];
    let mut m4 = vec![0f64; num_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        let nb = n_in.value(i);
        if nb == 0 {
            continue;
        }
        let g = g as usize;
        let na = n[g];
        if na == 0 {
            n[g] = nb;
            mean[g] = mean_in.value(i);
            m2[g] = m2_in.value(i);
            m3[g] = m3_in.value(i);
            m4[g] = m4_in.value(i);
            continue;
        }
        let ntot = na + nb;
        let (naf, nbf, nf) = (na as f64, nb as f64, ntot as f64);
        let delta = mean_in.value(i) - mean[g];
        let delta2 = delta * delta;
        let delta3 = delta2 * delta;
        let delta4 = delta2 * delta2;
        let (m2a, m2b) = (m2[g], m2_in.value(i));
        let (m3a, m3b) = (m3[g], m3_in.value(i));
        let (m4a, m4b) = (m4[g], m4_in.value(i));
        let new_mean = mean[g] + delta * nbf / nf;
        let new_m2 = m2a + m2b + delta2 * naf * nbf / nf;
        let new_m3 = m3a
            + m3b
            + delta3 * naf * nbf * (naf - nbf) / (nf * nf)
            + 3.0 * delta * (naf * m2b - nbf * m2a) / nf;
        let new_m4 = m4a
            + m4b
            + delta4 * naf * nbf * (naf * naf - naf * nbf + nbf * nbf) / (nf * nf * nf)
            + 6.0 * delta2 * (naf * naf * m2b + nbf * nbf * m2a) / (nf * nf)
            + 4.0 * delta * (naf * m3b - nbf * m3a) / nf;
        n[g] = ntot;
        mean[g] = new_mean;
        m2[g] = new_m2;
        m3[g] = new_m3;
        m4[g] = new_m4;
    }
    Ok(vec![
        Arc::new(Int64Array::from(n)),
        Arc::new(Float64Array::from(mean)),
        Arc::new(Float64Array::from(m2)),
        Arc::new(Float64Array::from(m3)),
        Arc::new(Float64Array::from(m4)),
    ])
}

/// `covar_pop = C2/n` (n ≥ 1) or `covar_samp = C2/(n−1)` (n ≥ 2). Null when the count
/// is too small.
pub(crate) fn finalize_covar(state: &[ArrayRef], sample: bool) -> Result<ArrayRef, RuntimeError> {
    let n = state[0].as_primitive::<Int64Type>();
    let c2 = state[3].as_primitive::<Float64Type>();
    let mut b = Float64Builder::with_capacity(n.len());
    for i in 0..n.len() {
        let cnt = n.value(i);
        if sample {
            if cnt < 2 {
                b.append_null();
            } else {
                b.append_value(c2.value(i) / (cnt - 1) as f64);
            }
        } else if cnt < 1 {
            b.append_null();
        } else {
            b.append_value(c2.value(i) / cnt as f64);
        }
    }
    Ok(Arc::new(b.finish()))
}

/// `corr = C2 / sqrt(M2x · M2y)`. Null when n < 2 or either variable has zero variance
/// (a flat column has no correlation).
pub(crate) fn finalize_corr(state: &[ArrayRef]) -> Result<ArrayRef, RuntimeError> {
    let n = state[0].as_primitive::<Int64Type>();
    let c2 = state[3].as_primitive::<Float64Type>();
    let m2x = state[4].as_primitive::<Float64Type>();
    let m2y = state[5].as_primitive::<Float64Type>();
    let mut b = Float64Builder::with_capacity(n.len());
    for i in 0..n.len() {
        if n.value(i) < 2 {
            b.append_null();
            continue;
        }
        // `sqrt(M2x) · sqrt(M2y)`, never `sqrt(M2x · M2y)`. The product overflows to
        // infinity once either centered sum passes ~1e154 — a column of values around 1e80,
        // which a physics or financial dataset reaches — and the correlation of two perfectly
        // correlated columns then came back as 0 rather than 1. Taking each root first keeps
        // every intermediate inside the representable range, and underflows gracefully too.
        let denom = m2x.value(i).max(0.0).sqrt() * m2y.value(i).max(0.0).sqrt();
        if denom == 0.0 || !denom.is_finite() {
            b.append_null();
        } else {
            // Cauchy-Schwarz bounds the true value to [-1, 1]; rounding in the three
            // accumulated moments can put the quotient a few ulps outside it, and a
            // correlation of 1.0000000000000002 is a wrong answer, not a rounding detail.
            b.append_value((c2.value(i) / denom).clamp(-1.0, 1.0));
        }
    }
    Ok(Arc::new(b.finish()))
}

/// Sample skewness (adjusted Fisher–Pearson, matching DuckDB):
/// `g1·√(n(n−1))/(n−2)` where `g1 = m3 / m2^1.5` and `mk` are the population central
/// moments. Null when n < 3 or the variance is zero.
pub(crate) fn finalize_skewness(state: &[ArrayRef]) -> Result<ArrayRef, RuntimeError> {
    moment_finalize(state, |n, m2, m3, _m4| {
        if n < 3.0 || m2 <= 0.0 {
            return None;
        }
        // `m2 · sqrt(m2)`, not `m2.powf(1.5)`: `powf` goes through `exp(1.5·ln m2)` and
        // carries the rounding of both transcendentals, where the explicit form is one
        // correctly-rounded square root and one multiply.
        let g1 = m3 / (m2 * m2.sqrt());
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

/// `kurtosis_pop` — the **population** excess kurtosis `m4/m2² - 3`, where
/// [`finalize_kurtosis`] applies the sample correction. DuckDB has both under these two
/// names, and the difference is large on a small group (5.71 against 1.63 on seven
/// values), so mapping one name to the other is a wrong answer rather than a rounding.
/// Null for a group with no variance, where the ratio is 0/0.
pub(crate) fn finalize_kurtosis_pop(state: &[ArrayRef]) -> Result<ArrayRef, RuntimeError> {
    moment_finalize(state, |n, m2, _m3, m4| {
        if n < 1.0 || m2 <= 0.0 {
            return None;
        }
        Some(m4 / (m2 * m2) - 3.0)
    })
}

/// Shared finalize for the moment aggregates: read the population central moments
/// `m2/m3/m4` (per element, dividing the accumulated `Mk` by n) from the state and
/// apply `f(n, m2, m3, m4)` (which returns `None` for the null cases).
fn moment_finalize(
    state: &[ArrayRef],
    f: impl Fn(f64, f64, f64, f64) -> Option<f64>,
) -> Result<ArrayRef, RuntimeError> {
    let n = state[0].as_primitive::<Int64Type>();
    let m2 = state[2].as_primitive::<Float64Type>();
    let m3 = state[3].as_primitive::<Float64Type>();
    let m4 = state[4].as_primitive::<Float64Type>();
    let mut b = Float64Builder::with_capacity(n.len());
    for i in 0..n.len() {
        let cnt = n.value(i);
        if cnt < 1 {
            b.append_null();
            continue;
        }
        let nf = cnt as f64;
        match f(nf, m2.value(i) / nf, m3.value(i) / nf, m4.value(i) / nf) {
            Some(v) => b.append_value(v),
            None => b.append_null(),
        }
    }
    Ok(Arc::new(b.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f64s(v: &[f64]) -> ArrayRef {
        Arc::new(Float64Array::from(v.to_vec()))
    }

    /// The whole-input central-moment state, finalized.
    fn covar_whole(x: &[f64], y: &[f64], sample: bool) -> Option<f64> {
        let g = vec![0u32; x.len()];
        let st = covar_state(&f64s(x), &f64s(y), &g, 1).unwrap();
        let out = finalize_covar(&st, sample).unwrap();
        let a = out.as_primitive::<Float64Type>();
        a.is_valid(0).then(|| a.value(0))
    }

    /// covar at a large offset must not catastrophically cancel: covar([1e9+i],[…])
    /// with the sum-of-products formula returned 0; the co-moment form returns 2.
    #[test]
    fn covar_stable_at_large_offset() {
        let x = [1e9 + 1.0, 1e9 + 2.0, 1e9 + 3.0, 1e9 + 4.0, 1e9 + 5.0];
        let y = [1e9 + 1.0, 1e9 + 3.0, 1e9 + 2.0, 1e9 + 7.0, 1e9 + 4.0];
        let pop = covar_whole(&x, &y, false).unwrap();
        assert!((pop - 2.0).abs() < 1e-6, "covar_pop {pop} != 2.0");
        let samp = covar_whole(&x, &y, true).unwrap();
        assert!((samp - 2.5).abs() < 1e-6, "covar_samp {samp} != 2.5");
    }

    /// corr at a large offset: the naive form denominator cancelled to 0 → NULL; the
    /// centered form recovers ~0.6868 (DuckDB).
    #[test]
    fn corr_stable_at_large_offset() {
        let x = [1e9 + 1.0, 1e9 + 2.0, 1e9 + 3.0, 1e9 + 4.0, 1e9 + 5.0];
        let y = [1e9 + 1.0, 1e9 + 3.0, 1e9 + 2.0, 1e9 + 7.0, 1e9 + 4.0];
        let g = vec![0u32; x.len()];
        let st = covar_state(&f64s(&x), &f64s(&y), &g, 1).unwrap();
        let out = finalize_corr(&st).unwrap();
        let a = out.as_primitive::<Float64Type>();
        assert!(a.is_valid(0), "corr must not be NULL");
        assert!(
            (a.value(0) - 0.6868028194537991).abs() < 1e-9,
            "corr {} != 0.6868",
            a.value(0)
        );
    }

    /// Partial→merge→finalize must equal the whole-input covar/corr (single-node ==
    /// distributed), even split across an uneven chunk boundary at a large offset.
    #[test]
    fn covar_corr_merge_equals_whole() {
        let x: Vec<f64> = (0..97).map(|i| 1000.0 + (i as f64) * 0.5).collect();
        let y: Vec<f64> = (0..97)
            .map(|i| 1000.0 - (i as f64) * 0.3 + (i % 7) as f64)
            .collect();
        let g = vec![0u32; x.len()];
        let whole = covar_state(&f64s(&x), &f64s(&y), &g, 1).unwrap();
        // Split into three chunks, build a partial per chunk, concat, and merge.
        let bounds = [0usize, 13, 55, 97];
        let mut ns = Vec::new();
        let (mut mx, mut my, mut c2, mut m2x, mut m2y) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for w in bounds.windows(2) {
            let (a, b) = (w[0], w[1]);
            let g = vec![0u32; b - a];
            let st = covar_state(&f64s(&x[a..b]), &f64s(&y[a..b]), &g, 1).unwrap();
            ns.push(st[0].as_primitive::<Int64Type>().value(0));
            mx.push(st[1].as_primitive::<Float64Type>().value(0));
            my.push(st[2].as_primitive::<Float64Type>().value(0));
            c2.push(st[3].as_primitive::<Float64Type>().value(0));
            m2x.push(st[4].as_primitive::<Float64Type>().value(0));
            m2y.push(st[5].as_primitive::<Float64Type>().value(0));
        }
        let cat: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(ns)),
            Arc::new(Float64Array::from(mx)),
            Arc::new(Float64Array::from(my)),
            Arc::new(Float64Array::from(c2)),
            Arc::new(Float64Array::from(m2x)),
            Arc::new(Float64Array::from(m2y)),
        ];
        let gids: Vec<u32> = vec![0; 3];
        let merged = merge_covar(&cat, &gids, 1).unwrap();
        for stat in [false, true] {
            let w = finalize_covar(&whole, stat).unwrap();
            let m = finalize_covar(&merged, stat).unwrap();
            let wv = w.as_primitive::<Float64Type>().value(0);
            let mv = m.as_primitive::<Float64Type>().value(0);
            assert!((wv - mv).abs() < 1e-6, "covar merge {mv} != whole {wv}");
        }
        let wc = finalize_corr(&whole).unwrap();
        let mc = finalize_corr(&merged).unwrap();
        assert!(
            (wc.as_primitive::<Float64Type>().value(0) - mc.as_primitive::<Float64Type>().value(0))
                .abs()
                < 1e-9,
            "corr merge != whole"
        );
    }

    /// skewness/kurtosis stable at a large offset and mergeable across chunks.
    #[test]
    fn moments_merge_equals_whole() {
        let x: Vec<f64> = (0..120).map(|i| 1000.0 + ((i * 7) % 13) as f64).collect();
        let g = vec![0u32; x.len()];
        let whole = moment_state(&f64s(&x), &g, 1, AggFunc::Skewness).unwrap();
        let bounds = [0usize, 17, 60, 120];
        let (mut ns, mut me, mut m2, mut m3, mut m4) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for w in bounds.windows(2) {
            let (a, b) = (w[0], w[1]);
            let g = vec![0u32; b - a];
            let st = moment_state(&f64s(&x[a..b]), &g, 1, AggFunc::Skewness).unwrap();
            ns.push(st[0].as_primitive::<Int64Type>().value(0));
            me.push(st[1].as_primitive::<Float64Type>().value(0));
            m2.push(st[2].as_primitive::<Float64Type>().value(0));
            m3.push(st[3].as_primitive::<Float64Type>().value(0));
            m4.push(st[4].as_primitive::<Float64Type>().value(0));
        }
        let cat: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(ns)),
            Arc::new(Float64Array::from(me)),
            Arc::new(Float64Array::from(m2)),
            Arc::new(Float64Array::from(m3)),
            Arc::new(Float64Array::from(m4)),
        ];
        let merged = merge_moments(&cat, &[0u32; 3], 1).unwrap();
        for (wcol, mcol) in whole.iter().zip(&merged) {
            // Compare each moment column (skip n, which is exact).
            if let (Some(wf), Some(mf)) = (
                wcol.as_any().downcast_ref::<Float64Array>(),
                mcol.as_any().downcast_ref::<Float64Array>(),
            ) {
                let rel = (wf.value(0) - mf.value(0)).abs() / (wf.value(0).abs() + 1.0);
                assert!(
                    rel < 1e-9,
                    "moment merge {} != whole {}",
                    mf.value(0),
                    wf.value(0)
                );
            }
        }
        let ws = finalize_skewness(&whole).unwrap();
        let ms = finalize_skewness(&merged).unwrap();
        assert!(
            (ws.as_primitive::<Float64Type>().value(0) - ms.as_primitive::<Float64Type>().value(0))
                .abs()
                < 1e-9
        );
        let wk = finalize_kurtosis(&whole).unwrap();
        let mk = finalize_kurtosis(&merged).unwrap();
        assert!(
            (wk.as_primitive::<Float64Type>().value(0) - mk.as_primitive::<Float64Type>().value(0))
                .abs()
                < 1e-9
        );
    }

    /// `corr` of two perfectly correlated columns must be 1, even when the centered sums are
    /// large enough that their *product* overflows to infinity.
    ///
    /// `sqrt(M2x · M2y)` overflows once either moment passes ~1e154; the denominator became
    /// infinite and the correlation came back as 0 — the opposite of the truth — for data
    /// that is entirely representable. Taking each square root first keeps every intermediate
    /// finite.
    #[test]
    fn corr_survives_moments_whose_product_overflows() {
        let x: Vec<f64> = (1..=5).map(|i| i as f64 * 1e120).collect();
        let y: Vec<f64> = (1..=5).map(|i| i as f64 * 1e120).collect();
        let g = vec![0u32; x.len()];
        let st = covar_state(&f64s(&x), &f64s(&y), &g, 1).unwrap();
        // The guard the fix exists for: the product really does overflow here.
        let m2x = st[4].as_primitive::<Float64Type>().value(0);
        assert!(
            !(m2x * m2x).is_finite(),
            "test no longer exercises the overflow"
        );
        let out = finalize_corr(&st).unwrap();
        let a = out.as_primitive::<Float64Type>();
        assert!(a.is_valid(0), "corr must not be null here");
        assert!((a.value(0) - 1.0).abs() < 1e-12, "corr = {}", a.value(0));
    }

    /// A correlation is bounded to [-1, 1] by Cauchy-Schwarz; rounding across three
    /// accumulated moments must not be allowed to report a value outside it.
    #[test]
    fn corr_is_clamped_to_the_unit_interval() {
        for (x, y) in [
            (vec![1.0, 2.0, 3.0, 4.0], vec![2.0, 4.0, 6.0, 8.0]),
            (vec![1.0, 2.0, 3.0, 4.0], vec![-2.0, -4.0, -6.0, -8.0]),
        ] {
            let g = vec![0u32; x.len()];
            let st = covar_state(&f64s(&x), &f64s(&y), &g, 1).unwrap();
            let out = finalize_corr(&st).unwrap();
            let v = out.as_primitive::<Float64Type>().value(0);
            assert!((-1.0..=1.0).contains(&v), "corr {v} outside [-1, 1]");
        }
    }

    /// The two-pass moment accumulators condition everything on the pass-1 mean, so that sum
    /// is compensated. A naive running sum of many values around a large offset loses the
    /// low bits of each addend and biases the mean, which the centering then amplifies.
    #[test]
    fn compensated_mean_keeps_the_moment_state_accurate() {
        // 1e6 values whose true variance is exactly known, offset far enough that a naive
        // sum drops bits on every addition.
        let n = 200_000;
        let values: Vec<f64> = (0..n).map(|i| 1e9 + (i % 2) as f64).collect();
        let g = vec![0u32; values.len()];
        let st = moment_state(&f64s(&values), &g, 1, AggFunc::Skewness).unwrap();
        let m2 = st[2].as_primitive::<Float64Type>().value(0);
        // Half the values are 1e9 and half 1e9+1, so Σ(x-x̄)² is exactly n/4.
        let expected = n as f64 / 4.0;
        assert!(
            (m2 - expected).abs() / expected < 1e-9,
            "M2 {m2} differs from the exact {expected}"
        );
    }
}
