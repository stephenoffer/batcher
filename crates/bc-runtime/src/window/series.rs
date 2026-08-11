//! Series kernels over an ordered partition: EWM statistics, interpolation, run ids.
//!
//! These are the time-series shapes that a `ROWS`/`RANGE` frame cannot express, because
//! each row's answer depends on the *whole prefix* through a recurrence rather than on a
//! bounded set of rows. They share [`crate::window::fill`]'s contract: one sequential pass
//! per ordered partition, an ORDER BY is required, and the result is scattered back to
//! original row order by the caller's `ordered` index lists.
//!
//! * **EWM** (`ewm_mean`/`ewm_var`/`ewm_std`) — exponentially weighted moving statistics.
//!   The weights are `(1-alpha)^(t-i)` over *absolute* positions, which is pandas'
//!   `adjust=True, ignore_na=False` and Polars' `ewm_mean(adjust=True, ignore_nulls=False)`,
//!   the default in both. A null input row yields a null output and contributes nothing,
//!   but still advances the decay — again matching both.
//! * **EWM by elapsed time** (`ewm_mean` with a half-life instead of an alpha) — the same
//!   smoother, decayed by how far apart two readings *are* rather than by how many rows
//!   separate them. It is the form an irregular feed needs: a per-row decay charges an hour
//!   of silence the same weight as a second, which smooths a gappy sensor into nonsense.
//! * **`interpolate`** — fill an interior null run by drawing a straight line between the
//!   bracketing non-null values, weighted by row position (Polars `interpolate`). Nulls
//!   before the first or after the last non-null have nothing to interpolate between and
//!   stay null; `forward_fill`/`backward_fill` are the tools for those.
//! * **`rle_id`** — the 0-based index of the current run of equal values, incrementing
//!   whenever the value changes along the order. It is the segmentation primitive behind
//!   "how long has this sensor been in its current state", and it is type-generic because
//!   it compares arrow's row encoding rather than a typed value.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Float64Array, Int64Array};
use arrow::datatypes::{DataType, Float64Type, Int64Type};
use arrow::row::{RowConverter, SortField};

use crate::error::RuntimeError;
use crate::window::frame::RangeOrder;
use crate::window::WindowFn;

/// Decaying `(sum_w, mean, m2, sum_w2)` state behind the EWM statistics.
///
/// `mean` and `m2` are West's weighted incremental form — the same recurrence
/// [`crate::window::agg`]'s Welford state uses, with the weights decayed in place. The
/// naive alternative (`Σwx²/Σw - mean²`) cancels catastrophically once the values are
/// large relative to their spread, which is the ordinary case for a sensor reading
/// around a large offset.
#[derive(Default)]
struct EwmState {
    /// `Σ wᵢ` over the observed rows, weights decayed to the current position.
    sum_w: f64,
    /// The weighted mean, `Σ wᵢxᵢ / Σ wᵢ`.
    mean: f64,
    /// `Σ wᵢ(xᵢ - mean)²` — the weighted second central moment, undivided.
    m2: f64,
    /// `Σ wᵢ²`, which is what turns the biased variance into the sample one.
    sum_w2: f64,
    /// Whether any value has been observed yet (`sum_w > 0` is not a safe test once
    /// the weights have decayed for many rows).
    seen: bool,
}

impl EwmState {
    /// Age every weight by one position: `w → w·decay`.
    #[inline]
    fn decay(&mut self, decay: f64) {
        self.sum_w *= decay;
        self.m2 *= decay;
        self.sum_w2 *= decay * decay;
    }

    /// Observe `v` with weight 1 (the newest row always carries the full weight).
    #[inline]
    fn push(&mut self, v: f64) {
        self.sum_w += 1.0;
        self.sum_w2 += 1.0;
        let delta = v - self.mean;
        self.mean += delta / self.sum_w;
        self.m2 += delta * (v - self.mean);
        self.seen = true;
    }

    /// The *sample* (debiased) exponentially weighted variance, or `None` when the
    /// effective sample size is one — where it is undefined, exactly as
    /// [`crate::window::agg`]'s `var` returns null for a single row.
    #[inline]
    fn variance(&self) -> Option<f64> {
        let sw2 = self.sum_w * self.sum_w;
        let denom = sw2 - self.sum_w2;
        if !self.seen || denom <= 0.0 {
            return None;
        }
        // `m2` is a sum of squared deviations, so it is non-negative by construction;
        // the clamp only keeps a `-0.0` out of the `sqrt` in the stddev arm.
        Some((self.m2.max(0.0) / self.sum_w) * (sw2 / denom))
    }
}

/// Exponentially weighted `mean`/`var`/`stddev` along each ordered partition.
///
/// `alpha` is the smoothing factor in `(0, 1]`; the caller validates it. Output is
/// Float64 for every numeric input, and null wherever the input row is null.
pub(crate) fn ewm_window(
    func: WindowFn,
    ordered: &[Vec<usize>],
    values: &ArrayRef,
    alpha: f64,
    num_rows: usize,
) -> Result<ArrayRef, RuntimeError> {
    let decay = 1.0 - alpha;
    let mut out = vec![None::<f64>; num_rows];
    // One closure per input type rather than a per-row `match`: the branch is loop
    // invariant, and this kernel is the inner loop of a smoothing pipeline.
    match values.data_type() {
        DataType::Float64 => {
            let arr = values.as_primitive::<Float64Type>();
            run_ewm(func, ordered, decay, &mut out, |row| {
                arr.is_valid(row).then(|| arr.value(row))
            });
        }
        DataType::Int64 => {
            let arr = values.as_primitive::<Int64Type>();
            run_ewm(func, ordered, decay, &mut out, |row| {
                arr.is_valid(row).then(|| arr.value(row) as f64)
            });
        }
        other => {
            return Err(RuntimeError::UnsupportedWindow {
                func: func.name().to_string(),
                dtype: other.to_string(),
            })
        }
    }
    Ok(Arc::new(Float64Array::from(out)))
}

/// The shared EWM scan, parameterized by how a row's value is read.
fn run_ewm(
    func: WindowFn,
    ordered: &[Vec<usize>],
    decay: f64,
    out: &mut [Option<f64>],
    value_at: impl Fn(usize) -> Option<f64>,
) {
    for part in ordered {
        let mut st = EwmState::default();
        for &row in part {
            st.decay(decay);
            let Some(v) = value_at(row) else {
                continue; // null in → null out, but the decay above still aged the state
            };
            st.push(v);
            out[row] = match func {
                WindowFn::EwmMean => st.seen.then_some(st.mean),
                WindowFn::EwmVar => st.variance(),
                WindowFn::EwmStd => st.variance().map(f64::sqrt),
                _ => None,
            };
        }
    }
}

/// Exponentially weighted mean decayed by the ORDER BY key's *elapsed value*.
///
/// `y[i] = (1 - w)·x[i] + w·y[i-1]` where `w = exp(-ln2 · Δ / half_life)` and `Δ` is the gap
/// between this row's order key and the last observed one — Polars' `ewm_mean_by`. A null
/// value yields a null output and leaves the anchor where it was, so the next reading decays
/// from the last one actually seen rather than from a row that had nothing in it.
///
/// `half_life` is in the key's own units, and in microseconds for a temporal key.
pub(crate) fn ewm_by_window(
    ordered: &[Vec<usize>],
    values: &ArrayRef,
    half_life: f64,
    order: &RangeOrder,
    num_rows: usize,
) -> Result<ArrayRef, RuntimeError> {
    let mut out = vec![None::<f64>; num_rows];
    let rate = -std::f64::consts::LN_2 / half_life;
    let run = |out: &mut [Option<f64>], value_at: &dyn Fn(usize) -> Option<f64>| {
        for part in ordered {
            // The last row that carried a value, and the smoothed value there.
            let mut prev: Option<(usize, f64)> = None;
            for &row in part {
                let Some(x) = value_at(row) else { continue };
                let y = match prev {
                    None => x,
                    Some((prow, py)) => match order.gap(prow, row) {
                        // A null order key has no elapsed distance, so there is no weight to
                        // compute; the row starts the recurrence again rather than guessing.
                        None => x,
                        Some(gap) => {
                            let w = (rate * gap).exp();
                            (1.0 - w) * x + w * py
                        }
                    },
                };
                out[row] = Some(y);
                prev = Some((row, y));
            }
        }
    };
    match values.data_type() {
        DataType::Float64 => {
            let arr = values.as_primitive::<Float64Type>();
            run(&mut out, &|row| arr.is_valid(row).then(|| arr.value(row)));
        }
        DataType::Int64 => {
            let arr = values.as_primitive::<Int64Type>();
            run(&mut out, &|row| {
                arr.is_valid(row).then(|| arr.value(row) as f64)
            });
        }
        other => {
            return Err(RuntimeError::UnsupportedWindow {
                func: "ewm_mean".to_string(),
                dtype: other.to_string(),
            })
        }
    }
    Ok(Arc::new(Float64Array::from(out)))
}

/// Linearly interpolate interior null runs along each ordered partition.
///
/// A null at ordered position `p` bracketed by non-null values at `a < p < b` takes
/// `x[a] + (x[b] - x[a]) · (p - a) / (b - a)`. Leading and trailing nulls have no
/// bracket and stay null. Integer input widens to Float64, because an interpolated
/// value between two integers is generally not one.
pub(crate) fn interpolate_window(
    ordered: &[Vec<usize>],
    values: &ArrayRef,
    num_rows: usize,
) -> Result<ArrayRef, RuntimeError> {
    let mut out = vec![None::<f64>; num_rows];
    match values.data_type() {
        DataType::Float64 => {
            let arr = values.as_primitive::<Float64Type>();
            run_interpolate(ordered, &mut out, |row| {
                arr.is_valid(row).then(|| arr.value(row))
            });
        }
        DataType::Int64 => {
            let arr = values.as_primitive::<Int64Type>();
            run_interpolate(ordered, &mut out, |row| {
                arr.is_valid(row).then(|| arr.value(row) as f64)
            });
        }
        other => {
            return Err(RuntimeError::UnsupportedWindow {
                func: "interpolate".to_string(),
                dtype: other.to_string(),
            })
        }
    }
    Ok(Arc::new(Float64Array::from(out)))
}

/// The shared interpolation scan: walk each partition, and on closing a null run whose
/// left edge exists, fill it from the straight line between the two edges.
fn run_interpolate(
    ordered: &[Vec<usize>],
    out: &mut [Option<f64>],
    value_at: impl Fn(usize) -> Option<f64>,
) {
    for part in ordered {
        // Position and value of the last non-null seen; `None` until the first one, which
        // is what leaves a leading null run untouched.
        let mut prev: Option<(usize, f64)> = None;
        for (pos, &row) in part.iter().enumerate() {
            let Some(v) = value_at(row) else { continue };
            out[row] = Some(v);
            if let Some((p_pos, p_val)) = prev {
                let span = (pos - p_pos) as f64;
                for (gap, &grow) in part[p_pos + 1..pos].iter().enumerate() {
                    let t = (gap + 1) as f64 / span;
                    out[grow] = Some(p_val + (v - p_val) * t);
                }
            }
            prev = Some((pos, v));
        }
        // A trailing null run ends the partition with no right edge, so it stays null —
        // no work to undo, since `out` starts null.
    }
}

/// The 0-based run index of each row within its ordered partition: 0 for the first run,
/// incrementing every time the value differs from the previous row's.
///
/// Type-generic by comparing arrow's row encoding, the same equality
/// [`crate::window::frame::PeerGroups`] uses for peer detection — so nulls are equal to
/// each other and distinct from every value, consistently with `GROUP BY`.
pub(crate) fn rle_id_window(
    ordered: &[Vec<usize>],
    values: &ArrayRef,
    num_rows: usize,
) -> Result<ArrayRef, RuntimeError> {
    let converter = RowConverter::new(vec![SortField::new(values.data_type().clone())])?;
    let rows = converter.convert_columns(std::slice::from_ref(values))?;
    let mut out = vec![0i64; num_rows];
    for part in ordered {
        let mut run = 0i64;
        let mut prev = None;
        for &row in part {
            let cur = rows.row(row);
            if prev.is_some_and(|p| p != cur) {
                run += 1;
            }
            prev = Some(cur);
            out[row] = run;
        }
    }
    Ok(Arc::new(Int64Array::from(out)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f64s(arr: &ArrayRef) -> Vec<Option<f64>> {
        let a = arr.as_primitive::<Float64Type>();
        (0..a.len())
            .map(|i| a.is_valid(i).then(|| a.value(i)))
            .collect()
    }

    fn close(got: &[Option<f64>], want: &[Option<f64>], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            match (g, w) {
                (None, None) => {}
                (Some(g), Some(w)) => {
                    assert!((g - w).abs() < 1e-9, "{what}[{i}]: got {g}, want {w}")
                }
                _ => panic!("{what}[{i}]: got {g:?}, want {w:?}"),
            }
        }
    }

    /// `ewm_mean` must reproduce pandas `ewm(alpha=.5).mean()` / Polars
    /// `ewm_mean(alpha=.5, adjust=True, ignore_nulls=False)` on `[1, null, 3, 4]` —
    /// the null yields null but still ages the decay, so the third value is 2.6 and
    /// not the 2.333 an `ignore_nulls=True` reading gives.
    #[test]
    fn ewm_mean_matches_the_reference_semantics() {
        let values: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(1.0),
            None,
            Some(3.0),
            Some(4.0),
        ]));
        let ordered = vec![vec![0usize, 1, 2, 3]];
        let got = ewm_window(WindowFn::EwmMean, &ordered, &values, 0.5, 4).unwrap();
        close(
            &f64s(&got),
            &[Some(1.0), None, Some(2.6), Some(3.4615384615384617)],
            "ewm_mean",
        );
    }

    /// `ewm_std` must match pandas `ewm(alpha=.5).std()` on `[1,2,3,4]`, including the
    /// null first row: the sample form is undefined for one observation, exactly as
    /// the window `stddev` aggregate returns null for a single-row frame.
    #[test]
    fn ewm_std_and_var_match_the_sample_reference() {
        let values: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0]));
        let ordered = vec![vec![0usize, 1, 2, 3]];
        let std = ewm_window(WindowFn::EwmStd, &ordered, &values, 0.5, 4).unwrap();
        let want_std = [
            None,
            // pandas' second value is 1/sqrt(2) exactly: with alpha=.5 the two-observation
            // sample form reduces to |x1 - x0| / sqrt(2). Spelled as the constant because
            // the literal is bit-identical to it and clippy reads a hand-typed copy of a
            // std constant as a typo waiting to happen.
            Some(std::f64::consts::FRAC_1_SQRT_2),
            Some(0.9636241116594315),
            Some(1.1771636613972953),
        ];
        close(&f64s(&std), &want_std, "ewm_std");

        let var = ewm_window(WindowFn::EwmVar, &ordered, &values, 0.5, 4).unwrap();
        let want_var: Vec<Option<f64>> = want_std.iter().map(|v| v.map(|s| s * s)).collect();
        close(&f64s(&var), &want_var, "ewm_var");
    }

    /// The decaying West state must agree with a direct `O(n²)` weighted recompute over
    /// large-magnitude values, where the textbook `Σwx²/Σw − mean²` form cancels away
    /// every significant digit.
    #[test]
    fn ewm_var_is_stable_against_a_large_offset() {
        let raw: Vec<f64> = (0..64).map(|i| 1e9 + (i % 7) as f64).collect();
        let values: ArrayRef = Arc::new(Float64Array::from(raw.clone()));
        let ordered = vec![(0..64).collect::<Vec<usize>>()];
        let alpha = 0.3;
        let got = f64s(&ewm_window(WindowFn::EwmVar, &ordered, &values, alpha, 64).unwrap());
        for (t, slot) in got.iter().enumerate().take(64).skip(1) {
            let w: Vec<f64> = (0..=t)
                .map(|i| (1.0 - alpha).powi((t - i) as i32))
                .collect();
            let sw: f64 = w.iter().sum();
            let sw2: f64 = w.iter().map(|x| x * x).sum();
            let mean: f64 = w.iter().zip(&raw).map(|(w, x)| w * x).sum::<f64>() / sw;
            let m2: f64 = w
                .iter()
                .zip(&raw)
                .map(|(w, x)| w * (x - mean) * (x - mean))
                .sum();
            let want = (m2 / sw) * (sw * sw / (sw * sw - sw2));
            let g = slot.expect("variance defined past the first row");
            assert!(
                (g - want).abs() <= 1e-6 * want.abs().max(1.0),
                "t={t}: got {g}, want {want}"
            );
        }
    }

    /// Partitions are independent: a second series' first row restarts the recurrence
    /// rather than inheriting the first series' state.
    #[test]
    fn ewm_restarts_per_partition() {
        let values: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 100.0, 2.0, 200.0]));
        // Rows 0,2 are one partition; rows 1,3 the other (deliberately interleaved, to
        // exercise the scatter back to original row order).
        let ordered = vec![vec![0usize, 2], vec![1usize, 3]];
        let got = f64s(&ewm_window(WindowFn::EwmMean, &ordered, &values, 0.5, 4).unwrap());
        close(
            &got,
            &[Some(1.0), Some(100.0), Some(5.0 / 3.0), Some(500.0 / 3.0)],
            "ewm_mean per partition",
        );
    }

    /// `alpha = 1` keeps only the current row, so the mean is the identity and the
    /// variance is undefined everywhere (the effective sample size never exceeds one).
    #[test]
    fn ewm_alpha_one_is_the_identity() {
        let values: ArrayRef = Arc::new(Float64Array::from(vec![3.0, 9.0, 4.0]));
        let ordered = vec![vec![0usize, 1, 2]];
        let mean = f64s(&ewm_window(WindowFn::EwmMean, &ordered, &values, 1.0, 3).unwrap());
        close(&mean, &[Some(3.0), Some(9.0), Some(4.0)], "alpha=1 mean");
        let var = f64s(&ewm_window(WindowFn::EwmVar, &ordered, &values, 1.0, 3).unwrap());
        assert_eq!(var, vec![None, None, None]);
    }

    /// Interior null runs are drawn as a straight line between their bracketing values;
    /// leading and trailing runs have no bracket and stay null.
    #[test]
    fn interpolate_fills_only_bracketed_gaps() {
        let values: ArrayRef = Arc::new(Float64Array::from(vec![
            None,
            Some(10.0),
            None,
            None,
            Some(40.0),
            None,
        ]));
        let ordered = vec![vec![0usize, 1, 2, 3, 4, 5]];
        let got = f64s(&interpolate_window(&ordered, &values, 6).unwrap());
        close(
            &got,
            &[None, Some(10.0), Some(20.0), Some(30.0), Some(40.0), None],
            "interpolate",
        );
    }

    /// Integers widen to Float64, because the value between two integers generally is
    /// not one. An all-null partition stays all null.
    #[test]
    fn interpolate_widens_integers_and_tolerates_all_null() {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![Some(0), None, Some(1)]));
        let ordered = vec![vec![0usize, 1, 2]];
        let got = f64s(&interpolate_window(&ordered, &values, 3).unwrap());
        close(&got, &[Some(0.0), Some(0.5), Some(1.0)], "int interpolate");

        let empty: ArrayRef = Arc::new(Float64Array::from(vec![None::<f64>, None, None]));
        let got = f64s(&interpolate_window(&ordered, &empty, 3).unwrap());
        assert_eq!(got, vec![None, None, None]);
    }

    /// Interpolation follows the *ordered* positions, not the physical row order, so a
    /// partition whose rows arrive shuffled interpolates along its sort order.
    #[test]
    fn interpolate_follows_the_partition_order_not_row_order() {
        // Physical rows [10, null, 30] but the order visits them as 2, 1, 0 — so the
        // gap is bracketed by 30 (first in order) and 10 (last).
        let values: ArrayRef = Arc::new(Float64Array::from(vec![Some(10.0), None, Some(30.0)]));
        let ordered = vec![vec![2usize, 1, 0]];
        let got = f64s(&interpolate_window(&ordered, &values, 3).unwrap());
        close(
            &got,
            &[Some(10.0), Some(20.0), Some(30.0)],
            "reverse-ordered interpolate",
        );
    }

    /// Run ids increment on every change and treat consecutive nulls as one run.
    #[test]
    fn rle_id_numbers_runs_and_groups_nulls() {
        use arrow::array::StringArray;
        let values: ArrayRef = Arc::new(StringArray::from(vec![
            Some("on"),
            Some("on"),
            Some("off"),
            None,
            None,
            Some("off"),
        ]));
        let ordered = vec![vec![0usize, 1, 2, 3, 4, 5]];
        let got = rle_id_window(&ordered, &values, 6).unwrap();
        assert_eq!(
            got.as_primitive::<Int64Type>().values(),
            &[0, 0, 1, 2, 2, 3],
            "a value returning after a gap opens a NEW run, it does not rejoin the old one"
        );
    }

    /// Each partition numbers its own runs from zero.
    #[test]
    fn rle_id_restarts_per_partition() {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![1, 5, 1, 5]));
        let ordered = vec![vec![0usize, 2], vec![1usize, 3]];
        let got = rle_id_window(&ordered, &values, 4).unwrap();
        assert_eq!(got.as_primitive::<Int64Type>().values(), &[0, 0, 0, 0]);
    }
}
