//! Variance / standard-deviation / mean finalizers and their shared
//! (sum, sum_of_squares, count) partial-state producer.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Float64Array, Float64Builder, Int64Array};
use arrow::datatypes::{DataType, Float64Type, Int64Type};

use super::AggFunc;
use crate::error::RuntimeError;

/// Kahan–Babuška–Neumaier compensated summation.
///
/// A running `f64` sum loses the low bits of every addend whose magnitude is far below the
/// accumulator's, so the naive error grows as `O(n·ε·Σ|xᵢ|)` — over ten million values that
/// is already visible in the eighth significant digit, and it is *systematic* rather than
/// random. Neumaier's variant keeps the lost part in a separate compensation term and adds it
/// back at the end, bringing the error down to `O(ε·Σ|xᵢ|)` independent of `n`.
///
/// It is the Neumaier form rather than plain Kahan because it also handles the case where the
/// **addend** is larger than the accumulator (the first values of a group, or a stream whose
/// magnitudes grow), which plain Kahan silently gets wrong.
///
/// Used where an accurate mean is the foundation of everything computed afterwards: the
/// centered two-pass moment accumulators condition their whole result on it, so an error in
/// the mean is amplified by the centering rather than averaged away.
#[derive(Clone, Copy, Default)]
pub(crate) struct NeumaierSum {
    sum: f64,
    compensation: f64,
}

impl NeumaierSum {
    #[inline]
    pub(crate) fn add(&mut self, value: f64) {
        let t = self.sum + value;
        self.compensation += if self.sum.abs() >= value.abs() {
            // The accumulator dominates: the low bits of `value` are what got lost.
            (self.sum - t) + value
        } else {
            // The addend dominates: the low bits of the accumulator are what got lost.
            (value - t) + self.sum
        };
        self.sum = t;
    }

    #[inline]
    pub(crate) fn total(&self) -> f64 {
        self.sum + self.compensation
    }
}

/// One-pass **Welford** (mean, M2, count) per group, read as f64.
///
/// `M2` is the sum of squared deviations from the group mean. The earlier state was
/// `(Σx, Σx², n)`, and `finalize` recovered the variance as `Σx² − (Σx)²/n` — a
/// subtraction of two nearly equal large numbers that catastrophically cancels when the
/// mean dwarfs the spread: `var([1e9+1, 1e9+2, 1e9+3])` came back as exactly `0` instead
/// of `1`. Welford accumulates the centered `M2` directly, so no such subtraction ever
/// happens, and the state stays mergeable via Chan's parallel formula ([`merge_welford`]).
pub(crate) fn var_state(
    values: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
    func: AggFunc,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    let mut mean = vec![0f64; num_groups];
    let mut m2 = vec![0f64; num_groups];
    let mut count = vec![0i64; num_groups];

    let mut update = |g: usize, v: f64| {
        count[g] += 1;
        let delta = v - mean[g];
        mean[g] += delta / count[g] as f64;
        let delta2 = v - mean[g];
        m2[g] += delta * delta2;
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
        Arc::new(Float64Array::from(mean)),
        Arc::new(Float64Array::from(m2)),
        Arc::new(Int64Array::from(count)),
    ])
}

/// Merge partial `(mean, M2, count)` states by group using Chan's parallel algorithm —
/// the mergeable combine for [`var_state`]. Each concatenated partial row `i` carries one
/// group's partial mean/M2/count and lands in output group `group_ids[i]`; folding them
/// with Chan's mean-difference correction is associative and commutative, so partials
/// merge in any order and single-node == distributed.
pub(crate) fn merge_welford(
    mean_in: &ArrayRef,
    m2_in: &ArrayRef,
    count_in: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
) -> Vec<ArrayRef> {
    let mean_in = mean_in.as_primitive::<Float64Type>();
    let m2_in = m2_in.as_primitive::<Float64Type>();
    let count_in = count_in.as_primitive::<Int64Type>();

    let mut mean = vec![0f64; num_groups];
    let mut m2 = vec![0f64; num_groups];
    let mut count = vec![0i64; num_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        let g = g as usize;
        let nb = count_in.value(i);
        if nb == 0 {
            continue;
        }
        let mb = mean_in.value(i);
        let m2b = m2_in.value(i);
        let na = count[g];
        if na == 0 {
            // First partial for this group: copy it exactly rather than folding it through
            // the correction formula, which is algebraically identical but rounds on the way.
            // A group whose rows all landed in one partition must finalize to exactly the
            // value that partition computed.
            count[g] = nb;
            mean[g] = mb;
            m2[g] = m2b;
            continue;
        }
        let n = na + nb;
        let (naf, nbf, nf) = (na as f64, nb as f64, n as f64);
        let delta = mb - mean[g];
        // The mean is combined as the weighted average `(na·ma + nb·mb)/n` when the two
        // counts are comparable, and as the incremental `ma + delta·nb/n` when one partial
        // is much smaller. Chan's paper makes exactly this distinction: the incremental form
        // is the accurate one for an unbalanced merge (it perturbs a good mean slightly),
        // while the weighted average is the accurate one for a balanced merge (the
        // incremental form there subtracts two nearly equal means and scales the difference
        // back up, amplifying its rounding). A morsel-parallel aggregate merges *balanced*
        // partials by construction, which is the case the incremental form serves worst.
        mean[g] = if naf.max(nbf) <= 4.0 * naf.min(nbf) {
            (naf * mean[g] + nbf * mb) / nf
        } else {
            mean[g] + delta * nbf / nf
        };
        m2[g] += m2b + delta * delta * naf * nbf / nf;
        count[g] = n;
    }
    vec![
        Arc::new(Float64Array::from(mean)),
        Arc::new(Float64Array::from(m2)),
        Arc::new(Int64Array::from(count)),
    ]
}

pub(crate) fn count_non_null(values: &ArrayRef, group_ids: &[u32], num_groups: usize) -> ArrayRef {
    // Global (single-group) fast path: the global-aggregate partial passes an empty
    // `group_ids` with `num_groups == 1` (every row is the one group), so the count is the
    // whole column's non-null total — no per-row group-id buffer needed (the same
    // single-group short-circuit `sum_acc`/`minmax_acc` take for a keyless COUNT/AVG).
    if num_groups == 1 && group_ids.is_empty() {
        let c = (values.len() - values.null_count()) as i64;
        return Arc::new(Int64Array::from(vec![c]));
    }
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

/// Finalize sample variance (or its sqrt for stddev) from Welford `(mean, M2, count)`.
/// `var = M2 / (n − 1)`; null when `n < 2`. (The first arg is `mean`, unused here but
/// kept in the state triple because `covar`/`corr` need it; named `_mean` for clarity.)
pub(crate) fn finalize_var(
    _mean: &ArrayRef,
    m2: &ArrayRef,
    count: &ArrayRef,
    stddev: bool,
) -> Result<ArrayRef, RuntimeError> {
    let m2 = m2.as_primitive::<Float64Type>();
    let count = count.as_primitive::<Int64Type>();
    let mut b = Float64Builder::with_capacity(count.len());
    for i in 0..count.len() {
        let n = count.value(i);
        if n < 2 {
            b.append_null();
            continue;
        }
        let var = (m2.value(i) / (n - 1) as f64).max(0.0); // guard tiny negatives
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Neumaier compensation must recover a sum a naive accumulator cannot.
    #[test]
    fn compensated_summation_recovers_what_naive_addition_drops() {
        let mut naive = 1e16f64;
        let mut compensated = NeumaierSum::default();
        compensated.add(1e16);
        for _ in 0..10_000 {
            naive += 1.0;
            compensated.add(1.0);
        }
        assert_eq!(
            naive, 1e16,
            "the naive sum is expected to lose every addend"
        );
        assert_eq!(compensated.total(), 1e16 + 10_000.0);
    }

    /// The Neumaier form (unlike plain Kahan) must also be correct when the addend is much
    /// larger than the accumulator.
    #[test]
    fn compensated_summation_handles_a_growing_magnitude() {
        let mut s = NeumaierSum::default();
        for v in [1.0, 1e100, 1.0, -1e100] {
            s.add(v);
        }
        assert_eq!(s.total(), 2.0);
    }
}
