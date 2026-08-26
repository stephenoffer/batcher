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
///
/// # Which Welford, and why not the other one
///
/// This is **Knuth's** recurrence, `M2 += δ·(x − mean_after)`. The alternative — Youngs–Cramer,
/// `M2 += δ²·n_before/n` — is algebraically identical, is what [`merge_welford`] uses for its
/// correction term, and is *better on exactly one thing*: it never reads back the mean it has
/// just rounded, so over `{2^53, 2^53+2}` it returns the exact variance 2 where this form
/// returns 4. It was tried here and **reverted**, because that is the only thing it is better
/// at and the trade is bad:
///
/// | shape | Knuth rel. err | Youngs–Cramer |
/// |---|---:|---:|
/// | plain small vectors | 9.7e-17 | 1.0e-16 |
/// | offset 1e15, unit spread | **1.8e-03** | 1.4e-02 |
/// | `[1e12 + (i%5)]`, grouped | **5.0e-06** | 1.3e-05 |
///
/// Knuth wins five of seven measured shapes, is *eight times* better at an offset of 1e15, and
/// is 2.5x better on the exact data in `test_diff_numeric_edges::
/// test_grouped_variance_stable_and_matches_duckdb` — which Youngs–Cramer turned red. Reading
/// the deviation against the *updated* mean is not a defect in Knuth's form; it is what makes
/// it partially self-correcting for the mean's own rounding, which is the dominant error term
/// whenever a large offset swamps the spread. That is the shape this module exists to serve
/// (see the `1e9` case below), so the rare `2^53` case does not buy it.
///
/// Two traps for whoever revisits this. **DuckDB is not an oracle for the `2^53` shape** — 1.5.5
/// answers `2.0` for those two doubles via a SQL literal union and `4.0` via a registered Arrow
/// table, same session, values echoing back identically; it is the aggregate path that differs.
/// And **"agrees with DuckDB more often on random data" is not the property to optimize**: it
/// measures which engine's rounding you happen to match, not which answer is right. Compare
/// against exact rational arithmetic, on the shapes the engine actually sees.
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
    mean: &ArrayRef,
    m2: &ArrayRef,
    count: &ArrayRef,
    stddev: bool,
) -> Result<ArrayRef, RuntimeError> {
    let mean = mean.as_primitive::<Float64Type>();
    let m2 = m2.as_primitive::<Float64Type>();
    let count = count.as_primitive::<Int64Type>();
    let mut b = Float64Builder::with_capacity(count.len());
    for i in 0..count.len() {
        let n = count.value(i);
        if n < 2 {
            b.append_null();
            continue;
        }
        // **A non-finite mean means a non-finite answer, and the mean is what decides it.**
        //
        // `M2` alone cannot: the recurrence's `M2` for a group containing an infinity is
        // `NaN` or `+inf` depending on *where in the group the infinity fell* — verified over
        // every permutation of `[1.0, inf, 2.0]`. An aggregate whose answer depends on row
        // order is not mergeable, and mergeability is the invariant that makes a distributed
        // result equal a single-node one, so this cannot be left to `M2`.
        //
        // The mean is non-finite in *every* one of those permutations, and in every merge
        // order too, because one non-finite value makes `delta` non-finite and the mean never
        // recovers. Keying on it restores an order-independent answer, and it also catches the
        // finite input whose running mean overflows (`{-1.7e308, 1.7e308}`), which used to
        // reach the clip below as `-inf` and come back as a confident `0.0`.
        if !mean.value(i).is_finite() {
            b.append_value(f64::NAN);
            continue;
        }
        // `max(0.0)` clips the tiny negative M2 that cancellation can leave behind, but
        // `f64::max` returns the *other* operand when one is NaN — so it also turned a
        // NaN M2 (any NaN or infinity in the input) into a confident `0.0`, which reads as
        // "this column is constant". `mean` and `sum` propagate NaN over the same input, so
        // the two disagreed, and a zero-variance check silently mis-classified the column.
        // Clip only a genuine negative; let a non-finite M2 through as itself.
        let raw = m2.value(i) / (n - 1) as f64;
        let var = if raw < 0.0 { 0.0 } else { raw };
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
        // The exact 128-bit accumulator an integer AVG sums into (`MEAN_INT_ACCUMULATOR`).
        // The whole sum is carried exactly to here, so the single division below is the
        // only rounding in the aggregate — which is what makes `AVG` over large integers
        // correct rather than merely close. `scale` is 0 for that accumulator; honouring a
        // non-zero one anyway keeps this correct if a decimal sum state ever reaches it.
        DataType::Decimal128(_, scale) => {
            let sums = sum.as_primitive::<arrow::datatypes::Decimal128Type>();
            let divisor = 10f64.powi(i32::from(*scale));
            for i in 0..counts.len() {
                push_mean(
                    &mut b,
                    sums.is_valid(i).then(|| sums.value(i) as f64 / divisor),
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

    /// `finalize_var` over a one-group state with the given `M2` and count.
    fn finalized(m2: f64, n: i64, stddev: bool) -> Option<f64> {
        let mean: ArrayRef = Arc::new(Float64Array::from(vec![0.0]));
        let m2: ArrayRef = Arc::new(Float64Array::from(vec![m2]));
        let count: ArrayRef = Arc::new(Int64Array::from(vec![n]));
        let out = finalize_var(&mean, &m2, &count, stddev).unwrap();
        let a = out.as_primitive::<Float64Type>();
        a.is_valid(0).then(|| a.value(0))
    }

    /// One group's `(mean, M2, count)` from [`var_state`] over `values`.
    fn one_group_state(values: Vec<f64>) -> (f64, f64, i64) {
        let n = values.len();
        let arr: ArrayRef = Arc::new(Float64Array::from(values));
        let ids: Vec<u32> = vec![0; n];
        let out = var_state(&arr, &ids, 1, AggFunc::Var).expect("f64 is supported");
        (
            out[0].as_primitive::<Float64Type>().value(0),
            out[1].as_primitive::<Float64Type>().value(0),
            out[2].as_primitive::<Int64Type>().value(0),
        )
    }

    /// The case the centered accumulator was introduced for still holds: a large offset with a
    /// unit spread must not cancel to zero.
    #[test]
    fn a_large_offset_with_unit_spread_keeps_its_variance() {
        let base = 1e9;
        let (_, m2, n) = one_group_state(vec![base + 1.0, base + 2.0, base + 3.0]);
        assert_eq!(n, 3);
        assert_eq!(finalized(m2, n, false), Some(1.0));
    }

    /// The one-pass update and [`merge_welford`] use **different** recurrences, and that is a
    /// known, bounded imprecision rather than an oversight.
    ///
    /// `var_state` uses Knuth (`M2 += δ·(x − mean_after)`); the merge uses Youngs–Cramer
    /// (`δ²·na·nb/n`). So adding rows one at a time and folding one-row partials pairwise do
    /// not produce a bit-identical `M2` — they agree to within float reassociation, which is
    /// the same bound `python-control-plane.md` states for every distributed float reduction.
    /// Making them identical means adopting one recurrence for both, and the accuracy table on
    /// `var_state` is why that is not free.
    ///
    /// This test pins the *bound*, not equality, so it fails if the two ever drift beyond it.
    #[test]
    fn the_one_pass_update_and_the_merge_agree_to_within_reassociation() {
        let xs = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let (_, m2_one_pass, n) = one_group_state(xs.clone());

        let mut mean: ArrayRef = Arc::new(Float64Array::from(vec![xs[0]]));
        let mut m2: ArrayRef = Arc::new(Float64Array::from(vec![0.0]));
        let mut count: ArrayRef = Arc::new(Int64Array::from(vec![1i64]));
        for &x in &xs[1..] {
            let merged = merge_welford(
                &concat_f64(&mean, x),
                &concat_f64(&m2, 0.0),
                &concat_i64(&count, 1),
                &[0, 0],
                1,
            );
            mean = Arc::clone(&merged[0]);
            m2 = Arc::clone(&merged[1]);
            count = Arc::clone(&merged[2]);
        }
        assert_eq!(count.as_primitive::<Int64Type>().value(0), n);
        let merged_m2 = m2.as_primitive::<Float64Type>().value(0);
        let rel = (merged_m2 - m2_one_pass).abs() / m2_one_pass.abs();
        assert!(
            rel < 1e-12,
            "one-pass M2 {m2_one_pass} vs merged {merged_m2} (relative {rel:e})"
        );
    }

    fn concat_f64(a: &ArrayRef, extra: f64) -> ArrayRef {
        let a = a.as_primitive::<Float64Type>();
        Arc::new(Float64Array::from(vec![a.value(0), extra]))
    }

    fn concat_i64(a: &ArrayRef, extra: i64) -> ArrayRef {
        let a = a.as_primitive::<Int64Type>();
        Arc::new(Int64Array::from(vec![a.value(0), extra]))
    }

    /// One group's variance from `var_state` + `finalize_var`, end to end.
    fn variance_of(values: Vec<f64>) -> Option<f64> {
        let n = values.len();
        let arr: ArrayRef = Arc::new(Float64Array::from(values));
        let state = var_state(&arr, &vec![0u32; n], 1, AggFunc::Var).expect("f64 is supported");
        let out = finalize_var(&state[0], &state[1], &state[2], false).unwrap();
        let a = out.as_primitive::<Float64Type>();
        a.is_valid(0).then(|| a.value(0))
    }

    /// A group containing an infinity gives the same answer whatever order it arrives in.
    ///
    /// This is a mergeability property, not a cosmetic one: partitioning is free to put the
    /// infinity anywhere, so an order-sensitive answer means a distributed result that differs
    /// from the single-node one. `M2` alone *is* order-sensitive here — over these six
    /// permutations the recurrence yields `NaN` for some and `+inf` for others, because the
    /// first value of a group is weighted by zero and `inf * 0.0` is `NaN`. The mean is
    /// non-finite in all six, which is why `finalize_var` keys on it.
    #[test]
    fn an_infinity_gives_the_same_variance_in_every_row_order() {
        let perms = [
            [1.0, f64::INFINITY, 2.0],
            [1.0, 2.0, f64::INFINITY],
            [2.0, 1.0, f64::INFINITY],
            [2.0, f64::INFINITY, 1.0],
            [f64::INFINITY, 1.0, 2.0],
            [f64::INFINITY, 2.0, 1.0],
        ];
        for p in perms {
            let got = variance_of(p.to_vec()).expect("three rows is not null");
            assert!(
                got.is_nan(),
                "var({p:?}) = {got}, expected NaN in every order"
            );
        }
        // Negative infinity is the same argument, and a NaN input must not become a number.
        assert!(variance_of(vec![1.0, f64::NEG_INFINITY, 2.0])
            .unwrap()
            .is_nan());
        assert!(variance_of(vec![1.0, f64::NAN, 2.0]).unwrap().is_nan());
    }

    /// A *finite* input whose running mean overflows must not report zero variance.
    ///
    /// `{-1.7e308, 1.7e308}` is two ordinary finite doubles whose difference is not
    /// representable. The mean overflows, `M2` reached the negative-clip as `-inf`, and the
    /// clip returned `0.0` — "this column is constant", for the most spread-out pair of
    /// doubles that exists.
    #[test]
    fn a_finite_input_that_overflows_the_mean_is_not_zero_variance() {
        let got = variance_of(vec![-1.7e308, 1.7e308]).expect("two rows is not null");
        assert!(got.is_nan(), "expected NaN, got {got}");
        assert_ne!(got, 0.0);
    }

    /// A NaN `M2` must stay NaN rather than becoming a confident zero.
    ///
    /// `f64::max` returns the *non*-NaN operand, so the `max(0.0)` that clips a tiny
    /// negative M2 also reported `var = 0` — "this column is constant" — for any input
    /// containing a NaN or an infinity, while `mean` and `sum` over the same input
    /// propagated NaN. A zero-variance test then silently mis-classified the column.
    #[test]
    fn a_non_finite_second_moment_does_not_become_zero_variance() {
        assert!(finalized(f64::NAN, 3, false).expect("not null").is_nan());
        assert!(finalized(f64::NAN, 3, true).expect("not null").is_nan());
        assert_eq!(finalized(f64::INFINITY, 3, false), Some(f64::INFINITY));
    }

    /// ...while the negative-M2 clip it was written for still works.
    #[test]
    fn a_tiny_negative_second_moment_is_still_clipped_to_zero() {
        assert_eq!(finalized(-1e-18, 5, false), Some(0.0));
        assert_eq!(finalized(-1e-18, 5, true), Some(0.0));
    }

    /// The ordinary path is unchanged: `var = M2 / (n − 1)`, null below two values.
    #[test]
    fn the_ordinary_variance_is_unchanged() {
        assert_eq!(finalized(8.0, 5, false), Some(2.0));
        assert_eq!(finalized(8.0, 5, true), Some(2.0f64.sqrt()));
        assert_eq!(finalized(0.0, 1, false), None);
    }
}
