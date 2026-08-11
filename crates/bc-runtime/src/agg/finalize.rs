//! The finalize dispatch — turning each aggregate's merged partial into its output column.
//!
//! Split from `agg/mod.rs` on a real seam rather than for size alone: that module owns the
//! `AggFunc` vocabulary and the partial/combine plumbing every aggregate shares, while this
//! owns the one-arm-per-function step where they stop being alike. It was extracted when the
//! file hit its ceiling at 796 of 800 lines, which meant the next aggregate anyone added —
//! whichever it was — could not fit.

use arrow::array::ArrayRef;

use super::*;

/// Step 3: turn merged state into output columns.
pub fn finalize(funcs: &[AggFunc], p: &Partial) -> Result<Vec<ArrayRef>, RuntimeError> {
    let mut out = Vec::with_capacity(funcs.len());
    for (a, &func) in funcs.iter().enumerate() {
        let state = &p.states[a];
        out.push(match func {
            AggFunc::Mean => finalize_mean(&state[0], &state[1])?,
            AggFunc::Var => finalize_var(&state[0], &state[1], &state[2], false)?,
            AggFunc::Stddev => finalize_var(&state[0], &state[1], &state[2], true)?,
            // The distinct-set state's per-group list length IS the distinct count.
            AggFunc::CountDistinct => finalize_count_distinct(&state[0]),
            AggFunc::Median => finalize_median(&state[0])?,
            AggFunc::Quantile(permille) => finalize_quantile(&state[0], permille as f64 / 1000.0)?,
            // array_agg: the collected per-group list IS the result, except a non-null
            // *empty* list (an aggregate over zero rows) becomes NULL to match DuckDB.
            AggFunc::ListAgg => finalize_list_agg(&state[0])?,
            AggFunc::ApproxCountDistinct => finalize_approx_distinct(&state[0]),
            AggFunc::ApproxQuantile(permille) => {
                finalize_approx_quantile(&state[0], permille as f64 / 1000.0)
            }
            AggFunc::Mode => finalize_mode(&state[0])?,
            AggFunc::NLength(p) => {
                median::finalize_contiguity(&state[0], median::Contiguity::NLength(p))?
            }
            AggFunc::LCount(p) => {
                median::finalize_contiguity(&state[0], median::Contiguity::LCount(p))?
            }
            AggFunc::AuN => median::finalize_contiguity(&state[0], median::Contiguity::AuN)?,
            // arg_min/arg_max: the value is state column 1 (column 0 is the key).
            AggFunc::ArgMin | AggFunc::ArgMax => state[1].clone(),
            AggFunc::CovarPop => finalize_covar(state, false)?,
            AggFunc::CovarSamp => finalize_covar(state, true)?,
            AggFunc::Corr => finalize_corr(state)?,
            AggFunc::Skewness => finalize_skewness(state)?,
            AggFunc::Kurtosis => finalize_kurtosis(state)?,
            AggFunc::Histogram => finalize_histogram(&state[0])?,
            AggFunc::Entropy => finalize_entropy(&state[0])?,
            AggFunc::Mad => finalize_mad(&state[0])?,
            AggFunc::QuantileDisc(permille) => {
                finalize_quantile_disc(&state[0], permille as f64 / 1000.0)?
            }
            AggFunc::ApproxTopK(k) => finalize_top_k(&state[0], k as usize)?,
            AggFunc::KurtosisPop => finalize_kurtosis_pop(state)?,
            // The compensation is added back exactly once, at the end.
            AggFunc::KahanSum => finalize_kahan(&state[0], &state[1])?.remove(0),
            // All other functions' state IS their output.
            _ => state[0].clone(),
        });
    }
    Ok(out)
}
