//! The per-function dispatch: the two steps where the aggregates stop being alike.
//!
//! `accumulate` turns a morsel into one aggregate's partial state and `finalize` turns merged
//! state back into an output column. Everything between them -- the `AggFunc` vocabulary, the
//! group-id assignment, the `combine` plumbing -- is shared machinery that treats every
//! aggregate the same way. These two are the one-arm-per-function tables, so they belong
//! together and away from it.
//!
//! Split from `agg/mod.rs` on that seam rather than for size alone, though size is what forced
//! the question twice: `finalize` moved out when the file hit 796 of its 800 lines, and
//! `accumulate` followed when a new state shape (`agg::counted`) pushed it over again.

use arrow::array::ArrayRef;

use super::*;

/// Produce the partial-state columns for one aggregate in a single scan.
pub(super) fn accumulate(
    func: AggFunc,
    values: Option<&ArrayRef>,
    group_ids: &[u32],
    num_groups: usize,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    Ok(match func {
        AggFunc::CountStar => {
            let mut counts = vec![0i64; num_groups];
            for &g in group_ids {
                counts[g as usize] += 1;
            }
            vec![Arc::new(Int64Array::from(counts))]
        }
        AggFunc::Count => vec![count_non_null(
            require(values, func)?,
            group_ids,
            num_groups,
        )],
        AggFunc::CountDistinct => {
            vec![distinct_state(
                require(values, func)?,
                group_ids,
                num_groups,
            )?]
        }
        AggFunc::Sum => vec![sum_acc(
            require(values, func)?,
            group_ids,
            num_groups,
            func,
        )?],
        AggFunc::Min => vec![minmax_acc(
            require(values, func)?,
            group_ids,
            num_groups,
            true,
            func,
        )?],
        AggFunc::Max => vec![minmax_acc(
            require(values, func)?,
            group_ids,
            num_groups,
            false,
            func,
        )?],
        AggFunc::Mean => {
            let v = require(values, func)?;
            vec![
                sum_acc(v, group_ids, num_groups, func)?,
                count_non_null(v, group_ids, num_groups),
            ]
        }
        // Variance/stddev carry a Welford (mean, M2, count) state, mergeable via Chan's
        // parallel formula (see `merge_welford`) — so they distribute like every other
        // aggregate, but without the sum-of-squares cancellation the naive form suffered.
        AggFunc::Var | AggFunc::Stddev => {
            var_state(require(values, func)?, group_ids, num_groups, func)?
        }
        AggFunc::Median
        | AggFunc::Quantile(_)
        | AggFunc::Histogram
        // The contiguity statistics differ from `Median` only in their finalize.
        | AggFunc::NLength(_)
        | AggFunc::LCount(_)
        | AggFunc::AuN => {
            vec![median_state(require(values, func)?, group_ids, num_groups)?]
        }
        // `array_agg`/`list_agg` KEEPS null elements (SQL semantics), unlike the
        // null-filtering `median_state` the others share.
        AggFunc::ListAgg => {
            vec![listagg_state(
                require(values, func)?,
                group_ids,
                num_groups,
            )?]
        }
        AggFunc::BoolAnd => vec![bool_acc(
            require(values, func)?,
            group_ids,
            num_groups,
            true,
            func,
        )?],
        AggFunc::BoolOr => vec![bool_acc(
            require(values, func)?,
            group_ids,
            num_groups,
            false,
            func,
        )?],
        AggFunc::ApproxCountDistinct => {
            vec![approx_distinct_state(
                require(values, func)?,
                group_ids,
                num_groups,
            )?]
        }
        AggFunc::ApproxQuantile(_) => {
            vec![approx_quantile_state(
                require(values, func)?,
                group_ids,
                num_groups,
            )?]
        }
        AggFunc::Product => vec![product_acc(require(values, func)?, group_ids, num_groups)?],
        AggFunc::KahanSum => kahan_acc(require(values, func)?, group_ids, num_groups)?,
        AggFunc::BitAnd | AggFunc::BitOr | AggFunc::BitXor => {
            vec![bitfold_acc(
                require(values, func)?,
                group_ids,
                num_groups,
                func,
            )?]
        }
        AggFunc::Skewness | AggFunc::Kurtosis | AggFunc::KurtosisPop => {
            moment_state(require(values, func)?, group_ids, num_groups, func)?
        }
        // The value-list family: entropy, the median absolute deviation and the discrete
        // quantile read a group's whole list, so they share `Median`'s state.
        AggFunc::Entropy | AggFunc::Mad | AggFunc::QuantileDisc(_) => {
            vec![median_state(require(values, func)?, group_ids, num_groups)?]
        }
        // `mode`/`top_k` count values instead of keeping them — `agg::counted` says why.
        AggFunc::Mode | AggFunc::ApproxTopK(_) => {
            counted_state(require(values, func)?, group_ids, num_groups)?
        }
        // `any_value` folds with the same min reducer its combine uses, so a partial
        // and a combined partial are the same shape and the fold is order-independent.
        AggFunc::AnyValue => accumulate(AggFunc::Min, values, group_ids, num_groups)?,
        // arg_min/arg_max and covar/corr are two-input; `partial` builds their state
        // directly (it has access to the second input), so they never reach the
        // single-input `accumulate`.
        AggFunc::ArgMin | AggFunc::ArgMax => unreachable!("arg_extreme handled in partial"),
        AggFunc::CovarPop | AggFunc::CovarSamp | AggFunc::Corr => {
            unreachable!("covar/corr handled in partial")
        }
    })
}

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
            AggFunc::Mode => counted::finalize_mode(state)?,
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
            AggFunc::ApproxTopK(k) => counted::finalize_top_k(state, k as usize)?,
            AggFunc::KurtosisPop => finalize_kurtosis_pop(state)?,
            // The compensation is added back exactly once, at the end.
            AggFunc::KahanSum => finalize_kahan(&state[0], &state[1])?.remove(0),
            // All other functions' state IS their output.
            _ => state[0].clone(),
        });
    }
    Ok(out)
}
