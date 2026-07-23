//! Hash aggregation — built mergeable so the SAME code runs single-node and
//! distributed.
//!
//! Every aggregate is expressed as three composable steps:
//!
//! * **partial**  — group partition-local rows and emit *partial state* columns
//!   (e.g. `mean` emits `sum` and `count`, not the average).
//! * **combine**  — regroup partial states by key and merge them with an
//!   associative reducer. `combine(partial(A), partial(B)) == partial(A ∪ B)`.
//! * **finalize** — turn merged state into the output value (e.g. `sum / count`).
//!
//! Single-node execution is `finalize(partial(all_rows))`. Distributed execution
//! is `finalize(combine(partial(p) for each partition p))` after a shuffle by key.
//! Because the only difference is whether `combine` runs across partitions, an
//! operator that passes the distributive-equivalence test works both ways.
//!
//! Keys are encoded with arrow's row format (any key type, no per-type code);
//! per-group reductions reuse arrow's typed kernels (correctness-first).

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int64Array};
use arrow::datatypes::DataType;
use rayon::prelude::*;

use crate::error::RuntimeError;

mod accum;
mod argextreme;
mod distinct;
mod fused;
mod group;
mod hll;
mod median;
mod qsketch;
pub mod spill;
mod stats;
mod var;

use accum::{bitfold_acc, bool_acc, concat_col, minmax_acc, product_acc, require, sum_acc};
use argextreme::{arg_extreme_state, merge_arg_extreme};
use distinct::{bucket_values_into_list, distinct_state, finalize_count_distinct, merge_distinct};
pub use distinct::{distinct_batch, distinct_dense};
pub(crate) use group::assign_groups;
use hll::{approx_distinct_state, finalize_approx_distinct, merge_approx_distinct};
use median::{
    finalize_histogram, finalize_list_agg, finalize_median, finalize_mode, finalize_quantile,
    listagg_state, median_state, merge_median,
};
use qsketch::{approx_quantile_state, finalize_approx_quantile, merge_approx_quantile};
use stats::{
    covar_state, finalize_corr, finalize_covar, finalize_kurtosis, finalize_skewness, merge_covar,
    merge_moments, moment_state,
};
use var::{count_non_null, finalize_mean, finalize_var, merge_welford, var_state};

/// An aggregate function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunc {
    CountStar,
    Count,
    /// COUNT(DISTINCT x). Exact and mergeable: the partial state is the set of
    /// distinct (non-null) values per group, held as one `List` column; combining
    /// unions the sets, finalizing counts them.
    CountDistinct,
    Sum,
    Min,
    Max,
    Mean,
    /// Sample variance (Bessel-corrected). State: (sum, sum_of_squares, count).
    Var,
    /// Sample standard deviation = sqrt(Var).
    Stddev,
    /// Median. Exact and mergeable: the partial state is each group's non-null
    /// values as one `List` column; combining concatenates the lists, finalizing
    /// sorts each list and takes the middle (averaging the two middle for an even
    /// count, matching DuckDB).
    Median,
    /// Continuous quantile (`percentile_cont`) at permille `p` (e.g. 250 = 0.25).
    /// Same list-state machinery as `Median` (which is the p=500 case); finalizing
    /// sorts and linearly interpolates at position `p/1000 · (n-1)`.
    Quantile(u16),
    /// `array_agg` — collect each group's non-null values into a `List` (in
    /// arrival order). Same list-state as `Median`; finalize returns the list.
    ListAgg,
    /// `bool_and` — logical AND of a group's non-null boolean values (null if the
    /// group has none). Mergeable: AND associates/commutes, so the partial boolean
    /// state re-folds identically.
    BoolAnd,
    /// `bool_or` — logical OR of a group's non-null boolean values (null if none).
    BoolOr,
    /// `approx_count_distinct` — bounded-memory distinct count via a per-group HLL
    /// sketch (mergeable; ~2% error). The skew-safe alternative to `CountDistinct`,
    /// whose exact per-group value list can OOM on a hot key.
    ApproxCountDistinct,
    /// `approx_quantile` at permille `p` (e.g. 500 = median) via a per-group KLL
    /// sketch (mergeable, bounded memory). The skew-safe alternative to `Median`/
    /// `Quantile`, whose exact per-group value list can OOM on a hot key.
    ApproxQuantile(u16),
    /// `mode` — the most frequent value per group (same list state as `Median`).
    /// Ties broken by the smallest value, so it is deterministic / mergeable.
    Mode,
    /// `arg_min` — the value at the row with the minimum ordering key (two-input;
    /// 2-column state). Key ties break to the smallest value (mergeable).
    ArgMin,
    /// `arg_max` — the value at the row with the maximum ordering key.
    ArgMax,
    /// `product` — product of a group's non-null values as Float64 (DuckDB
    /// `product`). Mergeable: multiplication associates/commutes, and f64 avoids
    /// the integer overflow a wrapping i64 product would hit.
    Product,
    /// `bit_and` — bitwise AND of a group's non-null Int64 values (mergeable).
    BitAnd,
    /// `bit_or` — bitwise OR of a group's non-null Int64 values (mergeable).
    BitOr,
    /// `bit_xor` — bitwise XOR of a group's non-null Int64 values (mergeable).
    BitXor,
    /// `covar_pop`/`covar_samp` — population/sample covariance of two inputs.
    /// Two-input, 6-column sum-of-powers state, mergeable by summing.
    CovarPop,
    CovarSamp,
    /// `corr` — Pearson correlation of two inputs (same 6-column state as covar).
    Corr,
    /// `skewness`/`kurtosis` — sample skewness / excess kurtosis of one input.
    /// Single-input, 5-column moment state, mergeable by summing.
    Skewness,
    Kurtosis,
    /// `histogram` — a `Map<value, count>` of each group's values (DuckDB
    /// `histogram`). Same per-group value-list state as `Median`; finalize counts.
    Histogram,
}

impl AggFunc {
    /// Number of partial-state columns this aggregate carries (1 for most;
    /// `mean` and `arg_min`/`arg_max` are 2; `var`/`stddev` are 3). The spill path
    /// *and* the distributed flatten/unflatten use this to pack/unpack a
    /// [`Partial`]'s state columns — it is the single source of truth for arity.
    pub fn state_arity(self) -> usize {
        match self {
            AggFunc::Mean | AggFunc::ArgMin | AggFunc::ArgMax => 2,
            AggFunc::Var | AggFunc::Stddev => 3,
            AggFunc::Skewness | AggFunc::Kurtosis => 5,
            AggFunc::CovarPop | AggFunc::CovarSamp | AggFunc::Corr => 6,
            _ => 1,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            AggFunc::CountStar => "count_star",
            AggFunc::Count => "count",
            AggFunc::CountDistinct => "count_distinct",
            AggFunc::Sum => "sum",
            AggFunc::Min => "min",
            AggFunc::Max => "max",
            AggFunc::Mean => "mean",
            AggFunc::Var => "var",
            AggFunc::Stddev => "stddev",
            AggFunc::Median => "median",
            AggFunc::Quantile(_) => "quantile",
            AggFunc::ListAgg => "list_agg",
            AggFunc::BoolAnd => "bool_and",
            AggFunc::BoolOr => "bool_or",
            AggFunc::ApproxCountDistinct => "approx_count_distinct",
            AggFunc::ApproxQuantile(_) => "approx_quantile",
            AggFunc::Mode => "mode",
            AggFunc::ArgMin => "arg_min",
            AggFunc::ArgMax => "arg_max",
            AggFunc::Product => "product",
            AggFunc::BitAnd => "bit_and",
            AggFunc::BitOr => "bit_or",
            AggFunc::BitXor => "bit_xor",
            AggFunc::CovarPop => "covar_pop",
            AggFunc::CovarSamp => "covar_samp",
            AggFunc::Corr => "corr",
            AggFunc::Skewness => "skewness",
            AggFunc::Kurtosis => "kurtosis",
            AggFunc::Histogram => "histogram",
        }
    }
}

/// One aggregate to compute: a function and its (optional) pre-evaluated input.
pub struct AggCall {
    pub func: AggFunc,
    pub values: Option<ArrayRef>,
    /// Second input — the ordering key for `arg_min`/`arg_max`. `None` for all
    /// single-input aggregates.
    pub key: Option<ArrayRef>,
}

impl AggCall {
    /// A single-input aggregate call (no ordering key).
    pub fn new(func: AggFunc, values: Option<ArrayRef>) -> Self {
        Self {
            func,
            values,
            key: None,
        }
    }

    /// A two-input aggregate call (`arg_min`/`arg_max`): value + ordering key.
    pub fn with_key(func: AggFunc, values: Option<ArrayRef>, key: Option<ArrayRef>) -> Self {
        Self { func, values, key }
    }
}

/// Partition-local partial aggregation result: the distinct group-key columns,
/// and per-aggregate *state* columns (1 column for most, 2 for `mean`).
pub struct Partial {
    pub group_columns: Vec<ArrayRef>,
    pub states: Vec<Vec<ArrayRef>>,
}

/// Final aggregation result: group-key columns followed by one column per aggregate.
pub struct GroupAggResult {
    pub group_columns: Vec<ArrayRef>,
    pub agg_columns: Vec<ArrayRef>,
}

/// Single-node convenience: `finalize(partial(...))`.
pub fn group_aggregate(
    group_keys: &[ArrayRef],
    calls: &[AggCall],
    num_rows: usize,
) -> Result<GroupAggResult, RuntimeError> {
    let funcs: Vec<AggFunc> = calls.iter().map(|c| c.func).collect();
    let partial = partial(group_keys, calls, num_rows)?;
    let agg_columns = finalize(&funcs, &partial)?;
    Ok(GroupAggResult {
        group_columns: partial.group_columns,
        agg_columns,
    })
}

/// Step 1: partition-local partial aggregation.
///
/// Single-pass and vectorized: group keys are hashed to dense group ids once,
/// then each aggregate scatters its values into per-group accumulators in one
/// linear scan. This is the hot path on large inputs (no per-group `take`).
pub fn partial(
    group_keys: &[ArrayRef],
    calls: &[AggCall],
    num_rows: usize,
) -> Result<Partial, RuntimeError> {
    // Widen a `Mean`'s Int64 input to Float64 once, up front, so both the global and
    // grouped paths (and every fused/combine/distributed step downstream) carry an f64
    // sum state uniformly. AVG is a float result, and the exact overflow-checked i64 SUM
    // accumulator errors on a large-magnitude integer column (e.g. ClickBench `UserID`);
    // an f64 accumulator can't overflow — matching DuckDB, which sums into a HUGEINT —
    // at ~2^-52 relative rounding, far inside the differential tolerance. `SUM` itself is
    // untouched (it keeps the exact i64 accumulator that errors rather than wrap).
    // Decode any dictionary-encoded value/ordering inputs to their plain value type before
    // the typed accumulator kernels (which downcast to a concrete array) run. Group *keys*
    // need no such step — `assign_groups` routes a dictionary key through arrow's
    // `RowConverter`, which encodes it natively. Identity (no realloc) when no input is a
    // dictionary, so the common case pays only one `data_type()` check per call.
    let decoded = decode_dict_call_inputs(calls)?;
    let calls = decoded.as_deref().unwrap_or(calls);
    let denulled = coerce_null_call_inputs(calls)?;
    let calls = denulled.as_deref().unwrap_or(calls);
    let widened = widen_mean_inputs(calls)?;
    let calls = widened.as_deref().unwrap_or(calls);
    // Global aggregate (no GROUP BY): every row is one group, so each aggregate's partial
    // state is the whole-column reduction — computable with arrow's SIMD kernels and, for
    // those, WITHOUT the `vec![0u32; num_rows]` group-id buffer the grouped path allocates
    // (and zeroes) per morsel but never reads on the single-group fast paths. That
    // allocation dominated a global `SUM`/`MIN`/`MAX`/`COUNT` (measured ~2 ms at 6 M rows).
    if group_keys.is_empty() {
        return accum::global_partial(calls, num_rows);
    }

    let (group_ids, num_groups, group_columns) = assign_groups(group_keys, num_rows)?;

    // Fused fast path: the simple scalar aggregates (sum/count/min/max/mean) read
    // `group_ids` *once* in a single fused scan instead of once per aggregate. It
    // fills `slots[idx]` for the calls it fused (positions match `calls`) and leaves
    // the rest `None` for the per-call loop below — so the result is identical, just
    // with fewer passes over `group_ids`. A no-op when <2 calls are fusable.
    let mut slots: Vec<Option<Vec<ArrayRef>>> = vec![None; calls.len()];
    fused::run_fused(calls, &group_ids, num_groups, &mut slots)?;

    let mut states = Vec::with_capacity(calls.len());
    for (idx, call) in calls.iter().enumerate() {
        let state = match slots[idx].take() {
            Some(fused_state) => fused_state, // already computed in the fused scan
            None => accum::accumulate_call(call, &group_ids, num_groups)?,
        };
        states.push(state);
    }
    Ok(Partial {
        group_columns,
        states,
    })
}

/// Decode any dictionary-encoded value/ordering-key input to its plain value type,
/// returning a fresh call list only when some input was a dictionary (else `None`, so the
/// common non-dictionary path allocates nothing). Keeps the typed accumulator kernels
/// oblivious to dictionary encoding, mirroring the scalar `decode_dict` at the `Col` leaf.
fn decode_dict_call_inputs(calls: &[AggCall]) -> Result<Option<Vec<AggCall>>, RuntimeError> {
    let is_dict = |a: &Option<ArrayRef>| {
        matches!(
            a.as_ref().map(|x| x.data_type()),
            Some(DataType::Dictionary(..))
        )
    };
    if !calls.iter().any(|c| is_dict(&c.values) || is_dict(&c.key)) {
        return Ok(None);
    }
    let decode = |a: &Option<ArrayRef>| -> Result<Option<ArrayRef>, RuntimeError> {
        match a {
            Some(arr) => match arr.data_type() {
                DataType::Dictionary(_, v) => Ok(Some(arrow::compute::cast(arr, v)?)),
                _ => Ok(Some(arr.clone())),
            },
            None => Ok(None),
        }
    };
    let out = calls
        .iter()
        .map(|c| {
            Ok(AggCall {
                func: c.func,
                values: decode(&c.values)?,
                key: decode(&c.key)?,
            })
        })
        .collect::<Result<Vec<_>, RuntimeError>>()?;
    Ok(Some(out))
}

/// Widen every `Mean` call's Int64 **or Decimal128/Decimal256** input to Float64,
/// returning a fresh call list only when some widening happened (else `None`, so the
/// common no-`AVG(int)` path allocates nothing). See the note in [`partial`] for why AVG
/// sums in f64 while SUM stays i64.
///
/// Decimal is widened for the same reason as integers, and additionally because the
/// downstream `sum_acc`/`finalize_mean` kernels only understand Int64/Float64 sum state —
/// without this, `avg`/`mean` over a `Decimal128` column raised "unsupported". DuckDB
/// returns a DOUBLE average, which Float64 matches.
fn widen_mean_inputs(calls: &[AggCall]) -> Result<Option<Vec<AggCall>>, RuntimeError> {
    let needs_widen = |c: &AggCall| {
        c.func == AggFunc::Mean
            && c.values.as_ref().is_some_and(|v| {
                matches!(
                    v.data_type(),
                    DataType::Int64 | DataType::Decimal128(_, _) | DataType::Decimal256(_, _)
                )
            })
    };
    if !calls.iter().any(needs_widen) {
        return Ok(None);
    }
    let mut out = Vec::with_capacity(calls.len());
    for c in calls {
        let values = if needs_widen(c) {
            Some(arrow::compute::cast(
                c.values.as_ref().unwrap(),
                &DataType::Float64,
            )?)
        } else {
            c.values.clone()
        };
        out.push(AggCall::with_key(c.func, values, c.key.clone()));
    }
    Ok(Some(out))
}

/// Coerce any `Null`-typed value/ordering input to an all-null `Int64` column, returning a
/// fresh call list only when some coercion happened (else `None`, so the common path allocates
/// nothing). A column that is entirely null carries Arrow's `Null` data type, which the typed
/// accumulator kernels reject ("aggregate `sum` is not supported for column type Null") — so
/// `SUM`/`MIN`/`MAX`/`AVG` over an all-null column *errored* where DuckDB returns NULL, and
/// `COUNT` over it counted every row instead of 0. An all-null `Int64` array flows through
/// every kernel correctly (sum/min/max/mean → NULL; count of non-null → 0), and the exact
/// result type is immaterial since the value is null. Runs before [`widen_mean_inputs`] so a
/// `Null` `AVG` input becomes `Int64` here, then Float64 there.
fn coerce_null_call_inputs(calls: &[AggCall]) -> Result<Option<Vec<AggCall>>, RuntimeError> {
    let is_null =
        |a: &Option<ArrayRef>| matches!(a.as_ref().map(|x| x.data_type()), Some(DataType::Null));
    if !calls.iter().any(|c| is_null(&c.values) || is_null(&c.key)) {
        return Ok(None);
    }
    let coerce = |a: &Option<ArrayRef>| -> Result<Option<ArrayRef>, RuntimeError> {
        match a {
            Some(arr) if matches!(arr.data_type(), DataType::Null) => {
                Ok(Some(arrow::compute::cast(arr, &DataType::Int64)?))
            }
            other => Ok(other.clone()),
        }
    };
    let out = calls
        .iter()
        .map(|c| {
            Ok(AggCall::with_key(
                c.func,
                coerce(&c.values)?,
                coerce(&c.key)?,
            ))
        })
        .collect::<Result<Vec<_>, RuntimeError>>()?;
    Ok(Some(out))
}

/// The row count above which `combine` groups in parallel (hash-radix). Below it the
/// serial path wins — the radix machinery (per-row hash store, bucket bins, parallel
/// dispatch) is pure overhead on a small input.
const RADIX_PARALLEL_THRESHOLD: usize = 200_000;

/// Produce the partial-state columns for one aggregate in a single scan.
fn accumulate(
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
        AggFunc::Median | AggFunc::Quantile(_) | AggFunc::Mode | AggFunc::Histogram => {
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
        AggFunc::BitAnd | AggFunc::BitOr | AggFunc::BitXor => {
            vec![bitfold_acc(
                require(values, func)?,
                group_ids,
                num_groups,
                func,
            )?]
        }
        AggFunc::Skewness | AggFunc::Kurtosis => {
            moment_state(require(values, func)?, group_ids, num_groups, func)?
        }
        // arg_min/arg_max and covar/corr are two-input; `partial` builds their state
        // directly (it has access to the second input), so they never reach the
        // single-input `accumulate`.
        AggFunc::ArgMin | AggFunc::ArgMax => unreachable!("arg_extreme handled in partial"),
        AggFunc::CovarPop | AggFunc::CovarSamp | AggFunc::Corr => {
            unreachable!("covar/corr handled in partial")
        }
    })
}

/// Step 2: merge partial results (across partitions) into one partial result.
/// `combine([p]) ≡ p`; combining is associative for all supported functions. Uses
/// the default radix threshold; the executor calls [`combine_with`] to tune it.
pub fn combine(parts: &[Partial], funcs: &[AggFunc]) -> Result<Partial, RuntimeError> {
    combine_with(parts, funcs, RADIX_PARALLEL_THRESHOLD)
}

/// [`combine`] with a caller-supplied radix-parallel threshold (performance-only —
/// above it the large regroup runs parallel hash-radix, below it serial; the result
/// is identical, group order being unspecified for a hash aggregate either way).
pub fn combine_with(
    parts: &[Partial],
    funcs: &[AggFunc],
    radix_parallel_threshold: usize,
) -> Result<Partial, RuntimeError> {
    assert!(!parts.is_empty(), "combine requires at least one partial");

    // A single partial is already grouped (`combine([p]) ≡ p`), so re-folding it is
    // identity for every associative reducer — skip the concat + re-encode + re-group
    // (the common single-morsel small-query path; the clone is an Arc refcount bump).
    if parts.len() == 1 {
        let p = &parts[0];
        return Ok(Partial {
            group_columns: p.group_columns.clone(),
            states: p.states.clone(),
        });
    }

    let n_keys = parts[0].group_columns.len();
    let total_rows = partial_rows(parts);

    // High-cardinality combine (the distinct / many-group case) regroups a large relation.
    // Above the threshold, hash-radix partitions by key and groups AND merges each partition
    // independently across threads (partitions are key-disjoint, so no cross-partition merge)
    // — parallelizing the otherwise-serial merge scan that dominates a many-group combine.
    // Below it, and for global aggregates (no keys), the serial path wins (the radix machinery
    // is pure overhead on a small/single group), and only that path needs the partials
    // concatenated at all.
    if total_rows > radix_parallel_threshold && n_keys > 0 {
        // One radix partition per core so the independent group-and-merge tasks fill the
        // pool. The old `64` ceiling left a >64-core box merging a high-cardinality combine
        // (DISTINCT / many-group) on at most 64 cores while the extra cores idled — the
        // combine plateaued, then regressed, past ~16 cores. Partitioning is now flat-CSR
        // (see `combine_radix`), so more partitions cost bounded allocation, not a growing-
        // vector storm. The 512 ceiling caps per-partition setup overhead on huge boxes.
        let partitions = rayon::current_num_threads().clamp(2, 512);
        let (group_columns, states) = group::combine_radix(parts, funcs, total_rows, partitions)?;
        return Ok(Partial {
            group_columns,
            states,
        });
    }

    // The serial regroup reads its input as one array per column, so this path — and only
    // this path — concatenates the partials.
    let group_concat: Vec<ArrayRef> = (0..n_keys)
        .into_par_iter()
        .map(|i| concat_col(parts.iter().map(|p| &p.group_columns[i])))
        .collect::<Result<_, _>>()?;
    let state_concats: Vec<Vec<ArrayRef>> = (0..funcs.len())
        .map(|a| {
            (0..parts[0].states[a].len())
                .map(|c| concat_col(parts.iter().map(|p| &p.states[a][c])))
                .collect::<Result<_, _>>()
        })
        .collect::<Result<_, _>>()?;
    let (group_ids, num_groups, merged_group_columns) = assign_groups(&group_concat, total_rows)?;
    let mut states = Vec::with_capacity(funcs.len());
    for (a, &func) in funcs.iter().enumerate() {
        states.push(group::merge_state(
            func,
            &state_concats[a],
            &group_ids,
            num_groups,
        )?);
    }
    Ok(Partial {
        group_columns: merged_group_columns,
        states,
    })
}

/// [`combine`], keeping the hash-radix partitions **separate** instead of concatenating them
/// into one `Partial`.
///
/// The partitions are key-disjoint, so their union is exactly the relation [`combine`]
/// returns — a caller that can emit several morsels (every executor's aggregate tail can)
/// gets the same rows in the same order without paying the concat, which on a
/// high-cardinality string key is the largest single term in the merge. Returns one `Partial`
/// per non-empty partition, in partition order.
///
/// Falls back to a one-element `Vec` — plain [`combine_with`] — whenever the radix regroup
/// would not have run anyway (a small merge, a single partial, or a global aggregate with no
/// group keys), so callers get a uniform shape and never a second semantics.
pub fn combine_partitioned(
    parts: &[Partial],
    funcs: &[AggFunc],
    radix_parallel_threshold: usize,
) -> Result<Vec<Partial>, RuntimeError> {
    assert!(!parts.is_empty(), "combine requires at least one partial");
    let n_keys = parts[0].group_columns.len();
    let total_rows = partial_rows(parts);
    if parts.len() == 1 || n_keys == 0 || total_rows <= radix_parallel_threshold {
        return Ok(vec![combine_with(parts, funcs, radix_parallel_threshold)?]);
    }
    let partitions = rayon::current_num_threads().clamp(2, 512);
    let per = group::combine_radix_parts(parts, funcs, total_rows, partitions)?;
    Ok(per
        .into_iter()
        .filter(|(g, _)| g.first().is_none_or(|c| !c.is_empty()))
        .map(|(group_columns, states)| Partial {
            group_columns,
            states,
        })
        .collect())
}

/// Rows the partials carry between them: the key-column length, or — for a GLOBAL aggregate,
/// which has no key columns — one state row per partial.
fn partial_rows(parts: &[Partial]) -> usize {
    match parts[0].group_columns.first() {
        Some(_) => parts
            .iter()
            .map(|p| p.group_columns[0].len())
            .sum(),
        None => parts
            .iter()
            .map(|p| {
                p.states
                    .first()
                    .and_then(|s| s.first())
                    .map_or(0, |c| c.len())
            })
            .sum(),
    }
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
            AggFunc::Mode => finalize_mode(&state[0])?,
            // arg_min/arg_max: the value is state column 1 (column 0 is the key).
            AggFunc::ArgMin | AggFunc::ArgMax => state[1].clone(),
            AggFunc::CovarPop => finalize_covar(state, false)?,
            AggFunc::CovarSamp => finalize_covar(state, true)?,
            AggFunc::Corr => finalize_corr(state)?,
            AggFunc::Skewness => finalize_skewness(state)?,
            AggFunc::Kurtosis => finalize_kurtosis(state)?,
            AggFunc::Histogram => finalize_histogram(&state[0])?,
            // All other functions' state IS their output.
            _ => state[0].clone(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{AsArray, Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{Float64Type, Int64Type};
    use arrow::row::{RowConverter, SortField};
    use hashbrown::hash_table::Entry;
    use hashbrown::HashTable;

    #[test]
    fn aggregate_over_dictionary_inputs_equals_decoded() {
        use arrow::array::DictionaryArray;
        use arrow::datatypes::Int32Type;
        // A dictionary-encoded VALUE column and a dictionary-encoded group KEY column, plus
        // their decoded forms — group_aggregate must give the same result over either.
        let val_dict: DictionaryArray<Int32Type> = {
            let v: DictionaryArray<Int32Type> = [Some("x"), Some("y"), Some("x"), Some("z")]
                .into_iter()
                .collect();
            v
        };
        let key_dict: DictionaryArray<Int32Type> = [Some("g1"), Some("g1"), Some("g2"), Some("g2")]
            .into_iter()
            .collect();
        let val_arr: ArrayRef = Arc::new(val_dict);
        let key_arr: ArrayRef = Arc::new(key_dict);
        let val_plain = arrow::compute::cast(&val_arr, &DataType::Utf8).unwrap();
        let key_plain = arrow::compute::cast(&key_arr, &DataType::Utf8).unwrap();

        // COUNT(val) grouped by key: count is non-null values per group; MIN(val) too.
        let run = |k: &ArrayRef, v: &ArrayRef| {
            let calls = [
                AggCall::new(AggFunc::Count, Some(v.clone())),
                AggCall::new(AggFunc::Min, Some(v.clone())),
            ];
            let r = group_aggregate(std::slice::from_ref(k), &calls, 4).unwrap();
            // Sort by the (string) group key so the two runs line up regardless of order.
            let gk = arrow::compute::cast(&r.group_columns[0], &DataType::Utf8).unwrap();
            let idx = arrow::compute::sort_to_indices(&gk, None, None).unwrap();
            let sorted: Vec<ArrayRef> = std::iter::once(&gk)
                .chain(r.agg_columns.iter())
                .map(|c| arrow::compute::take(c, &idx, None).unwrap())
                .collect();
            sorted
        };
        let from_dict = run(&key_arr, &val_arr);
        let from_plain = run(&key_plain, &val_plain);
        for (a, b) in from_dict.iter().zip(from_plain.iter()) {
            assert_eq!(a.as_ref(), b.as_ref(), "dict vs decoded aggregate mismatch");
        }
    }

    #[test]
    fn minmax_over_booleans_orders_false_below_true() {
        // SQL orders `false < true`, so `min` is the AND of a group's values and `max` is the
        // OR. Before this arm existed, `min(flag)` raised "not supported for column type
        // Boolean" — while a Parquet footer, which *does* record an exact boolean min/max,
        // happily answered the same query `false` from metadata. The engine and the metadata
        // shortcut disagreed, and the shortcut was the one that looked right.
        use arrow::array::BooleanArray;

        // Two groups: g0 = [true, false, null] → min false, max true;
        //             g1 = [true, true]        → min true,  max true.
        let keys: ArrayRef = Arc::new(Int64Array::from(vec![0, 0, 0, 1, 1]));
        let values: ArrayRef = Arc::new(BooleanArray::from(vec![
            Some(true),
            Some(false),
            None,
            Some(true),
            Some(true),
        ]));
        let calls = [
            AggCall::new(AggFunc::Min, Some(values.clone())),
            AggCall::new(AggFunc::Max, Some(values.clone())),
            // `bool_and`/`bool_or` must give exactly the same answer — they are the same fold.
            AggCall::new(AggFunc::BoolAnd, Some(values.clone())),
            AggCall::new(AggFunc::BoolOr, Some(values)),
        ];
        let out = group_aggregate(std::slice::from_ref(&keys), &calls, 5).unwrap();
        let group = out.group_columns[0].as_primitive::<Int64Type>();
        let col = |i: usize| {
            out.agg_columns[i]
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("boolean min/max output")
                .clone()
        };
        let (min, max, and, or) = (col(0), col(1), col(2), col(3));
        for row in 0..group.len() {
            let (want_min, want_max) = if group.value(row) == 0 {
                (false, true)
            } else {
                (true, true)
            };
            assert_eq!(
                min.value(row),
                want_min,
                "min of group {}",
                group.value(row)
            );
            assert_eq!(
                max.value(row),
                want_max,
                "max of group {}",
                group.value(row)
            );
            assert_eq!(min.value(row), and.value(row), "min must equal bool_and");
            assert_eq!(max.value(row), or.value(row), "max must equal bool_or");
        }
    }

    #[test]
    fn boolean_minmax_is_mergeable_across_partitions() {
        // The invariant every stateful primitive owes: combining the partials of a partition
        // must equal aggregating the whole. A boolean min/max folds with AND/OR, which are
        // associative and commutative, so any partition order merges to the same answer.
        use arrow::array::BooleanArray;

        let whole: ArrayRef = Arc::new(BooleanArray::from(vec![
            Some(true),
            Some(false),
            Some(true),
            Some(true),
        ]));
        let keys: ArrayRef = Arc::new(Int64Array::from(vec![7, 7, 7, 7]));

        let single = group_aggregate(
            std::slice::from_ref(&keys),
            &[
                AggCall::new(AggFunc::Min, Some(whole.clone())),
                AggCall::new(AggFunc::Max, Some(whole)),
            ],
            4,
        )
        .unwrap();

        // The same rows split across two partitions, each partially aggregated, then combined.
        let split = |vals: Vec<Option<bool>>| -> GroupAggResult {
            let rows = vals.len();
            let k: ArrayRef = Arc::new(Int64Array::from(vec![7i64; rows]));
            let v: ArrayRef = Arc::new(BooleanArray::from(vals));
            group_aggregate(
                std::slice::from_ref(&k),
                &[
                    AggCall::new(AggFunc::Min, Some(v.clone())),
                    AggCall::new(AggFunc::Max, Some(v)),
                ],
                rows,
            )
            .unwrap()
        };
        let left = split(vec![Some(true), Some(false)]);
        let right = split(vec![Some(true), Some(true)]);

        // Combining two partials is the same fold over their outputs (min-of-mins, max-of-maxs).
        let merged_keys: ArrayRef = Arc::new(Int64Array::from(vec![7i64, 7]));
        let merge = |a: &ArrayRef, b: &ArrayRef, func: AggFunc| {
            let values: ArrayRef = arrow::compute::concat(&[a.as_ref(), b.as_ref()]).unwrap();
            group_aggregate(
                std::slice::from_ref(&merged_keys),
                &[AggCall::new(func, Some(values))],
                2,
            )
            .unwrap()
            .agg_columns[0]
                .clone()
        };
        let combined_min = merge(&left.agg_columns[0], &right.agg_columns[0], AggFunc::Min);
        let combined_max = merge(&left.agg_columns[1], &right.agg_columns[1], AggFunc::Max);

        assert_eq!(combined_min.as_ref(), single.agg_columns[0].as_ref());
        assert_eq!(combined_max.as_ref(), single.agg_columns[1].as_ref());
    }

    #[test]
    fn mean_over_int64_does_not_overflow() {
        // Two i64::MAX values sum to > i64::MAX (a SUM must error there); AVG sums in f64,
        // so it returns their mean (≈ i64::MAX) instead of overflowing. This is exactly
        // ClickBench's `AVG(UserID)`, which errored before the f64 mean accumulator.
        let values: ArrayRef = Arc::new(Int64Array::from(vec![i64::MAX, i64::MAX]));
        // Global (no group) and grouped both route through `partial`; test the grouped
        // path (single group) so `assign_groups` + fused/per-call scans are exercised.
        let keys: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![7i64, 7]))];
        let calls = [AggCall::new(AggFunc::Mean, Some(values))];
        let p = partial(&keys, &calls, 2).expect("AVG(int64) must not overflow");
        let out = finalize(&[AggFunc::Mean], &p).expect("finalize");
        let got = out[0].as_primitive::<Float64Type>().value(0);
        assert!((got - i64::MAX as f64).abs() < 1.0, "got {got}");
    }

    #[test]
    fn mean_over_decimal128_returns_double() {
        use arrow::array::Decimal128Array;
        // A Decimal128(10,2) column: raw 150/250/350 == 1.50/2.50/3.50 → mean 2.50.
        // Before Decimal was widened, this raised "aggregate mean is not supported for
        // column type Decimal128(..)".
        let dec = Decimal128Array::from(vec![150i128, 250, 350])
            .with_precision_and_scale(10, 2)
            .unwrap();
        let values: ArrayRef = Arc::new(dec);
        let keys: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![1i64, 1, 1]))];
        let calls = [AggCall::new(AggFunc::Mean, Some(values))];
        let p = partial(&keys, &calls, 3).expect("AVG(decimal) must be supported");
        let out = finalize(&[AggFunc::Mean], &p).expect("finalize");
        assert_eq!(out[0].data_type(), &DataType::Float64);
        let got = out[0].as_primitive::<Float64Type>().value(0);
        assert!((got - 2.5).abs() < 1e-9, "got {got}");

        // The keyless global path widens too.
        let dec2 = Decimal128Array::from(vec![150i128, 250, 350])
            .with_precision_and_scale(10, 2)
            .unwrap();
        let v2: ArrayRef = Arc::new(dec2);
        let pg = partial(&[], &[AggCall::new(AggFunc::Mean, Some(v2))], 3).expect("global");
        let og = finalize(&[AggFunc::Mean], &pg).expect("finalize");
        assert!((og[0].as_primitive::<Float64Type>().value(0) - 2.5).abs() < 1e-9);
    }

    #[test]
    fn global_mean_over_int64_does_not_overflow() {
        // The keyless fast path (`global_partial`) must widen too.
        let values: ArrayRef = Arc::new(Int64Array::from(vec![i64::MAX, i64::MAX]));
        let calls = [AggCall::new(AggFunc::Mean, Some(values))];
        let p = partial(&[], &calls, 2).expect("global AVG(int64) must not overflow");
        let out = finalize(&[AggFunc::Mean], &p).expect("finalize");
        let got = out[0].as_primitive::<Float64Type>().value(0);
        assert!((got - i64::MAX as f64).abs() < 1.0, "got {got}");
    }

    fn i64s(v: &[i64]) -> ArrayRef {
        Arc::new(Int64Array::from(v.to_vec()))
    }
    fn strs(v: &[&str]) -> ArrayRef {
        Arc::new(StringArray::from(v.to_vec()))
    }

    /// The native-value fast paths (single string, multi-column) must assign exactly the
    /// same groups — same partition of rows, same group count — as the row-encoder
    /// fallback. Verified by comparing the group *partition* each path induces (the
    /// group ids are dense but their numbering may differ, so compare the equivalence
    /// relation: two rows share a group under one iff they do under the other).
    fn same_partition(a: &[u32], b: &[u32]) {
        assert_eq!(a.len(), b.len(), "row count");
        for i in 0..a.len() {
            for j in 0..a.len() {
                assert_eq!(
                    a[i] == a[j],
                    b[i] == b[j],
                    "rows {i},{j} disagree on co-grouping"
                );
            }
        }
    }

    fn via_row_encoder(keys: &[ArrayRef], n: usize) -> (Vec<u32>, usize) {
        let fields: Vec<SortField> = keys
            .iter()
            .map(|a| SortField::new(a.data_type().clone()))
            .collect();
        let conv = RowConverter::new(fields).unwrap();
        let rows = conv.convert_columns(keys).unwrap();
        let state = ahash::RandomState::with_seeds(0x9E37, 0x79B9, 0x7F4A, 0x7C15);
        let mut table: HashTable<u32> = HashTable::with_capacity(n.max(1));
        let mut reps: Vec<u32> = Vec::new();
        let mut ids = Vec::with_capacity(n);
        for i in 0..n {
            let row_i = rows.row(i);
            let h = state.hash_one(row_i);
            let gid = match table.entry(
                h,
                |&g| rows.row(reps[g as usize] as usize) == row_i,
                |&g| state.hash_one(rows.row(reps[g as usize] as usize)),
            ) {
                Entry::Occupied(e) => *e.get(),
                Entry::Vacant(e) => {
                    let g = reps.len() as u32;
                    reps.push(i as u32);
                    e.insert(g);
                    g
                }
            };
            ids.push(gid);
        }
        (ids, reps.len())
    }

    #[test]
    fn string_fast_path_matches_row_encoder() {
        // A single Utf8 key with a null and repeats — the byte fast path must induce the
        // same group partition as the row-encoder oracle (group ids may be renumbered).
        let keys: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec![
            Some("a"),
            Some("b"),
            Some("a"),
            None,
            Some("b"),
            None,
        ])) as ArrayRef];
        let n = keys[0].len();
        let (fast_ids, fast_n, _) = assign_groups(&keys, n).unwrap();
        let (slow_ids, slow_n) = via_row_encoder(&keys, n);
        assert_eq!(fast_n, slow_n, "group count");
        same_partition(&fast_ids, &slow_ids);
    }

    #[test]
    fn int64_multi_fast_path_matches_row_encoder() {
        // Two Int64 key columns with repeats and a repeated composite pair — the
        // multi-column fast path must induce the same group partition (and reps, so the
        // same group-key columns) as the row-encoder oracle.
        let k0 = i64s(&[1, 2, 1, 3, 2, 1]);
        let k1 = i64s(&[7, 8, 7, 9, 8, 8]); // (1,7) repeats; (1,8) is its own group
        let keys = vec![k0, k1];
        let n = keys[0].len();
        let (fast_ids, fast_n, fast_cols) = assign_groups(&keys, n).unwrap();
        let (slow_ids, slow_n) = via_row_encoder(&keys, n);
        assert_eq!(fast_n, slow_n, "group count");
        same_partition(&fast_ids, &slow_ids);
        // The representative key columns carry the first-seen distinct pairs:
        // (1,7), (2,8), (3,9), (1,8) in first-seen order.
        assert_eq!(fast_cols.len(), 2);
        let c0 = fast_cols[0].as_primitive::<Int64Type>();
        let c1 = fast_cols[1].as_primitive::<Int64Type>();
        let pairs: Vec<(i64, i64)> = (0..fast_n).map(|g| (c0.value(g), c1.value(g))).collect();
        assert_eq!(pairs, vec![(1, 7), (2, 8), (3, 9), (1, 8)]);
    }

    #[test]
    fn multi_raw_mixed_fast_path_matches_row_encoder() {
        // Mixed Int64 + Utf8 composite key (the `GROUP BY <id>, <status>` shape) with
        // repeats — the raw mixed multi-key path must induce the same group partition and
        // first-seen representative columns as the row-encoder oracle.
        let k0 = i64s(&[1, 2, 1, 3, 2, 1]);
        let k1: ArrayRef = Arc::new(StringArray::from(vec!["x", "y", "x", "y", "y", "z"]));
        let keys = vec![k0, k1]; // (1,x) repeats; (1,z) is its own group; (2,y) repeats
        let n = keys[0].len();
        let (fast_ids, fast_n, fast_cols) = assign_groups(&keys, n).unwrap();
        let (slow_ids, slow_n) = via_row_encoder(&keys, n);
        assert_eq!(fast_n, slow_n, "group count");
        same_partition(&fast_ids, &slow_ids);
        // First-seen distinct pairs: (1,x), (2,y), (3,y), (1,z).
        let c0 = fast_cols[0].as_primitive::<Int64Type>();
        let c1 = fast_cols[1].as_string::<i32>();
        let pairs: Vec<(i64, &str)> = (0..fast_n).map(|g| (c0.value(g), c1.value(g))).collect();
        assert_eq!(pairs, vec![(1, "x"), (2, "y"), (3, "y"), (1, "z")]);
    }

    const FUNCS: [AggFunc; 5] = [
        AggFunc::Sum,
        AggFunc::CountStar,
        AggFunc::Mean,
        AggFunc::Min,
        AggFunc::Max,
    ];

    /// Build the standard call set (Sum, Count(*), Mean, Min, Max) over `v`.
    fn calls(v: &ArrayRef) -> Vec<AggCall> {
        vec![
            AggCall::new(AggFunc::Sum, Some(v.clone())),
            AggCall::new(AggFunc::CountStar, None),
            AggCall::new(AggFunc::Mean, Some(v.clone())),
            AggCall::new(AggFunc::Min, Some(v.clone())),
            AggCall::new(AggFunc::Max, Some(v.clone())),
        ]
    }

    /// The core distribution-readiness property: splitting the input into chunks,
    /// running `partial` on each, then `combine`+`finalize`, must equal running
    /// the whole input through `group_aggregate` in one shot.
    #[test]
    fn partial_combine_equals_whole() {
        let keys = strs(&["a", "b", "a", "b", "a", "c"]);
        let vals = i64s(&[1, 2, 3, 4, 5, 6]);

        let whole = group_aggregate(std::slice::from_ref(&keys), &calls(&vals), 6).unwrap();

        // Split into two partitions [0..3] and [3..6] and go through the
        // distributed path (partial per partition → combine → finalize).
        let (k1, v1) = (keys.slice(0, 3), vals.slice(0, 3));
        let (k2, v2) = (keys.slice(3, 3), vals.slice(3, 3));
        let p1 = partial(std::slice::from_ref(&k1), &calls(&v1), 3).unwrap();
        let p2 = partial(std::slice::from_ref(&k2), &calls(&v2), 3).unwrap();

        let merged = combine(&[p1, p2], &FUNCS).unwrap();
        let dist_cols = finalize(&FUNCS, &merged).unwrap();

        // Compare as group->values maps (output order may differ between paths).
        let whole_map = to_map(&whole.group_columns[0], &whole.agg_columns);
        let dist_map = to_map(&merged.group_columns[0], &dist_cols);
        assert_eq!(whole_map, dist_map);
    }

    /// [`combine_partitioned`] must produce the same relation as [`combine`], as the union of
    /// its partitions — that equivalence is the whole licence for emitting them as separate
    /// morsels instead of concatenating them.
    ///
    /// Two things it has to get right, and both are wrong answers rather than slow ones: the
    /// partitions must be **key-disjoint** (a group split across two of them would finalize
    /// twice and the query would return it twice), and every partial row must land in exactly
    /// one (a dropped row is a missing group). The input is deliberately many small partials
    /// over few distinct keys, so most groups genuinely span several partials, and the radix
    /// threshold is forced to `0` so the partitioned path runs on a test-sized input.
    #[test]
    fn combine_partitioned_is_combine_split_by_key() {
        let n = 4_000usize;
        let keys: ArrayRef = Arc::new(StringArray::from(
            (0..n).map(|i| format!("k{}", i % 97)).collect::<Vec<_>>(),
        ));
        let vals: ArrayRef = Arc::new(Int64Array::from(
            (0..n as i64).map(|i| i % 13).collect::<Vec<_>>(),
        ));

        let chunk = 250;
        let partials: Vec<Partial> = (0..n / chunk)
            .map(|c| {
                let (k, v) = (keys.slice(c * chunk, chunk), vals.slice(c * chunk, chunk));
                partial(std::slice::from_ref(&k), &calls(&v), chunk).unwrap()
            })
            .collect();

        let merged = combine(&partials, &FUNCS).unwrap();
        let want = to_map(&merged.group_columns[0], &finalize(&FUNCS, &merged).unwrap());

        let parts = combine_partitioned(&partials, &FUNCS, 0).unwrap();
        assert!(parts.len() > 1, "the partitioned path did not engage");
        let mut got = std::collections::BTreeMap::new();
        for p in &parts {
            for (k, row) in to_map(&p.group_columns[0], &finalize(&FUNCS, p).unwrap()) {
                assert!(
                    got.insert(k.clone(), row).is_none(),
                    "group {k} appeared in two partitions — they are not key-disjoint"
                );
            }
        }
        assert_eq!(got, want);
    }

    #[test]
    fn int_fast_path_groups_with_nulls() {
        // Single Int64 key with duplicates and nulls → the direct-hash fast path.
        let key: ArrayRef = Arc::new(Int64Array::from(vec![
            Some(5),
            None,
            Some(5),
            Some(7),
            None,
            Some(7),
        ]));
        let (ids, ng, cols) = assign_groups(std::slice::from_ref(&key), 6).unwrap();

        // Three groups: 5, null, 7 (first-seen order).
        assert_eq!(ng, 3);
        let g = cols[0].as_any().downcast_ref::<Int64Array>().unwrap();
        // Every row maps back to its own key (null row → the null group's null key).
        let want = [Some(5i64), None, Some(5), Some(7), None, Some(7)];
        for (i, w) in want.iter().enumerate() {
            let gid = ids[i] as usize;
            if g.is_null(gid) {
                assert!(w.is_none());
            } else {
                assert_eq!(Some(g.value(gid)), *w);
            }
        }
        // Rows 0 & 2 share a group; 1 & 4 (null) share; 3 & 5 share.
        assert_eq!(ids[0], ids[2]);
        assert_eq!(ids[1], ids[4]);
        assert_eq!(ids[3], ids[5]);
    }

    /// Sum each group via the parallel radix `combine` (partials → combine with a tiny
    /// threshold that forces the radix path) and via the single-node serial oracle, and
    /// assert the per-group sums are identical. This exercises grouping *and* the
    /// parallel per-partition merge — the whole `combine_radix` path.
    fn assert_radix_combine_sum_matches(keys: &[ArrayRef], values: &ArrayRef, n: usize) {
        use std::collections::BTreeMap;
        let key_rendered = |cols: &[ArrayRef], r: usize| -> String {
            cols.iter()
                .map(|c| {
                    if c.is_null(r) {
                        "∅".to_string()
                    } else if let Some(a) = c.as_any().downcast_ref::<Int64Array>() {
                        a.value(r).to_string()
                    } else {
                        c.as_any()
                            .downcast_ref::<StringArray>()
                            .unwrap()
                            .value(r)
                            .to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join("|")
        };
        let sum_map = |gc: &[ArrayRef], ac: &[ArrayRef]| -> BTreeMap<String, i64> {
            let s = ac[0].as_any().downcast_ref::<Int64Array>().unwrap();
            (0..s.len())
                .map(|r| (key_rendered(gc, r), s.value(r)))
                .collect()
        };

        // Serial oracle: one whole-input group_aggregate.
        let call = AggCall::with_key(AggFunc::Sum, Some(values.clone()), None);
        let oracle = group_aggregate(keys, std::slice::from_ref(&call), n).unwrap();
        let oracle_map = sum_map(&oracle.group_columns, &oracle.agg_columns);

        // Parallel path: split into chunks, partial each, then combine (threshold 1 →
        // radix). Mirrors how the executor combines per-morsel partials.
        let chunks = 11usize;
        let step = n.div_ceil(chunks);
        let partials: Vec<Partial> = (0..n)
            .step_by(step)
            .map(|off| {
                let len = step.min(n - off);
                let ck: Vec<ArrayRef> = keys.iter().map(|k| k.slice(off, len)).collect();
                let cv = values.slice(off, len);
                let call = AggCall::with_key(AggFunc::Sum, Some(cv), None);
                partial(&ck, std::slice::from_ref(&call), len).unwrap()
            })
            .collect();
        let merged = combine_with(&partials, &[AggFunc::Sum], 1).unwrap();
        let agg = finalize(&[AggFunc::Sum], &merged).unwrap();
        let radix_map = sum_map(&merged.group_columns, &agg);

        assert_eq!(
            radix_map, oracle_map,
            "radix combine sums must match the oracle"
        );
    }

    #[test]
    fn radix_combine_matches_serial_on_high_cardinality() {
        // 250k rows over 5000 distinct int keys — crosses the radix threshold.
        let n = 250_000usize;
        let key: ArrayRef = Arc::new(Int64Array::from(
            (0..n).map(|i| (i % 5000) as i64).collect::<Vec<_>>(),
        ));
        let vals: ArrayRef = Arc::new(Int64Array::from(
            (0..n).map(|i| (i % 7) as i64).collect::<Vec<_>>(),
        ));
        assert_radix_combine_sum_matches(std::slice::from_ref(&key), &vals, n);
    }

    #[test]
    fn radix_combine_handles_null_keys_like_serial() {
        // Nulls scattered across the input must all merge into the one null group.
        let n = 200_001usize;
        let key: ArrayRef = Arc::new(Int64Array::from(
            (0..n)
                .map(|i| {
                    if i % 7 == 0 {
                        None
                    } else {
                        Some((i % 9000) as i64)
                    }
                })
                .collect::<Vec<Option<i64>>>(),
        ));
        let vals: ArrayRef = Arc::new(Int64Array::from(
            (0..n).map(|i| (i % 5 + 1) as i64).collect::<Vec<_>>(),
        ));
        assert_radix_combine_sum_matches(std::slice::from_ref(&key), &vals, n);
    }

    #[test]
    fn radix_combine_matches_serial_on_multikey() {
        // Two-column key forces the general RowConverter bucketing path.
        let n = 250_000usize;
        let a: ArrayRef = Arc::new(Int64Array::from(
            (0..n).map(|i| (i % 500) as i64).collect::<Vec<_>>(),
        ));
        let b: ArrayRef = Arc::new(StringArray::from(
            (0..n).map(|i| format!("g{}", i % 400)).collect::<Vec<_>>(),
        ));
        let vals: ArrayRef = Arc::new(Int64Array::from(
            (0..n).map(|i| (i % 9) as i64).collect::<Vec<_>>(),
        ));
        assert_radix_combine_sum_matches(&[a, b], &vals, n);
    }

    /// `bool_and`/`bool_or` must satisfy the same partial→combine→finalize ==
    /// whole-input invariant (AND/OR associate and commute), including null skip.
    #[test]
    fn bool_aggregates_combine_across_partitions() {
        use arrow::array::BooleanArray;
        let keys = strs(&["a", "a", "b", "b", "a", "b"]);
        // group a: T, T, (null) → and=T, or=T; group b: F, T, T → and=F, or=T
        let bools: ArrayRef = Arc::new(BooleanArray::from(vec![
            Some(true),
            Some(true),
            Some(false),
            Some(true),
            None,
            Some(true),
        ]));
        let funcs = [AggFunc::BoolAnd, AggFunc::BoolOr];
        let mk = |v: &ArrayRef| {
            vec![
                AggCall::new(AggFunc::BoolAnd, Some(v.clone())),
                AggCall::new(AggFunc::BoolOr, Some(v.clone())),
            ]
        };
        let whole = group_aggregate(std::slice::from_ref(&keys), &mk(&bools), 6).unwrap();
        let p1 = partial(
            std::slice::from_ref(&keys.slice(0, 3)),
            &mk(&bools.slice(0, 3)),
            3,
        )
        .unwrap();
        let p2 = partial(
            std::slice::from_ref(&keys.slice(3, 3)),
            &mk(&bools.slice(3, 3)),
            3,
        )
        .unwrap();
        let merged = combine(&[p1, p2], &funcs).unwrap();
        let dist = finalize(&funcs, &merged).unwrap();

        let want = whole.agg_columns[0]
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap()
            .clone();
        let got = dist[0]
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap()
            .clone();
        // Compare per group (output order may differ) via the group->bool map.
        let wmap = bool_map(&whole.group_columns[0], &want);
        let gmap = bool_map(&merged.group_columns[0], &got);
        assert_eq!(wmap, gmap);
    }

    fn bool_map(
        keys: &ArrayRef,
        vals: &arrow::array::BooleanArray,
    ) -> std::collections::BTreeMap<String, Option<bool>> {
        let k = keys.as_any().downcast_ref::<StringArray>().unwrap();
        (0..k.len())
            .map(|i| {
                (
                    k.value(i).to_string(),
                    if vals.is_valid(i) {
                        Some(vals.value(i))
                    } else {
                        None
                    },
                )
            })
            .collect()
    }

    /// `approx_count_distinct` must merge across partitions (HLLs union) and land
    /// within HLL error of the exact distinct count — the bounded-memory, skew-safe
    /// distinct path.
    #[test]
    fn approx_count_distinct_combines_within_error() {
        // 3000 rows, all key "a", with 1500 distinct values → split across 3 chunks.
        let n = 3000usize;
        let keys: ArrayRef = Arc::new(StringArray::from(vec!["a"; n]));
        let vals: ArrayRef = Arc::new(Int64Array::from(
            (0..n as i64).map(|i| i % 1500).collect::<Vec<_>>(),
        ));
        let funcs = [AggFunc::ApproxCountDistinct];
        let call = |v: &ArrayRef| vec![AggCall::new(AggFunc::ApproxCountDistinct, Some(v.clone()))];

        let chunk = n / 3;
        let mut partials = Vec::new();
        for c in 0..3 {
            let (k, v) = (keys.slice(c * chunk, chunk), vals.slice(c * chunk, chunk));
            partials.push(partial(std::slice::from_ref(&k), &call(&v), chunk).unwrap());
        }
        let merged = combine(&partials, &funcs).unwrap();
        let out = finalize(&funcs, &merged).unwrap();
        let est = out[0]
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0);
        // Exact distinct is 1500; HLL is within a few percent.
        let err = (est - 1500).abs() as f64 / 1500.0;
        assert!(
            err < 0.05,
            "approx distinct {est} too far from 1500 (err {err})"
        );
    }

    /// `approx_quantile` (DDSketch) must be *bit-identical* across merge topologies:
    /// the whole-input sketch equals the partial→combine of any chunking, because
    /// DDSketch buckets merge by summing counts (order-independent). This is the
    /// single-node==distributed invariant for the approximate quantile path.
    #[test]
    fn approx_quantile_is_merge_order_independent() {
        let n = 4000usize;
        let keys: ArrayRef = Arc::new(StringArray::from(vec!["a"; n]));
        let vals: ArrayRef = Arc::new(Float64Array::from(
            (0..n).map(|i| (i % 200) as f64).collect::<Vec<_>>(),
        ));
        let funcs = [AggFunc::ApproxQuantile(900)];
        let call = |v: &ArrayRef| vec![AggCall::new(AggFunc::ApproxQuantile(900), Some(v.clone()))];
        // Whole-input (one partial).
        let whole = group_aggregate(std::slice::from_ref(&keys), &call(&vals), n).unwrap();
        let whole_v = whole.agg_columns[0]
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0);
        // Split into 5 uneven chunks → partial each → combine → finalize.
        let bounds = [0usize, 137, 900, 2001, 3499, n];
        let partials: Vec<_> = bounds
            .windows(2)
            .map(|w| {
                let (k, v) = (keys.slice(w[0], w[1] - w[0]), vals.slice(w[0], w[1] - w[0]));
                partial(std::slice::from_ref(&k), &call(&v), w[1] - w[0]).unwrap()
            })
            .collect();
        let merged = combine(&partials, &funcs).unwrap();
        let dist_v = finalize(&funcs, &merged).unwrap()[0]
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0);
        assert_eq!(
            whole_v.to_bits(),
            dist_v.to_bits(),
            "approx_quantile not bit-identical across merge topology: {whole_v} vs {dist_v}"
        );
    }

    /// Global aggregation (no group keys) must also merge correctly across
    /// partitions — the path where partial rows are counted from state columns.
    #[test]
    fn global_aggregate_combines_across_partitions() {
        let vals = i64s(&[1, 2, 3, 4, 5, 6]);
        let funcs = [AggFunc::Sum, AggFunc::CountStar, AggFunc::Mean];
        let mk = |v: &ArrayRef| {
            vec![
                AggCall::new(AggFunc::Sum, Some(v.clone())),
                AggCall::new(AggFunc::CountStar, None),
                AggCall::new(AggFunc::Mean, Some(v.clone())),
            ]
        };
        let p1 = partial(&[], &mk(&vals.slice(0, 3)), 3).unwrap();
        let p2 = partial(&[], &mk(&vals.slice(3, 3)), 3).unwrap();
        let merged = combine(&[p1, p2], &funcs).unwrap();
        let cols = finalize(&funcs, &merged).unwrap();
        // sum=21, count=6, mean=3.5 — one output row.
        assert_eq!(
            cols[0]
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0),
            21
        );
        assert_eq!(
            cols[1]
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0),
            6
        );
        assert_eq!(
            cols[2]
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(0),
            3.5
        );
    }

    fn to_map(
        keys: &ArrayRef,
        aggs: &[ArrayRef],
    ) -> std::collections::BTreeMap<String, Vec<String>> {
        let keys = keys.as_any().downcast_ref::<StringArray>().unwrap();
        let mut m = std::collections::BTreeMap::new();
        for i in 0..keys.len() {
            let row: Vec<String> = aggs.iter().map(|a| scalar_str(a, i)).collect();
            m.insert(keys.value(i).to_string(), row);
        }
        m
    }

    fn scalar_str(a: &ArrayRef, i: usize) -> String {
        if let Some(x) = a.as_any().downcast_ref::<Int64Array>() {
            return x.value(i).to_string();
        }
        if let Some(x) = a.as_any().downcast_ref::<Float64Array>() {
            return format!("{:.6}", x.value(i));
        }
        "?".to_string()
    }

    fn count_map(keys: &ArrayRef, counts: &ArrayRef) -> std::collections::BTreeMap<String, i64> {
        let keys = keys.as_any().downcast_ref::<StringArray>().unwrap();
        let counts = counts.as_any().downcast_ref::<Int64Array>().unwrap();
        let mut m = std::collections::BTreeMap::new();
        for i in 0..keys.len() {
            m.insert(keys.value(i).to_string(), counts.value(i));
        }
        m
    }

    fn cd_calls(v: &ArrayRef) -> Vec<AggCall> {
        vec![AggCall::new(AggFunc::CountDistinct, Some(v.clone()))]
    }

    #[test]
    fn count_distinct_exact_and_mergeable() {
        // groups a,b,a,b,a,c with values 1,2,1,4,5,6
        // distinct: a->{1,5}=2, b->{2,4}=2, c->{6}=1
        let keys = strs(&["a", "b", "a", "b", "a", "c"]);
        let vals = i64s(&[1, 2, 1, 4, 5, 6]);

        let whole = group_aggregate(std::slice::from_ref(&keys), &cd_calls(&vals), 6).unwrap();
        let m = count_map(&whole.group_columns[0], &whole.agg_columns[0]);
        assert_eq!(m["a"], 2);
        assert_eq!(m["b"], 2);
        assert_eq!(m["c"], 1);

        // Distributed path: split into two partitions and merge.
        let (k1, v1) = (keys.slice(0, 3), vals.slice(0, 3));
        let (k2, v2) = (keys.slice(3, 3), vals.slice(3, 3));
        let p1 = partial(std::slice::from_ref(&k1), &cd_calls(&v1), 3).unwrap();
        let p2 = partial(std::slice::from_ref(&k2), &cd_calls(&v2), 3).unwrap();
        let merged = combine(&[p1, p2], &[AggFunc::CountDistinct]).unwrap();
        let cols = finalize(&[AggFunc::CountDistinct], &merged).unwrap();
        let dm = count_map(&merged.group_columns[0], &cols[0]);
        assert_eq!(dm["a"], 2);
        assert_eq!(dm["b"], 2);
        assert_eq!(dm["c"], 1);
    }

    #[test]
    fn count_distinct_excludes_nulls() {
        let keys = strs(&["a", "a", "a", "b"]);
        let vals: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None, Some(1), None]));
        let whole = group_aggregate(std::slice::from_ref(&keys), &cd_calls(&vals), 4).unwrap();
        let m = count_map(&whole.group_columns[0], &whole.agg_columns[0]);
        assert_eq!(m["a"], 1); // distinct non-null {1}
        assert_eq!(m["b"], 0); // all null → 0
    }

    /// The radix-parallel `combine` must equal the serial one — including float key
    /// identity on a COMPOSITE key.
    ///
    /// Regression: `hash_keys` stated the float-canonicalization policy per encoder, and
    /// its `RowConverter` fallback omitted it. Arrow's row format is deliberately
    /// non-canonical for floats, so a group whose representative is `-0.0` in one partial
    /// and `0.0` in another — legal, since `assign_groups` takes reps from the original
    /// column — hashed into different radix buckets. Buckets merge by plain `concat` on
    /// the "key-disjoint" assumption, so the two halves were never reconciled and the
    /// query returned two groups where the oracle returns one. Only the fallback was
    /// affected (a composite key mixing a float with a non-`is_hashable_mixed` type, here
    /// `Date32`), and only above `RADIX_PARALLEL_THRESHOLD` — which is why `combine_with`
    /// is called directly with a threshold of 1 rather than building a 200k-row fixture.
    #[test]
    fn radix_combine_preserves_float_key_identity_on_a_composite_key() {
        use arrow::array::Date32Array;

        let part_for = |f: f64| {
            let d: ArrayRef = Arc::new(Date32Array::from(vec![19723]));
            let fl: ArrayRef = Arc::new(Float64Array::from(vec![f]));
            let one: ArrayRef = Arc::new(Int64Array::from(vec![1i64]));
            partial(&[d, fl], &[AggCall::new(AggFunc::Sum, Some(one))], 1).unwrap()
        };
        // Same SQL group (-0.0 == 0.0), different representatives across the two partials.
        let parts = [part_for(-0.0), part_for(0.0)];

        let serial = combine_with(&parts, &[AggFunc::Sum], usize::MAX).unwrap();
        let radix = combine_with(&parts, &[AggFunc::Sum], 1).unwrap();
        assert_eq!(
            serial.group_columns[0].len(),
            1,
            "serial combine must fold -0.0 and 0.0 into one group"
        );
        assert_eq!(
            radix.group_columns[0].len(),
            serial.group_columns[0].len(),
            "radix combine must agree with serial on float key identity"
        );
        // And the counts must have merged, not split.
        let sums = finalize(&[AggFunc::Sum], &radix).unwrap();
        assert_eq!(sums[0].as_primitive::<Int64Type>().value(0), 2);
    }
}
