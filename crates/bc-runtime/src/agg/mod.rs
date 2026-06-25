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

use arrow::array::{Array, ArrayRef, AsArray, Int64Array, UInt32Array};
use arrow::datatypes::{
    ArrowPrimitiveType, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type,
    UInt64Type, UInt8Type,
};
use arrow::row::{RowConverter, SortField};
use hashbrown::hash_table::Entry;
use hashbrown::HashTable;

use crate::error::RuntimeError;

mod accum;
mod argextreme;
mod distinct;
mod hll;
mod median;
mod qsketch;
mod radix;
pub mod spill;
mod stats;
mod var;

use accum::{bitfold_acc, bool_acc, concat_col, minmax_acc, product_acc, require, sum_acc};
use argextreme::{arg_extreme_state, merge_arg_extreme};
use distinct::{bucket_values_into_list, distinct_state, finalize_count_distinct, merge_distinct};
use hll::{approx_distinct_state, finalize_approx_distinct, merge_approx_distinct};
use median::{
    finalize_histogram, finalize_median, finalize_mode, finalize_quantile, median_state,
    merge_median,
};
use qsketch::{approx_quantile_state, finalize_approx_quantile, merge_approx_quantile};
use radix::assign_groups_radix;
use stats::{
    covar_state, finalize_corr, finalize_covar, finalize_kurtosis, finalize_skewness, moment_state,
};
use var::{count_non_null, finalize_mean, finalize_var, var_state};

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
    let (group_ids, num_groups, group_columns) = assign_groups(group_keys, num_rows)?;
    let mut states = Vec::with_capacity(calls.len());
    for call in calls {
        let state = match call.func {
            // Two-input: arg_min/arg_max need both the value and the ordering key.
            AggFunc::ArgMin | AggFunc::ArgMax => arg_extreme_state(
                require(call.values.as_ref(), call.func)?,
                require(call.key.as_ref(), call.func)?,
                &group_ids,
                num_groups,
                matches!(call.func, AggFunc::ArgMax),
            )?,
            // covar/corr are two-input: `values` is x, `key` carries y.
            AggFunc::CovarPop | AggFunc::CovarSamp | AggFunc::Corr => covar_state(
                require(call.values.as_ref(), call.func)?,
                require(call.key.as_ref(), call.func)?,
                &group_ids,
                num_groups,
            )?,
            _ => accumulate(call.func, call.values.as_ref(), &group_ids, num_groups)?,
        };
        states.push(state);
    }
    Ok(Partial {
        group_columns,
        states,
    })
}

/// Assign each row a dense group id, returning the ids, the group count, and the
/// distinct group-key columns (in first-seen order).
pub(crate) fn assign_groups(
    group_keys: &[ArrayRef],
    num_rows: usize,
) -> Result<(Vec<u32>, usize, Vec<ArrayRef>), RuntimeError> {
    if group_keys.is_empty() {
        // Global aggregate: a single group over all rows.
        return Ok((vec![0; num_rows], 1, Vec::new()));
    }
    // Fast path: a single integer key column hashes its native values directly,
    // skipping the RowConverter encoding pass (a per-row allocation + copy) that the
    // general path needs for multi-column / variable-length / float keys. Integers are
    // exact under raw hashing — floats (NaN, ±0.0) and strings keep the RowConverter,
    // which imposes a correct total order. This is the common GROUP BY <int id> case.
    if group_keys.len() == 1 {
        use arrow::datatypes::DataType;
        let arr = &group_keys[0];
        match arr.data_type() {
            DataType::Int8 => return assign_groups_int::<Int8Type>(arr, num_rows),
            DataType::Int16 => return assign_groups_int::<Int16Type>(arr, num_rows),
            DataType::Int32 => return assign_groups_int::<Int32Type>(arr, num_rows),
            DataType::Int64 => return assign_groups_int::<Int64Type>(arr, num_rows),
            DataType::UInt8 => return assign_groups_int::<UInt8Type>(arr, num_rows),
            DataType::UInt16 => return assign_groups_int::<UInt16Type>(arr, num_rows),
            DataType::UInt32 => return assign_groups_int::<UInt32Type>(arr, num_rows),
            DataType::UInt64 => return assign_groups_int::<UInt64Type>(arr, num_rows),
            _ => {}
        }
    }
    let fields: Vec<SortField> = group_keys
        .iter()
        .map(|a| SortField::new(a.data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields)?;
    let rows = converter.convert_columns(group_keys)?;

    // Group via a raw hash table keyed by *row index* — we store only the
    // first-seen row of each group and compare encoded rows directly, avoiding
    // the per-row owned-key allocation an `IndexMap<OwnedRow, _>` would incur.
    // Size for the worst case (all rows distinct): the table holds at most
    // `num_rows` entries, so pre-sizing avoids the rehash cascade a small initial
    // capacity forces on a high-cardinality group-by (the hot per-morsel path).
    let state = ahash::RandomState::with_seeds(0x9E37, 0x79B9, 0x7F4A, 0x7C15);
    let mut table: HashTable<u32> = HashTable::with_capacity(num_rows.max(1));
    let mut reps: Vec<u32> = Vec::new(); // group_id -> first-seen row index
    let mut group_ids = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let row_i = rows.row(i);
        let hash = state.hash_one(row_i);
        let gid = match table.entry(
            hash,
            |&g| rows.row(reps[g as usize] as usize) == row_i,
            |&g| state.hash_one(rows.row(reps[g as usize] as usize)),
        ) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let gid = reps.len() as u32;
                reps.push(i as u32);
                e.insert(gid);
                gid
            }
        };
        group_ids.push(gid);
    }

    let num_groups = reps.len();
    let group_columns = converter.convert_rows(reps.iter().map(|&i| rows.row(i as usize)))?;
    Ok((group_ids, num_groups, group_columns))
}

/// Single-integer-key `assign_groups`: hash the native values directly (no row
/// encoding). Nulls form one group (SQL semantics); the output key column is the
/// representative rows `take`n from the input, so type and the null carry through.
fn assign_groups_int<T>(
    arr: &ArrayRef,
    num_rows: usize,
) -> Result<(Vec<u32>, usize, Vec<ArrayRef>), RuntimeError>
where
    T: ArrowPrimitiveType,
    T::Native: std::hash::Hash + Eq,
{
    let a = arr.as_primitive::<T>();
    let state = ahash::RandomState::with_seeds(0x9E37, 0x79B9, 0x7F4A, 0x7C15);
    let mut table: HashTable<u32> = HashTable::with_capacity(num_rows.max(1));
    let mut reps: Vec<u32> = Vec::new(); // group_id -> first-seen row index
    let mut group_ids = Vec::with_capacity(num_rows);
    let mut null_gid: Option<u32> = None;

    for i in 0..num_rows {
        if a.is_null(i) {
            let gid = *null_gid.get_or_insert_with(|| {
                let g = reps.len() as u32;
                reps.push(i as u32);
                g
            });
            group_ids.push(gid);
            continue;
        }
        let v = a.value(i);
        let hash = state.hash_one(v);
        // The table holds only non-null groups, so a rep is always a valid value.
        let gid = match table.entry(
            hash,
            |&g| a.value(reps[g as usize] as usize) == v,
            |&g| state.hash_one(a.value(reps[g as usize] as usize)),
        ) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let gid = reps.len() as u32;
                reps.push(i as u32);
                e.insert(gid);
                gid
            }
        };
        group_ids.push(gid);
    }

    let num_groups = reps.len();
    let group_columns = vec![arrow::compute::take(arr, &UInt32Array::from(reps), None)?];
    Ok((group_ids, num_groups, group_columns))
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
        // Variance/stddev share the (sum, sum_of_squares, count) state, all
        // mergeable by summing — so they distribute like every other aggregate.
        AggFunc::Var | AggFunc::Stddev => {
            var_state(require(values, func)?, group_ids, num_groups, func)?
        }
        AggFunc::Median
        | AggFunc::Quantile(_)
        | AggFunc::ListAgg
        | AggFunc::Mode
        | AggFunc::Histogram => {
            vec![median_state(require(values, func)?, group_ids, num_groups)?]
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

    // Concatenate the group-key columns and each aggregate's state columns.
    let group_concat: Vec<ArrayRef> = (0..n_keys)
        .map(|i| concat_col(parts.iter().map(|p| &p.group_columns[i])))
        .collect::<Result<_, _>>()?;
    // Number of partial rows to regroup. With group keys it's the key-column
    // length; for a GLOBAL aggregate there are no key columns, so each partial
    // contributes exactly one state row — count those instead.
    let total_rows = match group_concat.first() {
        Some(col) => col.len(),
        None => parts
            .iter()
            .map(|p| {
                p.states
                    .first()
                    .and_then(|s| s.first())
                    .map_or(0, |c| c.len())
            })
            .sum(),
    };

    // High-cardinality combine (the distinct / many-group case) regroups a large
    // concatenation; parallelize it via hash-radix when it's big enough to amortize.
    // The serial path stays for global aggregates (no key columns) and small inputs.
    let (group_ids, num_groups, merged_group_columns) =
        if total_rows > radix_parallel_threshold && !group_concat.is_empty() {
            let partitions = rayon::current_num_threads().clamp(2, 64);
            assign_groups_radix(&group_concat, total_rows, partitions)?
        } else {
            assign_groups(&group_concat, total_rows)?
        };

    let mut states = Vec::with_capacity(funcs.len());
    for (a, &func) in funcs.iter().enumerate() {
        let state_concat: Vec<ArrayRef> = (0..parts[0].states[a].len())
            .map(|c| concat_col(parts.iter().map(|p| &p.states[a][c])))
            .collect::<Result<_, _>>()?;
        states.push(merge_state(func, &state_concat, &group_ids, num_groups)?);
    }
    Ok(Partial {
        group_columns: merged_group_columns,
        states,
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
            // array_agg: the collected per-group list IS the result.
            AggFunc::ListAgg => state[0].clone(),
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

// --- partial / merge state production ----------------------------------------

/// Merge already-partial state columns into one group via the function's
/// associative reducer (single-pass, reusing `accumulate`). Counts/sums merge by
/// summing the partial states; min/max by min/max; mean by summing both the
/// partial sums and the partial counts.
fn merge_state(
    func: AggFunc,
    state: &[ArrayRef],
    group_ids: &[u32],
    num_groups: usize,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    Ok(match func {
        AggFunc::CountStar | AggFunc::Count | AggFunc::Sum => {
            accumulate(AggFunc::Sum, Some(&state[0]), group_ids, num_groups)?
        }
        // Distinct sets merge by unioning the per-group value lists (dedup again).
        AggFunc::CountDistinct => vec![merge_distinct(&state[0], group_ids, num_groups)?],
        AggFunc::Median
        | AggFunc::Quantile(_)
        | AggFunc::ListAgg
        | AggFunc::Mode
        | AggFunc::Histogram => {
            vec![merge_median(&state[0], group_ids, num_groups)?]
        }
        AggFunc::Min => accumulate(AggFunc::Min, Some(&state[0]), group_ids, num_groups)?,
        AggFunc::Max => accumulate(AggFunc::Max, Some(&state[0]), group_ids, num_groups)?,
        // Boolean state re-folds via the same AND/OR reducer (associative).
        AggFunc::BoolAnd | AggFunc::BoolOr => {
            accumulate(func, Some(&state[0]), group_ids, num_groups)?
        }
        // Product / bitwise state re-folds via the same associative op.
        AggFunc::Product | AggFunc::BitAnd | AggFunc::BitOr | AggFunc::BitXor => {
            accumulate(func, Some(&state[0]), group_ids, num_groups)?
        }
        // Per-group HLL sketches union across partitions.
        AggFunc::ApproxCountDistinct => {
            vec![merge_approx_distinct(&state[0], group_ids, num_groups)?]
        }
        // Per-group KLL sketches merge across partitions.
        AggFunc::ApproxQuantile(_) => {
            vec![merge_approx_quantile(&state[0], group_ids, num_groups)?]
        }
        // 2-column (key, value) state: keep the extreme-key pair per group.
        AggFunc::ArgMin | AggFunc::ArgMax => merge_arg_extreme(
            state,
            group_ids,
            num_groups,
            matches!(func, AggFunc::ArgMax),
        )?,
        AggFunc::Mean => vec![
            accumulate(AggFunc::Sum, Some(&state[0]), group_ids, num_groups)?
                .into_iter()
                .next()
                .unwrap(),
            accumulate(AggFunc::Sum, Some(&state[1]), group_ids, num_groups)?
                .into_iter()
                .next()
                .unwrap(),
        ],
        // (sum, sumsq, count) all merge by summing.
        AggFunc::Var | AggFunc::Stddev => (0..3)
            .map(|c| {
                accumulate(AggFunc::Sum, Some(&state[c]), group_ids, num_groups)
                    .map(|mut v| v.swap_remove(0))
            })
            .collect::<Result<Vec<_>, _>>()?,
        // The sum-of-powers state columns (5 for skew/kurt, 6 for covar/corr) all
        // merge by summing — the property that makes these mergeable.
        AggFunc::Skewness | AggFunc::Kurtosis => sum_each_column(state, group_ids, num_groups)?,
        AggFunc::CovarPop | AggFunc::CovarSamp | AggFunc::Corr => {
            sum_each_column(state, group_ids, num_groups)?
        }
    })
}

/// Merge each partial-state column by summing it across partitions (the shared
/// reducer for every sum-of-powers aggregate). Column 0 is an Int64 count; summing
/// it stays Int64, the Float64 moment columns stay Float64.
fn sum_each_column(
    state: &[ArrayRef],
    group_ids: &[u32],
    num_groups: usize,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    (0..state.len())
        .map(|c| {
            accumulate(AggFunc::Sum, Some(&state[c]), group_ids, num_groups)
                .map(|mut v| v.swap_remove(0))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array, StringArray};

    fn i64s(v: &[i64]) -> ArrayRef {
        Arc::new(Int64Array::from(v.to_vec()))
    }
    fn strs(v: &[&str]) -> ArrayRef {
        Arc::new(StringArray::from(v.to_vec()))
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

    #[test]
    fn radix_groups_equal_serial_on_large_input() {
        // 250k rows over 5000 distinct keys — crosses RADIX_PARALLEL_THRESHOLD, so this
        // is the path `combine` takes for a high-cardinality distinct/group-by.
        let n = 250_000usize;
        let vals: Vec<i64> = (0..n).map(|i| (i % 5000) as i64).collect();
        let key: ArrayRef = Arc::new(Int64Array::from(vals.clone()));
        let keys = std::slice::from_ref(&key);

        let (ids_s, ng_s, cols_s) = assign_groups(keys, n).unwrap();
        let (ids_r, ng_r, cols_r) = assign_groups_radix(keys, n, 8).unwrap();
        assert_eq!(ng_s, 5000);
        assert_eq!(ng_r, 5000); // same number of groups as serial

        let key_set = |c: &ArrayRef| {
            let a = c.as_any().downcast_ref::<Int64Array>().unwrap();
            (0..a.len())
                .map(|i| a.value(i))
                .collect::<std::collections::BTreeSet<_>>()
        };
        assert_eq!(key_set(&cols_s[0]), key_set(&cols_r[0])); // same distinct keys

        // Round-trip: each row's assigned group key equals its input key (both paths).
        let check = |ids: &[u32], cols: &ArrayRef| {
            let g = cols.as_any().downcast_ref::<Int64Array>().unwrap();
            for (i, &v) in vals.iter().enumerate() {
                assert_eq!(g.value(ids[i] as usize), v);
            }
        };
        check(&ids_s, &cols_s[0]);
        check(&ids_r, &cols_r[0]);
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
}
