//! Per-batch / per-side operator primitives shared by the sequential reference
//! executor (`crate::execute`) and the parallel executor (`crate::par`).
//!
//! Keeping the actual operator logic here — and having both executors call it —
//! is what guarantees the parallel path computes exactly what the sequential
//! oracle does (asserted by a Rust test and by the differential suite). The
//! executors differ only in *scheduling* (sequential vs rayon + hash-shuffle),
//! never in operator semantics.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanArray, RecordBatch, UInt64Array};
use arrow::compute::SortOptions;
use arrow::compute::{filter_record_batch, lexsort_to_indices, sort_to_indices, SortColumn};
use arrow::datatypes::{Field, Schema};
use bc_ir::{
    AggFunc, AggregateItem, FrameBound, FrameUnits, ProjectionItem, SortKey, WindowFn, WindowFrame,
    WindowFunc,
};
use bc_runtime::agg::{self, AggCall};
use bc_runtime::window::{self, WindowCall};
use bc_runtime::window_frame;

use crate::error::InterpError;

mod external_sort;
mod joins;
mod materialize;
mod mixed_spill;
mod morsel;
mod project_field;
mod quantile_spill;
mod radix_sort;
mod repartition;
mod reshape;
mod sample_sort;
mod str_sort;
pub(crate) use external_sort::{external_merge_sort, external_sort_to_final_store};
pub(crate) use joins::{
    asof_join_batches, columns_by_name, gather_join_output, gather_join_output_with, join_batches,
    join_batches_with, join_output_schema, join_top_n, key_indices, map_join_type,
};
pub(crate) use materialize::materialize;
pub(crate) use mixed_spill::try_bounded_mixed_spill;
pub(crate) use morsel::{morselize_par, remorselize, sliced_batch_bytes};
pub(crate) use quantile_spill::{
    try_bounded_distinct_spill, try_bounded_histogram_spill, try_bounded_mode_spill,
    try_bounded_quantile_spill,
};
pub(crate) use repartition::partition_morsels;
pub(crate) use reshape::{
    add_row_ids, sample_batch, sample_n_batches, unnest_batch, unpivot_batch,
};
pub(crate) use sample_sort::parallel_sort_batch;

// --- filter / project --------------------------------------------------------

/// A compiled expression (JIT fast path), or `None` to use the interpreter.
/// `CompiledExpr` is `Send + Sync`, so this is shared across rayon workers.
pub(crate) type Jit = Option<std::sync::Arc<bc_codegen::CompiledExpr>>;

/// Compile `expr` once for an operator using `sample` as a representative batch.
/// Returns `None` if the expression is outside the JIT's supported subset — the
/// interpreter then handles it. (Compiling once and reusing across morsels is
/// what makes the JIT win; a per-morsel compile would lose to the interpreter.)
///
/// Memoized process-wide on `(expr, schema, simd)` — the compile is a pure function of those,
/// so a query shape pays Cranelift once rather than once per `execute_plan` call. That matters
/// most where `execute_plan` is itself the loop body: the per-batch streaming path and the
/// per-operator UDF path.
pub(crate) fn try_compile(expr: &bc_expr::Expr, sample: &RecordBatch) -> Jit {
    bc_codegen::compile_expr_cached(expr, sample, bc_arrow::SimdOverride::default())
}

/// Evaluate an expression, using the compiled fast path when available and
/// falling back to the interpreter for batches the JIT can't handle (e.g. one
/// that contains nulls in a referenced column).
fn eval_jit(jit: &Jit, expr: &bc_expr::Expr, batch: &RecordBatch) -> Result<ArrayRef, InterpError> {
    if let Some(compiled) = jit {
        if let Ok(arr) = compiled.eval(batch) {
            return Ok(arr);
        }
    }
    Ok(expr.eval(batch)?)
}

pub(crate) fn filter_batch(
    batch: &RecordBatch,
    predicate: &bc_expr::Expr,
) -> Result<RecordBatch, InterpError> {
    filter_batch_jit(batch, predicate, &None)
}

/// Filter using a pre-compiled predicate when possible.
pub(crate) fn filter_batch_jit(
    batch: &RecordBatch,
    predicate: &bc_expr::Expr,
    jit: &Jit,
) -> Result<RecordBatch, InterpError> {
    let mask = eval_jit(jit, predicate, batch)?;
    let mask = mask
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| InterpError::NonBooleanPredicate {
            got: mask.data_type().to_string(),
        })?;
    // NB: short-circuiting an all-true / all-false mask here (Arc-clone / empty slice
    // instead of the gather) was measured and does NOT pay off: at the 16,384-row morsel
    // granularity `filter_record_batch`'s copy is L2-resident, and mask evaluation plus
    // rayon scheduling dominate. It only added a `true_count` pass to every morsel.
    Ok(filter_record_batch(batch, mask)?)
}

pub(crate) fn project_batch(
    batch: &RecordBatch,
    exprs: &[ProjectionItem],
) -> Result<RecordBatch, InterpError> {
    let jits: Vec<Jit> = exprs.iter().map(|_| None).collect();
    project_batch_jit(batch, exprs, &jits)
}

/// Project using pre-compiled expressions (one `Jit` per output column).
pub(crate) fn project_batch_jit(
    batch: &RecordBatch,
    exprs: &[ProjectionItem],
    jits: &[Jit],
) -> Result<RecordBatch, InterpError> {
    let mut fields = Vec::with_capacity(exprs.len());
    let mut columns = Vec::with_capacity(exprs.len());
    for (item, jit) in exprs.iter().zip(jits) {
        let array = eval_jit(jit, &item.expr, batch)?;
        fields.push(project_field::output_field(item, &array, batch));
        columns.push(array);
    }
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

// --- aggregation -------------------------------------------------------------

pub(crate) fn agg_funcs(aggregates: &[AggregateItem]) -> Vec<agg::AggFunc> {
    aggregates.iter().map(map_agg_func).collect()
}

/// Compile `expr` only if it is a *computed* expression worth JIT-ing. A bare
/// `Col` is skipped: the interpreter evaluates it as a zero-copy `Arc` clone, while
/// a compiled column would pay a compile cost and materialize a fresh buffer — a
/// loss. Returns `None` (interpreter) for bare columns and anything outside the
/// JIT subset; `Some` for compiled arithmetic/comparison/etc.
pub(crate) fn try_compile_computed(expr: &bc_expr::Expr, sample: &RecordBatch) -> Jit {
    match expr {
        bc_expr::Expr::Col { .. } => None,
        _ => try_compile(expr, sample),
    }
}

/// Per-operator compiled expressions for an [`Aggregate`](bc_ir::RelOp::Aggregate):
/// the group keys and each aggregate's value / ordering inputs. Compiled once from a
/// sample batch and reused across every morsel's partial aggregation (the compile
/// cost amortizes exactly as it does for Filter/Project). `CompiledExpr` is
/// `Send + Sync`, so this is shared across rayon workers.
pub(crate) struct AggJit {
    group: Vec<Jit>,
    input: Vec<Jit>,
    input2: Vec<Jit>,
}

/// Compile the group-key and aggregate-input expressions once, using `sample` as a
/// representative batch. Computed expressions (`GROUP BY a + b`, `SUM(price * qty)`)
/// get the JIT fast path; bare columns and unsupported expressions stay on the
/// interpreter (see [`try_compile_computed`]).
pub(crate) fn compile_agg(
    group_keys: &[ProjectionItem],
    aggregates: &[AggregateItem],
    sample: &RecordBatch,
) -> AggJit {
    AggJit {
        group: group_keys
            .iter()
            .map(|k| try_compile_computed(&k.expr, sample))
            .collect(),
        input: aggregates
            .iter()
            .map(|a| {
                a.input
                    .as_ref()
                    .and_then(|e| try_compile_computed(e, sample))
            })
            .collect(),
        input2: aggregates
            .iter()
            .map(|a| {
                a.input2
                    .as_ref()
                    .and_then(|e| try_compile_computed(e, sample))
            })
            .collect(),
    }
}

/// Partition-local partial aggregation of one batch (interpreter only — the
/// sequential oracle and callers without a compiled plan).
pub(crate) fn eval_partial(
    batch: &RecordBatch,
    group_keys: &[ProjectionItem],
    aggregates: &[AggregateItem],
) -> Result<agg::Partial, InterpError> {
    let group_arrays: Vec<ArrayRef> = group_keys
        .iter()
        .map(|k| k.expr.eval(batch))
        .collect::<Result<_, _>>()?;
    let mut calls = Vec::with_capacity(aggregates.len());
    for item in aggregates {
        let values = match &item.input {
            Some(expr) => Some(expr.eval(batch)?),
            None => None,
        };
        // The ordering key for arg_min/arg_max (the aggregate's second input).
        let key = match &item.input2 {
            Some(expr) => Some(expr.eval(batch)?),
            None => None,
        };
        calls.push(AggCall::with_key(map_agg_func(item), values, key));
    }
    Ok(agg::partial(&group_arrays, &calls, batch.num_rows())?)
}

/// Partial aggregation using the per-operator compiled expressions ([`compile_agg`]).
/// Identical result to [`eval_partial`] — the JIT is bit-for-bit with the
/// interpreter and falls back per batch where it can't apply.
pub(crate) fn eval_partial_jit(
    batch: &RecordBatch,
    group_keys: &[ProjectionItem],
    aggregates: &[AggregateItem],
    jit: &AggJit,
) -> Result<agg::Partial, InterpError> {
    let group_arrays: Vec<ArrayRef> = group_keys
        .iter()
        .zip(&jit.group)
        .map(|(k, j)| eval_jit(j, &k.expr, batch))
        .collect::<Result<_, _>>()?;
    let mut calls = Vec::with_capacity(aggregates.len());
    for (i, item) in aggregates.iter().enumerate() {
        let values = match &item.input {
            Some(expr) => Some(eval_jit(&jit.input[i], expr, batch)?),
            None => None,
        };
        let key = match &item.input2 {
            Some(expr) => Some(eval_jit(&jit.input2[i], expr, batch)?),
            None => None,
        };
        calls.push(AggCall::with_key(map_agg_func(item), values, key));
    }
    Ok(agg::partial(&group_arrays, &calls, batch.num_rows())?)
}

/// Assemble the output batch from finalized group + aggregate columns.
pub(crate) fn build_agg_batch(
    group_keys: &[ProjectionItem],
    aggregates: &[AggregateItem],
    group_columns: &[ArrayRef],
    agg_columns: &[ArrayRef],
) -> Result<RecordBatch, InterpError> {
    let mut fields = Vec::with_capacity(group_keys.len() + aggregates.len());
    let mut columns = Vec::with_capacity(group_keys.len() + aggregates.len());
    for (item, col) in group_keys.iter().zip(group_columns) {
        fields.push(Field::new(&item.alias, col.data_type().clone(), true));
        columns.push(col.clone());
    }
    for (item, col) in aggregates.iter().zip(agg_columns) {
        fields.push(Field::new(&item.alias, col.data_type().clone(), true));
        columns.push(col.clone());
    }
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

/// Deduplicate the merged partials of an all-column group-by into distinct rows.
pub(crate) fn distinct_partial(batch: &RecordBatch) -> Result<agg::Partial, InterpError> {
    let keys: Vec<ArrayRef> = batch.columns().to_vec();
    Ok(agg::partial(&keys, &[], batch.num_rows())?)
}

fn map_agg_func(item: &AggregateItem) -> agg::AggFunc {
    match item.func {
        AggFunc::CountStar => agg::AggFunc::CountStar,
        AggFunc::Count => agg::AggFunc::Count,
        AggFunc::CountDistinct => agg::AggFunc::CountDistinct,
        AggFunc::Sum => agg::AggFunc::Sum,
        AggFunc::Min => agg::AggFunc::Min,
        AggFunc::Max => agg::AggFunc::Max,
        AggFunc::Mean => agg::AggFunc::Mean,
        AggFunc::Var => agg::AggFunc::Var,
        AggFunc::Stddev => agg::AggFunc::Stddev,
        AggFunc::Median => agg::AggFunc::Median,
        // Quantile in [0,1] → permille (median is the 0.5 default).
        AggFunc::Quantile => {
            agg::AggFunc::Quantile((item.param.unwrap_or(0.5) * 1000.0).round() as u16)
        }
        AggFunc::ListAgg => agg::AggFunc::ListAgg,
        AggFunc::BoolAnd => agg::AggFunc::BoolAnd,
        AggFunc::BoolOr => agg::AggFunc::BoolOr,
        AggFunc::ApproxCountDistinct => agg::AggFunc::ApproxCountDistinct,
        AggFunc::ApproxQuantile => {
            agg::AggFunc::ApproxQuantile((item.param.unwrap_or(0.5) * 1000.0).round() as u16)
        }
        AggFunc::Mode => agg::AggFunc::Mode,
        AggFunc::ArgMin => agg::AggFunc::ArgMin,
        AggFunc::ArgMax => agg::AggFunc::ArgMax,
        AggFunc::Product => agg::AggFunc::Product,
        AggFunc::BitAnd => agg::AggFunc::BitAnd,
        AggFunc::BitOr => agg::AggFunc::BitOr,
        AggFunc::BitXor => agg::AggFunc::BitXor,
        AggFunc::CovarPop => agg::AggFunc::CovarPop,
        AggFunc::CovarSamp => agg::AggFunc::CovarSamp,
        AggFunc::Corr => agg::AggFunc::Corr,
        AggFunc::Skewness => agg::AggFunc::Skewness,
        AggFunc::Kurtosis => agg::AggFunc::Kurtosis,
        AggFunc::Histogram => agg::AggFunc::Histogram,
    }
}

// --- sort / limit / materialize ---------------------------------------------

/// Sort a single (already-materialized) batch by the given keys.
pub(crate) fn sort_batch(
    batch: &RecordBatch,
    keys: &[SortKey],
    limit: Option<usize>,
) -> Result<RecordBatch, InterpError> {
    if batch.num_rows() == 0 {
        return Ok(batch.clone());
    }
    let indices = sort_indices(batch, keys, limit)?;
    take_batch(batch, &indices)
}

/// The permutation that sorts `batch` by `keys` (the first `limit` rows for a top-N). Shared
/// by the serial [`sort_batch`] and the [`parallel_sort_batch`] gather so both order rows
/// identically — the parallel path only parallelizes the *take*, never the comparison.
fn sort_indices(
    batch: &RecordBatch,
    keys: &[SortKey],
    limit: Option<usize>,
) -> Result<arrow::array::UInt32Array, InterpError> {
    let indices = if limit.is_some() {
        // A `limit` makes this a top-N: arrow returns only the first `limit` indices via a
        // *partial* sort (far cheaper than fully sorting then slicing) — but that partial
        // sort is UNSTABLE, so which tied rows survive and in what order is arbitrary and
        // input-size-dependent. That makes single-node and the distributed range-sort (whose
        // per-bucket reduce runs this same top-N over a differently-sized slice) disagree on
        // ties. Append the original row position as a final ascending tie-break key: ties now
        // resolve to input order, so the top-N is deterministic and identical to a stable
        // sort-then-slice — single-node == every partitioning. One extra unique key over the
        // same O(n log k) partial sort; the radix/parallel fast paths are full-sort only.
        let mut sort_columns = eval_sort_columns(batch, keys)?;
        let row_index = Arc::new(UInt64Array::from_iter_values(0..batch.num_rows() as u64));
        sort_columns.push(SortColumn {
            values: row_index,
            options: Some(SortOptions {
                descending: false,
                nulls_first: false,
            }),
        });
        lexsort_to_indices(&sort_columns, limit)?
    } else {
        let vals: Vec<ArrayRef> = keys
            .iter()
            .map(|k| k.expr.eval(batch))
            .collect::<Result<_, _>>()?;
        return sort_indices_of(&vals, keys);
    };
    Ok(indices)
}

/// The permutation that sorts already-evaluated key columns — the full-sort core of
/// [`sort_indices`], split out so the parallel sample-sort can evaluate each key once over
/// the whole batch and reuse the arrays per range.
pub(crate) fn sort_indices_of(
    vals: &[ArrayRef],
    keys: &[SortKey],
) -> Result<arrow::array::UInt32Array, InterpError> {
    if let ([k], [v]) = (keys, vals) {
        // Single-key *full* sort uses arrow's specialized per-type `sort_to_indices` (a
        // dedicated primitive path) rather than the general multi-column `lexsort`.
        let opts = SortOptions {
            descending: k.descending,
            nulls_first: k.nulls_first,
        };
        // A string key sorts through the stable permutation builder: arrow's
        // `sort_to_indices` leaves ties in an arbitrary, input-size-dependent order, which
        // would make the parallel sample-sort's per-range results disagree with this
        // sequential oracle. Ties resolve to input order instead — deterministic, and the
        // same guarantee the radix path already gives fixed-width keys.
        if let Some(idx) = str_sort::stable_sort_indices_str(v, opts) {
            return Ok(idx);
        }
        // Radix fast path on a fixed-width integer/temporal key: O(n) vs the comparison
        // sort's O(n log n), producing the identical relation. Falls back for other types.
        return Ok(match radix_sort::radix_sort_indices(v, opts) {
            Some(idx) => idx,
            None => sort_to_indices(v, Some(opts), None)?,
        });
    }
    let mut columns: Vec<SortColumn> = vals
        .iter()
        .zip(keys)
        .map(|(values, k)| SortColumn {
            values: values.clone(),
            options: Some(SortOptions {
                descending: k.descending,
                nulls_first: k.nulls_first,
            }),
        })
        .collect();
    // Append an ascending row-index as the final tie-break so `lexsort` (which is unstable in
    // arrow) resolves rows equal on every real key to input order — the stability the single-key
    // radix/string paths already guarantee. Without it, the parallel sample-sort and the external
    // merge sort (each calling this over a differently-sized slice) order fully-tied rows
    // differently from this sequential oracle, breaking seq == par bit-for-bit. The slice this
    // sorts is always gathered in ascending original-row order, so a slice-local `0..n` preserves
    // the input's relative order of tied rows.
    let n = vals.first().map(|v| v.len()).unwrap_or(0);
    let row_index: ArrayRef = Arc::new(arrow::array::UInt32Array::from_iter_values(0..n as u32));
    columns.push(SortColumn {
        values: row_index,
        options: Some(SortOptions {
            descending: false,
            nulls_first: false,
        }),
    });
    Ok(lexsort_to_indices(&columns, None)?)
}

/// Gather `batch`'s rows in `indices` order (a single-threaded take of every column).
fn take_batch(
    batch: &RecordBatch,
    indices: &arrow::array::UInt32Array,
) -> Result<RecordBatch, InterpError> {
    let columns = batch
        .columns()
        .iter()
        .map(|c| bc_runtime::gather::take_column(c.as_ref(), indices))
        .collect::<Result<Vec<ArrayRef>, _>>()?;
    Ok(RecordBatch::try_new(batch.schema(), columns)?)
}

/// Late-materialized parallel top-N over already-morselized `parts`.
///
/// The eager parallel top-N gathers **every column** of each morsel's local top-k before
/// merging (`sort_batch(morsel, keys, Some(k))` per morsel, then a merge). On a wide row that
/// copies `morsels × k` full rows only to discard all but the final `k` — measured the
/// dominant cost of a `SELECT * … ORDER BY … LIMIT` once the scan is parallel.
///
/// Instead, each morsel emits only its top-k **sort-key values** plus a `(morsel, row)`
/// locator; the merge sorts those narrow candidates and the wide columns are gathered **once**,
/// for just the `k` survivors, via `interleave` across the source morsels.
///
/// Result-identical to the eager path: the candidates are concatenated in morsel order — the
/// same order the eager merge produces — and the final sort uses the same keys and the same
/// trailing row-position tie-break, so it selects the same rows in the same order; the locator
/// gather then reproduces those exact rows. Callers pass a non-empty `parts`.
pub(crate) fn parallel_top_n(
    parts: &[RecordBatch],
    keys: &[SortKey],
    k: usize,
) -> Result<RecordBatch, InterpError> {
    use arrow::array::{UInt32Array, UInt32Builder};
    use rayon::prelude::*;

    let schema = parts[0].schema();
    // Per morsel (parallel): its ≤k local top-k indices, and the key columns gathered to those
    // rows — narrow (only the ORDER BY expressions), never the payload.
    let per: Vec<(usize, UInt32Array, Vec<ArrayRef>)> = parts
        .par_iter()
        .enumerate()
        .filter(|(_, b)| b.num_rows() > 0)
        .map(|(p, b)| -> Result<_, InterpError> {
            let idx = sort_indices(b, keys, Some(k))?;
            let key_cols = keys
                .iter()
                .map(|key| {
                    let col = key.expr.eval(b)?;
                    Ok(bc_runtime::gather::take_column(col.as_ref(), &idx)?)
                })
                .collect::<Result<Vec<ArrayRef>, InterpError>>()?;
            Ok((p, idx, key_cols))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Flatten the candidates in morsel order into: per-key concatenated columns + the two
    // locator arrays (which source morsel, which row within it).
    let total: usize = per.iter().map(|(_, idx, _)| idx.len()).sum();
    let mut morsel_of = UInt32Builder::with_capacity(total);
    let mut row_of = UInt32Builder::with_capacity(total);
    for (p, idx, _) in &per {
        for r in idx.values() {
            morsel_of.append_value(*p as u32);
            row_of.append_value(*r);
        }
    }
    let morsel_of = morsel_of.finish();
    let row_of = row_of.finish();

    // Sort the narrow candidates by the same keys + a trailing row-position tie-break (matching
    // `sort_indices`' limit path over the eager-merged batch), keeping the global top-k.
    let mut sort_columns: Vec<SortColumn> = Vec::with_capacity(keys.len() + 1);
    for (j, key) in keys.iter().enumerate() {
        let col_j = per
            .iter()
            .map(|(_, _, kc)| kc[j].as_ref())
            .collect::<Vec<_>>();
        let values = if col_j.is_empty() {
            key.expr.eval(&parts[0])? // empty: unreachable shape, keeps types
        } else {
            arrow::compute::concat(&col_j)?
        };
        sort_columns.push(SortColumn {
            values,
            options: Some(SortOptions {
                descending: key.descending,
                nulls_first: key.nulls_first,
            }),
        });
    }
    sort_columns.push(SortColumn {
        values: Arc::new(UInt64Array::from_iter_values(0..total as u64)),
        options: Some(SortOptions {
            descending: false,
            nulls_first: false,
        }),
    });
    let winners = lexsort_to_indices(&sort_columns, Some(k))?;

    // Gather the payload once, for the k survivors, straight from the source morsels.
    let pairs: Vec<(usize, usize)> = winners
        .values()
        .iter()
        .map(|&w| {
            (
                morsel_of.value(w as usize) as usize,
                row_of.value(w as usize) as usize,
            )
        })
        .collect();
    let columns = (0..schema.fields().len())
        .map(|c| {
            let arrays: Vec<&dyn Array> = parts.iter().map(|b| b.column(c).as_ref()).collect();
            Ok(arrow::compute::interleave(&arrays, &pairs)?)
        })
        .collect::<Result<Vec<ArrayRef>, InterpError>>()?;
    Ok(RecordBatch::try_new(schema, columns)?)
}

/// Evaluate each sort key against `batch` into an arrow `SortColumn` (values + options).
fn eval_sort_columns(
    batch: &RecordBatch,
    keys: &[SortKey],
) -> Result<Vec<SortColumn>, InterpError> {
    keys.iter()
        .map(|k| {
            Ok(SortColumn {
                values: k.expr.eval(batch)?,
                options: Some(SortOptions {
                    descending: k.descending,
                    nulls_first: k.nulls_first,
                }),
            })
        })
        .collect()
}

/// Window over a single (already-materialized) batch, at the default parallel-row
/// threshold. Evaluates partition/order keys + each function input, runs the runtime
/// window kernel, and appends one column per function (named by alias) to the input.
pub(crate) fn window_batch(
    batch: &RecordBatch,
    partition_keys: &[bc_expr::Expr],
    order_keys: &[SortKey],
    functions: &[WindowFunc],
    rank_limit: Option<usize>,
) -> Result<RecordBatch, InterpError> {
    window_batch_with(
        batch,
        partition_keys,
        order_keys,
        functions,
        rank_limit,
        bc_arrow::RuntimeTuning::default().window_parallel_row_threshold,
    )
}

/// [`window_batch`] with a caller-supplied parallel-row threshold (perf-only — it only
/// decides whether per-partition sorts run across cores; the output is identical).
pub(crate) fn window_batch_with(
    batch: &RecordBatch,
    partition_keys: &[bc_expr::Expr],
    order_keys: &[SortKey],
    functions: &[WindowFunc],
    rank_limit: Option<usize>,
    parallel_row_threshold: usize,
) -> Result<RecordBatch, InterpError> {
    let num_rows = batch.num_rows();

    let part_arrays: Vec<ArrayRef> = partition_keys
        .iter()
        .map(|e| e.eval(batch))
        .collect::<Result<_, _>>()?;

    let order_arrays: Vec<(ArrayRef, SortOptions)> = order_keys
        .iter()
        .map(|k| {
            Ok((
                k.expr.eval(batch)?,
                SortOptions {
                    descending: k.descending,
                    nulls_first: k.nulls_first,
                },
            ))
        })
        .collect::<Result<_, InterpError>>()?;

    let mut calls = Vec::with_capacity(functions.len());
    for f in functions {
        let values = match &f.input {
            Some(expr) => Some(expr.eval(batch)?),
            None => None,
        };
        calls.push(WindowCall {
            func: map_window_func(f.func),
            values,
            offset: f.offset,
            frame: map_frame(f.frame),
        });
    }

    let cols = window::window_with(
        &part_arrays,
        &order_arrays,
        &calls,
        num_rows,
        parallel_row_threshold,
    )?;

    // input columns + one appended column per function alias.
    let in_schema = batch.schema();
    let mut fields: Vec<Field> = in_schema
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
    for (f, col) in functions.iter().zip(&cols) {
        fields.push(Field::new(&f.alias, col.data_type().clone(), true));
        columns.push(col.clone());
    }
    let out = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;
    // Fused `QUALIFY <rank> <= k`: keep only rows whose ranking value is within the
    // limit. The optimizer sets `rank_limit` only for a single ranking function, so
    // the bound applies to the first appended column (`cols[0]`). This is exactly
    // `Filter(Window, rank <= k)` — but fused, so the full windowed batch is never
    // emitted downstream and the separate filter is gone.
    match (rank_limit, cols.first()) {
        (Some(k), Some(rank_col)) => Ok(filter_by_rank_limit(&out, rank_col, k)?),
        _ => Ok(out),
    }
}

/// Keep rows of `batch` whose `rank_col` value is `<= limit` (a fused per-partition
/// top-N). `rank_col` is a ranking output (`row_number`/`rank`/`dense_rank`), whose
/// per-partition values start at 1, so a global `<= limit` mask selects the top rows
/// of every partition at once.
fn filter_by_rank_limit(
    batch: &RecordBatch,
    rank_col: &ArrayRef,
    limit: usize,
) -> Result<RecordBatch, InterpError> {
    use arrow::array::Int64Array;
    use arrow::compute::filter_record_batch;

    let ranks = rank_col
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("ranking window functions (row_number/rank/dense_rank) produce Int64 output");
    let limit = limit as i64;
    let mask: BooleanArray = ranks
        .iter()
        .map(|v| Some(v.is_some_and(|r| r <= limit)))
        .collect();
    Ok(filter_record_batch(batch, &mask)?)
}

fn map_window_func(f: WindowFn) -> window::WindowFn {
    match f {
        WindowFn::RowNumber => window::WindowFn::RowNumber,
        WindowFn::Rank => window::WindowFn::Rank,
        WindowFn::DenseRank => window::WindowFn::DenseRank,
        WindowFn::PercentRank => window::WindowFn::PercentRank,
        WindowFn::CumeDist => window::WindowFn::CumeDist,
        WindowFn::Ntile => window::WindowFn::Ntile,
        WindowFn::Sum => window::WindowFn::Sum,
        WindowFn::Avg => window::WindowFn::Avg,
        WindowFn::Min => window::WindowFn::Min,
        WindowFn::Max => window::WindowFn::Max,
        WindowFn::Count => window::WindowFn::Count,
        WindowFn::FirstValue => window::WindowFn::FirstValue,
        WindowFn::LastValue => window::WindowFn::LastValue,
        WindowFn::Lag => window::WindowFn::Lag,
        WindowFn::Lead => window::WindowFn::Lead,
        WindowFn::NthValue => window::WindowFn::NthValue,
        WindowFn::ForwardFill => window::WindowFn::ForwardFill,
        WindowFn::BackwardFill => window::WindowFn::BackwardFill,
    }
}

/// Map an IR window frame to the runtime frame. `ROWS` and `GROUPS` frames are
/// honored directly. A `RANGE` frame is honored only for peer bounds (CURRENT ROW /
/// UNBOUNDED); a numeric `RANGE` offset is value-based (typed order-key arithmetic
/// we don't implement), so it falls back to `None` — the default peer-`RANGE`
/// running aggregate the runtime already provides.
fn map_frame(frame: Option<WindowFrame>) -> Option<window_frame::Frame> {
    let f = frame?;
    let unit = match f.units {
        FrameUnits::Rows => window_frame::FrameUnit::Rows,
        FrameUnits::Groups => window_frame::FrameUnit::Groups,
        FrameUnits::Range => {
            if is_numeric_offset(f.start) || is_numeric_offset(f.end) {
                return None;
            }
            window_frame::FrameUnit::Range
        }
    };
    Some(window_frame::Frame {
        unit,
        start: map_bound(f.start),
        end: map_bound(f.end),
    })
}

/// Whether a frame bound carries a numeric `n` offset (`<n> PRECEDING/FOLLOWING`).
fn is_numeric_offset(b: FrameBound) -> bool {
    matches!(
        b,
        FrameBound::Preceding { .. } | FrameBound::Following { .. }
    )
}

fn map_bound(b: FrameBound) -> window_frame::FrameBound {
    use window_frame::FrameBound as R;
    match b {
        FrameBound::UnboundedPreceding => R::UnboundedPreceding,
        FrameBound::Preceding { n } => R::Preceding(n),
        FrameBound::CurrentRow => R::CurrentRow,
        FrameBound::Following { n } => R::Following(n),
        FrameBound::UnboundedFollowing => R::UnboundedFollowing,
    }
}

/// Keep at most `n` rows after skipping `offset`, slicing morsels in order.
pub(crate) fn limit(batches: Vec<RecordBatch>, n: usize, offset: usize) -> Vec<RecordBatch> {
    // Capture the input schema before consuming `batches`, so a fully-truncated
    // result (notably `Limit(_, 0)`, the canonical empty marker) still carries a
    // schema-only batch. A downstream pipeline breaker — a join or aggregate — needs
    // a schema even over zero rows; returning a bare empty `Vec` would lose it.
    let schema = batches.first().map(|b| b.schema());
    let mut remaining_skip = offset;
    let mut remaining_take = n;
    let mut out = Vec::with_capacity(batches.len());
    for batch in batches {
        if remaining_take == 0 {
            break;
        }
        let rows = batch.num_rows();
        if remaining_skip >= rows {
            remaining_skip -= rows;
            continue;
        }
        let start = remaining_skip;
        remaining_skip = 0;
        let take_n = (rows - start).min(remaining_take);
        out.push(batch.slice(start, take_n));
        remaining_take -= take_n;
    }
    if out.is_empty() {
        if let Some(schema) = schema {
            out.push(RecordBatch::new_empty(schema));
        }
    }
    out
}

#[cfg(test)]
mod filter_tests {
    use super::*;
    use arrow::array::Int64Array;
    use bc_expr::{BinaryOp, Expr, Literal};
    use std::sync::Arc;

    fn batch() -> RecordBatch {
        RecordBatch::try_from_iter(vec![
            (
                "a",
                Arc::new(Int64Array::from(vec![1i64, 2, 3, 4])) as ArrayRef,
            ),
            (
                "b",
                Arc::new(Int64Array::from(vec![
                    Some(10i64),
                    None,
                    Some(30),
                    Some(40),
                ])) as ArrayRef,
            ),
        ])
        .unwrap()
    }

    fn cmp(op: BinaryOp, v: i64) -> bc_expr::Expr {
        Expr::Binary {
            op,
            left: Box::new(Expr::Col { name: "a".into() }),
            right: Box::new(Expr::Lit {
                value: Literal::Int(v),
            }),
        }
    }

    /// An all-true mask returns the input rows unchanged (the short-circuit must not
    /// reorder, drop, or alter validity).
    #[test]
    fn all_true_mask_returns_input_unchanged() {
        let b = batch();
        let out = filter_batch(&b, &cmp(BinaryOp::Gt, 0)).unwrap();
        assert_eq!(out, b);
    }

    /// An all-false mask returns an empty batch with the same schema.
    #[test]
    fn all_false_mask_returns_empty() {
        let b = batch();
        let out = filter_batch(&b, &cmp(BinaryOp::Gt, 100)).unwrap();
        assert_eq!(out.num_rows(), 0);
        assert_eq!(out.schema(), b.schema());
    }

    /// The ordinary partial mask still gathers, and matches arrow's filter exactly.
    #[test]
    fn partial_mask_matches_arrow_filter() {
        let b = batch();
        let out = filter_batch(&b, &cmp(BinaryOp::Gt, 2)).unwrap();
        assert_eq!(out.num_rows(), 2);
        let a = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(a.values(), &[3, 4]);
    }

    /// A NULL in the predicate means "not selected". A mask that is true everywhere it is
    /// non-null must NOT take the all-true short-circuit — the null row has to be dropped.
    #[test]
    fn null_in_mask_is_not_all_true() {
        let b = batch();
        // b > 0 is true for rows 0,2,3 and NULL for row 1.
        let pred = Expr::Binary {
            op: BinaryOp::Gt,
            left: Box::new(Expr::Col { name: "b".into() }),
            right: Box::new(Expr::Lit {
                value: Literal::Int(0),
            }),
        };
        let out = filter_batch(&b, &pred).unwrap();
        assert_eq!(out.num_rows(), 3, "the NULL row must be filtered out");
        let a = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(a.values(), &[1, 3, 4]);
    }

    /// A mask that is entirely NULL selects nothing.
    #[test]
    fn all_null_mask_returns_empty() {
        let b = RecordBatch::try_from_iter(vec![(
            "b",
            Arc::new(Int64Array::from(vec![None, None, None] as Vec<Option<i64>>)) as ArrayRef,
        )])
        .unwrap();
        let pred = Expr::Binary {
            op: BinaryOp::Gt,
            left: Box::new(Expr::Col { name: "b".into() }),
            right: Box::new(Expr::Lit {
                value: Literal::Int(0),
            }),
        };
        assert_eq!(filter_batch(&b, &pred).unwrap().num_rows(), 0);
    }
}

#[cfg(test)]
mod sort_tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array};
    use bc_expr::Expr;
    use std::sync::Arc;

    /// The parallel sample-sort must produce a relation byte-identical to the serial
    /// stable sort — same order for every column, including the tie / null / NaN /
    /// descending cases — across a large float key (the path that engages it).
    #[test]
    fn parallel_sort_matches_serial_sort() {
        let n = 200_000usize; // > PARALLEL_SORT_MIN_ROWS so the parallel path engages
                              // Keys with heavy ties (low precision), scattered nulls and a few NaNs, plus a
                              // distinct payload so a tie-break difference would show as a column mismatch.
        let keyv: Vec<Option<f64>> = (0..n)
            .map(|i| {
                if i % 101 == 0 {
                    None
                } else if i % 997 == 0 {
                    Some(f64::NAN)
                } else {
                    Some(((i * 7) % 500) as f64 / 4.0)
                }
            })
            .collect();
        let payload: Vec<i64> = (0..n as i64).collect();
        let batch = RecordBatch::try_from_iter(vec![
            ("k", Arc::new(Float64Array::from(keyv)) as ArrayRef),
            ("p", Arc::new(Int64Array::from(payload)) as ArrayRef),
        ])
        .unwrap();

        let names = ["k", "p"];
        for descending in [false, true] {
            for nulls_first in [false, true] {
                let keys = vec![SortKey {
                    expr: Expr::Col { name: "k".into() },
                    descending,
                    nulls_first,
                }];
                check_parallel_matches_serial(&batch, &keys, &names, descending, nulls_first);
            }
        }
    }

    /// Single integer key and a two-key (int leading) sort — the integer / multi-key
    /// generalization of the float sample-sort. Same invariant: identical key-column
    /// sequence and row multiset vs the serial sort.
    #[test]
    fn parallel_int_and_multikey_sort_match_serial() {
        let n = 200_000usize;
        let k: Vec<Option<i64>> = (0..n)
            .map(|i| {
                if i % 101 == 0 {
                    None
                } else {
                    Some(((i * 13) % 700) as i64)
                }
            })
            .collect();
        let s: Vec<i64> = (0..n as i64).map(|i| (i * 31) % 50).collect();
        let p: Vec<i64> = (0..n as i64).collect();
        let batch = RecordBatch::try_from_iter(vec![
            ("k", Arc::new(Int64Array::from(k)) as ArrayRef),
            ("s", Arc::new(Int64Array::from(s)) as ArrayRef),
            ("p", Arc::new(Int64Array::from(p)) as ArrayRef),
        ])
        .unwrap();
        for descending in [false, true] {
            for nulls_first in [false, true] {
                // Single int key.
                let one = vec![SortKey {
                    expr: Expr::Col { name: "k".into() },
                    descending,
                    nulls_first,
                }];
                check_parallel_matches_serial(
                    &batch,
                    &one,
                    &["k", "s", "p"],
                    descending,
                    nulls_first,
                );
                // Two-key sort (int leading): the secondary key sorts within each range.
                let two = vec![
                    SortKey {
                        expr: Expr::Col { name: "k".into() },
                        descending,
                        nulls_first,
                    },
                    SortKey {
                        expr: Expr::Col { name: "s".into() },
                        descending: false,
                        nulls_first,
                    },
                ];
                check_parallel_matches_serial(
                    &batch,
                    &two,
                    &["k", "s", "p"],
                    descending,
                    nulls_first,
                );
            }
        }
    }

    /// Late-materialized `parallel_top_n` must be byte-identical to the eager top-N (each
    /// morsel's full-row local top-k, merged, re-topped) — same rows, same order, every
    /// column — across ties, descending, nulls, and `k` smaller/larger than a morsel.
    #[test]
    fn parallel_top_n_matches_eager() {
        let n = 40_000usize;
        // Heavy ties on the key (so the row-position tie-break is exercised), a distinct
        // payload so any tie-break disagreement surfaces as a column mismatch, and nulls.
        let key: Vec<Option<i64>> = (0..n)
            .map(|i| (i % 97 != 0).then_some(((i * 13) % 200) as i64))
            .collect();
        let payload: Vec<i64> = (0..n as i64).collect();
        let batch = RecordBatch::try_from_iter(vec![
            ("k", Arc::new(Int64Array::from(key)) as ArrayRef),
            ("p", Arc::new(Int64Array::from(payload)) as ArrayRef),
        ])
        .unwrap();
        // Split into uneven morsels (the parallel executor's shape).
        let parts: Vec<RecordBatch> = [0usize, 7000, 16384, 23000, 32768, n]
            .windows(2)
            .map(|w| batch.slice(w[0], w[1] - w[0]))
            .collect();

        for descending in [false, true] {
            for nulls_first in [false, true] {
                let keys = vec![SortKey {
                    expr: Expr::Col { name: "k".into() },
                    descending,
                    nulls_first,
                }];
                for k in [5usize, 100, 20_000, 50_000] {
                    // Eager reference: per-morsel full-row top-k, merged, re-topped.
                    let locals: Vec<RecordBatch> = parts
                        .iter()
                        .map(|b| sort_batch(b, &keys, Some(k)).unwrap())
                        .collect();
                    let merged = materialize(&locals).unwrap();
                    let eager = sort_batch(&merged, &keys, Some(k)).unwrap();

                    let late = parallel_top_n(&parts, &keys, k).unwrap();

                    assert_eq!(late.num_rows(), eager.num_rows(), "k={k} desc={descending}");
                    for name in ["k", "p"] {
                        let ci = eager.schema().index_of(name).unwrap();
                        let (le, ea) = (late.column(ci), eager.column(ci));
                        let le = le.as_any().downcast_ref::<Int64Array>().unwrap();
                        let ea = ea.as_any().downcast_ref::<Int64Array>().unwrap();
                        for r in 0..ea.len() {
                            assert_eq!(le.is_null(r), ea.is_null(r), "null@{r} col={name} k={k}");
                            if !ea.is_null(r) {
                                assert_eq!(
                                    le.value(r),
                                    ea.value(r),
                                    "{name}@{r} k={k} desc={descending} nf={nulls_first}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Parallel sample-sort must match the serial sort in the **key-column sequence**
    /// (identical regardless of tie order — fully-tied rows carry identical key values)
    /// and in the **full-row multiset**.
    fn check_parallel_matches_serial(
        batch: &RecordBatch,
        keys: &[SortKey],
        col_names: &[&str],
        descending: bool,
        nulls_first: bool,
    ) {
        let serial = sort_batch(batch, keys, None).unwrap();
        // The sample-sort returns its ranges in key order; the sorted relation is their
        // concatenation.
        let ranges = parallel_sort_batch(batch, keys, None)
            .unwrap()
            .expect("parallel sort should engage");
        let parallel = arrow::compute::concat_batches(&batch.schema(), ranges.iter()).unwrap();

        // Encode a column's values as comparable tokens (null distinct from any value).
        let col_tokens = |b: &RecordBatch, name: &str| -> Vec<(u8, u64)> {
            let c = b.column(b.schema().index_of(name).unwrap());
            (0..c.len())
                .map(|i| {
                    if c.is_null(i) {
                        (0u8, 0)
                    } else if let Some(a) = c.as_any().downcast_ref::<Float64Array>() {
                        (1, a.value(i).to_bits())
                    } else {
                        (
                            1,
                            c.as_any().downcast_ref::<Int64Array>().unwrap().value(i) as u64,
                        )
                    }
                })
                .collect()
        };
        // Key-column sequences must match position-for-position.
        let key_names: Vec<&str> = keys
            .iter()
            .map(|k| match &k.expr {
                Expr::Col { name } => name.as_str(),
                _ => unreachable!("test uses column keys"),
            })
            .collect();
        for name in &key_names {
            assert_eq!(
                col_tokens(&serial, name),
                col_tokens(&parallel, name),
                "key '{name}' ordering differs (descending={descending}, nulls_first={nulls_first})"
            );
        }
        // Full-row multiset must be preserved.
        let rows = |b: &RecordBatch| -> Vec<Vec<(u8, u64)>> {
            let cols: Vec<Vec<(u8, u64)>> = col_names.iter().map(|n| col_tokens(b, n)).collect();
            let mut rows: Vec<Vec<(u8, u64)>> = (0..b.num_rows())
                .map(|i| cols.iter().map(|c| c[i]).collect())
                .collect();
            rows.sort_unstable();
            rows
        };
        assert_eq!(
            rows(&serial),
            rows(&parallel),
            "row multiset differs (descending={descending}, nulls_first={nulls_first})"
        );
    }

    /// The parallel path declines for a small input (sampling + partition overhead would
    /// dominate), so the caller uses the serial sort.
    #[test]
    fn parallel_sort_declines_small() {
        let small = RecordBatch::try_from_iter(vec![(
            "k",
            Arc::new(Float64Array::from(vec![3.0, 1.0, 2.0])) as ArrayRef,
        )])
        .unwrap();
        let keys = vec![SortKey {
            expr: Expr::Col { name: "k".into() },
            descending: false,
            nulls_first: false,
        }];
        assert!(parallel_sort_batch(&small, &keys, None).unwrap().is_none());
    }

    /// A large **string** leading key now range-partitions and sorts in parallel, and the
    /// result is identical to the serial oracle (ties resolve to input order on both paths
    /// via the stable string permutation).
    #[test]
    fn parallel_sort_handles_string_key_identically_to_serial() {
        let n = 200_000usize;
        let strs: Vec<String> = (0..n).map(|i| format!("s{}", i % 1000)).collect();
        let str_batch = RecordBatch::try_from_iter(vec![
            (
                "k",
                Arc::new(arrow::array::StringArray::from(strs)) as ArrayRef,
            ),
            (
                "p",
                Arc::new(arrow::array::Int64Array::from(
                    (0..n as i64).collect::<Vec<_>>(),
                )) as ArrayRef,
            ),
        ])
        .unwrap();
        for (descending, nulls_first) in [(false, false), (true, false), (false, true)] {
            let keys = vec![SortKey {
                expr: Expr::Col { name: "k".into() },
                descending,
                nulls_first,
            }];
            let ranges = parallel_sort_batch(&str_batch, &keys, None)
                .unwrap()
                .expect("string sample-sort should engage on a large input");
            let par = arrow::compute::concat_batches(&str_batch.schema(), ranges.iter()).unwrap();
            let seq = sort_batch(&str_batch, &keys, None).unwrap();
            assert_eq!(
                seq, par,
                "descending={descending} nulls_first={nulls_first}"
            );
        }
    }
}
