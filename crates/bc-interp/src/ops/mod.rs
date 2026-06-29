//! Per-batch / per-side operator primitives shared by the sequential reference
//! executor (`crate::execute`) and the parallel executor (`crate::par`).
//!
//! Keeping the actual operator logic here — and having both executors call it —
//! is what guarantees the parallel path computes exactly what the sequential
//! oracle does (asserted by a Rust test and by the differential suite). The
//! executors differ only in *scheduling* (sequential vs rayon + hash-shuffle),
//! never in operator semantics.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanArray, RecordBatch};
use arrow::compute::SortOptions;
use arrow::compute::{
    concat_batches, filter_record_batch, lexsort_to_indices, sort_to_indices, take, SortColumn,
};
use arrow::datatypes::{DataType, Field, Schema};
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
mod mixed_spill;
mod morsel;
mod quantile_spill;
mod radix_sort;
mod reshape;
pub(crate) use external_sort::{external_merge_sort, external_sort_to_final_store};
pub(crate) use joins::{asof_join_batches, join_batches, join_batches_with, key_indices};
pub(crate) use mixed_spill::try_bounded_mixed_spill;
pub(crate) use morsel::{morselize, remorselize};
pub(crate) use quantile_spill::{
    try_bounded_distinct_spill, try_bounded_histogram_spill, try_bounded_mode_spill,
    try_bounded_quantile_spill,
};
pub(crate) use reshape::{
    add_row_ids, sample_batch, sample_n_batches, unnest_batch, unpivot_batch,
};

// --- filter / project --------------------------------------------------------

/// A compiled expression (JIT fast path), or `None` to use the interpreter.
/// `CompiledExpr` is `Send + Sync`, so this is shared across rayon workers.
pub(crate) type Jit = Option<std::sync::Arc<bc_codegen::CompiledExpr>>;

/// Compile `expr` once for an operator using `sample` as a representative batch.
/// Returns `None` if the expression is outside the JIT's supported subset — the
/// interpreter then handles it. (Compiling once and reusing across morsels is
/// what makes the JIT win; a per-morsel compile would lose to the interpreter.)
pub(crate) fn try_compile(expr: &bc_expr::Expr, sample: &RecordBatch) -> Jit {
    bc_codegen::compile_expr(expr, sample)
        .ok()
        .map(std::sync::Arc::new)
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
        // A bare-column passthrough carries the *source field* through (renamed),
        // preserving its metadata — notably the Arrow extension type (e.g.
        // FixedShapeTensor for embeddings/decoded media). Rebuilding from
        // `array.data_type()` would drop that metadata, downgrading a tensor column
        // to its plain storage type. Computed expressions get a fresh field.
        let field = match &item.expr {
            bc_expr::Expr::Col { name } => match batch.schema().index_of(name) {
                Ok(idx) => batch
                    .schema()
                    .field(idx)
                    .clone()
                    .with_name(item.alias.clone()),
                Err(_) => Field::new(&item.alias, array.data_type().clone(), true),
            },
            _ => Field::new(&item.alias, array.data_type().clone(), true),
        };
        fields.push(field);
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
fn try_compile_computed(expr: &bc_expr::Expr, sample: &RecordBatch) -> Jit {
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

/// Concatenate morsels into one batch. Errors if there are none (no schema).
pub(crate) fn materialize(batches: &[RecordBatch]) -> Result<RecordBatch, InterpError> {
    match batches.first() {
        Some(first) => Ok(concat_batches(&first.schema(), batches)?),
        None => Err(InterpError::EmptyJoinInput),
    }
}

/// Sort a single (already-materialized) batch by the given keys.
pub(crate) fn sort_batch(
    batch: &RecordBatch,
    keys: &[SortKey],
    limit: Option<usize>,
) -> Result<RecordBatch, InterpError> {
    if batch.num_rows() == 0 {
        return Ok(batch.clone());
    }
    // Single-key sort uses arrow's specialized per-type `sort_to_indices` (a
    // dedicated primitive path) rather than the general multi-column `lexsort`.
    let indices = if let [k] = keys {
        let opts = SortOptions {
            descending: k.descending,
            nulls_first: k.nulls_first,
        };
        let vals = k.expr.eval(batch)?;
        // Radix fast path for a *full* sort (no top-N partial sort to beat) on a
        // fixed-width integer/temporal key: O(n) vs the comparison sort's O(n log n),
        // producing the identical relation. Falls back for limits / other types.
        let radix = limit
            .is_none()
            .then(|| radix_sort::radix_sort_indices(&vals, opts))
            .flatten();
        match radix {
            Some(idx) => idx,
            None => sort_to_indices(&vals, Some(opts), limit)?,
        }
    } else {
        let sort_columns: Vec<SortColumn> = keys
            .iter()
            .map(|k| {
                Ok(SortColumn {
                    values: k.expr.eval(batch)?,
                    options: Some(SortOptions {
                        descending: k.descending,
                        nulls_first: k.nulls_first,
                    }),
                })
            })
            .collect::<Result<_, InterpError>>()?;
        // A `limit` makes this a top-N: arrow returns only the first `limit`
        // indices via a partial sort, far cheaper than fully sorting then slicing.
        lexsort_to_indices(&sort_columns, limit)?
    };
    let columns = batch
        .columns()
        .iter()
        .map(|c| take(c.as_ref(), &indices, None))
        .collect::<Result<Vec<ArrayRef>, _>>()?;
    Ok(RecordBatch::try_new(batch.schema(), columns)?)
}

/// Rows below which the single-node sample-sort stays serial — the sampling + range
/// partition + concat overhead only pays off on a large full sort.
const PARALLEL_SORT_MIN_ROWS: usize = 1 << 17;

/// Parallel single-node full sort by sample-sort: range-partition the rows by the
/// leading key (sampled quantile boundaries), sort each range in parallel, and
/// concatenate in key order — no final merge, because the ranges are globally ordered
/// relative to each other. This is the single-node form of the distributed range sort
/// (`dist/flight_sort.py`), so one implementation serves both.
///
/// Returns `None` (caller falls back to the serial [`sort_batch`]) unless it applies: a
/// full sort (no `LIMIT` — top-N is already cheap), a large input, and a **float or
/// integer leading key** (the bucket boundaries route it *exactly* — floats by `f64`,
/// integers by `i64`; a string leading key falls back). Multi-key sorts are supported:
/// rows are bucketed by the leading key (equal leading keys never span a boundary, so
/// they stay in one range), then each range is sorted by the *full* key list — so a plain
/// concatenation in leading-key order is the globally sorted multi-key relation, no merge.
pub(crate) fn parallel_sort_batch(
    batch: &RecordBatch,
    keys: &[SortKey],
    limit: Option<usize>,
) -> Result<Option<RecordBatch>, InterpError> {
    use rayon::prelude::*;

    let Some(k0) = keys.first() else {
        return Ok(None);
    };
    if limit.is_some() || batch.num_rows() < PARALLEL_SORT_MIN_ROWS {
        return Ok(None);
    }
    let key = k0.expr.eval(batch)?;
    let parts = rayon::current_num_threads().clamp(2, 64);

    // Range-partition by the leading key — exactly (f64 for floats, i64 for integers).
    // A string/other leading key can't be range-partitioned here, so fall back.
    let buckets = if matches!(key.data_type(), DataType::Float64 | DataType::Float32) {
        let key_f64 = arrow::compute::cast(&key, &DataType::Float64)?;
        let keyv = key_f64
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .expect("cast to Float64");
        let Some(b) = sample_boundaries_f64(keyv, parts) else {
            return Ok(None);
        };
        bc_runtime::shuffle::range_partition_by_key_array(
            batch,
            &key_f64,
            &b,
            parts,
            k0.nulls_first,
            k0.descending,
        )?
    } else if key.data_type().is_integer() {
        let key_i64 = arrow::compute::cast(&key, &DataType::Int64)?;
        let keyv = key_i64
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .expect("cast to Int64");
        let Some(b) = sample_boundaries_i64(keyv, parts) else {
            return Ok(None);
        };
        bc_runtime::shuffle::range_partition_by_i64_key(
            batch,
            &key_i64,
            &b,
            parts,
            k0.nulls_first,
            k0.descending,
        )?
    } else {
        return Ok(None);
    };

    // Each range sorts by the *full* key list in parallel.
    let sorted: Vec<RecordBatch> = buckets
        .par_iter()
        .map(|b| sort_batch(b, keys, None))
        .collect::<Result<_, InterpError>>()?;

    // Ascending → ranges 0..P; descending → reversed (range P-1 holds the largest keys).
    let ordered: Vec<&RecordBatch> = if k0.descending {
        sorted.iter().rev().collect()
    } else {
        sorted.iter().collect()
    };
    Ok(Some(concat_batches(&batch.schema(), ordered)?))
}

/// Sample `parts-1` ascending f64 quantile boundaries from a float key column. Returns
/// `None` if fewer than `parts` finite values exist (nothing meaningful to split).
fn sample_boundaries_f64(key: &arrow::array::Float64Array, parts: usize) -> Option<Vec<f64>> {
    let n = key.len();
    let target = 8192.min(n).max(parts);
    let stride = (n / target).max(1);
    let mut sample: Vec<f64> = (0..n)
        .step_by(stride)
        .filter(|&i| key.is_valid(i))
        .map(|i| key.value(i))
        .filter(|v| !v.is_nan())
        .collect();
    if sample.len() < parts {
        return None;
    }
    sample.sort_unstable_by(|a, b| a.total_cmp(b));
    let m = sample.len();
    Some(
        (1..parts)
            .map(|j| sample[(j * m / parts).min(m - 1)])
            .collect(),
    )
}

/// Sample `parts-1` ascending i64 quantile boundaries from an integer key column (the
/// exact-integer analog of [`sample_boundaries_f64`]). `None` if too few non-null values.
fn sample_boundaries_i64(key: &arrow::array::Int64Array, parts: usize) -> Option<Vec<i64>> {
    let n = key.len();
    let target = 8192.min(n).max(parts);
    let stride = (n / target).max(1);
    let mut sample: Vec<i64> = (0..n)
        .step_by(stride)
        .filter(|&i| key.is_valid(i))
        .map(|i| key.value(i))
        .collect();
    if sample.len() < parts {
        return None;
    }
    sample.sort_unstable();
    let m = sample.len();
    Some(
        (1..parts)
            .map(|j| sample[(j * m / parts).min(m - 1)])
            .collect(),
    )
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
    let mut out = Vec::new();
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
        let parallel = parallel_sort_batch(batch, keys, None)
            .unwrap()
            .expect("parallel sort should engage");

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

    /// The parallel path declines for a small input and for a non-numeric (string)
    /// leading key, so the caller uses the serial sort.
    #[test]
    fn parallel_sort_declines_small_and_string() {
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

        let n = 200_000usize;
        let strs: Vec<String> = (0..n).map(|i| format!("s{}", i % 1000)).collect();
        let str_batch = RecordBatch::try_from_iter(vec![(
            "k",
            Arc::new(arrow::array::StringArray::from(strs)) as ArrayRef,
        )])
        .unwrap();
        // String leading key → declines (can't range-partition here); serial sort handles it.
        assert!(parallel_sort_batch(&str_batch, &keys, None)
            .unwrap()
            .is_none());
    }
}
