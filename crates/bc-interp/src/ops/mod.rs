//! Per-batch / per-side operator primitives shared by the sequential reference
//! executor (`crate::execute`) and the parallel executor (`crate::par`).
//!
//! Keeping the actual operator logic here — and having both executors call it —
//! is what guarantees the parallel path computes exactly what the sequential
//! oracle does (asserted by a Rust test and by the differential suite). The
//! executors differ only in *scheduling* (sequential vs rayon + hash-shuffle),
//! never in operator semantics.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanArray, RecordBatch};
use arrow::compute::SortOptions;
use arrow::compute::{
    concat_batches, filter_record_batch, interleave, lexsort_to_indices, sort_to_indices, take,
    SortColumn,
};
use arrow::datatypes::{Field, Schema, SchemaRef};
use arrow::row::{OwnedRow, RowConverter, Rows, SortField};
use bc_ir::{
    AggFunc, AggregateItem, FrameBound, FrameUnits, ProjectionItem, SortKey, WindowFn, WindowFrame,
    WindowFunc,
};
use bc_runtime::agg::{self, AggCall};
use bc_runtime::window::{self, WindowCall};
use bc_runtime::window_frame;

use crate::error::InterpError;

mod joins;
mod reshape;
pub(crate) use joins::{asof_join_batches, join_batches, key_indices};
pub(crate) use reshape::{sample_batch, sample_n_batches, unnest_batch, unpivot_batch};

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

/// Partition-local partial aggregation of one batch.
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
        sort_to_indices(&k.expr.eval(batch)?, Some(opts), limit)?
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

/// Out-of-core sort: sort each input morsel into a run and spill it (dropping the
/// input batch as we go), then merge the runs with a **bounded-fan-in, streaming**
/// k-way merge. Peak memory is O(`FANIN` morsels) regardless of input size — only
/// one batch per run in the active merge group is ever resident, and the output is
/// streamed back to disk between passes rather than concatenated. The result equals
/// a single in-memory `sort_batch` over the whole input (the merge is
/// order-preserving). Disk spill uses the runtime's Arrow-IPC [`DiskSpillStore`].
pub(crate) fn external_merge_sort(
    parts: Vec<RecordBatch>,
    keys: &[SortKey],
    dir: &std::path::Path,
) -> Result<Vec<RecordBatch>, InterpError> {
    use bc_runtime::agg::spill::{DiskSpillStore, SpillStore};

    // Pass 0: sort each input morsel into a run and spill it, dropping each input
    // batch as it is consumed so the sorted runs never co-reside with the full input.
    let mut store =
        DiskSpillStore::new(dir.to_path_buf(), parts.len().max(1)).map_err(InterpError::from)?;
    let mut n_runs = 0usize;
    for b in parts.into_iter() {
        if b.num_rows() == 0 {
            continue;
        }
        let run = sort_batch(&b, keys, None)?;
        store.append(n_runs, &run).map_err(InterpError::from)?;
        n_runs += 1;
        // `b` and `run` drop here — the input morsel's memory is released.
    }
    if n_runs == 0 {
        return Ok(Vec::new());
    }

    // Merge passes: each merges groups of <= FANIN runs into one larger (spilled) run,
    // streaming so only one batch per run is resident. Repeats until a single run
    // remains. Fan-in bounds the resident working set independent of the run count.
    const FANIN: usize = 16;
    while n_runs > 1 {
        let n_groups = n_runs.div_ceil(FANIN);
        let mut next =
            DiskSpillStore::new(dir.to_path_buf(), n_groups).map_err(InterpError::from)?;
        for g in 0..n_groups {
            let lo = g * FANIN;
            let hi = (lo + FANIN).min(n_runs);
            let mut readers = Vec::with_capacity(hi - lo);
            for i in lo..hi {
                if let Some(r) = store.open_reader(i).map_err(InterpError::from)? {
                    readers.push(r);
                }
            }
            stream_merge_group(readers, keys, &mut next, g)?;
        }
        store = next;
        n_runs = n_groups;
    }

    // The final run holds the globally sorted result; stream its morsels out.
    let mut out = Vec::new();
    if let Some(reader) = store.open_reader(0).map_err(InterpError::from)? {
        for batch in reader {
            let batch = batch?;
            if batch.num_rows() > 0 {
                out.push(batch);
            }
        }
    }
    Ok(out)
}

/// A streaming reader over one spilled run's batches.
type RunReader = arrow::ipc::reader::StreamReader<std::io::BufReader<std::fs::File>>;

/// Build the key-row converter for a run group from a sample batch, baking each
/// key's asc/desc/nulls options into the encoding so encoded rows compare in order.
fn build_key_converter(batch: &RecordBatch, keys: &[SortKey]) -> Result<RowConverter, InterpError> {
    let key_cols = eval_sort_keys(batch, keys)?;
    let fields: Vec<SortField> = key_cols
        .iter()
        .zip(keys)
        .map(|(arr, k)| {
            SortField::new_with_options(
                arr.data_type().clone(),
                SortOptions {
                    descending: k.descending,
                    nulls_first: k.nulls_first,
                },
            )
        })
        .collect();
    Ok(RowConverter::new(fields)?)
}

/// Advance reader `ri` to its next non-empty batch, encoding that batch's key rows.
/// Sets `cur[ri]`/`cur_rows[ri]` to `None` when the reader is exhausted. Builds the
/// shared `converter`/`schema` from the first batch seen across the group.
#[allow(clippy::too_many_arguments)]
fn load_next_run_batch(
    ri: usize,
    readers: &mut [RunReader],
    cur: &mut [Option<RecordBatch>],
    cur_rows: &mut [Option<Rows>],
    idx: &mut [usize],
    converter: &mut Option<RowConverter>,
    schema: &mut Option<SchemaRef>,
    keys: &[SortKey],
) -> Result<(), InterpError> {
    loop {
        match readers[ri].next() {
            Some(batch) => {
                let batch = batch?;
                if batch.num_rows() == 0 {
                    continue;
                }
                if schema.is_none() {
                    *schema = Some(batch.schema());
                }
                if converter.is_none() {
                    *converter = Some(build_key_converter(&batch, keys)?);
                }
                let key_cols = eval_sort_keys(&batch, keys)?;
                let rows = converter
                    .as_ref()
                    .expect("converter built above")
                    .convert_columns(&key_cols)?;
                cur[ri] = Some(batch);
                cur_rows[ri] = Some(rows);
                idx[ri] = 0;
                return Ok(());
            }
            None => {
                cur[ri] = None;
                cur_rows[ri] = None;
                return Ok(());
            }
        }
    }
}

/// Flush the accumulated `(slot, row)` selections into one output batch via
/// `interleave` and append it to `store`'s `out_partition`. Exhausted (`None`) slots
/// get a type-correct empty placeholder; they are never indexed by `sel` because a
/// flush always precedes loading a slot's next batch.
fn flush_selection(
    sel: &mut Vec<(usize, usize)>,
    cur: &[Option<RecordBatch>],
    schema: &SchemaRef,
    store: &mut dyn bc_runtime::agg::spill::SpillStore,
    out_partition: usize,
) -> Result<(), InterpError> {
    if sel.is_empty() {
        return Ok(());
    }
    let mut cols: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
    for (c, field) in schema.fields().iter().enumerate() {
        let owned: Vec<ArrayRef> = cur
            .iter()
            .map(|b| match b {
                Some(batch) => batch.column(c).clone(),
                None => arrow::array::new_empty_array(field.data_type()),
            })
            .collect();
        let refs: Vec<&dyn Array> = owned.iter().map(|a| a.as_ref()).collect();
        cols.push(interleave(&refs, sel)?);
    }
    let batch = RecordBatch::try_new(schema.clone(), cols)?;
    store.append(out_partition, &batch).map_err(InterpError::from)?;
    sel.clear();
    Ok(())
}

/// Streaming k-way merge of `readers` (each a sorted run) into `store`'s
/// `out_partition`. Holds at most one batch per reader plus one output morsel of
/// `(slot, row)` selections, so memory is bounded by the fan-in — not the run sizes.
fn stream_merge_group(
    mut readers: Vec<RunReader>,
    keys: &[SortKey],
    store: &mut dyn bc_runtime::agg::spill::SpillStore,
    out_partition: usize,
) -> Result<(), InterpError> {
    let k = readers.len();
    if k == 0 {
        return Ok(());
    }
    let mut cur: Vec<Option<RecordBatch>> = (0..k).map(|_| None).collect();
    let mut cur_rows: Vec<Option<Rows>> = (0..k).map(|_| None).collect();
    let mut idx: Vec<usize> = vec![0; k];
    let mut converter: Option<RowConverter> = None;
    let mut schema: Option<SchemaRef> = None;
    // Min-heap over the current head key of each live reader (owned, so it survives
    // the reader advancing to its next batch).
    let mut heap: BinaryHeap<Reverse<(OwnedRow, usize)>> = BinaryHeap::new();

    for ri in 0..k {
        load_next_run_batch(
            ri,
            &mut readers,
            &mut cur,
            &mut cur_rows,
            &mut idx,
            &mut converter,
            &mut schema,
            keys,
        )?;
        if let Some(rows) = &cur_rows[ri] {
            heap.push(Reverse((rows.row(0).owned(), ri)));
        }
    }
    // The output schema is fixed once the first batch is seen; `schema` (the Option)
    // stays threaded through later `load_next_run_batch` calls (a no-op once set).
    let Some(out_schema) = schema.clone() else {
        return Ok(()); // every reader was empty
    };

    let target = bc_arrow::DEFAULT_MORSEL_ROWS;
    let mut sel: Vec<(usize, usize)> = Vec::with_capacity(target);

    while let Some(Reverse((_key, ri))) = heap.pop() {
        sel.push((ri, idx[ri]));
        idx[ri] += 1;
        let n = cur[ri].as_ref().map_or(0, |b| b.num_rows());
        if idx[ri] < n {
            heap.push(Reverse((
                cur_rows[ri].as_ref().expect("live cursor").row(idx[ri]).owned(),
                ri,
            )));
        } else {
            // Reader `ri` exhausted its current batch. The pending selections still
            // reference the current batches, so flush before swapping `ri`'s batch.
            flush_selection(&mut sel, &cur, &out_schema, store, out_partition)?;
            load_next_run_batch(
                ri,
                &mut readers,
                &mut cur,
                &mut cur_rows,
                &mut idx,
                &mut converter,
                &mut schema,
                keys,
            )?;
            if let Some(rows) = &cur_rows[ri] {
                heap.push(Reverse((rows.row(0).owned(), ri)));
            }
        }
        if sel.len() >= target {
            flush_selection(&mut sel, &cur, &out_schema, store, out_partition)?;
        }
    }
    flush_selection(&mut sel, &cur, &out_schema, store, out_partition)
}

/// Evaluate the sort-key expressions of `batch` into their key columns.
fn eval_sort_keys(batch: &RecordBatch, keys: &[SortKey]) -> Result<Vec<ArrayRef>, InterpError> {
    keys.iter()
        .map(|k| k.expr.eval(batch).map_err(InterpError::from))
        .collect()
}

/// Window over a single (already-materialized) batch: evaluate partition keys,
/// order keys, and each function's optional input, run the runtime window kernel,
/// and append the resulting columns (one per function) to the input batch. Input
/// columns are preserved; the appended columns are named by each function alias.
pub(crate) fn window_batch(
    batch: &RecordBatch,
    partition_keys: &[bc_expr::Expr],
    order_keys: &[SortKey],
    functions: &[WindowFunc],
    rank_limit: Option<usize>,
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

    let cols = window::window(&part_arrays, &order_arrays, &calls, num_rows)?;

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
    }
}

/// Map an IR window frame to the runtime frame. Only `ROWS` frames are honored;
/// an explicit `RANGE` frame falls back to `None` (the default peer-`RANGE`
/// running aggregate the runtime already implements).
fn map_frame(frame: Option<WindowFrame>) -> Option<window_frame::Frame> {
    let f = frame?;
    if f.units != FrameUnits::Rows {
        return None;
    }
    Some(window_frame::Frame {
        start: map_bound(f.start),
        end: map_bound(f.end),
    })
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

/// Split morsels that exceed the target's row OR byte bound, so there is enough
/// work to parallelize and no morsel's working set balloons on wide/variable-width
/// data. A row whose own width already exceeds the byte budget is emitted as a
/// one-row morsel (never coalesced): a giant cell cannot co-reside with 16 k
/// others, and it is never dropped or silently OOM'd.
///
/// With a row-only target (`MorselTarget::rows`, byte bound = `usize::MAX`) this is
/// byte-for-byte identical to the historical row-count morselizer.
pub(crate) fn morselize(
    batches: &[RecordBatch],
    target: bc_arrow::MorselTarget,
) -> Vec<RecordBatch> {
    let mut out = Vec::new();
    for b in batches {
        let n = b.num_rows();
        let chunk = morsel_chunk_rows(b, n, target);
        if n <= chunk {
            out.push(b.clone());
        } else {
            let mut off = 0;
            while off < n {
                let len = (n - off).min(chunk);
                out.push(b.slice(off, len));
                off += len;
            }
        }
    }
    out
}

/// Rows per output chunk for one batch: the row target, further capped so a chunk
/// stays within the byte budget. Returns `target.rows` for a row-only target or an
/// empty/zero-byte batch (the historical behavior); a single row wider than the
/// whole budget yields a chunk of 1 (a one-row morsel).
fn morsel_chunk_rows(b: &RecordBatch, n: usize, target: bc_arrow::MorselTarget) -> usize {
    if !target.byte_bounded() || n == 0 {
        return target.rows;
    }
    let bytes = b.get_array_memory_size();
    if bytes == 0 {
        return target.rows;
    }
    let avg = (bytes as f64 / n as f64).max(1.0);
    let by_bytes = ((target.bytes as f64 / avg).floor() as usize).max(1);
    target.rows.min(by_bytes)
}
