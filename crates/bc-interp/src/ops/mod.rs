//! Per-batch / per-side operator primitives shared by the sequential reference
//! executor (`crate::execute`) and the parallel executor (`crate::par`).
//!
//! Keeping the actual operator logic here — and having both executors call it —
//! is what guarantees the parallel path computes exactly what the sequential
//! oracle does (asserted by a Rust test and by the differential suite). The
//! executors differ only in *scheduling* (sequential vs rayon + hash-shuffle),
//! never in operator semantics.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanArray, Int64Array, RecordBatch, UInt64Array};
use arrow::compute::SortOptions;
use arrow::compute::{filter_record_batch, lexsort_to_indices, SortColumn};
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
pub(crate) use external_sort::{
    external_merge_sort, external_sort_to_final_store, DEFAULT_RUN_TARGET_BYTES,
};
pub(crate) use joins::{
    asof_join_batches, columns_by_name, gather_join_output, gather_join_output_with, join_batches,
    join_batches_with, join_output_schema, join_top_n, key_indices, map_join_type,
    range_join_batches,
};
pub(crate) use materialize::materialize;
pub(crate) use mixed_spill::try_bounded_mixed_spill;
pub(crate) use morsel::{morselize_par, remorselize, sliced_batch_bytes};
pub(crate) use quantile_spill::{
    try_bounded_distinct_spill, try_bounded_histogram_spill, try_bounded_mode_spill,
    try_bounded_quantile_spill,
};
pub(crate) use repartition::{partition_morsels, partition_morsels_by_index_salted};
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

/// Filter with no compiled predicate and no cross-morsel ordering state.
///
/// This is what the sequential oracle uses. It stays on the *static* conjunct order
/// deliberately: the oracle's job is to be the reference every other path is checked
/// against, and a reference that carries adaptive state is a worse reference even when the
/// adaptation cannot change its answer (which, for the conjuncts of an `AND`, it cannot).
pub(crate) fn filter_batch(
    batch: &RecordBatch,
    predicate: &bc_expr::Expr,
) -> Result<RecordBatch, InterpError> {
    filter_batch_jit(batch, predicate, &None, None)
}

/// Filter using a pre-compiled predicate when possible, and a measured conjunct order when
/// the caller keeps one per operator.
pub(crate) fn filter_batch_jit(
    batch: &RecordBatch,
    predicate: &bc_expr::Expr,
    jit: &Jit,
    order: Option<&bc_expr::ConjunctOrder>,
) -> Result<RecordBatch, InterpError> {
    // A conjunctive predicate the JIT did not take whole gets its conjuncts
    // short-circuited: the cheap one runs at full width, the rest only over the rows
    // it kept (`bc_expr::Expr::short_circuit_filter_mask`, which owns the argument for
    // why that is the same mask). Gated on `jit.is_none()` because a fully compiled
    // predicate is already a single fused pass with no intermediate mask to save —
    // splitting it would trade the better fast path for the lesser one. The mask is
    // interchangeable with the one below, so both feed the same gather.
    if jit.is_none() {
        if let Some(mask) = predicate.short_circuit_filter_mask_with(batch, order)? {
            return Ok(filter_record_batch(batch, &mask)?);
        }
    }
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
/// The shared body of partial aggregation: evaluate the group keys and each aggregate's
/// input(s), then hand them to `agg::partial`. The *only* thing the interpreter and the JIT
/// differ on is how a single expression is evaluated, so that is the one thing passed in —
/// `eval_group`/`eval_input`/`eval_input2` receive `(aggregate_index, expr)`. Keeping the
/// assembly here rather than in two twin functions is what makes invariant #6 (interp and
/// JIT are bit-for-bit identical) structural rather than a promise: there is one way to
/// build the `Partial`, and the two callers cannot drift apart on grouping, on the
/// arg_min/arg_max ordering key, or on the row count.
fn eval_partial_with(
    batch: &RecordBatch,
    group_keys: &[ProjectionItem],
    aggregates: &[AggregateItem],
    mut eval_group: impl FnMut(usize, &bc_expr::Expr) -> Result<ArrayRef, InterpError>,
    mut eval_input: impl FnMut(usize, &bc_expr::Expr) -> Result<ArrayRef, InterpError>,
    mut eval_input2: impl FnMut(usize, &bc_expr::Expr) -> Result<ArrayRef, InterpError>,
) -> Result<agg::Partial, InterpError> {
    let group_arrays: Vec<ArrayRef> = group_keys
        .iter()
        .enumerate()
        .map(|(i, k)| eval_group(i, &k.expr))
        .collect::<Result<_, _>>()?;
    let mut calls = Vec::with_capacity(aggregates.len());
    for (i, item) in aggregates.iter().enumerate() {
        let values = match &item.input {
            Some(expr) => Some(eval_input(i, expr)?),
            None => None,
        };
        // The ordering key for arg_min/arg_max (the aggregate's second input).
        let key = match &item.input2 {
            Some(expr) => Some(eval_input2(i, expr)?),
            None => None,
        };
        calls.push(AggCall::with_key(map_agg_func(item), values, key));
    }
    Ok(agg::partial(&group_arrays, &calls, batch.num_rows())?)
}

pub(crate) fn eval_partial(
    batch: &RecordBatch,
    group_keys: &[ProjectionItem],
    aggregates: &[AggregateItem],
) -> Result<agg::Partial, InterpError> {
    eval_partial_with(
        batch,
        group_keys,
        aggregates,
        |_, e| e.eval(batch).map_err(Into::into),
        |_, e| e.eval(batch).map_err(Into::into),
        |_, e| e.eval(batch).map_err(Into::into),
    )
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
    eval_partial_with(
        batch,
        group_keys,
        aggregates,
        |i, e| eval_jit(&jit.group[i], e, batch),
        |i, e| eval_jit(&jit.input[i], e, batch),
        |i, e| eval_jit(&jit.input2[i], e, batch),
    )
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

/// Deduplicate whole rows across `batches` in parallel, as the same *multiset* of unique rows
/// the serial [`distinct_partial`] oracle produces (DISTINCT's group order is unspecified). The
/// streaming executor otherwise handed DISTINCT to the single-threaded oracle.
///
/// A single dense-integer key takes the presence-bitmap fast path ([`agg::distinct_dense`]).
/// Otherwise this is a mergeable dedup: dedup each batch into a `Partial` in parallel, then
/// `combine` the partials into the global distinct set. It stays fast across the whole
/// cardinality range — a low-cardinality key yields tiny per-batch partials that merge in one
/// cheap pass (no per-core hash-partition of millions of rows), a high-cardinality key
/// parallelizes both the per-batch dedup and the combine — so it never regresses the few-groups
/// case the way an unconditional hash-partition would.
pub(crate) fn parallel_distinct(batches: &[RecordBatch]) -> Result<Vec<RecordBatch>, InterpError> {
    use rayon::prelude::*;
    if let Some(out) = agg::distinct_dense(batches)? {
        return Ok(vec![out]);
    }
    let partials: Vec<agg::Partial> = batches
        .par_iter()
        .map(distinct_partial)
        .collect::<Result<_, InterpError>>()?;
    let combined = agg::combine(&partials, &[])?;
    Ok(vec![RecordBatch::try_new(
        batches[0].schema(),
        combined.group_columns,
    )?])
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
        AggFunc::AnyValue => agg::AggFunc::AnyValue,
        AggFunc::Entropy => agg::AggFunc::Entropy,
        AggFunc::Mad => agg::AggFunc::Mad,
        // Same permille encoding as `Quantile`: the param is a fraction on the wire and
        // an integer permille in the runtime, so the two cannot drift apart.
        AggFunc::QuantileDisc => {
            agg::AggFunc::QuantileDisc((item.param.unwrap_or(0.5) * 1000.0).round() as u16)
        }
        // `k` is a count, not a fraction, so it rides `param` unscaled.
        AggFunc::ApproxTopK => agg::AggFunc::ApproxTopK(item.param.unwrap_or(1.0).round() as u16),
        AggFunc::KurtosisPop => agg::AggFunc::KurtosisPop,
        AggFunc::KahanSum => agg::AggFunc::KahanSum,
    }
}

// --- sort / limit / materialize ---------------------------------------------

/// Normalize an evaluated sort-key column so arrow's order-based kernels rank it the way
/// the engine does. Two independent corrections, both needed before *any* ordering kernel.
///
/// **An all-`Null`-typed key.** A sort key that evaluates to the `Null` type (an all-null
/// column, e.g. a `from_pydict` column that is entirely `None`) has no natural order, and
/// `lexsort_to_indices` / `RowConverter` reject it outright — turning `ORDER BY <all-null
/// col>` into a hard error on a path DuckDB executes fine. Such a key is all-equal, so it
/// must contribute *nothing* to the ordering: substitute a constant (all-equal) column, and
/// ties fall through to the following keys and the trailing row-index tie-break — exactly
/// DuckDB's "order by the remaining keys". The asc/desc/nulls-first flags on the key are then
/// irrelevant (a constant sorts identically under any of them).
///
/// **A float key.** Arrow's ordering kernels rank floats on the **raw** bits
/// (`f64::total_cmp`), which ranks a *negative* NaN below `-inf` while a positive one ranks
/// above `+inf`, and splits `-0.0` from `0.0`. So `ORDER BY f` put a `-NaN` **first** where
/// DuckDB puts every NaN last, and where the engine's own `MIN`/`MAX`/`=` rank it greatest —
/// `ORDER BY x DESC LIMIT 1` and `max(x)` disagreed on the same column. That is not an exotic
/// input: on x86 `0.0/0.0` and `sqrt(-1)` both *produce* a negative NaN, so `SELECT x/y AS r
/// ... ORDER BY r` reaches it with ordinary data. `bc_arrow::canon_float_array` folds `-0.0`
/// into `0.0` and every NaN into one, so the same kernels compute the engine's relation.
/// Only the *key* is canonicalized; the sort then gathers the original rows, so a `-NaN` in
/// is still a `-NaN` out — the ordering is corrected without rewriting the user's data.
///
/// Applied at every sort-key eval site so the serial, parallel sample-sort, radix, top-N, and
/// spilling merge paths all agree — they route through here, so they cannot drift apart.
/// (`bc-runtime`'s `keys::canonicalize_float_keys` is the sibling for grouping/join keys.)
pub(crate) fn normalize_sort_key(arr: ArrayRef) -> ArrayRef {
    if matches!(arr.data_type(), DataType::Null) {
        Arc::new(Int64Array::from(vec![0i64; arr.len()]))
    } else {
        // A float key is canonicalized so the ordering kernels rank it the way the engine's
        // float identity says — see the doc comment above.
        bc_arrow::canon_float_array(&arr)
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
    // SINGLE-KEY top-N: take the stable full-sort permutation and keep its first `limit`.
    //
    // That routes through the single-key fast paths in [`sort_indices_of`] (the stable string
    // permutation builder, the integer/temporal radix) instead of the multi-column
    // `lexsort_to_indices` the row-index tie-break below forces — which RowConverter-encodes
    // EVERY row just to keep `limit` of them. On a 6M-row `ORDER BY <utf8> LIMIT 100` that
    // encode *was* the operator: the plain full sort of the same column measured ~7x faster
    // than the "top-N" it was supposed to beat. Both paths produce the same stable order (ties
    // resolve to input order), so the first `limit` rows are identical either way — this only
    // changes how they are found. Multi-key top-N keeps the partial `lexsort` below, where the
    // `limit` genuinely buys an O(n log k) partial sort.
    if let (Some(k), 1) = (limit, keys.len()) {
        let vals: Vec<ArrayRef> = keys
            .iter()
            .map(|key| key.expr.eval(batch))
            .collect::<Result<_, _>>()?;
        let full = sort_indices_of(&vals, keys)?;
        let n = k.min(full.len());
        return Ok(full.slice(0, n));
    }
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
        // NB: a row-encoded partial sort was tried here and does NOT pay. Arrow's
        // `partial_sort` keeps a bounded region of size `limit` (O(n log limit)) and barely
        // touches the tail, while the encode pass is O(n) in row *width* whatever the limit
        // is; selecting the limit-th encoded row and sorting the prefix measured 0.60x to
        // 0.66x at every limit tried. Multi-key top-N keeps arrow's path — see
        // `bc_arrow::row_sort`, which declines to offer a limited form for this reason.
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
    // An all-null `Null`-typed key has no natural order (arrow's kernels reject it) and is
    // all-equal anyway; substitute a constant so it contributes nothing to the ordering.
    let coerced: Vec<ArrayRef> = vals.iter().cloned().map(normalize_sort_key).collect();
    let vals: &[ArrayRef] = &coerced;
    if let ([k], [v]) = (keys, vals) {
        // Single-key *full* sort uses a stable specialized path per key type (string / radix)
        // rather than the general multi-column `lexsort`; anything neither handles falls
        // through to the stable lexsort path below.
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
        // sort's O(n log n), producing the identical relation.
        if let Some(idx) = radix_sort::radix_sort_indices(v, opts) {
            return Ok(idx);
        }
        // Any other single key (boolean, decimal, a NaN-bearing float the radix declines)
        // has no stable specialized path. Arrow's `sort_to_indices` is UNSTABLE, so its tie
        // order is arbitrary and input-size-dependent — which makes the serial oracle, the
        // per-range parallel sample-sort, and the per-run external merge sort disagree on
        // rows equal on the key. Fall through to the general lexsort path below, which
        // appends an ascending row-index tie-break: ties resolve to input order, the same
        // stability the radix/string single-key paths guarantee. (Not the specialized arrow
        // primitive, but this type is off the fast path anyway.)
    }
    let options = sort_options(keys);
    // Row-encoded stable sort. Identical permutation to the `lexsort` fallback below, but
    // the ascending row-index tie-break lives in the *comparator* instead of being encoded
    // as a trailing key column — so the encoder writes and every comparison memcmps four
    // fewer bytes per row, which on a narrow two-key sort is about a third of the encoded
    // width. `bc_arrow::row_sort` owns that equivalence argument and pins it against this
    // very fallback.
    if let Some(idx) = bc_arrow::row_sort::stable_lexsort_indices(vals, &options) {
        return Ok(idx);
    }

    let mut columns: Vec<SortColumn> = vals
        .iter()
        .zip(&options)
        .map(|(values, o)| SortColumn {
            values: values.clone(),
            options: Some(*o),
        })
        .collect();
    // Reached only for a key type the row encoder rejects. Append an ascending row-index as
    // the final tie-break so `lexsort` (which is unstable in arrow) resolves rows equal on
    // every real key to input order — the stability the single-key radix/string paths already
    // guarantee. Without it, the parallel sample-sort and the external merge sort (each
    // calling this over a differently-sized slice) order fully-tied rows differently from this
    // sequential oracle, breaking seq == par bit-for-bit. The slice this sorts is always
    // gathered in ascending original-row order, so a slice-local `0..n` preserves the input's
    // relative order of tied rows.
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

/// The per-key `SortOptions` for `keys`, in key order.
fn sort_options(keys: &[SortKey]) -> Vec<SortOptions> {
    keys.iter()
        .map(|k| SortOptions {
            descending: k.descending,
            nulls_first: k.nulls_first,
        })
        .collect()
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
/// The ≤`k` indices of one morsel's rows in sorted order — the per-morsel step of
/// [`parallel_top_n`], with the same deterministic input-order tie-break the eager oracle uses.
///
/// For a **single** sort key this is a *stable full sort* (the radix / specialized path,
/// no arrow row-format encoding) sliced to `k`, not the multi-column partial sort: `sort_indices`'
/// limit path appends a `row_index` tie-break key to make ties deterministic, which forces
/// `lexsort_to_indices` to encode **every row of every morsel** into the arrow row format — the
/// dominant cost, and independent of `k`, so a `LIMIT 10` top-N paid the same ~full-encode as
/// `LIMIT 10000`. A stable single-key sort keeps ties in input order already, so its first `k` is
/// bit-identical to the `(key, row_index)` partial sort at a fraction of the cost (radix is O(n)
/// and touches the values directly). Multi-key top-N keeps the partial `lexsort` (the row format
/// is inherent to comparing several columns; the win is specific to the one-key case).
/// The `k` best rows of a morsel, over key columns the caller has already evaluated and
/// normalized.
///
/// Taking pre-evaluated keys is what lets `parallel_top_n` run each ORDER BY expression once per
/// morsel and reuse it for the selection, the top-N bound check and the candidate gather.
/// Evaluating them here instead would repeat a computed key's work, and would repeat
/// `normalize_sort_key`'s whole-column scan for a float key.
fn top_k_indices_of(
    key_arrays: &[ArrayRef],
    keys: &[SortKey],
    num_rows: usize,
    k: usize,
) -> Result<arrow::array::UInt32Array, InterpError> {
    use arrow::array::UInt32Array;
    // A single key takes the O(n) specialized full sort ONLY where `sort_indices_of` has one:
    // the string permutation builder or the integer/temporal radix. Both are linear, so
    // sorting the whole morsel to keep `k` costs no more than selecting `k`.
    //
    // A float, decimal or boolean key has no specialized path. It used to full-`lexsort` every
    // morsel to keep `k` rows — an O(n log n) sort. Measured on 6M rows: `ORDER BY <f64> DESC
    // LIMIT 100` took **26.3 ms against DuckDB's 8.7 ms (3.0x)**, while the *three*-key form of
    // the same query ran in 18 ms because it reached the O(n) quickselect below. Fewer sort
    // keys costing more was the tell. So those keys now fall through to that same quickselect,
    // which is O(n) for any type and (with the fixed `parallel_top_n` tie-break) selects
    // exactly the stable sort's top-k — proven for a float key with `-0.0`/`0.0`, NaN and
    // heavy ties by `parallel_top_n_float_key_matches_eager`.
    if keys.len() == 1 {
        let v = key_arrays[0].clone();
        let opts = SortOptions {
            descending: keys[0].descending,
            nulls_first: keys[0].nulls_first,
        };
        // The string builder and the integer/temporal radix are stable *full* sorts, and for
        // top-k that only pays when the full sort is genuinely as cheap as selecting k. Integer
        // radix is (a few cache-friendly LSD passes over a compact key, measured a win vs
        // DuckDB). A **float** key is not: its radix runs 8 LSD passes scattering by a random
        // key byte, so sorting a whole morsel to keep 100 rows costs ~8x an O(n) selection and
        // thrashes cache. So exclude float here and let it fall to the quickselect below —
        // which the fixed `parallel_top_n` tie-break makes result-identical to this full sort
        // (`parallel_top_n_float_key_matches_eager`).
        let is_float = matches!(
            v.data_type(),
            DataType::Float16 | DataType::Float32 | DataType::Float64
        );
        let full = str_sort::stable_sort_indices_str(&v, opts).or_else(|| {
            (!is_float)
                .then(|| radix_sort::radix_sort_indices(&v, opts))
                .flatten()
        });
        if let Some(full) = full {
            let take = k.min(full.len());
            return Ok(UInt32Array::from_iter_values(
                full.values().iter().take(take).copied(),
            ));
        }
    }
    // General top-k (multi-key, or a single key with no radix/string fast path): an O(n)
    // `select_nth_unstable_by` (quickselect) over a total-order row comparator — each ORDER BY
    // key in turn, then the row index. This selects the k best rows while touching each value
    // directly, avoiding the arrow row-format encode of *every* row that
    // `lexsort_to_indices(Some(k))` pays regardless of k (that encode was the dominant,
    // k-independent cost — a `LIMIT 10` cost the same as `LIMIT 10000`). Because the comparator
    // is a strict total order (the trailing `row index` tie-break breaks every remaining tie),
    // the selected *set* is exactly the eager `(keys, row_index)` sort's top-k. The within-morsel
    // *order* of that set is unstable, which is immaterial because `parallel_top_n` re-sorts the
    // survivors globally and breaks ties by original `(morsel, row)` — never by this order.
    // Covered by `parallel_top_n_matches_eager` (int key) and
    // `parallel_top_n_float_key_matches_eager` (float key with -0.0/NaN/heavy ties).
    use arrow::array::make_comparator;
    use std::cmp::Ordering;
    let n = num_rows;
    let comparators = keys
        .iter()
        .zip(key_arrays)
        .map(|(key, arr)| {
            make_comparator(
                arr.as_ref(),
                arr.as_ref(),
                SortOptions {
                    descending: key.descending,
                    nulls_first: key.nulls_first,
                },
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let cmp = |a: &u32, b: &u32| -> Ordering {
        for c in &comparators {
            match c(*a as usize, *b as usize) {
                Ordering::Equal => continue,
                other => return other,
            }
        }
        a.cmp(b) // strict total order: earlier row wins a tie, matching the eager sort
    };
    let mut idx: Vec<u32> = (0..n as u32).collect();
    if n > k {
        idx.select_nth_unstable_by(k - 1, cmp);
        idx.truncate(k);
    }
    Ok(UInt32Array::from(idx))
}

pub(crate) fn parallel_top_n(
    parts: &[RecordBatch],
    keys: &[SortKey],
    k: usize,
) -> Result<RecordBatch, InterpError> {
    use arrow::array::{UInt32Array, UInt32Builder};
    use rayon::prelude::*;

    let schema = parts[0].schema();
    // A bound on the first key's cut-off, shared across workers. Once any morsel has produced
    // `k` candidates, a morsel whose entire first-key range is strictly worse than that cannot
    // contribute and is dropped for the price of one min/max pass — the cheap half of
    // DataFusion's dynamic top-K filter, applied where the data is already in hand.
    // `bc_runtime::topn` owns the soundness argument; the bound only ever tightens, so a stale
    // read costs a missed skip and never a wrong answer.
    let bound = bc_runtime::topn::TopNBound::new(keys[0].descending);
    // Per morsel (parallel): its ≤k local top-k indices, and the key columns gathered to those
    // rows — narrow (only the ORDER BY expressions), never the payload. `None` for a morsel the
    // bound excluded.
    let per: Vec<(usize, UInt32Array, Vec<ArrayRef>)> = parts
        .par_iter()
        .enumerate()
        .filter(|(_, b)| b.num_rows() > 0)
        .map(|(p, b)| -> Result<Option<_>, InterpError> {
            // Evaluate the ORDER BY expressions ONCE per morsel and reuse them for the
            // selection, the bound check and the candidate gather. They used to be evaluated
            // twice — once inside the selection and again here — which for a computed key is
            // the expression run twice, and for a float key is `normalize_sort_key` scanning
            // the whole column twice looking for `-0.0`/NaN.
            let key_arrays: Vec<ArrayRef> = keys
                .iter()
                .map(|key| Ok(normalize_sort_key(key.expr.eval(b)?)))
                .collect::<Result<_, InterpError>>()?;

            if let Some((min, max)) = bc_runtime::topn::i64_key_range(&key_arrays[0]) {
                if bound.excludes_range(min, max) {
                    return Ok(None);
                }
            }

            let idx = top_k_indices_of(&key_arrays, keys, b.num_rows(), k)?;
            let key_cols = key_arrays
                .iter()
                .map(|col| Ok(bc_runtime::gather::take_column(col.as_ref(), &idx)?))
                .collect::<Result<Vec<ArrayRef>, InterpError>>()?;
            // Only a *full* candidate set of `k` rows bounds the global cut-off; fewer than `k`
            // proves nothing about the k-th best.
            if idx.len() == k {
                if let Some(v) = bound.candidate_bound(&key_cols[0]) {
                    bound.publish(v);
                }
            }
            Ok(Some((p, idx, key_cols)))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect();

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
            normalize_sort_key(key.expr.eval(&parts[0])?) // empty: unreachable shape, keeps types
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
    // Tie-break by the candidate's ORIGINAL position — its source morsel, then its row within
    // that morsel — NOT by its position in the flattened candidate array. Those differ exactly
    // when a morsel's `top_k_indices_of` returns its rows in some order other than ascending row:
    // the multi-key (and single-key non-radix/non-string) path uses an *unstable* quickselect,
    // so among rows tied on the key the flatten order is arbitrary. The eager oracle breaks
    // such ties by original row, so tie-breaking on the flatten position let a different tied
    // row survive at the same rank — a data-size-dependent wrong answer that only appears with
    // real ties on the key (a distinct second key hides it). `(morsel, row)` is the original
    // order, so this matches the oracle regardless of how each morsel selected its candidates.
    sort_columns.push(SortColumn {
        values: Arc::new(morsel_of.clone()),
        options: Some(SortOptions::default()),
    });
    sort_columns.push(SortColumn {
        values: Arc::new(row_of.clone()),
        options: Some(SortOptions::default()),
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
                values: normalize_sort_key(k.expr.eval(batch)?),
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
            frame: map_frame(f.frame)?,
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
        WindowFn::Var => window::WindowFn::Var,
        WindowFn::Stddev => window::WindowFn::Stddev,
        WindowFn::Product => window::WindowFn::Product,
        WindowFn::BoolAnd => window::WindowFn::BoolAnd,
        WindowFn::BoolOr => window::WindowFn::BoolOr,
        WindowFn::BitAnd => window::WindowFn::BitAnd,
        WindowFn::BitOr => window::WindowFn::BitOr,
        WindowFn::BitXor => window::WindowFn::BitXor,
        WindowFn::CountDistinct => window::WindowFn::CountDistinct,
    }
}

/// Map an IR window frame to the runtime frame. `ROWS` and `GROUPS` frames are
/// honored directly. A `RANGE` frame is honored only for peer bounds (CURRENT ROW /
/// UNBOUNDED).
///
/// A numeric `RANGE` offset is *value*-based — the frame covers rows whose ORDER BY
/// value lies within `n` of the current row's, which needs typed order-key arithmetic
/// the runtime does not implement. It is rejected rather than approximated: silently
/// substituting the peer-`RANGE` running aggregate returns a *wrong answer* for any
/// frame that is not already peer-shaped, and a wrong answer is worse than an error.
///
/// The Python control plane rejects this shape first (`plan/logical/window.py` raises
/// `PlanError`, and the SQL parser raises `NotImplementedError`), so this is the
/// data-plane half of that contract — reachable only via directly-constructed IR.
fn map_frame(frame: Option<WindowFrame>) -> Result<Option<window_frame::Frame>, InterpError> {
    let Some(f) = frame else {
        return Ok(None);
    };
    let unit = match f.units {
        FrameUnits::Rows => window_frame::FrameUnit::Rows,
        FrameUnits::Groups => window_frame::FrameUnit::Groups,
        FrameUnits::Range => {
            if is_numeric_offset(f.start) || is_numeric_offset(f.end) {
                return Err(InterpError::ValueBasedRangeFrame);
            }
            window_frame::FrameUnit::Range
        }
    };
    Ok(Some(window_frame::Frame {
        unit,
        start: map_bound(f.start),
        end: map_bound(f.end),
    }))
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

    /// A **float** single sort key must agree with the eager stable oracle too.
    ///
    /// A float key has no O(n) single-key path (neither the string builder nor the radix), so
    /// `top_k_indices_of` routes it through the quickselect. That path selects rather than sorts,
    /// and its within-morsel order is unstable — so this pins that `parallel_top_n` still yields
    /// the stable sort's exact top-k, which it does because it breaks ties by original
    /// `(morsel, row)`, not by the quickselect's output order. The bug this guards: with `-0.0`
    /// and `0.0` present, tie-breaking on candidate-array position (the old code) surfaced the
    /// `0.0` row where the stable sort surfaces the earlier `-0.0` row. Covers heavy ties, both
    /// zeros, NaN and nulls, ascending and descending, nulls first and last.
    #[test]
    fn parallel_top_n_float_key_matches_eager() {
        let n = 40_000usize;
        // Heavy ties, both zeros, NaN, and nulls — every case the fast paths refuse.
        let key: Vec<Option<f64>> = (0..n)
            .map(|i| match i % 101 {
                0 => None,
                1 => Some(f64::NAN),
                2 => Some(-0.0),
                3 => Some(0.0),
                _ => Some(((i * 13) % 200) as f64),
            })
            .collect();
        let payload: Vec<i64> = (0..n as i64).collect();
        let batch = RecordBatch::try_from_iter(vec![
            (
                "k",
                Arc::new(arrow::array::Float64Array::from(key)) as ArrayRef,
            ),
            ("p", Arc::new(Int64Array::from(payload)) as ArrayRef),
        ])
        .unwrap();
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
                for k in [5usize, 100, 20_000] {
                    let locals: Vec<RecordBatch> = parts
                        .iter()
                        .map(|b| sort_batch(b, &keys, Some(k)).unwrap())
                        .collect();
                    let merged = materialize(&locals).unwrap();
                    let eager = sort_batch(&merged, &keys, Some(k)).unwrap();
                    let late = parallel_top_n(&parts, &keys, k).unwrap();

                    assert_eq!(late.num_rows(), eager.num_rows(), "k={k} desc={descending}");
                    // The payload identifies the exact rows, so a tie-order divergence shows here.
                    let ci = eager.schema().index_of("p").unwrap();
                    let le = late
                        .column(ci)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap();
                    let ea = eager
                        .column(ci)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap();
                    for r in 0..ea.len() {
                        assert_eq!(
                            le.value(r),
                            ea.value(r),
                            "row {r} k={k} desc={descending} nf={nulls_first}"
                        );
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

    /// A single-key sort whose key type has no specialized stable path (boolean here, and a
    /// NaN-bearing float that the radix declines) MUST still be stable: rows equal on the key
    /// keep input order. Before the fix this branch called arrow's UNSTABLE `sort_to_indices`,
    /// so ties came back in an arbitrary, input-size-dependent order — making the serial
    /// oracle, the parallel sample-sort's per-range sorts, and the external-merge-sort runs
    /// disagree on the payload of tied rows (a seq != par != spill divergence). The payload is
    /// a distinct ascending id, so any tie-order scramble shows as a non-ascending run.
    #[test]
    fn single_key_fallback_is_stable_on_ties() {
        use arrow::array::BooleanArray;
        let n = 4096usize;

        // Boolean key: two big tie groups (false, then true).
        let bk: Vec<bool> = (0..n).map(|i| i % 2 == 1).collect();
        let bp: Vec<i64> = (0..n as i64).collect();
        let bb = RecordBatch::try_from_iter(vec![
            ("k", Arc::new(BooleanArray::from(bk)) as ArrayRef),
            ("p", Arc::new(Int64Array::from(bp)) as ArrayRef),
        ])
        .unwrap();
        // Float key with a NaN so the radix declines and the fallback path is taken; heavy
        // ties on 0.0 exercise the tie-break.
        let fk: Vec<f64> = (0..n)
            .map(|i| if i == 0 { f64::NAN } else { (i % 3) as f64 })
            .collect();
        let fp: Vec<i64> = (0..n as i64).collect();
        let fb = RecordBatch::try_from_iter(vec![
            ("k", Arc::new(Float64Array::from(fk)) as ArrayRef),
            ("p", Arc::new(Int64Array::from(fp)) as ArrayRef),
        ])
        .unwrap();

        for (batch, label) in [(&bb, "bool"), (&fb, "float-nan")] {
            let keys = vec![SortKey {
                expr: Expr::Col { name: "k".into() },
                descending: false,
                nulls_first: false,
            }];
            let out = sort_batch(batch, &keys, None).unwrap();
            let kc = out.column(0);
            let pc = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
            // A comparable per-row key token (null distinct from any value; NaN by bits) so a
            // run of equal keys can be identified without gathering.
            let tok = |i: usize| -> (u8, u64) {
                if kc.is_null(i) {
                    (0, 0)
                } else if let Some(b) = kc.as_any().downcast_ref::<BooleanArray>() {
                    (1, b.value(i) as u64)
                } else {
                    let f = kc.as_any().downcast_ref::<Float64Array>().unwrap();
                    (2, f.value(i).to_bits())
                }
            };
            // Within each run of equal keys the payload ids must be strictly ascending
            // (stable = input order preserved).
            for i in 1..out.num_rows() {
                if tok(i) == tok(i - 1) {
                    assert!(
                        pc.value(i) > pc.value(i - 1),
                        "{label}: tie order not stable at row {i}: {} !> {}",
                        pc.value(i),
                        pc.value(i - 1)
                    );
                }
            }
        }
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

    /// A `Null`-typed (all-null) sort key has no natural order — arrow's sort kernels reject
    /// it, which turned `ORDER BY <all-null col>` into a hard error on a path DuckDB runs
    /// fine. It must instead sort as all-equal: with a real secondary key present, the result
    /// is ordered by that key; with only the null key, the input order is preserved (stable).
    /// Covers both the top-N (`limit`) and full-sort code paths.
    #[test]
    fn null_typed_sort_key_orders_by_remaining_keys() {
        // `n` (Null type) is the leading key; `y` is a real Int64 tiebreak; `p` a distinct
        // payload proving stability. Rows deliberately out of `y` order on input.
        let ys = vec![3i64, 1, 2, 1, 3];
        let ps = vec![10i64, 11, 12, 13, 14];
        let batch = RecordBatch::try_from_iter(vec![
            (
                "n",
                Arc::new(arrow::array::NullArray::new(ys.len())) as ArrayRef,
            ),
            ("y", Arc::new(Int64Array::from(ys)) as ArrayRef),
            ("p", Arc::new(Int64Array::from(ps)) as ArrayRef),
        ])
        .unwrap();

        // ORDER BY n, y  → the null key contributes nothing; rows come out in `y` order,
        // ties on `y` (rows p=11,13) resolved to input order (11 before 13).
        let keys = vec![
            SortKey {
                expr: Expr::Col { name: "n".into() },
                descending: false,
                nulls_first: false,
            },
            SortKey {
                expr: Expr::Col { name: "y".into() },
                descending: false,
                nulls_first: false,
            },
        ];
        let out = sort_batch(&batch, &keys, None).expect("null-typed key must not error");
        let p = out.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(
            p.values(),
            &[11, 13, 12, 10, 14],
            "ordered by y, stable ties"
        );

        // Top-N (limit) path over the same keys: the first 3 by `y`.
        let topn = sort_batch(&batch, &keys, Some(3)).expect("null-typed top-N must not error");
        let p = topn
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(p.values(), &[11, 13, 12]);

        // ORDER BY n alone → all rows equal, input order preserved (stable no-op sort).
        let only_null = vec![SortKey {
            expr: Expr::Col { name: "n".into() },
            descending: true,
            nulls_first: true,
        }];
        let out = sort_batch(&batch, &only_null, None).expect("sole null key must not error");
        let p = out.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(p.values(), &[10, 11, 12, 13, 14], "input order preserved");
    }
}

#[cfg(test)]
mod window_frame_tests {
    use super::*;
    use bc_ir::{FrameBound, FrameUnits, WindowFrame};

    fn frame(units: FrameUnits, start: FrameBound, end: FrameBound) -> Option<WindowFrame> {
        Some(WindowFrame { units, start, end })
    }

    /// A value-based `RANGE` offset must be rejected, not approximated. Before this
    /// was an error it silently mapped to `None` — the peer-`RANGE` running aggregate
    /// — which is a *different frame* and therefore a wrong answer for any input whose
    /// order key is not already peer-shaped.
    #[test]
    fn numeric_range_offset_is_rejected_not_downgraded() {
        for (start, end) in [
            (FrameBound::Preceding { n: 2 }, FrameBound::CurrentRow),
            (FrameBound::CurrentRow, FrameBound::Following { n: 3 }),
            (
                FrameBound::Preceding { n: 1 },
                FrameBound::Following { n: 1 },
            ),
        ] {
            let got = map_frame(frame(FrameUnits::Range, start, end));
            assert!(
                matches!(got, Err(InterpError::ValueBasedRangeFrame)),
                "RANGE {start:?}..{end:?} must error, got {got:?}"
            );
        }
    }

    /// Peer-shaped `RANGE` bounds are exactly representable, so they still map.
    #[test]
    fn peer_range_bounds_still_map() {
        let got = map_frame(frame(
            FrameUnits::Range,
            FrameBound::UnboundedPreceding,
            FrameBound::CurrentRow,
        ))
        .expect("peer RANGE must not error")
        .expect("peer RANGE must produce a frame");
        assert_eq!(got.unit, window_frame::FrameUnit::Range);
    }

    /// `ROWS` and `GROUPS` count positions, so a numeric offset is exact for both and
    /// must keep working — the rejection is specific to value-based `RANGE`.
    #[test]
    fn rows_and_groups_offsets_are_unaffected() {
        for units in [FrameUnits::Rows, FrameUnits::Groups] {
            let got = map_frame(frame(
                units,
                FrameBound::Preceding { n: 2 },
                FrameBound::Following { n: 1 },
            ))
            .expect("numeric offset is exact for rows/groups")
            .expect("must produce a frame");
            assert!(matches!(got.start, window_frame::FrameBound::Preceding(2)));
        }
    }

    /// No frame at all stays `None` (the default running frame), and that is not an error.
    #[test]
    fn absent_frame_is_none() {
        assert!(map_frame(None)
            .expect("absent frame is not an error")
            .is_none());
    }
}

#[cfg(test)]
mod topn_bound_tests {
    use super::*;
    use arrow::array::Int64Array;
    use bc_expr::Expr;
    use std::sync::Arc;

    /// `clustered` puts each morsel's keys in its own ascending band, the way time-ordered or
    /// key-partitioned data arrives; otherwise every morsel draws from the whole range, which is
    /// the shape the bound provably cannot help.
    fn morsels(count: usize, rows: usize, clustered: bool) -> Vec<RecordBatch> {
        (0..count)
            .map(|m| {
                let k: Int64Array = (0..rows)
                    .map(|i| {
                        Some(if clustered {
                            (m * rows + i) as i64
                        } else {
                            ((m * rows + i).wrapping_mul(2_654_435_761) % 1_000_000_007) as i64
                        })
                    })
                    .collect();
                let payload: Int64Array = (0..rows).map(|i| Some(i as i64)).collect();
                RecordBatch::try_from_iter(vec![
                    ("k", Arc::new(k) as ArrayRef),
                    ("p", Arc::new(payload) as ArrayRef),
                ])
                .expect("morsel")
            })
            .collect()
    }

    fn keys() -> Vec<SortKey> {
        vec![SortKey {
            expr: Expr::Col { name: "k".into() },
            descending: false,
            nulls_first: false,
        }]
    }

    /// What the bound is worth, as the two costs it trades: selecting a morsel's local top-k
    /// against reading its key range. Ignored by default — it is a measurement, not a contract.
    ///
    /// ```text
    /// cargo test --release -p bc-interp topn_bound_tests -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "timing study for the top-N morsel skip; run with --release -- --ignored"]
    fn report_the_top_n_skip_saving() {
        const MORSELS: usize = 256;
        const ROWS: usize = 16_384;
        let keys = keys();

        for (shape, clustered) in [("clustered", true), ("random", false)] {
            let parts = morsels(MORSELS, ROWS, clustered);
            for k in [10_usize, 100, 1_000] {
                // Cost of the work a skip avoids: one morsel's selection, plus the narrow gather.
                let key_arrays: Vec<ArrayRef> = keys
                    .iter()
                    .map(|key| normalize_sort_key(key.expr.eval(&parts[0]).unwrap()))
                    .collect();
                let mut select_ms = f64::MAX;
                let mut range_ms = f64::MAX;
                for _ in 0..20 {
                    let t = std::time::Instant::now();
                    std::hint::black_box(top_k_indices_of(&key_arrays, &keys, ROWS, k).unwrap());
                    select_ms = select_ms.min(t.elapsed().as_secs_f64() * 1e3);
                    let t = std::time::Instant::now();
                    std::hint::black_box(bc_runtime::topn::i64_key_range(&key_arrays[0]));
                    range_ms = range_ms.min(t.elapsed().as_secs_f64() * 1e3);
                }

                // How many morsels the bound actually excludes, replaying the operator's sequence
                // single-threaded so the count is deterministic.
                let bound = bc_runtime::topn::TopNBound::new(false);
                let mut skipped = 0usize;
                for b in &parts {
                    let ka: Vec<ArrayRef> = keys
                        .iter()
                        .map(|key| normalize_sort_key(key.expr.eval(b).unwrap()))
                        .collect();
                    if let Some((min, max)) = bc_runtime::topn::i64_key_range(&ka[0]) {
                        if bound.excludes_range(min, max) {
                            skipped += 1;
                            continue;
                        }
                    }
                    let idx = top_k_indices_of(&ka, &keys, b.num_rows(), k).unwrap();
                    if idx.len() == k {
                        let cand = bc_runtime::gather::take_column(ka[0].as_ref(), &idx).unwrap();
                        if let Some(v) = bound.candidate_bound(&cand) {
                            bound.publish(v);
                        }
                    }
                }
                println!(
                "{shape:>9} k={k:>5}: select {select_ms:>7.3} ms/morsel, range {range_ms:>7.3} \
                 ms/morsel ({:>5.1}x cheaper), skipped {skipped}/{MORSELS}, gauge_off={}",
                select_ms / range_ms,
                bound.is_off()
            );
            }
        }
    }
}
