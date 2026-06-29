//! Distributed-execution primitives.
//!
//! These are the building blocks the (Python) distributed orchestrator composes
//! across Ray workers. They are exactly the mergeable pieces proven in
//! `bc-runtime`, surfaced at a granularity the orchestrator can map over
//! partitions:
//!
//! * [`partial_aggregate`] — a map task runs this on its partition, emitting
//!   *partial state* (group keys + per-aggregate state columns) as one batch.
//! * [`partition_batches`] — hash-shuffle a batch into one bucket per reducer.
//! * [`combine_finalize`] — a reduce task merges the partial states routed to it
//!   and finalizes them into output rows.
//!
//! `combine_finalize(partition(partial(pₖ)))` over all partitions equals a
//! single-node aggregation — the same property the `bc-runtime` tests assert,
//! now spanning machines.

use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch};
use arrow::datatypes::{Field, Schema};
use bc_ir::{AggregateItem, ProjectionItem};
use bc_runtime::{agg, shuffle};
use rayon::prelude::*;

use crate::error::InterpError;
use crate::ops;

/// Map step: aggregate one partition into partial state.
///
/// The output batch is `[group_key_columns..., state_columns...]`; state column
/// names are synthetic (`__s{agg}_{col}`) and decoded by [`combine_finalize`]
/// using the aggregate list (only `mean` has two state columns).
///
/// Parallel across the morsels of the input (rayon): a distributed map worker folds
/// tens of millions of rows here, and a single-threaded partial would pin it to one
/// core while the read (now ~16-way concurrent) finishes in a fraction of the time —
/// leaving the fold the whole bottleneck. Partial-aggregate per morsel and `combine`
/// (the same mergeable path the parallel executor uses): the combine of per-morsel
/// partials equals one partial over the whole input, so the result is bit-identical to
/// the sequential fold — only the core count changes. A single morsel stays sequential.
pub fn partial_aggregate(
    group_keys: &[ProjectionItem],
    aggregates: &[AggregateItem],
    batches: &[RecordBatch],
) -> Result<RecordBatch, InterpError> {
    // The input is already morsel-sized (the map prefix's output), so partial-aggregate
    // each batch in parallel and `combine` — no `materialize` concat (it would serialize
    // the whole partition through one core, defeating the point). One batch stays
    // sequential. Runs on a dedicated pool sized to the worker's cores: Ray actors can
    // leave the *global* rayon pool sized to 1 (it is built before the cgroup affinity
    // lands), so `par_iter` on it would run single-threaded — the explicit pool is what
    // actually spreads the fold across all cores.
    let non_empty: Vec<&RecordBatch> = batches.iter().filter(|b| b.num_rows() > 0).collect();
    if non_empty.is_empty() {
        let combined = ops::materialize(batches).map_err(|_| InterpError::EmptyAggregateInput)?;
        let partial = ops::eval_partial(&combined, group_keys, aggregates)?;
        return partial_to_batch(group_keys, &partial);
    }
    if non_empty.len() == 1 {
        let partial = ops::eval_partial(non_empty[0], group_keys, aggregates)?;
        return partial_to_batch(group_keys, &partial);
    }
    let funcs = ops::agg_funcs(aggregates);
    let agg_jit = ops::compile_agg(group_keys, aggregates, non_empty[0]);
    // Share the executor's width-sized pool (NOT rayon's global pool, which a Ray worker
    // leaves at 1 thread — see `par::execute_parallel_with_metrics`), so the fold spreads
    // across every core. `available_parallelism` reads the actor's applied CPU affinity.
    let width = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1);
    let partials: Vec<agg::Partial> = crate::par::pool_for(width)?.install(|| {
        non_empty
            .par_iter()
            .map(|b| ops::eval_partial_jit(b, group_keys, aggregates, &agg_jit))
            .collect::<Result<_, InterpError>>()
    })?;
    let merged = agg::combine(&partials, &funcs)?;
    partial_to_batch(group_keys, &merged)
}

/// Per-aggregate partial-state column count (mean keeps sum+count; var/stddev keep
/// count+mean+M2; everything else a single accumulator).
fn agg_widths(aggregates: &[AggregateItem]) -> Vec<usize> {
    // Reuse the runtime's `state_arity` (the single source of truth) rather than a
    // duplicate table here — so a new multi-column aggregate (e.g. arg_min/arg_max)
    // works in the distributed path automatically.
    ops::agg_funcs(aggregates)
        .iter()
        .map(|f| f.state_arity())
        .collect()
}

/// Serialize a `Partial` into the wire batch `partial_aggregate` emits:
/// `[group_key_columns..., state_columns...]` with synthetic state names.
fn partial_to_batch(
    group_keys: &[ProjectionItem],
    partial: &agg::Partial,
) -> Result<RecordBatch, InterpError> {
    let mut fields = Vec::new();
    let mut columns = Vec::new();
    for (k, c) in group_keys.iter().zip(&partial.group_columns) {
        fields.push(Field::new(&k.alias, c.data_type().clone(), true));
        columns.push(c.clone());
    }
    for (a, state) in partial.states.iter().enumerate() {
        for (c, col) in state.iter().enumerate() {
            fields.push(Field::new(
                format!("__s{a}_{c}"),
                col.data_type().clone(),
                true,
            ));
            columns.push(col.clone());
        }
    }
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

/// Decode partial-state batches back into `Partial`s, splitting the synthetic
/// state columns by each aggregate's width.
///
/// The batches arrive from other Ray workers, so their column count is validated
/// against the wire format (`n_keys + Σ widths`) before any column is indexed: a
/// version-skewed or corrupt partial yields a typed [`InterpError::MalformedPartial`]
/// the orchestrator can treat as a failed task (recompute) rather than panicking the
/// reducer on an out-of-bounds access.
fn batches_to_partials(
    n_keys: usize,
    widths: &[usize],
    partial_batches: &[RecordBatch],
) -> Result<Vec<agg::Partial>, InterpError> {
    let state: usize = widths.iter().sum();
    let expected = n_keys + state;
    let mut partials = Vec::with_capacity(partial_batches.len());
    for batch in partial_batches {
        if batch.num_columns() != expected {
            return Err(InterpError::MalformedPartial {
                expected,
                n_keys,
                state,
                got: batch.num_columns(),
            });
        }
        let group_columns: Vec<ArrayRef> = (0..n_keys).map(|i| batch.column(i).clone()).collect();
        let mut states = Vec::with_capacity(widths.len());
        let mut off = n_keys;
        for &w in widths {
            states.push((0..w).map(|c| batch.column(off + c).clone()).collect());
            off += w;
        }
        partials.push(agg::Partial {
            group_columns,
            states,
        });
    }
    Ok(partials)
}

/// Combine step (no finalize): merge partial-state batches into a single partial
/// batch in the *same* wire format. This lets a streaming/incremental driver keep
/// one running state, bounded by the number of groups, instead of accumulating
/// every micro-batch's partials before a final `combine_finalize`.
pub fn combine(
    group_keys: &[ProjectionItem],
    aggregates: &[AggregateItem],
    partial_batches: &[RecordBatch],
) -> Result<RecordBatch, InterpError> {
    let widths = agg_widths(aggregates);
    let partials = batches_to_partials(group_keys.len(), &widths, partial_batches)?;
    if partials.is_empty() {
        return Err(InterpError::EmptyAggregateInput);
    }
    let funcs = ops::agg_funcs(aggregates);
    let merged = in_worker_pool(|| agg::combine(&partials, &funcs))??;
    partial_to_batch(group_keys, &merged)
}

/// Reduce step: merge the partial-state batches routed to one reducer and
/// finalize them into the output schema (group aliases + aggregate aliases).
pub fn combine_finalize(
    group_keys: &[ProjectionItem],
    aggregates: &[AggregateItem],
    partial_batches: &[RecordBatch],
) -> Result<RecordBatch, InterpError> {
    let widths = agg_widths(aggregates);
    let partials = batches_to_partials(group_keys.len(), &widths, partial_batches)?;
    if partials.is_empty() {
        return Err(InterpError::EmptyAggregateInput);
    }

    let funcs = ops::agg_funcs(aggregates);
    let merged = in_worker_pool(|| agg::combine(&partials, &funcs))??;
    let agg_cols = agg::finalize(&funcs, &merged)?;
    ops::build_agg_batch(group_keys, aggregates, &merged.group_columns, &agg_cols)
}

/// Run a rayon-parallel data-plane step inside the worker's **width-sized** pool rather
/// than rayon's global pool. A Ray map/reduce actor leaves the global pool sized to one
/// thread — it is built before the actor's cgroup CPU affinity lands — so the
/// rayon-parallel kernels in `bc_runtime` (the high-cardinality combine's regroup/merge,
/// the shuffle's hash + scatter) would pin a worker that processes millions of rows to a
/// single core. The width-sized pool (the same fix `partial_aggregate` applies to the map
/// fold) spreads them across every core the actor owns. Result-identical; scheduling only.
fn in_worker_pool<T: Send>(f: impl FnOnce() -> T + Send) -> Result<T, InterpError> {
    let width = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1);
    Ok(crate::par::pool_for(width)?.install(f))
}

/// Hash-shuffle `batches` into `num_partitions` buckets by the given key columns.
/// Returns one (single-batch) relation per bucket — the unit a reducer consumes.
pub fn partition_batches(
    batches: &[RecordBatch],
    key_indices: &[usize],
    num_partitions: usize,
) -> Result<Vec<Vec<RecordBatch>>, InterpError> {
    let combined = ops::materialize(batches)?;
    let parts =
        in_worker_pool(|| shuffle::partition_by_keys(&combined, key_indices, num_partitions))??;
    Ok(parts.into_iter().map(|b| vec![b]).collect())
}

/// Range-shuffle `batches` into `n_buckets` globally-ordered buckets by the leading
/// sort key at `key_index` and the ascending `boundaries` — the distributed-sort
/// counterpart of the hash [`partition_batches`]. Bucket order is the sort order, so a
/// reducer sorts its bucket and the driver concatenates buckets (reversed when
/// `descending`) with no merge. Returns one (single-batch) relation per bucket.
pub fn range_partition_batches(
    batches: &[RecordBatch],
    key_index: usize,
    boundaries: &[f64],
    n_buckets: usize,
    nulls_first: bool,
    descending: bool,
) -> Result<Vec<Vec<RecordBatch>>, InterpError> {
    let combined = ops::materialize(batches)?;
    let parts = in_worker_pool(|| {
        shuffle::range_partition_by_key(
            &combined,
            key_index,
            boundaries,
            n_buckets,
            nulls_first,
            descending,
        )
    })??;
    Ok(parts.into_iter().map(|b| vec![b]).collect())
}

/// Skew-aware shuffle for a single-key distributed join: like [`partition_batches`],
/// but a *hot* key's rows are salted across reducers instead of overloading one.
/// `replicate=false` (probe side) fans each hot row to one salted bucket;
/// `replicate=true` (build side) replicates each hot row to all salted buckets, so
/// every salted probe bucket can match it. Cold keys hash exactly as the unsalted
/// shuffle, so the joined relation is unchanged — only the hot key's work moves off
/// a single reducer. See [`shuffle::salted_partition_by_keys`].
pub fn salted_partition_batches(
    batches: &[RecordBatch],
    key_indices: &[usize],
    num_partitions: usize,
    hot_keys: &std::collections::HashSet<String>,
    salt_count: u32,
    replicate: bool,
) -> Result<Vec<Vec<RecordBatch>>, InterpError> {
    let combined = ops::materialize(batches)?;
    let parts = in_worker_pool(|| {
        shuffle::salted_partition_by_keys(
            &combined,
            key_indices,
            num_partitions,
            hot_keys,
            salt_count,
            replicate,
        )
    })??;
    Ok(parts.into_iter().map(|b| vec![b]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use arrow::datatypes::DataType;

    fn batch(n_cols: usize) -> RecordBatch {
        let fields: Vec<Field> = (0..n_cols)
            .map(|i| Field::new(format!("c{i}"), DataType::Int64, true))
            .collect();
        let cols: Vec<ArrayRef> = (0..n_cols)
            .map(|_| Arc::new(Int64Array::from(vec![1i64])) as ArrayRef)
            .collect();
        RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap()
    }

    /// A version-skewed/corrupt partial (wrong column count) from another worker is
    /// rejected with a typed error before any column is indexed — never an
    /// out-of-bounds panic on the reducer.
    #[test]
    fn malformed_partial_is_typed_error_not_panic() {
        // Expect n_keys (1) + widths (2 + 1 = 3) = 4 columns; give 2.
        match batches_to_partials(1, &[2, 1], &[batch(2)]) {
            Err(InterpError::MalformedPartial {
                expected,
                n_keys,
                state,
                got,
            }) => assert_eq!((expected, n_keys, state, got), (4, 1, 3, 2)),
            _ => panic!("expected Err(MalformedPartial)"),
        }
    }

    /// A correctly-shaped batch decodes into one partial with the right arity split.
    #[test]
    fn well_formed_partial_decodes() {
        let partials = batches_to_partials(1, &[2, 1], &[batch(4)]).unwrap();
        assert_eq!(partials.len(), 1);
        assert_eq!(partials[0].group_columns.len(), 1);
        let widths: Vec<usize> = partials[0].states.iter().map(|s| s.len()).collect();
        assert_eq!(widths, vec![2, 1]);
    }
}
