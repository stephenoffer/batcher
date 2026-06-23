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

use crate::error::InterpError;
use crate::ops;

/// Map step: aggregate one partition into partial state.
///
/// The output batch is `[group_key_columns..., state_columns...]`; state column
/// names are synthetic (`__s{agg}_{col}`) and decoded by [`combine_finalize`]
/// using the aggregate list (only `mean` has two state columns).
pub fn partial_aggregate(
    group_keys: &[ProjectionItem],
    aggregates: &[AggregateItem],
    batches: &[RecordBatch],
) -> Result<RecordBatch, InterpError> {
    let combined = ops::materialize(batches).map_err(|_| InterpError::EmptyAggregateInput)?;
    let partial = ops::eval_partial(&combined, group_keys, aggregates)?;
    partial_to_batch(group_keys, &partial)
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
fn batches_to_partials(
    n_keys: usize,
    widths: &[usize],
    partial_batches: &[RecordBatch],
) -> Vec<agg::Partial> {
    let mut partials = Vec::with_capacity(partial_batches.len());
    for batch in partial_batches {
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
    partials
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
    let partials = batches_to_partials(group_keys.len(), &widths, partial_batches);
    if partials.is_empty() {
        return Err(InterpError::EmptyAggregateInput);
    }
    let funcs = ops::agg_funcs(aggregates);
    let merged = agg::combine(&partials, &funcs)?;
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
    let partials = batches_to_partials(group_keys.len(), &widths, partial_batches);
    if partials.is_empty() {
        return Err(InterpError::EmptyAggregateInput);
    }

    let funcs = ops::agg_funcs(aggregates);
    let merged = agg::combine(&partials, &funcs)?;
    let agg_cols = agg::finalize(&funcs, &merged)?;
    ops::build_agg_batch(group_keys, aggregates, &merged.group_columns, &agg_cols)
}

/// Hash-shuffle `batches` into `num_partitions` buckets by the given key columns.
/// Returns one (single-batch) relation per bucket — the unit a reducer consumes.
pub fn partition_batches(
    batches: &[RecordBatch],
    key_indices: &[usize],
    num_partitions: usize,
) -> Result<Vec<Vec<RecordBatch>>, InterpError> {
    let combined = ops::materialize(batches)?;
    let parts = shuffle::partition_by_keys(&combined, key_indices, num_partitions)?;
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
    let parts = shuffle::salted_partition_by_keys(
        &combined,
        key_indices,
        num_partitions,
        hot_keys,
        salt_count,
        replicate,
    )?;
    Ok(parts.into_iter().map(|b| vec![b]).collect())
}
