//! Join per-batch primitives: equi (`join_batches`) and ASOF (`asof_join_batches`).
//! Split out of `ops` to keep that module under the size limit. Both materialize
//! their (co-partitioned) sides, compute index pairs via `bc_runtime::join`, and
//! gather the planner-specified output columns through one shared assembler — so the
//! equi and ASOF joins cannot drift in how they build their output.

use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch};
use arrow::compute::take;
use arrow::datatypes::{Field, Schema};
use bc_ir::{JoinOutputCol, JoinSide, JoinStrategy, JoinType};
use bc_runtime::join::{self, JoinType as RtJoinType};

use crate::error::InterpError;

/// Join two already-materialized (and, in the parallel path, co-partitioned)
/// batches into the planner-specified output columns.
pub(crate) fn join_batches(
    left: &RecordBatch,
    right: &RecordBatch,
    left_keys: &[String],
    right_keys: &[String],
    join_type: JoinType,
    output: &[JoinOutputCol],
    strategy: JoinStrategy,
) -> Result<RecordBatch, InterpError> {
    let left_key_cols = columns_by_name(left, left_keys)?;
    let right_key_cols = columns_by_name(right, right_keys)?;
    let rt = map_join_type(join_type);
    // The physical index builder. Broadcast still builds a hash table per call
    // (its "no shuffle" win is the executor's, not this primitive's); SortMerge
    // sorts both sides and merges. All produce the same relation.
    let idx = match strategy {
        JoinStrategy::SortMerge => {
            join::sort_merge_join_indices(&left_key_cols, &right_key_cols, rt)?
        }
        _ => join::hash_join_indices(&left_key_cols, &right_key_cols, rt)?,
    };
    gather_join_output(left, right, &idx, output)
}

/// ASOF (nearest-match) join: each left row matched to the right row whose `on` key
/// is nearest in `direction` within its `by` group. A breaker (both sides fully
/// materialized), left-style (every left row emitted; unmatched → null right cols).
#[allow(clippy::too_many_arguments)]
pub(crate) fn asof_join_batches(
    left: &RecordBatch,
    right: &RecordBatch,
    left_on: &str,
    right_on: &str,
    left_by: &[String],
    right_by: &[String],
    backward: bool,
    output: &[JoinOutputCol],
) -> Result<RecordBatch, InterpError> {
    let left_on_col = left
        .column_by_name(left_on)
        .ok_or_else(|| InterpError::UnknownJoinColumn(left_on.to_string()))?
        .clone();
    let right_on_col = right
        .column_by_name(right_on)
        .ok_or_else(|| InterpError::UnknownJoinColumn(right_on.to_string()))?
        .clone();
    let left_by_cols = columns_by_name(left, left_by)?;
    let right_by_cols = columns_by_name(right, right_by)?;
    let idx = join::asof_join_indices(
        &left_on_col,
        &right_on_col,
        &left_by_cols,
        &right_by_cols,
        backward,
    )?;
    gather_join_output(left, right, &idx, output)
}

/// Build a join's output batch by gathering each output column from its side with
/// the computed indices (`take` yields null for a null index). Shared by the equi
/// and ASOF joins so their output assembly cannot drift.
fn gather_join_output(
    left: &RecordBatch,
    right: &RecordBatch,
    idx: &join::JoinIndices,
    output: &[JoinOutputCol],
) -> Result<RecordBatch, InterpError> {
    let mut fields = Vec::with_capacity(output.len());
    let mut columns = Vec::with_capacity(output.len());
    for col in output {
        let (batch, indices) = match col.side {
            JoinSide::Left => (left, &idx.left),
            JoinSide::Right => (right, &idx.right),
        };
        let source = batch
            .column_by_name(&col.name)
            .ok_or_else(|| InterpError::UnknownJoinColumn(col.name.clone()))?;
        let gathered = take(source.as_ref(), indices, None)?;
        fields.push(Field::new(&col.alias, gathered.data_type().clone(), true));
        columns.push(gathered);
    }
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

pub(crate) fn columns_by_name(
    batch: &RecordBatch,
    names: &[String],
) -> Result<Vec<ArrayRef>, InterpError> {
    names
        .iter()
        .map(|n| {
            batch
                .column_by_name(n)
                .cloned()
                .ok_or_else(|| InterpError::UnknownJoinColumn(n.clone()))
        })
        .collect()
}

/// Indices of the named key columns within a batch's schema.
pub(crate) fn key_indices(
    batch: &RecordBatch,
    names: &[String],
) -> Result<Vec<usize>, InterpError> {
    names
        .iter()
        .map(|n| {
            batch
                .schema()
                .index_of(n)
                .map_err(|_| InterpError::UnknownJoinColumn(n.clone()))
        })
        .collect()
}

pub(crate) fn map_join_type(t: JoinType) -> RtJoinType {
    match t {
        JoinType::Inner => RtJoinType::Inner,
        JoinType::Left => RtJoinType::Left,
        JoinType::Right => RtJoinType::Right,
        JoinType::Full => RtJoinType::Full,
        JoinType::Semi => RtJoinType::Semi,
        JoinType::Anti => RtJoinType::Anti,
    }
}
