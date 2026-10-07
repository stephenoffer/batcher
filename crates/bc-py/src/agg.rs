//! The distributed aggregate's map and reduce steps: `partial_aggregate`, `combine` and
//! `combine_finalize`, the mergeable folds the shuffle composes across workers.
//!
//! All three run with the GIL released. A fleet actor runs several methods at once on its
//! own threads, and these are what it folds a large partition with; holding the GIL
//! through the fold stalled every other actor thread -- a Flight gather, a split read, a
//! heartbeat -- for its whole duration, serializing the concurrency the actor was granted.

use arrow::array::RecordBatch;
use arrow_pyarrow::PyArrowType;
use pyo3::prelude::*;

use crate::{parse_aggregates, parse_group_keys, rebase_batch, to_pyerr, unwrap_batches};

/// One of the mergeable aggregate folds; all three take the same inputs.
type AggFold = fn(
    &[bc_ir::ProjectionItem],
    &[bc_ir::AggregateItem],
    &[RecordBatch],
) -> Result<RecordBatch, bc_interp::InterpError>;

/// Run `fold` with the GIL released, as `execute_plan_aggregated` does. Nothing in the
/// fold touches Python; the engine error is mapped to a Python exception after it.
fn fold_detached(
    py: Python<'_>,
    group_keys_json: &str,
    aggregates_json: &str,
    batches: Vec<PyArrowType<RecordBatch>>,
    fold: AggFold,
) -> PyResult<PyArrowType<RecordBatch>> {
    let group_keys = parse_group_keys(group_keys_json)?;
    let aggregates = parse_aggregates(aggregates_json)?;
    let out = py.detach(|| {
        let batches = unwrap_batches(batches)?;
        Ok::<_, PyErr>(fold(&group_keys, &aggregates, &batches).map_err(|e| e.to_string()))
    })?;
    Ok(PyArrowType(rebase_batch(out.map_err(to_pyerr)?)))
}

/// Distributed map step: aggregate one partition into partial state.
#[pyfunction]
pub(crate) fn partial_aggregate(
    py: Python<'_>,
    group_keys_json: &str,
    aggregates_json: &str,
    batches: Vec<PyArrowType<RecordBatch>>,
) -> PyResult<PyArrowType<RecordBatch>> {
    let fold: AggFold = bc_interp::dist::partial_aggregate;
    fold_detached(py, group_keys_json, aggregates_json, batches, fold)
}

/// Distributed reduce step: merge partial-state batches and finalize.
#[pyfunction]
pub(crate) fn combine_finalize(
    py: Python<'_>,
    group_keys_json: &str,
    aggregates_json: &str,
    partials: Vec<PyArrowType<RecordBatch>>,
) -> PyResult<PyArrowType<RecordBatch>> {
    let fold: AggFold = bc_interp::dist::combine_finalize;
    fold_detached(py, group_keys_json, aggregates_json, partials, fold)
}

/// Combine step WITHOUT finalize: merge partial-state batches into a single partial
/// batch (same wire format), so a streaming driver can keep one running state across
/// micro-batches, bounded by the number of groups, and `combine_finalize` once.
#[pyfunction]
pub(crate) fn combine(
    py: Python<'_>,
    group_keys_json: &str,
    aggregates_json: &str,
    partials: Vec<PyArrowType<RecordBatch>>,
) -> PyResult<PyArrowType<RecordBatch>> {
    let fold: AggFold = bc_interp::dist::combine;
    fold_detached(py, group_keys_json, aggregates_json, partials, fold)
}
