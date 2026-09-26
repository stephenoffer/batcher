//! The FFI entry point for streaming one source into the engine chunk by chunk.
//!
//! `execute_plan` takes every source resident, which is what forces the control plane to decode
//! a whole table before the engine may start (see `bc_interp::stream::chunked`). Here the
//! driving source is a Python iterator of batch lists instead. The engine pulls the next chunk
//! only between its own parallel steps, on the calling thread, reacquiring the GIL for exactly
//! as long as it takes to advance the iterator and take ownership of the batches — so a Python
//! producer that decodes on a background thread (the control plane's prefetching Parquet reader)
//! keeps decoding while the engine computes.

use arrow::array::RecordBatch;
use arrow_pyarrow::PyArrowType;
use pyo3::exceptions::PyStopIteration;
use pyo3::prelude::*;

use crate::normalize::{narrow_output, normalize_batch, rebase_nested_offsets};
use crate::{errors, prepare_exec, ExecSetup};

/// Whether `plan_json` can run with source `driving` streamed in chunks.
#[pyfunction]
pub(crate) fn plan_chunkable(plan_json: &str, driving: usize) -> PyResult<bool> {
    let plan = bc_ir::RelOp::from_json(plan_json).map_err(errors::ir_to_pyerr)?;
    Ok(bc_interp::chunkable(&plan, driving))
}

/// Execute `plan_json` with `sources[driving]` read from `chunks`, an iterator of batch lists.
///
/// `sources[driving]` carries only the driving relation's schema (a zero-row batch), and
/// `memory_budget` bounds what the path may hold (the tighter of it and the engine config's).
/// Raises `MemoryBudgetExceededError` when that is exceeded, which the caller
/// answers by taking an out-of-core path; the rows returned otherwise are exactly those
/// `execute_plan` returns over the concatenated chunks.
#[pyfunction]
#[pyo3(signature = (plan_json, sources, driving, chunks, engine_config="", query_id=None, memory_budget=0))]
pub(crate) fn execute_plan_chunked(
    py: Python<'_>,
    plan_json: &str,
    sources: Vec<Vec<PyArrowType<RecordBatch>>>,
    driving: usize,
    chunks: Bound<'_, PyAny>,
    engine_config: &str,
    query_id: Option<&str>,
    memory_budget: usize,
) -> PyResult<Vec<PyArrowType<RecordBatch>>> {
    let ExecSetup {
        plan,
        sources,
        opts,
        narrow,
        budget,
        ..
    } = prepare_exec(plan_json, sources, engine_config, query_id)?;
    // What the chunked path holds — aggregate state, or a spine's collected output — must stay
    // inside an envelope even when the engine config names none, because the chunked path's only
    // fallback is the caller's out-of-core one: over budget, it must say so rather than grow.
    let budget = match (budget, memory_budget) {
        (0, m) => m,
        (b, 0) => b,
        (b, m) => b.min(m),
    };
    let iterator: Py<PyAny> = chunks.try_iter()?.into_any().unbind();
    // `auto_width` caps the width by the resident sources' size, and the driving one is not
    // resident here: honour an explicit width, and otherwise give the chunks the operator cores
    // a resident run over them would get.
    let workers = if opts.parallelism > 0 {
        opts.parallelism
    } else {
        bc_arrow::operator_cores().max(1)
    };
    // A Python exception raised by the chunk iterator (a strict-schema `SchemaError`, say) is
    // kept here and re-raised as itself: the engine only sees that the producer failed, and the
    // caller must get the same exception the resident read would have raised.
    let raised: std::sync::Mutex<Option<PyErr>> = std::sync::Mutex::new(None);
    let out = py.detach(|| {
        let mut next = || {
            Python::attach(|py| next_chunk(py, &iterator))
                .map_err(|e| {
                    let message = e.to_string();
                    if let Ok(mut slot) = raised.lock() {
                        slot.get_or_insert(e);
                    }
                    bc_interp::InterpError::ChunkSource(message)
                })
                .transpose()
        };
        bc_interp::execute_chunked(&plan, &sources, driving, &mut next, workers, budget, &opts)
    });
    if let Some(err) = raised.into_inner().ok().flatten() {
        return Err(err);
    }
    let out = out.map_err(errors::interp_to_pyerr)?;
    let out = bc_interp::coalesce_small_batches(rebase_nested_offsets(narrow_output(out, &narrow)));
    Ok(out.into_iter().map(PyArrowType).collect())
}

/// The iterator's next chunk, normalized exactly as `prepare_exec` normalizes a resident source.
fn next_chunk(py: Python<'_>, iterator: &Py<PyAny>) -> PyResult<Option<Vec<RecordBatch>>> {
    let item = match iterator.bind(py).call_method0("__next__") {
        Ok(item) => item,
        Err(e) if e.is_instance_of::<PyStopIteration>(py) => return Ok(None),
        Err(e) => return Err(e),
    };
    let batches: Vec<PyArrowType<RecordBatch>> = item.extract()?;
    batches
        .iter()
        .map(|b| normalize_batch(&b.0))
        .collect::<PyResult<Vec<_>>>()
        .map(Some)
}

/// The row groups of a list of Parquet files, read one at a time by the engine's own workers.
///
/// Each read decodes on the worker that asked for it (`bc_io::read_parquet_row_group`), which then
/// computes over the rows and frees them on the same thread. Reading ahead onto the I/O pool was
/// measured and dropped: with the decode already on every worker it only added threads competing
/// for the same cores (TPC-H sf10 q6, 158 ms without it, 182 ms with).
struct ParquetUnits<'a> {
    uris: &'a [String],
    columns: Option<&'a [String]>,
    predicate: Option<&'a str>,
    batch_size: usize,
    units: Vec<(usize, usize)>,
}

impl bc_interp::UnitSource for ParquetUnits<'_> {
    fn units(&self) -> usize {
        self.units.len()
    }

    fn read(&self, unit: usize) -> Result<Vec<RecordBatch>, bc_interp::InterpError> {
        let source = |e: String| bc_interp::InterpError::ChunkSource(e);
        let (file, rg) = self.units[unit];
        bc_io::read_parquet_row_group(
            &self.uris[file],
            rg,
            self.columns,
            self.batch_size,
            self.predicate,
        )
        .map_err(|e| source(e.to_string()))?
        .iter()
        .map(|b| normalize_batch(b).map_err(|e| source(e.to_string())))
        .collect()
    }
}

/// Execute `plan_json` with `sources[driving]` read from Parquet `uris` by the engine's workers.
///
/// Each worker decodes a contiguous range of the files' row groups one at a time and pushes each
/// straight through its pipeline (`bc_interp::execute_units`), so decoding and computing overlap
/// across the pool and no worker holds more than one decoded row group of the driving relation.
/// `columns` and `predicate` are the scan's pushed projection and native predicate, exactly as
/// the resident read takes them; `sources[driving]` is the zero-row schema carrier. Budget and
/// errors as [`execute_plan_chunked`]; a plan that cannot run this way raises the engine's
/// not-chunkable error, and the caller reads the source resident instead.
///
/// Returns the rows and the per-operator metrics document `execute_plan_metered` returns: every
/// driving row passes through exactly once here, so the counts are the query's own and the
/// learning loop records them (`bc_interp::execute_units_metered`).
#[pyfunction]
#[pyo3(signature = (plan_json, sources, driving, uris, columns=None, predicate=None, batch_size=65536, engine_config="", query_id=None, memory_budget=0))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_plan_parquet(
    py: Python<'_>,
    plan_json: &str,
    sources: Vec<Vec<PyArrowType<RecordBatch>>>,
    driving: usize,
    uris: Vec<String>,
    columns: Option<Vec<String>>,
    predicate: Option<String>,
    batch_size: usize,
    engine_config: &str,
    query_id: Option<&str>,
    memory_budget: usize,
) -> PyResult<(Vec<PyArrowType<RecordBatch>>, String)> {
    let ExecSetup {
        plan,
        sources,
        opts,
        narrow,
        budget,
        ..
    } = prepare_exec(plan_json, sources, engine_config, query_id)?;
    let budget = match (budget, memory_budget) {
        (0, m) => m,
        (b, 0) => b,
        (b, m) => b.min(m),
    };
    let query_watch = bc_interp::QueryStopwatch::start();
    // Every usable core, not `operator_cores`' SMT-discounted width: each worker here spends
    // most of its time decoding Parquet (Snappy, then value copies), which is the work `bc-io`
    // already sizes its own pool by `usable_cores` for. At the operator width a 48-thread host
    // decoded on 30 threads and left the rest idle. An explicit width still wins, which is how
    // a concurrency grant or a user's `parallelism` bounds it.
    let workers = if opts.parallelism > 0 {
        opts.parallelism
    } else {
        bc_arrow::usable_cores().max(1)
    };
    let out = py.detach(|| {
        let units = bc_io::parquet_row_groups(&uris)
            .map_err(|e| bc_interp::InterpError::ChunkSource(e.to_string()))?
            .into_iter()
            .map(|(file, rg, _)| (file, rg))
            .collect();
        let src = ParquetUnits {
            uris: &uris,
            columns: columns.as_deref(),
            predicate: predicate.as_deref(),
            batch_size: batch_size.max(1),
            units,
        };
        bc_interp::execute_units_metered(&plan, &sources, driving, &src, workers, budget, &opts)
    });
    let (out, metrics) = out.map_err(errors::interp_to_pyerr)?;
    let metrics = metrics.with_query(query_watch);
    let out = bc_interp::coalesce_small_batches(rebase_nested_offsets(narrow_output(out, &narrow)));
    Ok((
        out.into_iter().map(PyArrowType).collect(),
        metrics.to_json(),
    ))
}
