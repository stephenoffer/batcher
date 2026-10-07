//! The plan's other Parquet scans, read by the engine before it runs instead of by the control
//! plane.
//!
//! [`super::execute_plan_parquet`] streams the driving scan through the workers, but every other
//! scan the plan reads (the build sides) used to arrive resident from Python: decoded on the
//! control plane's reader, exported batch by batch to `pyarrow`, then imported back. On TPC-H at
//! sf100 that round trip was the slowest stretch of several queries on 64 cores -- q18 spent
//! ~300 ms after its reads had finished with nearly every core idle, exporting, collecting
//! garbage and freeing on the interpreter's threads, before the engine started.
//!
//! Here those scans are read with the same footer pruning and the same per-file row order as the
//! control plane's multi-file read, one row group per task across the query's pool, normalized
//! as `prepare_exec` normalizes a resident source, and handed to the plan in place of the schema
//! carriers the control plane sent for them.

use arrow::array::RecordBatch;
use rayon::prelude::*;

use crate::normalize::normalize_batch;

/// One scan to read here: its source id, files, pushed projection and native predicate, and the
/// rows to decode at once -- what the control plane's `ParquetUnitRead` carries.
pub(crate) type ResidentRead = (
    usize,
    Vec<String>,
    Option<Vec<String>>,
    Option<String>,
    usize,
);

/// Read every scan in `reads` across a pool of `workers`, and put each into `sources` in place
/// of its carrier.
///
/// The carrier is the zero-row batch of the scan's declared schema, which the control plane
/// checked every file matches (`parquet.units`), so `prepare_exec` already recorded the columns'
/// pre-normalization widths from it. A scan whose every row group the footers prune keeps its
/// carrier, which is what the control plane's read returns for the same case.
pub(crate) fn read_into(
    reads: &[ResidentRead],
    sources: &mut [Vec<RecordBatch>],
    workers: usize,
) -> Result<(), bc_interp::InterpError> {
    let source = |e: String| bc_interp::InterpError::ChunkSource(e);
    // Every scan's surviving row groups, as one flat list of tasks in (scan, file, row group)
    // order, so a small scan's row groups do not wait behind a large one's.
    let mut tasks: Vec<(usize, usize, usize)> = Vec::new();
    for (k, (_, uris, _, predicate, _)) in reads.iter().enumerate() {
        let groups = bc_io::parquet_row_groups_surviving(uris, predicate.as_deref())
            .map_err(|e| source(e.to_string()))?;
        tasks.extend(groups.into_iter().map(|(file, rg, _)| (k, file, rg)));
    }
    let decoded: Vec<Vec<RecordBatch>> = bc_interp::install_on_pool(workers, || {
        tasks
            .par_iter()
            .map(|&(k, file, rg)| {
                let (_, uris, columns, predicate, batch_size) = &reads[k];
                bc_io::read_parquet_row_group(
                    &uris[file],
                    rg,
                    columns.as_deref(),
                    (*batch_size).max(1),
                    predicate.as_deref(),
                )
                .map_err(|e| source(e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()
    })??;
    let mut per_scan: Vec<Vec<RecordBatch>> = vec![Vec::new(); reads.len()];
    for (&(k, _, _), batches) in tasks.iter().zip(decoded) {
        per_scan[k].extend(batches.into_iter().filter(|b| b.num_rows() > 0));
    }
    let normalized: Vec<Vec<RecordBatch>> = bc_interp::install_on_pool(workers, || {
        per_scan
            .par_iter()
            .map(|batches| batches.par_iter().map(normalize_batch).collect())
            .collect::<pyo3::PyResult<Vec<Vec<RecordBatch>>>>()
    })?
    .map_err(|e| source(e.to_string()))?;
    for ((id, ..), batches) in reads.iter().zip(normalized) {
        if !batches.is_empty() {
            sources[*id] = batches;
        }
    }
    Ok(())
}
