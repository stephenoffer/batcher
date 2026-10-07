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

use std::sync::Arc;

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
///
/// The stack of `Filter`s directly over a scan read once in `plan` is applied as each row group
/// is read -- decoded first as late-materialization stages, as the driving scan's are, then
/// evaluated exactly -- so a selective build side (TPC-H's `orders` under a one-year
/// `o_orderdate` range) is never held, concatenated or filtered whole, which on 64 cores was a
/// stretch of the build at a fifth of the machine. The `Filter`s stay in the plan and keep only
/// what they kept already.
///
/// Returns each scan read with filters applied, with the rows it read before them, for the
/// caller to restate the scan's measured counts (`late::restate_prefiltered`).
pub(crate) fn read_into(
    plan: &bc_ir::RelOp,
    reads: &[ResidentRead],
    sources: &mut [Vec<RecordBatch>],
    workers: usize,
) -> Result<Vec<(usize, u64)>, bc_interp::InterpError> {
    let source = |e: String| bc_interp::InterpError::ChunkSource(e);
    // Per scan: the filters to apply as it is read, and the late filter decoding them first.
    let prepared: Vec<(Vec<&bc_expr::Expr>, Option<Arc<bc_io::LateFilter>>)> = reads
        .iter()
        .map(|(id, _, columns, _, _)| {
            if scans_of(plan, *id) != 1 {
                return (Vec::new(), None);
            }
            let carrier = &sources[*id];
            let read = super::late::read_columns(carrier, columns.as_deref());
            let late = super::late::late_of(super::late::plan_stages(plan, *id, carrier), &read);
            (super::late::scan_filters(plan, *id), late)
        })
        .collect();
    // Every scan's surviving row groups, as one flat list of tasks in (scan, file, row group)
    // order, so a small scan's row groups do not wait behind a large one's.
    let mut tasks: Vec<(usize, usize, usize)> = Vec::new();
    let mut prefiltered: Vec<(usize, u64)> = Vec::new();
    for (k, (id, uris, _, predicate, _)) in reads.iter().enumerate() {
        let groups = bc_io::parquet_row_groups_surviving(uris, predicate.as_deref())
            .map_err(|e| source(e.to_string()))?;
        if !prepared[k].0.is_empty() || prepared[k].1.is_some() {
            prefiltered.push((*id, groups.iter().map(|&(_, _, n)| n as u64).sum()));
        }
        tasks.extend(groups.into_iter().map(|(file, rg, _)| (k, file, rg)));
    }
    let decoded: Vec<Vec<RecordBatch>> = bc_interp::install_on_pool(workers, || {
        tasks
            .par_iter()
            .map(|&(k, file, rg)| {
                let (_, uris, columns, predicate, batch_size) = &reads[k];
                let (filters, late) = &prepared[k];
                bc_io::read_parquet_row_group_late(
                    &uris[file],
                    rg,
                    columns.as_deref(),
                    (*batch_size).max(1),
                    predicate.as_deref(),
                    late.as_ref(),
                    false,
                )
                .map_err(|e| source(e.to_string()))?
                .iter()
                .map(|b| {
                    let b = normalize_batch(b).map_err(|e| source(e.to_string()))?;
                    super::late::apply_filters(filters, b).map_err(|e| source(e.to_string()))
                })
                .filter(|b| !matches!(b, Ok(b) if b.num_rows() == 0))
                .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()
    })??;
    let mut normalized: Vec<Vec<RecordBatch>> = vec![Vec::new(); reads.len()];
    for (&(k, _, _), batches) in tasks.iter().zip(decoded) {
        normalized[k].extend(batches);
    }
    for ((id, ..), batches) in reads.iter().zip(normalized) {
        if !batches.is_empty() {
            sources[*id] = batches;
        }
    }
    Ok(prefiltered)
}

/// How many times `plan` scans source `id`.
fn scans_of(plan: &bc_ir::RelOp, id: usize) -> usize {
    let here = usize::from(matches!(plan, bc_ir::RelOp::Scan { source_id } if *source_id == id));
    here + plan
        .children()
        .into_iter()
        .map(|c| scans_of(c, id))
        .sum::<usize>()
}
