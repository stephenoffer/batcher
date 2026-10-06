//! The chunk-driven entry points: [`execute_chunked`] and its metered twin.
//!
//! Kept apart from the unit-driven ones in `mod.rs` only for the file budget; both drive the
//! same `Run` over the same oriented plan.

use arrow::array::RecordBatch;
use bc_ir::RelOp;

use super::orient::{orient, orient_counted, original_ids};
use super::{node_at, oriented_core, run_post, Core, NextChunk, Run};
use crate::error::InterpError;
use crate::par::ExecOptions;

/// Execute `plan` with `sources[driving]` supplied by `next_chunk` rather than resident.
///
/// `sources[driving]` carries only the driving relation's schema — a zero-row batch — and is
/// what the plan runs over when the chunks hold no rows at all. Returns the same rows the
/// resident executors return for the same plan over the concatenated chunks.
///
/// # Errors
/// [`InterpError::NotChunkable`] when the plan is not [`super::chunkable`] — the caller should
/// run it resident instead; [`InterpError::MemoryBudgetExceeded`] when the aggregate state (or
/// the collected output of a spine) outgrows `budget` — the caller routes the query to an
/// executor that spills; anything a chunk producer or an operator reports.
pub fn execute_chunked(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    driving: usize,
    next_chunk: &mut NextChunk<'_>,
    workers: usize,
    budget: usize,
    opts: &ExecOptions,
) -> Result<Vec<RecordBatch>, InterpError> {
    let oriented = orient(plan, driving);
    chunked_with(
        &oriented, sources, driving, next_chunk, workers, budget, opts, None,
    )
}

/// [`execute_chunked`], with per-operator metrics for the operators it runs itself.
///
/// One meter spans every chunk: each chunk's operators add to the same counters, so the totals
/// are the query's real cardinalities, as [`super::execute_units_metered`]'s are. Taking any one
/// chunk's counts as the operator's would teach the learning loop a fraction of the truth; that,
/// not chunking, is what kept this path unmetered. A distributed broadcast join is the caller
/// it was added for: probing a replicated build side with one native call per probe chunk built
/// the hash table again on every call, and the chunked path builds it once.
///
/// # Errors
/// As [`execute_chunked`].
#[allow(clippy::too_many_arguments)]
pub fn execute_chunked_metered(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    driving: usize,
    next_chunk: &mut NextChunk<'_>,
    workers: usize,
    budget: usize,
    opts: &ExecOptions,
) -> Result<(Vec<RecordBatch>, crate::ExecMetrics), InterpError> {
    let (oriented, swaps) = orient_counted(plan, driving);
    let meter = super::super::Meter::new(&oriented, workers.max(1) as u32);
    let out = chunked_with(
        &oriented,
        sources,
        driving,
        next_chunk,
        workers,
        budget,
        opts,
        Some(&meter),
    )?;
    // Filed under the control plane's op ids, as `execute_units_metered` files its own.
    let metrics = if swaps == 0 {
        meter.finish()
    } else {
        meter.finish_renumbered(&original_ids(plan, &oriented, driving))
    };
    Ok((out, metrics))
}

/// The body both entry points share, over an already oriented `plan`.
#[allow(clippy::too_many_arguments)]
fn chunked_with(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    driving: usize,
    next_chunk: &mut NextChunk<'_>,
    workers: usize,
    budget: usize,
    opts: &ExecOptions,
    meter: Option<&super::super::Meter>,
) -> Result<Vec<RecordBatch>, InterpError> {
    let Some(core) = oriented_core(plan, driving) else {
        return Err(InterpError::NotChunkable);
    };
    let run = Run {
        driving,
        workers: workers.max(1),
        budget,
        opts,
        carrier: sources[driving].clone(),
        driving_rows: None,
        pool: crate::par::pool_for(workers.max(1))?,
    };
    let mut srcs: Vec<Vec<RecordBatch>> = sources.to_vec();
    srcs[driving] = next_chunk().transpose()?.unwrap_or_default();
    match core {
        Core::Aggregate { path, node } => {
            let result = run.aggregate(node, &mut srcs, next_chunk, meter)?;
            if path.is_empty() {
                return Ok(result);
            }
            // The rest of the plan reads the aggregate's result as a new source.
            let mut rest = plan.clone();
            *node_at(&mut rest, &path) = RelOp::Scan {
                source_id: srcs.len(),
            };
            let mut rest_srcs = srcs;
            rest_srcs[driving] = Vec::new();
            rest_srcs.push(result);
            match meter {
                Some(_) => run_post(plan, &rest, &path, &rest_srcs, opts, meter),
                None => crate::par::execute_parallel_with(&rest, &rest_srcs, opts),
            }
        }
        Core::Spine { depth, node } => {
            let result = run.collect(node, &mut srcs, next_chunk, meter)?;
            if depth == 0 {
                return Ok(result);
            }
            // The global sort/limit above the spine reads its collected rows as a new source.
            let path = vec![0; depth];
            let mut post = plan.clone();
            *node_at(&mut post, &path) = RelOp::Scan {
                source_id: srcs.len(),
            };
            let mut post_srcs = srcs;
            post_srcs[driving] = Vec::new();
            post_srcs.push(result);
            match meter {
                Some(_) => run_post(plan, &post, &path, &post_srcs, opts, meter),
                None => crate::execute(&post, &post_srcs),
            }
        }
    }
}
