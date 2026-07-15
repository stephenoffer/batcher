//! Streaming, across cores: one pipeline instance per worker over a shard of the driving scan.
//!
//! The sequential streaming executor fixes the memory model but gives up the parallelism the
//! materializing path has, which makes it the wrong default — a query that fits in RAM should
//! not lose every core to save memory it was not short of. This restores it, the way the RFC
//! describes: *"morselize the leaf, run one pipeline instance per worker over a shard of
//! morsels, each feeding a thread-local breaker partial; combine at the breaker."*
//!
//! The shape:
//!
//! 1. **Every hash-join build side is executed once**, up front, and shared. A worker must not
//!    rebuild the dimension tables it probes — that would be `workers x` the build cost, and on
//!    a chain of joins it is the dominant term. They are cached by plan-node identity and handed
//!    to every worker.
//! 2. **The driving scan is sharded** into contiguous row ranges, one per worker. Contiguous and
//!    in-order is what lets the outputs concatenate back into the oracle's row order.
//! 3. **Each worker runs the whole streaming pipeline** over its shard, holding one morsel.
//! 4. **The root combines**: an aggregate folds each worker's `Partial` (mergeable algebra, the
//!    same `combine` the distributed path uses); anything else concatenates the workers' output
//!    in shard order.
//!
//! Peak memory is `workers x morsel` + the shared build tables + the breaker's state — still
//! independent of the input size, which is the property the whole exercise is for.
//!
//! **When it declines.** Sharding rewrites what one `Scan` sees, so it is only sound when that
//! source is read in exactly one place. A self-join (the driving source also appearing under a
//! build side) would hand the build a *shard* and silently compute a different relation, so it
//! is refused and the sequential streaming path takes over. Same for a plan with no scan to
//! shard, or too few rows to be worth splitting.

use std::collections::HashMap;

use arrow::array::RecordBatch;
use bc_ir::RelOp;
use rayon::prelude::*;

use bc_runtime::agg;

use super::{build_with, combine_and_finalize, fold_partial, strip_empties, Ctx, Meter};
use crate::ops;
use crate::{ExecMetrics, InterpError};

/// Below this, sharding costs more than it saves — the shared build sides still have to be
/// executed, and a handful of morsels does not need four threads.
const MIN_ROWS_TO_SHARD: usize = 4 * bc_arrow::DEFAULT_MORSEL_ROWS;

/// Execute `plan` by streaming, with one pipeline per worker over a shard of the driving scan.
///
/// Falls back to the sequential streaming path (never to materializing) whenever the plan cannot
/// be sharded soundly.
pub fn execute_streaming_parallel(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    workers: usize,
    budget: usize,
) -> Result<Vec<RecordBatch>, InterpError> {
    run(plan, sources, workers, None, budget)
}

/// [`execute_streaming_parallel`], with per-operator metrics. Results are identical; the metrics
/// are a side-channel the control plane learns from.
pub fn execute_streaming_parallel_metered(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    workers: usize,
    budget: usize,
) -> Result<(Vec<RecordBatch>, ExecMetrics), InterpError> {
    let m = Meter::new(plan, workers.max(1) as u32);
    let out = run(plan, sources, workers, Some(&m), budget)?;
    Ok((out, m.finish()))
}

fn run(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    workers: usize,
    meter: Option<&Meter>,
    budget: usize,
) -> Result<Vec<RecordBatch>, InterpError> {
    let workers = workers.max(1);

    // A `Sort` (an `ORDER BY`, and the top of most TPC-H queries) is a breaker, but its *input*
    // is usually the expensive part — an aggregate, or a join chain. Parallelize that, then sort
    // its (small) result once. Sorting each shard and concatenating would of course not be
    // sorted; this is the difference between the two.
    if let RelOp::Sort { input, keys, limit } = plan {
        let rows = run(input, sources, workers, meter, budget)?;
        let rows_in = crate::count_rows(&rows);
        let held = crate::batch_bytes(&rows);
        if budget > 0 && held as usize > budget {
            return Err(InterpError::MemoryBudgetExceeded {
                needed: held as usize,
                budget,
                reason: "the streaming sort does not spill",
            });
        }
        let t = std::time::Instant::now();
        let out = match ops::materialize(&rows) {
            Ok(combined) => vec![ops::sort_batch(&combined, keys, *limit)?],
            Err(_) => Vec::new(),
        };
        if let Some(m) = meter {
            m.breaker(
                m.id(plan),
                rows_in,
                0,
                held,
                &out,
                t.elapsed().as_nanos() as u64,
            );
        }
        return Ok(out);
    }

    let Some(driving) = shardable_source(plan) else {
        return fallback(plan, sources, meter, budget);
    };
    let driving_rows: usize = sources
        .get(driving)
        .map(|b| b.iter().map(|x| x.num_rows()).sum())
        .unwrap_or(0);
    if workers == 1 || driving_rows < MIN_ROWS_TO_SHARD {
        return fallback(plan, sources, meter, budget);
    }

    // (1) The build sides, once. Executed on the streaming path themselves, so building them
    // never materializes their subtree either. Note they are built from the **unsharded**
    // `sources`, which is what lets a worker probe the whole build relation with its shard.
    let cache = super::prebuild_joins(plan, sources, meter, budget)?;

    // (2) Contiguous shards of the driving scan, in row order — each materialized as the
    // `sources` view its worker will scan. Built *before* the parallel loop so they outlive the
    // morsel streams that borrow them (a stream is a `Box<dyn Iterator + 'a>`, and `'a` has to
    // outlive the closure that produced it).
    let shard_sources: Vec<Vec<Vec<RecordBatch>>> = shard(&sources[driving], workers)
        .into_iter()
        .map(|sh| swap(sources, driving, sh))
        .collect();

    // (3)+(4) One streaming pipeline per shard; combine at the root.
    match plan {
        RelOp::Aggregate {
            input,
            group_keys,
            aggregates,
        } => {
            // Each worker folds its shard into a `Partial` — and stops there. Finalizing per
            // shard would be wrong, not merely slow: two shards' `mean`s do not average to the
            // relation's `mean`, and once a `count` and a `sum` are finalized into a row the
            // shape needed to merge them is gone. `partial` per shard, `combine` across shards,
            // `finalize` once is the mergeable algebra (invariant #7) — the very same fold the
            // distributed path runs across *nodes*. One implementation, one set of semantics.
            let t = std::time::Instant::now();
            let folded: Vec<(Option<agg::Partial>, u64)> = shard_sources
                .par_iter()
                .map(|srcs| {
                    let ctx = Ctx::new(srcs, &cache, meter, budget);
                    fold_partial(build_with(input, ctx)?, group_keys, aggregates)
                })
                .collect::<Result<Vec<_>, InterpError>>()?;

            let rows_in: u64 = folded.iter().map(|(_, n)| *n).sum();
            let partials: Vec<agg::Partial> = folded.into_iter().filter_map(|(p, _)| p).collect();
            if partials.is_empty() {
                // No shard saw a row. A global aggregate over nothing still yields one row
                // (`COUNT` 0, `SUM` NULL) — the oracle owns that.
                return crate::execute(plan, sources);
            }
            let state: u64 = partials
                .iter()
                .flat_map(|p| p.group_columns.iter())
                .map(|c| c.get_array_memory_size() as u64)
                .sum();
            // Over budget: the materializing executor spills this; streaming does not. Hand the
            // query back rather than OOM where it would have survived.
            if budget > 0 && state as usize > budget {
                return Err(InterpError::MemoryBudgetExceeded {
                    needed: state as usize,
                    budget,
                    reason: "the streaming aggregate does not spill",
                });
            }
            let out = combine_and_finalize(&partials, group_keys, aggregates)?;
            if let Some(m) = meter {
                m.breaker(
                    m.id(plan),
                    rows_in,
                    0,
                    state,
                    &out,
                    t.elapsed().as_nanos() as u64,
                );
            }
            Ok(out)
        }
        _ => {
            let parts: Vec<Vec<RecordBatch>> = shard_sources
                .par_iter()
                .map(|srcs| {
                    let ctx = Ctx::new(srcs, &cache, meter, budget);
                    build_with(plan, ctx)?.collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, InterpError>>()?;
            // Shards are contiguous, in-order row ranges, so concatenating them in shard order
            // reproduces exactly the relation — and the order — the sequential path emits.
            Ok(strip_empties(parts.into_iter().flatten().collect()))
        }
    }
}

/// The plan cannot be sharded: run it on the sequential streaming path. Still bounded-memory,
/// just single-threaded — never a fall back to materializing.
fn fallback(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    meter: Option<&Meter>,
    budget: usize,
) -> Result<Vec<RecordBatch>, InterpError> {
    let cache = super::prebuild_joins(plan, sources, meter, budget)?;
    let ctx = Ctx::new(sources, &cache, meter, budget);
    let out: Vec<RecordBatch> = build_with(plan, ctx)?.collect::<Result<_, _>>()?;
    Ok(strip_empties(out))
}

/// `sources` with `idx` replaced by `shard` — what one worker's pipeline scans.
fn swap(
    sources: &[Vec<RecordBatch>],
    idx: usize,
    shard: Vec<RecordBatch>,
) -> Vec<Vec<RecordBatch>> {
    let mut out = sources.to_vec();
    out[idx] = shard;
    out
}

/// Split a relation into `workers` contiguous, in-order shards of whole morsels.
fn shard(batches: &[RecordBatch], workers: usize) -> Vec<Vec<RecordBatch>> {
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    let per = total.div_ceil(workers).max(1);
    let mut out: Vec<Vec<RecordBatch>> = Vec::with_capacity(workers);
    let mut cur: Vec<RecordBatch> = Vec::new();
    let mut cur_rows = 0usize;

    for b in batches {
        let mut off = 0usize;
        while off < b.num_rows() {
            let want = per - cur_rows;
            let take = want.min(b.num_rows() - off);
            cur.push(b.slice(off, take)); // zero-copy
            cur_rows += take;
            off += take;
            if cur_rows >= per {
                out.push(std::mem::take(&mut cur));
                cur_rows = 0;
            }
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    if out.is_empty() {
        // An empty relation still owes its schema to whatever is above it.
        out.push(batches.to_vec());
    }
    out
}

/// The source id to shard, or `None` when this plan must not be sharded.
///
/// Two conditions, and both were learned the hard way — each one is a wrong answer, silently, if
/// it is skipped:
///
/// 1. **The spine must be shardable.** Sharding hands one worker a *slice* of the driving
///    relation, so every operator between the root and that scan has to treat its input as a
///    stream of independent morsels. A breaker on that path does not: sort each shard and
///    concatenate and the result is not sorted; dedup each shard and duplicates survive across
///    shards; `LIMIT n` each shard and you get `n x workers` rows. `Limit` and `RowId` are not
///    breakers but are just as positional — a per-shard limit, or a row counter that restarts at
///    zero in every worker.
/// 2. **The driving source must be read in exactly one place.** A self-join reads it again under
///    a build side, and that build must see the *whole* relation, not this worker's shard.
///
/// Anything else falls back to the sequential streaming path — still bounded-memory, just
/// single-threaded. Declining is always safe; guessing is not.
fn shardable_source(plan: &RelOp) -> Option<usize> {
    // An `Aggregate` is a breaker, and a breaker that sees only a shard computes the wrong
    // answer — *unless* it is the root, where each worker's `Partial` is combined rather than
    // finalized. So the root aggregate is allowed and checked through to its input; an aggregate
    // anywhere *below* is not, and `spine_is_shardable` refuses it.
    let spine = match plan {
        RelOp::Aggregate { input, .. } => input,
        other => other,
    };
    if !spine_is_shardable(spine) {
        return None;
    }
    let driving = leftmost_scan(spine)?;
    let mut counts: HashMap<usize, usize> = HashMap::new();
    count_scans(plan, &mut counts);
    (counts.get(&driving) == Some(&1)).then_some(driving)
}

/// Whether every operator from here down to the driving scan is morsel-independent.
///
/// A hash join qualifies because only its **probe** side is on this path — its build side is a
/// separate subtree, prepared once from the *unsharded* sources, so a worker still probes the
/// whole build relation.
///
/// Everything not listed is refused, and that is the point: a breaker here (`Aggregate`, `Sort`,
/// `Distinct`, `Window`, `Union`, …) would see one shard and answer for one shard, and `Limit` /
/// `RowId` are positional in the same way — a per-shard limit, or a row counter restarting at
/// zero in every worker. A new `RelOp` variant is un-shardable until someone proves otherwise.
fn spine_is_shardable(plan: &RelOp) -> bool {
    match plan {
        RelOp::Scan { .. } => true,
        RelOp::Filter { input, .. }
        | RelOp::Project { input, .. }
        | RelOp::Unnest { input, .. }
        | RelOp::Unpivot { input, .. } => spine_is_shardable(input),
        RelOp::HashJoin { left, .. } => spine_is_shardable(left),
        _ => false,
    }
}

fn leftmost_scan(plan: &RelOp) -> Option<usize> {
    match plan {
        RelOp::Scan { source_id } => Some(*source_id),
        other => other.children().first().and_then(|c| leftmost_scan(c)),
    }
}

fn count_scans(plan: &RelOp, counts: &mut HashMap<usize, usize>) {
    if let RelOp::Scan { source_id } = plan {
        *counts.entry(*source_id).or_insert(0) += 1;
    }
    for c in plan.children() {
        count_scans(c, counts);
    }
}
