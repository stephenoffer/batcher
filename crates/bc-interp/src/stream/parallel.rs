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
use std::sync::Arc;

use arrow::array::RecordBatch;
use bc_ir::RelOp;
use bc_resource::CancelToken;
use rayon::prelude::*;

use bc_runtime::agg;

use super::{
    build_with, combine_and_finalize, fold_partial, limit_stream, node_key, strip_empties,
    BuildCache, Ctx, MatCache, Meter,
};
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
    execute_streaming_parallel_or_hand_off(plan, sources, workers, budget, false, None)
}

/// [`execute_streaming_parallel`], optionally allowed to **decline** a plan it cannot shard.
///
/// With `handoff` set, a plan whose probe spine holds a hash join too large to probe per morsel
/// returns [`InterpError::PreferMaterializing`] instead of running. That is not a failure and not
/// a fallback the executor can take itself: the materializing executor needs the caller's full
/// resource policy (its spill options and memory pool), which this entry point is not given, and
/// the caller is also the only party that knows whether that executor's memory profile is
/// affordable for this query. So the decision is reported, not made.
///
/// Why here and not from the plan alone: whether a join gets a per-morsel probe depends on its
/// build side's *actual* row count, which a filter or an aggregate below it can change by orders
/// of magnitude. The build sides are prepared first and the answer read off them — exact, at the
/// cost of discarding that preparation when the answer is "hand it over".
pub fn execute_streaming_parallel_or_hand_off(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    workers: usize,
    budget: usize,
    handoff: bool,
    cancel: Option<&CancelToken>,
) -> Result<Vec<RecordBatch>, InterpError> {
    in_scoped_pool(workers, || {
        run(plan, sources, workers, None, budget, handoff, cancel)
    })
}

/// Wrap a pipeline's output iterator so each morsel it yields is preceded by a cancellation
/// check.
///
/// The streaming executor is pull-based: one `next()` drives one morsel through *every*
/// operator in the pipeline. So a single check here is a per-morsel check for the whole
/// pipeline, rather than one per operator — and it sits at the only point where nothing is
/// half-built, since the morsel it would have produced has not been started.
///
/// It does not bound a **breaker**. A sort or an aggregate build consumes its entire input
/// inside one `next()`, so a query that spends ten minutes building a hash table notices a
/// cancel when that build finishes. The materializing executor's operator-boundary check and
/// the external sort's per-merge-pass check cover the cases where that matters most; a
/// breaker-internal poll is a further step, not this one.
fn with_cancellation<'a>(
    it: Box<dyn Iterator<Item = Result<RecordBatch, InterpError>> + 'a>,
    cancel: Option<&CancelToken>,
) -> Box<dyn Iterator<Item = Result<RecordBatch, InterpError>> + 'a> {
    let Some(token) = cancel.cloned() else {
        return it; // not cancellable: the iterator is handed back untouched
    };
    Box::new(it.map(move |b| {
        if token.is_cancelled() {
            return Err(InterpError::Cancelled);
        }
        b
    }))
}

/// Run `f` inside a width-sized scoped rayon pool — **never** rayon's global pool.
///
/// The same contract [`crate::par::execute_parallel_with_metrics`] documents, and for the same
/// reason: on a Ray worker the global pool is built lazily at first use, *before* Ray pins the
/// actor's CPU affinity, so it is fixed at ONE thread and every `par_iter` beneath it runs
/// single-threaded. The materializing executor was given a scoped pool for exactly this; this
/// is the default executor, so without it the overwhelming majority of queries took the throttle.
///
/// It also makes `EngineConfig.parallelism` mean something here. On the global pool the setting
/// was silently ignored — a plan asked to run at width 1 still fanned out across every core
/// (measured: 9 cores used at `parallelism = 1`), so neither a user bounding a co-tenanted box
/// nor the control plane's own CPU-share accounting could hold.
fn in_scoped_pool<T>(
    workers: usize,
    f: impl FnOnce() -> Result<T, InterpError> + Send,
) -> Result<T, InterpError>
where
    T: Send,
{
    crate::par::pool_for(workers.max(1))?.install(f)
}

/// [`execute_streaming_parallel`], with per-operator metrics. Results are identical; the metrics
/// are a side-channel the control plane learns from.
pub fn execute_streaming_parallel_metered(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    workers: usize,
    budget: usize,
) -> Result<(Vec<RecordBatch>, ExecMetrics), InterpError> {
    execute_streaming_parallel_metered_or_hand_off(plan, sources, workers, budget, false, None)
}

/// [`execute_streaming_parallel_or_hand_off`], with per-operator metrics.
pub fn execute_streaming_parallel_metered_or_hand_off(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    workers: usize,
    budget: usize,
    handoff: bool,
    cancel: Option<&CancelToken>,
) -> Result<(Vec<RecordBatch>, ExecMetrics), InterpError> {
    let m = Meter::new(plan, workers.max(1) as u32);
    let out = in_scoped_pool(workers, || {
        run(plan, sources, workers, Some(&m), budget, handoff, cancel)
    })?;
    Ok((out, m.finish()))
}

pub(super) fn run(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    workers: usize,
    meter: Option<&Meter>,
    budget: usize,
    handoff: bool,
    cancel: Option<&CancelToken>,
) -> Result<Vec<RecordBatch>, InterpError> {
    run_with_cache(
        plan, sources, workers, meter, budget, None, None, handoff, cancel,
    )
}

/// [`run`], reusing everything the caller has already prepared for this plan — build sides and
/// materialized spine breakers alike.
///
/// Both caches are keyed by plan-node address and are independent of which rows a worker scans
/// (they were computed over the unsharded sources), so handing them to a re-entrant call on a
/// *subtree* of the same plan is sound — and it is what stops that subtree from evaluating them a
/// second time. Dropping either one is the recurring bug in this file: it does not change the
/// answer, so no test fails; it silently doubles the work, and only `explain(analyze=True)` (a
/// scan whose `actual` is 2x its table's row count) shows it.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_reusing(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    workers: usize,
    meter: Option<&Meter>,
    budget: usize,
    cache: Option<&BuildCache>,
    mats: Option<&MatCache>,
    cancel: Option<&CancelToken>,
) -> Result<Vec<RecordBatch>, InterpError> {
    // Never hands off: this re-entry is made from *inside* a running pipeline, which has already
    // produced work the caller cannot rewind. Whether to hand the plan over is settled once, at
    // the top, before any of that exists.
    run_with_cache(
        plan, sources, workers, meter, budget, cache, mats, false, cancel,
    )
}

/// `run`, optionally reusing build sides a caller has already executed.
///
/// `prebuilt` is the caller's `BuildCache` when this call is re-entering on a *subtree* of a plan
/// whose builds were prepared already. Without it, `peel_row_wise` recursing into its input
/// executed every build side a second time and discarded the first set: on TPC-H q17 the meter
/// showed `lineitem` scanned 12M times over a 6M-row table, because the correlated subquery's
/// aggregate — the single most expensive operator in that plan — was computed once for a cache
/// that was then dropped, and once for real.
///
/// Reuse is sound because the cache is keyed by plan-node *address* (`node_key`) and `input` is a
/// borrowed subtree of the very plan the cache was built from, so the joins under it are the same
/// nodes at the same addresses. It is also complete: `collect_builds` descends the whole tree, so
/// a cache built for the parent covers every join in the child.
#[allow(clippy::too_many_arguments)]
fn run_with_cache(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    workers: usize,
    meter: Option<&Meter>,
    budget: usize,
    prebuilt: Option<&BuildCache>,
    prebuilt_mats: Option<&MatCache>,
    handoff: bool,
    cancel: Option<&CancelToken>,
) -> Result<Vec<RecordBatch>, InterpError> {
    let workers = workers.max(1);

    // Already evaluated by the caller: hand back what it produced rather than producing it again.
    //
    // This must come first — before the `Sort` and `Limit` arms below, which would otherwise
    // happily re-run a node the caller has in hand. That is exactly the shape a root
    // `Project(Sort(Aggregate(…)))` takes (TPC-H q3): the pass materializes the `Sort`, then
    // `peel_row_wise` re-enters on that same `Sort`, and without this check the whole aggregate
    // underneath it runs a second time. Nothing in the *result* would betray it.
    if let Some(batches) = prebuilt_mats.and_then(|m| m.get(&node_key(plan))) {
        return Ok(batches.as_ref().clone());
    }

    // A `Sort` (an `ORDER BY`, and the top of most TPC-H queries) is a breaker, but its *input*
    // is usually the expensive part — an aggregate, or a join chain. Parallelize that, then sort
    // its (small) result once. Sorting each shard and concatenating would of course not be
    // sorted; this is the difference between the two.
    if let RelOp::Sort { input, keys, limit } = plan {
        // Hand the child whatever the caller already prepared — the builds *and* the materialized
        // spine breakers. Re-entering through plain `run` would execute every one of them again;
        // see `run_with_cache`'s contract above and `peel_row_wise`.
        let rows = run_with_cache(
            input,
            sources,
            workers,
            meter,
            budget,
            prebuilt,
            prebuilt_mats,
            handoff,
            cancel,
        )?;
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
        // Top-N: the mergeable `parallel_top_n` reduces each morsel to its local top-k and merges
        // the survivors — no concatenation of the whole input, no `LIMIT`-ed `lexsort` over every
        // row (result-identical to a full sort-then-slice; see `parallel_top_n_matches_eager`).
        // The unlimited sort still materializes + sorts. Mirrors the sequential breaker.
        let out = match limit {
            Some(k) if !rows.is_empty() => vec![ops::parallel_top_n(&rows, keys, *k)?],
            // Full sort: try the parallel sample-sort (range-partition + per-range parallel sort),
            // result-identical to the serial `sort_batch` oracle and used unchanged from `par.rs`.
            // It declines for small/limit/unsupported keys, where the serial sort runs. Without it
            // a large full sort ran arrow's single-threaded `lexsort` (~16x DuckDB at 6M rows).
            _ => match ops::materialize(&rows) {
                Ok(combined) => match ops::parallel_sort_batch(&combined, keys, *limit)? {
                    Some(sorted) => sorted,
                    None => vec![ops::sort_batch(&combined, keys, *limit)?],
                },
                Err(_) => Vec::new(),
            },
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

    // A root `Limit` over a **breaker** is the `ORDER BY … LIMIT n OFFSET m` tail, and it dragged
    // the whole query onto the serial fallback for the same reason a mid-spine breaker did:
    // `spine_is_shardable` refuses `Limit` (rightly — a per-shard limit returns `n x workers`
    // rows), and nothing peeled it off the root. ClickBench q38–q42 are all
    // `Limit(Sort(Aggregate(…)))` and measured 0.97–1.00x CPU parallelism — completely serial —
    // while the identical shape *without* the `OFFSET` (q33, q36) parallelized fine. Run the child
    // across every core, then take the window from its result.
    //
    // Sound only because this executor's contract is the oracle's rows **in the oracle's order**:
    // an offset into a differently-ordered relation is a wrong answer, not a reordered one. That
    // is asserted, not assumed — see `a_root_limit_with_an_offset_matches_the_oracle_in_order`.
    //
    // Restricted to a breaker child, and that restriction is the point: over a *pipeline* child
    // the streaming `Limit` stops pulling once it is satisfied, so `LIMIT 10` reads ten rows'
    // worth of morsels instead of the relation. Peeling there would trade that early exit — the
    // one place streaming changes complexity rather than memory — for parallelism over rows the
    // query was never going to return. A breaker child has no early exit to lose: it materializes
    // its whole input either way.
    if let RelOp::Limit { input, n, offset } = plan {
        if is_spine_breaker(input) {
            let rows = run_with_cache(
                input,
                sources,
                workers,
                meter,
                budget,
                prebuilt,
                prebuilt_mats,
                handoff,
                cancel,
            )?;
            let id = meter.map(|m| m.id(plan));
            let t = std::time::Instant::now();
            // The same kernel the streaming pipeline uses, over an already-materialized child —
            // so offset/limit semantics cannot drift between the two paths.
            let out: Vec<RecordBatch> =
                limit_stream(Box::new(rows.into_iter().map(Ok)), *n, *offset)
                    .collect::<Result<_, _>>()?;
            if let (Some(m), Some(id)) = (meter, id) {
                for b in &out {
                    m.morsel(id, b.num_rows() as u64, b, t.elapsed().as_nanos() as u64);
                }
            }
            return Ok(strip_empties(out));
        }
    }

    // (1) The build sides, once. Executed on the streaming path themselves, so building them
    // never materializes their subtree either. Note they are built from the **unsharded**
    // `sources`, which is what lets a worker probe the whole build relation with its shard.
    //
    // Prepared *before* the shardability decision, because that decision depends on what the
    // preparation produced: a join is only worth sharding if its build got a per-morsel probe.
    // Reuse the caller's builds when it already ran them for this subtree (see `run_with_cache`);
    // `owned` only exists to give the freshly built cache somewhere to live for this frame.
    let owned;
    let cache: &BuildCache = match prebuilt {
        Some(c) => c,
        None => {
            owned = super::prebuild_joins(plan, sources, meter, budget, workers)?;
            &owned
        }
    };

    // (1b) The spine breakers, once, in parallel. A breaker between the root and the driving scan
    // is the other thing that used to force the whole query onto one core — `spine_is_shardable`
    // refuses to shard through one, and rightly so. Evaluating it here, over the *unsharded*
    // sources, turns it into a materialized leaf: the spine above it becomes shardable, and the
    // subtree below it (usually the expensive half) has just been run on every core. See
    // [`MatCache`] for why this cannot leak a shard into a breaker.
    let owned_mats: MatCache;
    let mats: Option<&MatCache> = if prebuilt_mats.is_some() {
        prebuilt_mats
    } else if workers > 1 {
        owned_mats =
            materialize_spine_breakers(plan, sources, workers, meter, budget, cache, cancel)?;
        (!owned_mats.is_empty()).then_some(&owned_mats)
    } else {
        None
    };

    let Some(driving) = shardable_source(plan, cache, mats) else {
        // Not shardable as a whole — but a *row-wise root* over a child that is must not drag the
        // whole query onto one core. `Project(Aggregate(…))`, `Project(Filter(Aggregate(…)))` and
        // `Project(Sort(…))` are the shapes: the expensive child is parallelizable, and only the
        // projection/filter sitting on its (already reduced) output made `spine_is_shardable`
        // refuse the plan. Run the child in parallel and apply the row-wise op to its result —
        // exactly what the `Sort` arm above already does for `ORDER BY`.
        //
        // This is why TPC-H q15/q17/q20 were the last queries losing to DuckDB: q15's CTE reaches
        // the executor as `Project(Filter(Aggregate))` on a join's *build* side, so its 6M-row
        // lineitem aggregate — 26.8 ms sharded, and faster than DuckDB's on its own — ran serial
        // and cost ~5x that.
        // One reason for refusing is worth handing back to the caller rather than absorbing: a
        // hash join whose build side is past the per-morsel probe's ceiling. Everything else that
        // lands here still runs across cores in some fashion — the row-wise peel below, the
        // prepared build sides, the materialized spine breakers. That one does not: the probe
        // side is materialized whole and the join, the gather and the fold above it all run
        // through a single pipeline. Measured at sf10 on a 60M x 15M join: 5.7x parallelism
        // streaming against 62x on the materializing executor, which partitions the same join
        // across every core. The caller can run it there; see
        // [`execute_streaming_parallel_or_hand_off`] for why the choice is reported, not taken.
        if handoff && workers > 1 {
            if let Some(reason) = spine_join_blocks_sharding(plan, cache, mats, sources) {
                return Err(InterpError::PreferMaterializing { reason });
            }
        }
        return match plan {
            RelOp::Project { .. } | RelOp::Filter { .. } => {
                peel_row_wise(plan, sources, workers, meter, budget, cache, mats, cancel)
            }
            _ => fallback_with(plan, sources, meter, budget, cache, workers, mats, cancel),
        };
    };
    let driving_rows: usize = sources
        .get(driving)
        .map(|b| b.iter().map(|x| x.num_rows()).sum())
        .unwrap_or(0);
    if workers == 1 || driving_rows < MIN_ROWS_TO_SHARD {
        return fallback_with(plan, sources, meter, budget, cache, workers, mats, cancel);
    }

    // (2) Contiguous shards of the driving scan, in row order — each materialized as the
    // `sources` view its worker will scan. Built *before* the parallel loop so they outlive the
    // morsel streams that borrow them (a stream is a `Box<dyn Iterator + 'a>`, and `'a` has to
    // outlive the closure that produced it).
    // Capped so no shard is under a morsel (see `effective_shard_count`) — else a medium
    // relation's sub-morsel shards run the parallel path slower than sequential.
    let shard_workers = effective_shard_count(workers, driving_rows);
    let shard_sources: Vec<Vec<Vec<RecordBatch>>> = shard(&sources[driving], shard_workers)
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
            // One compiled JIT for the aggregate's expressions, shared by every shard — the
            // schema is identical across shards, so `compile_agg` runs exactly once (the first
            // shard to see a row) rather than once per core.
            let jit: std::sync::OnceLock<ops::AggJit> = std::sync::OnceLock::new();
            let folded: Vec<(Option<agg::Partial>, u64)> = shard_sources
                .par_iter()
                .map(|srcs| {
                    let ctx = Ctx::new(srcs, cache, meter, budget).with_mats(mats);
                    fold_partial(
                        with_cancellation(build_with(input, ctx)?, cancel),
                        group_keys,
                        aggregates,
                        &jit,
                    )
                })
                .collect::<Result<Vec<_>, InterpError>>()?;

            let rows_in: u64 = folded.iter().map(|(_, n)| *n).sum();
            let partials: Vec<agg::Partial> = folded.into_iter().filter_map(|(p, _)| p).collect();
            if partials.is_empty() {
                // No shard saw a row. A global aggregate over nothing still yields one row
                // (`COUNT` 0, `SUM` NULL) — the oracle owns that.
                return crate::execute(plan, sources);
            }
            // Keys *and* accumulators — see the sequential path in `stream::breaker`. The
            // holistic aggregates keep a per-group value list that grows with the input, so
            // counting only `group_columns` under-reads exactly the shape that OOMs.
            let state: u64 = partials
                .iter()
                .flat_map(|p| p.group_columns.iter().chain(p.states.iter().flatten()))
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
                // This is the arm the engine default actually reaches — the sharded fold — so a
                // backend tag recorded only on the sequential breaker would still read `interp`
                // for every real query. Empty when no shard saw a row, where `interp` is right.
                if let Some(compiled) = jit.get() {
                    m.note_backend(m.id(plan), compiled.backend_tag());
                }
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
        // `DISTINCT` over a shard is a *partial* dedup, so each worker runs the pipeline below the
        // dedup and the driver dedups the union — the same `partial → combine` the aggregate arm
        // above runs, with the empty aggregate list.
        //
        // The result is identical to the unsharded path, not merely equivalent. Shards are
        // contiguous in-order row ranges, so flattening them in shard order reproduces the
        // relation in the oracle's row order; `parallel_distinct` then sees the same rows in the
        // same order, differing only in where the morsel boundaries fall. Its `combine` keeps
        // first-seen order across the concatenated partials, and first-seen order over a sequence
        // does not depend on how that sequence is cut into morsels.
        //
        // Peak memory is what the sequential streaming breaker already held (it `drain`s its
        // input), so this buys the parallelism without widening the envelope.
        RelOp::Distinct { input } => {
            let t = std::time::Instant::now();
            let parts: Vec<Vec<RecordBatch>> = shard_sources
                .par_iter()
                .map(|srcs| {
                    let ctx = Ctx::new(srcs, cache, meter, budget).with_mats(mats);
                    with_cancellation(build_with(input, ctx)?, cancel)
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, InterpError>>()?;
            let batches: Vec<RecordBatch> = parts.into_iter().flatten().collect();
            if batches.is_empty() {
                // No shard saw a row. The oracle owns the correctly-typed empty relation.
                return crate::execute(plan, sources);
            }
            let rows_in = crate::count_rows(&batches);
            let held = crate::batch_bytes(&batches);
            if budget > 0 && held as usize > budget {
                return Err(InterpError::MemoryBudgetExceeded {
                    needed: held as usize,
                    budget,
                    reason: "the streaming distinct does not spill",
                });
            }
            let out = ops::parallel_distinct(&batches)?;
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
            Ok(out)
        }
        _ => {
            let parts: Vec<Vec<RecordBatch>> = shard_sources
                .par_iter()
                .map(|srcs| {
                    let ctx = Ctx::new(srcs, cache, meter, budget).with_mats(mats);
                    with_cancellation(build_with(plan, ctx)?, cancel).collect::<Result<Vec<_>, _>>()
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
/// Run a row-wise root's child in parallel, then apply the root to the child's result.
///
/// Only reached when the plan as a whole is un-shardable, so the child is the expensive half and
/// this op runs on what the child already reduced (an aggregate's groups, a sort's rows) — small
/// by the time it gets here, which is why applying it on the driver is not the thing to
/// parallelize. Row-wise ops commute with the child's sharding: `Project`/`Filter` are per-row, so
/// applying them after the child's workers combine is what applying them inside each worker would
/// have produced.
#[allow(clippy::too_many_arguments)]
fn peel_row_wise(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    workers: usize,
    meter: Option<&Meter>,
    budget: usize,
    cache: &BuildCache,
    mats: Option<&MatCache>,
    cancel: Option<&CancelToken>,
) -> Result<Vec<RecordBatch>, InterpError> {
    // `run` only routes the two row-wise roots here; anything else keeps the old behaviour.
    let input = match plan {
        RelOp::Project { input, .. } | RelOp::Filter { input, .. } => input,
        _ => return fallback_with(plan, sources, meter, budget, cache, workers, mats, cancel),
    };
    let id = meter.map(|m| m.id(plan));
    // Hand the child the builds *and* the materialized spine breakers we already have. Re-entering
    // through plain `run` would execute every one of them again — see `run_with_cache`.
    let rows = run_with_cache(
        input,
        sources,
        workers,
        meter,
        budget,
        Some(cache),
        mats,
        false,
        cancel,
    )?;
    let mut out = Vec::with_capacity(rows.len());
    // Hoisted out of the loop on purpose: this is one operator over many morsels, so the
    // conjunct order is measured across them rather than restarted at each.
    let order = match plan {
        RelOp::Filter { predicate, .. } => bc_expr::ConjunctOrder::new(predicate),
        _ => None,
    };
    for b in &rows {
        let t = std::time::Instant::now();
        let done = match plan {
            RelOp::Project { exprs, .. } => ops::project_batch(b, exprs)?,
            RelOp::Filter { predicate, .. } => {
                ops::filter_batch_jit(b, predicate, &None, order.as_ref())?
            }
            _ => unreachable!("guarded above"),
        };
        if let (Some(m), Some(id)) = (meter, id) {
            m.morsel(
                id,
                b.num_rows() as u64,
                &done,
                t.elapsed().as_nanos() as u64,
            );
        }
        out.push(done);
    }
    Ok(strip_empties(out))
}

#[allow(clippy::too_many_arguments)]
fn fallback_with(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    meter: Option<&Meter>,
    budget: usize,
    cache: &BuildCache,
    workers: usize,
    mats: Option<&MatCache>,
    cancel: Option<&CancelToken>,
) -> Result<Vec<RecordBatch>, InterpError> {
    // This path is *not* inside a rayon loop, so its stages may still fan out — which matters
    // because the plan reached it precisely by being un-shardable, and everything in it would
    // otherwise be serial. It reads `mats`, so even a plan that declines to shard collects the
    // parallel evaluation of whatever spine breaker made it decline.
    let ctx = Ctx::with_workers(sources, cache, meter, budget, workers).with_mats(mats);
    let out: Vec<RecordBatch> =
        with_cancellation(build_with(plan, ctx)?, cancel).collect::<Result<_, _>>()?;
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

/// Shards to split the driving relation into: `workers`, capped so no shard is under a morsel.
///
/// Splitting a medium relation across every core gives sub-morsel shards whose per-shard
/// dispatch + `combine` (~15 µs) dwarf their work. `driving_rows / MORSEL_ROWS` (floor — `shard`
/// spreads rows evenly, so ceiling would push the split back below a morsel) keeps every worker
/// at ≥ one morsel; a relation with ≥ `workers` morsels still fills every core. Scheduling only:
/// `combine` is associative + commutative over contiguous in-order shards, so the result stands.
fn effective_shard_count(workers: usize, driving_rows: usize) -> usize {
    workers
        .min(driving_rows / bc_arrow::DEFAULT_MORSEL_ROWS)
        .max(1)
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
fn shardable_source(plan: &RelOp, cache: &BuildCache, mats: Option<&MatCache>) -> Option<usize> {
    // An `Aggregate` is a breaker, and a breaker that sees only a shard computes the wrong
    // answer — *unless* it is the root, where each worker's `Partial` is combined rather than
    // finalized. So the root aggregate is allowed and checked through to its input; an aggregate
    // anywhere *below* is not, and `spine_is_shardable` refuses it.
    //
    // `Distinct` is the same shape and gets the same allowance: it is a mergeable all-column
    // group-by with no aggregates, so a worker's shard produces a *partial* dedup that merges
    // with the others exactly as an aggregate's `Partial` does. Without this the whole subtree
    // under a `DISTINCT` ran on one core — and since Kyber rewrites `COUNT(DISTINCT x) GROUP BY
    // k` into `count(*)` over `DISTINCT (k, x)`, that serialized every grouped `COUNT(DISTINCT)`
    // (ClickBench q10/q11/q13: 22 of 25 ms was the single-threaded scan+filter below the dedup).
    let spine = match plan {
        RelOp::Aggregate { input, .. } | RelOp::Distinct { input } => input,
        other => other,
    };
    if !spine_is_shardable(spine, cache, mats) {
        return None;
    }
    let driving = leftmost_scan(spine, mats)?;
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
///
/// A hash join qualifies **only when its build got a per-morsel probe**. That is not a
/// correctness condition — `materialized_join_from` answers either way — it is the difference
/// between sharding and duplicating: without a probe table that arm joins the *whole* build
/// against the worker's shard, so every worker rebuilds the same hash table. TPC-H q4
/// (`orders SEMI lineitem`) hashed its 3.8M-row build 16 times over to probe 3.5k rows each.
/// Sharding a join we cannot probe per morsel multiplies the build by the worker count; running
/// it once on the sequential streaming path is strictly less work.
fn spine_is_shardable(plan: &RelOp, cache: &BuildCache, mats: Option<&MatCache>) -> bool {
    // An already-materialized breaker is a leaf here, not a breaker: it was evaluated over every
    // row of its input before any sharding began, and what the spine above sees is a finished
    // relation. Sharding is therefore never extended *through* it — the thing that would answer
    // for one shard has already answered for the whole.
    if is_materialized(plan, mats) {
        return true;
    }
    match plan {
        RelOp::Scan { .. } => true,
        RelOp::Filter { input, .. }
        | RelOp::Project { input, .. }
        | RelOp::Unnest { input, .. }
        | RelOp::Unpivot { input, .. } => spine_is_shardable(input, cache, mats),
        RelOp::HashJoin { left, .. } => {
            let probe_driven = cache
                .get(&node_key(plan))
                .is_some_and(|b| b.has_morsel_probe());
            probe_driven && spine_is_shardable(left, cache, mats)
        }
        _ => false,
    }
}

/// Why the probe spine cannot be sharded, when the reason is a hash join — `None` otherwise.
///
/// This is [`spine_is_shardable`]'s one *interesting* refusal. The others (a breaker, a `Limit`,
/// a `RowId`) leave a plan the executor still spreads: a breaker was already evaluated across
/// every core by `materialize_spine_breakers`, and a row-wise root is peeled onto its
/// parallelized child. A join whose build side got no per-morsel probe leaves nothing — the
/// probe relation is drained and concatenated whole, joined, gathered, and folded through one
/// pipeline — so it is the refusal worth telling the caller about.
///
/// Walks the same spine `spine_is_shardable` walks, and stops where it stops: at a materialized
/// node (already a leaf) and at a scan.
///
/// **One hash join in the whole plan, and no more.** That bound is about memory, and it is the
/// reason this is not simply "prefer the faster executor". With a single join the two executors
/// hold the same thing: the streaming fallback already drains and concatenates the entire probe
/// relation to feed that join, so handing it over costs no peak the query was not already paying.
/// More than one join is the opposite — streaming holds one join's output at a time while the
/// materializing executor holds every one of them at once, which is precisely the intermediate
/// blow-up the streaming executor exists to remove. Measured: TPC-H q5 at sf10 (five joins)
/// reaches 99 GB and is OOM-killed on the materializing executor, while streaming answers it in
/// a few GB. Speed is worth trading for memory only where there is no memory to lose.
///
/// The count is over the **whole plan**, not the probe spine, and that distinction is the bug it
/// was written to fix: a bushy join tree puts most of its joins under *build* sides, so a q5 whose
/// spine holds one join still asks the materializing executor to hold five join outputs at once.
/// **And the probe side must be the bigger half.** What the materializing executor buys here is a
/// probe spread across every core; what it costs is the build side, executed once for the cache
/// this call is about to throw away. That trade only pays when the probe is the dominant term.
/// TPC-H q4 is the counter-example that motivated the condition: `orders SEMI lineitem` filters
/// its probe to ~57k rows and builds over ~3.8M, so the two executors measured within 1% of each
/// other (164 ms vs 163 ms at sf10) and handing it over bought nothing while paying for the
/// discarded build. Comparing the driving relation's row count against the build's exact one
/// keeps the hand-off to the shape where it was worth 3-11x.
fn spine_join_blocks_sharding(
    plan: &RelOp,
    cache: &BuildCache,
    mats: Option<&MatCache>,
    sources: &[Vec<RecordBatch>],
) -> Option<&'static str> {
    if count_hash_joins(plan) != 1 {
        return None;
    }
    // The root aggregate/distinct is not itself on the spine — each worker's `Partial` is
    // combined rather than finalized — so look through it, exactly as `shardable_source` does.
    let spine = match plan {
        RelOp::Aggregate { input, .. } | RelOp::Distinct { input } => input.as_ref(),
        other => other,
    };
    let driving_rows: usize = leftmost_scan(spine, mats)
        .and_then(|sid| sources.get(sid))
        .map(|b| b.iter().map(|x| x.num_rows()).sum())
        .unwrap_or(0);
    let mut node = spine;
    loop {
        if is_materialized(node, mats) {
            return None;
        }
        if let RelOp::HashJoin { .. } = node {
            let build = cache.get(&node_key(node))?;
            if build.has_morsel_probe() || driving_rows <= build.side.num_rows() {
                return None;
            }
            return Some("a hash join's build side is too large to probe one morsel at a time");
        }
        node = spine_child(node)?;
    }
}

/// Hash joins anywhere in `plan` — build sides included.
fn count_hash_joins(plan: &RelOp) -> usize {
    let here = usize::from(matches!(plan, RelOp::HashJoin { .. }));
    here + plan
        .children()
        .iter()
        .map(|c| count_hash_joins(c))
        .sum::<usize>()
}

/// The source the spine's driving scan reads, or `None` when there is nothing to shard.
///
/// **Stops at a materialized node**, and that is load-bearing rather than tidy. The scans under a
/// materialized breaker are not executed during the sharded phase — the cached relation is yielded
/// instead — so descending past one would nominate a source that no worker reads. Every worker
/// would then emit the *whole* materialized relation while believing it held a shard of it, and the
/// concatenated result would be `workers x` too many rows. Returning `None` declines the shard
/// instead, which is free: the expensive half already ran on every core.
fn leftmost_scan(plan: &RelOp, mats: Option<&MatCache>) -> Option<usize> {
    if is_materialized(plan, mats) {
        return None;
    }
    match plan {
        RelOp::Scan { source_id } => Some(*source_id),
        other => other
            .children()
            .first()
            .and_then(|c| leftmost_scan(c, mats)),
    }
}

fn is_materialized(plan: &RelOp, mats: Option<&MatCache>) -> bool {
    mats.is_some_and(|m| m.contains_key(&node_key(plan)))
}

/// The next node down the **probe spine** — the path sharding would travel.
///
/// A hash join contributes only its `left` (probe) input: the build side is a separate subtree,
/// prepared once from the unsharded sources by `prebuild_joins`, and is not on this path.
fn spine_child(plan: &RelOp) -> Option<&RelOp> {
    match plan {
        RelOp::Scan { .. } => None,
        RelOp::HashJoin { left, .. } => Some(left),
        other => other.children().first().copied(),
    }
}

/// Whether this node is one `spine_is_shardable` refuses — i.e. the thing that pins the query to
/// one core when it sits mid-spine.
///
/// A `HashJoin` is deliberately **not** included even when it cannot be probed per morsel. Its
/// build side has already been prepared in parallel, and materializing the join itself would only
/// move the same work; the walk descends through it instead, to reach whatever breaker lies below.
fn is_spine_breaker(plan: &RelOp) -> bool {
    !matches!(
        plan,
        RelOp::Scan { .. }
            | RelOp::Filter { .. }
            | RelOp::Project { .. }
            | RelOp::Unnest { .. }
            | RelOp::Unpivot { .. }
            | RelOp::HashJoin { .. }
    )
}

/// Evaluate the first breaker on the probe spine, in parallel, over the **unsharded** sources.
///
/// This is the counterpart to `prebuild_joins` for the other thing that used to serialize a whole
/// query. `spine_is_shardable` refuses a plan with a breaker between the root and the driving scan
/// — correctly, since a breaker handed one shard answers for one shard — and the refusal cost the
/// *entire* query its parallelism, including the breaker's own subtree, which is typically the
/// expensive half and is very often shardable in its own right. TPC-H q17 is the canonical shape:
/// `hash_join(project(aggregate(hash_join(scan lineitem, part))), part)`, whose aggregate reads 6M
/// rows and ran on one core.
///
/// Three properties make this safe, and each one is a wrong answer if it is dropped:
///
/// 1. **The root is never materialized.** `run_with_cache` is in the middle of evaluating it, so
///    caching it would make this function recurse forever. The walk starts at the root's spine
///    child.
/// 2. **Only the first breaker is taken.** Deeper ones are found by the recursive `run` on this
///    subtree, which performs its own pass — so each breaker is evaluated exactly once, by exactly
///    one call, and nested breakers compose without special handling.
/// 3. **The subtree is evaluated over `sources`, unsharded.** Sharding happens only *above* the
///    resulting leaf, never through it.
fn materialize_spine_breakers(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    workers: usize,
    meter: Option<&Meter>,
    budget: usize,
    cache: &BuildCache,
    cancel: Option<&CancelToken>,
) -> Result<MatCache, InterpError> {
    let mut mats = MatCache::new();
    let Some(mut node) = spine_child(plan) else {
        return Ok(mats);
    };
    loop {
        if is_spine_breaker(node) {
            // The builds are handed down (`collect_builds` descends the probe spine, so the cache
            // already covers every join under here); the mats are not, because this subtree owns
            // whatever lies below it.
            let batches = run_with_cache(
                node,
                sources,
                workers,
                meter,
                budget,
                Some(cache),
                None,
                false,
                cancel,
            )?;
            mats.insert(node_key(node), Arc::new(batches));
            return Ok(mats);
        }
        match spine_child(node) {
            Some(next) => node = next,
            None => return Ok(mats),
        }
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

/// Whether the streaming parallel executor can spread this plan across cores.
///
/// It cannot when a source is scanned **more than once** — a self-join or a correlated
/// subquery (TPC-H q21's `EXISTS`/`NOT EXISTS` over `lineitem`, q22's over `orders`). Sharding
/// the driving scan would hand a *build* side a shard instead of the whole relation, so
/// `shardable_source` refuses those plans and the entire query falls to the single-threaded
/// sequential streaming pipeline — where the joins probe one morsel at a time on one core,
/// while the materializing executor's `join_partitioned` spreads the probe across all of them.
/// Measured at sf1, q21 was 251 ms streaming vs 92 ms materializing for exactly this reason.
///
/// The caller uses this to prefer the materializing parallel executor for such a plan **only
/// when memory is unbounded** (`budget == 0`): with no memory cap the input already fits in RAM
/// and the far faster per-core join wins, while a capped run (large scale, distributed) keeps
/// the bounded-memory streaming path. Correctness is identical either way — both executors are
/// checked against the sequential oracle — so this only trades memory for speed.
pub fn streaming_parallelizes(plan: &RelOp) -> bool {
    let mut counts: HashMap<usize, usize> = HashMap::new();
    count_scans(plan, &mut counts);
    counts.values().all(|&c| c <= 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `streaming_parallelizes` is the routing predicate: false exactly when a source is read
    /// more than once, because that is what makes the streaming executor decline to shard and
    /// fall to its single-threaded pipeline. A plain scan and a two-*different*-source join are
    /// parallelizable; a self-join (same `source_id` twice) is not.
    #[test]
    fn streaming_parallelizes_is_false_only_for_a_repeated_source() {
        let scan = |sid| RelOp::Scan { source_id: sid };
        assert!(streaming_parallelizes(&scan(0)), "a bare scan shards");

        let join = |l, r| RelOp::HashJoin {
            left: Box::new(l),
            right: Box::new(r),
            left_keys: vec!["k".into()],
            right_keys: vec!["k".into()],
            join_type: bc_ir::JoinType::Inner,
            output: vec![],
            strategy: bc_ir::JoinStrategy::Hash,
        };
        assert!(
            streaming_parallelizes(&join(scan(0), scan(1))),
            "a join of two distinct sources shards (only the probe side is on the spine)"
        );
        assert!(
            !streaming_parallelizes(&join(scan(0), scan(0))),
            "a self-join reads source 0 twice, so the streaming executor cannot shard it"
        );
    }

    /// The streaming executor's body must run on a pool of the width it was asked for, never
    /// rayon's global pool.
    ///
    /// This is the regression that shipped: `run` was called directly, so `par_iter` beneath it
    /// bound to the global pool. Single-node that merely made `EngineConfig.parallelism` a no-op
    /// (a plan asked for width 1 still used every core); on a Ray worker, whose global pool is
    /// built before CPU affinity lands and is stuck at one thread, it silently throttled the
    /// *default* executor to a single core.
    ///
    /// Scope, stated honestly: this pins the helper, not that the entry points call it — a unit
    /// test cannot observe the pool width from inside a running plan without a hook that would
    /// cost more than it proves. The wiring was verified end to end by measurement instead:
    /// executing at `parallelism = 1` used 8.97 cores before the fix and 1.00 after, against a
    /// materializing executor that reads 1.00 either way.
    #[test]
    fn the_body_runs_on_a_pool_of_the_requested_width() {
        for width in [1usize, 2, 4] {
            let observed = in_scoped_pool(width, || Ok(rayon::current_num_threads())).unwrap();
            assert_eq!(
                observed, width,
                "streaming body ran on a {observed}-thread pool, expected {width}"
            );
        }
    }

    /// `workers()` resolves "all cores" from the machine, not from the global pool's width —
    /// the other half of the same throttle: a Ray worker's global pool reports 1, which would
    /// have shrunk the shard count to 1 even with the scoped pool in place.
    #[test]
    fn zero_parallelism_resolves_to_the_machines_cores() {
        let opts = crate::ExecOptions::default();
        assert_eq!(opts.parallelism, 0, "default is 'all cores'");
        let expected = bc_arrow::usable_cores();
        assert_eq!(opts.workers(), expected);
    }

    /// The shard count never produces a sub-morsel shard, and a relation big enough to fill
    /// every worker still fans out fully. Without the cap, a 96-core box split a 200k-row
    /// relation (~12 morsels) into 96 ~2k-row shards and the dispatch/`combine` overhead ran
    /// the parallel path *slower* than sequential.
    #[test]
    fn shard_count_never_goes_below_one_morsel() {
        let m = bc_arrow::DEFAULT_MORSEL_ROWS;
        // A medium relation is capped by its whole-morsel count, not the machine width.
        assert_eq!(effective_shard_count(96, 200_000), 200_000 / m);
        // Exactly the shard threshold (4 morsels) yields at most 4 shards, not 96.
        assert_eq!(effective_shard_count(96, 4 * m), 4);
        // A relation with ≥ `workers` morsels still uses every worker.
        assert_eq!(effective_shard_count(96, 200 * m), 96);
        // Degenerate inputs never yield zero shards.
        assert_eq!(effective_shard_count(96, 0), 1);
        assert_eq!(effective_shard_count(1, 200 * m), 1);
        // Every shard of a capped split holds at least one full morsel's worth of rows — the
        // whole point of the cap. `shard` spreads rows evenly, so the smallest shard is
        // `rows / shards` (floored), which must still clear a morsel.
        for rows in [4 * m, 65_536, 100_000, 200_000, 500_000, 96 * m + 1] {
            let shards = effective_shard_count(96, rows);
            assert!(
                rows / shards >= m,
                "{rows} rows split into {shards} shards gives a sub-morsel shard of {} rows",
                rows / shards
            );
        }
    }
}
