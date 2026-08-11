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
    in_scoped_pool(useful_workers(plan, sources, workers), || {
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

/// `workers`, capped by the shards this plan could actually keep busy.
///
/// The pool this executor installs was sized from the *machine* while the work was sized from
/// the *data*: [`effective_shard_count`] already refuses to cut a relation below one morsel per
/// shard, so a 100 K-row query runs 6 shards — inside a 96-thread pool. The other 90 threads are
/// not free. Rayon wakes them, they contend for the job queue and the epoch GC, and the cost is
/// paid on exactly the queries that can least absorb it. Measured on a 100 K-row point lookup:
/// **6.15 ms of CPU for 1.46 ms of wall time (4.2 cores)**, against 1.66 ms of CPU — and a
/// *faster* 1.25 ms wall — at a width of six.
///
/// This is the cap [`crate::par::auto_width`] already applies on the materializing path, for the
/// reason its docstring gives: "a one-row query would otherwise install and spin a 96-thread
/// pool. Batcher's stated goal of low fixed overhead on sub-second queries is exactly this
/// case." This is the *default* executor, so it is where that goal is mostly decided.
///
/// It is an upper bound over every source rather than the driving one alone, which is not yet
/// chosen here — so it can only ever be larger than the shard count, and can never remove
/// parallelism the plan could have used. Scheduling only: the shard count, and therefore the
/// result, is unchanged.
///
/// **Exception — media decode**, the same one `auto_width` carves out and for the same reason. A
/// `.image`/`.audio`/`.video` decode does heavy per-row work *inside* the morsel with its own
/// rayon fan-out, and its input is tiny encoded bytes, so the morsel count understates the
/// useful width by orders of magnitude. Those plans keep every core.
fn useful_workers(plan: &RelOp, sources: &[Vec<RecordBatch>], workers: usize) -> usize {
    if nothing_to_parallelize(plan) {
        return 1;
    }
    if plan.contains_media_decode() {
        return workers.max(1);
    }
    let widest = sources
        .iter()
        .map(|batches| batches.iter().map(|b| b.num_rows()).sum::<usize>())
        .max()
        .unwrap_or(0);
    workers.min(widest / bc_arrow::DEFAULT_MORSEL_ROWS).max(1)
}

/// A plan that computes nothing per row, so spreading it across cores cannot make it faster.
///
/// A bare `Scan` *is* its source: the streaming pipeline over it hands back the morsels it was
/// given. Running that in parallel means cutting the source into shards, running an identity
/// pipeline over each, concatenating them back, and installing a pool to do it in. All of that
/// machinery is overhead with nothing to amortize it against, and it is not free — measured on
/// the 3 M-row local Parquet read in `benchmarks/scenarios/formats/read.py`, a single-column
/// `read.parquet(...).collect()` cost **9.7 ms wall and 70.2 ms of CPU** at the automatic width
/// against **7.5 ms and 12.5 ms** at a width of one. The parallel path was both slower and five
/// times more expensive, on the most ordinary call a user makes: load a file.
///
/// This is a different bound from the one [`useful_workers`] otherwise applies, and neither
/// subsumes the other. That one asks *how many morsels exist* — a 3 M-row scan has 183, so it
/// caps nothing. This one asks *whether spreading them accomplishes anything*, and for an
/// identity pipeline the answer is no at any size.
///
/// Result-preserving in the strongest sense: the operator computes the same rows in the same
/// order either way, so this does not even move a float the way a partition-count change can.
fn nothing_to_parallelize(plan: &RelOp) -> bool {
    matches!(plan, RelOp::Scan { .. })
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
    let out = in_scoped_pool(useful_workers(plan, sources, workers), || {
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

    // A `UNION ALL` runs its branches **across the pool**, not one after another.
    //
    // Each branch is a whole plan in its own right — its own scans, its own joins, its own
    // aggregate — and nothing about one branch's rows depends on another's. But a `Union` root
    // is not shardable (there is no single driving relation to slice), so without this arm the
    // whole query fell to the sequential pipeline and *every branch lost its own parallelism
    // too*. That is a far larger cost than the missed overlap between branches, and it is the
    // dominant defect in TPC-DS: measured on q22's five grouping levels over
    // `inventory x date_dim x item`, one level alone runs at **63.6 cores in 106 ms**, and the
    // five of them unioned ran at **5.8 cores in 2,624 ms** — the same 15.1 s of CPU, spread
    // over a ninth of the machine. The five levels run standalone sum to 331 ms, so the union
    // was costing 8x the sum of its parts.
    //
    // Order is preserved: `par_iter().collect::<Result<Vec<_>>>()` keeps branch order, so the
    // rows appear exactly as the sequential loop emitted them. Every branch still goes through
    // `run_with_cache`, so each one shards, pre-builds and meters as it would on its own — the
    // `Meter`'s counters are atomics precisely so concurrent workers can share it.
    //
    // `UNION` (distinct) is deliberately excluded: it needs the dedup the fallback path applies
    // over the concatenated result, and returning early here would skip it.
    //
    // The memory budget is **divided** among the branches rather than handed to each in full.
    // Branches now run at the same time, so their peaks are concurrent, and giving each the
    // whole envelope would authorize `branches x budget` — the one way this change could turn a
    // query that fitted into one that does not.
    if let RelOp::Union { inputs, distinct } = plan {
        if !*distinct && inputs.len() > 1 && workers > 1 {
            let share = if budget == 0 {
                0
            } else {
                (budget / inputs.len()).max(1)
            };
            let per: Vec<Vec<RecordBatch>> = inputs
                .par_iter()
                .map(|branch| {
                    run_with_cache(
                        branch, sources, workers, meter, share, None, None, false, cancel,
                    )
                })
                .collect::<Result<_, _>>()?;
            let all: Vec<RecordBatch> = per.into_iter().flatten().collect();
            // Promotable-but-different branch types (`int64 ∪ float64`) coerce to the union's
            // advertised supertype, exactly as the sequential arm does.
            return Ok(strip_empties(crate::coerce_union_branches(all)?));
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
    // `nothing_to_parallelize` joins the two existing reasons not to cut the source up: a plan
    // that computes nothing per row gains nothing from being sharded, however many rows it has.
    // Without it the width cap alone would leave the worst of both — ~96 shards of an identity
    // pipeline, reassembled, on a pool of one.
    if workers == 1 || nothing_to_parallelize(plan) || driving_rows < MIN_ROWS_TO_SHARD {
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
                // (`COUNT` 0, `SUM` NULL) — the oracle owns that, over an empty input.
                return empty_shard_result(plan, sources);
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
        // A dedup over a shard is a *partial* dedup, so each worker runs the pipeline below the
        // dedup and the driver dedups the union — the same `partial → combine` the aggregate arm
        // above runs, with the empty aggregate list. A keyed dedup reduces the same way: its
        // `partial` and `combine` are one function, so the union of shard reductions reduced
        // again is the whole answer.
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
        RelOp::Distinct {
            input,
            keys,
            order,
            limit,
        } => {
            let t = std::time::Instant::now();

            // A fused `LIMIT k` makes each shard keep only its own first `k` distinct rows and
            // stop pulling, then combines those in shard order. The mergeable argument is on
            // `DistinctPrefix`: a row among the global first `k` is among its own shard's first
            // `k`, so the ordered union of the shard prefixes contains the answer and
            // re-applying the operator to it selects exactly that answer.
            //
            // `par_iter().collect()` preserves shard order, and the shards partition the source
            // in order, so the concatenation below is the input's own row order — which is what
            // makes this equal to the sequential oracle rather than merely the same size.
            // `DISTINCT ON` is excluded for the reason the sequential breaker gives.
            if let (Some(k), true) = (limit, keys.is_empty()) {
                // Each shard probes a bounded prefix of its own slice. It stops the moment it
                // holds `k` distinct rows — the win — and gives up hashing at
                // `PREFIX_PROBE_MORSELS` if it does not, because a key that sparse against the
                // limit is one the whole-column dense direct-map deduplicates faster than any
                // exit can be reached. A shard that gives up keeps *collecting* its batches,
                // which costs nothing, so the fallback below pays for the ordinary dedup and
                // sixteen morsels of wasted hashing rather than a second pass over the source.
                type Probe = (Option<RecordBatch>, Vec<RecordBatch>, bool);
                let probes: Vec<Probe> = shard_sources
                    .par_iter()
                    .map(|srcs| {
                        let ctx = Ctx::new(srcs, cache, meter, budget).with_mats(mats);
                        let mut acc = bc_runtime::agg::DistinctPrefix::new(*k);
                        let mut held: Vec<RecordBatch> = Vec::new();
                        let mut gave_up = false;
                        for batch in with_cancellation(build_with(input, ctx)?, cancel) {
                            let batch = batch?;
                            if !gave_up {
                                acc.push(&batch)?;
                                if acc.is_satisfied() {
                                    held.push(batch);
                                    break;
                                }
                                gave_up = held.len() + 1 >= super::breaker::PREFIX_PROBE_MORSELS;
                            }
                            held.push(batch);
                        }
                        Ok::<_, InterpError>((acc.finish(), held, gave_up))
                    })
                    .collect::<Result<Vec<_>, InterpError>>()?;

                // Only when *every* shard filled its prefix is the ordered union of those
                // prefixes the answer; one shard that gave up may still hold distinct rows the
                // global first `k` needs, and its prefix does not contain them.
                if probes.iter().all(|(_, _, gave_up)| !gave_up) {
                    let shard_rows: Vec<RecordBatch> =
                        probes.into_iter().filter_map(|(p, _, _)| p).collect();
                    if shard_rows.is_empty() {
                        // No shard saw a row. The oracle owns the correctly-typed empty relation.
                        return empty_shard_result(plan, sources);
                    }
                    let rows_in = crate::count_rows(&shard_rows);
                    let out: Vec<RecordBatch> = bc_runtime::agg::distinct_prefix(&shard_rows, *k)?
                        .into_iter()
                        .collect();
                    if let Some(m) = meter {
                        m.breaker(
                            m.id(plan),
                            rows_in,
                            0,
                            0,
                            &out,
                            t.elapsed().as_nanos() as u64,
                        );
                    }
                    return Ok(out);
                }

                // Measured low cardinality. Dedup the ordinary bucket-parallel way over
                // everything the shards collected; the limit binds only if that turns up more
                // than `k` distinct rows, and only then is the ordered prefix recomputed.
                let batches: Vec<RecordBatch> =
                    probes.into_iter().flat_map(|(_, held, _)| held).collect();
                if batches.is_empty() {
                    return empty_shard_result(plan, sources);
                }
                let rows_in = crate::count_rows(&batches);
                let held = crate::batch_bytes(&batches);
                let full = ops::parallel_distinct(&batches)?;
                let distinct_rows: usize = full.iter().map(|b| b.num_rows()).sum();
                let out = match distinct_rows > *k {
                    false => full,
                    true => bc_runtime::agg::distinct_prefix(&batches, *k)?
                        .into_iter()
                        .collect(),
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
                return empty_shard_result(plan, sources);
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
            let out = match keys.is_empty() {
                true => ops::parallel_distinct(&batches)?,
                false => ops::parallel_distinct_on(&batches, keys, order)?,
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

/// The plan's answer when sharded execution proved its input holds no rows.
///
/// Four arms below reach the same situation — every shard reported nothing — and each needs the
/// *shape* of an empty answer rather than its content: a keyless aggregate still owes one row
/// (`COUNT` 0, `SUM` NULL), a grouped one owes none, a `DISTINCT` owes an empty relation with the
/// right column types. The sequential oracle owns those rules and this path should not restate
/// them.
///
/// What it must not do is ask the oracle over the **whole relation**, which is what all four did:
/// having just established that the answer is empty, the executor re-ran the entire query
/// single-threaded to be told so again. That is not a small tax. TPC-H q18 and q19 both end in an
/// aggregate whose input is empty at sf1, and both spent ~120 ms of a ~140 ms query inside this
/// second, serial execution — measured at **2 runnable threads against 92**, immediately after a
/// correct 92-way parallel phase that had taken 5.9 ms and already had the answer. It is also why
/// adding one predicate to a join's build side could make a query *seven times slower* while
/// doing strictly less work: the extra predicate emptied the join, and emptying it bought a whole
/// extra execution.
///
/// Slicing each source to zero rows keeps every schema the oracle needs and leaves it nothing to
/// scan. Sound because the operators here are the plan's root: their input is empty in the real
/// execution and empty in this one, so they compute the same answer — and the arms only reach it
/// having proved that emptiness.
fn empty_shard_result(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
) -> Result<Vec<RecordBatch>, InterpError> {
    let empty: Vec<Vec<RecordBatch>> = sources
        .iter()
        .map(|s| s.first().map(|b| vec![b.slice(0, 0)]).unwrap_or_default())
        .collect();
    crate::execute(plan, &empty)
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
        RelOp::Aggregate { input, .. } | RelOp::Distinct { input, .. } => input,
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
    // The root aggregate/distinct is not itself on the spine — each worker's `Partial` is
    // combined rather than finalized — so look through it, exactly as `shardable_source` does.
    let spine = match plan {
        RelOp::Aggregate { input, .. } | RelOp::Distinct { input, .. } => input.as_ref(),
        other => other,
    };
    let driving_rows: usize = leftmost_scan(spine, mats)
        .and_then(|sid| sources.get(sid))
        .map(|b| b.iter().map(|x| x.num_rows()).sum())
        .unwrap_or(0);
    // Only the **first** join on the spine is judged, and only in a single-join plan. Both
    // restrictions look over-cautious and both were measured before being left in place.
    //
    // Extending the walk to every spine join, and dropping the single-join requirement, lets a
    // star-schema query hand off — which is what TPC-DS q17 (seven joins) appears to want, since
    // it runs 1,035 ms at 10 cores streaming against 447 ms at 22 materializing in isolation. In
    // the harness it is a **net loss**, measured 2026-08-08: q50 155 -> 115 ms, q45 126 -> 112 and
    // q85 101 -> 89 improve, but **q17 320 -> 975 ms and q25 201 -> 467** — the two it was aimed
    // at regress by 3.0x and 2.3x. Handing the whole plan to the materializing executor on the
    // evidence of one oversized build is too blunt: the other joins on that spine are served fine
    // per morsel, and their probe sides lose the sharding they were getting.
    if count_hash_joins(plan) != 1 {
        return None;
    }
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

/// Whether this plan is a **join-free grouped aggregation**, which the materializing
/// executor computes for about half the CPU the streaming one spends.
///
/// Both executors parallelize this shape — that was checked rather than assumed, and the
/// `cpu=` figure `explain(analyze)` prints for the operator is misleading here: measured
/// process CPU over wall time, streaming ran it on 14.2 cores and materializing on 11.4. The
/// difference is not parallelism but *work*. On the H2O `groupby` suite at its own 1e7-row
/// tier, streaming burned roughly twice the CPU for the identical answer:
///
/// | query | streaming | materializing |
/// |---|---|---|
/// | `sum(v1), mean(v3) by id3` (1e5 groups) | 269 ms / 3,813 cpu-ms | **174 ms / 1,993 cpu-ms** |
/// | `sum(v1:v3) by id6` (1e5 groups) | 178 ms / 2,505 cpu-ms | **97 ms / 955 cpu-ms** |
/// | `sum(v3), count by id1:id6` (1e7 groups) | 1,494 ms / 16,662 cpu-ms | **753 ms / 10,107 cpu-ms** |
///
/// A 100-group aggregate is the one case streaming wins, and it wins by 4.5 ms (31.8 against
/// 36.3) — against 741 ms on the 1e7-group case. The trade is one-sided at every size measured.
///
/// **Join-free is the whole restriction, and it is deliberate.** A grouped aggregate *under*
/// a join is a different plan with different intermediates, and none of it was measured here;
/// restricting to a plan with no `HashJoin` anywhere keeps this off every TPC-H shape. The
/// caller pairs it with the same `materialize_fits` envelope guard `streaming_parallelizes`
/// uses, so a large input still keeps the bounded streaming path.
/// **Seen through a row-wise root**, because most grouped aggregates do not have the aggregate
/// at the root and the shape of the projection above it says nothing about which executor
/// should run them. `SELECT id3, max(v1) - min(v2) ... GROUP BY id3` leaves a `Project` over the
/// `Aggregate` for the subtraction, and `stddev` leaves one for its `sqrt`; on H2O `groupby`
/// those are q7 and q6, and both were routed to streaming and ran **~1.8x slower** than the
/// materializing path they qualify for on every other count — q6 90.0 ms against 47.9 ms, q7
/// 100.4 ms against 56.9 ms. The projection itself is over the aggregate's *output*, which is
/// one row per group and therefore trivial next to the aggregation, so peeling it cannot change
/// which executor is the right one.
pub fn materializing_aggregate_is_faster(plan: &RelOp) -> bool {
    fn has_join(op: &RelOp) -> bool {
        if matches!(op, RelOp::HashJoin { .. }) {
            return true;
        }
        op.children().iter().any(|c| has_join(c))
    }
    let mut node = plan;
    while let RelOp::Project { input, .. } = node {
        node = input;
    }
    matches!(node, RelOp::Aggregate { group_keys, .. } if !group_keys.is_empty()) && !has_join(plan)
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
    /// The installed pool must be no wider than the shards it will run.
    ///
    /// The regression this guards is silent and only visible in CPU time: a 96-thread pool
    /// running six shards returns the right answer at the right wall-clock, while burning four
    /// cores doing nothing. On a shared box that is the difference between a query costing what
    /// it needs and a query costing what the machine has.
    #[test]
    fn the_pool_is_no_wider_than_the_shards_it_will_run() {
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        let m = bc_arrow::DEFAULT_MORSEL_ROWS;
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let rows = 6 * m + 7; // six whole morsels and a remainder
        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(
                (0..rows as i64).collect::<Vec<_>>(),
            ))],
        )
        .unwrap();
        let sources = vec![vec![batch]];
        // A plan that does per-row work, so the *morsel-count* bound is the one under test.
        // A bare scan is capped at one worker for a different reason — see
        // `an_identity_pipeline_is_never_sharded_or_spread`.
        let plan = RelOp::Filter {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            predicate: bc_expr::Expr::Col {
                name: "a".to_string(),
            },
        };

        let capped = useful_workers(&plan, &sources, 96);
        assert_eq!(
            capped, 6,
            "six morsels of data cannot keep more than six shards busy"
        );
        assert!(
            capped >= effective_shard_count(96, rows),
            "the pool must never be narrower than the shard count it has to run"
        );
        // A width the caller asked for explicitly is still an upper bound, never raised.
        assert_eq!(useful_workers(&plan, &sources, 2), 2);
        // And an empty relation still gets a usable pool rather than a zero-width one.
        assert_eq!(useful_workers(&plan, &[vec![]], 96), 1);
    }

    /// An identity pipeline is neither sharded nor spread, however much data it carries.
    ///
    /// `read.parquet(...).collect()` reaches the executor as a bare `Scan`: the file has already
    /// been decoded and the plan's whole job is to hand those batches back. The morsel-count
    /// bound cannot see this — 3 M rows are 183 morsels, so it permits every core — and the
    /// result was that the most ordinary call a user makes paid for ~96 shards of an identity
    /// pipeline and a 96-thread pool to run them on. Measured at 9.7 ms wall / 70.2 ms CPU
    /// against 7.5 ms / 12.5 ms: slower *and* five times the CPU.
    #[test]
    fn an_identity_pipeline_is_never_sharded_or_spread() {
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        let m = bc_arrow::DEFAULT_MORSEL_ROWS;
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        // Far more morsels than cores, so nothing but the identity rule can be doing the capping.
        let rows = 200 * m;
        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(
                (0..rows as i64).collect::<Vec<_>>(),
            ))],
        )
        .unwrap();
        let sources = vec![vec![batch]];

        let scan = RelOp::Scan { source_id: 0 };
        assert!(nothing_to_parallelize(&scan));
        assert_eq!(
            useful_workers(&scan, &sources, 96),
            1,
            "a scan computes nothing per row, so no width above one can help it"
        );

        // The moment an operator sits on top there *is* per-row work, and the morsel-count
        // bound takes over — this is the boundary the rule must not overreach past.
        let filtered = RelOp::Filter {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            predicate: bc_expr::Expr::Col {
                name: "a".to_string(),
            },
        };
        assert!(!nothing_to_parallelize(&filtered));
        assert_eq!(useful_workers(&filtered, &sources, 96), 96);
    }

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
