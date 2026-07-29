//! The breakers: operators that must see all of their input before they can emit any output.
//!
//! They stay breakers deliberately. A breaker is where the adaptive layer measures an *actual*
//! cardinality and re-plans the rest of the query (CLAUDE.md invariant #10 — the moat), so
//! collapsing them would trade the differentiator for a memory win available elsewhere. What
//! streaming removes is the *incidental* materialization of everything between them.
//!
//! Their input still arrives as a stream, so the subtree below a breaker never materializes more
//! than a morsel at a time — and the aggregate does not even hold its input, folding it
//! incrementally into state bounded by the group count.
//!
//! Every kernel here is the one [`crate::execute`] calls. The breakers are not re-implemented;
//! they are handed a collected input and left alone. That is what keeps the streaming executor a
//! *scheduling* change rather than a second set of semantics.

use std::time::Instant;

use arrow::array::RecordBatch;
use bc_ir::RelOp;
use bc_runtime::agg;
use rayon::prelude::*;

use super::{build_with, finalize_partial, fold_partial, Ctx, Morsels};
use crate::ops;
use crate::InterpError;

/// Morsels each worker gets per parallel fold round.
///
/// The round is the unit of buffering, so this is what the "streaming" aggregate holds of its
/// input at once: `workers x` this many morsels. Two keeps every worker fed across a round
/// boundary (one to fold, one queued) while leaving the buffer a small multiple of a morsel —
/// at the 16,384-row default and 96 workers, a few hundred MB at the very widest, and
/// proportional to the machine rather than to the relation.
const PAR_FOLD_MORSELS_PER_WORKER: usize = 2;

/// Pull a stream to exhaustion.
pub(super) fn drain(stream: Morsels<'_>) -> Result<Vec<RecordBatch>, InterpError> {
    stream.collect()
}

/// Pull a stream to exhaustion, **stopping** the moment what has been accumulated exceeds
/// `budget`.
///
/// The breakers below give way to the spilling executor when their input does not fit, which
/// is what turns a would-be OOM into a spill. That handoff only works if it happens *before*
/// the allocation it is avoiding. Draining first and checking afterwards inverts it: an input
/// ten times the envelope is ten envelopes of resident memory before a single byte of the
/// check runs, so the process dies at the drain and the executor that could have spilled is
/// never reached. The guard was, in the case it exists for, unreachable.
///
/// Checking per morsel bounds the held bytes at roughly `budget` plus one morsel. The
/// reported `needed` is therefore where the accumulation crossed the line rather than the
/// input's true size — a lower bound, and deliberately so: knowing the exact total requires
/// holding the whole thing, which is the thing being prevented. The decision it feeds ("this
/// does not fit, use the executor that spills") is the same either way.
///
/// `budget == 0` is "unbounded", where there is nothing to enforce and no spilling path to
/// prefer, so this is exactly [`drain`].
fn drain_within_budget(
    stream: Morsels<'_>,
    budget: usize,
    reason: &'static str,
) -> Result<Vec<RecordBatch>, InterpError> {
    if budget == 0 {
        return drain(stream);
    }
    let mut held: u64 = 0;
    let mut out = Vec::new();
    for batch in stream {
        let batch = batch?;
        held += batch.get_array_memory_size() as u64;
        if held as usize > budget {
            // `out` is dropped on the way out, so the bail releases what it accumulated.
            return Err(InterpError::MemoryBudgetExceeded {
                needed: held as usize,
                budget,
                reason,
            });
        }
        out.push(batch);
    }
    Ok(out)
}

/// [`fold_partial`], with the per-morsel `partial` step spread across `workers`.
///
/// The sequential fold is the right shape when this breaker is *inside* a sharded worker (there
/// the parallelism is the shard, and `Ctx::workers` is 1 to say so). On the **un-sharded**
/// fallback path it is not: that path is taken precisely when the plan could not be split — a
/// self-join, or a hash join whose build side is past the per-morsel probe's ceiling — and the
/// aggregate then folds the *whole* relation on one core while every other core idles. It is the
/// dominant term when it happens: a 60M-row `lineitem` join-then-group-by measured 2.7 s, of
/// which ~1.9 s was this loop.
///
/// Rounds are the only difference. Morsels are taken from the stream **in order** into a bounded
/// buffer, `partial`-ed in parallel, and combined in that same order — so this is the mergeable
/// algebra (invariant #7) with a wider `partial` step, exactly as the sharded aggregate in
/// `stream::parallel` already runs it, and the accumulated partial folds in last on each round
/// just as the sequential loop does.
///
/// Only the `partial` step fans out; the stream is still pulled one morsel at a time by this
/// thread. That is deliberate — the pull is where the *upstream* pipeline runs, and on this path
/// the upstream has either already materialized (the join fallback) or is un-shardable by
/// construction, so pulling it from several threads would be unsound rather than merely
/// unhelpful.
///
/// An oversized batch is **sliced here** rather than upstream. The un-shardable join emits its
/// output as one relation-sized batch on purpose (splitting it there makes the next join's
/// `materialize` copy the whole relation — see `materialized_join_from`), so without this the
/// round would hold exactly one unit of work and fold on one core. Slicing is zero-copy and
/// local to this fold, so it costs the pipeline nothing and buys the fan-out.
fn fold_partial_parallel(
    input: Morsels<'_>,
    group_keys: &[bc_ir::ProjectionItem],
    aggregates: &[bc_ir::AggregateItem],
    jit: &std::sync::OnceLock<ops::AggJit>,
    workers: usize,
) -> Result<(Option<agg::Partial>, u64), InterpError> {
    let funcs = ops::agg_funcs(aggregates);
    let per_round = workers.saturating_mul(PAR_FOLD_MORSELS_PER_WORKER).max(2);
    let mut buf: Vec<RecordBatch> = Vec::with_capacity(per_round);
    let mut folded: Option<agg::Partial> = None;
    let mut rows_in: u64 = 0;

    for morsel in input {
        let morsel = morsel?;
        if morsel.num_rows() == 0 {
            continue;
        }
        rows_in += morsel.num_rows() as u64;
        for piece in slice_to_morsels(morsel) {
            buf.push(piece);
            if buf.len() >= per_round {
                folded = Some(fold_round(
                    &mut buf, group_keys, aggregates, jit, &funcs, folded,
                )?);
            }
        }
    }
    if !buf.is_empty() {
        folded = Some(fold_round(
            &mut buf, group_keys, aggregates, jit, &funcs, folded,
        )?);
    }
    Ok((folded, rows_in))
}

/// One batch as morsel-sized, in-order, **zero-copy** slices — itself when it already fits.
///
/// `RecordBatch::slice` shares the parent's buffers, so this changes only where the fold's units
/// of work begin and end. Order is the slice order, which is the batch's own row order.
fn slice_to_morsels(batch: RecordBatch) -> Vec<RecordBatch> {
    let rows = batch.num_rows();
    if rows <= bc_arrow::DEFAULT_MORSEL_ROWS {
        return vec![batch];
    }
    (0..rows)
        .step_by(bc_arrow::DEFAULT_MORSEL_ROWS)
        .map(|start| batch.slice(start, bc_arrow::DEFAULT_MORSEL_ROWS.min(rows - start)))
        .collect()
}

/// One round of [`fold_partial_parallel`]: `partial` every buffered morsel in parallel, then
/// combine them (and the partial carried in from earlier rounds) into one.
fn fold_round(
    buf: &mut Vec<RecordBatch>,
    group_keys: &[bc_ir::ProjectionItem],
    aggregates: &[bc_ir::AggregateItem],
    jit: &std::sync::OnceLock<ops::AggJit>,
    funcs: &[agg::AggFunc],
    carried: Option<agg::Partial>,
) -> Result<agg::Partial, InterpError> {
    // Compiled once per query, before the fan-out, and shared by every worker — the same
    // `OnceLock` contract `fold_partial` documents. Compiling inside the `par_iter` would pay
    // Cranelift's per-expression cost once per core.
    let compiled = jit.get_or_init(|| ops::compile_agg(group_keys, aggregates, &buf[0]));
    let mut partials: Vec<agg::Partial> = buf
        .par_iter()
        .map(|m| ops::eval_partial_jit(m, group_keys, aggregates, compiled))
        .collect::<Result<Vec<_>, _>>()?;
    buf.clear();
    if let Some(prev) = carried {
        partials.push(prev);
    }
    agg::combine(&partials, funcs).map_err(Into::into)
}

/// Run a breaker over its (streamed) input and return its materialized output.
pub(super) fn exec_breaker(plan: &RelOp, ctx: Ctx<'_>) -> Result<Vec<RecordBatch>, InterpError> {
    let id = ctx.id(plan);
    match plan {
        RelOp::Aggregate {
            input,
            group_keys,
            aggregates,
        } => {
            let t = Instant::now();
            // Folded morsel by morsel, so the aggregate's *input* is never held — only its state,
            // which is O(groups). The fold reports the rows it consumed, which is this operator's
            // `rows_in`: Kyber's selectivity model reads `rows_out / rows_in`, so a zero here
            // would not be a missing number, it would be a *wrong* one.
            // A fresh cell, compiled once on the first morsel that carries rows.
            let jit = std::sync::OnceLock::new();
            // `ctx.workers` is 1 inside a sharded worker (the shard *is* the parallelism, and
            // fanning out again would nest rayon) and the real width on the un-sharded fallback
            // path — where nothing else is spreading this fold across the machine.
            let stream = build_with(input, ctx)?;
            let folded = if ctx.workers > 1 {
                fold_partial_parallel(stream, group_keys, aggregates, &jit, ctx.workers)?
            } else {
                fold_partial(stream, group_keys, aggregates, &jit)?
            };
            match folded {
                (Some(merged), rows_in) => {
                    // Both halves of the state, not just the keys. The accumulator columns
                    // are the half that can actually be unbounded: a holistic aggregate
                    // (`median`/`quantile`/`n_unique`/`mode`/`listagg`) keeps every value it
                    // has seen per group, so its state grows with the *input*, while the keys
                    // grow only with the group count. Counting keys alone let the one shape
                    // that OOMs here — few groups, huge per-group value lists — read as
                    // kilobytes and sail past the check.
                    let state_bytes = merged
                        .group_columns
                        .iter()
                        .chain(merged.states.iter().flatten())
                        .map(|c| c.get_array_memory_size() as u64)
                        .sum::<u64>();
                    // The streaming aggregate folds in memory. A group count too large for the
                    // envelope is exactly the case the materializing executor spills, so hand the
                    // query back rather than OOM where it would have survived.
                    ctx.check_budget(state_bytes, "the streaming aggregate does not spill")?;
                    let out = finalize_partial(&merged, group_keys, aggregates)?;
                    if let (Some(m), Some(id)) = (ctx.meter, id) {
                        // Peak is the *state* plus the result — not the input, which this
                        // executor never holds. That is a smaller and truer number than the
                        // materializing path's, and Carbonite should reserve against it.
                        m.breaker(
                            id,
                            rows_in,
                            0,
                            state_bytes,
                            &out,
                            t.elapsed().as_nanos() as u64,
                        );
                    }
                    Ok(out)
                }
                // No rows reached the aggregate at all. That is not the same as "no output": a
                // global `COUNT` over nothing is one row holding 0, and a `SUM` is one row holding
                // NULL. The oracle already defines that; defer to it rather than reproduce it.
                (None, _) => crate::execute(plan, ctx.sources),
            }
        }

        RelOp::Sort { input, keys, limit } => {
            // A sort genuinely holds its input; over budget, the external (spilling) sort in
            // the materializing executor is the right tool, so give way to it — and give way
            // *while draining*, so the input that does not fit is never fully resident.
            let batches = drain_within_budget(
                build_with(input, ctx)?,
                ctx.budget,
                "the streaming sort does not spill",
            )?;
            let rows_in = crate::count_rows(&batches);
            let held = crate::batch_bytes(&batches);
            let t = Instant::now();
            let out = match limit {
                // Top-N: reduce each morsel to its local top-k in parallel and merge the narrow
                // survivors — never concatenate or sort the whole input. `ops::parallel_top_n` is
                // the mergeable top-N, result-identical to a full sort-then-slice (asserted by
                // `parallel_top_n_matches_eager`), so this only changes throughput. The old path
                // `materialize`d all N rows into one batch and ran a `LIMIT`-ed `lexsort` over
                // every one of them — O(N) row-format encoding to keep k rows.
                Some(k) if !batches.is_empty() => vec![ops::parallel_top_n(&batches, keys, *k)?],
                // Full sort (no LIMIT), or the empty input: materialize, then try the parallel
                // sample-sort (range-partition + per-range parallel sort for a large float / int
                // / string key). It returns the ranges already in key order — their concatenation
                // is the sorted relation — and is result-identical to the serial `sort_batch`
                // oracle (`sample_sort` tests). It declines (`None`) for a small input, a top-N
                // `LIMIT`, or an unsupported key, where the serial sort runs. Without this a large
                // full sort ran arrow's single-threaded `lexsort` — ~16x DuckDB on a 6M-row sort.
                _ => match ops::materialize(&batches) {
                    Ok(combined) => match ops::parallel_sort_batch(&combined, keys, *limit)? {
                        Some(sorted) => sorted,
                        None => vec![ops::sort_batch(&combined, keys, *limit)?],
                    },
                    Err(_) => Vec::new(),
                },
            };
            if let (Some(m), Some(id)) = (ctx.meter, id) {
                // A sort genuinely does hold its input — it is a full breaker — so its peak is
                // that input plus the sorted result. Streaming changes nothing here, and the
                // metric says so honestly.
                m.breaker(id, rows_in, 0, held, &out, t.elapsed().as_nanos() as u64);
            }
            Ok(out)
        }

        // `DISTINCT` over all columns is a mergeable all-column group-by, so dedup it in
        // parallel here rather than on the single-threaded oracle the deferred path below would
        // use. Drain the input, and — exactly like the `Sort` breaker — give way to the spilling
        // executor if the held input exceeds the envelope (that path dedups out of core); with no
        // envelope (`budget == 0`, the common case) `check_budget` admits and the parallel dedup
        // runs. Empty input defers so the oracle supplies the correctly-typed empty relation.
        // Without this a 6M-row DISTINCT ran single-threaded — ~7x DuckDB.
        RelOp::Distinct { input } => {
            let t = Instant::now();
            let batches = drain_within_budget(
                build_with(input, ctx)?,
                ctx.budget,
                "the streaming distinct does not spill",
            )?;
            if batches.is_empty() {
                return exec_deferred_breaker(plan, ctx);
            }
            let rows_in = crate::count_rows(&batches);
            let held = crate::batch_bytes(&batches);
            let out = ops::parallel_distinct(&batches)?;
            if let (Some(m), Some(id)) = (ctx.meter, id) {
                m.breaker(id, rows_in, 0, held, &out, t.elapsed().as_nanos() as u64);
            }
            Ok(out)
        }

        // `UNION` (DISTINCT) is a concat of its branches followed by an all-column dedup — the
        // same mergeable dedup as `Distinct`, so parallelize it here too rather than on the
        // single-threaded oracle. Branches are coerced to the common supertype first (exactly as
        // the oracle does), then deduped in parallel; over an envelope the spilling executor takes
        // over (`check_budget`). `UNION ALL` (`distinct: false`) just streams the concat and stays
        // on the deferred path below. Without this a large `UNION` ran single-threaded — ~6x DuckDB.
        RelOp::Union {
            inputs,
            distinct: true,
        } => {
            let t = Instant::now();
            let inner = Ctx::new(ctx.sources, ctx.cache, None, ctx.budget).with_mats(ctx.mats);
            let mut all = Vec::new();
            let mut held_so_far: u64 = 0;
            for inp in inputs {
                // The envelope covers the whole union, so each branch is drained against
                // what the previous ones already hold rather than against the full budget.
                let remaining = ctx.budget.saturating_sub(held_so_far as usize);
                let branch = drain_within_budget(
                    build_with(inp, inner)?,
                    if ctx.budget == 0 { 0 } else { remaining.max(1) },
                    "the streaming union-distinct does not spill",
                )?;
                held_so_far += crate::batch_bytes(&branch);
                all.extend(branch);
            }
            let all = crate::coerce_union_branches(all)?;
            if all.is_empty() {
                return exec_deferred_breaker(plan, ctx);
            }
            let rows_in = crate::count_rows(&all);
            let held = crate::batch_bytes(&all);
            let out = ops::parallel_distinct(&all)?;
            if let (Some(m), Some(id)) = (ctx.meter, id) {
                m.breaker(id, rows_in, 0, held, &out, t.elapsed().as_nanos() as u64);
            }
            Ok(out)
        }

        // Everything else — `Window`, `Sample`, `AsofJoin`, `UNION ALL` — is run by the
        // sequential oracle over this subtree.
        //
        // That is a deliberate boundary, not an oversight. Each has a reason its streaming form
        // is more than a scheduling change: `Distinct` and `Window` need the spill-aware state the
        // oracle already threads; `Sample` draws against the whole relation, so a per-morsel draw
        // would be a *different* sample, not a faster one; `Union` must coerce its branches to a
        // common supertype it cannot know until it has seen them; `AsofJoin` orders both sides.
        // Handing them to the oracle keeps this executor honest — it streams what it can prove it
        // may, and defers the rest rather than guessing.
        //
        // But the oracle materializes the whole subtree in RAM, so under a memory envelope it
        // would OOM exactly where the spilling parallel executor survives (`Distinct`,
        // `Window` WITH PARTITION BY, `AsofJoin` WITH `by` keys, and `Union DISTINCT` all
        // grace-partition out of core there). `exec_deferred_breaker` closes that hole: it bounds
        // the oracle by the envelope and gives way to the spilling executor when the input does
        // not fit — turning a would-be OOM into a spill, never a crash.
        _ => exec_deferred_breaker(plan, ctx),
    }
}

/// Run a deferred breaker (`Distinct`/`Window`/`Sample`/`AsofJoin`/`Union`) under the memory
/// envelope. The streaming executor does not re-implement these — it hands them to the
/// sequential oracle — but the oracle holds their whole input in RAM, so a naive deferral OOMs
/// under a tight budget where the *spilling* parallel executor would not.
///
/// This mirrors the streaming `Sort` breaker: drain the input(s), measure the materialized bytes,
/// and if they exceed the envelope give way with [`InterpError::MemoryBudgetExceeded`] — the
/// caller (`bc_py::execute_plan`) then re-runs the whole plan on the executor that spills. Within
/// budget, the oracle runs over the *already-drained* input, wired in as synthetic `Scan`s, so no
/// source is read twice and the streaming run above continues exactly as before. With no envelope
/// (`budget == 0`) there is nothing to enforce and no spill path to prefer, so the subtree defers
/// to the oracle unchanged.
fn exec_deferred_breaker(plan: &RelOp, ctx: Ctx<'_>) -> Result<Vec<RecordBatch>, InterpError> {
    if ctx.budget == 0 {
        return crate::execute(plan, ctx.sources);
    }
    let children = plan.children();
    // Drain each child with a meter-less context: a deferred subtree reports no metrics on this
    // tier (the oracle emits its own), and draining to measure must not change that. The drained
    // batches are the same bytes the oracle would hold, so they are the honest quantity to budget.
    let measure = Ctx::new(ctx.sources, ctx.cache, None, ctx.budget).with_mats(ctx.mats);
    let mut drained: Vec<Vec<RecordBatch>> = Vec::with_capacity(children.len());
    let mut held: u64 = 0;
    for child in &children {
        // Budget against what the earlier children already hold: the oracle will hold all of
        // them at once, and draining a child that alone exceeds the envelope OOMs before the
        // check below could hand the query to the executor that spills.
        let remaining = ctx.budget.saturating_sub(held as usize);
        let batches = drain_within_budget(
            build_with(child, measure)?,
            remaining.max(1),
            "this streaming breaker does not spill",
        )?;
        held += crate::batch_bytes(&batches);
        drained.push(batches);
    }

    // Within budget: run the oracle over the drained inputs, wired in as synthetic scans so the
    // top operator runs exactly once over exactly the rows the oracle would have produced —
    // identical result, no re-scan.
    let Some(rewritten) = rebuild_with_scan_children(plan, ctx.sources.len()) else {
        // Not one of the deferred breakers (unreachable — `build_with` routes only those here).
        return crate::execute(plan, ctx.sources);
    };
    let mut sources: Vec<Vec<RecordBatch>> = ctx.sources.to_vec();
    sources.extend(drained);
    crate::execute(&rewritten, &sources)
}

/// Clone a deferred breaker with each child subtree replaced by a `Scan` of a synthetic source
/// (`base + child_index`), so the oracle can run the top operator over already-materialized input.
/// Returns `None` for any node that is not a deferred breaker (never reached in practice).
fn rebuild_with_scan_children(plan: &RelOp, base: usize) -> Option<RelOp> {
    let scan = |i: usize| {
        Box::new(RelOp::Scan {
            source_id: base + i,
        })
    };
    Some(match plan {
        RelOp::Distinct { .. } => RelOp::Distinct { input: scan(0) },
        RelOp::Window {
            partition_keys,
            order_keys,
            functions,
            rank_limit,
            ..
        } => RelOp::Window {
            input: scan(0),
            partition_keys: partition_keys.clone(),
            order_keys: order_keys.clone(),
            functions: functions.clone(),
            rank_limit: *rank_limit,
        },
        RelOp::Sample {
            fraction, seed, n, ..
        } => RelOp::Sample {
            input: scan(0),
            fraction: *fraction,
            seed: *seed,
            n: *n,
        },
        RelOp::Union { inputs, distinct } => RelOp::Union {
            inputs: (0..inputs.len())
                .map(|i| RelOp::Scan {
                    source_id: base + i,
                })
                .collect(),
            distinct: *distinct,
        },
        RelOp::AsofJoin {
            left_on,
            right_on,
            left_by,
            right_by,
            backward,
            output,
            ..
        } => RelOp::AsofJoin {
            left: scan(0),
            right: scan(1),
            left_on: left_on.clone(),
            right_on: right_on.clone(),
            left_by: left_by.clone(),
            right_by: right_by.clone(),
            backward: *backward,
            output: output.clone(),
        },
        _ => return None,
    })
}
