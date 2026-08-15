//! The breakers: operators that must see all of their input before they can emit any output.
//!
//! They stay breakers deliberately. A breaker is where the adaptive layer measures an *actual*
//! cardinality and re-plans the rest of the query (CLAUDE.md invariant #10 — the moat), so
//! collapsing them would trade the differentiator for a memory win available elsewhere. What
//! streaming removes is the *incidental* materialization of everything between them.
//!
//! Their input still arrives as a stream, so the subtree below a breaker never materializes more
//! than a morsel at a time. Three of them do not hold their input at all, folding it into state
//! bounded by their own *answer* instead (`super::folds`): the aggregate's groups, a top-N's `k`
//! survivors, and a reducing dedup's distinct rows.
//!
//! Every kernel here is the one [`crate::execute`] calls. The breakers are not re-implemented;
//! they are handed a collected input and left alone. That is what keeps the streaming executor a
//! *scheduling* change rather than a second set of semantics.

use std::time::Instant;

use arrow::array::RecordBatch;
use bc_ir::RelOp;

use super::folds::{fold_partial_parallel, stream_distinct, stream_sample_n, stream_top_n};
use super::union_all;
use super::{build_with, finalize_partial, fold_partial, Ctx, Morsels};
use crate::ops;
use crate::InterpError;

/// How many morsels a limited `DISTINCT` reads before deciding its early exit will not pay.
///
/// The exit wins when `k` distinct rows turn up in a short prefix. When they have not by here,
/// the key is low-cardinality relative to the limit, and the whole-column dense direct-map is
/// the faster answer — measured at 4.9x DuckDB on that shape, which an unconditional early exit
/// would hand back. Sixteen morsels is ~262,000 rows: long enough that a genuinely
/// high-cardinality key has filled the prefix many times over, short enough that the wasted
/// probe is a rounding error against the scan it saves.
pub(super) const PREFIX_PROBE_MORSELS: usize = 16;

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
                        // Only when something was actually compiled: the cell is empty if no
                        // morsel carried rows, and the default tag is already `"interp"`.
                        if let Some(compiled) = jit.get() {
                            m.note_backend(id, compiled.backend_tag());
                        }
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
            // Top-N (`ORDER BY … LIMIT k`) is **not** a breaker on this path: its state is the k
            // rows it is keeping, so it folds the input away morsel by morsel and never holds it.
            // `parallel_top_n` was already the mergeable top-N — reduce each morsel to its local
            // top-k, merge the narrow survivors, break ties by original position — but it was
            // handed a fully drained input, so a `SELECT … ORDER BY … LIMIT 10` over a hundred
            // million rows still had all hundred million resident to keep ten. See `stream_top_n`.
            if let Some(k) = limit {
                let t = Instant::now();
                let (kept, rows_in, held) = stream_top_n(build_with(input, ctx)?, keys, *k, ctx)?;
                let out: Vec<RecordBatch> = kept.into_iter().collect();
                if let (Some(m), Some(id)) = (ctx.meter, id) {
                    // The peak is the fold's, not the input's — a genuinely smaller and truer
                    // number than the drained path reported, and the one Carbonite should
                    // reserve against.
                    m.breaker(id, rows_in, 0, held, &out, t.elapsed().as_nanos() as u64);
                }
                return Ok(out);
            }
            // A full sort genuinely holds its input; over budget, the external (spilling) sort in
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
            // Materialize, then try the parallel sample-sort (range-partition + per-range parallel
            // sort for a large float / int / string key). It returns the ranges already in key
            // order — their concatenation is the sorted relation — and is result-identical to the
            // serial `sort_batch` oracle (`sample_sort` tests). It declines (`None`) for a small
            // input or an unsupported key, where the serial sort runs. Without this a large full
            // sort ran arrow's single-threaded `lexsort` — ~16x DuckDB on a 6M-row sort.
            let out = match ops::materialize(&batches) {
                Ok(combined) => match ops::parallel_sort_batch(&combined, keys, *limit)? {
                    Some(sorted) => sorted,
                    None => vec![ops::sort_batch(&combined, keys, *limit)?],
                },
                Err(_) => Vec::new(),
            };
            if let (Some(m), Some(id)) = (ctx.meter, id) {
                // A sort genuinely does hold its input — it is a full breaker — so its peak is
                // that input plus the sorted result. Streaming changes nothing here, and the
                // metric says so honestly.
                m.breaker(id, rows_in, 0, held, &out, t.elapsed().as_nanos() as u64);
            }
            Ok(out)
        }

        // Dedup is a mergeable reduction, so run it in parallel here rather than on the
        // single-threaded oracle the deferred path below would use. Drain the input, and —
        // exactly like the `Sort` breaker — give way to the spilling executor if the held input
        // exceeds the envelope (that path dedups out of core); with no envelope (`budget == 0`,
        // the common case) `check_budget` admits and the parallel dedup runs. Empty input defers
        // so the oracle supplies the correctly-typed empty relation. Without this a 6M-row
        // DISTINCT ran single-threaded — ~7x DuckDB.
        RelOp::Distinct {
            input,
            keys,
            order,
            limit,
        } => {
            let t = Instant::now();

            // A whole-row `DISTINCT` under a fused `LIMIT k` stops as soon as `k` distinct rows
            // exist, so it pulls a *prefix* of its input rather than draining it. That is the
            // point of the fusion: the work becomes proportional to how far in the input has to
            // be read to find `k` distinct rows, not to the input. It also needs no budget
            // check, because `DistinctPrefix` never holds more than `k` rows and the morsels
            // that fed it are dropped as they are consumed — this is the one breaker here that
            // is not one.
            //
            // `DISTINCT ON` is excluded: `order` chooses which row survives per key, so a
            // surviving row can be replaced by a later one and no prefix of the input
            // determines the answer.
            if let Some(k) = limit {
                // `DISTINCT ON` with a limit goes to the oracle rather than the parallel dedup
                // below. `ops::parallel_distinct_on` emits *bucket* order, not first-seen
                // order, so truncating its output would keep a different `k` than the oracle
                // does — the two tiers would disagree on the answer, which is invariant #6.
                // Kyber only fuses a limit into a whole-row `DISTINCT`, so this arm costs
                // nothing in practice; it exists so a hand-written plan cannot diverge.
                if !keys.is_empty() {
                    return exec_deferred_breaker(plan, ctx);
                }
                let mut stream = build_with(input, ctx)?;
                let mut acc = bc_runtime::agg::DistinctPrefix::new(*k);
                let mut probed: Vec<RecordBatch> = Vec::new();
                let mut rows_in = 0usize;
                let mut gave_up = false;
                for batch in stream.by_ref() {
                    let batch = batch?;
                    rows_in += batch.num_rows();
                    acc.push(&batch)?;
                    if acc.is_satisfied() {
                        break;
                    }
                    probed.push(batch);
                    if probed.len() >= PREFIX_PROBE_MORSELS {
                        gave_up = true;
                        break;
                    }
                }
                if !gave_up {
                    // Either the prefix filled (the early exit, and the whole point) or the
                    // input ran out first, in which case what it holds is every distinct row
                    // and the limit was never binding. Both are the finished answer.
                    //
                    // Nothing pushed means an empty input: defer, so the oracle supplies the
                    // correctly-typed empty relation exactly as the drained path below does.
                    let Some(out) = acc.finish() else {
                        return exec_deferred_breaker(plan, ctx);
                    };
                    let out = vec![out];
                    if let (Some(m), Some(id)) = (ctx.meter, id) {
                        m.breaker(
                            id,
                            rows_in as u64,
                            0,
                            0,
                            &out,
                            t.elapsed().as_nanos() as u64,
                        );
                    }
                    return Ok(out);
                }

                // The probe window closed without `k` distinct rows, so the key is
                // low-cardinality relative to the limit and the early exit is not going to pay.
                // Measured rather than estimated, which matters: the planner's fallback for an
                // unmeasured column is half the row count, and acting on that would fire the
                // exit on exactly the shape where the dense direct-map already wins by 4.9x.
                // This is the same shape `agg_par.rs` uses to pick an aggregate strategy on a
                // measured reduction ratio.
                //
                // So: drain the rest and dedup the ordinary way. `parallel_distinct` emits
                // bucket order, which is fine only while the limit cannot bind — when it turns
                // out there are more than `k` distinct rows after all, the ordered prefix is
                // recomputed over the input just drained, which is the one case that pays
                // twice and is bounded by a single extra pass.
                for batch in stream {
                    let batch = batch?;
                    rows_in += batch.num_rows();
                    probed.push(batch);
                }
                let held = crate::batch_bytes(&probed);
                let full = ops::parallel_distinct(&probed)?;
                let distinct_rows: usize = full.iter().map(|b| b.num_rows()).sum();
                let out = match distinct_rows > *k {
                    false => full,
                    true => bc_runtime::agg::distinct_prefix(&probed, *k)?
                        .into_iter()
                        .collect(),
                };
                if let (Some(m), Some(id)) = (ctx.meter, id) {
                    m.breaker(
                        id,
                        rows_in as u64,
                        0,
                        held,
                        &out,
                        t.elapsed().as_nanos() as u64,
                    );
                }
                return Ok(out);
            }

            // A whole-row `DISTINCT` folds its input away round by round (`stream_distinct`)
            // instead of holding it: the state is the surviving rows, which for the shape anyone
            // writes a `DISTINCT` for is a small fraction of what produced it. `DISTINCT ON` keeps
            // the drained path — `order` decides which row per key survives, and proving a fold
            // preserves that is a different argument from the whole-row dedup's idempotence.
            let (out, rows_in, held) = if keys.is_empty() {
                stream_distinct(build_with(input, ctx)?, ctx)?
            } else {
                let batches = drain_within_budget(
                    build_with(input, ctx)?,
                    ctx.budget,
                    "the streaming distinct does not spill",
                )?;
                let (rows_in, held) = (crate::count_rows(&batches), crate::batch_bytes(&batches));
                match batches.is_empty() {
                    true => (Vec::new(), rows_in, held),
                    false => (
                        ops::parallel_distinct_on(&batches, keys, order)?,
                        rows_in,
                        held,
                    ),
                }
            };
            // Nothing reached the operator at all: defer, so the oracle supplies the
            // correctly-typed empty relation rather than a schema-less nothing.
            if out.is_empty() {
                return exec_deferred_breaker(plan, ctx);
            }
            if let (Some(m), Some(id)) = (ctx.meter, id) {
                m.breaker(id, rows_in, 0, held, &out, t.elapsed().as_nanos() as u64);
            }
            Ok(out)
        }

        // `UNION` (DISTINCT) is a concat of its branches followed by an all-column dedup — the
        // same mergeable dedup as `Distinct`, so parallelize it here too rather than on the
        // single-threaded oracle. Branches are coerced to the common supertype first (exactly as
        // the oracle does), then deduped in parallel; over an envelope the spilling executor takes
        // over (`check_budget`). Without this a large `UNION` ran single-threaded — ~6x DuckDB.
        RelOp::Union {
            inputs,
            distinct: true,
        } => {
            let t = Instant::now();
            // The concat half of a `UNION DISTINCT` is a `UNION ALL`, and the dedup half is a
            // whole-row `DISTINCT`. Both already stream, so composing them streams: fold the
            // union's morsels into the dedup's survivors and hold neither branch. The `None` id
            // keeps the union's own metric a *breaker* record made below rather than a
            // per-morsel pipeline one — this operator does hold state, unlike a `UNION ALL`.
            //
            // `build_union_all` declines when the branch types cannot be settled from a peeked
            // morsel each, which is the one thing the drained path below can do that this
            // cannot; that path stays for exactly those.
            if let Some(stream) = union_all::build_union_all(inputs, ctx, None)? {
                let (out, rows_in, held) = stream_distinct(stream, ctx)?;
                if out.is_empty() {
                    return exec_deferred_breaker(plan, ctx);
                }
                if let (Some(m), Some(id)) = (ctx.meter, id) {
                    m.breaker(id, rows_in, 0, held, &out, t.elapsed().as_nanos() as u64);
                }
                return Ok(out);
            }
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

        // A fixed-count `SAMPLE n` keeps the `n` smallest-hash rows of the whole relation, so it
        // cannot be drawn per morsel — a per-morsel draw would keep `n` rows from *every* morsel,
        // which is a different sample rather than a faster one. But it is a top-N by hash, and
        // `sample_n_batches` already computes it with a bounded heap, so it folds exactly as the
        // top-N does and its state is the `n` rows it is keeping. See `folds::stream_sample_n`.
        RelOp::Sample {
            input,
            seed,
            n: Some(n),
            ..
        } => {
            let t = Instant::now();
            let (out, rows_in, held) = stream_sample_n(build_with(input, ctx)?, *n, *seed, ctx)?;
            if let (Some(m), Some(id)) = (ctx.meter, id) {
                m.breaker(id, rows_in, 0, held, &out, t.elapsed().as_nanos() as u64);
            }
            Ok(out)
        }

        // Everything else — `Window` and `AsofJoin` — is run by the sequential oracle over this
        // subtree.
        //
        // That is a deliberate boundary, not an oversight. Each has a reason its streaming form
        // is more than a scheduling change: `Window` needs the spill-aware state the oracle
        // already threads, and `AsofJoin` orders both sides. Handing them to the oracle keeps
        // this executor honest — it streams what it can prove it may, and defers the rest rather
        // than guessing.
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
    let t = Instant::now();
    let id = ctx.id(plan);
    let children = plan.children();
    // Drain each child through **this** executor, whether or not there is an envelope to
    // enforce. Handing the whole subtree to `crate::execute` instead — which is what the
    // unbudgeted case used to do, and `budget == 0` is the default — threw away the streaming
    // of everything below: the oracle is a tree-walk that returns every operator's full output,
    // so a `WINDOW` over a chain of joins held every one of those joins' results at once, on a
    // path whose entire purpose is not to. Only the deferred operator itself needs its input
    // materialized, and only its own input.
    let measure = Ctx::new(ctx.sources, ctx.cache, ctx.meter, ctx.budget).with_mats(ctx.mats);
    let mut drained: Vec<Vec<RecordBatch>> = Vec::with_capacity(children.len());
    let mut held: u64 = 0;
    for child in &children {
        // Budget against what the earlier children already hold: the oracle will hold all of
        // them at once, and draining a child that alone exceeds the envelope OOMs before the
        // check below could hand the query to the executor that spills. `budget == 0` is
        // "unbounded", where `drain_within_budget` is exactly `drain`.
        let remaining = match ctx.budget {
            0 => 0,
            b => b.saturating_sub(held as usize).max(1),
        };
        let batches = drain_within_budget(
            build_with(child, measure)?,
            remaining,
            "this streaming breaker does not spill",
        )?;
        held += crate::batch_bytes(&batches);
        drained.push(batches);
    }
    let rows_in = drained.iter().map(|b| crate::count_rows(b)).sum::<u64>();

    // Within budget: run the oracle over the drained inputs, wired in as synthetic scans so the
    // top operator runs exactly once over exactly the rows the oracle would have produced —
    // identical result, no re-scan.
    let Some(rewritten) = rebuild_with_scan_children(plan, ctx.sources.len()) else {
        // Not one of the deferred breakers (unreachable — `build_with` routes only those here).
        return crate::execute(plan, ctx.sources);
    };
    let mut sources: Vec<Vec<RecordBatch>> = ctx.sources.to_vec();
    sources.extend(drained);
    let out = crate::execute(&rewritten, &sources)?;
    if let (Some(m), Some(id)) = (ctx.meter, id) {
        // The deferred operators used to report nothing at all on this tier: the oracle they
        // are handed to emits into its own `ExecMetrics`, which this executor never reads. A
        // `WINDOW` or an `ASOF JOIN` therefore left a hole in exactly the feedback Kyber
        // re-plans from. Its peak genuinely is its held input plus its result — the operator
        // materializes, that is why it is here — so the number is honest as well as present.
        m.breaker(id, rows_in, 0, held, &out, t.elapsed().as_nanos() as u64);
    }
    Ok(out)
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
        RelOp::Distinct {
            keys, order, limit, ..
        } => RelOp::Distinct {
            input: scan(0),
            keys: keys.clone(),
            order: order.clone(),
            limit: *limit,
        },
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
            direction,
            tolerance,
            allow_exact_matches,
            output,
            ..
        } => RelOp::AsofJoin {
            left: scan(0),
            right: scan(1),
            left_on: left_on.clone(),
            right_on: right_on.clone(),
            left_by: left_by.clone(),
            right_by: right_by.clone(),
            direction: *direction,
            tolerance: *tolerance,
            allow_exact_matches: *allow_exact_matches,
            output: output.clone(),
        },
        _ => return None,
    })
}
