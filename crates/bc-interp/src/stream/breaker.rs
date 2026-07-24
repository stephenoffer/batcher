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

use super::{build_with, finalize_partial, fold_partial, Ctx, Morsels};
use crate::ops;
use crate::InterpError;

/// Pull a stream to exhaustion.
pub(super) fn drain(stream: Morsels<'_>) -> Result<Vec<RecordBatch>, InterpError> {
    stream.collect()
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
            // Sequential breaker: a fresh cell, compiled once on the first morsel.
            let jit = std::sync::OnceLock::new();
            match fold_partial(build_with(input, ctx)?, group_keys, aggregates, &jit)? {
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
            let batches = drain(build_with(input, ctx)?)?;
            let rows_in = crate::count_rows(&batches);
            let held = crate::batch_bytes(&batches);
            // A sort genuinely holds its input; over budget, the external (spilling) sort in the
            // materializing executor is the right tool, so give way to it.
            ctx.check_budget(held, "the streaming sort does not spill")?;
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
            let batches = drain(build_with(input, ctx)?)?;
            if batches.is_empty() {
                return exec_deferred_breaker(plan, ctx);
            }
            let rows_in = crate::count_rows(&batches);
            let held = crate::batch_bytes(&batches);
            ctx.check_budget(held, "the streaming distinct does not spill")?;
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
            for inp in inputs {
                all.extend(drain(build_with(inp, inner)?)?);
            }
            let all = crate::coerce_union_branches(all)?;
            if all.is_empty() {
                return exec_deferred_breaker(plan, ctx);
            }
            let rows_in = crate::count_rows(&all);
            let held = crate::batch_bytes(&all);
            ctx.check_budget(held, "the streaming union-distinct does not spill")?;
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
        let batches = drain(build_with(child, measure)?)?;
        held += crate::batch_bytes(&batches);
        drained.push(batches);
    }
    ctx.check_budget(held, "this streaming breaker does not spill")?;

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
