//! The mergeable folds a streaming breaker reduces its input with.
//!
//! Three operators here need their whole input only in the sense that they need to have *seen* it
//! — an aggregate's groups, a top-N's `k` survivors, a dedup's distinct rows. Each is a
//! `partial`/`combine` reduction (invariant #7), so each can consume its input morsel by morsel
//! and keep state proportional to its *answer* rather than to what produced it. That is the whole
//! content of this module: the loop that pulls a stream into bounded state, once per operator.
//!
//! `breaker.rs` owns which operator runs and what it reports; this owns how the reduction is
//! driven. They split because the reductions share their shape — buffer a round, reduce it
//! against the carried state, repeat — and reading them side by side is what shows that they do.
//!
//! Every kernel called here is the one [`crate::execute`] calls. Folding changes when the kernel
//! runs, never what it computes.

use arrow::array::RecordBatch;
use bc_runtime::agg;
use rayon::prelude::*;

use super::{Ctx, Morsels};
use crate::ops;
use crate::InterpError;

/// How many partials the aggregate lets pile up before folding them together.
///
/// The fold has to be bounded or the "streaming" aggregate quietly re-materializes its input as
/// a heap of per-morsel partials. Combining on *every* morsel would instead re-hash the whole
/// running state once per morsel. Batching the fold keeps state at `O(groups)` while paying the
/// combine only every `N` morsels.
const AGG_FOLD_EVERY: usize = 32;

/// Morsels each worker gets per parallel fold round.
///
/// The round is the unit of buffering, so this is what the "streaming" aggregate holds of its
/// input at once: `workers x` this many morsels. Two keeps every worker fed across a round
/// boundary (one to fold, one queued) while leaving the buffer a small multiple of a morsel —
/// at the 16,384-row default and 96 workers, a few hundred MB at the very widest, and
/// proportional to the machine rather than to the relation.
const PAR_FOLD_MORSELS_PER_WORKER: usize = 2;

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
pub(super) fn fold_partial_parallel(
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

/// Fold a stream into its global top `k` rows, holding only a bounded round of morsels and the
/// running `k` survivors — never the input.
///
/// [`ops::parallel_top_n`] is already the mergeable top-N: it reduces each batch it is handed to
/// that batch's local top-`k`, merges the narrow candidates, and breaks ties by the candidate's
/// **original position** (source batch, then row within it). Those two properties are exactly
/// what make folding it sound. Passing the carried survivors as batch 0 and this round's morsels
/// after them presents the rows in their original relative order, so a tie between a carried row
/// and a new one resolves toward the carried row — which is the earlier row, which is what a full
/// sort-then-slice keeps. Fold or no fold, the answer is the same `k` rows in the same order.
///
/// **The round is sized so the fold is not the cost.** Re-reading the carried `k` rows once per
/// round is the only work the fold adds, so a round is required to carry at least `TOPN_FOLD_RATIO`
/// times `k` rows before it is folded: the overhead is then bounded by that ratio's reciprocal
/// regardless of `k`, instead of growing without limit as `k` approaches the round size. Peak
/// memory becomes `O(k)` — proportional to the *result*, which is the bound the operator's own
/// semantics justify — rather than `O(input)`.
///
/// Returns the survivors (`None` for an input with no batches at all), the rows consumed, and the
/// peak bytes actually held.
pub(super) fn stream_top_n(
    input: Morsels<'_>,
    keys: &[bc_ir::SortKey],
    k: usize,
    ctx: Ctx<'_>,
) -> Result<(Option<RecordBatch>, u64, u64), InterpError> {
    let round_rows = round_rows_for(k, ctx.workers);
    let mut carried: Option<RecordBatch> = None;
    let mut buf: Vec<RecordBatch> = Vec::new();
    let mut buffered_rows = 0usize;
    let mut rows_in = 0u64;
    let mut peak = 0u64;
    let mut saw_batch = false;

    for morsel in input {
        let morsel = morsel?;
        saw_batch = true;
        if morsel.num_rows() == 0 {
            // Keep the schema of an all-empty input, exactly as draining did: the survivors of
            // nothing are still a typed, zero-row relation.
            if carried.is_none() && buf.is_empty() {
                buf.push(morsel);
            }
            continue;
        }
        rows_in += morsel.num_rows() as u64;
        buffered_rows += morsel.num_rows();
        buf.push(morsel);
        if buffered_rows >= round_rows {
            peak = peak.max(held_bytes(&carried, &buf));
            ctx.check_budget(peak, "the streaming top-N does not spill")?;
            carried = Some(fold_top_n(carried.take(), &mut buf, keys, k)?);
            buffered_rows = 0;
        }
    }
    if !saw_batch {
        return Ok((None, 0, 0));
    }
    if !buf.is_empty() {
        peak = peak.max(held_bytes(&carried, &buf));
        ctx.check_budget(peak, "the streaming top-N does not spill")?;
        carried = Some(fold_top_n(carried.take(), &mut buf, keys, k)?);
    }
    Ok((carried, rows_in, peak))
}

/// How many rows a top-N fold round buffers before folding.
///
/// The floor keeps every worker fed (a round is what [`ops::parallel_top_n`] fans out over); the
/// `k` term keeps the carried survivors a small fraction of each round, so the fold's overhead
/// does not grow with `k`.
fn round_rows_for(k: usize, workers: usize) -> usize {
    let per_worker = workers.max(1).saturating_mul(PAR_FOLD_MORSELS_PER_WORKER);
    let floor = per_worker.saturating_mul(bc_arrow::DEFAULT_MORSEL_ROWS);
    floor.max(k.saturating_mul(TOPN_FOLD_RATIO))
}

/// How many rows a top-N round must hold per row of carried state before it folds.
///
/// Sixteen bounds the fold's added work at ~6% of the round it is folding, which is inside the
/// noise of the scan that feeds it, while keeping peak memory at a small multiple of the result.
const TOPN_FOLD_RATIO: usize = 16;

/// Fold a stream into a fixed-count sample of `n` rows, holding the candidates and a bounded
/// round rather than the relation they are drawn from.
///
/// A fixed-`n` sample keeps the `n` smallest-hash rows of the **whole** relation — which is why
/// it is not a per-morsel operator like the fractional one, and why it was a breaker. But
/// "smallest `n` by a key" is a top-N by another name, and [`ops::sample_n_batches`] already
/// computes it as one: a bounded max-heap over `(hash, row, batch, row_index)`, with ties broken
/// by the candidate's position. Presenting the carried candidates ahead of each round preserves
/// those positions' relative order, so the fold selects the same `n` rows the single pass does
/// and emits them in the same order.
///
/// The sample is *the answer*, so peak memory becomes `O(n)`. `ds.sample(n=1000)` over a
/// hundred-million-row table is the shape this exists for, and it is exactly the one that held
/// all hundred million.
pub(super) fn stream_sample_n(
    input: Morsels<'_>,
    n: usize,
    seed: u64,
    ctx: Ctx<'_>,
) -> Result<(Vec<RecordBatch>, u64, u64), InterpError> {
    let round_rows = round_rows_for(n, ctx.workers);
    let mut carried: Vec<RecordBatch> = Vec::new();
    let mut buf: Vec<RecordBatch> = Vec::new();
    let mut buffered_rows = 0usize;
    let mut rows_in = 0u64;
    let mut peak = 0u64;

    for morsel in input {
        let morsel = morsel?;
        rows_in += morsel.num_rows() as u64;
        buffered_rows += morsel.num_rows();
        buf.push(morsel);
        if buffered_rows < round_rows {
            continue;
        }
        peak = peak.max(crate::batch_bytes(&carried) + crate::batch_bytes(&buf));
        ctx.check_budget(peak, "the streaming sample does not spill")?;
        carried.append(&mut buf);
        carried = ops::sample_n_batches(&carried, n, seed)?;
        buffered_rows = 0;
    }
    peak = peak.max(crate::batch_bytes(&carried) + crate::batch_bytes(&buf));
    ctx.check_budget(peak, "the streaming sample does not spill")?;
    carried.append(&mut buf);
    Ok((ops::sample_n_batches(&carried, n, seed)?, rows_in, peak))
}

/// Bytes currently held by a top-N fold: the carried survivors plus the buffered round.
fn held_bytes(carried: &Option<RecordBatch>, buf: &[RecordBatch]) -> u64 {
    carried
        .iter()
        .chain(buf.iter())
        .map(|b| b.get_array_memory_size() as u64)
        .sum()
}

/// One top-N round: the carried survivors (first, so ties resolve toward the earlier row) ahead of
/// this round's morsels, reduced back to `k`.
fn fold_top_n(
    carried: Option<RecordBatch>,
    buf: &mut Vec<RecordBatch>,
    keys: &[bc_ir::SortKey],
    k: usize,
) -> Result<RecordBatch, InterpError> {
    let mut parts: Vec<RecordBatch> = Vec::with_capacity(buf.len() + 1);
    parts.extend(carried);
    parts.append(buf);
    let out = ops::parallel_top_n(&parts, keys, k);
    buf.clear();
    out
}

/// Fold a stream into its distinct rows, holding a bounded round of morsels and the running
/// survivors rather than the whole input.
///
/// [`ops::parallel_distinct`] is `agg::combine` over one partial per batch with no aggregate
/// functions — the mergeable algebra (invariant #7) with the aggregate half empty. `combine` is
/// associative, and dedup is idempotent, so reducing the survivors against each round in turn
/// yields the same rows as one reduction over the concatenated input. Group order is unspecified
/// for a hash aggregate (`bc_runtime::agg::combine_with` says so explicitly), which is the same
/// licence the streaming aggregate's partitioned merge already runs under.
///
/// **It gives up when the data says it will not pay.** Folding buys memory only where the dedup
/// actually reduces; on a key that is nearly all-distinct the survivors approach the input, so the
/// fold would re-hash a growing state once per round and buy nothing. Measured rather than
/// guessed, exactly as `agg_par` picks an aggregate strategy: once the survivors exceed
/// [`DISTINCT_FOLD_MIN_REDUCTION`] of the rows seen, the rounds stop folding and the remaining
/// morsels accumulate for one final reduction — which is precisely what this operator did before,
/// so the shape that cannot benefit is also not charged.
pub(super) fn stream_distinct(
    input: Morsels<'_>,
    ctx: Ctx<'_>,
) -> Result<(Vec<RecordBatch>, u64, u64), InterpError> {
    let round_rows = round_rows_for(0, ctx.workers);
    let mut carried: Vec<RecordBatch> = Vec::new();
    let mut buf: Vec<RecordBatch> = Vec::new();
    let mut buffered_rows = 0usize;
    let mut rows_in = 0u64;
    let mut peak = 0u64;
    let mut folding = true;

    for morsel in input {
        let morsel = morsel?;
        rows_in += morsel.num_rows() as u64;
        buffered_rows += morsel.num_rows();
        buf.push(morsel);
        if !folding || buffered_rows < round_rows {
            // Not folding: this is the drained path, so the whole held input must still be
            // checked against the envelope morsel by morsel, or the bail happens after the
            // allocation it exists to avoid.
            if !folding {
                peak = peak.max(crate::batch_bytes(&carried) + crate::batch_bytes(&buf));
                ctx.check_budget(peak, "the streaming distinct does not spill")?;
            }
            continue;
        }
        peak = peak.max(crate::batch_bytes(&carried) + crate::batch_bytes(&buf));
        ctx.check_budget(peak, "the streaming distinct does not spill")?;
        carried.append(&mut buf);
        carried = ops::parallel_distinct(&carried)?;
        buffered_rows = 0;
        folding =
            crate::count_rows(&carried).saturating_mul(DISTINCT_FOLD_MIN_REDUCTION) <= rows_in;
    }
    peak = peak.max(crate::batch_bytes(&carried) + crate::batch_bytes(&buf));
    ctx.check_budget(peak, "the streaming distinct does not spill")?;
    carried.append(&mut buf);
    if carried.is_empty() {
        return Ok((Vec::new(), rows_in, peak));
    }
    let out = ops::parallel_distinct(&carried)?;
    Ok((out, rows_in, peak))
}

/// How many input rows a streaming dedup must reduce to one survivor for the fold to keep paying.
///
/// Below this the survivors are most of the input, so re-reducing them each round is work with no
/// memory to show for it. Two is the break-even in spirit as well as arithmetic: it is the point
/// at which the state the fold carries stops being smaller than the input it replaced.
const DISTINCT_FOLD_MIN_REDUCTION: u64 = 2;

/// Fold an aggregate's input **incrementally** into one `Partial` — `partial` each morsel,
/// `combine` them — without ever holding the input.
///
/// `None` means the stream held no rows at all. That is not the same as "the aggregate is
/// empty": a global `COUNT` over nothing is still one row containing `0`. Only the caller knows
/// whether it is looking at a whole relation (defer to the oracle for that row) or at one shard
/// of a parallel run (contribute nothing and let the other shards speak).
///
/// **Why a `Partial` and not a finished aggregate.** A finalized aggregate cannot be merged: two
/// shards' `mean`s do not average to the relation's `mean`, and two `count`s do not compose with
/// two `sum`s once the shape is lost. `partial`/`combine`/`finalize` is the mergeable algebra
/// (invariant #7) — the *same* fold the distributed path runs across nodes — and it is
/// associative, so folding morsel by morsel, or shard by shard, finalizes to exactly what one
/// `partial` over the concatenated input would.
pub(crate) fn fold_partial(
    input: Morsels<'_>,
    group_keys: &[bc_ir::ProjectionItem],
    aggregates: &[bc_ir::AggregateItem],
    jit: &std::sync::OnceLock<ops::AggJit>,
) -> Result<(Option<agg::Partial>, u64), InterpError> {
    let funcs = ops::agg_funcs(aggregates);
    let mut partials: Vec<agg::Partial> = Vec::new();
    let mut folded: Option<agg::Partial> = None;
    let mut rows_in: u64 = 0;
    // Compile the computed group-key and aggregate-input expressions once, from the first
    // morsel that carries rows — the JIT fast path the materializing executor already uses,
    // so arithmetic in aggregate inputs (`SUM(price * (1 - discount) * (1 + tax))`, the whole
    // TPC-H q1 shape) is compiled once and reused across morsels instead of interpreted per
    // row. `eval_jit` is bit-identical to the interpreter on its supported subset and falls
    // back to it otherwise, so this changes throughput only — the streaming-oracle tests pin
    // it against the same interpreter.
    //
    // The `OnceLock` is **shared across the shards** by the caller: `compile_agg` is a pure
    // function of the plan and the schema, and every shard's post-child morsel has the same
    // schema, so one compile serves all of them. Compiling per shard instead paid Cranelift's
    // per-expression cost once per core (~90× on a big box), which measured as a real fraction
    // of a low-cardinality aggregate — this hoists it to exactly one compile per query.

    for morsel in input {
        let morsel = morsel?;
        if morsel.num_rows() == 0 {
            continue;
        }
        rows_in += morsel.num_rows() as u64;
        let jit = jit.get_or_init(|| ops::compile_agg(group_keys, aggregates, &morsel));
        partials.push(ops::eval_partial_jit(&morsel, group_keys, aggregates, jit)?);
        // Bounded: without this the "streaming" aggregate quietly re-materializes its input as a
        // heap of per-morsel partials. Combining on *every* morsel would instead re-hash the
        // whole running state once per morsel; batching the fold keeps state at O(groups).
        if partials.len() >= AGG_FOLD_EVERY {
            if let Some(prev) = folded.take() {
                partials.push(prev);
            }
            folded = Some(agg::combine(&partials, &funcs)?);
            partials.clear();
        }
    }

    if let Some(prev) = folded.take() {
        partials.push(prev);
    }
    if partials.is_empty() {
        return Ok((None, rows_in));
    }
    Ok((Some(agg::combine(&partials, &funcs)?), rows_in))
}

/// `finalize` one (already combined) partial into the aggregate's output batch.
pub(crate) fn finalize_partial(
    merged: &agg::Partial,
    group_keys: &[bc_ir::ProjectionItem],
    aggregates: &[bc_ir::AggregateItem],
) -> Result<Vec<RecordBatch>, InterpError> {
    let funcs = ops::agg_funcs(aggregates);
    let agg_cols = agg::finalize(&funcs, merged)?;
    Ok(vec![ops::build_agg_batch(
        group_keys,
        aggregates,
        &merged.group_columns,
        &agg_cols,
    )?])
}

/// Combine shard-level partials into one, then finalize — the parallel aggregate's tail.
pub(crate) fn combine_and_finalize(
    partials: &[agg::Partial],
    group_keys: &[bc_ir::ProjectionItem],
    aggregates: &[bc_ir::AggregateItem],
) -> Result<Vec<RecordBatch>, InterpError> {
    let funcs = ops::agg_funcs(aggregates);
    // Keep the merge's hash-radix partitions as separate morsels rather than concatenating
    // them into one. They are key-disjoint, so the rows and their order are exactly what one
    // combined `Partial` would finalize to — but the concat is a second full copy of the
    // grouped relation (on a high-cardinality string key, the largest term in the merge), and
    // the next operator gets a batch per partition to fan back out over instead of one.
    let merged = agg::combine_partitioned(partials, &funcs, 0)?;
    let mut out = Vec::with_capacity(merged.len());
    for part in &merged {
        out.extend(finalize_partial(part, group_keys, aggregates)?);
    }
    Ok(out)
}
