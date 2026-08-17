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

/// Partial rows a unit may keep, per input row, before grouping at that unit size stops
/// paying — the streaming twin of `agg_par::REDUCTION_CEILING`, and the same number.
///
/// A morsel is 16,384 rows. A key with 10,000 groups therefore fills a morsel's hash table
/// almost completely: the partial keeps ~0.5 rows per input row, its group columns are `take`n
/// out at that width, and `combine` inherits half the relation — all to discover a reduction
/// the *shard* achieves 12x better. That is `agg_par::chunked_partials`' argument, driven by a
/// measurement the stream can take as it goes rather than from a sample of the whole relation.
///
/// The same number then answers the other direction, which is what makes it safe: a unit whose
/// partial keeps most of its rows has not reduced, whether that unit is a morsel or a chunk.
/// See [`Unit`].
const MORSEL_REDUCTION_CEILING: f64 = 0.20;

/// Rows the chunked fold buffers before it groups them.
///
/// The chunk is what bounds this path's extra memory: it holds the buffered morsels and, for
/// the moment `concat` runs, a copy of them — so a shard's transient footprint is ~2x this
/// many rows, not the relation. 128 k rows is eight morsels, enough for a 10,000-group key to
/// reduce ~12x over a single morsel, and small enough that a wide row (blobs, embeddings)
/// stays inside a few MB per shard.
const CHUNK_FOLD_ROWS: usize = 128 * 1024;

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

    // How large a unit this fold groups at a time, adapted from what the units it has already
    // grouped actually reduced. See [`Unit`]: a morsel is the default, and a key whose groups
    // outnumber a morsel's rows moves to a [`CHUNK_FOLD_ROWS`]-row chunk — but only while the
    // chunk earns its concatenation back. Same `partial`s over the same rows in the same order
    // either way, so `combine`, `finalize`, the spill path and the distributed reduce are
    // untouched; only the size of the unit changes.
    let mut unit = Unit::Morsel;
    let mut asked = false;
    let mut chunk: Vec<RecordBatch> = Vec::new();
    let mut chunk_rows = 0usize;

    for morsel in input {
        let morsel = morsel?;
        if morsel.num_rows() == 0 {
            continue;
        }
        rows_in += morsel.num_rows() as u64;
        let jit = jit.get_or_init(|| ops::compile_agg(group_keys, aggregates, &morsel));
        if unit == Unit::Chunk {
            chunk_rows += morsel.num_rows();
            chunk.push(morsel);
            if chunk_rows >= CHUNK_FOLD_ROWS {
                let partial = group_chunk(&mut chunk, group_keys, aggregates, jit)?;
                if !reduces(&partial, chunk_rows) {
                    // The chunk did not reduce either, so the key is near-unique over the whole
                    // shard and no unit size reduces it. Chunking then buys nothing and costs a
                    // `concat` per chunk, so fall back to the morsel — which is what the input
                    // already arrives as. This is what keeps the trade one-sided: a wrong guess
                    // costs one chunk's concatenation, not the relation's.
                    unit = Unit::Morsel;
                }
                partials.push(partial);
                chunk_rows = 0;
            }
        } else {
            let partial = ops::eval_partial_jit(&morsel, group_keys, aggregates, jit)?;
            // Asked once, of the first morsel, and never again. The projection inverts the
            // coupon-collector curve by bisection, which is far too much arithmetic to repeat
            // per morsel — and repeating it answers nothing new, because the reduction a key
            // achieves is a property of the key rather than of where the stream has reached.
            // A shard whose later morsels disagree is covered from the other side: the chunk's
            // own measurement below can send the fold back to the morsel.
            if !asked {
                asked = true;
                if chunk_would_reduce(&partial, morsel.num_rows()) {
                    unit = Unit::Chunk;
                }
            }
            partials.push(partial);
        }
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
    if !chunk.is_empty() {
        let jit = jit
            .get()
            .expect("a buffered chunk implies a compiled aggregate");
        partials.push(group_chunk(&mut chunk, group_keys, aggregates, jit)?);
    }

    if let Some(prev) = folded.take() {
        partials.push(prev);
    }
    if partials.is_empty() {
        return Ok((None, rows_in));
    }
    Ok((Some(agg::combine(&partials, &funcs)?), rows_in))
}

/// How much input [`fold_partial`] groups at a time.
///
/// Widening is decided *before* it is paid for, by projecting the key's group count forward to
/// the chunk's size (see [`chunk_would_reduce`]) — so a key that cannot benefit never buffers
/// one. The measurement of each grouped chunk is then the backstop, and it can move back to the
/// morsel: the projection assumes a uniform key, and a clustered or skewed one can clear the
/// entry test and still not reduce. Being wrong then costs one chunk's concatenation, not the
/// relation's.
#[derive(PartialEq, Eq, Clone, Copy)]
enum Unit {
    /// One morsel at a time — the input's own unit, and no copy.
    Morsel,
    /// [`CHUNK_FOLD_ROWS`] rows at a time, concatenated first.
    Chunk,
}

/// Whether a partial over `rows` reduced enough to be worth taking at that unit size.
fn reduces(partial: &agg::Partial, rows: usize) -> bool {
    rows > 0 && (kept_rows(partial) as f64 / rows as f64) < MORSEL_REDUCTION_CEILING
}

/// Partial rows this unit's `partial` kept — the measurement both decisions read.
fn kept_rows(partial: &agg::Partial) -> usize {
    partial.group_columns.first().map_or(0, |c| c.len())
}

/// Whether grouping [`CHUNK_FOLD_ROWS`] rows of this key *would* reduce, judged from what one
/// morsel of it kept.
///
/// This is the entry test, and it has to be a projection rather than the morsel's own ratio.
/// "A morsel did not reduce" is true of two completely different keys: one with a few tens of
/// thousands of groups, where a wider unit reduces enormously, and a **near-unique** one, where
/// no unit reduces because the group count grows with the rows. Widening on the morsel's ratio
/// alone treats them the same, so a `GROUP BY <customer id>` buffered and concatenated a chunk
/// to rediscover that its key is unique — measured at ~5% on TPC-H q13 and q18, both of which
/// are exactly that shape.
///
/// `agg_par::estimated_groups` is the projection: it inverts the coupon-collector curve on the
/// morsel's `(rows, groups)` to recover the key's domain, then reads the curve forward to the
/// chunk's row count. Shared with the materializing executor rather than restated, so the two
/// paths cannot come to different conclusions about the same key.
fn chunk_would_reduce(partial: &agg::Partial, rows: usize) -> bool {
    if rows == 0 || reduces(partial, rows) {
        return false; // the morsel already reduces; a wider unit is not needed
    }
    let projected = crate::agg_par::estimated_groups(rows, kept_rows(partial), 1, CHUNK_FOLD_ROWS);
    (projected as f64 / CHUNK_FOLD_ROWS as f64) < MORSEL_REDUCTION_CEILING
}

/// Group a buffered run of morsels as one unit, draining the buffer.
///
/// A single morsel needs no concatenation — copying it to group it would be pure loss — which
/// is also what makes the chunked path safe to enter on a short input.
fn group_chunk(
    chunk: &mut Vec<RecordBatch>,
    group_keys: &[bc_ir::ProjectionItem],
    aggregates: &[bc_ir::AggregateItem],
    jit: &ops::AggJit,
) -> Result<agg::Partial, InterpError> {
    let partial = match chunk.as_slice() {
        [only] => ops::eval_partial_jit(only, group_keys, aggregates, jit),
        many => {
            let joined = arrow::compute::concat_batches(&many[0].schema(), many)?;
            ops::eval_partial_jit(&joined, group_keys, aggregates, jit)
        }
    };
    chunk.clear();
    partial
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

#[cfg(test)]
mod fold_unit_tests {
    use super::*;
    use arrow::array::{ArrayRef, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use bc_ir::{AggFunc, AggregateItem, ProjectionItem, RelOp};
    use std::sync::Arc;

    /// `n` rows whose key cycles through `groups` distinct values — a key whose cardinality
    /// is exactly what the fold's unit choice is supposed to react to.
    fn keyed(n: usize, groups: usize) -> Vec<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
        ]));
        (0..n)
            .step_by(bc_arrow::DEFAULT_MORSEL_ROWS)
            .map(|start| {
                let end = (start + bc_arrow::DEFAULT_MORSEL_ROWS).min(n);
                let k: ArrayRef = Arc::new(Int64Array::from(
                    (start..end)
                        .map(|i| (i % groups) as i64)
                        .collect::<Vec<_>>(),
                ));
                let v: ArrayRef = Arc::new(Int64Array::from(
                    (start..end).map(|i| i as i64).collect::<Vec<_>>(),
                ));
                RecordBatch::try_new(schema.clone(), vec![k, v]).unwrap()
            })
            .collect()
    }

    fn group_by_k_sum_v() -> RelOp {
        RelOp::Aggregate {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            group_keys: vec![ProjectionItem {
                alias: "k".into(),
                expr: bc_expr::Expr::Col { name: "k".into() },
            }],
            aggregates: vec![AggregateItem {
                alias: "s".into(),
                func: AggFunc::Sum,
                input: Some(bc_expr::Expr::Col { name: "v".into() }),
                input2: None,
                param: None,
            }],
        }
    }

    /// Whatever unit the fold picks, the answer is the sequential oracle's.
    ///
    /// The three cardinalities are the three regimes the unit choice distinguishes, and the
    /// middle one is the only one the widening was written for: a key with far fewer groups
    /// than a morsel has rows never leaves the morsel; one with far more never reduces at any
    /// size and must come back to it; one in between is the case the chunk exists to serve.
    /// Routing is a performance decision, so all three must agree with `execute` exactly.
    #[test]
    fn every_unit_choice_agrees_with_the_sequential_oracle() {
        let rows = 400_000;
        for groups in [8usize, 50_000, rows] {
            let sources = vec![keyed(rows, groups)];
            let plan = group_by_k_sum_v();
            let streamed = crate::execute_streaming(&plan, &sources, 0).unwrap();
            let oracle = crate::execute(&plan, &sources).unwrap();
            let sums = |bs: &[RecordBatch]| -> Vec<(i64, i64)> {
                let mut out: Vec<(i64, i64)> = bs
                    .iter()
                    .flat_map(|b| {
                        let k = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
                        let s = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
                        (0..b.num_rows())
                            .map(|i| (k.value(i), s.value(i)))
                            .collect::<Vec<_>>()
                    })
                    .collect();
                out.sort_unstable();
                out
            };
            assert_eq!(
                sums(&streamed),
                sums(&oracle),
                "groups={groups}: the fold's unit choice changed the answer"
            );
            assert_eq!(sums(&streamed).len(), groups.min(rows));
        }
    }

    /// A near-unique key does not stay on the chunked path.
    ///
    /// This is the regression the two-directional [`Unit`] exists for: such a key fails the
    /// morsel test (it reduces at no unit size), and a widen-only rule would then pay a
    /// `concat` per chunk forever for a reduction that was never available. Asserted on the
    /// decision itself, because the cost it avoids is invisible in the result.
    #[test]
    fn a_key_that_reduces_at_no_size_returns_to_the_morsel() {
        let rows = bc_arrow::DEFAULT_MORSEL_ROWS;
        let unique = agg::Partial {
            group_columns: vec![
                Arc::new(Int64Array::from((0..rows as i64).collect::<Vec<_>>())) as ArrayRef,
            ],
            states: Vec::new(),
        };
        assert!(
            !reduces(&unique, rows),
            "a one-row-per-group partial reduces nothing"
        );
        let reduced = agg::Partial {
            group_columns: vec![Arc::new(Int64Array::from(vec![0i64; 8])) as ArrayRef],
            states: Vec::new(),
        };
        assert!(
            reduces(&reduced, rows),
            "eight groups over a morsel reduces"
        );
        assert!(
            !chunk_would_reduce(&unique, rows),
            "a near-unique key must not buffer a chunk it cannot benefit from"
        );
        assert!(
            !chunk_would_reduce(&reduced, rows),
            "a key the morsel already reduces needs no wider unit"
        );

        // The regime between them is the one the wider unit exists for: ~10,000 groups fill a
        // morsel almost completely, and a chunk of the same key reduces about twelve-fold. The
        // entry test has to tell it apart from the near-unique key *before* either pays a
        // concatenation, which is why it projects rather than reading the morsel's own ratio.
        let mid = agg::Partial {
            group_columns: vec![
                Arc::new(Int64Array::from((0..9_000i64).collect::<Vec<_>>())) as ArrayRef,
            ],
            states: Vec::new(),
        };
        assert!(!reduces(&mid, rows), "10k groups fill a morsel");
        assert!(
            chunk_would_reduce(&mid, rows),
            "but a chunk of them reduces"
        );
    }
}
