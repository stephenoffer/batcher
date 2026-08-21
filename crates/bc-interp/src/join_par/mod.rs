//! Parallel join strategies shared by the multi-core executor (`par`).
//!
//! These are the join shapes that need more than a plain per-bucket hash join:
//!
//! * **grace (spilling) hash join** — when the build side exceeds the memory
//!   envelope, partition both sides to disk and join one bucket at a time so only
//!   one build table is ever resident.
//! * **broadcast / chunked join** — when the build side is small enough to
//!   replicate, the large probe side joins without a key shuffle, parallelized over
//!   row-range chunks; also the per-bucket skew mitigation (`broadcast_join` of a
//!   hot bucket).
//! * **skew detection** — deciding when a co-partitioned bucket is hot enough to be
//!   spread across worker chunks.
//! * **build-side correction** — deciding, from the two inputs' *measured* sizes, that
//!   the planner picked the wrong side to build on.
//!
//! Extracted from `par` along the join-strategy seam to keep that file within the
//! size budget; the semantics are unchanged (the parallel join is still
//! bit-identical to the sequential oracle as a multiset).

use arrow::array::RecordBatch;
use bc_runtime::agg::spill::{DiskSpillStore, SpillStore};
use bc_runtime::{join, shuffle};
use rayon::prelude::*;
use std::sync::Arc;

use crate::error::InterpError;
use crate::ops;
use crate::par::SpillOptions;

pub(crate) mod probe_stream;
use crate::spill_split::{
    drain_repartition, grace_bucket_count, split_salt, MAX_GRACE_SPLIT_DEPTH,
};
use probe_stream::{chunk_by_bytes, ProbeStream};

/// How many times larger the planner's build side must be than its probe side before the
/// executor overrides the choice.
///
/// A swap is free — both relations are already in hand, so it rebinds two slices and
/// costs nothing to perform — but it is not free to *get wrong*, because the two sides
/// are not interchangeable downstream: the emitted row order follows the probe. So the
/// bar is not "is the left smaller" (which would churn the orientation on every near-tie
/// and on every estimate that was right to within a rounding error) but "is the planner's
/// build side big enough that building it is the thing that hurts".
///
/// 2× is where the asymmetry starts to bite rather than where it becomes dramatic. Below
/// it the two orientations cost within noise of each other and the planner's choice
/// carries information the executor does not have (it saw the *whole* plan, including
/// which side a downstream operator would rather have in order). Above it the build side
/// is paying for a hash table, a chain array and a null mask over rows that would have
/// been a streaming probe.
const SWAP_MIN_RATIO: usize = 2;

/// Build sides below this are never worth correcting.
///
/// One morsel is the granularity at which every other decision in the executor stops
/// caring, and a build table over fewer rows than that is a rounding error against the
/// probe regardless of the ratio: a 3-row probe against a 30-row build is a 10× ratio and
/// a 0 ms difference. Without this floor the rule would fire constantly on the small
/// dimension-table joins where it can only change the output order.
const SWAP_MIN_BUILD_ROWS: usize = bc_arrow::DEFAULT_MORSEL_ROWS;

/// Whether building on the *left* input would be materially cheaper than the planner's
/// choice of the right, judged from the two inputs' true sizes.
///
/// **This is the one join decision the planner cannot make correctly and the executor
/// can.** Kyber picks the build side in `kyber/rules/selection.py` from
/// `estimated rows × estimated row width`, and by the time this runs both relations are
/// materialized, so their sizes are *facts*. Where the estimate was right this changes
/// nothing. Where it was wrong — a `LIKE` predicate, a correlated filter, a join feeding
/// a join, or simply a source with no statistics on its first run — the executor is
/// otherwise obliged to build a hash table over the larger relation, and that single
/// wrong choice decides three things at once:
///
/// * **whether the join spills.** Admission compares the *build* side against the memory
///   envelope. Building the 50 M-row side of a 50 M ⋈ 200 k join spills to disk; building
///   the 200 k-row side does not. That is not a percentage, it is a grace hash join
///   against no disk I/O at all.
/// * **whether the streaming probe is reachable.** `streaming_supported` is gated on
///   `build_rows`, so an oversized build falls through to the shuffle, which copies the
///   probe — the query's largest relation — for no reason.
/// * **the hash table's cache residency**, which is what the radix path exists to
///   protect.
///
/// Restricted to `Inner`, which is the only flavor whose swap is a pure re-labeling.
/// `Semi`/`Anti` emit *left* rows and are not symmetric under a swap at all. `Left`/`Right`
/// are semantically swappable (and [`broadcast_join`] already performs exactly that
/// rewrite in the other direction), but the engine's probe-driven fast paths decline
/// `Right` — `streaming_supported` rejects it outright — so turning a `Left` join into a
/// `Right` one to shrink the build would trade a hash table for a slower path and could
/// lose. `Full` is symmetric but drives from both sides, so there is no build to shrink.
pub(crate) fn build_side_swap_pays(
    join_type: bc_ir::JoinType,
    probe_rows: usize,
    build_rows: usize,
) -> bool {
    matches!(join_type, bc_ir::JoinType::Inner)
        && build_rows >= SWAP_MIN_BUILD_ROWS
        && build_rows >= probe_rows.saturating_mul(SWAP_MIN_RATIO)
}

/// The join's output projection with each column's side re-labeled for swapped inputs.
///
/// The output columns name a side and a column *on* that side, so exchanging the two
/// relations is expressible entirely here: nothing about the column, its name, or its
/// alias changes, only which input it is read from. This is what makes a build-side swap
/// a re-labeling rather than a rewrite.
pub(crate) fn flip_output(output: &[bc_ir::JoinOutputCol]) -> Vec<bc_ir::JoinOutputCol> {
    use bc_ir::JoinSide;
    output
        .iter()
        .map(|o| bc_ir::JoinOutputCol {
            side: match o.side {
                JoinSide::Left => JoinSide::Right,
                JoinSide::Right => JoinSide::Left,
            },
            name: o.name.clone(),
            alias: o.alias.clone(),
        })
        .collect()
}

/// Grace hash join, streamed: the build (right) side exceeds the budget, so
/// partition both sides by key to disk **one input batch at a time** and join one
/// bucket at a time — only one input batch (plus its `p` shards) and one build
/// bucket are ever resident.
///
/// Unlike a partition-an-already-materialized-batch grace join, this never
/// concatenates the full build side into one `RecordBatch` first, so a build far
/// larger than memory spills instead of OOMing at the materialize step. Bucket count
/// is sized from the build batches' total bytes (no materialization) so each
/// bucket's build side ≈ one budget. Equal keys co-partition (fixed-seed
/// partitioner), so the union of per-bucket joins is the full join for every join
/// type — the result is the same multiset the in-memory path produces.
///
/// A bucket that exceeds the envelope on *either* side is recursively re-split before it is
/// joined (see [`join_bucket`]), so key skew cannot reintroduce the OOM the spill was there
/// to prevent.
///
/// Empty input is handled exactly as the in-memory path: a side with no batches at all
/// reports `EmptyJoinInput` (empty joins are shortcut upstream and never reach this spill
/// path). A *bucket* with no rows on one side is ordinary and joins as an empty relation of
/// that side's schema.
pub(crate) fn spilling_hash_join_streaming(
    left_batches: &[RecordBatch],
    right_batches: &[RecordBatch],
    left_keys: &[String],
    right_keys: &[String],
    join_type: bc_ir::JoinType,
    output: &[bc_ir::JoinOutputCol],
    sp: &SpillOptions,
) -> Result<(Vec<RecordBatch>, u64), InterpError> {
    // Enough partitions that each bucket lands near one budget on **both** sides — sized from
    // the batches' total size without materializing either, then capped (`MAX_GRACE_FANOUT`).
    //
    // Only the build side has to be resident, so only the build side bounds *memory*; the
    // probe side is streamed past it in envelope-sized chunks. But the build table is rebuilt
    // once per chunk, so sizing the fan-out from the build side alone is what decides how many
    // times. A star join — a 200 MB dimension against a 200 GB fact — asked for two buckets,
    // which left each probe bucket a thousand budgets long and rebuilt the same table a
    // thousand times. Taking the larger side puts each probe bucket near one chunk, so the
    // table is built once per bucket, which is what it was before the probe side was streamed.
    //
    // Past the fan-out cap the buckets are still too big for that, and the chunking is what
    // keeps them bounded rather than resident. Correctness does not depend on either: equal
    // keys co-partition for any `p`.
    let budget = sp.memory_budget_bytes.max(1);
    let p = grace_fanout(side_bytes(left_batches), side_bytes(right_batches), budget);

    let mut lstore = DiskSpillStore::with_codec(sp.dir.join("join-left"), p, sp.codec)?;
    let mut rstore = DiskSpillStore::with_codec(sp.dir.join("join-right"), p, sp.codec)?;
    // Stream each input batch through the key-partitioner into its `p` shards; only
    // one input batch and its shards are resident at a time, so neither side is ever
    // fully materialized in memory.
    partition_batches_to_store(left_batches, left_keys, p, 0, &mut lstore)?;
    partition_batches_to_store(right_batches, right_keys, p, 0, &mut rstore)?;

    // A side with no batches at all is shortcut upstream and never reaches this path, so
    // both schemas are available (and both are needed: a bucket may receive rows on one
    // side only, and an outer join still has to emit for it).
    let ctx = BucketJoin {
        left_keys,
        right_keys,
        join_type,
        output,
        budget,
        codec: sp.codec,
        dir: &sp.dir,
        lschema: left_batches
            .first()
            .ok_or(InterpError::EmptyJoinInput)?
            .schema(),
        rschema: right_batches
            .first()
            .ok_or(InterpError::EmptyJoinInput)?
            .schema(),
    };
    let mut out = Vec::with_capacity(p);
    for i in 0..p {
        join_bucket(&mut lstore, &mut rstore, i, &ctx, 0, &mut out)?;
    }
    // A join whose result is empty still has to say what its columns *are*. The per-bucket
    // path always pushed one batch per bucket, so it carried the schema even when every
    // bucket was empty; the streamed probe drops empty batches, so the empty relation is
    // stated once, here, by joining the two empty sides through the ordinary assembler.
    if out.is_empty() {
        out.push(ops::join_batches(
            &RecordBatch::new_empty(ctx.lschema.clone()),
            &RecordBatch::new_empty(ctx.rschema.clone()),
            left_keys,
            right_keys,
            join_type,
            output,
            bc_ir::JoinStrategy::Hash,
        )?);
    }
    // Both sides were streamed to disk; the spill volume is their combined written bytes.
    let spill_bytes = lstore.spilled_bytes() + rstore.spilled_bytes();
    Ok((out, spill_bytes))
}

/// A relation's in-memory footprint, without concatenating it.
fn side_bytes(batches: &[RecordBatch]) -> usize {
    batches.iter().map(|b| b.get_array_memory_size()).sum()
}

/// How many ways a grace join fans out, from the **larger** of its two sides.
///
/// Separated and named because the choice of side is the whole content of the decision, and
/// it is not the obvious one. See the call site for why the build side alone is wrong.
fn grace_fanout(left_bytes: usize, right_bytes: usize, budget: usize) -> usize {
    grace_bucket_count(left_bytes.max(right_bytes), budget)
}

/// The parts of a grace join that do not change as buckets are re-split.
struct BucketJoin<'a> {
    left_keys: &'a [String],
    right_keys: &'a [String],
    join_type: bc_ir::JoinType,
    output: &'a [bc_ir::JoinOutputCol],
    budget: usize,
    codec: bc_runtime::agg::spill::SpillCodec,
    dir: &'a std::path::Path,
    /// Each side's schema, so a bucket that received no rows on that side still joins as an
    /// empty relation of the right shape. Partitioning writes only non-empty shards — with
    /// a 256-way fan-out over thousands of morsels, writing the empty ones costs millions of
    /// IPC messages carrying nothing — so an untouched partition legitimately has no file
    /// at all, and `materialize`, which derives its schema from the first batch, has nothing
    /// to derive it from.
    lschema: arrow::datatypes::SchemaRef,
    rschema: arrow::datatypes::SchemaRef,
}

/// Concatenate a bucket's batches, or an empty relation of `schema` when it has none.
fn materialize_or_empty(
    batches: &[RecordBatch],
    schema: &arrow::datatypes::SchemaRef,
) -> Result<RecordBatch, InterpError> {
    if batches.is_empty() {
        return Ok(RecordBatch::new_empty(schema.clone()));
    }
    ops::materialize(batches)
}

/// Join co-partitioned bucket `i` of the two stores, re-splitting it first if the **build**
/// side does not fit in the envelope.
///
/// The bucket count is sized from the build side's *average* bytes per bucket, so it is only
/// an average-case fit. Under key skew one bucket holds far more than its share, and the
/// build side has to be resident to be a hash table — so a skewed bucket would OOM at exactly
/// the point spilling was supposed to have prevented it. This is the failure mode that makes
/// skewed joins the standard reason a Spark job dies, and the grace *aggregate* already
/// guards against it by recursively re-partitioning an over-large partition.
///
/// The build side is asked *before* being read, which is the whole point: the decision to
/// split has to happen without first pulling the partition that provably does not fit.
///
/// A re-split partitions **both** sides with a salt derived from the depth. The salt is a
/// function of the depth alone, never of the row, so equal keys still co-locate on both
/// sides and each sub-bucket remains an independent join whose union is the same relation.
///
/// ## Why the probe side is not part of that decision
///
/// It used to be: the split fired when *either* side was over budget, because both sides were
/// materialized whole before being joined. That was the wrong shape twice over.
///
/// It is unnecessary — a grace join needs only the *build* bucket resident. The probe side is
/// streamed through it batch by batch ([`stream_probe_bucket`]), so its size never bounds
/// anything, and re-splitting on account of it wrote both sides to disk again to reach a
/// state that was already fine.
///
/// And it does not work. Re-splitting is a re-hash, so it cannot separate rows that share a
/// key — and a probe bucket is over budget precisely when it holds a *hot key*. A fact table
/// whose `customer_id` is `-1` for 40% of its rows re-hashed to the same sub-bucket at every
/// one of the three permitted levels, paying three full re-spills of the whole bucket, and
/// then materialized it anyway. The dimension side it joins against holds one row for that
/// key: nothing about this join ever needed more than a morsel of memory.
fn join_bucket(
    lstore: &mut DiskSpillStore,
    rstore: &mut DiskSpillStore,
    i: usize,
    ctx: &BucketJoin<'_>,
    depth: u32,
    out: &mut Vec<RecordBatch>,
) -> Result<(), InterpError> {
    let rbytes = rstore.partition_bytes(i) as usize;
    if depth < MAX_GRACE_SPLIT_DEPTH && rbytes > ctx.budget {
        return split_and_join_bucket(lstore, rstore, i, ctx, depth, rbytes, out);
    }
    let build = materialize_or_empty(&rstore.read(i)?, &ctx.rschema)?;
    stream_probe_bucket(lstore, i, &build, ctx, out)
}

/// Join every probe batch of bucket `i` against the resident `build` bucket, one at a time.
///
/// The bucket's rows are read straight off disk and handed to [`ProbeStream`], so the peak is
/// the build bucket — which the caller has just proven fits the envelope — plus a single
/// probe morsel, regardless of how many rows the probe bucket holds or how they are spread
/// across keys. That is what makes the grace join's bound hold under the skew that
/// re-partitioning cannot help.
fn stream_probe_bucket(
    lstore: &mut DiskSpillStore,
    i: usize,
    build: &RecordBatch,
    ctx: &BucketJoin<'_>,
    out: &mut Vec<RecordBatch>,
) -> Result<(), InterpError> {
    let stream = ProbeStream {
        join_type: ctx.join_type,
        output: ctx.output,
        probe_schema: ctx.lschema.clone(),
    };
    let indices_of = |probe: &RecordBatch, build: &RecordBatch, jt: bc_ir::JoinType| {
        let lkeys = ops::columns_by_name(probe, ctx.left_keys)?;
        let rkeys = ops::columns_by_name(build, ctx.right_keys)?;
        Ok(join::hash_join_indices(
            &lkeys,
            &rkeys,
            ops::map_join_type(jt),
        )?)
    };

    // `open_reader` is the streaming counterpart of `read`; `None` is a bucket nothing was
    // ever routed to, which is ordinary at a 256-way fan-out (empty shards are not written).
    //
    // The shards are coalesced back to the envelope on the way in. Partitioning cut each input
    // morsel `p` ways, so they arrive as fragments of a few dozen rows, and the build table is
    // rebuilt once per chunk — one build per bucket when the probe side fits, which is what it
    // has always been, rather than one per fragment.
    let reader = lstore.open_reader(i)?;
    let probe_rows = match reader {
        Some(r) => stream.run(
            chunk_by_bytes(r.map(|b| b.map_err(InterpError::from)), ctx.budget),
            build,
            &indices_of,
            out,
        )?,
        None => stream.run(std::iter::empty(), build, &indices_of, out)?,
    };
    // `open_reader` streams, so it cannot make the short-read check `read`/`drain` make for
    // themselves — and a truncated IPC stream reads back as a shorter *valid* one, which here
    // would silently drop probe rows from the join. Asking the store to make its own
    // comparison is what keeps that an error rather than a wrong answer.
    lstore.verify_rows(i, probe_rows)?;
    Ok(())
}

/// Re-partition an over-large bucket on both sides and join the sub-buckets.
///
/// Each side is streamed out of its store one batch at a time and straight into a fresh
/// child store, so the bucket that did not fit is never held whole — the peak is one batch
/// plus one sub-bucket, which is what the caller's budget was meant to describe all along.
fn split_and_join_bucket(
    lstore: &mut DiskSpillStore,
    rstore: &mut DiskSpillStore,
    i: usize,
    ctx: &BucketJoin<'_>,
    depth: u32,
    bytes: usize,
    out: &mut Vec<RecordBatch>,
) -> Result<(), InterpError> {
    let sub_p = grace_bucket_count(bytes, ctx.budget);
    let salt = split_salt(depth + 1);
    let sub = ctx.dir.join(format!("join-split-{depth}"));
    let mut lsub = DiskSpillStore::with_codec(sub.join("left"), sub_p, ctx.codec)?;
    let mut rsub = DiskSpillStore::with_codec(sub.join("right"), sub_p, ctx.codec)?;
    let lidx = schema_key_indices(&ctx.lschema, ctx.left_keys)?;
    let ridx = schema_key_indices(&ctx.rschema, ctx.right_keys)?;
    drain_repartition(lstore, i, sub_p, salt, &columns_at(&lidx), &mut lsub)?;
    drain_repartition(rstore, i, sub_p, salt, &columns_at(&ridx), &mut rsub)?;
    for j in 0..sub_p {
        join_bucket(&mut lsub, &mut rsub, j, ctx, depth + 1, out)?;
    }
    Ok(())
}

/// Indices of the named key columns within a *schema*.
///
/// The batch-taking [`ops::key_indices`] cannot be used inside a store `drain`, whose
/// callback reports the store's own error type. Resolving against the schema once, before
/// the drain, keeps a missing key column an ordinary plan error instead of something that
/// has to be smuggled out of a closure — and it is one lookup per side rather than one per
/// batch.
fn schema_key_indices(
    schema: &arrow::datatypes::SchemaRef,
    names: &[String],
) -> Result<Vec<usize>, InterpError> {
    names
        .iter()
        .map(|n| {
            schema
                .index_of(n)
                .map_err(|_| InterpError::UnknownJoinColumn(n.clone()))
        })
        .collect()
}

/// A key extractor that takes the columns at `key_idx` — the join's key form, as the shape
/// [`drain_repartition`] takes.
fn columns_at(
    key_idx: &[usize],
) -> impl Fn(&RecordBatch) -> Result<Vec<arrow::array::ArrayRef>, InterpError> + '_ {
    move |b: &RecordBatch| Ok(key_idx.iter().map(|&k| b.column(k).clone()).collect())
}

/// Hash-partition one batch by the key columns at `key_idx` into `p` salted shards and
/// append each non-empty shard to `dest`.
///
/// Empty shards are skipped. With a 256-way fan-out over thousands of morsels, writing them
/// would be millions of IPC messages carrying no rows; the cost of skipping is that an
/// untouched partition has no file, which [`materialize_or_empty`] handles.
fn partition_batch_into(
    b: &RecordBatch,
    key_idx: &[usize],
    p: usize,
    salt: u64,
    dest: &mut DiskSpillStore,
) -> Result<(), bc_runtime::RuntimeError> {
    let key_cols: Vec<arrow::array::ArrayRef> =
        key_idx.iter().map(|&k| b.column(k).clone()).collect();
    let shards = shuffle::partition_by_key_arrays_salted(b, &key_cols, p, salt)?;
    for (i, shard) in shards.iter().enumerate() {
        if shard.num_rows() > 0 {
            dest.append(i, shard)?;
        }
    }
    Ok(())
}

/// Hash-partition each input batch by `keys` into `p` shards and append every non-empty
/// shard to its partition in `store` — one batch resident at a time (the bounded-memory
/// half of the streaming grace join).
fn partition_batches_to_store(
    batches: &[RecordBatch],
    keys: &[String],
    p: usize,
    salt: u64,
    store: &mut DiskSpillStore,
) -> Result<(), InterpError> {
    for b in batches {
        let idx = ops::key_indices(b, keys)?;
        partition_batch_into(b, &idx, p, salt, store)?;
    }
    Ok(())
}

/// Grace ASOF join: when an ASOF join with `by` keys is too large to hold both
/// sides in memory, partition both sides by the `by` keys to disk and ASOF-join one
/// bucket pair at a time — only one bucket of each side is ever resident. Equal `by`
/// values hash to the same bucket on both sides (the fixed-seed partitioner), and a
/// nearest-`on` match never crosses a `by` group, so each bucket is an independent
/// ASOF join and their union is the full result — identical to the in-memory path,
/// with bounded memory. Bucket count is sized so the larger side's bucket ≈ one
/// budget.
///
/// Takes the two sides as **morsels**, and streams each one through the partitioner into its
/// store ([`partition_batches_to_store`], shared with the grace hash join) — so at no point
/// is either input held whole. Taking them as one `RecordBatch` each, as this did, meant the
/// caller had concatenated both sides before it could ask whether they fit: an ASOF join over
/// inputs larger than the envelope OOMed at that concatenation, before the spill path it was
/// on its way to could bound anything. That is the same defect the grace hash join was
/// already fixed for, and this was the operator it was still live in.
#[allow(clippy::too_many_arguments)]
pub(crate) fn spilling_asof_join(
    left_batches: &[RecordBatch],
    right_batches: &[RecordBatch],
    left_on: &str,
    right_on: &str,
    left_by: &[String],
    right_by: &[String],
    direction: bc_ir::AsofDirection,
    tolerance: Option<f64>,
    allow_exact_matches: bool,
    output: &[bc_ir::JoinOutputCol],
    sp: &SpillOptions,
) -> Result<Vec<RecordBatch>, InterpError> {
    // Sized from the morsels' summed size, so the fan-out is chosen without materializing.
    let bytes = crate::batch_bytes(left_batches).max(crate::batch_bytes(right_batches)) as usize;
    // Capped like every other grace fan-out. Uncapped, a side three orders of magnitude over
    // the envelope asked for thousands of buckets per store — thousands of spill files, each
    // receiving shards too small to write efficiently — and the bucket that still did not fit
    // was materialized anyway, which is the failure the fan-out was trying to avoid.
    let budget = sp.memory_budget_bytes.max(1);
    let p = grace_bucket_count(bytes, budget);

    let mut lstore = DiskSpillStore::with_codec(sp.dir.join("asof-left"), p, sp.codec)?;
    let mut rstore = DiskSpillStore::with_codec(sp.dir.join("asof-right"), p, sp.codec)?;
    partition_batches_to_store(left_batches, left_by, p, 0, &mut lstore)?;
    partition_batches_to_store(right_batches, right_by, p, 0, &mut rstore)?;

    // A side with no batches at all is shortcut upstream and never reaches this path, so both
    // schemas are available — and both are needed, since a bucket may receive rows on one side
    // only and still has to ASOF-join against an empty relation of the right shape.
    let asof = AsofBuckets {
        left_on,
        right_on,
        left_by,
        right_by,
        direction,
        tolerance,
        allow_exact_matches,
        output,
        budget,
        codec: sp.codec,
        dir: &sp.dir,
        lschema: left_batches
            .first()
            .ok_or(InterpError::EmptyJoinInput)?
            .schema(),
        rschema: right_batches
            .first()
            .ok_or(InterpError::EmptyJoinInput)?
            .schema(),
    };
    let mut out = Vec::with_capacity(p);
    for i in 0..p {
        asof_bucket(&mut lstore, &mut rstore, i, &asof, 0, &mut out)?;
    }
    Ok(out)
}

/// The parts of a spilling ASOF join that do not change as buckets are re-split.
struct AsofBuckets<'a> {
    left_on: &'a str,
    right_on: &'a str,
    left_by: &'a [String],
    right_by: &'a [String],
    direction: bc_ir::AsofDirection,
    tolerance: Option<f64>,
    allow_exact_matches: bool,
    output: &'a [bc_ir::JoinOutputCol],
    budget: usize,
    codec: bc_runtime::agg::spill::SpillCodec,
    dir: &'a std::path::Path,
    lschema: arrow::datatypes::SchemaRef,
    rschema: arrow::datatypes::SchemaRef,
}

/// ASOF-join one co-partitioned bucket pair, re-splitting it first if it does not fit.
///
/// The same guard the hash join gets, and legal for the same reason with one extra step: a
/// nearest-`on` match never crosses a `by` group, so re-partitioning **by the `by` keys**
/// keeps every group whole in one sub-bucket and each sub-pair stays an independent ASOF
/// join. Ordering within a group is untouched, since the bucket's rows are re-ordered only
/// across sub-buckets and `asof_join_batches` orders what it is given.
///
/// Both sides are measured before either is read: the fan-out is sized from the *larger*
/// side's total, which says nothing about how any one `by` value is distributed.
fn asof_bucket(
    lstore: &mut dyn SpillStore,
    rstore: &mut dyn SpillStore,
    i: usize,
    ctx: &AsofBuckets<'_>,
    depth: u32,
    out: &mut Vec<RecordBatch>,
) -> Result<(), InterpError> {
    let biggest = (lstore.partition_bytes(i) as usize).max(rstore.partition_bytes(i) as usize);
    if depth < MAX_GRACE_SPLIT_DEPTH && biggest > ctx.budget {
        let sub_p = grace_bucket_count(biggest, ctx.budget);
        let salt = split_salt(depth + 1);
        let sub = ctx.dir.join(format!("asof-split-{depth}"));
        let mut lsub = DiskSpillStore::with_codec(sub.join("left"), sub_p, ctx.codec)?;
        let mut rsub = DiskSpillStore::with_codec(sub.join("right"), sub_p, ctx.codec)?;
        let lidx = schema_key_indices(&ctx.lschema, ctx.left_by)?;
        let ridx = schema_key_indices(&ctx.rschema, ctx.right_by)?;
        drain_repartition(lstore, i, sub_p, salt, &columns_at(&lidx), &mut lsub)?;
        drain_repartition(rstore, i, sub_p, salt, &columns_at(&ridx), &mut rsub)?;
        for j in 0..sub_p {
            asof_bucket(&mut lsub, &mut rsub, j, ctx, depth + 1, out)?;
        }
        return Ok(());
    }
    let lpart = materialize_or_empty(&lstore.read(i)?, &ctx.lschema)?;
    let rpart = materialize_or_empty(&rstore.read(i)?, &ctx.rschema)?;
    out.push(ops::asof_join_batches(
        &lpart,
        &rpart,
        ctx.left_on,
        ctx.right_on,
        ctx.left_by,
        ctx.right_by,
        ctx.direction,
        ctx.tolerance,
        ctx.allow_exact_matches,
        ctx.output,
    )?);
    Ok(())
}

/// Broadcast hash join: the build side is small enough to replicate, so the large
/// probe side is joined *without* being shuffled by key. Inner/left/semi/anti are
/// left-row-local, so the probe parallelizes over row-range chunks of the left
/// (each chunk joins the full right). A **right** join is its mirror — it is run as
/// a left join with the sides swapped and the output column sides flipped, so the
/// driving (right) side is the chunked probe (each right row lands in one chunk, no
/// duplication). **Full** must emit unmatched rows from *both* sides, which chunks
/// would duplicate, so it runs as a single pass. All cases avoid the shuffle.
/// Broadcast join **without materializing the probe side**: build the table once over the
/// (small) build batch, then probe each probe morsel independently, across cores.
///
/// Returns `Ok(None)` when the join cannot be streamed — a build-driven join type
/// (`Right`/`Full`), a non-integer key, or an empty probe side — and the caller keeps the
/// materialized [`broadcast_join`] path. Nothing silently changes shape.
///
/// The win is the copy that does not happen: [`ops::materialize`] on the probe side
/// concatenates the largest relation in the query into one `RecordBatch` before every
/// broadcast join. Streaming skips it, and skips the `remorselize` that had to undo the
/// concatenation afterwards — each morsel's output *is* a morsel. The emitted relation is
/// identical: morsels are contiguous in-order row ranges, so probing them in order emits the
/// same rows in the same order as slicing the concatenated batch by range (pinned by
/// `bc_runtime::join::stream`'s `morsel_by_morsel_matches_the_whole_relation`).
pub(crate) fn broadcast_join_streaming(
    probe_batches: &[RecordBatch],
    build: &RecordBatch,
    probe_keys: &[String],
    build_keys: &[String],
    join_type: bc_ir::JoinType,
    output: &[bc_ir::JoinOutputCol],
) -> Result<Option<Vec<RecordBatch>>, InterpError> {
    if probe_batches.is_empty() {
        return Ok(None);
    }
    let tuning = bc_arrow::RuntimeTuning::default();
    let build_key_cols = ops::columns_by_name(build, build_keys)?;
    let probe_rows: usize = probe_batches.iter().map(|b| b.num_rows()).sum();
    let Some(table) = join::BroadcastProbe::new(
        &build_key_cols,
        ops::map_join_type(join_type),
        probe_rows,
        tuning.bloom_fp_rate,
        tuning.bloom_min_build_rows,
    ) else {
        return Ok(None);
    };
    // Every morsel shares one schema, so one shape check covers them all and the per-morsel
    // probe below cannot fail.
    let first_keys = ops::columns_by_name(&probe_batches[0], probe_keys)?;
    if !table.accepts(&first_keys) {
        return Ok(None);
    }
    // One schema for every morsel's output: both sides' schemas are fixed for the join.
    let schema = ops::join_output_schema(&probe_batches[0], build, output)?;
    let out = probe_batches
        .par_iter()
        .map(|morsel| {
            let keys = ops::columns_by_name(morsel, probe_keys)?;
            let idx = table
                .probe(&keys)
                .ok_or_else(|| InterpError::UnknownJoinColumn(probe_keys.join(", ")))?;
            ops::gather_join_output_with(morsel, build, &idx, output, Arc::clone(&schema))
        })
        .collect::<Result<Vec<_>, InterpError>>()?;
    Ok(Some(out))
}

pub(crate) fn broadcast_join(
    left: &RecordBatch,
    right: &RecordBatch,
    left_keys: &[String],
    right_keys: &[String],
    join_type: bc_ir::JoinType,
    output: &[bc_ir::JoinOutputCol],
) -> Result<Vec<RecordBatch>, InterpError> {
    use bc_ir::{JoinStrategy, JoinType};
    let single_pass = || {
        ops::join_batches(
            left,
            right,
            left_keys,
            right_keys,
            join_type,
            output,
            JoinStrategy::Hash,
        )
    };
    // Full: a single pass (chunks would duplicate both sides' unmatched rows).
    if matches!(join_type, JoinType::Full) {
        return Ok(vec![single_pass()?]);
    }
    // An empty probe has no row-range chunks, so the chunked path below would produce
    // *zero* batches — and a batch is the only thing that carries a schema. Every
    // downstream pipeline breaker (join, aggregate, distinct) materializes its input and
    // needs that schema even over zero rows, so an empty relation must still be one
    // zero-row batch. The single-pass join produces exactly that. `Right` drives from
    // the right side, so test the side that will become the probe.
    let probe_is_right = matches!(join_type, JoinType::Right);
    let probe_rows = if probe_is_right {
        right.num_rows()
    } else {
        left.num_rows()
    };
    if probe_rows == 0 {
        return Ok(vec![single_pass()?]);
    }
    // Right: chunk the driving (right) side, joined against the full left as a LEFT
    // join with flipped keys + output sides. Mirror of the left-driven path.
    let (probe, build, pkeys, bkeys, jt, out): (_, _, _, _, _, Vec<bc_ir::JoinOutputCol>) =
        if matches!(join_type, JoinType::Right) {
            let flipped = flip_output(output);
            (right, left, right_keys, left_keys, JoinType::Left, flipped)
        } else {
            (
                left,
                right,
                left_keys,
                right_keys,
                join_type,
                output.to_vec(),
            )
        };
    // Build the (replicated) build-side hash table ONCE, then probe `probe` across `p`
    // parallel row-range chunks against that single shared table (rather than rebuilding
    // it per chunk — the old path's cost). Split the probe into ~one chunk per core so a
    // large probe is fully parallel; bound the count by the probe's morsel count so chunks
    // never shrink below a morsel (tiny chunks are pure scheduling overhead). The table is
    // built once, so chunk count governs *only* probe parallelism — it is independent of
    // the build size (the old `probe/build` cap throttled a 6 M-row probe to a handful of
    // chunks whenever the build was large). Each chunk gathers its own output batch; the
    // chunks concatenate to the full relation. Result-invariant: chunking only splits the probe.
    let max_chunks = (probe.num_rows() / bc_arrow::DEFAULT_MORSEL_ROWS).max(1);
    let p = rayon::current_num_threads().max(1).min(max_chunks);
    let probe_keys = ops::columns_by_name(probe, pkeys)?;
    let build_keys = ops::columns_by_name(build, bkeys)?;
    let tuning = bc_arrow::RuntimeTuning::default();
    let idxs = join::broadcast_hash_join_indices(
        &probe_keys,
        &build_keys,
        ops::map_join_type(jt),
        p,
        tuning.bloom_fp_rate,
        tuning.bloom_min_build_rows,
    )?;
    idxs.par_iter()
        .map(|idx| ops::gather_join_output(probe, build, idx, &out))
        .collect()
}

// A bucket is "skewed" when it holds far more probe rows than the average bucket
// (a hot key concentrating there) and is large enough that spreading it pays off.
// The live thresholds now flow in from `bc_arrow::RuntimeTuning` (default ==
// these values); the executor passes `opts.tuning.skew_*`. These consts remain as
// the canonical defaults the in-crate tests pin against — they equal
// `RuntimeTuning::default().skew_*`.
#[cfg(test)]
const SKEW_BUCKET_FACTOR: usize = 4;
#[cfg(test)]
pub(crate) const SKEW_MIN_BUCKET_ROWS: usize = 4 * bc_arrow::DEFAULT_MORSEL_ROWS;
/// Byte floor mirroring [`SKEW_MIN_BUCKET_ROWS`]. A bucket whose *bytes* dwarf the
/// average is a straggler even at a modest row count — wide rows (large strings,
/// blobs, embeddings) concentrate work the row-only test cannot see (65 k wide rows
/// look identical to 65 k narrow ones by row count).
#[cfg(test)]
pub(crate) const SKEW_MIN_BUCKET_BYTES: usize = 4 * bc_arrow::DEFAULT_MORSEL_BYTES;

/// `min_bucket_rows`/`bucket_factor` are performance-only (the default consts, or
/// the control plane's tuning): skew salting is result-invisible, so they change
/// only *which* buckets get spread across workers, never the relation.
pub(crate) fn is_skewed_bucket(
    bucket_rows: usize,
    avg_rows: usize,
    bucket_factor: usize,
    min_bucket_rows: usize,
) -> bool {
    bucket_rows >= min_bucket_rows && bucket_rows > bucket_factor * avg_rows.max(1)
}

/// Byte-aware companion to [`is_skewed_bucket`]: the same factor test on Arrow
/// bytes. The driving side of a bucket is hot if it is skewed by *either* rows or
/// bytes, so a hot key of wide rows triggers the same spread-the-bucket mitigation
/// that a hot key of many narrow rows already does.
pub(crate) fn is_skewed_bucket_bytes(
    bucket_bytes: usize,
    avg_bytes: usize,
    bucket_factor: usize,
    min_bucket_bytes: usize,
) -> bool {
    bucket_bytes >= min_bucket_bytes && bucket_bytes > bucket_factor * avg_bytes.max(1)
}

/// Skew salting (spreading a hot bucket's probe rows across worker chunks against
/// the full build bucket) is valid for the single-driving-side join types — each
/// probe row lands in exactly one chunk. `Right` qualifies via the flip in
/// `broadcast_join` (it chunks the driving right side). `Full` must emit unmatched
/// rows from both sides, so it keeps the single per-bucket join.
pub(crate) fn skew_salting_eligible(join_type: bc_ir::JoinType) -> bool {
    use bc_ir::JoinType;
    matches!(
        join_type,
        JoinType::Inner | JoinType::Left | JoinType::Semi | JoinType::Anti | JoinType::Right
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The build-side correction fires only where it can pay: a materially oversized
    /// build, on the one join flavor whose swap is a re-labeling.
    ///
    /// Each `assert!(!...)` here is a case the rule must decline, and each names the gate
    /// that declines it — a rule that fired on all of them would still pass a test that
    /// only checked the positive case.
    #[test]
    fn build_side_correction_fires_only_on_a_materially_oversized_inner_build() {
        use bc_ir::JoinType;
        let big = SWAP_MIN_BUILD_ROWS * 4;

        // The case it exists for: the planner nominated a build 4x the probe.
        assert!(build_side_swap_pays(JoinType::Inner, big / 4, big));
        // Exactly at the ratio still pays; a hair under it does not.
        assert!(build_side_swap_pays(JoinType::Inner, big / 2, big));
        assert!(!build_side_swap_pays(JoinType::Inner, big / 2 + 1, big));
        // A build the planner sized correctly is left alone.
        assert!(!build_side_swap_pays(JoinType::Inner, big, big / 4));

        // The one-morsel floor: a 10x ratio over a trivial build is still a 0 ms
        // difference, and swapping it would only churn the output order.
        assert!(!build_side_swap_pays(
            JoinType::Inner,
            SWAP_MIN_BUILD_ROWS / 10,
            SWAP_MIN_BUILD_ROWS - 1
        ));

        // Every non-`Inner` flavor declines, however lopsided. `Semi`/`Anti` emit left
        // rows and are not symmetric; `Left`/`Right` would trade the probe-driven fast
        // paths for the `Right` ones the engine declines; `Full` drives from both sides.
        for jt in [
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
            JoinType::Semi,
            JoinType::Anti,
        ] {
            assert!(!build_side_swap_pays(jt, big / 100, big));
        }
    }

    /// Flipping the output twice is the identity, which is the property that makes a
    /// build-side swap a re-labeling: only the side moves, never the column or its alias.
    #[test]
    fn flipping_the_output_sides_twice_is_the_identity() {
        use bc_ir::{JoinOutputCol, JoinSide};
        let output = vec![
            JoinOutputCol {
                side: JoinSide::Left,
                name: "a".into(),
                alias: "x".into(),
            },
            JoinOutputCol {
                side: JoinSide::Right,
                name: "b".into(),
                alias: "y".into(),
            },
        ];
        let once = flip_output(&output);
        assert!(matches!(once[0].side, JoinSide::Right));
        assert!(matches!(once[1].side, JoinSide::Left));

        let twice = flip_output(&once);
        for (before, after) in output.iter().zip(&twice) {
            assert!(matches!(
                (before.side, after.side),
                (JoinSide::Left, JoinSide::Left) | (JoinSide::Right, JoinSide::Right)
            ));
            assert_eq!(before.name, after.name);
            assert_eq!(before.alias, after.alias);
        }
    }

    /// The byte-aware skew test fires when a bucket's bytes dwarf the average even
    /// though its row count is far under the row-skew floor — the case a hot key of
    /// wide rows creates and the row-only test misses.
    #[test]
    fn byte_skew_fires_where_row_skew_cannot() {
        // A bucket well under SKEW_MIN_BUCKET_ROWS rows, so the row test is blind.
        let rows = 1_000;
        let avg_rows = 250;
        assert!(
            !is_skewed_bucket(rows, avg_rows, SKEW_BUCKET_FACTOR, SKEW_MIN_BUCKET_ROWS),
            "row test must not trip here"
        );

        // Same bucket carries wide rows: bytes exceed the floor and 4× the average.
        let bucket_bytes = SKEW_MIN_BUCKET_BYTES + 1;
        let avg_bytes = bucket_bytes / 8;
        assert!(
            is_skewed_bucket_bytes(
                bucket_bytes,
                avg_bytes,
                SKEW_BUCKET_FACTOR,
                SKEW_MIN_BUCKET_BYTES
            ),
            "byte test must detect a wide-row hot bucket"
        );
    }

    /// Byte skew respects both gates: a bucket above 4× average but below the byte
    /// floor is not hot (spreading a small bucket would not pay off).
    #[test]
    fn byte_skew_requires_the_floor() {
        let small = SKEW_MIN_BUCKET_BYTES / 2;
        assert!(!is_skewed_bucket_bytes(
            small,
            small / 8,
            SKEW_BUCKET_FACTOR,
            SKEW_MIN_BUCKET_BYTES
        ));
        // And a large bucket only modestly above average is not hot either.
        assert!(!is_skewed_bucket_bytes(
            SKEW_MIN_BUCKET_BYTES,
            SKEW_MIN_BUCKET_BYTES,
            SKEW_BUCKET_FACTOR,
            SKEW_MIN_BUCKET_BYTES
        ));
    }

    /// The fan-out follows the larger side, so a small build against a huge probe does not
    /// leave every probe bucket a thousand chunks long.
    ///
    /// Sizing from the build side alone was right while both sides were materialized whole —
    /// only the build bounded memory. Once the probe is streamed past the build, the fan-out
    /// stops being a memory decision and becomes a *rebuild-count* one: the build table is
    /// rebuilt once per probe chunk, and the chunk count is `probe_bucket / budget`.
    #[test]
    fn the_grace_fanout_follows_the_larger_side() {
        let budget = 1_000;
        // A star join: a small dimension, a fact table a hundred budgets long. Sized from the
        // build side this was the two-bucket floor; the probe side is what decides it.
        assert_eq!(grace_fanout(100_000, 1_500, budget), 100);
        // Symmetric — neither side is privileged.
        assert_eq!(grace_fanout(1_500, 100_000, budget), 100);
        // Both inside the budget is still the floor of two: a grace join that fans out one way
        // has not partitioned anything.
        assert_eq!(grace_fanout(500, 500, budget), 2);
        // And it stays under the fan-out cap however lopsided the input.
        assert_eq!(
            grace_fanout(budget * 100_000, 1, budget),
            crate::spill_split::MAX_GRACE_FANOUT
        );
    }

    // --- the streamed probe: grace join under key skew ---------------------------------

    use arrow::array::{Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use bc_ir::{JoinOutputCol, JoinSide, JoinType};

    /// A two-column `(k, v)` batch, nullable so a null key is expressible.
    fn kv(name_k: &str, name_v: &str, ks: &[Option<i64>], vs: &[i64]) -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new(name_k, DataType::Int64, true),
            Field::new(name_v, DataType::Int64, true),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int64Array::from(ks.to_vec())),
                Arc::new(Int64Array::from(vs.to_vec())),
            ],
        )
        .unwrap()
    }

    fn out_col(side: JoinSide, name: &str, alias: &str) -> JoinOutputCol {
        JoinOutputCol {
            side,
            name: name.into(),
            alias: alias.into(),
        }
    }

    /// The result as an order-independent multiset of stringified rows, so the spilled and
    /// in-memory relations compare as the unordered relations they are.
    fn multiset(batches: &[RecordBatch]) -> Vec<String> {
        let mut rows = Vec::new();
        for b in batches {
            let cols: Vec<&Int64Array> = (0..b.num_columns())
                .map(|c| b.column(c).as_any().downcast_ref::<Int64Array>().unwrap())
                .collect();
            for r in 0..b.num_rows() {
                let cells: Vec<String> = cols
                    .iter()
                    .map(|c| match c.is_null(r) {
                        true => "∅".to_string(),
                        false => c.value(r).to_string(),
                    })
                    .collect();
                rows.push(cells.join("|"));
            }
        }
        rows.sort();
        rows
    }

    fn spill_opts(tag: &str) -> SpillOptions {
        SpillOptions {
            // One byte: every bucket is "over budget", so the split guard is exercised to
            // its depth limit and the streamed probe is the only thing bounding memory.
            memory_budget_bytes: 1,
            dir: std::env::temp_dir().join(format!(
                "bc_join_stream_{}_{}_{tag}",
                std::process::id(),
                SPILL_TEST_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            )),
            codec: bc_runtime::agg::spill::SpillCodec::None,
        }
    }

    static SPILL_TEST_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    /// Probe morsels dominated by one key, against a build side holding that key once —
    /// the shape a `-1`/`unknown` sentinel in a fact table produces.
    ///
    /// `hot` rows of key `7` spread over `morsels` batches, plus a cold key per morsel and a
    /// null key (which matches nothing but must still be emitted by the outer flavors).
    fn skewed_inputs(morsels: usize, hot: usize) -> (Vec<RecordBatch>, Vec<RecordBatch>) {
        let per = hot / morsels;
        let left: Vec<RecordBatch> = (0..morsels)
            .map(|m| {
                let mut ks: Vec<Option<i64>> = vec![Some(7); per];
                let mut vs: Vec<i64> = (0..per as i64).map(|i| (m * per) as i64 + i).collect();
                ks.push(Some(100 + m as i64)); // a cold key, matched only for m == 0
                vs.push(-1);
                ks.push(None); // a null key matches nothing, on either side
                vs.push(-2);
                kv("lk", "lv", &ks, &vs)
            })
            .collect();
        // The build side is one row for the hot key, one for a cold key that matches, and one
        // for a key nothing probes (so `Right`/`Full` have a genuine unmatched remainder).
        let right = vec![kv(
            "rk",
            "rv",
            &[Some(7), Some(100), Some(999), None],
            &[70, 1000, 9990, -9],
        )];
        (left, right)
    }

    /// The streamed probe computes the same relation the in-memory join does, for every
    /// join flavor, on an input no re-partition can balance.
    ///
    /// This is the case the recursive split cannot fix and never could: the hot key re-hashes
    /// to one sub-bucket at every level, so bounding memory has to come from *not holding the
    /// probe bucket*, which is what the streamed probe does.
    #[test]
    fn a_skewed_grace_join_equals_the_in_memory_join_for_every_flavor() {
        let (left, right) = skewed_inputs(8, 400);
        let lmat = ops::materialize(&left).unwrap();
        let rmat = ops::materialize(&right).unwrap();

        let both = vec![
            out_col(JoinSide::Left, "lk", "lk"),
            out_col(JoinSide::Left, "lv", "lv"),
            out_col(JoinSide::Right, "rk", "rk"),
            out_col(JoinSide::Right, "rv", "rv"),
        ];
        // Semi/anti emit left columns only — the planner never names the other side for them.
        let left_only = vec![
            out_col(JoinSide::Left, "lk", "lk"),
            out_col(JoinSide::Left, "lv", "lv"),
        ];

        for jt in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
            JoinType::Semi,
            JoinType::Anti,
        ] {
            let output = match jt {
                JoinType::Semi | JoinType::Anti => &left_only,
                _ => &both,
            };
            let want = ops::join_batches(
                &lmat,
                &rmat,
                &["lk".into()],
                &["rk".into()],
                jt,
                output,
                bc_ir::JoinStrategy::Hash,
            )
            .unwrap();
            let (got, _) = spilling_hash_join_streaming(
                &left,
                &right,
                &["lk".into()],
                &["rk".into()],
                jt,
                output,
                &spill_opts(&format!("{jt:?}")),
            )
            .unwrap();
            assert_eq!(
                multiset(std::slice::from_ref(&want)),
                multiset(&got),
                "{jt:?}: spilled relation differs from the in-memory one"
            );
        }
    }

    /// The probe bucket is *streamed*, not materialized — the property the memory bound
    /// rests on, observed where a unit test can see it.
    ///
    /// Every probe row here carries the same key, so all of them land in one bucket however
    /// the hash is salted. Holding that bucket produces exactly one output batch; consuming it
    /// a morsel at a time produces one per morsel. The batch count is therefore a direct
    /// reading of whether the bucket was held.
    #[test]
    fn a_single_key_probe_bucket_is_consumed_a_morsel_at_a_time() {
        let morsels = 6;
        let left: Vec<RecordBatch> = (0..morsels)
            .map(|m| kv("lk", "lv", &[Some(7); 4], &[m as i64; 4]))
            .collect();
        let right = vec![kv("rk", "rv", &[Some(7)], &[70])];
        let output = vec![
            out_col(JoinSide::Left, "lv", "lv"),
            out_col(JoinSide::Right, "rv", "rv"),
        ];
        let (got, _) = spilling_hash_join_streaming(
            &left,
            &right,
            &["lk".into()],
            &["rk".into()],
            JoinType::Inner,
            &output,
            &spill_opts("stream"),
        )
        .unwrap();
        assert_eq!(
            got.len(),
            morsels,
            "one output batch per probe morsel; {} means the bucket was materialized",
            got.len()
        );
        assert_eq!(got.iter().map(|b| b.num_rows()).sum::<usize>(), morsels * 4);
    }

    /// An empty result still carries the join's schema.
    ///
    /// The streamed probe drops empty batches, so the empty relation has to be stated once
    /// rather than falling out of the per-bucket loop — and a caller that lost the schema
    /// would fail downstream, far from here.
    #[test]
    fn an_empty_spilled_join_still_reports_its_columns() {
        let left = vec![kv("lk", "lv", &[Some(1)], &[10])];
        let right = vec![kv("rk", "rv", &[Some(2)], &[20])];
        let output = vec![
            out_col(JoinSide::Left, "lv", "lv"),
            out_col(JoinSide::Right, "rv", "rv"),
        ];
        let (got, _) = spilling_hash_join_streaming(
            &left,
            &right,
            &["lk".into()],
            &["rk".into()],
            JoinType::Inner,
            &output,
            &spill_opts("empty"),
        )
        .unwrap();
        assert_eq!(got.iter().map(|b| b.num_rows()).sum::<usize>(), 0);
        let schema = got
            .first()
            .expect("an empty join still emits a schema")
            .schema();
        assert_eq!(
            schema
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .collect::<Vec<_>>(),
            vec!["lv", "rv"]
        );
    }
}
