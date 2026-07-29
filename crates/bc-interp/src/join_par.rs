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
    // Enough partitions that each bucket's build side ≈ one budget — sized from the build
    // batches' total size without materializing them, then capped (see `MAX_JOIN_FANOUT`).
    let build_bytes: usize = right_batches
        .iter()
        .map(|b| b.get_array_memory_size())
        .sum();
    let budget = sp.memory_budget_bytes.max(1);
    let p = build_bytes.div_ceil(budget).clamp(2, MAX_JOIN_FANOUT);

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
    // Both sides were streamed to disk; the spill volume is their combined written bytes.
    let spill_bytes = lstore.spilled_bytes() + rstore.spilled_bytes();
    Ok((out, spill_bytes))
}

/// Upper bound on how many ways one level of the grace join fans out.
///
/// The bucket count wants to be `build_bytes / budget`, which for a build side three orders
/// of magnitude over the envelope is thousands of buckets — thousands of open spill files
/// per side, each receiving shards too small to write efficiently. Capping the fan-out and
/// re-splitting the buckets that still do not fit (see [`join_bucket`]) reaches the same
/// per-bucket size with bounded descriptors and full-sized writes, at the cost of re-reading
/// only the buckets that were actually too big.
const MAX_JOIN_FANOUT: usize = 256;

/// How many times a bucket that still exceeds the budget may be re-split.
///
/// Each level multiplies the effective partition count by up to [`MAX_JOIN_FANOUT`], so
/// three levels is 16 million buckets — far past any real ratio of build side to envelope.
/// The bound exists for the case no hash can fix: when a *single* join key's rows exceed the
/// budget, every re-split sends them all to one sub-bucket again, and without a limit the
/// recursion would never terminate. At the limit the bucket is joined as-is, which is what
/// the join did unconditionally before.
const MAX_JOIN_SPLIT_DEPTH: u32 = 3;

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

/// Join co-partitioned bucket `i` of the two stores, re-splitting it first if either side
/// does not fit in the envelope.
///
/// The bucket count is sized from the build side's *average* bytes per bucket, so it is only
/// an average-case fit. Under key skew one bucket holds far more than its share — and both
/// sides were materialized whole before being joined, so a skewed bucket OOMs at exactly the
/// point spilling was supposed to have prevented it. This is the failure mode that makes
/// skewed joins the standard reason a Spark job dies, and the grace *aggregate* already
/// guards against it by recursively re-partitioning an over-large partition; the join did
/// not.
///
/// Both sides are asked *before* being read, which is the whole point: the decision to split
/// has to happen without first pulling the partition that provably does not fit into memory.
/// The probe side counts too, not just the build side — the bucket count is sized from the
/// build side alone, so a fact table with a hot key can leave a probe bucket orders of
/// magnitude over the envelope even when every build bucket fits.
///
/// A re-split partitions **both** sides with a salt derived from the depth. The salt is a
/// function of the depth alone, never of the row, so equal keys still co-locate on both
/// sides and each sub-bucket remains an independent join whose union is the same relation.
fn join_bucket(
    lstore: &mut dyn SpillStore,
    rstore: &mut dyn SpillStore,
    i: usize,
    ctx: &BucketJoin<'_>,
    depth: u32,
    out: &mut Vec<RecordBatch>,
) -> Result<(), InterpError> {
    let lbytes = lstore.partition_bytes(i) as usize;
    let rbytes = rstore.partition_bytes(i) as usize;
    if depth < MAX_JOIN_SPLIT_DEPTH && lbytes.max(rbytes) > ctx.budget {
        return split_and_join_bucket(lstore, rstore, i, ctx, depth, lbytes.max(rbytes), out);
    }
    let lpart = materialize_or_empty(&lstore.read(i)?, &ctx.lschema)?;
    let rpart = materialize_or_empty(&rstore.read(i)?, &ctx.rschema)?;
    out.push(ops::join_batches(
        &lpart,
        &rpart,
        ctx.left_keys,
        ctx.right_keys,
        ctx.join_type,
        ctx.output,
        bc_ir::JoinStrategy::Hash,
    )?);
    Ok(())
}

/// Re-partition an over-large bucket on both sides and join the sub-buckets.
///
/// Each side is streamed out of its store one batch at a time and straight into a fresh
/// child store, so the bucket that did not fit is never held whole — the peak is one batch
/// plus one sub-bucket, which is what the caller's budget was meant to describe all along.
fn split_and_join_bucket(
    lstore: &mut dyn SpillStore,
    rstore: &mut dyn SpillStore,
    i: usize,
    ctx: &BucketJoin<'_>,
    depth: u32,
    bytes: usize,
    out: &mut Vec<RecordBatch>,
) -> Result<(), InterpError> {
    let sub_p = bytes.div_ceil(ctx.budget).clamp(2, MAX_JOIN_FANOUT);
    let salt = split_salt(depth + 1);
    let sub = ctx.dir.join(format!("join-split-{depth}"));
    let mut lsub = DiskSpillStore::with_codec(sub.join("left"), sub_p, ctx.codec)?;
    let mut rsub = DiskSpillStore::with_codec(sub.join("right"), sub_p, ctx.codec)?;
    drain_into_store(
        lstore,
        i,
        &schema_key_indices(&ctx.lschema, ctx.left_keys)?,
        sub_p,
        salt,
        &mut lsub,
    )?;
    drain_into_store(
        rstore,
        i,
        &schema_key_indices(&ctx.rschema, ctx.right_keys)?,
        sub_p,
        salt,
        &mut rsub,
    )?;
    for j in 0..sub_p {
        join_bucket(&mut lsub, &mut rsub, j, ctx, depth + 1, out)?;
    }
    Ok(())
}

/// The salt for re-splitting at `depth` — any value that is non-zero (0 means "unsalted",
/// the cluster-wide assignment) and distinct per level, so successive splits of the same
/// rows are independent of one another.
fn split_salt(depth: u32) -> u64 {
    0x9E37_79B9_7F4A_7C15u64.wrapping_mul(depth as u64 + 1) | 1
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

/// Stream partition `i` out of `store` and re-partition it into `dest` by the key columns at
/// `key_idx`, one batch at a time, never holding the partition whole.
fn drain_into_store(
    store: &mut dyn SpillStore,
    i: usize,
    key_idx: &[usize],
    p: usize,
    salt: u64,
    dest: &mut DiskSpillStore,
) -> Result<(), InterpError> {
    store.drain(i, &mut |b| partition_batch_into(b, key_idx, p, salt, dest))?;
    Ok(())
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn spilling_asof_join(
    left: &RecordBatch,
    right: &RecordBatch,
    left_on: &str,
    right_on: &str,
    left_by: &[String],
    right_by: &[String],
    backward: bool,
    output: &[bc_ir::JoinOutputCol],
    sp: &SpillOptions,
) -> Result<Vec<RecordBatch>, InterpError> {
    let li = ops::key_indices(left, left_by)?;
    let ri = ops::key_indices(right, right_by)?;
    let bytes = left
        .get_array_memory_size()
        .max(right.get_array_memory_size());
    let p = bytes.div_ceil(sp.memory_budget_bytes.max(1)).max(2);
    let lb = shuffle::partition_by_keys(left, &li, p)?;
    let rb = shuffle::partition_by_keys(right, &ri, p)?;

    let mut lstore = DiskSpillStore::with_codec(sp.dir.join("asof-left"), p, sp.codec)?;
    let mut rstore = DiskSpillStore::with_codec(sp.dir.join("asof-right"), p, sp.codec)?;
    for i in 0..p {
        lstore.append(i, &lb[i])?;
        rstore.append(i, &rb[i])?;
    }
    drop(lb);
    drop(rb);

    let mut out = Vec::with_capacity(p);
    for i in 0..p {
        let lpart = ops::materialize(&lstore.read(i)?)?;
        let rpart = ops::materialize(&rstore.read(i)?)?;
        out.push(ops::asof_join_batches(
            &lpart, &rpart, left_on, right_on, left_by, right_by, backward, output,
        )?);
    }
    Ok(out)
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
    use bc_ir::{JoinSide, JoinStrategy, JoinType};
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
            let flipped = output
                .iter()
                .map(|o| bc_ir::JoinOutputCol {
                    side: match o.side {
                        JoinSide::Left => JoinSide::Right,
                        JoinSide::Right => JoinSide::Left,
                    },
                    name: o.name.clone(),
                    alias: o.alias.clone(),
                })
                .collect();
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
}
