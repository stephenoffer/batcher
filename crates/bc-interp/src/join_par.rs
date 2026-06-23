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
use bc_runtime::shuffle;
use rayon::prelude::*;

use crate::error::InterpError;
use crate::ops;
use crate::par::SpillOptions;

/// Grace hash join: the build (right) side exceeds the budget, so partition both
/// sides by key to disk and join one bucket at a time — only one build table is
/// resident. Bucket count is sized so each bucket's build side ≈ one budget.
pub(crate) fn spilling_hash_join(
    left: &RecordBatch,
    right: &RecordBatch,
    left_keys: &[String],
    right_keys: &[String],
    join_type: bc_ir::JoinType,
    output: &[bc_ir::JoinOutputCol],
    sp: &SpillOptions,
) -> Result<Vec<RecordBatch>, InterpError> {
    let li = ops::key_indices(left, left_keys)?;
    let ri = ops::key_indices(right, right_keys)?;
    // Enough partitions that each bucket's build side ≈ one budget.
    let p = right
        .get_array_memory_size()
        .div_ceil(sp.memory_budget_bytes.max(1))
        .max(2);
    let lb = shuffle::partition_by_keys(left, &li, p)?;
    let rb = shuffle::partition_by_keys(right, &ri, p)?;

    let mut lstore = DiskSpillStore::new(sp.dir.join("join-left"), p)?;
    let mut rstore = DiskSpillStore::new(sp.dir.join("join-right"), p)?;
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
        out.push(ops::join_batches(
            &lpart,
            &rpart,
            left_keys,
            right_keys,
            join_type,
            output,
            bc_ir::JoinStrategy::Hash,
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
pub(crate) fn broadcast_join(
    left: &RecordBatch,
    right: &RecordBatch,
    left_keys: &[String],
    right_keys: &[String],
    join_type: bc_ir::JoinType,
    output: &[bc_ir::JoinOutputCol],
) -> Result<Vec<RecordBatch>, InterpError> {
    use bc_ir::{JoinSide, JoinStrategy, JoinType};
    // Full: a single pass (chunks would duplicate both sides' unmatched rows).
    if matches!(join_type, JoinType::Full) {
        return Ok(vec![ops::join_batches(
            left,
            right,
            left_keys,
            right_keys,
            join_type,
            output,
            JoinStrategy::Hash,
        )?]);
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
    let p = rayon::current_num_threads().max(1);
    split_rows(probe, p)
        .par_iter()
        .map(|chunk| ops::join_batches(chunk, build, pkeys, bkeys, jt, &out, JoinStrategy::Hash))
        .collect()
}

/// A bucket is "skewed" when it holds far more probe rows than the average bucket
/// (a hot key concentrating there) and is large enough that spreading it pays off.
const SKEW_BUCKET_FACTOR: usize = 4;
pub(crate) const SKEW_MIN_BUCKET_ROWS: usize = 4 * bc_arrow::DEFAULT_MORSEL_ROWS;

pub(crate) fn is_skewed_bucket(bucket_rows: usize, avg_rows: usize) -> bool {
    bucket_rows >= SKEW_MIN_BUCKET_ROWS && bucket_rows > SKEW_BUCKET_FACTOR * avg_rows.max(1)
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

/// Split a batch into at most `parts` contiguous, near-equal row-range slices
/// (zero-copy). Empty batches yield a single empty slice.
fn split_rows(batch: &RecordBatch, parts: usize) -> Vec<RecordBatch> {
    let n = batch.num_rows();
    if n == 0 || parts <= 1 {
        return vec![batch.clone()];
    }
    let per = n.div_ceil(parts);
    let mut out = Vec::with_capacity(parts);
    let mut off = 0;
    while off < n {
        let len = per.min(n - off);
        out.push(batch.slice(off, len));
        off += len;
    }
    out
}
