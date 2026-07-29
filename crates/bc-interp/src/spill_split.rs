//! Re-splitting a grace bucket that did not fit — the shared skew guard.
//!
//! Every grace operator sizes its bucket count the same way: total bytes over the memory
//! envelope. That is an *average*-case fit, and it is the only thing standing between a
//! spilling operator and an OOM. Under key skew one bucket holds far more than its share, so
//! the operator materializes a bucket that provably does not fit at exactly the point
//! spilling was supposed to have prevented it. This is the failure mode that makes skewed
//! joins and skewed windows the standard reason a Spark job dies.
//!
//! The guard is the same in every case, which is why it lives here rather than being pasted
//! into each operator: measure the bucket *before* reading it, and if it is over budget,
//! stream it into a fresh child store re-partitioned by the same keys under a salt derived
//! from the recursion depth. The salt is a function of the depth alone and never of the row,
//! so equal keys still co-locate — which is what keeps each sub-bucket an independent
//! instance of the same operator whose union is the same relation.
//!
//! The grace *aggregate* has its own copy of this shape inside `bc-runtime`, where it can
//! see the partial-state layout it routes on; it is not reachable from here.

use arrow::array::{ArrayRef, RecordBatch};
use bc_runtime::agg::spill::{DiskSpillStore, SpillStore};
use bc_runtime::shuffle;

use crate::error::InterpError;

/// Upper bound on how many ways one level of a grace operator fans out.
///
/// The bucket count wants to be `bytes / budget`, which for an input three orders of
/// magnitude over the envelope is thousands of buckets — thousands of open spill files, each
/// receiving shards too small to write efficiently, and (before the store learned to release
/// them) thousands of open descriptors. Capping the fan-out and re-splitting the buckets
/// that still do not fit reaches the same per-bucket size with bounded descriptors and
/// full-sized writes, at the cost of re-reading only the buckets that were actually too big.
pub(crate) const MAX_GRACE_FANOUT: usize = 256;

/// How many times a bucket that still exceeds the budget may be re-split.
///
/// Each level multiplies the effective bucket count by up to [`MAX_GRACE_FANOUT`], so three
/// levels is 16 million buckets — far past any real ratio of input to envelope. The bound
/// exists for the case no hash can fix: when a *single* key's rows exceed the budget, every
/// re-split sends them all to one sub-bucket again, and without a limit the recursion would
/// never terminate. At the limit the bucket is processed as-is, which is what these
/// operators did unconditionally before.
pub(crate) const MAX_GRACE_SPLIT_DEPTH: u32 = 3;

/// Buckets to split `bytes` into so each lands inside `budget`, within the fan-out cap.
pub(crate) fn grace_bucket_count(bytes: usize, budget: usize) -> usize {
    bytes.div_ceil(budget.max(1)).clamp(2, MAX_GRACE_FANOUT)
}

/// The salt for re-splitting at `depth`.
///
/// Any value that is non-zero — 0 means "unsalted", which is the cluster-wide bucket
/// assignment and must never be used for a local re-split — and distinct per level, so
/// successive splits of the same rows are independent of one another.
pub(crate) fn split_salt(depth: u32) -> u64 {
    0x9E37_79B9_7F4A_7C15u64.wrapping_mul(depth as u64 + 1) | 1
}

/// Stream `partition` out of `store` and into `dest`, re-partitioned `p` ways by `keys_of`
/// under `salt`, one batch at a time.
///
/// The bucket that did not fit is never held whole: the store hands over one batch, it is
/// sharded and appended, and it is released before the next arrives. Empty shards are
/// skipped — at a 256-way fan-out over thousands of batches, writing them would be millions
/// of IPC messages carrying no rows — so a sub-bucket that received nothing has no file at
/// all, and callers must treat an empty read as an empty relation rather than an error.
pub(crate) fn drain_repartition(
    store: &mut dyn SpillStore,
    partition: usize,
    p: usize,
    salt: u64,
    keys_of: &dyn Fn(&RecordBatch) -> Result<Vec<ArrayRef>, InterpError>,
    dest: &mut DiskSpillStore,
) -> Result<(), InterpError> {
    // `drain`'s callback reports the *store's* error type, but deriving the partition keys
    // can fail with a plan error (an unknown column, an expression that does not evaluate).
    // Carrying it out here rather than flattening it into an I/O error is the difference
    // between "spill i/o error" and the real cause. The sentinel below only ever ends the
    // drain; `first_err` is what is actually reported.
    let mut first_err: Option<InterpError> = None;
    let drained = store.drain(partition, &mut |batch| match repartition_one(
        batch, p, salt, keys_of, dest,
    ) {
        Ok(()) => Ok(()),
        Err(e) => {
            first_err = Some(e);
            Err(bc_runtime::RuntimeError::MalformedPartial {
                expected: 0,
                got: 0,
            })
        }
    });
    if let Some(e) = first_err {
        return Err(e);
    }
    drained.map_err(InterpError::from)
}

/// Shard one batch by `keys_of` into `p` salted buckets and append each non-empty shard.
fn repartition_one(
    batch: &RecordBatch,
    p: usize,
    salt: u64,
    keys_of: &dyn Fn(&RecordBatch) -> Result<Vec<ArrayRef>, InterpError>,
    dest: &mut DiskSpillStore,
) -> Result<(), InterpError> {
    let keys = keys_of(batch)?;
    for (i, shard) in shuffle::partition_by_key_arrays_salted(batch, &keys, p, salt)?
        .iter()
        .enumerate()
    {
        if shard.num_rows() > 0 {
            dest.append(i, shard)?;
        }
    }
    Ok(())
}
