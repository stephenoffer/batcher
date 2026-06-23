//! Bounded-memory window execution via grace partitioning.
//!
//! Window functions are per-partition-independent, and equal `PARTITION BY` keys
//! hash to the same bucket, so the input can be grace-partitioned by those keys
//! into disk-backed buckets and the in-memory window kernel run one bucket at a
//! time. Each bucket holds *complete* partitions, so the result is the same
//! multiset as the single-pass kernel (window outputs are unordered relations),
//! with peak resident memory bounded to the largest bucket instead of the whole
//! input. This reuses the same grace algebra and `DiskSpillStore` as the aggregate
//! spill path — the one mechanism, applied to a different operator.

use std::path::Path;

use arrow::array::{ArrayRef, RecordBatch};
use bc_runtime::agg::spill::{DiskSpillStore, SpillStore};
use bc_runtime::shuffle;

use crate::batch_bytes;
use crate::error::InterpError;
use crate::ops;

/// Run a window operator under a memory envelope by grace-partitioning on the
/// `PARTITION BY` keys. Caller guarantees `partition_keys` is non-empty (a single
/// global partition cannot be split for ranking/running aggregates).
pub(crate) fn window_spilling(
    parts: &[RecordBatch],
    partition_keys: &[bc_expr::Expr],
    order_keys: &[bc_ir::SortKey],
    functions: &[bc_ir::WindowFunc],
    rank_limit: Option<usize>,
    budget_bytes: usize,
    dir: &Path,
) -> Result<Vec<RecordBatch>, InterpError> {
    let p = (batch_bytes(parts) as usize)
        .div_ceil(budget_bytes.max(1))
        .max(2);
    let mut store = DiskSpillStore::new(dir.join("window"), p)?;
    for batch in parts {
        let keys: Vec<ArrayRef> = partition_keys
            .iter()
            .map(|e| e.eval(batch))
            .collect::<Result<_, _>>()?;
        for (i, bucket) in shuffle::partition_by_key_arrays(batch, &keys, p)?
            .iter()
            .enumerate()
        {
            if bucket.num_rows() > 0 {
                store.append(i, bucket)?;
            }
        }
    }
    let mut out = Vec::new();
    for i in 0..p {
        let bucket = store.read(i)?;
        if bucket.is_empty() {
            continue;
        }
        let combined = ops::materialize(&bucket)?;
        out.push(ops::window_batch(
            &combined,
            partition_keys,
            order_keys,
            functions,
            rank_limit,
        )?);
    }
    Ok(out)
}
