//! Bounded-memory `DISTINCT ON` via grace partitioning.
//!
//! Deduplicating on a key subset is per-key independent, and equal keys hash to the same
//! bucket, so the input can be grace-partitioned by the dedup key into disk-backed buckets and
//! reduced one bucket at a time. Each bucket holds *every* row of the keys it holds, so each
//! bucket's reduction is final and their union is the whole answer — the same relation the
//! in-memory path returns, with peak resident memory bounded to the largest bucket.
//!
//! Two things make this cheaper than the window spill it is modelled on. Each morsel is
//! reduced *before* it is written, so what reaches the disk is one row per key per morsel
//! rather than the whole relation — on a low-cardinality key that is most of the spill gone.
//! And the reduction is a `min`, not a ranking, so a bucket that has to be re-split does not
//! need its keys kept whole for correctness, only for efficiency.
//!
//! This reuses the same grace algebra, `DiskSpillStore` and re-split machinery as the
//! aggregate and window spill paths: one mechanism, a third operator.

use std::path::Path;

use arrow::array::{ArrayRef, RecordBatch};
use bc_runtime::agg::spill::{DiskSpillStore, SpillCodec, SpillStore};
use bc_runtime::agg::{self, OrderKey};
use bc_runtime::shuffle;

use crate::batch_bytes;
use crate::error::InterpError;
use crate::ops;
use crate::spill_split::{
    drain_repartition, grace_bucket_count, split_salt, MAX_GRACE_SPLIT_DEPTH,
};

/// Reduce a `DISTINCT ON` under a memory envelope by grace-partitioning on its dedup key.
///
/// `parts` must already carry every ordering column (`ops::distinct_on_widen`), so a bucket
/// read back from disk can be reduced without re-evaluating anything. Returns the reduced
/// batches — still widened; the caller narrows — and the measured spill volume.
pub(crate) fn distinct_on_spilling(
    parts: &[RecordBatch],
    key_indices: &[usize],
    order: &[OrderKey],
    budget_bytes: usize,
    dir: &Path,
    codec: SpillCodec,
) -> Result<(Vec<RecordBatch>, u64), InterpError> {
    let p = grace_bucket_count(batch_bytes(parts) as usize, budget_bytes);
    let mut store = DiskSpillStore::with_codec(dir.join("distinct-on"), p, codec)?;
    let keys_of = |batch: &RecordBatch| -> Result<Vec<ArrayRef>, InterpError> {
        Ok(key_indices
            .iter()
            .map(|&i| batch.column(i).clone())
            .collect())
    };
    for batch in parts {
        // Reduce the morsel before writing it. The reduction is idempotent, so this changes
        // no answer, and it is the difference between spilling the relation and spilling its
        // per-morsel distinct keys.
        let reduced = agg::distinct_on(batch, key_indices, order)?;
        for (i, bucket) in shuffle::partition_by_keys(&reduced, key_indices, p)?
            .iter()
            .enumerate()
        {
            if bucket.num_rows() > 0 {
                store.append(i, bucket)?;
            }
        }
    }
    let spill_bytes = store.spilled_bytes(); // measured volume routed to disk
    let ctx = DistinctBuckets {
        key_indices,
        order,
        budget_bytes,
        codec,
        dir,
        keys_of: &keys_of,
    };
    let mut out = Vec::with_capacity(p);
    for i in 0..p {
        distinct_bucket(&mut store, i, &ctx, 0, &mut out)?;
    }
    Ok((out, spill_bytes))
}

/// The parts of a spilling `DISTINCT ON` that do not change as buckets are re-split.
struct DistinctBuckets<'a> {
    key_indices: &'a [usize],
    order: &'a [OrderKey],
    budget_bytes: usize,
    codec: SpillCodec,
    dir: &'a Path,
    keys_of: &'a dyn Fn(&RecordBatch) -> Result<Vec<ArrayRef>, InterpError>,
}

/// Reduce bucket `i`, re-splitting it first if it does not fit the envelope.
///
/// The bucket count is sized from the input's *average* bytes per bucket, so a skewed key
/// leaves one bucket over the envelope; asking the store for a bucket's size before reading it
/// is what lets the split happen without first pulling in the thing that does not fit.
///
/// A re-split re-partitions by the same dedup key under a depth-derived salt, so equal keys
/// still land together and each sub-bucket's reduction is still final. Unlike the window's,
/// this split is an efficiency measure rather than a correctness one — a key split across two
/// buckets would yield two survivors, but the depth bound below is what keeps a single hot key
/// from recursing forever, and a hot key's rows re-hash together however they are salted. Past
/// the bound the bucket is reduced as-is, which is safe because the reduction that runs there
/// is exactly the in-memory one.
fn distinct_bucket(
    store: &mut dyn SpillStore,
    i: usize,
    ctx: &DistinctBuckets<'_>,
    depth: u32,
    out: &mut Vec<RecordBatch>,
) -> Result<(), InterpError> {
    let bytes = store.partition_bytes(i) as usize;
    if depth < MAX_GRACE_SPLIT_DEPTH && bytes > ctx.budget_bytes {
        let sub_p = grace_bucket_count(bytes, ctx.budget_bytes);
        let mut sub = DiskSpillStore::with_codec(
            ctx.dir.join(format!("distinct-on-split-{depth}")),
            sub_p,
            ctx.codec,
        )?;
        drain_repartition(
            store,
            i,
            sub_p,
            split_salt(depth + 1),
            ctx.keys_of,
            &mut sub,
        )?;
        for j in 0..sub_p {
            distinct_bucket(&mut sub, j, ctx, depth + 1, out)?;
        }
        return Ok(());
    }
    let bucket = store.read(i)?;
    if bucket.is_empty() {
        return Ok(());
    }
    let combined = ops::materialize(&bucket)?;
    out.push(agg::distinct_on(&combined, ctx.key_indices, ctx.order)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, ArrayRef, Int64Array, StringArray};
    use arrow::compute::SortOptions;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::collections::HashMap;
    use std::sync::Arc;

    const SQL_ASC: SortOptions = SortOptions {
        descending: false,
        nulls_first: false,
    };

    /// A per-process scratch directory, the convention `window_spill`'s tests already use —
    /// the store removes its own subdirectory on drop, so nothing outlives the test.
    fn scratch(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("{tag}_{}", std::process::id()))
    }

    fn batch(k: &[i64], ts: &[i64], v: &[&str]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("ts", DataType::Int64, true),
            Field::new("v", DataType::Utf8, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(k.to_vec())) as ArrayRef,
                Arc::new(Int64Array::from(ts.to_vec())) as ArrayRef,
                Arc::new(StringArray::from(v.to_vec())) as ArrayRef,
            ],
        )
        .unwrap()
    }

    fn rows(batches: &[RecordBatch]) -> HashMap<i64, (i64, String)> {
        let mut out = HashMap::new();
        for b in batches {
            let k = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let ts = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
            let v = b.column(2).as_any().downcast_ref::<StringArray>().unwrap();
            for i in 0..b.num_rows() {
                let prev = out.insert(k.value(i), (ts.value(i), v.value(i).to_string()));
                assert!(prev.is_none(), "a key survived twice: {}", k.value(i));
            }
        }
        out
    }

    /// The spilling reduction returns exactly what the in-memory one does, at a budget small
    /// enough to force many buckets and at least one re-split. The value asserted is the whole
    /// surviving *row*, so a bucket boundary that separated a key's rows would show up as a
    /// wrong payload rather than merely a wrong count.
    #[test]
    fn spilled_reduction_equals_the_in_memory_one() {
        let mut s: u64 = 31;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 33) as i64
        };
        let n = 20_000usize;
        let ks: Vec<i64> = (0..n).map(|_| next() % 211).collect();
        let tss: Vec<i64> = (0..n).map(|_| next() % 1_000_003).collect();
        let vs: Vec<String> = (0..n).map(|i| format!("payload-{i}")).collect();
        let refs: Vec<&str> = vs.iter().map(String::as_str).collect();
        let morsels: Vec<RecordBatch> = (0..n)
            .step_by(1024)
            .map(|start| {
                let end = (start + 1024).min(n);
                batch(&ks[start..end], &tss[start..end], &refs[start..end])
            })
            .collect();

        let whole = ops::materialize(&morsels).unwrap();
        let want = rows(&[agg::distinct_on(&whole, &[0], &[(1, SQL_ASC)]).unwrap()]);

        let dir = scratch("bc_dedupspill_ordered");
        // A budget far under the input forces a wide fan-out and a re-split.
        let (got, spilled) = distinct_on_spilling(
            &morsels,
            &[0],
            &[(1, SQL_ASC)],
            16 * 1024,
            &dir,
            SpillCodec::None,
        )
        .unwrap();
        assert!(spilled > 0, "nothing was routed to disk");
        assert_eq!(rows(&got), want, "spilled result differs from in-memory");
    }

    /// With no ordering the spilling path still returns one row per key, and every returned row
    /// is one the input actually held.
    #[test]
    fn unordered_spill_keeps_one_real_row_per_key() {
        let ks: Vec<i64> = (0..4_000).map(|i| i % 97).collect();
        let tss: Vec<i64> = (0..4_000).map(|i| i as i64 * 3).collect();
        let vs: Vec<String> = (0..4_000).map(|i| format!("v{i}")).collect();
        let refs: Vec<&str> = vs.iter().map(String::as_str).collect();
        let morsels: Vec<RecordBatch> = (0..4_000usize)
            .step_by(512)
            .map(|s| {
                let e = (s + 512).min(4_000);
                batch(&ks[s..e], &tss[s..e], &refs[s..e])
            })
            .collect();
        let real: std::collections::HashSet<(i64, i64, String)> =
            (0..4_000).map(|i| (ks[i], tss[i], vs[i].clone())).collect();

        let dir = scratch("bc_dedupspill_unordered");
        let (got, _) =
            distinct_on_spilling(&morsels, &[0], &[], 8 * 1024, &dir, SpillCodec::None).unwrap();
        let got = rows(&got);
        assert_eq!(got.len(), 97);
        for (k, (ts, v)) in got {
            assert!(real.contains(&(k, ts, v)), "row {k} was never in the input");
        }
    }
}
