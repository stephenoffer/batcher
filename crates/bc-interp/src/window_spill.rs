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
use bc_runtime::agg::spill::{DiskSpillStore, SpillCodec, SpillStore};
use bc_runtime::shuffle;

use crate::batch_bytes;
use crate::error::InterpError;
use crate::ops;
use crate::spill_split::{
    drain_repartition, grace_bucket_count, split_salt, MAX_GRACE_SPLIT_DEPTH,
};

/// Run a window operator under a memory envelope by grace-partitioning on the
/// `PARTITION BY` keys. Caller guarantees `partition_keys` is non-empty (a single
/// global partition cannot be split for ranking/running aggregates).
#[allow(clippy::too_many_arguments)] // a window kernel legitimately needs all of these
pub(crate) fn window_spilling(
    parts: &[RecordBatch],
    partition_keys: &[bc_expr::Expr],
    order_keys: &[bc_ir::SortKey],
    functions: &[bc_ir::WindowFunc],
    rank_limit: Option<usize>,
    budget_bytes: usize,
    dir: &Path,
    codec: SpillCodec,
) -> Result<(Vec<RecordBatch>, u64), InterpError> {
    let p = grace_bucket_count(batch_bytes(parts) as usize, budget_bytes);
    let mut store = DiskSpillStore::with_codec(dir.join("window"), p, codec)?;
    let keys_of = |batch: &RecordBatch| -> Result<Vec<ArrayRef>, InterpError> {
        partition_keys
            .iter()
            .map(|e| e.eval(batch).map_err(InterpError::from))
            .collect()
    };
    for batch in parts {
        for (i, bucket) in shuffle::partition_by_key_arrays(batch, &keys_of(batch)?, p)?
            .iter()
            .enumerate()
        {
            if bucket.num_rows() > 0 {
                store.append(i, bucket)?;
            }
        }
    }
    let spill_bytes = store.spilled_bytes(); // measured volume routed to disk
    let ctx = WindowBuckets {
        partition_keys,
        order_keys,
        functions,
        rank_limit,
        budget_bytes,
        codec,
        dir,
        keys_of: &keys_of,
    };
    let mut out = Vec::with_capacity(p);
    for i in 0..p {
        window_bucket(&mut store, i, &ctx, 0, &mut out)?;
    }
    Ok((out, spill_bytes))
}

/// The parts of a spilling window that do not change as buckets are re-split.
struct WindowBuckets<'a> {
    partition_keys: &'a [bc_expr::Expr],
    order_keys: &'a [bc_ir::SortKey],
    functions: &'a [bc_ir::WindowFunc],
    rank_limit: Option<usize>,
    budget_bytes: usize,
    codec: SpillCodec,
    dir: &'a Path,
    keys_of: &'a dyn Fn(&RecordBatch) -> Result<Vec<ArrayRef>, InterpError>,
}

/// Run the window kernel over bucket `i`, re-splitting it first if it does not fit.
///
/// The bucket count is sized from the input's *average* bytes per bucket, so a skewed
/// `PARTITION BY` leaves one bucket far over the envelope — and the kernel needs its bucket
/// materialized, so that bucket OOMs at exactly the point spilling was meant to prevent it.
/// Asking the store for the bucket's size *before* reading it is what allows the split to
/// happen without first pulling in the thing that does not fit.
///
/// A re-split re-partitions by the **same** `PARTITION BY` keys under a depth-derived salt,
/// so every window partition still lands whole in one sub-bucket. That is the correctness
/// condition here and it is stronger than the join's: a window partition split across two
/// buckets would produce two independent rankings, not merely a slower join.
///
/// It also means the split cannot help a *single* hot key — its rows re-hash together
/// however they are salted. The depth bound is what makes that terminate; past it the bucket
/// is run as-is. What the split does fix is the common case the fan-out cap creates: a
/// bucket holding many distinct partitions, too large only in aggregate.
fn window_bucket(
    store: &mut dyn SpillStore,
    i: usize,
    ctx: &WindowBuckets<'_>,
    depth: u32,
    out: &mut Vec<RecordBatch>,
) -> Result<(), InterpError> {
    let bytes = store.partition_bytes(i) as usize;
    if depth < MAX_GRACE_SPLIT_DEPTH && bytes > ctx.budget_bytes {
        let sub_p = grace_bucket_count(bytes, ctx.budget_bytes);
        let mut sub = DiskSpillStore::with_codec(
            ctx.dir.join(format!("window-split-{depth}")),
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
            window_bucket(&mut sub, j, ctx, depth + 1, out)?;
        }
        return Ok(());
    }
    let bucket = store.read(i)?;
    if bucket.is_empty() {
        return Ok(());
    }
    let combined = ops::materialize(&bucket)?;
    out.push(ops::window_batch(
        &combined,
        ctx.partition_keys,
        ctx.order_keys,
        ctx.functions,
        ctx.rank_limit,
    )?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use bc_ir::{SortKey, WindowFn, WindowFunc};
    use std::sync::Arc;

    fn fbatch(ks: &[f64], vs: &[i64]) -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("k", DataType::Float64, true),
            Field::new("v", DataType::Int64, true),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Float64Array::from(ks.to_vec())),
                Arc::new(Int64Array::from(vs.to_vec())),
            ],
        )
        .unwrap()
    }

    /// Canonical multiset of `(k-bits, rn, s)` rows (window output is an unordered
    /// relation, so compare as a multiset).
    fn rows(batches: &[RecordBatch]) -> Vec<(u64, i64, i64)> {
        let mut out = Vec::new();
        for b in batches {
            let k = b.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
            let rn = b.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
            let s = b.column(3).as_any().downcast_ref::<Int64Array>().unwrap();
            for i in 0..b.num_rows() {
                out.push((k.value(i).to_bits(), rn.value(i), s.value(i)));
            }
        }
        out.sort();
        out
    }

    /// A spilling window whose buckets are wildly uneven must still equal the in-memory
    /// kernel, and the re-split must keep every window partition whole.
    ///
    /// The bucket count is sized from the input's *average* bytes per bucket, so a skewed
    /// `PARTITION BY` leaves one bucket far over the envelope — and the kernel materializes
    /// its bucket, so that is an OOM at exactly the point spilling was meant to prevent one.
    /// Re-splitting is the fix, and it is more delicate here than for a join: a window
    /// partition landing in two sub-buckets would produce two independent rankings rather
    /// than merely a slower operator. Salting the *same* partition keys is what prevents
    /// that, and `row_number` is the assertion, because it is the function that cannot
    /// survive a split partition.
    #[test]
    fn window_spill_skewed_partitions_match_in_memory() {
        // One hot key carrying most rows, plus a spread of cold ones — so the buckets are
        // uneven, some sub-buckets hold several distinct partitions, and the hot key exercises
        // the case no salt can separate.
        let mut ks: Vec<f64> = Vec::new();
        let mut vs: Vec<i64> = Vec::new();
        for i in 0..400 {
            ks.push(7.0);
            vs.push(i);
        }
        for k in 0..60 {
            for r in 0..3 {
                ks.push(k as f64);
                vs.push(1000 + k * 10 + r);
            }
        }
        // Split into several morsels so a partition's rows arrive spread across runs.
        let parts: Vec<RecordBatch> = ks
            .chunks(64)
            .zip(vs.chunks(64))
            .map(|(k, v)| fbatch(k, v))
            .collect();
        let whole = crate::ops::materialize(&parts).unwrap();

        let pk = vec![bc_expr::Expr::Col { name: "k".into() }];
        let ok = vec![SortKey {
            expr: bc_expr::Expr::Col { name: "v".into() },
            descending: false,
            nulls_first: false,
        }];
        let funcs = vec![
            WindowFunc {
                func: WindowFn::RowNumber,
                input: None,
                offset: 1,
                frame: None,
                alpha: None,
                half_life: None,
                alias: "rn".into(),
            },
            WindowFunc {
                func: WindowFn::Sum,
                input: Some(bc_expr::Expr::Col { name: "v".into() }),
                offset: 1,
                frame: None,
                alpha: None,
                half_life: None,
                alias: "s".into(),
            },
        ];

        let oracle = crate::ops::window_batch(&whole, &pk, &ok, &funcs, None).unwrap();
        let dir = std::env::temp_dir().join(format!("bc_winspill_skew_{}", std::process::id()));
        // Small but not degenerate: the cold buckets fit and the hot one does not, so the
        // split is a real decision taken on a measured size.
        let (spilled, _) =
            window_spilling(&parts, &pk, &ok, &funcs, None, 2048, &dir, SpillCodec::None).unwrap();

        assert_eq!(
            rows(std::slice::from_ref(&oracle)),
            rows(&spilled),
            "a re-split window bucket diverged from the in-memory kernel — a partition was \
             split across sub-buckets and ranked twice"
        );
    }

    /// A `PARTITION BY <float>` window over a key holding `-0.0`, `0.0` and NaN must give
    /// the same relation spilled (grace-partitioned) as in memory: the shuffle folds the
    /// two zeros / all NaNs into one bucket exactly as the in-memory partition identity
    /// treats them, so no partition is split across buckets under memory pressure.
    #[test]
    fn window_spill_signed_zero_nan_partition_matches_in_memory() {
        let nan = f64::NAN;
        let ks = [-0.0, 0.0, nan, 1.0, -0.0, nan, 0.0, 1.0, nan];
        let vs = [10, 20, 30, 40, 50, 60, 70, 80, 90];
        // Split across morsels so the two zeros / NaNs arrive in different runs.
        let parts = vec![fbatch(&ks[0..4], &vs[0..4]), fbatch(&ks[4..9], &vs[4..9])];
        let whole = crate::ops::materialize(&parts).unwrap();

        let pk = vec![bc_expr::Expr::Col { name: "k".into() }];
        let ok = vec![SortKey {
            expr: bc_expr::Expr::Col { name: "v".into() },
            descending: false,
            nulls_first: false,
        }];
        let funcs = vec![
            WindowFunc {
                func: WindowFn::RowNumber,
                input: None,
                offset: 1,
                frame: None,
                alpha: None,
                half_life: None,
                alias: "rn".into(),
            },
            WindowFunc {
                func: WindowFn::Sum,
                input: Some(bc_expr::Expr::Col { name: "v".into() }),
                offset: 1,
                frame: None,
                alpha: None,
                half_life: None,
                alias: "s".into(),
            },
        ];

        let oracle = crate::ops::window_batch(&whole, &pk, &ok, &funcs, None).unwrap();
        let dir = std::env::temp_dir().join(format!("bc_winspill_negz_{}", std::process::id()));
        // budget 1 → many grace buckets, forcing the two zeros / NaNs to co-partition.
        let (spilled, _) =
            window_spilling(&parts, &pk, &ok, &funcs, None, 1, &dir, SpillCodec::None).unwrap();

        assert_eq!(rows(std::slice::from_ref(&oracle)), rows(&spilled));
    }
}
