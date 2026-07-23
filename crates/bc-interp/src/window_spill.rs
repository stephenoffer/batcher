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
    let p = (batch_bytes(parts) as usize)
        .div_ceil(budget_bytes.max(1))
        .max(2);
    let mut store = DiskSpillStore::with_codec(dir.join("window"), p, codec)?;
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
    let spill_bytes = store.spilled_bytes(); // measured volume routed to disk
    let mut out = Vec::with_capacity(p);
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
    Ok((out, spill_bytes))
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
                alias: "rn".into(),
            },
            WindowFunc {
                func: WindowFn::Sum,
                input: Some(bc_expr::Expr::Col { name: "v".into() }),
                offset: 1,
                frame: None,
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
