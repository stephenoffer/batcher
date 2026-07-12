//! Late-materialized fused join + top-N.
//!
//! `… JOIN … ORDER BY <cols> LIMIT k` over a wide `SELECT *` otherwise gathers the join's
//! **whole payload** for every matched row (14 GB for a 60 M-row TPC-H `lineitem ⋈ orders`)
//! only for the top-N to keep `k` of it. This fuses the two: the join emits only the
//! **sort-key columns** plus a `(bucket, left-row, right-row)` locator per matched row, the
//! top-N runs on that narrow relation, and the wide payload is gathered **once**, for just the
//! `k` survivors, straight from the co-partitioned source buckets via `interleave`.
//!
//! Result-identical to gathering the full join then topping it: [`super::parallel_top_n`] is
//! morselization-independent (its `(key, candidate-position)` order reproduces the global
//! `(key, input-position)` total order), and the narrow relation carries the same per-bucket
//! row order the wide join produces — so it selects the same rows, and the locator gather
//! reproduces them exactly.
//!
//! Scoped to the case that pays and is simple to prove: an **inner** hash join whose ORDER BY
//! keys are **bare columns** of the join output. The caller falls back to the ordinary
//! join-then-top-N for anything else (outer joins, computed sort keys, sort-merge, broadcast,
//! or a spilling build).

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, RecordBatch, UInt32Array};
use arrow::compute::interleave;
use arrow::datatypes::{Field, Schema};
use bc_expr::Expr;
use bc_ir::{JoinOutputCol, JoinSide, SortKey};
use bc_runtime::join::{self, JoinType as RtJoinType};
use rayon::prelude::*;

use crate::error::InterpError;

// Locator column names in the narrow relation. Prefixed to not collide with a user alias.
const BUCKET_COL: &str = "__jtn_bucket";
const LROW_COL: &str = "__jtn_lrow";
const RROW_COL: &str = "__jtn_rrow";

/// The output columns the ORDER BY references, or `None` if any sort key is not a bare column
/// of the join output (then the caller keeps the ordinary path). Preserves `output` order.
fn key_output_cols<'a>(
    output: &'a [JoinOutputCol],
    sort_keys: &[SortKey],
) -> Option<Vec<&'a JoinOutputCol>> {
    let mut names = Vec::with_capacity(sort_keys.len());
    for key in sort_keys {
        match &key.expr {
            Expr::Col { name } => names.push(name.as_str()),
            _ => return None,
        }
    }
    let picked: Vec<&JoinOutputCol> = output
        .iter()
        .filter(|c| names.contains(&c.alias.as_str()))
        .collect();
    // Every referenced key must resolve to an output column.
    names
        .iter()
        .all(|n| output.iter().any(|c| c.alias == *n))
        .then_some(picked)
}

/// Fused co-partitioned inner-hash join + top-N, or `None` to fall back to join-then-top-N.
///
/// `p` is the bucket count the ordinary join uses; `left_batches`/`right_batches` are the
/// already-executed (uncopartitioned) inputs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn join_top_n(
    left_batches: &[RecordBatch],
    right_batches: &[RecordBatch],
    left_keys: &[String],
    right_keys: &[String],
    output: &[JoinOutputCol],
    sort_keys: &[SortKey],
    k: usize,
    p: usize,
    tuning: &bc_arrow::RuntimeTuning,
) -> Result<Option<Vec<RecordBatch>>, InterpError> {
    let Some(key_cols) = key_output_cols(output, sort_keys) else {
        return Ok(None);
    };
    // Co-partition both sides exactly as the ordinary join (`ops::partition_morsels`), so equal
    // keys share a bucket and the per-bucket row order matches.
    let rb = super::partition_morsels(right_batches, right_keys, p)?;
    let lb = super::partition_morsels(left_batches, left_keys, p)?;

    // Per bucket (parallel): inner-join indices, then the narrow relation — only the sort-key
    // columns (gathered at the indices) plus the (bucket, left-row, right-row) locator.
    let narrow: Vec<RecordBatch> = (0..p)
        .into_par_iter()
        .map(|i| -> Result<RecordBatch, InterpError> {
            let left_key_cols = super::columns_by_name(&lb[i], left_keys)?;
            let right_key_cols = super::columns_by_name(&rb[i], right_keys)?;
            let idx = join::hash_join_indices_with(
                &left_key_cols,
                &right_key_cols,
                RtJoinType::Inner,
                tuning.bloom_fp_rate,
                tuning.bloom_min_build_rows,
            )?;
            let n = idx.left.len();
            let mut fields: Vec<Field> = Vec::with_capacity(key_cols.len() + 3);
            let mut columns: Vec<ArrayRef> = Vec::with_capacity(key_cols.len() + 3);
            for col in &key_cols {
                let (batch, indices) = match col.side {
                    JoinSide::Left => (&lb[i], &idx.left),
                    JoinSide::Right => (&rb[i], &idx.right),
                };
                let source = batch
                    .column_by_name(&col.name)
                    .ok_or_else(|| InterpError::UnknownJoinColumn(col.name.clone()))?;
                let gathered = bc_runtime::gather::take_column(source.as_ref(), indices)?;
                fields.push(Field::new(&col.alias, gathered.data_type().clone(), true));
                columns.push(gathered);
            }
            // Locators (inner join → no null indices).
            let bucket = Arc::new(UInt32Array::from(vec![i as u32; n])) as ArrayRef;
            fields.push(Field::new(BUCKET_COL, bucket.data_type().clone(), false));
            columns.push(bucket);
            fields.push(Field::new(LROW_COL, idx.left.data_type().clone(), false));
            columns.push(Arc::new(idx.left) as ArrayRef);
            fields.push(Field::new(RROW_COL, idx.right.data_type().clone(), false));
            columns.push(Arc::new(idx.right) as ArrayRef);
            Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?)
        })
        .collect::<Result<Vec<_>, _>>()?;

    if narrow.iter().all(|b| b.num_rows() == 0) {
        // No matches: an empty result with the join's output schema.
        return Ok(Some(vec![empty_output(&lb, &rb, output)?]));
    }

    // Top-N over the narrow relation (keys + locators). Identical selection to topping the
    // full join, because `parallel_top_n` is morselization-independent.
    let narrow: Vec<RecordBatch> = narrow.into_iter().filter(|b| b.num_rows() > 0).collect();
    let topk = super::parallel_top_n(&narrow, sort_keys, k)?;

    // Late-materialize: gather the wide payload for just the k survivors, in sort order, via
    // `interleave` across the source buckets.
    let bucket = u32_col(&topk, BUCKET_COL)?;
    let lrow = u32_col(&topk, LROW_COL)?;
    let rrow = u32_col(&topk, RROW_COL)?;
    let out = gather_survivors(&lb, &rb, output, bucket, lrow, rrow)?;
    Ok(Some(vec![out]))
}

/// Gather the full join output for the surviving `(bucket, left-row, right-row)` locators.
fn gather_survivors(
    lb: &[RecordBatch],
    rb: &[RecordBatch],
    output: &[JoinOutputCol],
    bucket: &UInt32Array,
    lrow: &UInt32Array,
    rrow: &UInt32Array,
) -> Result<RecordBatch, InterpError> {
    let n = bucket.len();
    let left_pairs: Vec<(usize, usize)> = (0..n)
        .map(|s| (bucket.value(s) as usize, lrow.value(s) as usize))
        .collect();
    let right_pairs: Vec<(usize, usize)> = (0..n)
        .map(|s| (bucket.value(s) as usize, rrow.value(s) as usize))
        .collect();
    let mut fields = Vec::with_capacity(output.len());
    let mut columns = Vec::with_capacity(output.len());
    for col in output {
        let (buckets, pairs) = match col.side {
            JoinSide::Left => (lb, &left_pairs),
            JoinSide::Right => (rb, &right_pairs),
        };
        let arrays: Vec<&dyn Array> = buckets
            .iter()
            .map(|b| -> Result<&dyn Array, InterpError> {
                Ok(b.column_by_name(&col.name)
                    .ok_or_else(|| InterpError::UnknownJoinColumn(col.name.clone()))?
                    .as_ref())
            })
            .collect::<Result<_, _>>()?;
        let gathered = interleave(&arrays, pairs)?;
        fields.push(Field::new(&col.alias, gathered.data_type().clone(), true));
        columns.push(gathered);
    }
    Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?)
}

/// An empty batch with the join's output schema (types taken from the source buckets).
fn empty_output(
    lb: &[RecordBatch],
    rb: &[RecordBatch],
    output: &[JoinOutputCol],
) -> Result<RecordBatch, InterpError> {
    let mut fields = Vec::with_capacity(output.len());
    let mut columns = Vec::with_capacity(output.len());
    for col in output {
        let batches = match col.side {
            JoinSide::Left => lb,
            JoinSide::Right => rb,
        };
        let dt = batches
            .first()
            .and_then(|b| b.column_by_name(&col.name))
            .map(|c| c.data_type().clone())
            .unwrap_or(arrow::datatypes::DataType::Null);
        fields.push(Field::new(&col.alias, dt.clone(), true));
        columns.push(arrow::array::new_empty_array(&dt));
    }
    Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?)
}

fn u32_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a UInt32Array, InterpError> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
        .ok_or_else(|| InterpError::UnknownJoinColumn(name.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use arrow::datatypes::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use bc_ir::{JoinStrategy, JoinType};

    fn int_batch(name: &str, vals: Vec<i64>) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(name, DataType::Int64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vals)) as ArrayRef]).unwrap()
    }

    fn two_col(n0: &str, v0: Vec<i64>, n1: &str, v1: Vec<i64>) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new(n0, DataType::Int64, false),
            ArrowField::new(n1, DataType::Int64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(v0)) as ArrayRef,
                Arc::new(Int64Array::from(v1)) as ArrayRef,
            ],
        )
        .unwrap()
    }

    fn i64s(b: &RecordBatch, name: &str) -> Vec<i64> {
        let c = b.column_by_name(name).unwrap();
        let a = c.as_any().downcast_ref::<Int64Array>().unwrap();
        (0..a.len()).map(|i| a.value(i)).collect()
    }

    /// The fused join+top-N must match the eager parallel path (per-bucket full join, then
    /// `parallel_top_n`) row-for-row — same survivors, same order, every column — including on
    /// heavy sort-key ties, across bucket counts and `k`.
    #[test]
    fn join_top_n_matches_eager_parallel() {
        // Left: key `lk`, sort value `sv` (heavy ties), payload `lp`.
        let nl = 6000i64;
        let left = two_col("lk", (0..nl).map(|i| i % 300).collect(), "sv", (0..nl).map(|i| (i * 7) % 40).collect());
        let left = {
            // add payload column lp = i
            let mut cols: Vec<ArrayRef> = left.columns().to_vec();
            let mut fields: Vec<ArrowField> = left.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
            cols.push(Arc::new(Int64Array::from((0..nl).collect::<Vec<_>>())) as ArrayRef);
            fields.push(ArrowField::new("lp", DataType::Int64, false));
            RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), cols).unwrap()
        };
        // Right: key `rk` (dense so most left rows match), payload `rp`.
        let nr = 300i64;
        let right = two_col("rk", (0..nr).collect(), "rp", (0..nr).map(|i| i * 1000).collect());

        let left_keys = vec!["lk".to_string()];
        let right_keys = vec!["rk".to_string()];
        let output = vec![
            JoinOutputCol { side: JoinSide::Left, name: "sv".into(), alias: "sv".into() },
            JoinOutputCol { side: JoinSide::Left, name: "lp".into(), alias: "lp".into() },
            JoinOutputCol { side: JoinSide::Right, name: "rp".into(), alias: "rp".into() },
        ];
        let sort_keys = vec![SortKey { expr: Expr::Col { name: "sv".into() }, descending: true, nulls_first: false }];
        let tuning = bc_arrow::RuntimeTuning::default();

        // Split inputs into morsels to exercise multi-batch bucketing.
        let l_morsels: Vec<RecordBatch> = (0..nl as usize).step_by(700).map(|s| left.slice(s, (700).min(nl as usize - s))).collect();
        let r_morsels = vec![right.clone()];

        for p in [1usize, 3, 8] {
            for k in [3usize, 50, 500, 10_000] {
                // Eager parallel reference: per-bucket full join, then parallel_top_n.
                let rb = super::super::partition_morsels(&r_morsels, &right_keys, p).unwrap();
                let lb = super::super::partition_morsels(&l_morsels, &left_keys, p).unwrap();
                let full: Vec<RecordBatch> = (0..p)
                    .map(|i| super::super::join_batches(&lb[i], &rb[i], &left_keys, &right_keys, JoinType::Inner, &output, JoinStrategy::Hash).unwrap())
                    .filter(|b| b.num_rows() > 0)
                    .collect();
                let eager = super::super::parallel_top_n(&full, &sort_keys, k).unwrap();

                let fused = join_top_n(&l_morsels, &r_morsels, &left_keys, &right_keys, &output, &sort_keys, k, p, &tuning)
                    .unwrap()
                    .expect("bare-column sort key should fuse");
                let fused = &fused[0];

                assert_eq!(fused.num_rows(), eager.num_rows(), "row count p={p} k={k}");
                for col in ["sv", "lp", "rp"] {
                    assert_eq!(i64s(fused, col), i64s(&eager, col), "col {col} p={p} k={k}");
                }
            }
        }
    }

    /// A computed (non-bare-column) sort key declines the fused path (caller falls back).
    #[test]
    fn computed_sort_key_declines() {
        let left = int_batch("lk", vec![1, 2, 3]);
        let right = int_batch("rk", vec![1, 2, 3]);
        let output = vec![JoinOutputCol { side: JoinSide::Left, name: "lk".into(), alias: "lk".into() }];
        let sort_keys = vec![SortKey {
            expr: Expr::Binary {
                op: bc_expr::BinaryOp::Add,
                left: Box::new(Expr::Col { name: "lk".into() }),
                right: Box::new(Expr::Lit { value: bc_expr::Literal::Int(1) }),
            },
            descending: false,
            nulls_first: false,
        }];
        let got = join_top_n(
            std::slice::from_ref(&left),
            std::slice::from_ref(&right),
            &["lk".to_string()],
            &["rk".to_string()],
            &output,
            &sort_keys,
            2,
            1,
            &bc_arrow::RuntimeTuning::default(),
        )
        .unwrap();
        assert!(got.is_none(), "computed sort key must decline to fuse");
    }
}
