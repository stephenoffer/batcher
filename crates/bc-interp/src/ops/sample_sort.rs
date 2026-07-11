//! Single-node parallel full sort by **sample-sort**.
//!
//! Range-partition the rows by the leading key (sampled quantile boundaries) and sort each
//! range in parallel. The ranges are globally ordered relative to each other, so the sorted
//! relation is simply the ranges in key order — no final merge, and no concat: the executor
//! consumes a `Vec<RecordBatch>` already.
//!
//! **The payload is gathered exactly once.** Routing produces per-range *row indices*, not
//! range batches; each range sorts a cheap gather of just its key columns, then composes
//! that permutation with its row indices and gathers every column once. Materializing the
//! ranges up front (and again to sort them, and a third time to concatenate) copied every
//! column three times — on a 5 M-row, 6-column sort that was two thirds of the work.
//!
//! This is the single-node form of the distributed range sort (`dist/flight_sort.py`), so
//! one implementation serves both: the boundaries and the routing come from the same
//! `bc_runtime::shuffle` range partitioners.

use arrow::array::{
    Array, ArrayRef, GenericStringArray, OffsetSizeTrait, RecordBatch, UInt32Array,
};
use arrow::compute::take;
use arrow::datatypes::DataType;
use bc_ir::SortKey;
use rayon::prelude::*;

use crate::error::InterpError;

/// Rows below which the single-node sample-sort stays serial — the sampling + range
/// partition overhead only pays off on a large full sort.
const PARALLEL_SORT_MIN_ROWS: usize = 1 << 17;

/// Rows sampled to estimate the quantile boundaries. Enough to balance 64 ranges well
/// while staying a negligible fraction of a large sort.
const SAMPLE_TARGET: usize = 8192;

/// Parallel single-node full sort by sample-sort.
///
/// Returns `None` (caller uses the serial `sort_batch`) unless it applies: a full sort (no
/// `LIMIT` — top-N is already cheap), a large input, and a **float, integer, or string**
/// leading key (the boundaries route it exactly — floats by `f64`, integers by `i64`,
/// strings lexicographically by bytes). Multi-key sorts are supported: rows bucket by the
/// leading key (equal leading keys never span a boundary), then each range sorts by the
/// full key list, so a plain concatenation in leading-key order is the globally sorted
/// multi-key relation.
pub(crate) fn parallel_sort_batch(
    batch: &RecordBatch,
    keys: &[SortKey],
    limit: Option<usize>,
) -> Result<Option<Vec<RecordBatch>>, InterpError> {
    let Some(k0) = keys.first() else {
        return Ok(None);
    };
    if limit.is_some() || batch.num_rows() < PARALLEL_SORT_MIN_ROWS {
        return Ok(None);
    }
    // Evaluate every sort key once over the whole batch: the per-range sorts reuse these
    // arrays, so a computed `ORDER BY` expression is evaluated once, not once per range.
    let key_arrays: Vec<ArrayRef> = keys
        .iter()
        .map(|k| k.expr.eval(batch))
        .collect::<Result<_, _>>()?;
    let key = &key_arrays[0];
    // 64 ranges saturate the gather: measured 32/64/96/128/192 ranges on 96 cores at
    // 157/123/124/120/116 ms — past 64 the sort is memory-bandwidth bound, not
    // parallelism bound, so more ranges only add sampling and concat overhead.
    let parts = rayon::current_num_threads().clamp(2, 64);

    // Route each row to a range, as *indices only*. Gathering the payload into range
    // batches here (and again to sort each one, and a third time to concatenate) copies
    // every column three times; composing the range's indices with its sort permutation
    // gathers exactly once.
    let part_of = match key.data_type() {
        DataType::Float64 | DataType::Float32 => {
            let key_f64 = arrow::compute::cast(key, &DataType::Float64)?;
            let keyv = key_f64
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .expect("cast to Float64");
            let Some(b) = sample_boundaries_f64(keyv, parts) else {
                return Ok(None);
            };
            bc_runtime::shuffle::range_part_of_f64(
                &key_f64,
                &b,
                parts,
                k0.nulls_first,
                k0.descending,
            )?
        }
        DataType::Utf8 => {
            let a = key
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .expect("Utf8 downcast");
            let Some(b) = sample_boundaries_str(a, parts) else {
                return Ok(None);
            };
            bc_runtime::shuffle::range_part_of_str(key, &b, parts, k0.nulls_first, k0.descending)?
        }
        DataType::LargeUtf8 => {
            let a = key
                .as_any()
                .downcast_ref::<arrow::array::LargeStringArray>()
                .expect("LargeUtf8 downcast");
            let Some(b) = sample_boundaries_str(a, parts) else {
                return Ok(None);
            };
            bc_runtime::shuffle::range_part_of_str(key, &b, parts, k0.nulls_first, k0.descending)?
        }
        dt if dt.is_integer() => {
            let key_i64 = arrow::compute::cast(key, &DataType::Int64)?;
            let keyv = key_i64
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .expect("cast to Int64");
            let Some(b) = sample_boundaries_i64(keyv, parts) else {
                return Ok(None);
            };
            bc_runtime::shuffle::range_part_of_i64(
                &key_i64,
                &b,
                parts,
                k0.nulls_first,
                k0.descending,
            )?
        }
        _ => return Ok(None),
    };
    let buckets = bc_runtime::shuffle::bucket_indices(&part_of, parts);

    // Each range sorts independently: gather only its *key* columns (one or two narrow
    // arrays), sort those, then map the range-local permutation back through the range's
    // row indices and gather the payload once.
    let mut sorted: Vec<RecordBatch> = buckets
        .par_iter()
        .map(|idx| -> Result<RecordBatch, InterpError> {
            let take_idx = UInt32Array::from(idx.clone());
            let range_keys: Vec<ArrayRef> = key_arrays
                .iter()
                .map(|a| take(a.as_ref(), &take_idx, None))
                .collect::<Result<_, _>>()?;
            let local = super::sort_indices_of(&range_keys, keys)?;
            let global: Vec<u32> = local.values().iter().map(|&l| idx[l as usize]).collect();
            Ok(bc_runtime::shuffle::gather_rows(batch, &global)?)
        })
        .collect::<Result<_, InterpError>>()?;

    // Ranges are globally ordered relative to each other, so the sorted relation is simply
    // the ranges in key order.
    if k0.descending {
        sorted.reverse();
    }
    Ok(Some(sorted))
}

/// Sample `parts-1` ascending f64 quantile boundaries from a float key column. Returns
/// `None` if fewer than `parts` finite values exist (nothing meaningful to split).
fn sample_boundaries_f64(key: &arrow::array::Float64Array, parts: usize) -> Option<Vec<f64>> {
    let n = key.len();
    let target = SAMPLE_TARGET.min(n).max(parts);
    let stride = (n / target).max(1);
    let mut sample: Vec<f64> = (0..n)
        .step_by(stride)
        .filter(|&i| key.is_valid(i))
        .map(|i| key.value(i))
        .filter(|v| !v.is_nan())
        .collect();
    if sample.len() < parts {
        return None;
    }
    sample.sort_unstable_by(|a, b| a.total_cmp(b));
    let m = sample.len();
    Some(
        (1..parts)
            .map(|j| sample[(j * m / parts).min(m - 1)])
            .collect(),
    )
}

/// Sample `parts-1` ascending i64 quantile boundaries from an integer key column (the
/// exact-integer analog of [`sample_boundaries_f64`]). `None` if too few non-null values.
fn sample_boundaries_i64(key: &arrow::array::Int64Array, parts: usize) -> Option<Vec<i64>> {
    let n = key.len();
    let target = SAMPLE_TARGET.min(n).max(parts);
    let stride = (n / target).max(1);
    let mut sample: Vec<i64> = (0..n)
        .step_by(stride)
        .filter(|&i| key.is_valid(i))
        .map(|i| key.value(i))
        .collect();
    if sample.len() < parts {
        return None;
    }
    sample.sort_unstable();
    let m = sample.len();
    Some(
        (1..parts)
            .map(|j| sample[(j * m / parts).min(m - 1)])
            .collect(),
    )
}

/// Sample `parts-1` ascending string quantile boundaries, compared byte-lexicographically
/// (Rust's `str` `Ord`), which is exactly how arrow orders a `Utf8` column.
///
/// Returns `None` when there are too few non-null values to split, or when the sample is
/// so skewed that the boundaries collapse to a single distinct value — in that case every
/// row would route to one bucket and the sample-sort would be pure overhead, so the caller
/// falls back to the serial sort.
fn sample_boundaries_str<O: OffsetSizeTrait>(
    key: &GenericStringArray<O>,
    parts: usize,
) -> Option<Vec<String>> {
    let n = key.len();
    let target = SAMPLE_TARGET.min(n).max(parts);
    let stride = (n / target).max(1);
    let mut sample: Vec<&str> = (0..n)
        .step_by(stride)
        .filter(|&i| key.is_valid(i))
        .map(|i| key.value(i))
        .collect();
    if sample.len() < parts {
        return None;
    }
    sample.sort_unstable();
    let m = sample.len();
    let bounds: Vec<String> = (1..parts)
        .map(|j| sample[(j * m / parts).min(m - 1)].to_string())
        .collect();
    if bounds.first() == bounds.last() {
        return None;
    }
    Some(bounds)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, StringArray};
    use arrow::compute::concat_batches;
    use arrow::datatypes::{Field, Schema};
    use bc_expr::Expr;

    use super::super::sort_batch;
    use super::*;

    /// The sample-sort returns the ranges in key order; the sorted relation is their
    /// concatenation, which is what the serial oracle produces as one batch.
    fn concat_ranges(schema: &std::sync::Arc<Schema>, ranges: Vec<RecordBatch>) -> RecordBatch {
        concat_batches(schema, ranges.iter()).unwrap()
    }

    fn str_batch(vals: Vec<Option<&str>>, payload: Vec<i64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Utf8, true),
            Field::new("p", DataType::Int64, false),
        ]));
        let s: ArrayRef = Arc::new(StringArray::from(vals));
        let p: ArrayRef = Arc::new(Int64Array::from(payload));
        RecordBatch::try_new(schema, vec![s, p]).unwrap()
    }

    fn key(descending: bool, nulls_first: bool) -> Vec<SortKey> {
        vec![SortKey {
            expr: Expr::Col { name: "s".into() },
            descending,
            nulls_first,
        }]
    }

    /// The sample-sort must produce exactly what the serial `sort_batch` oracle produces.
    fn assert_matches_serial(batch: &RecordBatch, keys: &[SortKey]) {
        let want = sort_batch(batch, keys, None).unwrap();
        let ranges = parallel_sort_batch(batch, keys, None)
            .unwrap()
            .expect("sample-sort should engage");
        assert_eq!(want, concat_ranges(&batch.schema(), ranges));
    }

    fn big_str_batch(n: usize, nulls: bool) -> RecordBatch {
        let mut s: u64 = 99;
        let mut vals = Vec::with_capacity(n);
        let mut pay = Vec::with_capacity(n);
        for i in 0..n {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = (s >> 33) % 5000;
            if nulls && i % 97 == 0 {
                vals.push(None);
            } else {
                vals.push(Some(format!("str_{v:05}")));
            }
            pay.push(i as i64);
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Utf8, true),
            Field::new("p", DataType::Int64, false),
        ]));
        let sa: ArrayRef = Arc::new(StringArray::from(vals));
        let pa: ArrayRef = Arc::new(Int64Array::from(pay));
        RecordBatch::try_new(schema, vec![sa, pa]).unwrap()
    }

    #[test]
    fn string_sample_sort_matches_serial_ascending() {
        let b = big_str_batch(1 << 18, false);
        assert_matches_serial(&b, &key(false, false));
    }

    #[test]
    fn string_sample_sort_matches_serial_descending() {
        let b = big_str_batch(1 << 18, false);
        assert_matches_serial(&b, &key(true, false));
    }

    #[test]
    fn string_sample_sort_matches_serial_with_nulls() {
        let b = big_str_batch(1 << 18, true);
        assert_matches_serial(&b, &key(false, false));
        assert_matches_serial(&b, &key(false, true));
        assert_matches_serial(&b, &key(true, true));
    }

    #[test]
    fn string_sample_sort_is_stable_on_ties() {
        // One distinct-ish key repeated: equal keys must keep input order (payload
        // ascending), exactly as the stable serial sort does.
        let n = 1 << 18;
        let vals: Vec<Option<&str>> = (0..n)
            .map(|i| Some(if i % 2 == 0 { "aaa" } else { "bbb" }))
            .collect();
        let b = str_batch(vals, (0..n as i64).collect());
        // Two distinct values only: boundaries collapse is possible, so just assert the
        // result equals the serial oracle whichever path is taken.
        let want = sort_batch(&b, &key(false, false), None).unwrap();
        let got = match parallel_sort_batch(&b, &key(false, false), None).unwrap() {
            Some(ranges) => concat_ranges(&b.schema(), ranges),
            None => sort_batch(&b, &key(false, false), None).unwrap(),
        };
        assert_eq!(want, got);
    }

    #[test]
    fn small_input_declines_sample_sort() {
        let b = str_batch(vec![Some("b"), Some("a")], vec![1, 2]);
        assert!(parallel_sort_batch(&b, &key(false, false), None)
            .unwrap()
            .is_none());
    }

    #[test]
    fn single_distinct_key_declines_sample_sort() {
        let n = 1 << 18;
        let vals: Vec<Option<&str>> = (0..n).map(|_| Some("same")).collect();
        let b = str_batch(vals, (0..n as i64).collect());
        assert!(parallel_sort_batch(&b, &key(false, false), None)
            .unwrap()
            .is_none());
    }
}
