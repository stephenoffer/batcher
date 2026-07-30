//! Hash-partition a relation held as morsels, gathering each row exactly **once**.
//!
//! The shuffle join used to `materialize` its probe side into one batch and then
//! `partition_by_keys` it, which gathers every row again: two full copies of the query's
//! largest relation, back to back. `interleave` can gather from many source arrays at once,
//! so the concatenation is unnecessary — each bucket is built directly from the morsels.
//!
//! **The result is identical to partitioning the concatenated relation.** A row's bucket is a
//! deterministic function of its key value (`shuffle::bucket_of_rows`), so it lands in the
//! same bucket whichever morsel carries it; and the gather visits morsels in order and rows in
//! order within a morsel, so each bucket holds its rows in the relation's original order —
//! exactly what a single `scatter_into_buckets` over the concatenated batch produces. The
//! per-bucket join and the `seq == par` oracle both depend on that.
//!
//! The buckets stay **contiguous** — one `RecordBatch` each. That is the difference between
//! this and the obvious "partition each morsel independently" approach, which was tried and
//! reverted: it leaves each bucket holding one small piece per morsel (366 pieces of ~170 rows
//! at sf10), and the per-piece overhead of the downstream join swamps the copy it saved.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, ArrowPrimitiveType, PrimitiveArray, RecordBatch};
use arrow::compute::interleave;
use arrow::datatypes::{
    DataType, Date32Type, Date64Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type,
    Int8Type, TimeUnit, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType, TimestampSecondType, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
};
use bc_runtime::shuffle;
use rayon::prelude::*;

use crate::error::InterpError;
use crate::ops::columns_by_name;

/// Hash-partition `batches` into `parts` buckets by `keys`, one contiguous batch per bucket.
///
/// A single bucket needs no gather at all — the relation is concatenated as-is, which is what
/// the caller would have done anyway.
pub(crate) fn partition_morsels(
    batches: &[RecordBatch],
    keys: &[String],
    parts: usize,
) -> Result<Vec<RecordBatch>, InterpError> {
    partition_morsels_with(batches, parts, |b| columns_by_name(b, keys), 0)
}

/// [`partition_morsels`] keyed by column *index* — the form the distributed shuffle
/// speaks (`dist::partition_batches` receives key indices, not names).
/// Hash-partition morsels by the key columns at `key_indices`, re-mixing the hash with
/// `salt`.
///
/// `salt == 0` is the cluster-wide bucket assignment, which a shuffle must never perturb. A
/// non-zero salt exists for **re-splitting a bucket that did not fit**, where re-using the
/// unsalted hash is not merely suboptimal but inert: `bucket_of` reads the low bits at a
/// power-of-two bucket count, so re-partitioning a 16-way bucket into 8 sub-buckets sends
/// every row to `bucket & 7` — one sub-bucket, always. The re-partition writes and re-reads
/// the whole bucket and changes nothing.
pub(crate) fn partition_morsels_by_index_salted(
    batches: &[RecordBatch],
    key_indices: &[usize],
    parts: usize,
    salt: u64,
) -> Result<Vec<RecordBatch>, InterpError> {
    partition_morsels_with(
        batches,
        parts,
        |b| Ok(key_indices.iter().map(|&i| b.column(i).clone()).collect()),
        salt,
    )
}

/// The shared body: everything but *how a morsel's key columns are selected* is
/// independent of whether the caller names its keys or indexes them.
fn partition_morsels_with(
    batches: &[RecordBatch],
    parts: usize,
    key_cols_of: impl Fn(&RecordBatch) -> Result<Vec<ArrayRef>, InterpError> + Sync,
    salt: u64,
) -> Result<Vec<RecordBatch>, InterpError> {
    debug_assert!(parts >= 1);
    if parts == 1 || batches.is_empty() {
        return Ok(vec![crate::ops::materialize(batches)?]);
    }

    // One hash pass per morsel, then that morsel's row ids binned by bucket into a flat
    // CSR array. The `Vec<Vec<u32>>` shape would ask for one growing vector per
    // (morsel, bucket) — ~350k of them here — where the whole step is a single pass.
    let per_morsel: Vec<(Vec<u32>, Vec<u32>)> = batches
        .par_iter()
        .map(|batch| {
            let key_cols = key_cols_of(batch)?;
            let part_of = shuffle::bucket_of_rows_salted(&key_cols, batch.num_rows(), parts, salt)?;
            Ok(shuffle::bucket_csr(&part_of, parts))
        })
        .collect::<Result<_, InterpError>>()?;

    let schema = batches[0].schema();
    let ncols = schema.fields().len();
    // `interleave`'s sources are the same for every bucket; build the pointer table once,
    // and decide once per column how it will be moved (`plan_column`).
    let sources: Vec<Vec<&dyn Array>> = (0..ncols)
        .map(|c| batches.iter().map(|b| b.column(c).as_ref()).collect())
        .collect();
    let plans: Vec<ColGather> = sources.iter().map(|s| plan_column(s)).collect();
    let buckets: Vec<RecordBatch> = (0..parts)
        .into_par_iter()
        .map(|bucket| {
            // `(morsel, row)` pairs for this bucket, morsels in order, rows in order — the
            // relation's own order, restricted to the rows that hashed here. Sized exactly,
            // so the gather never reallocates.
            let total: usize = per_morsel
                .iter()
                .map(|(_, off)| (off[bucket + 1] - off[bucket]) as usize)
                .sum();
            // Built lazily: only a column that declined the flat gather needs index pairs.
            let mut pairs: Option<Vec<(usize, usize)>> = None;
            let columns: Vec<ArrayRef> = plans
                .iter()
                .zip(&sources)
                .map(|(plan, src)| match plan {
                    ColGather::Fast(cols) => Ok(cols.gather(&per_morsel, bucket, total)),
                    ColGather::Interleave => {
                        let pairs = pairs.get_or_insert_with(|| {
                            let mut p = Vec::with_capacity(total);
                            for (morsel, (rows, off)) in per_morsel.iter().enumerate() {
                                p.extend(
                                    rows[off[bucket] as usize..off[bucket + 1] as usize]
                                        .iter()
                                        .map(|&row| (morsel, row as usize)),
                                );
                            }
                            p
                        });
                        interleave(src, pairs).map_err(InterpError::from)
                    }
                })
                .collect::<Result<_, InterpError>>()?;
            RecordBatch::try_new(schema.clone(), columns).map_err(InterpError::from)
        })
        .collect::<Result<_, InterpError>>()?;
    Ok(buckets)
}

/// One column's source arrays, downcast once for the whole partition.
///
/// The downcast is per (column, morsel) — 3,663 morsels here — and the gather runs per
/// (column, bucket). Doing the downcast *inside* the bucket loop makes it
/// `buckets × columns × morsels`, which is invisible at 96 buckets and costs 20% of TPC-H
/// Q9 at 576. It depends only on the column, so it is hoisted to exactly that.
enum ColGather<'a> {
    Fast(FastCols<'a>),
    /// A string, a nested type, or any source carrying a null: `interleave` owns it.
    Interleave,
}

/// Gather `cols`' rows for `bucket` into a flat value array.
///
/// `interleave` needs a materialized `&[(usize, usize)]` — **sixteen bytes of index per
/// output row**. Partitioning a 60 M-row probe side that way writes and re-reads a 960 MB
/// scratch array to move 480 MB of payload: ~2.9 GB of traffic, and the ~120 ms it measured.
/// The row ids already exist as `u32` in the CSR bins, so a column whose copy is a plain
/// value move reads them in place and writes the output directly. Traffic drops to ~1.2 GB.
fn gather_from<T: ArrowPrimitiveType>(
    cols: &[&PrimitiveArray<T>],
    per_morsel: &[(Vec<u32>, Vec<u32>)],
    bucket: usize,
    total: usize,
) -> ArrayRef {
    let mut out: Vec<T::Native> = Vec::with_capacity(total);
    for (array, (rows, off)) in cols.iter().zip(per_morsel) {
        let values = array.values();
        out.extend(
            rows[off[bucket] as usize..off[bucket + 1] as usize]
                .iter()
                .map(|&r| values[r as usize]),
        );
    }
    Arc::new(PrimitiveArray::<T>::new(out.into(), None))
}

macro_rules! fast_cols {
    ($($variant:ident => $ty:ty),* $(,)?) => {
        /// The concrete primitive types the flat gather handles, downcast once per column.
        enum FastCols<'a> { $($variant(Vec<&'a PrimitiveArray<$ty>>)),* }

        impl FastCols<'_> {
            fn gather(
                &self,
                per_morsel: &[(Vec<u32>, Vec<u32>)],
                bucket: usize,
                total: usize,
            ) -> ArrayRef {
                match self {
                    $(FastCols::$variant(cols) => gather_from(cols, per_morsel, bucket, total)),*
                }
            }
        }

        /// Downcast every source of one column, or `None` if any is not `$ty`.
        fn downcast_all<'a, T: ArrowPrimitiveType>(
            sources: &[&'a dyn Array],
        ) -> Option<Vec<&'a PrimitiveArray<T>>> {
            sources
                .iter()
                .map(|a| a.as_any().downcast_ref::<PrimitiveArray<T>>())
                .collect()
        }
    };
}

fast_cols! {
    I8 => Int8Type, I16 => Int16Type, I32 => Int32Type, I64 => Int64Type,
    U8 => UInt8Type, U16 => UInt16Type, U32 => UInt32Type, U64 => UInt64Type,
    F32 => Float32Type, F64 => Float64Type,
    D32 => Date32Type, D64 => Date64Type,
    TsS => TimestampSecondType, TsMs => TimestampMillisecondType,
    TsUs => TimestampMicrosecondType, TsNs => TimestampNanosecondType,
}

/// Decide once, per column, how its rows will be moved into each bucket.
fn plan_column<'a>(sources: &[&'a dyn Array]) -> ColGather<'a> {
    if sources.iter().any(|a| a.null_count() > 0) {
        return ColGather::Interleave; // a null buffer to rebuild: `interleave` owns it
    }
    let Some(dtype) = sources.first().map(|a| a.data_type()) else {
        return ColGather::Interleave;
    };
    macro_rules! fast {
        ($variant:ident, $ty:ty) => {
            match downcast_all::<$ty>(sources) {
                Some(cols) => ColGather::Fast(FastCols::$variant(cols)),
                None => ColGather::Interleave,
            }
        };
    }
    match dtype {
        DataType::Int8 => fast!(I8, Int8Type),
        DataType::Int16 => fast!(I16, Int16Type),
        DataType::Int32 => fast!(I32, Int32Type),
        DataType::Int64 => fast!(I64, Int64Type),
        DataType::UInt8 => fast!(U8, UInt8Type),
        DataType::UInt16 => fast!(U16, UInt16Type),
        DataType::UInt32 => fast!(U32, UInt32Type),
        DataType::UInt64 => fast!(U64, UInt64Type),
        DataType::Float32 => fast!(F32, Float32Type),
        DataType::Float64 => fast!(F64, Float64Type),
        DataType::Date32 => fast!(D32, Date32Type),
        DataType::Date64 => fast!(D64, Date64Type),
        DataType::Timestamp(TimeUnit::Second, None) => fast!(TsS, TimestampSecondType),
        DataType::Timestamp(TimeUnit::Millisecond, None) => fast!(TsMs, TimestampMillisecondType),
        DataType::Timestamp(TimeUnit::Microsecond, None) => fast!(TsUs, TimestampMicrosecondType),
        DataType::Timestamp(TimeUnit::Nanosecond, None) => fast!(TsNs, TimestampNanosecondType),
        _ => ColGather::Interleave,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, Int64Array, StringArray};
    use std::sync::Arc;

    fn batch(keys: &[i64], vals: &[&str]) -> RecordBatch {
        RecordBatch::try_from_iter(vec![
            ("k", Arc::new(Int64Array::from(keys.to_vec())) as ArrayRef),
            ("v", Arc::new(StringArray::from(vals.to_vec())) as ArrayRef),
        ])
        .unwrap()
    }

    fn rows_of(batches: &[RecordBatch]) -> Vec<(i64, String)> {
        let mut out = Vec::new();
        for b in batches {
            let k = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let v = b.column(1).as_any().downcast_ref::<StringArray>().unwrap();
            for i in 0..b.num_rows() {
                out.push((k.value(i), v.value(i).to_string()));
            }
        }
        out
    }

    /// The invariant everything rests on: partitioning morsels is byte-for-byte what
    /// partitioning the concatenated relation produces — same buckets, same order in each.
    #[test]
    fn matches_partitioning_the_concatenated_relation() {
        let morsels = [
            batch(&[1, 2, 3, 4], &["a", "b", "c", "d"]),
            batch(&[5, 1, 6, 2], &["e", "f", "g", "h"]),
            batch(&[3, 7], &["i", "j"]),
        ];
        let parts = 4;
        let whole = crate::ops::materialize(&morsels).unwrap();
        let expected = shuffle::partition_by_keys(&whole, &[0], parts).unwrap();
        let got = partition_morsels(&morsels, &["k".into()], parts).unwrap();

        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(&expected) {
            assert_eq!(
                rows_of(std::slice::from_ref(g)),
                rows_of(std::slice::from_ref(e))
            );
        }
    }

    /// Equal keys co-partition across morsels — the invariant the per-bucket join needs.
    #[test]
    fn equal_keys_share_a_bucket_across_morsels() {
        let morsels = [
            batch(&[7], &["a"]),
            batch(&[7], &["b"]),
            batch(&[9], &["c"]),
        ];
        let got = partition_morsels(&morsels, &["k".into()], 8).unwrap();
        let holding_seven: Vec<usize> = got
            .iter()
            .enumerate()
            .filter(|(_, b)| {
                rows_of(std::slice::from_ref(b))
                    .iter()
                    .any(|(k, _)| *k == 7)
            })
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            holding_seven.len(),
            1,
            "key 7 must land in exactly one bucket"
        );
        let bucket = &got[holding_seven[0]];
        assert_eq!(rows_of(std::slice::from_ref(bucket)).len(), 2);
    }

    /// Every row is placed exactly once; nothing is dropped or duplicated.
    #[test]
    fn every_row_is_placed_exactly_once() {
        let morsels = [
            batch(&[1, 2, 3], &["a", "b", "c"]),
            batch(&[4, 5], &["d", "e"]),
        ];
        let got = partition_morsels(&morsels, &["k".into()], 3).unwrap();
        let mut all = rows_of(&got);
        all.sort();
        assert_eq!(
            all,
            vec![
                (1, "a".into()),
                (2, "b".into()),
                (3, "c".into()),
                (4, "d".into()),
                (5, "e".into())
            ]
        );
    }

    /// One bucket is the degenerate case: no hashing, no gather.
    #[test]
    fn a_single_bucket_is_the_concatenated_relation() {
        let morsels = [batch(&[1, 2], &["a", "b"]), batch(&[3], &["c"])];
        let got = partition_morsels(&morsels, &["k".into()], 1).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(rows_of(&got).len(), 3);
    }

    /// **A sliced morsel indexes its own rows.** `PrimitiveArray::values()` is offset-
    /// adjusted, but a gather that read the whole backing buffer would silently take the
    /// wrong values. Morsels are almost always slices of a larger batch, so this is the
    /// invariant the fast gather lives or dies on.
    #[test]
    fn a_sliced_morsel_gathers_only_its_own_values() {
        let whole = batch(&[10, 11, 12, 13, 14, 15], &["a", "b", "c", "d", "e", "f"]);
        let morsels = [whole.slice(2, 2), whole.slice(4, 2)]; // keys 12,13 then 14,15
        let got = partition_morsels(&morsels, &["k".into()], 4).unwrap();
        let mut all = rows_of(&got);
        all.sort();
        assert_eq!(
            all,
            vec![
                (12, "c".into()),
                (13, "d".into()),
                (14, "e".into()),
                (15, "f".into())
            ]
        );
    }

    /// A nullable column falls back to `interleave`, and the nulls survive the round trip.
    #[test]
    fn a_column_with_nulls_falls_back_and_keeps_them() {
        let mk = |keys: Vec<i64>, vals: Vec<Option<i64>>| {
            RecordBatch::try_from_iter(vec![
                ("k", Arc::new(Int64Array::from(keys)) as ArrayRef),
                ("v", Arc::new(Int64Array::from(vals)) as ArrayRef),
            ])
            .unwrap()
        };
        let morsels = [mk(vec![1, 2], vec![None, Some(7)]), mk(vec![3], vec![None])];
        let got = partition_morsels(&morsels, &["k".into()], 4).unwrap();
        let mut seen: Vec<(i64, Option<i64>)> = Vec::new();
        for b in &got {
            let k = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let v = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
            for i in 0..b.num_rows() {
                seen.push((k.value(i), (!v.is_null(i)).then(|| v.value(i))));
            }
        }
        seen.sort();
        assert_eq!(seen, vec![(1, None), (2, Some(7)), (3, None)]);
    }

    /// Float and date columns take the fast gather; a string column takes `interleave`.
    /// Both must land in the same bucket batch with the same rows.
    #[test]
    fn mixed_fast_and_fallback_columns_agree() {
        use arrow::array::{Date32Array, Float64Array};
        let mk = |k: Vec<i64>, f: Vec<f64>, d: Vec<i32>, s: Vec<&str>| {
            RecordBatch::try_from_iter(vec![
                ("k", Arc::new(Int64Array::from(k)) as ArrayRef),
                ("f", Arc::new(Float64Array::from(f)) as ArrayRef),
                ("d", Arc::new(Date32Array::from(d)) as ArrayRef),
                ("s", Arc::new(StringArray::from(s)) as ArrayRef),
            ])
            .unwrap()
        };
        let morsels = [
            mk(vec![1, 2], vec![1.5, 2.5], vec![100, 200], vec!["a", "b"]),
            mk(vec![3], vec![3.5], vec![300], vec!["c"]),
        ];
        let got = partition_morsels(&morsels, &["k".into()], 4).unwrap();
        let mut seen = Vec::new();
        for b in &got {
            let k = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let f = b.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
            let d = b.column(2).as_any().downcast_ref::<Date32Array>().unwrap();
            let st = b.column(3).as_any().downcast_ref::<StringArray>().unwrap();
            for i in 0..b.num_rows() {
                seen.push((k.value(i), f.value(i), d.value(i), st.value(i).to_string()));
            }
        }
        seen.sort_by_key(|r| r.0);
        assert_eq!(
            seen,
            vec![
                (1, 1.5, 100, "a".into()),
                (2, 2.5, 200, "b".into()),
                (3, 3.5, 300, "c".into())
            ]
        );
    }

    /// A bucket that receives no rows is still an empty batch with the right schema.
    #[test]
    fn empty_buckets_keep_their_schema() {
        let morsels = [batch(&[42], &["a"])];
        let got = partition_morsels(&morsels, &["k".into()], 8).unwrap();
        assert_eq!(got.len(), 8);
        assert_eq!(got.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
        for b in &got {
            assert_eq!(b.schema(), morsels[0].schema());
        }
    }
}
