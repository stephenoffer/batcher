//! Bucket-parallel window execution: hash-partition rows by the PARTITION BY keys so
//! every window partition lands wholly inside one bucket, run the serial window kernel
//! ([`crate::window::window_serial`]) on each bucket across rayon cores, and scatter each
//! function's output column back to original row order. Split out of `window` along the
//! parallelism seam; bit-identical to the serial kernel as a per-row result.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, PrimitiveArray, UInt32Array};
use arrow::buffer::NullBuffer;
use arrow::compute::take;
use arrow::compute::SortOptions;
use arrow::datatypes::{ArrowPrimitiveType, DataType, Float64Type, Int64Type};
use arrow::row::{RowConverter, SortField};
use rayon::prelude::*;

use crate::error::RuntimeError;
use crate::window::{window_serial, WindowCall};

/// Bucket-parallel window: hash-partition rows by `partition_keys` into `nbuckets`
/// (equal keys together), run [`window_serial`] on each bucket across cores, and
/// scatter each function's output column back to original row order. Partitioning only
/// regroups whole partitions across buckets, and the final scatter restores original
/// positions, so the per-row result is identical to the serial kernel.
pub(crate) fn window_parallel(
    partition_keys: &[ArrayRef],
    order_keys: &[(ArrayRef, SortOptions)],
    funcs: &[WindowCall],
    num_rows: usize,
    nbuckets: usize,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    let buckets = partition_row_indices(partition_keys, num_rows, nbuckets)?;

    // Load-balance guard for the pathological case: when a *single* partition holds
    // most rows (a near-constant PARTITION BY), bucketing collapses to one busy bucket
    // and the partition/gather/scatter plumbing is pure overhead over the serial kernel.
    // A handful of *balanced* large partitions still parallelizes fine (each rides a
    // core, and the order-key encoding fans out too), so only bail when one bucket
    // dominates — otherwise stay on the parallel path.
    let max_bucket = buckets.iter().map(Vec::len).max().unwrap_or(0);
    if max_bucket > num_rows / 2 {
        return window_serial(partition_keys, order_keys, funcs, num_rows);
    }

    // Each bucket gathers its rows' keys/order/values and runs the serial kernel over
    // just those rows (a huge per-bucket threshold keeps that call serial — the cores
    // are already spent on the buckets, not on nested per-partition sorts).
    let per_bucket: Vec<Vec<ArrayRef>> = buckets
        .par_iter()
        .map(|idx| -> Result<Vec<ArrayRef>, RuntimeError> {
            let idx_arr = UInt32Array::from(idx.clone());
            let g = |a: &ArrayRef| take(a.as_ref(), &idx_arr, None);
            let bk: Vec<ArrayRef> = partition_keys.iter().map(g).collect::<Result<_, _>>()?;
            let bo: Vec<(ArrayRef, SortOptions)> = order_keys
                .iter()
                .map(|(a, o)| Ok((take(a.as_ref(), &idx_arr, None)?, *o)))
                .collect::<Result<_, RuntimeError>>()?;
            let bc: Vec<WindowCall> = funcs
                .iter()
                .map(|c| -> Result<WindowCall, RuntimeError> {
                    Ok(WindowCall {
                        func: c.func,
                        offset: c.offset,
                        frame: c.frame,
                        values: c.values.as_ref().map(g).transpose()?,
                    })
                })
                .collect::<Result<_, _>>()?;
            window_serial(&bk, &bo, &bc, idx.len())
        })
        .collect::<Result<_, _>>()?;

    // Scatter each function's per-bucket results back to original row order. For the
    // primitive output types (every window function here yields Int64 or Float64) this is
    // a cache-blocked parallel scatter ([`scatter_blocked`], the dominant window cost done
    // right); other output types fall back to a concat + inverse-permutation gather.
    (0..funcs.len())
        .map(|f| {
            let cols: Vec<&ArrayRef> = per_bucket.iter().map(|b| &b[f]).collect();
            match cols[0].data_type() {
                DataType::Int64 => scatter_blocked::<Int64Type>(&cols, &buckets, num_rows),
                DataType::Float64 => scatter_blocked::<Float64Type>(&cols, &buckets, num_rows),
                _ => scatter_by_gather(&cols, &buckets, num_rows),
            }
        })
        .collect()
}

/// Cache-blocked parallel scatter of the per-bucket result columns back to original row
/// order — the window's dominant cost, done right.
///
/// `cols[b]` are bucket `b`'s results in bucket-local order; `buckets[b][k]` is the original
/// row index of that bucket's `k`-th row, and **each bucket's indices are ascending**
/// (guaranteed by [`partition_row_indices`]). The output is split into `nthreads` disjoint
/// contiguous ranges (`chunks_mut` — safe, no aliasing), and each range, on its own core,
/// binary-searches every bucket for the ascending slice of indices that fall in it and
/// writes only those. So every core writes solely to its own cache-local output range (no
/// cross-core cache-line contention that sinks a naive parallel scatter) and reads each
/// bucket's values sequentially. Result-identical to a `take` by the inverse permutation.
fn scatter_blocked<T>(
    cols: &[&ArrayRef],
    buckets: &[Vec<u32>],
    num_rows: usize,
) -> Result<ArrayRef, RuntimeError>
where
    T: ArrowPrimitiveType,
    T::Native: Copy + Default + Send + Sync,
{
    let any_null = cols.iter().any(|c| c.null_count() > 0);
    let arrs: Vec<&PrimitiveArray<T>> = cols.iter().map(|c| c.as_primitive::<T>()).collect();
    let nthreads = rayon::current_num_threads().max(1);
    let range = num_rows.div_ceil(nthreads).max(1);
    let mut values = vec![T::Native::default(); num_rows];

    // Write each output range on its own core. `lo` is the range's first original index;
    // for each bucket the rows in `[lo, hi)` are the ascending slice `[start, end)`.
    let fill = |lo: usize, hi: usize, vc: &mut [T::Native], bc: Option<&mut [bool]>| {
        let mut bc = bc;
        for (b, bucket) in buckets.iter().enumerate() {
            let start = bucket.partition_point(|&x| (x as usize) < lo);
            let end = bucket.partition_point(|&x| (x as usize) < hi);
            let arr = arrs[b];
            for (offset, &orig) in bucket[start..end].iter().enumerate() {
                let k = start + offset; // bucket-local position in `arr`
                let o = orig as usize - lo;
                vc[o] = arr.value(k);
                if let Some(bc) = bc.as_deref_mut() {
                    bc[o] = arr.is_valid(k);
                }
            }
        }
    };

    let nulls = if any_null {
        let mut valid = vec![false; num_rows];
        values
            .par_chunks_mut(range)
            .zip(valid.par_chunks_mut(range))
            .enumerate()
            .for_each(|(ci, (vc, bc))| {
                let lo = ci * range;
                fill(lo, lo + vc.len(), vc, Some(bc));
            });
        Some(NullBuffer::from(valid))
    } else {
        values
            .par_chunks_mut(range)
            .enumerate()
            .for_each(|(ci, vc)| {
                let lo = ci * range;
                fill(lo, lo + vc.len(), vc, None);
            });
        None
    };
    Ok(Arc::new(PrimitiveArray::<T>::new(values.into(), nulls)))
}

/// Fallback scatter for non-primitive window outputs: concatenate the per-bucket columns in
/// bucket order and `take` by the inverse permutation. (Ranking / running / value functions
/// here all yield Int64 / Float64, so this is reached only by exotic output types.)
fn scatter_by_gather(
    cols: &[&ArrayRef],
    buckets: &[Vec<u32>],
    num_rows: usize,
) -> Result<ArrayRef, RuntimeError> {
    let perm: Vec<u32> = buckets.iter().flatten().copied().collect();
    let mut inv = vec![0u32; num_rows];
    for (k, &orig) in perm.iter().enumerate() {
        inv[orig as usize] = k as u32;
    }
    let refs: Vec<&dyn Array> = cols.iter().map(|a| a.as_ref()).collect();
    let concatenated = arrow::compute::concat(&refs)?;
    Ok(take(concatenated.as_ref(), &UInt32Array::from(inv), None)?)
}

const SEED: ahash::RandomState =
    ahash::RandomState::with_seeds(0x1234_5678, 0x9abc_def0, 0x0fed_cba9, 0x8765_4321);

/// Hash-partition `0..num_rows` into `nbuckets` index lists by the partition keys, so
/// equal keys share a bucket (whole window partitions never split). The per-row bucket
/// id is computed in parallel; the per-bucket index lists are then assembled by
/// concatenating each chunk's lists in chunk order (so each bucket's indices stay
/// ascending, which the scatter-back relies on for a deterministic permutation).
fn partition_row_indices(
    partition_keys: &[ArrayRef],
    num_rows: usize,
    nbuckets: usize,
) -> Result<Vec<Vec<u32>>, RuntimeError> {
    let part_of = bucket_of_each_row(partition_keys, num_rows, nbuckets)?;

    // Parallel stable counting sort into per-bucket lists: each row-range chunk builds
    // its own per-bucket lists, then bucket `b`'s global list is the chunks' `b`-lists
    // concatenated in chunk order. Mirrors `shuffle::scatter_into_buckets`.
    let nthreads = rayon::current_num_threads().max(1);
    let chunk = num_rows.div_ceil(nthreads).max(1);
    let per_chunk: Vec<Vec<Vec<u32>>> = part_of
        .par_chunks(chunk)
        .enumerate()
        .map(|(ci, slice)| {
            let base = (ci * chunk) as u32;
            let mut lists: Vec<Vec<u32>> = vec![Vec::new(); nbuckets];
            for (j, &b) in slice.iter().enumerate() {
                lists[b as usize].push(base + j as u32);
            }
            lists
        })
        .collect();
    Ok((0..nbuckets)
        .into_par_iter()
        .map(|b| {
            let total: usize = per_chunk.iter().map(|c| c[b].len()).sum();
            let mut out = Vec::with_capacity(total);
            for c in &per_chunk {
                out.extend_from_slice(&c[b]);
            }
            out
        })
        .collect())
}

/// The bucket id (`hash(key) % nbuckets`) of each row, computed in parallel. A single
/// integer / string key hashes its native value directly (no `RowConverter`).
fn bucket_of_each_row(
    partition_keys: &[ArrayRef],
    num_rows: usize,
    nbuckets: usize,
) -> Result<Vec<u32>, RuntimeError> {
    let n = nbuckets as u64;
    if partition_keys.len() == 1 {
        use arrow::array::{Int64Array, StringArray};
        use arrow::datatypes::DataType;
        let a = &partition_keys[0];
        match a.data_type() {
            DataType::Int64 => {
                let v = a.as_any().downcast_ref::<Int64Array>().expect("i64");
                return Ok((0..num_rows)
                    .into_par_iter()
                    .map(|i| {
                        let h = if v.is_null(i) {
                            0
                        } else {
                            SEED.hash_one(v.value(i))
                        };
                        (h % n) as u32
                    })
                    .collect());
            }
            DataType::Utf8 => {
                let v = a.as_any().downcast_ref::<StringArray>().expect("utf8");
                return Ok((0..num_rows)
                    .into_par_iter()
                    .map(|i| {
                        let h = if v.is_null(i) {
                            0
                        } else {
                            SEED.hash_one(v.value(i))
                        };
                        (h % n) as u32
                    })
                    .collect());
            }
            _ => {}
        }
    }

    // General path: row-encode the (multi-column / non-int) keys, hash the bytes.
    let fields: Vec<SortField> = partition_keys
        .iter()
        .map(|a| SortField::new(a.data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields)?;
    let rows = converter.convert_columns(partition_keys)?;
    Ok((0..num_rows)
        .map(|i| (SEED.hash_one(rows.row(i)) % n) as u32)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::window::{window_serial, window_with, WindowCall, WindowFn};
    use arrow::array::{BooleanArray, Int64Array, StringArray};

    fn i64s(v: &[i64]) -> ArrayRef {
        Arc::new(Int64Array::from(v.to_vec()))
    }
    fn asc(a: ArrayRef) -> (ArrayRef, SortOptions) {
        (
            a,
            SortOptions {
                descending: false,
                nulls_first: false,
            },
        )
    }

    /// The parallel scatter for NON-primitive outputs (`scatter_by_gather`) must equal the
    /// serial kernel. Prior parity tests only exercised Int64/Float64 outputs (the
    /// `scatter_blocked` path); a String `first_value`/`last_value` and a Boolean min/max
    /// output route through the fallback gather, which was never checked seq==par.
    #[test]
    fn parallel_matches_serial_nonprimitive_outputs() {
        let n = 400usize;
        let part = i64s(&(0..n as i64).map(|i| i % 30).collect::<Vec<_>>());
        let ord = i64s(&(0..n as i64).map(|i| (i * 7 + 13) % 50).collect::<Vec<_>>());
        let svals: ArrayRef = Arc::new(StringArray::from(
            (0..n).map(|i| format!("s{}", i % 17)).collect::<Vec<_>>(),
        ));
        let bvals: ArrayRef = Arc::new(BooleanArray::from(
            (0..n).map(|i| i % 3 == 0).collect::<Vec<_>>(),
        ));
        let order = [asc(ord)];

        let cases: Vec<WindowCall> = vec![
            WindowCall {
                func: WindowFn::FirstValue,
                values: Some(svals.clone()),
                offset: 1,
                frame: None,
            },
            WindowCall {
                func: WindowFn::LastValue,
                values: Some(svals.clone()),
                offset: 1,
                frame: None,
            },
            WindowCall {
                func: WindowFn::Min,
                values: Some(bvals.clone()),
                offset: 1,
                frame: None,
            },
            WindowCall {
                func: WindowFn::Max,
                values: Some(bvals.clone()),
                offset: 1,
                frame: None,
            },
            WindowCall {
                func: WindowFn::Min,
                values: Some(svals.clone()),
                offset: 1,
                frame: None,
            },
        ];
        for call in cases {
            let f = [call];
            let par = window_with(std::slice::from_ref(&part), &order, &f, n, 1).unwrap();
            let ser = window_serial(std::slice::from_ref(&part), &order, &f, n).unwrap();
            assert_eq!(par[0].as_ref(), ser[0].as_ref(), "{:?}", f[0].func);
        }
    }
}
