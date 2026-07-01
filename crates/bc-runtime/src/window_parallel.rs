//! Bucket-parallel window execution: hash-partition rows by the PARTITION BY keys so
//! every window partition lands wholly inside one bucket, run the serial window kernel
//! ([`crate::window::window_serial`]) on each bucket across rayon cores, and scatter each
//! function's output column back to original row order. Split out of `window` along the
//! parallelism seam; bit-identical to the serial kernel as a per-row result.

use arrow::array::{Array, ArrayRef, UInt32Array};
use arrow::compute::take;
use arrow::compute::SortOptions;
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
    parallel_row_threshold: usize,
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
        return window_serial(
            partition_keys,
            order_keys,
            funcs,
            num_rows,
            parallel_row_threshold,
        );
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
            window_serial(&bk, &bo, &bc, idx.len(), usize::MAX)
        })
        .collect::<Result<_, _>>()?;

    // Scatter back: concatenate each function's per-bucket columns in bucket order,
    // then `take` by the inverse permutation to restore original row positions. `perm`
    // is the original index of each bucket-order row; `inv[orig] = k` inverts it.
    let perm: Vec<u32> = buckets.iter().flatten().copied().collect();
    let mut inv = vec![0u32; num_rows];
    for (k, &orig) in perm.iter().enumerate() {
        inv[orig as usize] = k as u32;
    }
    let inv_arr = UInt32Array::from(inv);
    (0..funcs.len())
        .map(|f| {
            let cols: Vec<&dyn Array> = per_bucket.iter().map(|b| b[f].as_ref()).collect();
            let concatenated = arrow::compute::concat(&cols)?;
            Ok(take(concatenated.as_ref(), &inv_arr, None)?)
        })
        .collect()
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
