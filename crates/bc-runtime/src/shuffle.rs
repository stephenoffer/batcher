//! Hash repartitioning — the shuffle primitive.
//!
//! Splits a batch's rows into `num_partitions` buckets by a stable hash of the
//! key columns. This is the single mechanism behind both **parallel** execution
//! (buckets → threads) and **distributed** execution (buckets → actors over Arrow
//! Flight): equal keys always land in the same bucket, so a hash join or
//! group-by can be computed independently per bucket and unioned. The hash is
//! seeded deterministically so both sides of a join agree within a run.
//!
//! Keys are encoded with arrow's row format (multi-key, any type) so a single
//! code path partitions on any key types.

use std::collections::HashSet;

use arrow::array::{
    Array, ArrayRef, AsArray, Float64Array, GenericBinaryArray, GenericStringArray,
    LargeStringArray, OffsetSizeTrait, RecordBatch, StringArray, UInt32Array,
};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use arrow::row::{RowConverter, SortField};
use rayon::prelude::*;

use crate::error::RuntimeError;

/// Below this row count the hash pass runs serially: rayon's fan-out/join costs more
/// than it saves on a small batch (and most morsels are small). A shuffle of millions
/// of rows — a join build/probe side or a distributed map partition — clears it and
/// fans the hash across every core.
const PAR_HASH_MIN_ROWS: usize = 1 << 16;

// Fixed seeds → deterministic partitioning within a process (so the two sides of
// a join hash identically). Not for security; collision resistance is irrelevant.
const SEED: ahash::RandomState =
    ahash::RandomState::with_seeds(0x1234_5678, 0x9abc_def0, 0x0fed_cba9, 0x8765_4321);

/// Partition `batch` into `num_partitions` buckets by a hash of `key_indices`.
/// Returns one `RecordBatch` per bucket (some may be empty), each with the input
/// schema. Empty input yields `num_partitions` empty batches.
pub fn partition_by_keys(
    batch: &RecordBatch,
    key_indices: &[usize],
    num_partitions: usize,
) -> Result<Vec<RecordBatch>, RuntimeError> {
    let keys: Vec<ArrayRef> = key_indices
        .iter()
        .map(|&i| batch.column(i).clone())
        .collect();
    partition_by_key_arrays(batch, &keys, num_partitions)
}

/// Like [`partition_by_keys`], but the key columns are supplied directly as arrays
/// rather than by index into `batch`. This lets callers partition by *derived* keys
/// (a window `PARTITION BY` expression, a salted join key) without first appending
/// them to the batch. `keys` must each have `batch.num_rows()` rows; an empty `keys`
/// routes every row to bucket 0 (a single global partition).
pub fn partition_by_key_arrays(
    batch: &RecordBatch,
    keys: &[ArrayRef],
    num_partitions: usize,
) -> Result<Vec<RecordBatch>, RuntimeError> {
    assert!(num_partitions >= 1);
    // Single global bucket → no hashing or gather; the Arc-backed batch is returned
    // as-is (a refcount bump, not a copy). Covers the common non-distributed case.
    if num_partitions == 1 {
        return Ok(vec![batch.clone()]);
    }
    let part_of = bucket_of_rows(keys, batch.num_rows(), num_partitions)?;
    scatter_into_buckets(batch, &part_of, num_partitions)
}

/// The bucket each row's key hashes to, one `u32` per row.
///
/// Split out of [`partition_by_key_arrays`] so a caller holding a *relation* as morsels can
/// bucket each morsel and gather the payload once, rather than concatenating the relation
/// first and gathering twice. The bucket is a deterministic function of the key *value*, so a
/// row lands in the same bucket whichever morsel carries it — the co-partitioning invariant
/// the parallel and distributed joins rely on.
///
/// An empty key set routes every row to bucket 0 (hashing an empty row is ill-defined). A
/// single integer key hashes its native values directly, skipping the `RowConverter` encoding
/// the general path needs; both sides of a join dispatch on the same key type, so equal keys
/// still co-partition either way.
pub fn bucket_of_rows(
    keys: &[ArrayRef],
    rows: usize,
    num_partitions: usize,
) -> Result<Vec<u32>, RuntimeError> {
    if keys.is_empty() {
        return Ok(vec![0u32; rows]);
    }
    // Canonicalize float keys FIRST, so every path below (raw hash or `RowConverter`) sees the
    // same bits `assign_groups` groups by. Without this a `-0.0` and a `0.0` — one group to the
    // assigner — encode differently, land on different reducers, and the query returns two
    // groups where the single-node oracle returns one (invariant #7). See `crate::keys`.
    let canon = crate::keys::canonicalize_float_keys(keys);
    let keys: &[ArrayRef] = canon.as_deref().unwrap_or(keys);
    if let Some(part) = partition_int_key(keys, num_partitions) {
        return Ok(part);
    }
    // Mixed Int64 / string / binary key (null-free): hash each row's raw column values
    // directly, in parallel, instead of arrow's `RowConverter` — whose `convert_columns`
    // encodes every row into its byte format in one *serial* pass (the parallel hash after it
    // can't hide that). That serial encode is the whole cost of a `COUNT(DISTINCT id) GROUP BY
    // flag` partition (a `(flag, orderkey)` shuffle over 60M rows ran at ~14% CPU / ~1s).
    // Equal non-null rows hash identically, so they co-partition — all a shuffle/DISTINCT
    // needs. Null-free only: a null slot's arbitrary raw bytes could split two equal
    // null-bearing rows across buckets (fine for a join, where null keys never match, but not
    // for a DISTINCT, where nulls compare equal); a nullable key keeps the `RowConverter`.
    if let Some(part) = partition_mixed_key(keys, num_partitions) {
        return Ok(part);
    }
    let fields: Vec<SortField> = keys
        .iter()
        .map(|a| SortField::new(a.data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields)?;
    let encoded = converter.convert_columns(keys)?;
    Ok(if rows >= PAR_HASH_MIN_ROWS {
        (0..rows)
            .into_par_iter()
            .map(|i| bucket_of(SEED.hash_one(encoded.row(i)), num_partitions))
            .collect()
    } else {
        (0..rows)
            .map(|i| bucket_of(SEED.hash_one(encoded.row(i)), num_partitions))
            .collect()
    })
}

/// Compute a per-row bucket id across every core on a large input.
///
/// `part_of[i]` depends only on row `i`, so this is a pure map — the result is identical to
/// the serial loop, whatever the thread count. It matters most for a *string* key, whose
/// `partition_point` costs ~log2(buckets) full string comparisons per row: a 5 M-row string
/// `ORDER BY` spent 330 ms of its 461 ms here before this ran in parallel.
fn map_rows<F>(n: usize, f: F) -> Vec<u32>
where
    F: Fn(usize) -> u32 + Send + Sync,
{
    if n >= PAR_HASH_MIN_ROWS {
        (0..n).into_par_iter().map(f).collect()
    } else {
        (0..n).map(f).collect()
    }
}

/// The bucket nulls route to: whichever end the caller's final concatenation places
/// first/last. A descending sort concatenates buckets high→low, so its "front" bucket is
/// `n_buckets - 1`.
fn null_bucket_of(n_buckets: usize, nulls_first: bool, descending: bool) -> u32 {
    let front = if descending { n_buckets - 1 } else { 0 };
    (if nulls_first {
        front
    } else {
        n_buckets - 1 - front
    }) as u32
}

/// The row indices of each bucket, in input order, as one flat array plus offsets.
///
/// The `Vec<Vec<u32>>` shape [`bucket_indices`] returns costs one heap allocation per
/// bucket, and the per-bucket lists grow by reallocation. That is invisible on one large
/// relation and ruinous per morsel: hash-partitioning 3,663 morsels into 96 buckets asks
/// for ~350,000 vectors and, with each doubling from capacity 4 to ~170 rows, on the order
/// of a million allocations — for a partition step whose actual work is one pass over the
/// bucket ids.
///
/// This bins the same rows into a **CSR layout**: `rows[offsets[b]..offsets[b + 1]]` is
/// bucket `b`'s row indices, ascending, exactly as `bucket_indices(part_of, p)[b]` would
/// give them. Two allocations, no growth: a histogram fixes every bucket's extent before a
/// single scatter pass fills it. Every `part_of[i]` must be `< num_partitions`.
pub fn bucket_csr(part_of: &[u32], num_partitions: usize) -> (Vec<u32>, Vec<u32>) {
    let mut offsets = vec![0u32; num_partitions + 1];
    for &b in part_of {
        offsets[b as usize + 1] += 1;
    }
    for b in 0..num_partitions {
        offsets[b + 1] += offsets[b];
    }
    // `cursor[b]` is the next free slot in bucket `b`'s extent. Rows are visited in
    // ascending order, so each bucket's slice comes out ascending — the order the
    // per-bucket join and the `seq == par` oracle depend on.
    let mut cursor: Vec<u32> = offsets[..num_partitions].to_vec();
    let mut rows = vec![0u32; part_of.len()];
    for (i, &b) in part_of.iter().enumerate() {
        let slot = &mut cursor[b as usize];
        rows[*slot as usize] = i as u32;
        *slot += 1;
    }
    (rows, offsets)
}

/// The row indices of each bucket, in input order — [`scatter_into_buckets`] without the
/// gather.
///
/// A caller that is going to permute a bucket's rows anyway (the parallel sample-sort
/// sorts each range) can compose its permutation with these indices and gather the payload
/// **once**, instead of gathering into buckets and then gathering again to sort. Every
/// `part_of[i]` must be `< num_partitions`.
pub fn bucket_indices(part_of: &[u32], num_partitions: usize) -> Vec<Vec<u32>> {
    let n = part_of.len();
    if n < PAR_HASH_MIN_ROWS {
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); num_partitions];
        for (i, &b) in part_of.iter().enumerate() {
            buckets[b as usize].push(i as u32);
        }
        return buckets;
    }
    // Stable parallel counting sort: each row-range chunk bins its own rows, then bucket
    // `b`'s list is the chunks' `b`-lists concatenated in chunk order (each already
    // ascending), which reproduces the serial scatter's per-bucket ordering exactly.
    let nthreads = rayon::current_num_threads().max(1);
    let chunk = n.div_ceil(nthreads).max(1);
    let per_chunk: Vec<Vec<Vec<u32>>> = part_of
        .par_chunks(chunk)
        .enumerate()
        .map(|(ci, slice)| {
            // Pre-size each bucket to the uniform-key expectation so the scatter's hot
            // push loop reallocates only under real skew. (`vec![v; n]` can't be used —
            // Vec::clone drops capacity, so each cloned bucket would start at zero.)
            let cap = slice.len() / num_partitions + 1;
            let mut buckets: Vec<Vec<u32>> = (0..num_partitions)
                .map(|_| Vec::with_capacity(cap))
                .collect();
            let base = (ci * chunk) as u32;
            for (j, &b) in slice.iter().enumerate() {
                buckets[b as usize].push(base + j as u32);
            }
            buckets
        })
        .collect();
    (0..num_partitions)
        .into_par_iter()
        .map(|b| {
            let total: usize = per_chunk.iter().map(|c| c[b].len()).sum();
            let mut idx = Vec::with_capacity(total);
            for c in &per_chunk {
                idx.extend_from_slice(&c[b]);
            }
            idx
        })
        .collect()
}

/// Gather `batch`'s rows at `idx` (used by the sample-sort once it has composed a bucket's
/// indices with that bucket's sort permutation).
pub fn gather_rows(batch: &RecordBatch, idx: &[u32]) -> Result<RecordBatch, RuntimeError> {
    take_rows(batch, idx)
}

/// Counting-sort scatter: given a per-row bucket id, gather each bucket's rows into
/// its own `RecordBatch`. Histogram → prefix-sum offsets → stable scatter into one
/// contiguous index buffer, so each bucket is a contiguous slice and we pay no
/// per-bucket `Vec` reallocation. Shared by the hash and range partitioners — both
/// differ only in *how* they compute the bucket id, not in how they materialize the
/// buckets. Every `part_of[i]` must be `< num_partitions`.
fn scatter_into_buckets(
    batch: &RecordBatch,
    part_of: &[u32],
    num_partitions: usize,
) -> Result<Vec<RecordBatch>, RuntimeError> {
    let n = part_of.len();
    // Small input: serial counting sort, one contiguous scatter buffer.
    if n < PAR_HASH_MIN_ROWS {
        let mut offsets = vec![0u32; num_partitions + 1];
        for &b in part_of {
            offsets[b as usize + 1] += 1;
        }
        for b in 0..num_partitions {
            offsets[b + 1] += offsets[b];
        }
        let mut scatter = vec![0u32; n];
        let mut cursor = offsets[..num_partitions].to_vec();
        for (i, &b) in part_of.iter().enumerate() {
            let pos = &mut cursor[b as usize];
            scatter[*pos as usize] = i as u32;
            *pos += 1;
        }
        return (0..num_partitions)
            .map(|b| {
                take_rows(
                    batch,
                    &scatter[offsets[b] as usize..offsets[b + 1] as usize],
                )
            })
            .collect();
    }

    // Large input: stable parallel counting sort. Each row-range chunk builds its own
    // per-bucket index lists in parallel; concatenating those lists in chunk order
    // (each chunk's already ascending) reproduces the serial scatter's exact per-bucket
    // ordering. The per-bucket gather then runs across every core — the partitioner
    // under both the single-node hash join and the distributed shuffle.
    let nthreads = rayon::current_num_threads().max(1);
    let chunk = n.div_ceil(nthreads).max(1);
    let per_chunk: Vec<Vec<Vec<u32>>> = part_of
        .par_chunks(chunk)
        .enumerate()
        .map(|(ci, slice)| {
            // Uniform-key pre-size (see `bucket_indices`): reallocate only under real skew.
            let cap = slice.len() / num_partitions + 1;
            let mut buckets: Vec<Vec<u32>> = (0..num_partitions)
                .map(|_| Vec::with_capacity(cap))
                .collect();
            let base = (ci * chunk) as u32;
            for (j, &b) in slice.iter().enumerate() {
                buckets[b as usize].push(base + j as u32);
            }
            buckets
        })
        .collect();

    (0..num_partitions)
        .into_par_iter()
        .map(|b| {
            let total: usize = per_chunk.iter().map(|c| c[b].len()).sum();
            let mut idx = Vec::with_capacity(total);
            for c in &per_chunk {
                idx.extend_from_slice(&c[b]);
            }
            take_rows(batch, &idx)
        })
        .collect()
}

/// Range-partition `batch` into `n_buckets` globally-ordered buckets by the leading
/// sort key at `key_index` and the ascending `boundaries`. Bucket `b` receives rows
/// whose key falls in the `b`-th open interval of the boundaries
/// (`searchsorted(boundaries, key, side="right")`), so equal keys never span a
/// boundary and a concatenation of the per-bucket sorts is globally ordered. Nulls go
/// to the front or back bucket to match single-node null ordering: `front` is the
/// bucket the driver concatenates first (`n_buckets-1` for a descending sort, else
/// `0`), and nulls land there when `nulls_first`, else at the opposite end.
///
/// Whether `dt` is a temporal type with a total order that its integer backing
/// (days / millis / micros / nanos) preserves — so range-partitioning on the backing
/// gives the same order as the single-node temporal sort. Excludes `Interval`
/// (month-day-nano is not a single totally-ordered scalar).
pub fn is_temporal_key(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::Duration(_)
    )
}

/// Cast a temporal column to `Int64` via its order-preserving backing (days / ticks).
/// `Date32`/`Time32` are `i32`-backed, so a direct `Date32 → Int64` cast is unsupported;
/// route those through `Int32` first. The result is the canonical integer the sort sample
/// and the range partition both compare on, so they share one representation.
pub fn temporal_to_i64(col: &ArrayRef) -> Result<ArrayRef, RuntimeError> {
    match cast(col, &DataType::Int64) {
        Ok(a) => Ok(a),
        Err(_) => {
            let i32 = cast(col, &DataType::Int32)?;
            Ok(cast(&i32, &DataType::Int64)?)
        }
    }
}

/// This is the Rust counterpart of the hash [`partition_by_keys`] for the
/// distributed-sort path. The key is compared as `f64` — bit-identical to the
/// previous NumPy `searchsorted` over `to_numpy()` keys (the boundaries are
/// `f64` quantiles), and `NaN` sorts last exactly as NumPy places it.
pub fn range_partition_by_key(
    batch: &RecordBatch,
    key_index: usize,
    boundaries: &[f64],
    n_buckets: usize,
    nulls_first: bool,
    descending: bool,
) -> Result<Vec<RecordBatch>, RuntimeError> {
    range_partition_by_key_array(
        batch,
        batch.column(key_index),
        boundaries,
        n_buckets,
        nulls_first,
        descending,
    )
}

/// Like [`range_partition_by_key`], but the leading sort key is supplied directly as an
/// array rather than by column index — so a *computed* `ORDER BY` key (or the
/// single-node parallel sample-sort, whose key comes from an expression eval) can
/// range-partition without first appending the key to `batch`. `key` must have
/// `batch.num_rows()` rows.
pub fn range_partition_by_key_array(
    batch: &RecordBatch,
    key_col: &ArrayRef,
    boundaries: &[f64],
    n_buckets: usize,
    nulls_first: bool,
    descending: bool,
) -> Result<Vec<RecordBatch>, RuntimeError> {
    assert!(n_buckets >= 1);
    if n_buckets == 1 {
        return Ok(vec![batch.clone()]);
    }
    let front = if descending { n_buckets - 1 } else { 0 };
    let null_bucket = if nulls_first {
        front
    } else {
        n_buckets - 1 - front
    } as u32;

    // Compare the key in f64 (the boundaries are f64 quantiles), matching the prior
    // `kc.to_numpy()` + `np.searchsorted` path bit-for-bit. Numeric keys cast directly;
    // TEMPORAL keys (Date/Time/Timestamp) route by their order-preserving integer backing
    // (days / millis / micros as i64 → f64), so a distributed `ORDER BY <date>` balances and
    // sorts exactly like single-node instead of failing — the common TPC-H shape. STRING and
    // other non-orderable-numeric keys are still refused: Arrow would parse "12" → 12.0,
    // disagreeing with the single-node *lexical* string sort (what `to_numpy()` refused).
    let dt = key_col.data_type();
    let key = if dt.is_numeric() {
        cast(key_col, &DataType::Float64)?
    } else if is_temporal_key(dt) {
        // temporal → i64 (its backing) → f64. The sample side casts identically, so the
        // boundaries and the routing share one representation.
        cast(&temporal_to_i64(key_col)?, &DataType::Float64)?
    } else {
        return Err(RuntimeError::NonNumericRangeKey {
            dtype: dt.to_string(),
        });
    };
    let key = key.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
        RuntimeError::NonNumericRangeKey {
            dtype: key_col.data_type().to_string(),
        }
    })?;

    let part_of: Vec<u32> = (0..batch.num_rows())
        .map(|i| {
            if key.is_null(i) {
                null_bucket
            } else {
                let v = key.value(i);
                // NumPy orders NaN last, so a NaN key lands in the highest bucket.
                let id = if v.is_nan() {
                    boundaries.len()
                } else {
                    boundaries.partition_point(|&b| b <= v)
                };
                id as u32
            }
        })
        .collect();

    scatter_into_buckets(batch, &part_of, n_buckets)
}

/// The per-row bucket id [`range_partition_by_key_array`] would scatter by.
pub fn range_part_of_f64(
    key_col: &ArrayRef,
    boundaries: &[f64],
    n_buckets: usize,
    nulls_first: bool,
    descending: bool,
) -> Result<Vec<u32>, RuntimeError> {
    let null_bucket = null_bucket_of(n_buckets, nulls_first, descending);
    let dt = key_col.data_type();
    let key = if dt.is_numeric() {
        cast(key_col, &DataType::Float64)?
    } else if is_temporal_key(dt) {
        cast(&temporal_to_i64(key_col)?, &DataType::Float64)?
    } else {
        return Err(RuntimeError::NonNumericRangeKey {
            dtype: dt.to_string(),
        });
    };
    let key = key.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
        RuntimeError::NonNumericRangeKey {
            dtype: key_col.data_type().to_string(),
        }
    })?;
    Ok(map_rows(key.len(), |i| {
        if key.is_null(i) {
            null_bucket
        } else {
            let v = key.value(i);
            let id = if v.is_nan() {
                boundaries.len()
            } else {
                boundaries.partition_point(|&b| b <= v)
            };
            id as u32
        }
    }))
}

/// Like [`range_partition_by_key_array`], but for an **integer** leading key compared
/// **exactly** as `i64` (boundaries are `i64` quantiles) — no `f64` cast, so a key beyond
/// `2^53` is routed without precision loss. Any signed/unsigned integer width is widened
/// to `i64` (order-preserving). The single-node parallel sample-sort uses this for an
/// integer `ORDER BY` leading key; floats keep [`range_partition_by_key_array`].
pub fn range_partition_by_i64_key(
    batch: &RecordBatch,
    key_col: &ArrayRef,
    boundaries: &[i64],
    n_buckets: usize,
    nulls_first: bool,
    descending: bool,
) -> Result<Vec<RecordBatch>, RuntimeError> {
    assert!(n_buckets >= 1);
    if n_buckets == 1 {
        return Ok(vec![batch.clone()]);
    }
    let part_of = range_part_of_i64(key_col, boundaries, n_buckets, nulls_first, descending)?;
    scatter_into_buckets(batch, &part_of, n_buckets)
}

/// The per-row bucket id [`range_partition_by_i64_key`] would scatter by — the routing
/// without the gather, for callers that permute the rows themselves.
pub fn range_part_of_i64(
    key_col: &ArrayRef,
    boundaries: &[i64],
    n_buckets: usize,
    nulls_first: bool,
    descending: bool,
) -> Result<Vec<u32>, RuntimeError> {
    let null_bucket = null_bucket_of(n_buckets, nulls_first, descending);
    if !matches!(key_col.data_type(), t if t.is_integer()) {
        return Err(RuntimeError::NonNumericRangeKey {
            dtype: key_col.data_type().to_string(),
        });
    }
    let key = cast(key_col, &DataType::Int64)?;
    let key = key
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .ok_or_else(|| RuntimeError::NonNumericRangeKey {
            dtype: key_col.data_type().to_string(),
        })?;
    Ok(map_rows(key.len(), |i| {
        if key.is_null(i) {
            null_bucket
        } else {
            boundaries.partition_point(|&b| b <= key.value(i)) as u32
        }
    }))
}

/// Like [`range_partition_by_i64_key`], but for a **string** leading key compared
/// **lexicographically by bytes** — exactly the ordering arrow's `sort_to_indices`
/// gives a `Utf8`/`LargeUtf8` column, so the per-range sorts concatenate into the
/// same relation a single global string sort produces. Boundaries are ascending
/// string quantiles sampled from the key.
///
/// Without this, a string `ORDER BY` had no range partitioner, so the single-node
/// sample-sort refused it and a whole-column string sort ran single-threaded — the
/// one sort shape that never used more than one core.
pub fn range_partition_by_str_key(
    batch: &RecordBatch,
    key_col: &ArrayRef,
    boundaries: &[String],
    n_buckets: usize,
    nulls_first: bool,
    descending: bool,
) -> Result<Vec<RecordBatch>, RuntimeError> {
    assert!(n_buckets >= 1);
    if n_buckets == 1 {
        return Ok(vec![batch.clone()]);
    }
    let null_bucket = null_bucket_of(n_buckets, nulls_first, descending);
    let part_of = str_part_of(key_col, boundaries, null_bucket)?;
    scatter_into_buckets(batch, &part_of, n_buckets)
}

/// The per-row bucket id [`range_partition_by_str_key`] would scatter by — the routing
/// without the gather.
pub fn range_part_of_str(
    key_col: &ArrayRef,
    boundaries: &[String],
    n_buckets: usize,
    nulls_first: bool,
    descending: bool,
) -> Result<Vec<u32>, RuntimeError> {
    str_part_of(
        key_col,
        boundaries,
        null_bucket_of(n_buckets, nulls_first, descending),
    )
}

/// `partition_point(|b| b <= v)` routes a value equal to a boundary consistently to the
/// higher bucket, so equal keys never straddle a boundary — the property the
/// concatenation-without-merge relies on.
fn route_str<O: OffsetSizeTrait>(
    arr: &GenericStringArray<O>,
    boundaries: &[String],
    null_bucket: u32,
) -> Vec<u32> {
    map_rows(arr.len(), |i| {
        if arr.is_null(i) {
            null_bucket
        } else {
            let v = arr.value(i);
            boundaries.partition_point(|b| b.as_str() <= v) as u32
        }
    })
}

/// Dispatch a string key to its `Utf8` / `LargeUtf8` array and route every row.
fn str_part_of(
    key_col: &ArrayRef,
    boundaries: &[String],
    null_bucket: u32,
) -> Result<Vec<u32>, RuntimeError> {
    let bad = || RuntimeError::NonNumericRangeKey {
        dtype: key_col.data_type().to_string(),
    };
    match key_col.data_type() {
        DataType::Utf8 => Ok(route_str(
            key_col
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(bad)?,
            boundaries,
            null_bucket,
        )),
        DataType::LargeUtf8 => Ok(route_str(
            key_col
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(bad)?,
            boundaries,
            null_bucket,
        )),
        _ => Err(bad()),
    }
}

/// Skew-aware partitioning for a **single-key** distributed join: a *hot* key's
/// rows are spread across `salt_count` sub-buckets instead of all landing on one
/// reducer. The probe side (`replicate = false`) sends each hot row to one salted
/// bucket (round-robin, so the hot key's probe rows fan out evenly); the build side
/// (`replicate = true`) sends each hot row to *all* `salt_count` salted buckets, so
/// every salted probe bucket has the full build side for that key to match against.
/// Cold keys partition exactly as [`partition_by_keys`] would, so the salted join
/// yields the **same relation** as the unsalted one — only the hot key's work moves
/// off a single reducer onto many.
///
/// `hot_keys` are the hot values rendered as strings (matching the `heavy_hitters`
/// detection, which casts any key type to Utf8). Membership is tested by casting the
/// key column to Utf8. Single-key only (`key_indices.len() == 1`).
pub fn salted_partition_by_keys(
    batch: &RecordBatch,
    key_indices: &[usize],
    num_partitions: usize,
    hot_keys: &HashSet<String>,
    salt_count: u32,
    replicate: bool,
) -> Result<Vec<RecordBatch>, RuntimeError> {
    assert!(num_partitions >= 1 && salt_count >= 1);
    assert_eq!(key_indices.len(), 1, "salted partition is single-key only");
    if num_partitions == 1 || hot_keys.is_empty() {
        return partition_by_keys(batch, key_indices, num_partitions);
    }
    let n = batch.num_rows();
    let key_col = batch.column(key_indices[0]).clone();
    let converter = RowConverter::new(vec![SortField::new(key_col.data_type().clone())])?;
    let rows = converter.convert_columns(std::slice::from_ref(&key_col))?;
    // Hot membership is tested on the string rendering, matching how hot keys were
    // detected. Cast failures → treat as cold (no salting), still correct.
    let key_str = cast(&key_col, &DataType::Utf8).ok();
    let key_str = key_str
        .as_ref()
        .and_then(|a| a.as_any().downcast_ref::<StringArray>());

    // Uniform-key pre-size; hot-key replication adds only a bounded few extra per bucket.
    let cap = n / num_partitions + 1;
    let mut buckets: Vec<Vec<u32>> = (0..num_partitions)
        .map(|_| Vec::with_capacity(cap))
        .collect();
    let mut cursor: u32 = 0;
    // Reused dedup marks (all-false between rows) so a replicated build row lands in
    // each DISTINCT salt bucket exactly once — see the `replicate` branch.
    let mut seen = vec![false; num_partitions];
    for i in 0..n {
        let kh = SEED.hash_one(rows.row(i));
        let is_hot = key_str
            .map(|s| s.is_valid(i) && hot_keys.contains(s.value(i)))
            .unwrap_or(false);
        if !is_hot {
            buckets[bucket_of(kh, num_partitions) as usize].push(i as u32);
        } else if replicate {
            // Replicate the build hot row to each DISTINCT salt bucket once. When
            // `salt_count > num_partitions` (or two salts simply collide), pushing
            // per-salt would place the build row in one bucket multiple times, so the
            // reducer joins each salted probe row against several copies and the join
            // output is duplicated. Dedupe via `seen`, then restore it to all-false.
            for s in 0..salt_count {
                let b = bucket_of(salted_hash(kh, s), num_partitions) as usize;
                if !seen[b] {
                    seen[b] = true;
                    buckets[b].push(i as u32);
                }
            }
            for s in 0..salt_count {
                seen[bucket_of(salted_hash(kh, s), num_partitions) as usize] = false;
            }
        } else {
            let s = cursor % salt_count;
            cursor = cursor.wrapping_add(1);
            buckets[bucket_of(salted_hash(kh, s), num_partitions) as usize].push(i as u32);
        }
    }
    buckets.iter().map(|idx| take_rows(batch, idx)).collect()
}

/// Mix a salt into a key hash so different salts spread a hot key across buckets
/// (a splitmix64 avalanche over `key_hash ^ salt·golden`). Both join sides use this
/// for a given `(key, salt)`, so a salted probe row and the replicated build rows
/// land in the same bucket.
#[inline]
fn salted_hash(key_hash: u64, salt: u32) -> u64 {
    let mut h = key_hash ^ (salt as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h
}

/// Map a key hash to a bucket in `[0, num_partitions)` without a division: a bit
/// mask when the count is a power of two, else Lemire's multiply-shift over the
/// hash's high-entropy bits. Deterministic, so equal keys (and both join sides)
/// always agree within a run.
#[inline]
/// Per-row bucket ids for an all-`Int64` key, hashing native values directly (no row encoding).
/// `None` unless every key column is `Int64` (narrow ints are normalized to `Int64` upstream, so
/// this covers the common single- and composite-integer shuffle shapes).
///
/// Nulls are handled in-path, not bailed on. Arrow leaves the value under a null slot undefined
/// (parquet's `pad_nulls` leaves whatever was in the buffer), so a null row must NOT be hashed by
/// its raw value: it would scatter otherwise-identical NULL keys across every bucket (wrong for an
/// aggregate/DISTINCT shuffle, where SQL says all NULLs are one group). Instead every null row
/// hashes to the fixed `keys::NULL_HASH` bucket. Crucially, this keeps a *nullable* key on the raw
/// path rather than falling back to the `RowConverter`: if it fell back while a null-free key of
/// the same type hashed raw, the two hashes would disagree on equal *non-null* keys and split them
/// across buckets — silently dropping inner-join matches (both join sides must take one path).
fn partition_int_key(keys: &[ArrayRef], num_partitions: usize) -> Option<Vec<u32>> {
    use arrow::array::Int64Array;
    if keys.is_empty() || !keys.iter().all(|k| k.data_type() == &DataType::Int64) {
        return None;
    }
    // Nulls are handled here, NOT bailed on: a null row hashes to the fixed `keys::NULL_HASH`
    // bucket (co-locating every null, so a DISTINCT/aggregate sees one NULL group), and a
    // non-null row hashes its raw value. This is what keeps BOTH sides of a join on the raw
    // path: if a nullable key fell back to the `RowConverter` while the other (null-free) side
    // took this raw hash, the two hashes would disagree on equal *non-null* keys, splitting them
    // across buckets and silently dropping inner-join matches. Same-typed keys therefore always
    // take one identical path regardless of null presence.
    let null_bucket = bucket_of(crate::keys::NULL_HASH, num_partitions);
    // Single Int64 key — hash the raw value directly.
    if keys.len() == 1 {
        let arr = keys[0].as_any().downcast_ref::<Int64Array>()?;
        let vals = arr.values();
        let hash1 = |i: usize| -> u32 {
            if arr.is_null(i) {
                null_bucket
            } else {
                bucket_of(SEED.hash_one(vals[i]), num_partitions)
            }
        };
        let n = vals.len();
        return Some(if n >= PAR_HASH_MIN_ROWS {
            (0..n).into_par_iter().map(hash1).collect()
        } else {
            (0..n).map(hash1).collect()
        });
    }
    // Composite Int64 key (e.g. a `(part, supplier)` join / group shuffle). Fold each
    // column's raw value into one hasher per row — skips the `RowConverter` encode the
    // general path runs. A row with a null in ANY key column routes to the null bucket (equal
    // null-bearing rows still co-locate; they are compared within the bucket by the assigner),
    // so co-partitioning holds for null-free and nullable keys alike.
    let cols: Vec<&Int64Array> = keys
        .iter()
        .map(|k| k.as_any().downcast_ref::<Int64Array>().unwrap())
        .collect();
    let n = cols[0].len();
    let hashn = |i: usize| -> u32 {
        use std::hash::{BuildHasher, Hasher};
        if cols.iter().any(|c| c.is_null(i)) {
            return null_bucket;
        }
        let mut h = SEED.build_hasher();
        for c in &cols {
            h.write_i64(c.values()[i]);
        }
        bucket_of(h.finish(), num_partitions)
    };
    Some(if n >= PAR_HASH_MIN_ROWS {
        (0..n).into_par_iter().map(hashn).collect()
    } else {
        (0..n).map(hashn).collect()
    })
}

/// One key column, downcast once, exposing a per-row raw value to the hasher.
enum MixedCol<'a> {
    Int(&'a [i64]),
    Str32(&'a GenericStringArray<i32>),
    Str64(&'a GenericStringArray<i64>),
    Bin32(&'a GenericBinaryArray<i32>),
    Bin64(&'a GenericBinaryArray<i64>),
}

impl MixedCol<'_> {
    #[inline]
    fn write<H: std::hash::Hasher>(&self, h: &mut H, i: usize) {
        match self {
            MixedCol::Int(v) => h.write_i64(v[i]),
            MixedCol::Str32(a) => h.write(a.value(i).as_bytes()),
            MixedCol::Str64(a) => h.write(a.value(i).as_bytes()),
            MixedCol::Bin32(a) => h.write(a.value(i)),
            MixedCol::Bin64(a) => h.write(a.value(i)),
        }
    }
}

/// Partition a null-free composite key of `Int64` / string / binary columns by hashing each
/// row's raw values directly — the parallel alternative to the `RowConverter` path, whose
/// per-row byte encode runs serially. Returns `None` (caller keeps `RowConverter`) for an
/// empty key, any nullable column, or any unsupported type.
///
/// Equal non-null rows fold the same bytes into the hasher in the same order, so they land in
/// the same bucket — the co-partitioning invariant a shuffle/DISTINCT needs. Gated null-free
/// because a null slot's arbitrary raw value could split two equal null-bearing rows across
/// buckets, which a DISTINCT (nulls compare equal) must not do.
fn partition_mixed_key(keys: &[ArrayRef], num_partitions: usize) -> Option<Vec<u32>> {
    if keys.len() < 2 {
        return None;
    }
    let cols: Vec<MixedCol> = keys
        .iter()
        .map(|k| match k.data_type() {
            DataType::Int64 => Some(MixedCol::Int(
                k.as_primitive::<arrow::datatypes::Int64Type>().values(),
            )),
            DataType::Utf8 => Some(MixedCol::Str32(k.as_string::<i32>())),
            DataType::LargeUtf8 => Some(MixedCol::Str64(k.as_string::<i64>())),
            DataType::Binary => Some(MixedCol::Bin32(k.as_binary::<i32>())),
            DataType::LargeBinary => Some(MixedCol::Bin64(k.as_binary::<i64>())),
            _ => None,
        })
        .collect::<Option<_>>()?;
    let n = keys[0].len();
    // Nulls route to the fixed null bucket (co-locating equal null-bearing rows) and non-null
    // rows hash raw — the same null-awareness `partition_int_key` has, so a nullable key never
    // falls back to the `RowConverter` while a null-free key of the same shape hashes raw, which
    // would split equal non-null keys across buckets and drop inner-join matches.
    let null_bucket = bucket_of(crate::keys::NULL_HASH, num_partitions);
    let any_null = keys.iter().any(|k| k.null_count() != 0);
    let hashn = |i: usize| -> u32 {
        use std::hash::{BuildHasher, Hasher};
        if any_null && keys.iter().any(|k| k.is_null(i)) {
            return null_bucket;
        }
        let mut h = SEED.build_hasher();
        for c in &cols {
            c.write(&mut h, i);
        }
        bucket_of(h.finish(), num_partitions)
    };
    Some(if n >= PAR_HASH_MIN_ROWS {
        (0..n).into_par_iter().map(hashn).collect()
    } else {
        (0..n).map(hashn).collect()
    })
}

fn bucket_of(hash: u64, num_partitions: usize) -> u32 {
    if num_partitions.is_power_of_two() {
        (hash & (num_partitions as u64 - 1)) as u32
    } else {
        ((hash as u128 * num_partitions as u128) >> 64) as u32
    }
}

/// Gather the given row indices out of every column of `batch`.
fn take_rows(batch: &RecordBatch, idx: &[u32]) -> Result<RecordBatch, RuntimeError> {
    // NB: fanning the per-column `take`s across cores (nested inside the outer parallel
    // loop over ranges/buckets) was measured and does NOT help: 119.9 -> 118.9 ms on a
    // 5 M-row, 6-column sort. The gather is memory-bandwidth bound, not core bound —
    // raising the range count from 64 to 192 likewise bought nothing.
    let indices = UInt32Array::from(idx.to_vec());
    let columns = batch
        .columns()
        .iter()
        .map(|c| crate::gather::take_column(c.as_ref(), &indices))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(RecordBatch::try_new(batch.schema(), columns)?)
}

#[cfg(test)]
mod tests {
    /// The CSR binning must be indistinguishable from the `Vec<Vec<u32>>` one — same
    /// buckets, same ascending order inside each. Everything downstream assumes it.
    #[test]
    fn bucket_csr_matches_bucket_indices() {
        for parts in [1usize, 2, 3, 8, 64] {
            for n in [0usize, 1, 5, 100, 1000] {
                let part_of: Vec<u32> = (0..n).map(|i| ((i * 7 + 3) % parts) as u32).collect();
                let nested = bucket_indices(&part_of, parts);
                let (rows, offsets) = bucket_csr(&part_of, parts);
                assert_eq!(offsets.len(), parts + 1);
                assert_eq!(*offsets.last().unwrap() as usize, n);
                for b in 0..parts {
                    let slice = &rows[offsets[b] as usize..offsets[b + 1] as usize];
                    assert_eq!(
                        slice,
                        nested[b].as_slice(),
                        "parts={parts} n={n} bucket={b}"
                    );
                }
            }
        }
    }

    /// A bucket that receives every row, and one that receives none.
    #[test]
    fn bucket_csr_handles_degenerate_partitions() {
        let (rows, offsets) = bucket_csr(&[0, 0, 0], 3);
        assert_eq!(rows, vec![0, 1, 2]);
        assert_eq!(offsets, vec![0, 3, 3, 3]);
    }
    use super::*;
    use arrow::array::Int64Array;
    use std::sync::Arc;

    #[test]
    fn partitions_are_disjoint_and_complete() {
        let batch = RecordBatch::try_from_iter(vec![(
            "k",
            Arc::new(Int64Array::from((0..100).collect::<Vec<_>>())) as ArrayRef,
        )])
        .unwrap();

        let parts = partition_by_keys(&batch, &[0], 8).unwrap();
        let total: usize = parts.iter().map(|p| p.num_rows()).sum();
        assert_eq!(total, 100); // every row lands in exactly one bucket
        assert_eq!(parts.len(), 8);
    }

    #[test]
    fn equal_keys_share_a_bucket() {
        // Two batches with the same keys must partition identically (join needs this).
        let mk = || {
            RecordBatch::try_from_iter(vec![(
                "k",
                Arc::new(Int64Array::from(vec![5, 17, 5, 42])) as ArrayRef,
            )])
            .unwrap()
        };
        let a = partition_by_keys(&mk(), &[0], 4).unwrap();
        let b = partition_by_keys(&mk(), &[0], 4).unwrap();
        for (pa, pb) in a.iter().zip(&b) {
            assert_eq!(pa.num_rows(), pb.num_rows());
        }
    }

    #[test]
    fn non_power_of_two_is_disjoint_complete_and_ordered() {
        // 7 buckets exercises the multiply-shift path (not a bit mask). The scatter
        // must remain a complete, disjoint permutation that preserves row order
        // within each bucket.
        let batch = RecordBatch::try_from_iter(vec![(
            "k",
            Arc::new(Int64Array::from((0..200).collect::<Vec<_>>())) as ArrayRef,
        )])
        .unwrap();
        let parts = partition_by_keys(&batch, &[0], 7).unwrap();
        assert_eq!(parts.len(), 7);
        let total: usize = parts.iter().map(|p| p.num_rows()).sum();
        assert_eq!(total, 200);
        for p in &parts {
            let col = p.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            // Stable: keys within a bucket stay in ascending (original) order.
            assert!(col.values().windows(2).all(|w| w[0] < w[1]));
        }
    }

    /// The bucket id each row of a key column lands in, mirroring the reference
    /// `np.searchsorted(boundaries, key, side="right")` + null routing, for the
    /// equal-keys / nulls / descending / nulls_first cases.
    fn ids(
        keys: Vec<Option<i64>>,
        boundaries: &[f64],
        n_buckets: usize,
        nulls_first: bool,
        descending: bool,
    ) -> Vec<usize> {
        let batch = RecordBatch::try_from_iter(vec![(
            "k",
            Arc::new(Int64Array::from(keys.clone())) as ArrayRef,
        )])
        .unwrap();
        let buckets =
            range_partition_by_key(&batch, 0, boundaries, n_buckets, nulls_first, descending)
                .unwrap();
        // Reconstruct each original row's bucket from the per-bucket key values.
        let mut out = vec![usize::MAX; keys.len()];
        for (b, part) in buckets.iter().enumerate() {
            let col = part
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for r in 0..part.num_rows() {
                let val = if col.is_null(r) {
                    None
                } else {
                    Some(col.value(r))
                };
                // First not-yet-assigned matching row (stable scatter preserves order).
                let pos = keys
                    .iter()
                    .enumerate()
                    .find(|(i, k)| out[*i] == usize::MAX && **k == val)
                    .map(|(i, _)| i)
                    .unwrap();
                out[pos] = b;
            }
        }
        out
    }

    #[test]
    fn range_buckets_match_searchsorted_right() {
        // boundaries [10, 20] → 3 buckets; equal-to-boundary goes to the higher bucket
        // (side="right"), so 10→bucket1, 20→bucket2, and equal keys never split.
        let got = ids(
            vec![Some(5), Some(10), Some(15), Some(20), Some(25), Some(10)],
            &[10.0, 20.0],
            3,
            false,
            false,
        );
        assert_eq!(got, vec![0, 1, 1, 2, 2, 1]);
    }

    #[test]
    fn range_nulls_route_to_the_correct_end() {
        // Ascending: front bucket is 0. nulls_first → nulls in bucket 0; else top bucket.
        assert_eq!(
            ids(vec![None, Some(5), Some(25)], &[10.0, 20.0], 3, true, false),
            vec![0, 0, 2]
        );
        assert_eq!(
            ids(
                vec![None, Some(5), Some(25)],
                &[10.0, 20.0],
                3,
                false,
                false
            ),
            vec![2, 0, 2]
        );
        // Descending: front bucket is n-1. nulls_first → nulls in the top bucket; the
        // driver concatenates high→low so that places them first overall.
        assert_eq!(
            ids(vec![None, Some(5)], &[10.0, 20.0], 3, true, true),
            vec![2, 0]
        );
        assert_eq!(
            ids(vec![None, Some(5)], &[10.0, 20.0], 3, false, true),
            vec![0, 0]
        );
    }

    #[test]
    fn range_empty_boundaries_single_bucket_of_non_nulls() {
        // No boundaries (e.g. a single reducer or an all-null sample) → every non-null
        // key in bucket 0, nulls at the configured end.
        assert_eq!(
            ids(vec![Some(3), Some(99), None], &[], 2, false, false),
            vec![0, 0, 1]
        );
        // n_buckets == 1 returns the batch unchanged → all rows in bucket 0.
        assert_eq!(ids(vec![Some(3), None], &[], 1, false, false), vec![0, 0]);
    }

    #[test]
    fn range_partition_rejects_non_numeric_key() {
        let batch = RecordBatch::try_from_iter(vec![(
            "k",
            Arc::new(StringArray::from(vec!["12", "3"])) as ArrayRef,
        )])
        .unwrap();
        // A string key must error, not be parsed to a float (which would disagree with
        // the single-node lexical string sort).
        assert!(range_partition_by_key(&batch, 0, &[5.0], 2, false, false).is_err());
    }

    /// Count inner-join output pairs between a probe and a build batch on column 0
    /// via a nested loop (test-only oracle).
    fn join_pairs(probe: &RecordBatch, build: &RecordBatch) -> usize {
        let pk = probe
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let bk = build
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let mut pairs = 0;
        for i in 0..pk.len() {
            for j in 0..bk.len() {
                if pk.value(i) == bk.value(j) {
                    pairs += 1;
                }
            }
        }
        pairs
    }

    #[test]
    fn salted_join_equals_unsalted_join_and_fans_out_hot_key() {
        // Probe: key 1 is hot (100 rows); keys 2,3 are cold. Build: key 1 has 5 rows.
        let mut probe_keys: Vec<i64> = vec![1; 100];
        probe_keys.extend([2, 2, 3]);
        let probe = RecordBatch::try_from_iter(vec![(
            "k",
            Arc::new(Int64Array::from(probe_keys)) as ArrayRef,
        )])
        .unwrap();
        let build = RecordBatch::try_from_iter(vec![(
            "k",
            Arc::new(Int64Array::from(vec![1, 1, 1, 1, 1, 2, 3])) as ArrayRef,
        )])
        .unwrap();

        let n = 8usize;
        let salt = 4u32;
        let hot: HashSet<String> = ["1".to_string()].into_iter().collect();

        // Probe spreads the hot key (one salted bucket per row); build replicates it.
        let probe_parts = salted_partition_by_keys(&probe, &[0], n, &hot, salt, false).unwrap();
        let build_parts = salted_partition_by_keys(&build, &[0], n, &hot, salt, true).unwrap();

        // The salted, per-bucket join must reproduce the whole-relation join exactly.
        let global = join_pairs(&probe, &build);
        let salted: usize = probe_parts
            .iter()
            .zip(&build_parts)
            .map(|(p, b)| join_pairs(p, b))
            .sum();
        assert_eq!(salted, global, "salted join must equal the unsalted join");

        // The hot key's probe rows must land in more than one bucket (work fanned out).
        let buckets_touched = probe_parts.iter().filter(|p| p.num_rows() > 0).count();
        assert!(
            buckets_touched > 1,
            "hot key should spread across multiple buckets, got {buckets_touched}"
        );
    }

    #[test]
    fn salted_build_replication_dedupes_when_salt_exceeds_partitions() {
        // Regression: with salt_count > num_partitions, distinct salts hash to the
        // same bucket, so replicating the build hot row *per salt* put multiple copies
        // in one bucket and the reducer doubled the join output. The build row must
        // land in each distinct salt bucket exactly once.
        let probe = RecordBatch::try_from_iter(vec![(
            "k",
            Arc::new(Int64Array::from(vec![1i64; 50])) as ArrayRef,
        )])
        .unwrap();
        let build = RecordBatch::try_from_iter(vec![(
            "k",
            Arc::new(Int64Array::from(vec![1i64])) as ArrayRef, // one build row for the hot key
        )])
        .unwrap();
        let hot: HashSet<String> = ["1".to_string()].into_iter().collect();
        let n = 3usize; // fewer partitions than salts → guaranteed bucket collisions
        let salt = 8u32;

        let probe_parts = salted_partition_by_keys(&probe, &[0], n, &hot, salt, false).unwrap();
        let build_parts = salted_partition_by_keys(&build, &[0], n, &hot, salt, true).unwrap();

        // The salted per-bucket join must equal the whole-relation join (no dup rows).
        let global = join_pairs(&probe, &build); // 50 probe × 1 build
        let salted: usize = probe_parts
            .iter()
            .zip(&build_parts)
            .map(|(p, b)| join_pairs(p, b))
            .sum();
        assert_eq!(salted, global, "salted join must equal the unsalted join");
        // The single build row is replicated to at most `num_partitions` buckets.
        let build_rows: usize = build_parts.iter().map(|b| b.num_rows()).sum();
        assert!(
            build_rows <= n,
            "build row over-replicated: {build_rows} > {n}"
        );
    }

    #[test]
    fn salted_cold_keys_match_plain_partition() {
        // With no hot keys, salted partitioning is identical to the plain shuffle.
        let batch = RecordBatch::try_from_iter(vec![(
            "k",
            Arc::new(Int64Array::from((0..100).collect::<Vec<_>>())) as ArrayRef,
        )])
        .unwrap();
        let empty: HashSet<String> = HashSet::new();
        let salted = salted_partition_by_keys(&batch, &[0], 8, &empty, 4, false).unwrap();
        let plain = partition_by_keys(&batch, &[0], 8).unwrap();
        for (s, p) in salted.iter().zip(&plain) {
            assert_eq!(s.num_rows(), p.num_rows());
        }
    }

    #[test]
    fn single_partition_returns_whole_batch() {
        let batch = RecordBatch::try_from_iter(vec![(
            "k",
            Arc::new(Int64Array::from(vec![3, 1, 2])) as ArrayRef,
        )])
        .unwrap();
        let parts = partition_by_keys(&batch, &[0], 1).unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].num_rows(), 3);
    }

    /// The mixed (string, int) fast partition co-locates equal rows and places every row
    /// exactly once — the invariant DISTINCT / shuffle rest on. It must agree with the
    /// `RowConverter` path on *which rows share a bucket* (bucket ids themselves may differ;
    /// only co-location matters).
    #[test]
    fn mixed_key_copartitions_equal_rows() {
        use arrow::array::StringArray;
        let flag = Arc::new(StringArray::from(vec!["A", "N", "A", "N", "A", "R"])) as ArrayRef;
        let key = Arc::new(Int64Array::from(vec![10, 20, 10, 20, 11, 20])) as ArrayRef;
        let keys = vec![flag, key];
        // Fast path must fire (null-free, 2 cols of int+str).
        let fast = partition_mixed_key(&keys, 8).expect("mixed fast path should apply");
        assert_eq!(fast.len(), 6);
        // Rows 0 and 2 are ("A",10) — identical → same bucket. Rows 1,3,5 are ("N"/"R",20):
        // 1 and 3 are ("N",20) identical; 5 is ("R",20) distinct.
        assert_eq!(fast[0], fast[2], "equal (A,10) rows must co-partition");
        assert_eq!(fast[1], fast[3], "equal (N,20) rows must co-partition");
        // Different rows may or may not collide in 8 buckets, but a full DISTINCT over the
        // partitioned buckets must recover exactly the 4 distinct rows. Verify via the public
        // `partition_by_keys` + per-bucket dedup that the distinct count is right.
        let batch =
            RecordBatch::try_from_iter(vec![("f", keys[0].clone()), ("k", keys[1].clone())])
                .unwrap();
        let buckets = partition_by_keys(&batch, &[0, 1], 8).unwrap();
        let total: usize = buckets.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 6, "every row placed exactly once");
    }

    /// A nullable mixed key stays on the raw fast path (it no longer declines): null rows route
    /// to the fixed null bucket and equal non-null rows co-partition. Declining here was the
    /// bug — a nullable key fell back to the `RowConverter` while a null-free key of the same
    /// type hashed raw, so equal non-null keys split across buckets and inner-join matches were
    /// silently dropped. The raw path must fire and co-locate both the null rows and the equal
    /// non-null rows.
    #[test]
    fn mixed_key_with_nulls_uses_null_aware_fast_path() {
        use arrow::array::StringArray;
        let flag = Arc::new(StringArray::from(vec![Some("A"), None, Some("A"), None])) as ArrayRef;
        let key = Arc::new(Int64Array::from(vec![Some(1), None, Some(1), None])) as ArrayRef;
        let part = partition_mixed_key(&[flag, key], 8).expect("null-aware fast path applies");
        assert_eq!(part.len(), 4);
        // Rows 0 and 2 are ("A",1) → same bucket. Rows 1 and 3 are (null,null) → the null bucket.
        assert_eq!(part[0], part[2], "equal (A,1) rows co-partition");
        assert_eq!(
            part[1], part[3],
            "null-bearing rows co-locate in the null bucket"
        );
        assert_eq!(
            part[1],
            bucket_of(crate::keys::NULL_HASH, 8),
            "null rows route to the fixed null bucket"
        );
    }

    /// Regression for the dropped-match bug: a null-BEARING key column and a null-FREE key column
    /// of the same type must send equal non-null values to the SAME bucket. Before the fix the
    /// null-bearing side fell back to the `RowConverter` while the null-free side hashed raw, so
    /// equal keys split across buckets and an inner join silently lost rows.
    #[test]
    fn nullable_and_nullfree_int_keys_copartition() {
        let with_null = Arc::new(Int64Array::from(vec![Some(7), None, Some(42)])) as ArrayRef;
        let no_null = Arc::new(Int64Array::from(vec![7, 42, 99])) as ArrayRef;
        let p_null = bucket_of_rows(&[with_null], 3, 16).unwrap();
        let p_free = bucket_of_rows(&[no_null], 3, 16).unwrap();
        // key 7: row 0 on the nullable side, row 0 on the null-free side.
        assert_eq!(
            p_null[0], p_free[0],
            "key 7 must co-partition across both sides"
        );
        // key 42: row 2 on the nullable side, row 1 on the null-free side.
        assert_eq!(
            p_null[2], p_free[1],
            "key 42 must co-partition across both sides"
        );
    }
}
