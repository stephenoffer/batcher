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

use arrow::array::{Array, ArrayRef, RecordBatch, StringArray, UInt32Array};
use arrow::compute::{cast, take};
use arrow::datatypes::DataType;
use arrow::row::{RowConverter, SortField};

use crate::error::RuntimeError;

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
    let n = batch.num_rows();

    // One hash pass → the bucket id per row. An empty key set routes every row to
    // bucket 0 (hashing an empty row is ill-defined).
    let part_of: Vec<u32> = if keys.is_empty() {
        vec![0u32; n]
    } else {
        let fields: Vec<SortField> = keys
            .iter()
            .map(|a| SortField::new(a.data_type().clone()))
            .collect();
        let converter = RowConverter::new(fields)?;
        let rows = converter.convert_columns(keys)?;
        (0..n)
            .map(|i| bucket_of(SEED.hash_one(rows.row(i)), num_partitions))
            .collect()
    };

    // Counting-sort partition: histogram → prefix-sum offsets → stable scatter into
    // one contiguous index buffer. This avoids the per-bucket `Vec` reallocation the
    // naive push-per-row layout incurs, and each bucket ends up a contiguous slice.
    let mut offsets = vec![0u32; num_partitions + 1];
    for &b in &part_of {
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

    (0..num_partitions)
        .map(|b| {
            take_rows(
                batch,
                &scatter[offsets[b] as usize..offsets[b + 1] as usize],
            )
        })
        .collect()
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

    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); num_partitions];
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
fn bucket_of(hash: u64, num_partitions: usize) -> u32 {
    if num_partitions.is_power_of_two() {
        (hash & (num_partitions as u64 - 1)) as u32
    } else {
        ((hash as u128 * num_partitions as u128) >> 64) as u32
    }
}

/// Gather the given row indices out of every column of `batch`.
fn take_rows(batch: &RecordBatch, idx: &[u32]) -> Result<RecordBatch, RuntimeError> {
    let indices = UInt32Array::from(idx.to_vec());
    let columns = batch
        .columns()
        .iter()
        .map(|c| take(c.as_ref(), &indices, None))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(RecordBatch::try_new(batch.schema(), columns)?)
}

#[cfg(test)]
mod tests {
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
}
