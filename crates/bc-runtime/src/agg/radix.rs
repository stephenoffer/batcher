//! Parallel hash-radix grouping for the large-input `combine` regroup.
//!
//! Split out of `agg` along the parallel-grouping seam: the serial `assign_groups`
//! stays in the parent (the per-morsel hot path and the correctness reference),
//! while this is the high-cardinality `combine` fast path that the executor reaches
//! once the concatenated input crosses the radix-parallel threshold.

use arrow::array::ArrayRef;
use arrow::row::{RowConverter, SortField};
use hashbrown::hash_table::Entry;
use hashbrown::HashTable;

use crate::error::RuntimeError;

/// Parallel `assign_groups` via hash-radix partitioning — for the large concatenated
/// input `combine` builds on a high-cardinality group-by/distinct, where the serial
/// single-thread grouping dominates.
///
/// Each row's group is fixed by its key encoding, so all rows of a group hash equal
/// and land in the same partition; partitions are then grouped independently across
/// threads with **no cross-partition merge**. The result is identical to the serial
/// path (group *order* differs, which callers already treat as unspecified, like any
/// hash aggregate).
pub(super) fn assign_groups_radix(
    group_keys: &[ArrayRef],
    num_rows: usize,
    partitions: usize,
) -> Result<(Vec<u32>, usize, Vec<ArrayRef>), RuntimeError> {
    use rayon::prelude::*;

    let fields: Vec<SortField> = group_keys
        .iter()
        .map(|a| SortField::new(a.data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields)?;
    let rows = converter.convert_columns(group_keys)?;
    let state = ahash::RandomState::with_seeds(0x9E37, 0x79B9, 0x7F4A, 0x7C15);

    // Hash every row once (vectorized), then bin row indices by `hash % partitions`.
    let hashes: Vec<u64> = (0..num_rows)
        .into_par_iter()
        .map(|i| state.hash_one(rows.row(i)))
        .collect();
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); partitions];
    for (i, &h) in hashes.iter().enumerate() {
        buckets[(h % partitions as u64) as usize].push(i as u32);
    }

    // Group each partition independently in parallel (rows sharing a key are all here),
    // reusing the precomputed hashes. Returns (local group id per bucket row, reps).
    let grouped: Vec<(Vec<u32>, Vec<u32>)> = buckets
        .par_iter()
        .map(|bucket| {
            let mut table: HashTable<u32> = HashTable::with_capacity(bucket.len());
            let mut reps: Vec<u32> = Vec::new();
            let mut local: Vec<u32> = Vec::with_capacity(bucket.len());
            for &i in bucket {
                let row_i = rows.row(i as usize);
                let gid = match table.entry(
                    hashes[i as usize],
                    |&g| rows.row(reps[g as usize] as usize) == row_i,
                    |&g| hashes[reps[g as usize] as usize],
                ) {
                    Entry::Occupied(e) => *e.get(),
                    Entry::Vacant(e) => {
                        let gid = reps.len() as u32;
                        reps.push(i);
                        e.insert(gid);
                        gid
                    }
                };
                local.push(gid);
            }
            (local, reps)
        })
        .collect();

    // Prefix-sum partition group counts → global offsets, then scatter global ids back
    // into original row order.
    let mut offsets = Vec::with_capacity(partitions + 1);
    offsets.push(0u32);
    for (_, reps) in &grouped {
        offsets.push(offsets.last().unwrap() + reps.len() as u32);
    }
    let num_groups = *offsets.last().unwrap() as usize;

    let mut group_ids = vec![0u32; num_rows];
    for (p, (local, _)) in grouped.iter().enumerate() {
        let off = offsets[p];
        for (k, &i) in buckets[p].iter().enumerate() {
            group_ids[i as usize] = off + local[k];
        }
    }

    // Group columns = each group's representative row, in (partition, local) order.
    let rep_rows = grouped
        .iter()
        .flat_map(|(_, reps)| reps.iter().map(|&r| rows.row(r as usize)));
    let group_columns = converter.convert_rows(rep_rows)?;
    Ok((group_ids, num_groups, group_columns))
}
