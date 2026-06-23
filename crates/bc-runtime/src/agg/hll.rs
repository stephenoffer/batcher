//! APPROX_COUNT_DISTINCT — bounded-memory distinct count via per-group HyperLogLog.
//!
//! Exact `COUNT(DISTINCT)` keeps every group's full value list as its partial state,
//! so a single hot group (heavy skew) can OOM. This keeps a *fixed-size* HLL sketch
//! per group instead: memory is bounded regardless of skew, at the cost of a small
//! (~2%) relative error. It is fully mergeable — HLLs union exactly — so
//! `partial → combine → finalize` equals the sketch built over the whole input, and
//! the distributed path inherits it unchanged.
//!
//! The per-group state is serialized to a `Binary` column (row `g` = group `g`'s HLL
//! bytes), which flows through `combine`'s generic state concatenation like any other
//! state column; `merge` deserializes, unions per group, and re-serializes.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, BinaryArray, Int64Array};
use arrow::row::{RowConverter, SortField};
use bc_sketches::{HyperLogLog, Mergeable};

use crate::error::RuntimeError;

// Fixed seed so value hashing is deterministic within a process — partials built on
// different morsels hash identically, which the HLL union relies on.
const SEED: ahash::RandomState =
    ahash::RandomState::with_seeds(0x9e37_79b9, 0x7f4a_7c15, 0xf39c_c060, 0x5ced_c834);

/// Serialize one HLL per group into a `Binary` column (row `g` = group `g`).
fn serialize(hlls: &[HyperLogLog]) -> ArrayRef {
    Arc::new(BinaryArray::from_iter_values(
        hlls.iter().map(|h| h.to_bytes()),
    ))
}

/// Partial state: one HLL per group over `values`, hashing each non-null value
/// (nulls excluded, matching `COUNT(DISTINCT)` semantics). Any value type is hashed
/// via arrow's row encoding, so this is type-generic like the exact path.
pub(crate) fn approx_distinct_state(
    values: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
) -> Result<ArrayRef, RuntimeError> {
    let mut hlls: Vec<HyperLogLog> = (0..num_groups)
        .map(|_| HyperLogLog::default_precision())
        .collect();
    let converter = RowConverter::new(vec![SortField::new(values.data_type().clone())])?;
    let rows = converter.convert_columns(std::slice::from_ref(values))?;
    for (i, &g) in group_ids.iter().enumerate() {
        if values.is_valid(i) {
            hlls[g as usize].add_hash(SEED.hash_one(rows.row(i)));
        }
    }
    Ok(serialize(&hlls))
}

/// Merge per-group HLLs across partitions: `combine` has concatenated the partial
/// `Binary` state columns; union the sketches routed to each output group.
pub(crate) fn merge_approx_distinct(
    state: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
) -> Result<ArrayRef, RuntimeError> {
    let blobs = state.as_binary::<i32>();
    let mut hlls: Vec<HyperLogLog> = (0..num_groups)
        .map(|_| HyperLogLog::default_precision())
        .collect();
    for row in 0..blobs.len() {
        if blobs.is_valid(row) {
            if let Some(h) = HyperLogLog::from_bytes(blobs.value(row)) {
                hlls[group_ids[row] as usize].merge(&h);
            }
        }
    }
    Ok(serialize(&hlls))
}

/// Approximate distinct count per group = its HLL's cardinality estimate.
pub(crate) fn finalize_approx_distinct(state: &ArrayRef) -> ArrayRef {
    let blobs = state.as_binary::<i32>();
    let counts: Vec<i64> = (0..blobs.len())
        .map(|i| {
            if blobs.is_valid(i) {
                HyperLogLog::from_bytes(blobs.value(i)).map_or(0, |h| h.estimate().round() as i64)
            } else {
                0
            }
        })
        .collect();
    Arc::new(Int64Array::from(counts))
}
