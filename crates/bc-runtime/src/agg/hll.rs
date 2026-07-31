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

use std::hash::BuildHasher;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, BinaryArray, Int64Array};
use arrow::row::{RowConverter, SortField};
use bc_sketches::{HyperLogLog, Mergeable};

use crate::error::RuntimeError;

// Deterministic value hashing, *across* processes and not merely within one. HLL partials
// are serialized into a Binary column and merged by `combine_finalize` on another machine,
// so a register set built with one hash and unioned with a register set built with another
// does not error — it silently returns a wrong distinct count. `ahash` cannot promise that
// (it selects an AES-NI backend at compile time); `PortableBuildHasher` can. See
// `crate::keys::SHUFFLE_HASHER`.
const SEED: bc_arrow::PortableBuildHasher =
    bc_arrow::PortableBuildHasher::with_seed(0x484C_4C5F_5245_4753);

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
    // Canonicalize a Float64 value column FIRST so `-0.0`/`0.0` and every NaN bit pattern
    // hash to one value — the same distinct identity the EXACT path uses (its raw-multikey
    // dedup routes float values through `canon_f64`). Without this the row encoding, which is
    // NOT canonical for floats, counted `-0.0` and `0.0` as two distinct values, so
    // `approx_count_distinct` over a float column disagreed with `count(distinct)` and DuckDB
    // (e.g. `{-0.0, 0.0, NaN, 1.5}` estimated 4 where the exact count is 3). Nulls carry
    // through unchanged, so the validity check below is still correct.
    let canon = crate::keys::canonicalize_float_keys(std::slice::from_ref(values));
    let encode = canon.as_ref().map_or(values, |c| &c[0]);
    let converter = RowConverter::new(vec![SortField::new(encode.data_type().clone())])?;
    let rows = converter.convert_columns(std::slice::from_ref(encode))?;
    // `values` is an `Arc<dyn Array>`, so checking validity on it per row is a virtual call.
    // Resolve the null buffer once; a null-free column then hashes every row unconditionally.
    match values.nulls() {
        None => {
            for (i, &g) in group_ids.iter().enumerate() {
                hlls[g as usize].add_hash(SEED.hash_one(rows.row(i)));
            }
        }
        Some(nulls) => {
            for (i, &g) in group_ids.iter().enumerate() {
                if nulls.is_valid(i) {
                    hlls[g as usize].add_hash(SEED.hash_one(rows.row(i)));
                }
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;
    use arrow::datatypes::Int64Type;

    /// `-0.0`/`+0.0` are one value and every NaN is one value for distinct identity, so
    /// `approx_count_distinct({-0.0, 0.0, NaN, NaN, 1.5})` must estimate 3 — matching the
    /// exact `count(distinct)` and DuckDB — not 4. Small cardinalities are in HLL's exact
    /// linear-counting regime, so the estimate is exact here.
    #[test]
    fn approx_distinct_canonicalizes_signed_zero_and_nan() {
        let values: ArrayRef = Arc::new(Float64Array::from(vec![
            -0.0,
            0.0,
            f64::NAN,
            f64::from_bits(0x7ff8_0000_0000_0001), // a different NaN bit pattern
            1.5,
        ]));
        let group_ids = [0u32, 0, 0, 0, 0];
        let state = approx_distinct_state(&values, &group_ids, 1).unwrap();
        let out = finalize_approx_distinct(&state);
        let est = out.as_primitive::<Int64Type>().value(0);
        assert_eq!(
            est, 3,
            "signed-zero/NaN must collapse to 3 distinct, got {est}"
        );
    }
}
