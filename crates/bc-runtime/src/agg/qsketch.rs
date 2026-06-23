//! APPROX_QUANTILE / APPROX_MEDIAN — bounded-memory quantiles via per-group DDSketch.
//!
//! Exact `MEDIAN`/`QUANTILE` keep every group's full value list as their partial
//! state, so a hot group (heavy skew) can OOM. This keeps a *fixed relative-error*
//! DDSketch per group instead: bounded memory regardless of skew, at the cost of a
//! small (~1%) relative error.
//!
//! DDSketch is chosen over KLL specifically because its merge is **exactly
//! order-independent** — values land in fixed logarithmic buckets and merging sums
//! bucket counts — so the sketch over a multiset is identical regardless of how the
//! input was partitioned or in what order partials are combined. That makes
//! `partial → combine → finalize` *bit-identical* whether single-node or
//! distributed, preserving the single-node==distributed invariant (KLL's
//! compaction is order-sensitive and would only match within error bounds).
//!
//! Numeric only (like the exact path): values are cast to `Float64`; a non-numeric
//! input is rejected. The per-group state is serialized to a `Binary` column.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, BinaryArray, Float64Array};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Float64Type};
use bc_sketches::{DDSketch, Mergeable};

use crate::error::RuntimeError;

fn serialize(sketches: &[DDSketch]) -> ArrayRef {
    Arc::new(BinaryArray::from_iter_values(
        sketches.iter().map(|s| s.to_bytes()),
    ))
}

/// Partial state: one DDSketch per group over `values` (cast to `Float64`,
/// non-null only). Returns one serialized sketch per group as a `Binary` column.
pub(crate) fn approx_quantile_state(
    values: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
) -> Result<ArrayRef, RuntimeError> {
    let f = cast(values, &DataType::Float64).map_err(|_| RuntimeError::UnsupportedAggregate {
        func: "approx_quantile".to_string(),
        dtype: values.data_type().to_string(),
    })?;
    let f = f.as_primitive::<Float64Type>();
    let mut sketches: Vec<DDSketch> = (0..num_groups).map(|_| DDSketch::default()).collect();
    for (i, &g) in group_ids.iter().enumerate() {
        if f.is_valid(i) {
            sketches[g as usize].add(f.value(i));
        }
    }
    Ok(serialize(&sketches))
}

/// Merge per-group DDSketches across partitions (`combine` has concatenated the
/// partial `Binary` state columns). Order-independent: bucket counts sum.
pub(crate) fn merge_approx_quantile(
    state: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
) -> Result<ArrayRef, RuntimeError> {
    let blobs = state.as_binary::<i32>();
    let mut sketches: Vec<DDSketch> = (0..num_groups).map(|_| DDSketch::default()).collect();
    for row in 0..blobs.len() {
        if blobs.is_valid(row) {
            if let Some(s) = DDSketch::from_bytes(blobs.value(row)) {
                sketches[group_ids[row] as usize].merge(&s);
            }
        }
    }
    Ok(serialize(&sketches))
}

/// The `q`-quantile per group (`q ∈ [0,1]`); null for an empty group.
pub(crate) fn finalize_approx_quantile(state: &ArrayRef, q: f64) -> ArrayRef {
    let blobs = state.as_binary::<i32>();
    let out: Float64Array = (0..blobs.len())
        .map(|i| {
            if blobs.is_valid(i) {
                DDSketch::from_bytes(blobs.value(i)).and_then(|s| s.quantile(q))
            } else {
                None
            }
        })
        .collect();
    Arc::new(out)
}
