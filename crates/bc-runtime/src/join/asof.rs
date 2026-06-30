//! ASOF (nearest-match) join: each left row matched to the right row whose `on` key
//! is nearest in a direction within its `by` group. Left-style (every left row
//! emitted; unmatched → null right). Split out of the join module along the
//! algorithm seam; like the equi-join it carries no single-node assumption —
//! partitioning both sides by `by` makes a global ASOF the union of per-partition
//! ASOFs (the distributed seam).

use arrow::array::{Array, ArrayRef, UInt32Array};
use arrow::row::{OwnedRow, RowConverter, SortField};
use indexmap::IndexMap;

use super::JoinIndices;
use crate::error::RuntimeError;

/// Compute ASOF (nearest-match) join indices. Every left row is emitted (left-style);
/// it is matched to the right row whose `on` key is nearest *in `direction`* within
/// the same `by` group (exact `by` equality). Unmatched left rows get a null right
/// index (arrow `take` then yields null), exactly like a left outer join.
///
/// `backward = true` picks the largest right.on ≤ left.on; `false` picks the smallest
/// right.on ≥ left.on. Keys are arrow row-encoded, so `on` (order-preserving) and
/// `by` (equality) work for any type. Rows with a null `on` never match. As with the
/// equi-join primitive, partitioning both sides by `by` makes a global ASOF equal the
/// union of per-partition ASOFs — the seam the distributed path can use.
pub fn asof_join_indices(
    left_on: &ArrayRef,
    right_on: &ArrayRef,
    left_by: &[ArrayRef],
    right_by: &[ArrayRef],
    backward: bool,
) -> Result<JoinIndices, RuntimeError> {
    let n_left = left_on.len();
    let n_right = right_on.len();

    // One shared converter so left/right `on` encodings are mutually order-comparable.
    let on_conv = RowConverter::new(vec![SortField::new(right_on.data_type().clone())])?;
    let left_on_enc = on_conv.convert_columns(std::slice::from_ref(left_on))?;
    let right_on_enc = on_conv.convert_columns(std::slice::from_ref(right_on))?;

    let by_conv = if left_by.is_empty() {
        None
    } else {
        Some(RowConverter::new(
            right_by
                .iter()
                .map(|a| SortField::new(a.data_type().clone()))
                .collect(),
        )?)
    };
    let left_by_enc = by_conv
        .as_ref()
        .map(|c| c.convert_columns(left_by))
        .transpose()?;
    let right_by_enc = by_conv
        .as_ref()
        .map(|c| c.convert_columns(right_by))
        .transpose()?;

    // Group right rows by `by` key (byte-encoded; empty key when there are no `by`
    // columns), each group sorted ascending by `on` for binary search.
    let mut groups: IndexMap<Vec<u8>, Vec<(OwnedRow, u32)>> = IndexMap::new();
    for j in 0..n_right {
        if right_on.is_null(j) {
            continue;
        }
        let key = right_by_enc
            .as_ref()
            .map_or_else(Vec::new, |e| e.row(j).as_ref().to_vec());
        groups
            .entry(key)
            .or_default()
            .push((right_on_enc.row(j).owned(), j as u32));
    }
    for v in groups.values_mut() {
        v.sort_by(|a, b| a.0.row().cmp(&b.0.row()));
    }

    let mut right_idx: Vec<Option<u32>> = Vec::with_capacity(n_left);
    for i in 0..n_left {
        if left_on.is_null(i) {
            right_idx.push(None);
            continue;
        }
        let key = left_by_enc
            .as_ref()
            .map_or_else(Vec::new, |e| e.row(i).as_ref().to_vec());
        let target = left_on_enc.row(i);
        let matched = groups.get(&key).and_then(|g| {
            if backward {
                // largest on ≤ target
                let pp = g.partition_point(|(on, _)| on.row() <= target);
                (pp > 0).then(|| g[pp - 1].1)
            } else {
                // smallest on ≥ target
                let pp = g.partition_point(|(on, _)| on.row() < target);
                (pp < g.len()).then(|| g[pp].1)
            }
        });
        right_idx.push(matched);
    }

    Ok(JoinIndices {
        left: UInt32Array::from((0..n_left as u32).collect::<Vec<_>>()),
        right: UInt32Array::from(right_idx),
    })
}
