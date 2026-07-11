//! Sort-merge equi-join: the no-hash-table join for two large (or already-sorted)
//! inputs. Produces the same [`JoinIndices`](super::JoinIndices) relation as the
//! hash join for every join type (output order differs — these are unordered
//! relations), so the executor can pick it on the build side without changing
//! semantics. Split out of the join module along the algorithm seam.

use std::cmp::Ordering;

use arrow::array::{Array, ArrayRef, UInt32Array};
use arrow::row::{RowConverter, Rows, SortField};

use super::{null_mask, JoinIndices, JoinType};
use crate::error::RuntimeError;

/// Sort `idx` into ascending encoded-key order, skipping the sort when the indices
/// already arrive that way (one O(n) pass). Pre-sorted input — time-series, an
/// upstream `Sort`, sorted lakehouse files — then merges without the O(n log n) sort,
/// which is what makes sort-merge the right pick for already-ordered inputs.
/// Result-identical: the merge consumes ascending keys either way, and equal-key
/// group order does not affect the unordered join relation.
fn sort_indices_if_unsorted(idx: &mut [u32], enc: &Rows) {
    let already = idx
        .windows(2)
        .all(|w| enc.row(w[0] as usize) <= enc.row(w[1] as usize));
    if !already {
        idx.sort_by(|&a, &b| enc.row(a as usize).cmp(&enc.row(b as usize)));
    }
}

/// Sort-merge join: sort both sides by key, then merge. Produces the **same
/// [`JoinIndices`](super::JoinIndices) relation** as
/// [`hash_join_indices`](super::hash_join_indices) for every join type (output order
/// differs — these are unordered relations). The win is no hash table: both sides
/// stream in key order, so it suits two large (or already-sorted) inputs the way
/// Spark's default join does. NULL keys never match (`NULL ≠ NULL`).
pub fn sort_merge_join_indices(
    left_keys: &[ArrayRef],
    right_keys: &[ArrayRef],
    join_type: JoinType,
) -> Result<JoinIndices, RuntimeError> {
    let n_left = left_keys.first().map_or(0, |a| a.len());
    let n_right = right_keys.first().map_or(0, |a| a.len());

    // One shared converter so left/right encoded keys are mutually comparable.
    let fields: Vec<SortField> = right_keys
        .iter()
        .map(|a| SortField::new(a.data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields)?;
    let left_enc = converter.convert_columns(left_keys)?;
    let right_enc = converter.convert_columns(right_keys)?;
    let left_null = null_mask(left_keys, n_left);
    let right_null = null_mask(right_keys, n_right);

    // Sort the non-null-key rows of each side by encoded key (null keys never match
    // and are handled with the unmatched rows below).
    let mut l: Vec<u32> = (0..n_left as u32)
        .filter(|&i| !left_null[i as usize])
        .collect();
    let mut r: Vec<u32> = (0..n_right as u32)
        .filter(|&i| !right_null[i as usize])
        .collect();
    // Skip the O(n log n) sort on a side that already arrives in ascending key order
    // (pre-sorted lakehouse / time-series input, or an upstream `Sort`): a one-pass
    // check is O(n). The merge only needs ascending keys — equal-key group order is
    // irrelevant to the unordered result — so the as-is order is bit-equivalent.
    sort_indices_if_unsorted(&mut l, &left_enc);
    sort_indices_if_unsorted(&mut r, &right_enc);

    // Left/Full/Anti preserve unmatched left rows; Right/Full preserve unmatched
    // right rows (Semi emits only *matched* left rows, once each).
    let emit_left_unmatched = matches!(join_type, JoinType::Left | JoinType::Full | JoinType::Anti);
    let emit_right_unmatched = matches!(join_type, JoinType::Right | JoinType::Full);

    // The output is at least as large as the bigger side (each matched/unmatched row emits
    // once); pre-size to that lower bound so the common near-1:1 join skips early reallocs.
    let out_hint = l.len().max(r.len());
    let mut left_out: Vec<Option<u32>> = Vec::with_capacity(out_hint);
    let mut right_out: Vec<Option<u32>> = Vec::with_capacity(out_hint);
    let mut push = |lo: Option<u32>, ro: Option<u32>| {
        left_out.push(lo);
        right_out.push(ro);
    };

    let (mut i, mut j) = (0usize, 0usize);
    while i < l.len() && j < r.len() {
        match left_enc
            .row(l[i] as usize)
            .cmp(&right_enc.row(r[j] as usize))
        {
            Ordering::Less => {
                if emit_left_unmatched {
                    push(Some(l[i]), None);
                }
                i += 1;
            }
            Ordering::Greater => {
                if emit_right_unmatched {
                    push(None, Some(r[j]));
                }
                j += 1;
            }
            Ordering::Equal => {
                // Extents of the equal-key group on each side.
                let key = left_enc.row(l[i] as usize);
                let mut i2 = i + 1;
                while i2 < l.len() && left_enc.row(l[i2] as usize) == key {
                    i2 += 1;
                }
                let mut j2 = j + 1;
                while j2 < r.len() && right_enc.row(r[j2] as usize) == key {
                    j2 += 1;
                }
                match join_type {
                    // Semi: each matched left row once (no right column).
                    JoinType::Semi => {
                        for &li in &l[i..i2] {
                            push(Some(li), None);
                        }
                    }
                    // Anti: matched rows are dropped (only unmatched left survives).
                    JoinType::Anti => {}
                    // Inner/Left/Right/Full: the group cross product.
                    _ => {
                        for &li in &l[i..i2] {
                            for &rj in &r[j..j2] {
                                push(Some(li), Some(rj));
                            }
                        }
                    }
                }
                i = i2;
                j = j2;
            }
        }
    }
    // Tails: rows past the other side's end are all unmatched.
    while i < l.len() {
        if emit_left_unmatched {
            push(Some(l[i]), None);
        }
        i += 1;
    }
    while j < r.len() {
        if emit_right_unmatched {
            push(None, Some(r[j]));
        }
        j += 1;
    }
    // Null-key rows match nothing but are still part of their relation for outer joins.
    if emit_left_unmatched {
        for (li, &is_null) in left_null.iter().enumerate() {
            if is_null {
                push(Some(li as u32), None);
            }
        }
    }
    if emit_right_unmatched {
        for (rj, &is_null) in right_null.iter().enumerate() {
            if is_null {
                push(None, Some(rj as u32));
            }
        }
    }

    Ok(JoinIndices {
        left: UInt32Array::from(left_out),
        right: UInt32Array::from(right_out),
    })
}
