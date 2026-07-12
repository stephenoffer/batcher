//! COUNT(DISTINCT) — exact, mergeable via a per-group value list — plus the
//! `bucket_values_into_list` helper shared with the median path and the single-pass
//! whole-row `distinct_batch` dedup.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Int64Array, ListArray, RecordBatch, UInt32Array};
use arrow::buffer::OffsetBuffer;
use arrow::compute::take;
use arrow::datatypes::{DataType, Field, Int64Type};
use rayon::prelude::*;

use super::assign_groups;
use super::group::dense_budget;
use crate::error::RuntimeError;

/// Whole-relation `DISTINCT` over a **single dense integer column**, without hashing,
/// partitioning, or materializing the input.
///
/// When the column's value range is small (dictionary codes, dense ids, enums), "which
/// values occur" is a presence bitmap indexed by `value - min`. Each morsel-chunk ORs into
/// its own bitmap across cores, the bitmaps reduce with a word-wise OR, and the distinct
/// values fall out of a scan of the set bits. That is two linear passes over the key and
/// nothing else — no `group_ids` buffer, no gather, no per-morsel partial to re-merge.
///
/// Returns `None` (caller keeps its usual path) for anything else: more than one column, a
/// nullable or non-integer key, or a value range too sparse to be worth a map.
///
/// Rows come back in **ascending value order**. `DISTINCT` has no defined row order — the
/// existing paths already differ (the sequential oracle emits first-seen order, the
/// parallel bucket dedup emits bucket order) — so callers must not depend on it.
pub fn distinct_dense(parts: &[RecordBatch]) -> Result<Option<RecordBatch>, RuntimeError> {
    let Some(first) = parts.first() else {
        return Ok(None);
    };
    if first.num_columns() != 1 || !matches!(first.column(0).data_type(), DataType::Int64) {
        return Ok(None);
    }
    let cols: Vec<&Int64Array> = parts
        .iter()
        .map(|b| b.column(0).as_primitive::<Int64Type>())
        .collect();
    if cols.iter().any(|c| c.null_count() != 0) {
        return Ok(None);
    }
    let rows: usize = cols.iter().map(|c| c.len()).sum();
    if rows == 0 {
        return Ok(None);
    }

    // Pass 1: the value range. `i128` so `i64::MIN..=i64::MAX` cannot overflow the span.
    let (lo, hi) = cols
        .par_iter()
        .filter(|c| !c.is_empty())
        .map(|c| {
            c.values()
                .iter()
                .fold((i64::MAX, i64::MIN), |(l, h), &v| (l.min(v), h.max(v)))
        })
        .reduce(
            || (i64::MAX, i64::MIN),
            |(l1, h1), (l2, h2)| (l1.min(l2), h1.max(h2)),
        );
    let span_i = (hi as i128) - (lo as i128) + 1;
    let Ok(span) = usize::try_from(span_i) else {
        return Ok(None);
    };
    if span > dense_budget(rows) {
        return Ok(None);
    }

    // Pass 2: presence bitmap, OR-reduced across cores.
    let words = span.div_ceil(64);
    let bits = cols
        .par_iter()
        .map(|c| {
            let mut w = vec![0u64; words];
            for &v in c.values().iter() {
                let i = (v.wrapping_sub(lo)) as usize;
                w[i >> 6] |= 1u64 << (i & 63);
            }
            w
        })
        .reduce(
            || vec![0u64; words],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x |= y;
                }
                a
            },
        );

    let mut vals: Vec<i64> = Vec::new();
    for (wi, &w) in bits.iter().enumerate() {
        let mut w = w;
        while w != 0 {
            let b = w.trailing_zeros() as usize;
            vals.push(lo.wrapping_add(((wi << 6) + b) as i64));
            w &= w - 1;
        }
    }
    let out: ArrayRef = Arc::new(Int64Array::from(vals));
    Ok(Some(RecordBatch::try_new(first.schema(), vec![out])?))
}

/// Single-pass whole-row DISTINCT: hash-partition the batch's rows by *all* columns into
/// `num_partitions` buckets (equal rows co-partition), then dedup each bucket independently
/// across cores and concatenate the per-bucket distinct rows. Unlike the per-morsel
/// `partial` + `combine` path, this hashes each row **once** — the win for a
/// high-cardinality DISTINCT whose per-morsel partial reduces nothing yet is still hashed
/// again in the combine. The caller gates this on null-free key columns (so the fast
/// integer/byte partition co-locates equal rows) and an in-memory working set.
pub fn distinct_batch(
    batch: &RecordBatch,
    num_partitions: usize,
) -> Result<RecordBatch, RuntimeError> {
    let ncols = batch.num_columns();
    let key_idx: Vec<usize> = (0..ncols).collect();
    let buckets = crate::shuffle::partition_by_keys(batch, &key_idx, num_partitions)?;
    // Each bucket's distinct rows are its `assign_groups` representatives (first-seen);
    // the buckets partition the key space, so their union is the global distinct set.
    let per: Vec<Vec<ArrayRef>> = buckets
        .par_iter()
        .map(|b| {
            let keys: Vec<ArrayRef> = b.columns().to_vec();
            let (_ids, _n, group_cols) = assign_groups(&keys, b.num_rows())?;
            Ok::<_, RuntimeError>(group_cols)
        })
        .collect::<Result<_, _>>()?;
    let out_cols: Vec<ArrayRef> = (0..ncols)
        .into_par_iter()
        .map(|c| {
            let arrs: Vec<&dyn Array> = per.iter().map(|g| g[c].as_ref()).collect();
            Ok::<_, RuntimeError>(arrow::compute::concat(&arrs)?)
        })
        .collect::<Result<_, _>>()?;
    Ok(RecordBatch::try_new(batch.schema(), out_cols)?)
}

/// Partial state for COUNT(DISTINCT): each group's distinct non-null values as one
/// `List` column (row `g` = group `g`). Nulls are excluded (SQL semantics). The
/// dedup reuses `assign_groups` on `(group, value)` pairs — no bespoke set code.
pub(crate) fn distinct_state(
    values: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
) -> Result<ArrayRef, RuntimeError> {
    // At most one entry per input row; reserve up front so a large group-by does not
    // repeatedly reallocate (each growth copies the whole buffer and transiently holds ~1.5×).
    let mut keep: Vec<u32> = Vec::with_capacity(group_ids.len());
    let mut kept_groups: Vec<i64> = Vec::with_capacity(group_ids.len());
    for (i, &g) in group_ids.iter().enumerate() {
        if values.is_valid(i) {
            keep.push(i as u32);
            kept_groups.push(g as i64);
        }
    }
    let kept_values = take(values.as_ref(), &UInt32Array::from(keep), None)?;
    let group_col: ArrayRef = Arc::new(Int64Array::from(kept_groups));
    distinct_pairs_to_list(group_col, kept_values, num_groups)
}

/// Merge per-group distinct lists across partitions: flatten to `(group, value)`
/// pairs, dedup, re-bucket. `combine` has already concatenated the list columns.
pub(crate) fn merge_distinct(
    state: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
) -> Result<ArrayRef, RuntimeError> {
    let list = state.as_list::<i32>();
    let offsets = list.value_offsets();
    let child = list.values();
    // The concatenated child holds every partial list's values in list-row order with contiguous
    // offsets, so flattening in row order visits child element `e` at output position `e` — the
    // index would be `0..child.len()` and take-ing through it just copies every value (~tens of
    // millions on a big COUNT(DISTINCT)) without reordering. Expand only the per-element group id
    // and hand the child straight to the deduping bucketer.
    let contiguous = offsets.first().copied().unwrap_or(0) == 0
        && offsets.last().map(|&o| o as usize) == Some(child.len());
    let mut elem_groups: Vec<i64> = Vec::with_capacity(child.len());
    for row in 0..list.len() {
        let n = (offsets[row + 1] - offsets[row]) as usize;
        let g = group_ids[row] as i64;
        elem_groups.extend(std::iter::repeat_n(g, n));
    }
    let group_col: ArrayRef = Arc::new(Int64Array::from(elem_groups));
    if contiguous {
        return distinct_pairs_to_list(group_col, child.clone(), num_groups);
    }
    let elem_idx: Vec<u32> = (0..list.len())
        .flat_map(|row| offsets[row] as u32..offsets[row + 1] as u32)
        .collect();
    let values = take(child.as_ref(), &UInt32Array::from(elem_idx), None)?;
    distinct_pairs_to_list(group_col, values, num_groups)
}

/// Dedup `(group, value)` pairs and bucket the distinct values into a per-group
/// `List` column — the shared core of the distinct partial and merge steps.
fn distinct_pairs_to_list(
    groups: ArrayRef,
    values: ArrayRef,
    num_groups: usize,
) -> Result<ArrayRef, RuntimeError> {
    let n = values.len();
    // Fast path: an `Int64` value column (the common `count(distinct <int id>)`, e.g.
    // TPC-H Q16's `count(distinct ps_suppkey)`) dedups the raw `(group, value)` pairs
    // through a hash set, skipping the `RowConverter` encoding the general path runs
    // over every row. First-seen order is preserved (matching `assign_groups`), so the
    // bucketed lists are identical.
    if let Some(vals) = values.as_any().downcast_ref::<Int64Array>() {
        let grp = groups.as_primitive::<Int64Type>();
        // Dedup through hashbrown + a fixed-seed ahash hasher, not `std::HashSet`'s
        // cryptographic SipHash — ~5-10× faster on these small `(i64, i64)` integer keys,
        // and the result is hasher-independent (first-seen order is preserved by the
        // insert branch below), so the deterministic distinct set is unchanged.
        let mut seen: hashbrown::HashSet<(i64, i64), ahash::RandomState> =
            hashbrown::HashSet::with_hasher(ahash::RandomState::with_seed(0));
        let (mut dgroups, mut dvalues): (Vec<i64>, Vec<i64>) = (Vec::new(), Vec::new());
        for i in 0..n {
            let pair = (grp.value(i), vals.value(i));
            if seen.insert(pair) {
                dgroups.push(pair.0);
                dvalues.push(pair.1);
            }
        }
        let distinct_values: ArrayRef = Arc::new(Int64Array::from(dvalues));
        return bucket_values_into_list(&Int64Array::from(dgroups), &distinct_values, num_groups);
    }
    let (_ids, _n_pairs, pair_cols) = assign_groups(&[groups, values], n)?;
    let distinct_groups = pair_cols[0].as_primitive::<Int64Type>();
    bucket_values_into_list(distinct_groups, &pair_cols[1], num_groups)
}

/// Bucket `values` into a `List` column by their `group_ids` (each in
/// `0..num_groups`), preserving stable order within each group.
pub(crate) fn bucket_values_into_list(
    group_ids: &Int64Array,
    values: &ArrayRef,
    num_groups: usize,
) -> Result<ArrayRef, RuntimeError> {
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); num_groups];
    for i in 0..group_ids.len() {
        buckets[group_ids.value(i) as usize].push(i as u32);
    }
    let mut order: Vec<u32> = Vec::with_capacity(values.len());
    let mut offsets: Vec<i32> = Vec::with_capacity(num_groups + 1);
    offsets.push(0);
    for bucket in &buckets {
        order.extend_from_slice(bucket);
        offsets.push(order.len() as i32);
    }
    let ordered = take(values.as_ref(), &UInt32Array::from(order), None)?;
    let field = Arc::new(Field::new("item", values.data_type().clone(), true));
    let list = ListArray::try_new(field, OffsetBuffer::new(offsets.into()), ordered, None)?;
    Ok(Arc::new(list))
}

/// Distinct count per group = the length of its distinct-value list.
pub(crate) fn finalize_count_distinct(state: &ArrayRef) -> ArrayRef {
    let list = state.as_list::<i32>();
    let offsets = list.value_offsets();
    let counts: Vec<i64> = (0..list.len())
        .map(|i| (offsets[i + 1] - offsets[i]) as i64)
        .collect();
    Arc::new(Int64Array::from(counts))
}

#[cfg(test)]
mod dense_tests {
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    fn batches(chunks: Vec<Vec<Option<i64>>>) -> Vec<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, true)]));
        chunks
            .into_iter()
            .map(|c| {
                let a: ArrayRef = Arc::new(Int64Array::from(c));
                RecordBatch::try_new(schema.clone(), vec![a]).unwrap()
            })
            .collect()
    }

    fn values(b: &RecordBatch) -> Vec<i64> {
        b.column(0).as_primitive::<Int64Type>().values().to_vec()
    }

    /// Distinct values across morsels, ascending, with duplicates collapsed.
    #[test]
    fn dense_distinct_across_morsels() {
        let parts = batches(vec![
            vec![Some(3), Some(1), Some(3)],
            vec![Some(2), Some(1)],
            vec![Some(5)],
        ]);
        let out = distinct_dense(&parts)
            .unwrap()
            .expect("dense should engage");
        assert_eq!(values(&out), vec![1, 2, 3, 5]);
    }

    /// Negative values exercise the `value - min` offset.
    #[test]
    fn dense_distinct_negative_values() {
        let parts = batches(vec![vec![Some(-5), Some(0), Some(-5), Some(3)]]);
        let out = distinct_dense(&parts).unwrap().unwrap();
        assert_eq!(values(&out), vec![-5, 0, 3]);
    }

    /// A single repeated value: span 1.
    #[test]
    fn dense_distinct_single_value() {
        let parts = batches(vec![vec![Some(7); 100]]);
        let out = distinct_dense(&parts).unwrap().unwrap();
        assert_eq!(values(&out), vec![7]);
    }

    /// A nullable column must decline (nulls are a DISTINCT group the bitmap has no slot for).
    #[test]
    fn nullable_declines() {
        let parts = batches(vec![vec![Some(1), None, Some(2)]]);
        assert!(distinct_dense(&parts).unwrap().is_none());
    }

    /// A sparse range exceeds the budget and declines rather than allocating a huge map.
    #[test]
    fn sparse_range_declines() {
        let parts = batches(vec![vec![Some(0), Some(1 << 40)]]);
        assert!(distinct_dense(&parts).unwrap().is_none());
    }

    /// `i64::MIN`/`i64::MAX` would overflow the span: decline, never wrap.
    #[test]
    fn extreme_range_declines() {
        let parts = batches(vec![vec![Some(i64::MIN), Some(i64::MAX)]]);
        assert!(distinct_dense(&parts).unwrap().is_none());
    }

    /// More than one column, or a non-integer column, declines.
    #[test]
    fn wrong_shape_declines() {
        let schema = Arc::new(Schema::new(vec![Field::new("f", DataType::Float64, false)]));
        let a: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0]));
        let f = RecordBatch::try_new(schema, vec![a]).unwrap();
        assert!(distinct_dense(std::slice::from_ref(&f)).unwrap().is_none());

        let schema2 = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]));
        let x: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 2]));
        let y: ArrayRef = Arc::new(Int64Array::from(vec![3i64, 4]));
        let two = RecordBatch::try_new(schema2, vec![x, y]).unwrap();
        assert!(distinct_dense(std::slice::from_ref(&two))
            .unwrap()
            .is_none());
    }

    /// The dense path returns exactly the set `assign_groups` would find (order aside).
    #[test]
    fn dense_matches_assign_groups_set() {
        let mut s: u64 = 5;
        let vals: Vec<Option<i64>> = (0..5000)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                Some(((s >> 33) % 700) as i64 - 300)
            })
            .collect();
        let parts = batches(vec![vals.clone()]);
        let dense = distinct_dense(&parts).unwrap().unwrap();
        let keys: Vec<ArrayRef> = parts[0].columns().to_vec();
        let (_ids, n, cols) = assign_groups(&keys, parts[0].num_rows()).unwrap();
        let mut want = cols[0].as_primitive::<Int64Type>().values().to_vec();
        want.sort_unstable();
        assert_eq!(n, want.len());
        assert_eq!(values(&dense), want);
    }
}
