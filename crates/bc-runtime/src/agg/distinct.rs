//! COUNT(DISTINCT) — exact, mergeable via a per-group value list — plus the
//! `bucket_values_into_list` helper shared with the median path and the single-pass
//! whole-row `distinct_parts` dedup, and [`DistinctPrefix`], the order-preserving dedup that
//! stops once `k` distinct rows exist.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Int64Array, ListArray, RecordBatch, UInt32Array};
use arrow::buffer::OffsetBuffer;
use arrow::compute::{concat_batches, take};
use arrow::datatypes::{DataType, Field, Int64Type};
use rayon::prelude::*;

use super::assign_groups;
use super::group::dense_budget;
use crate::error::RuntimeError;

/// A `DISTINCT` that keeps the **first `k` distinct rows in input order** and can say when it
/// has them, so the caller stops pulling.
///
/// # Why first-in-order rather than any `k`
///
/// A `DISTINCT` under a `LIMIT k` currently deduplicates its whole input and then throws
/// nearly all of it away. That is asymptotic, not a constant factor: on a high-cardinality key
/// the work is proportional to the *input* to answer a question about `k` rows. DuckDB fuses
/// the pair (`PhysicalLimitedDistinct`) and stops its `Sink` the moment the hash table holds
/// `limit` groups, taking whichever `k` its threads reach first.
///
/// Batcher cannot take whichever `k` wins the race. Invariant #7 requires a result identical
/// on one node and on many, and a thread-order-dependent `k` is not. Keeping the *first* `k`
/// in input order is deterministic, costs nothing extra ([`assign_groups`] already returns
/// representatives in first-seen order), and still permits the early exit — because once a
/// prefix of the input holds `k` distinct rows, nothing after it can change the answer.
///
/// # Why it is still mergeable
///
/// `combine` is ordered concatenation followed by re-applying this operator, which is exactly
/// the shape `sample_n_batches` already uses. It is associative but **not commutative**, so
/// the distributed driver must assemble partitions by index — the same `preserve_order=True`
/// contract `dist/executor.py` already applies to a plain `Limit`, and for the same reason.
///
/// The argument that this is sound: if a row `v` is among the global first `k` distinct rows,
/// then within `v`'s own partition the number of distinct rows occurring before it is at most
/// the global number before it, which is `< k`. So `v` survives its own partition's first-`k`,
/// the union of the per-partition results contains the global answer, and re-applying the
/// operator to that ordered union selects exactly it.
#[derive(Debug)]
pub struct DistinctPrefix {
    /// The distinct rows seen so far, in first-seen order. Never more than `target` rows, so
    /// this is bounded by the limit rather than by the input.
    kept: Option<RecordBatch>,
    target: usize,
}

impl DistinctPrefix {
    /// A prefix collector for `target` distinct rows. `target == 0` is satisfied immediately.
    pub fn new(target: usize) -> Self {
        Self { kept: None, target }
    }

    /// True once `target` distinct rows are held, so the caller can stop pulling its input.
    pub fn is_satisfied(&self) -> bool {
        self.kept
            .as_ref()
            .map_or(self.target == 0, |b| b.num_rows() >= self.target)
    }

    /// How many distinct rows are held.
    pub fn len(&self) -> usize {
        self.kept.as_ref().map_or(0, |b| b.num_rows())
    }

    /// True when nothing has been kept yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fold one batch in, keeping first-seen order and never growing past `target`.
    ///
    /// Cost is `O(|kept| + |batch|)` with `|kept| <= target`, so a small limit makes this
    /// proportional to the batch rather than to everything seen so far.
    pub fn push(&mut self, batch: &RecordBatch) -> Result<(), RuntimeError> {
        if self.is_satisfied() || batch.num_rows() == 0 {
            return Ok(());
        }
        let schema = batch.schema();
        // Order matters: `kept` first, so `assign_groups`' first-seen representatives put
        // already-kept rows ahead of anything new in this batch.
        let combined = match self.kept.take() {
            Some(kept) => concat_batches(&schema, [&kept, batch])?,
            None => batch.clone(),
        };
        let keys: Vec<ArrayRef> = combined.columns().to_vec();
        let (_ids, _n, group_cols) = assign_groups(&keys, combined.num_rows())?;
        let deduped = RecordBatch::try_new(schema, group_cols)?;
        self.kept = Some(match deduped.num_rows() > self.target {
            true => deduped.slice(0, self.target),
            false => deduped,
        });
        Ok(())
    }

    /// The kept rows, or `None` when nothing was ever pushed.
    ///
    /// `None` rather than an empty batch because the caller owns the schema decision: an
    /// empty input defers to the path that supplies a correctly-typed empty relation.
    pub fn finish(self) -> Option<RecordBatch> {
        self.kept
    }
}

/// Whole-relation `DISTINCT` over `parts` keeping the first `target` distinct rows in order.
///
/// The single-shot form of [`DistinctPrefix`], for callers that already hold their input.
pub fn distinct_prefix(
    parts: &[RecordBatch],
    target: usize,
) -> Result<Option<RecordBatch>, RuntimeError> {
    let mut acc = DistinctPrefix::new(target);
    for batch in parts {
        acc.push(batch)?;
        if acc.is_satisfied() {
            break;
        }
    }
    Ok(acc.finish())
}

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
    // Validate *every* part is a single `Int64` column, not just the first. Heterogeneous
    // batches reach here — a `UNION`'s branches keep their own types, so an `Int64` branch
    // and a `Float64` branch arrive as differently-typed single-column batches — and blindly
    // `as_primitive::<Int64Type>()`-ing a non-`Int64` one panics ("primitive array"). Decline
    // (return `None`) instead, so the caller's general path handles (or cleanly rejects) the
    // mismatch rather than crashing the query.
    let cols: Vec<&Int64Array> = match parts
        .iter()
        .map(|b| {
            (b.num_columns() == 1)
                .then(|| b.column(0).as_any().downcast_ref::<Int64Array>())
                .flatten()
        })
        .collect::<Option<Vec<_>>>()
    {
        Some(cols) => cols,
        None => return Ok(None),
    };
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

/// Single-pass whole-row DISTINCT over a relation held as morsels: hash-partition every
/// morsel's rows by *all* columns into `num_partitions` buckets (equal rows co-partition),
/// then dedup each bucket independently across cores and concatenate the per-bucket distinct
/// rows. Unlike the per-morsel `partial` + `combine` path, this hashes each row **once** —
/// the win for a high-cardinality DISTINCT whose per-morsel partial reduces nothing yet is
/// still hashed again in the combine. The caller gates it on an in-memory working set.
///
/// Each partition's rows are gathered from the morsels in **one** `interleave` pass, rather
/// than scattered per morsel and concatenated back. Both shapes move the relation once; the
/// difference is fragmentation, and it is not small — a 600-morsel relation cut into 600xP
/// pieces costs tens of thousands of `concat` calls on ~600-row arrays to stitch back, which
/// measured slower than materializing the whole relation first and partitioning that.
///
/// Nulls are fine here and are not the caller's business to screen for: `bucket_of_rows` routes
/// a null-bearing row deterministically (a fixed bucket for an integer key, the row encoding
/// otherwise), so equal null-bearing rows land together, and `assign_groups` inside the bucket
/// compares them exactly.
pub fn distinct_parts(
    parts: &[RecordBatch],
    num_partitions: usize,
) -> Result<Vec<RecordBatch>, RuntimeError> {
    let Some(first) = parts.first() else {
        return Ok(Vec::new());
    };
    let ncols = first.num_columns();
    let csr: Vec<(Vec<u32>, Vec<u32>)> = parts
        .par_iter()
        .map(|b| {
            let keys: Vec<ArrayRef> = b.columns().to_vec();
            let part_of = crate::shuffle::bucket_of_rows(&keys, b.num_rows(), num_partitions)?;
            Ok::<_, RuntimeError>(crate::shuffle::bucket_csr(&part_of, num_partitions))
        })
        .collect::<Result<_, _>>()?;
    // Each bucket's distinct rows are its `assign_groups` representatives (first-seen); the
    // buckets partition the key space, so their union is the global distinct set.
    (0..num_partitions)
        .into_par_iter()
        .map(|p| {
            let mut coords: Vec<(usize, usize)> = Vec::new();
            for (m, (rows, offsets)) in csr.iter().enumerate() {
                let span = offsets[p] as usize..offsets[p + 1] as usize;
                coords.extend(rows[span].iter().map(|&r| (m, r as usize)));
            }
            if coords.is_empty() {
                return Ok(None);
            }
            let keys: Vec<ArrayRef> = (0..ncols)
                .map(|c| {
                    let arrs: Vec<&dyn Array> =
                        parts.iter().map(|b| b.column(c).as_ref()).collect();
                    arrow::compute::interleave(&arrs, &coords)
                })
                .collect::<Result<_, _>>()?;
            let (_ids, _n, group_cols) = assign_groups(&keys, coords.len())?;
            Ok(Some(RecordBatch::try_new(first.schema(), group_cols)?))
        })
        .collect::<Result<Vec<_>, RuntimeError>>()
        .map(|v| v.into_iter().flatten().collect())
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
    // `values` is an `Arc<dyn Array>`, so `values.is_valid(i)` is a **virtual call per row** —
    // and one the optimizer cannot see through, so it also blocks inlining the loop body.
    // Resolving the null buffer once turns the per-row check into an inlinable bit test, and
    // the null-free case (much the commonest) into no check at all.
    match values.nulls() {
        None => {
            for (i, &g) in group_ids.iter().enumerate() {
                keep.push(i as u32);
                kept_groups.push(g as i64);
            }
        }
        Some(nulls) => {
            for (i, &g) in group_ids.iter().enumerate() {
                if nulls.is_valid(i) {
                    keep.push(i as u32);
                    kept_groups.push(g as i64);
                }
            }
        }
    }
    let kept_values = take(values.as_ref(), &UInt32Array::from(keep), None)?;
    let group_col: ArrayRef = Arc::new(Int64Array::from(kept_groups));
    distinct_pairs_to_list(group_col, kept_values, num_groups)
}

/// Merge per-group distinct lists across partitions: flatten to `(group, value)`
/// pairs, dedup, re-bucket. `combine` has already concatenated the list columns.
/// Flatten a `List` aggregation state into per-element `(group_id, value)` columns.
///
/// COUNT(DISTINCT) and MEDIAN/QUANTILE both combine by concatenating each group's partial
/// value lists, then flattening the result to feed a per-group bucketer. The concatenated
/// child already holds every value in list-row order, so when the list offsets are
/// contiguous (`0, len0, len0+len1, ...`) flattening in row order visits child element `e`
/// exactly at output position `e` -- the flattened value column *is* the child, and the
/// obvious "build `0..child.len()` and `take` through it" is a full copy of every value
/// (~tens of millions on a low-cardinality aggregate) that reorders nothing. This skips it,
/// falling back to an explicit gather only for a sliced/offset child.
///
/// Returns the per-element group ids and the flat value column; the caller buckets them.
pub(crate) fn flatten_list_state(
    state: &ArrayRef,
    group_ids: &[u32],
) -> Result<(Int64Array, ArrayRef), RuntimeError> {
    let list = state.as_list::<i32>();
    let offsets = list.value_offsets();
    let child = list.values();
    let contiguous = offsets.first().copied().unwrap_or(0) == 0
        && offsets.last().map(|&o| o as usize) == Some(child.len());
    let mut elem_groups: Vec<i64> = Vec::with_capacity(child.len());
    for row in 0..list.len() {
        let n = (offsets[row + 1] - offsets[row]) as usize;
        let g = group_ids[row] as i64;
        elem_groups.extend(std::iter::repeat_n(g, n));
    }
    let elem_groups = Int64Array::from(elem_groups);
    if contiguous {
        return Ok((elem_groups, child.clone()));
    }
    let elem_idx: Vec<u32> = (0..list.len())
        .flat_map(|row| offsets[row] as u32..offsets[row + 1] as u32)
        .collect();
    let values = take(child.as_ref(), &UInt32Array::from(elem_idx), None)?;
    Ok((elem_groups, values))
}

pub(crate) fn merge_distinct(
    state: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
) -> Result<ArrayRef, RuntimeError> {
    let (elem_groups, values) = flatten_list_state(state, group_ids)?;
    distinct_pairs_to_list(Arc::new(elem_groups), values, num_groups)
}
/// Dedup `(group, value)` pairs and bucket the distinct values into a per-group
/// `List` column — the shared core of the distinct partial and merge steps.
/// Dedup `(group, value)` `i64` pairs across cores by hash-partitioning them, then deduping
/// each partition independently. Equal pairs share a partition, so the union of the partitions'
/// distinct pairs is the global distinct set (order within a group is not preserved — only the
/// COUNT(DISTINCT) caller uses this, and it counts, not orders). Returns the distinct groups and
/// values as parallel vecs.
fn par_dedup_pairs(grp: &Int64Array, vals: &Int64Array, n: usize) -> (Vec<i64>, Vec<i64>) {
    use rayon::prelude::*;

    let parts = rayon::current_num_threads().clamp(2, 256);
    let state = ahash::RandomState::with_seed(0);
    let g = grp.values();
    let v = vals.values();
    // Bucket the row indices by pair hash, in parallel: per-chunk CSR (histogram → prefix-sum →
    // scatter), then each partition's list is the chunks' slices concatenated (same shape as the
    // combine's radix bucketing). Cheap flat allocations, no per-(chunk,bucket) growing vectors.
    let nthreads = rayon::current_num_threads().max(1);
    let chunk = n.div_ceil(nthreads).max(1);
    // `crate::shuffle::bucket_of`, not `% parts`: a 64-bit modulo by a runtime value is a
    // hardware divide, and this closure runs twice per row — once to histogram, once to
    // scatter. `parts` is the thread count, so it is a power of two only by luck. Which
    // bucket a pair lands in is a purely internal choice (the dedup compares real keys
    // inside each bucket), and the bucket count is already `current_num_threads()`, so
    // nothing downstream could have depended on the old mapping either.
    let bucket_of =
        |i: usize| crate::shuffle::bucket_of(state.hash_one((g[i], v[i])), parts) as usize;
    let per_chunk: Vec<(Vec<u32>, Vec<u32>)> = (0..n)
        .into_par_iter()
        .step_by(chunk)
        .map(|start| {
            let end = (start + chunk).min(n);
            let mut off = vec![0u32; parts + 1];
            for i in start..end {
                off[bucket_of(i) + 1] += 1;
            }
            for b in 0..parts {
                off[b + 1] += off[b];
            }
            let mut cursor = off[..parts].to_vec();
            let mut rows = vec![0u32; end - start];
            for i in start..end {
                let b = bucket_of(i);
                rows[cursor[b] as usize] = i as u32;
                cursor[b] += 1;
            }
            (rows, off)
        })
        .collect();
    // Dedup each partition independently, in parallel.
    let per: Vec<(Vec<i64>, Vec<i64>)> = (0..parts)
        .into_par_iter()
        .map(|b| {
            let mut seen: hashbrown::HashSet<(i64, i64), ahash::RandomState> =
                hashbrown::HashSet::with_hasher(ahash::RandomState::with_seed(1));
            let (mut dg, mut dv) = (Vec::new(), Vec::new());
            for (rows, off) in &per_chunk {
                for &i in &rows[off[b] as usize..off[b + 1] as usize] {
                    let pair = (g[i as usize], v[i as usize]);
                    if seen.insert(pair) {
                        dg.push(pair.0);
                        dv.push(pair.1);
                    }
                }
            }
            (dg, dv)
        })
        .collect();
    let total: usize = per.iter().map(|(dg, _)| dg.len()).sum();
    let (mut dgroups, mut dvalues) = (Vec::with_capacity(total), Vec::with_capacity(total));
    for (dg, dv) in per {
        dgroups.extend_from_slice(&dg);
        dvalues.extend_from_slice(&dv);
    }
    (dgroups, dvalues)
}

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
        // Large input: dedup in parallel by hash-partitioning the `(group, value)` pairs across
        // cores. Equal pairs hash equally so they co-locate in one partition; deduping each
        // partition independently and unioning the survivors yields the same distinct SET as the
        // serial pass. COUNT(DISTINCT) only counts each group's list length, so the (now
        // per-partition-first-seen) order within a group is irrelevant to the result — this path
        // feeds only CountDistinct. Turns the serial hash-set scan (the whole cost of a big
        // COUNT(DISTINCT id) combine) into a parallel one.
        const PAR_DEDUP_MIN: usize = 1 << 18;
        if n >= PAR_DEDUP_MIN {
            let (dgroups, dvalues) = par_dedup_pairs(grp, vals, n);
            let distinct_values: ArrayRef = Arc::new(Int64Array::from(dvalues));
            return bucket_values_into_list(
                &Int64Array::from(dgroups),
                &distinct_values,
                num_groups,
            );
        }
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
///
/// A counting sort, not a vector of vectors. The obvious shape — `vec![Vec::new();
/// num_groups]`, push each row into its group's vector, concatenate — costs one heap
/// allocation *per group* and grows each of them geometrically. That is invisible on the
/// low-cardinality aggregate this started life on (`count(distinct x) GROUP BY region`, a
/// handful of groups) and dominates the high-cardinality one: a 6 M-row
/// `count(distinct u) GROUP BY k` over a near-unique key ends with ~3.8 M groups, so the
/// bookkeeping allocated 3.8 M vectors and chased 3.8 M pointers to concatenate them.
///
/// The histogram → prefix-sum → scatter below is the same CSR shape [`par_dedup_pairs`]
/// already uses, in two flat allocations that do not depend on the group count for their
/// *number*. It is **bit-identical** to the vector-of-vectors: `offsets` is still the
/// running per-group count, and scattering rows in ascending `i` through a per-group cursor
/// leaves each group's slice holding its rows in ascending `i` — exactly what pushing into
/// that group's vector produced. Both the group order and the within-group order are the
/// same, so callers that depend on stable order (the median/quantile path) are unaffected.
pub(crate) fn bucket_values_into_list(
    group_ids: &Int64Array,
    values: &ArrayRef,
    num_groups: usize,
) -> Result<ArrayRef, RuntimeError> {
    let groups = group_ids.values();
    let mut offsets: Vec<i32> = vec![0; num_groups + 1];
    for &g in groups.iter() {
        offsets[g as usize + 1] += 1;
    }
    for b in 0..num_groups {
        offsets[b + 1] += offsets[b];
    }
    let mut cursor: Vec<i32> = offsets[..num_groups].to_vec();
    let mut order: Vec<u32> = vec![0; groups.len()];
    for (i, &g) in groups.iter().enumerate() {
        let b = g as usize;
        order[cursor[b] as usize] = i as u32;
        cursor[b] += 1;
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

    /// The prefix keeps the FIRST `k` distinct rows in input order, not an arbitrary `k`.
    ///
    /// This is the property the whole fusion rests on: an arbitrary `k` would make the
    /// distributed result depend on which worker won a race, which invariant #7 forbids.
    #[test]
    fn prefix_keeps_the_first_k_in_input_order() {
        let parts = batches(vec![
            vec![Some(7), Some(3), Some(7)],
            vec![Some(1), Some(3), Some(9)],
        ]);
        let out = distinct_prefix(&parts, 3).unwrap().expect("rows exist");
        // First-seen order is 7, 3, 1 — NOT ascending, and not whichever three hash first.
        assert_eq!(values(&out), vec![7, 3, 1]);
    }

    /// A limit larger than the distinct count is simply not binding.
    #[test]
    fn prefix_under_target_returns_every_distinct_row() {
        let parts = batches(vec![vec![Some(2), Some(2), Some(5)]]);
        let out = distinct_prefix(&parts, 100).unwrap().expect("rows exist");
        assert_eq!(values(&out), vec![2, 5]);
    }

    /// An empty input yields `None`, so the caller can defer to the path that owns the
    /// correctly-typed empty relation rather than inventing a schema here.
    #[test]
    fn prefix_of_nothing_is_none() {
        assert!(distinct_prefix(&[], 5).unwrap().is_none());
        assert!(distinct_prefix(&batches(vec![vec![]]), 5)
            .unwrap()
            .is_none());
    }

    /// `target == 0` is satisfied before anything is pushed, so no input is read at all.
    #[test]
    fn prefix_of_zero_reads_nothing() {
        let mut acc = DistinctPrefix::new(0);
        assert!(acc.is_satisfied());
        acc.push(&batches(vec![vec![Some(1)]])[0]).unwrap();
        assert!(acc.finish().is_none());
    }

    /// **The mergeability invariant.** Splitting the input into partitions, taking each
    /// partition's own first-`k` prefix, then re-applying the operator to the ordered union
    /// must equal the single-node answer — for every split point and every `k`.
    ///
    /// This is what makes `combine` sound despite being non-commutative: the ordered union is
    /// what the distributed driver assembles under `preserve_order=True`.
    #[test]
    fn combine_finalize_of_partitions_equals_single_node() {
        let rows: Vec<Option<i64>> = vec![
            Some(5),
            Some(1),
            Some(5),
            Some(9),
            Some(1),
            Some(3),
            Some(7),
            Some(3),
            Some(8),
        ];
        let whole = batches(vec![rows.clone()]);
        for k in 1..=6 {
            let single = distinct_prefix(&whole, k).unwrap().expect("rows exist");
            for cut in 1..rows.len() {
                let (a, b) = rows.split_at(cut);
                let parts = batches(vec![a.to_vec(), b.to_vec()]);
                // `partial`: each partition keeps its own first `k`.
                let per_partition: Vec<RecordBatch> = parts
                    .iter()
                    .filter_map(|p| distinct_prefix(std::slice::from_ref(p), k).unwrap())
                    .collect();
                // `combine` + `finalize`: re-apply to the ordered union.
                let merged = distinct_prefix(&per_partition, k)
                    .unwrap()
                    .expect("rows exist");
                assert_eq!(
                    values(&merged),
                    values(&single),
                    "k={k} cut={cut}: partitioned answer differs from single-node"
                );
            }
        }
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

    /// Heterogeneous batch types (a `UNION` of an `Int64` branch and a `Float64` branch
    /// arrives as differently-typed single-column batches) must DECLINE, not panic. Before
    /// the fix, `distinct_dense` validated only the first batch's type, then
    /// `as_primitive::<Int64Type>()`-ed the `Float64` batch — an "primitive array" panic on a
    /// reachable data path.
    #[test]
    fn heterogeneous_batch_types_decline() {
        let int_schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
        let flt_schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, false)]));
        let ints: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 2, 3]));
        let flts: ArrayRef = Arc::new(Float64Array::from(vec![2.0, 3.5]));
        let parts = vec![
            RecordBatch::try_new(int_schema, vec![ints]).unwrap(),
            RecordBatch::try_new(flt_schema, vec![flts]).unwrap(),
        ];
        // Must return None (decline), never panic.
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

    /// The parallel `(group, value)` dedup path (n ≥ 2^18) yields the exact same distinct SET
    /// per group as a serial reference — the invariant COUNT(DISTINCT) rests on. Order within a
    /// group is unspecified (parallel first-seen), so both sides compare sorted.
    #[test]
    fn par_dedup_pairs_matches_serial() {
        use std::collections::HashSet;
        let n = (1usize << 18) + 12_345; // over the parallel threshold
        let mut s: u64 = 99;
        let mut gv: Vec<i64> = Vec::with_capacity(n);
        let mut vv: Vec<i64> = Vec::with_capacity(n);
        for _ in 0..n {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            gv.push(((s >> 40) % 4) as i64); // 4 groups
            vv.push(((s >> 20) % 50_000) as i64); // ~50k distinct values → lots of dupes
        }
        let (dg, dv) = super::par_dedup_pairs(
            &Int64Array::from(gv.clone()),
            &Int64Array::from(vv.clone()),
            n,
        );
        // Parallel result as a set of pairs.
        let got: HashSet<(i64, i64)> = dg.iter().copied().zip(dv.iter().copied()).collect();
        // Serial reference set.
        let want: HashSet<(i64, i64)> = gv.iter().copied().zip(vv.iter().copied()).collect();
        assert_eq!(
            got.len(),
            dg.len(),
            "parallel path emitted a duplicate pair"
        );
        assert_eq!(got, want, "parallel distinct set differs from serial");
    }

    /// The counting sort in `bucket_values_into_list` must reproduce the vector-of-vectors
    /// it replaced **exactly** — same group order, same order within a group. Within-group
    /// order is not free to change: the median/quantile path buckets through this helper
    /// too, and `COUNT(DISTINCT)`'s own offsets are the answer it finalizes. The reference
    /// here is the old implementation, kept verbatim as the oracle.
    #[test]
    fn bucketing_matches_the_vector_of_vectors_it_replaced() {
        fn reference(
            group_ids: &Int64Array,
            n_values: usize,
            num_groups: usize,
        ) -> (Vec<u32>, Vec<i32>) {
            let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); num_groups];
            for i in 0..group_ids.len() {
                buckets[group_ids.value(i) as usize].push(i as u32);
            }
            let mut order: Vec<u32> = Vec::with_capacity(n_values);
            let mut offsets: Vec<i32> = vec![0];
            for bucket in &buckets {
                order.extend_from_slice(bucket);
                offsets.push(order.len() as i32);
            }
            (order, offsets)
        }

        let mut s: u64 = 4242;
        // Include an empty group set, a single group, and a near-unique key — the shape the
        // counting sort exists for — plus groups that receive no rows at all.
        for (n, num_groups) in [
            (0usize, 0usize),
            (0, 5),
            (1, 1),
            (37, 4),
            (5_000, 4_997),
            (20_000, 64),
        ] {
            let mut gv: Vec<i64> = Vec::with_capacity(n);
            for _ in 0..n {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                gv.push(if num_groups == 0 {
                    0
                } else {
                    ((s >> 33) as usize % num_groups) as i64
                });
            }
            let groups = Int64Array::from(gv);
            let values: ArrayRef = Arc::new(Int64Array::from(
                (0..n).map(|i| (i as i64) * 7).collect::<Vec<_>>(),
            ));
            let (want_order, want_offsets) = reference(&groups, n, num_groups);

            let list = super::bucket_values_into_list(&groups, &values, num_groups).unwrap();
            let list = list.as_list::<i32>();
            assert_eq!(
                list.value_offsets(),
                &want_offsets[..],
                "offsets differ (n={n}, groups={num_groups})"
            );
            // The gathered child must equal the reference gather through `want_order`.
            let want_child = take(values.as_ref(), &UInt32Array::from(want_order), None).unwrap();
            assert_eq!(
                list.values().as_ref(),
                want_child.as_ref(),
                "bucketed values differ (n={n}, groups={num_groups})"
            );
        }
    }
}

#[cfg(test)]
mod scratch_timing {
    use super::*;
    use std::time::Instant;

    #[test]
    #[ignore]
    fn time_ungrouped() {
        let n = 8_000_000usize;
        let card = 5_000_000i64;
        let mut s: u64 = 7;
        let vals: Vec<i64> = (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((s >> 20) % card as u64) as i64
            })
            .collect();
        let values: ArrayRef = Arc::new(Int64Array::from(vals.clone()));
        let gids = vec![0u32; n];

        let t = Instant::now();
        let st = distinct_state(&values, &gids, 1).unwrap();
        println!(
            "distinct_state(num_groups=1): {:?} -> len {}",
            t.elapsed(),
            st.len()
        );

        // sub-parts
        let grp: ArrayRef = Arc::new(Int64Array::from(vec![0i64; n]));
        let t = Instant::now();
        let (dg, dv) = par_dedup_pairs(
            grp.as_primitive::<Int64Type>(),
            values.as_any().downcast_ref::<Int64Array>().unwrap(),
            n,
        );
        println!(
            "  par_dedup_pairs: {:?} -> {} distinct",
            t.elapsed(),
            dg.len()
        );

        let dvals: ArrayRef = Arc::new(Int64Array::from(dv));
        let t = Instant::now();
        let l = bucket_values_into_list(&Int64Array::from(dg), &dvals, 1).unwrap();
        println!(
            "  bucket_values_into_list: {:?} -> {}",
            t.elapsed(),
            l.len()
        );

        let t = Instant::now();
        let m = merge_distinct(&st, &[0u32], 1).unwrap();
        println!("merge_distinct: {:?} -> {}", t.elapsed(), m.len());
    }

    /// Simulate the ungrouped executor path: per-morsel partials (parallel) then combine.
    #[test]
    #[ignore]
    fn time_ungrouped_pipeline() {
        use crate::agg::{combine_with, finalize, partial, AggCall, AggFunc};
        use arrow::datatypes::{Field, Schema};
        use rayon::prelude::*;

        let n = 8_000_000usize;
        let card = 5_000_000i64;
        let morsel = 16_384usize;
        let mut s: u64 = 7;
        let vals: Vec<i64> = (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((s >> 20) % card as u64) as i64
            })
            .collect();
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, true)]));
        let morsels: Vec<RecordBatch> = vals
            .chunks(morsel)
            .map(|c| {
                let a: ArrayRef = Arc::new(Int64Array::from(c.to_vec()));
                RecordBatch::try_new(schema.clone(), vec![a]).unwrap()
            })
            .collect();
        println!("morsels: {}", morsels.len());

        let t = Instant::now();
        let partials: Vec<_> = morsels
            .par_iter()
            .map(|b| {
                let v: ArrayRef = b.column(0).clone();
                partial(
                    &[],
                    &[AggCall::new(AggFunc::CountDistinct, Some(v))],
                    b.num_rows(),
                )
            })
            .collect::<Result<_, _>>()
            .unwrap();
        println!("partial (parallel): {:?}", t.elapsed());

        let t = Instant::now();
        let merged = combine_with(&partials, &[AggFunc::CountDistinct], 200_000).unwrap();
        println!("combine_with: {:?}", t.elapsed());

        let t = Instant::now();
        let out = finalize(&[AggFunc::CountDistinct], &merged).unwrap();
        println!("finalize: {:?} -> {:?}", t.elapsed(), out[0].len());
    }
}
