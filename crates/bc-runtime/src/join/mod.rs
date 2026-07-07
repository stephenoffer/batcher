//! Hash join — produces match index-pairs, built to distribute.
//!
//! The join computes two row-index vectors (`left`, `right`) describing the
//! output: output column `c` is `take(side_c, side_indices)`. Unmatched rows on
//! the null-supplying side get a null index (arrow `take` yields null), which is
//! exactly how outer joins are expressed — so inner/left/right/full/semi/anti all
//! fall out of one index-pair builder.
//!
//! Distribution: because matching is purely by key equality, a global join equals
//! the union of per-partition joins when both sides are hash-partitioned by the
//! join key. This module is the partition-local primitive the shuffle layer calls;
//! it carries no single-node assumptions.
//!
//! Keys are encoded with arrow's row format (multi-key, any type). SQL null
//! semantics are honored: a row with any null key never matches (NULL ≠ NULL).

use arrow::array::{Array, ArrayRef, UInt32Array};
use arrow::buffer::NullBuffer;
use arrow::row::{RowConverter, Rows, SortField};
use bc_sketches::BloomFilter;
use hashbrown::hash_table::Entry;
use hashbrown::HashTable;
use rayon::prelude::*;

use crate::error::RuntimeError;

mod asof;
mod sort_merge;

pub use asof::asof_join_indices;
pub use sort_merge::sort_merge_join_indices;

/// False-positive rate for the probe-side runtime bloom (see [`use_probe_bloom_with`]).
/// At 1% a bloom costs ~1.2 bytes/key — far less than the ~9 bytes/entry chained
/// hash table — so it stays cache-resident after the table spills.
const BLOOM_FP_RATE: f64 = 0.01;

/// Build-row floor above which a probe-side bloom pre-filter pays for itself.
///
/// For a small build side the chained hash table is cache-resident and a probe
/// lookup is already cheap, so a bloom only adds work. Above this size the table
/// spills L2/L3 and a probe lookup becomes a random cache miss the compact bloom
/// can skip for non-matching keys. ~64K build rows ≈ a ~600 KB hash table, past
/// typical L2. Tunable; the conservative side never regresses the small-join case.
const BLOOM_MIN_BUILD_ROWS: usize = 1 << 16;

/// Whether a probe-side bloom pre-filter pays for a hash join of these sizes.
///
/// Engage only when the build side is large enough to spill cache *and* the probe
/// side is at least as large, so the per-probe-row saving on non-matching keys
/// amortizes the one-pass cost of building the bloom over the build side. The bloom
/// has no false negatives, so it can only ever skip a provably-empty chain — the
/// join result is identical whether it is used or not.
///
/// `min_build_rows` is the build-row floor (the default [`BLOOM_MIN_BUILD_ROWS`], or
/// the control plane's tuning).
fn use_probe_bloom_with(build_rows: usize, probe_rows: usize, min_build_rows: usize) -> bool {
    build_rows >= min_build_rows && probe_rows >= build_rows
}

/// Extra resident bytes a hash-join build side costs *beyond* its Arrow columns.
///
/// The build phase allocates a chained hash table over the right side — a
/// `HashTable<u32>` of ~`rows` entries (hashbrown holds them at a 7/8 load factor,
/// one control byte each), a `next: Vec<u32>` chain, and a per-row null mask — none
/// of which `RecordBatch::get_array_memory_size` (columns only) counts. On narrow
/// keys that hidden overhead is 2–10× the column bytes, so an admission estimate
/// based on columns alone undercounts the resident build table and can OOM before
/// spilling. This is a tight, measured estimate (not worst case) so it never spills
/// an in-memory join that would actually have fit.
pub fn estimate_build_bytes(rows: usize) -> usize {
    // heads (u32 slot + control byte at the load factor) + next (u32) + null mask (1B).
    rows.saturating_mul(2 * std::mem::size_of::<u32>() + 4)
}

/// Join flavors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    /// Left-semi: left rows that have ≥1 match (left columns only).
    Semi,
    /// Left-anti: left rows that have no match (left columns only).
    Anti,
}

/// Row indices describing the join output. Nullable: a null entry means "no row
/// on this side" (the null-supplying side of an outer join).
pub struct JoinIndices {
    pub left: UInt32Array,
    pub right: UInt32Array,
}

/// Compute join output indices from the (pre-evaluated) key columns of each side.
///
/// The hash table is built on the **right** side and probed with the **left**,
/// so left-outer/semi/anti stream the left side; right/full additionally emit
/// right rows that never matched. A probe-side bloom pre-filter is engaged
/// automatically for selective large joins ([`use_probe_bloom_with`]).
pub fn hash_join_indices(
    left_keys: &[ArrayRef],
    right_keys: &[ArrayRef],
    join_type: JoinType,
) -> Result<JoinIndices, RuntimeError> {
    hash_join_indices_with(
        left_keys,
        right_keys,
        join_type,
        BLOOM_FP_RATE,
        BLOOM_MIN_BUILD_ROWS,
    )
}

/// [`hash_join_indices`] with the probe-bloom knobs supplied by the caller.
///
/// `bloom_fp_rate` and `bloom_min_build_rows` are performance-only: the bloom has
/// no false negatives, so it can only skip a provably-empty chain — the produced
/// [`JoinIndices`] relation is identical for any setting. The parallel executor
/// threads these from the control plane's tuning.
pub fn hash_join_indices_with(
    left_keys: &[ArrayRef],
    right_keys: &[ArrayRef],
    join_type: JoinType,
    bloom_fp_rate: f64,
    bloom_min_build_rows: usize,
) -> Result<JoinIndices, RuntimeError> {
    let left_rows = left_keys.first().map_or(0, |a| a.len());
    let right_rows = right_keys.first().map_or(0, |a| a.len());
    let use_bloom = use_probe_bloom_with(right_rows, left_rows, bloom_min_build_rows);
    hash_join_indices_impl(left_keys, right_keys, join_type, use_bloom, bloom_fp_rate)
}

/// The hash-join index builder, with the probe-side bloom pre-filter made explicit.
///
/// `use_bloom` is decided by [`use_probe_bloom_with`] on the public path; tests drive it
/// both ways to prove the bloom is a pure performance short-circuit (the produced
/// [`JoinIndices`] relation is identical with the filter on or off).
pub(crate) fn hash_join_indices_impl(
    left_keys: &[ArrayRef],
    right_keys: &[ArrayRef],
    join_type: JoinType,
    use_bloom: bool,
    bloom_fp_rate: f64,
) -> Result<JoinIndices, RuntimeError> {
    let left_rows = left_keys.first().map_or(0, |a| a.len());
    let right_rows = right_keys.first().map_or(0, |a| a.len());
    let left_null = null_mask(left_keys, left_rows);
    let right_null = null_mask(right_keys, right_rows);

    // Fast path: a single integer key column hashes/compares its native values
    // directly, skipping the `RowConverter` encoding pass (a per-row allocation +
    // copy, plus a byte-slice compare on every chain walk) the general path needs for
    // multi-column / variable-length / float keys. This is the dominant join shape in
    // analytical workloads (a fact joined to a dimension on an integer id). The
    // *same* `build_probe` loop drives both paths, so the int path is bit-identical to
    // the row-encoded oracle by construction — only the key accessor differs.
    if let Some(keys) = I64Keys::try_new(left_keys, right_keys) {
        return Ok(build_probe(
            &keys,
            left_rows,
            right_rows,
            &left_null,
            &right_null,
            join_type,
            use_bloom,
            bloom_fp_rate,
        ));
    }

    // Two-`Int64`-key fast path: a composite integer key (e.g. a fact joined on
    // `(part_id, supplier_id)`, TPC-H Q9's `partsupp ⋈ lineitem`) hashes/compares the
    // raw `(i64, i64)` pair, skipping the `RowConverter` encoding the general path runs
    // over every row. Same `build_probe` loop → bit-identical to the row-encoded oracle.
    if let Some(keys) = I64x2Keys::try_new(left_keys, right_keys) {
        return Ok(build_probe(
            &keys,
            left_rows,
            right_rows,
            &left_null,
            &right_null,
            join_type,
            use_bloom,
            bloom_fp_rate,
        ));
    }

    // General path: encode both sides' keys with one shared converter (types align).
    let fields: Vec<SortField> = right_keys
        .iter()
        .map(|a| SortField::new(a.data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields)?;
    let keys = RowKeys {
        right: converter.convert_columns(right_keys)?,
        left: converter.convert_columns(left_keys)?,
    };
    Ok(build_probe(
        &keys,
        left_rows,
        right_rows,
        &left_null,
        &right_null,
        join_type,
        use_bloom,
        bloom_fp_rate,
    ))
}

/// Key access for the hash-join build/probe: how to hash a build (right) or probe
/// (left) row and how to compare two rows for equality. One trait, two
/// implementations (row-encoded for any type, raw `i64` for the integer fast path),
/// so a single [`build_probe`] loop produces an identical result either way — equal
/// keys on the two sides hash equally within an implementation, which is all the join
/// needs.
trait JoinKeys {
    fn hash_right(&self, state: &ahash::RandomState, i: usize) -> u64;
    fn hash_left(&self, state: &ahash::RandomState, i: usize) -> u64;
    /// Whether build rows `a` and `b` carry the same key (chain comparison).
    fn right_eq_right(&self, a: usize, b: usize) -> bool;
    /// Whether build row `r` and probe row `l` carry the same key (probe comparison).
    fn right_eq_left(&self, r: usize, l: usize) -> bool;
}

/// Row-encoded keys (the general path): equal keys produce equal byte rows, so they
/// hash and compare correctly across any number of columns and any type.
struct RowKeys {
    right: Rows,
    left: Rows,
}

impl JoinKeys for RowKeys {
    fn hash_right(&self, state: &ahash::RandomState, i: usize) -> u64 {
        state.hash_one(self.right.row(i))
    }
    fn hash_left(&self, state: &ahash::RandomState, i: usize) -> u64 {
        state.hash_one(self.left.row(i))
    }
    fn right_eq_right(&self, a: usize, b: usize) -> bool {
        self.right.row(a) == self.right.row(b)
    }
    fn right_eq_left(&self, r: usize, l: usize) -> bool {
        self.right.row(r) == self.left.row(l)
    }
}

/// Raw `i64` keys (the fast path): a single `Int64` key column on each side. Narrow
/// integer types are normalized to `Int64` at the FFI boundary, so this covers the
/// common integer-id join without the row-format detour.
struct I64Keys<'a> {
    right: &'a [i64],
    left: &'a [i64],
}

impl<'a> I64Keys<'a> {
    /// `Some` when both sides are exactly one `Int64` column — the fast-path shape.
    fn try_new(left_keys: &'a [ArrayRef], right_keys: &'a [ArrayRef]) -> Option<Self> {
        use arrow::array::Int64Array;
        use arrow::datatypes::DataType;
        if left_keys.len() != 1 || right_keys.len() != 1 {
            return None;
        }
        if left_keys[0].data_type() != &DataType::Int64
            || right_keys[0].data_type() != &DataType::Int64
        {
            return None;
        }
        let left = left_keys[0].as_any().downcast_ref::<Int64Array>()?;
        let right = right_keys[0].as_any().downcast_ref::<Int64Array>()?;
        // `values()` is the raw slice; null rows are masked out by the null mask in
        // `build_probe`, so a null slot's arbitrary value is never hashed or compared.
        Some(Self {
            right: right.values(),
            left: left.values(),
        })
    }
}

impl JoinKeys for I64Keys<'_> {
    fn hash_right(&self, state: &ahash::RandomState, i: usize) -> u64 {
        state.hash_one(self.right[i])
    }
    fn hash_left(&self, state: &ahash::RandomState, i: usize) -> u64 {
        state.hash_one(self.left[i])
    }
    fn right_eq_right(&self, a: usize, b: usize) -> bool {
        self.right[a] == self.right[b]
    }
    fn right_eq_left(&self, r: usize, l: usize) -> bool {
        self.right[r] == self.left[l]
    }
}

/// Two-`Int64`-key fast path: the raw value slices of both key columns per side.
struct I64x2Keys<'a> {
    right: (&'a [i64], &'a [i64]),
    left: (&'a [i64], &'a [i64]),
}

impl<'a> I64x2Keys<'a> {
    /// `Some` when both sides are exactly two `Int64` columns.
    fn try_new(left_keys: &'a [ArrayRef], right_keys: &'a [ArrayRef]) -> Option<Self> {
        use arrow::array::Int64Array;
        use arrow::datatypes::DataType;
        if left_keys.len() != 2 || right_keys.len() != 2 {
            return None;
        }
        if left_keys
            .iter()
            .chain(right_keys)
            .any(|k| k.data_type() != &DataType::Int64)
        {
            return None;
        }
        let col = |a: &'a ArrayRef| {
            a.as_any()
                .downcast_ref::<Int64Array>()
                .map(Int64Array::values)
        };
        Some(Self {
            right: (col(&right_keys[0])?, col(&right_keys[1])?),
            left: (col(&left_keys[0])?, col(&left_keys[1])?),
        })
    }
}

impl JoinKeys for I64x2Keys<'_> {
    fn hash_right(&self, state: &ahash::RandomState, i: usize) -> u64 {
        state.hash_one((self.right.0[i], self.right.1[i]))
    }
    fn hash_left(&self, state: &ahash::RandomState, i: usize) -> u64 {
        state.hash_one((self.left.0[i], self.left.1[i]))
    }
    fn right_eq_right(&self, a: usize, b: usize) -> bool {
        self.right.0[a] == self.right.0[b] && self.right.1[a] == self.right.1[b]
    }
    fn right_eq_left(&self, r: usize, l: usize) -> bool {
        self.right.0[r] == self.left.0[l] && self.right.1[r] == self.left.1[l]
    }
}

/// A chained hash table over a build (right) side, ready to be probed.
///
/// Built once by [`JoinTable::build`] and then probed — either in a single pass
/// (the [`build_probe`] oracle path) or **many times, read-only and in parallel**,
/// by the broadcast executor, which builds the table once and fans the probe of a
/// large left side across worker chunks instead of rebuilding the table per chunk.
/// `heads` maps a key to the head of a singly-linked chain of right-row indices
/// sharing it; `next` threads the rest (`u32::MAX` terminates). Null-key build rows
/// are never inserted (NULL ≠ NULL), so a present chain head is always a real match.
struct JoinTable {
    heads: HashTable<u32>,
    next: Vec<u32>,
    state: ahash::RandomState,
    bloom: Option<BloomFilter>,
}

impl JoinTable {
    /// Build the chained hash table over the right (build) side. The optional probe
    /// bloom is populated in this same pass (no extra hashing) — see
    /// [`use_probe_bloom_with`].
    fn build<K: JoinKeys>(
        keys: &K,
        right_rows: usize,
        right_null: &[bool],
        use_bloom: bool,
        bloom_fp_rate: f64,
    ) -> Self {
        let state = ahash::RandomState::with_seeds(0x9E37, 0x79B9, 0x7F4A, 0x7C15);
        let mut heads: HashTable<u32> = HashTable::with_capacity(right_rows);
        let mut next: Vec<u32> = vec![u32::MAX; right_rows];
        let mut bloom =
            use_bloom.then(|| BloomFilter::with_params(right_rows as u64, bloom_fp_rate));
        for (i, &is_null) in right_null.iter().enumerate() {
            if is_null {
                continue;
            }
            let hash = keys.hash_right(&state, i);
            if let Some(b) = bloom.as_mut() {
                b.add_hash(hash);
            }
            match heads.entry(
                hash,
                |&h| keys.right_eq_right(h as usize, i),
                |&h| keys.hash_right(&state, h as usize),
            ) {
                // Prepend i to the chain — order within a key is irrelevant (the join
                // output is an unordered relation).
                Entry::Occupied(mut e) => {
                    next[i] = *e.get();
                    *e.get_mut() = i as u32;
                }
                Entry::Vacant(e) => {
                    e.insert(i as u32);
                }
            }
        }
        Self {
            heads,
            next,
            state,
            bloom,
        }
    }

    /// The chain head for probe (left) row `l` — `None` for a null key, a bloom miss,
    /// or no match; otherwise a real right-row index (`is_some()` ⇒ ≥1 match).
    #[inline]
    fn head_for<K: JoinKeys>(&self, keys: &K, l: usize, is_null: bool) -> Option<u32> {
        if is_null {
            return None;
        }
        let hash = keys.hash_left(&self.state, l);
        // A bloom miss is definitive (no false negatives): the key is not on the build
        // side, so the chain is provably empty — skip the hash-table lookup.
        if self.bloom.as_ref().is_some_and(|b| !b.contains_hash(hash)) {
            return None;
        }
        self.heads
            .find(hash, |&h| keys.right_eq_left(h as usize, l))
            .copied()
    }

    /// Probe left rows `range` against the table, appending index pairs for the
    /// **left-driven** join types (Inner/Left/Semi/Anti). `right_matched`, when
    /// supplied, records which build rows were hit (for the Right/Full unmatched
    /// emission the single-pass oracle does after probing). The broadcast path passes
    /// `None` — it never emits build-side-unmatched rows (Full/Right run single-pass).
    #[allow(clippy::too_many_arguments)]
    fn probe_range<K: JoinKeys>(
        &self,
        keys: &K,
        range: std::ops::Range<usize>,
        left_null: &[bool],
        join_type: JoinType,
        left_out: &mut Vec<Option<u32>>,
        right_out: &mut Vec<Option<u32>>,
        mut right_matched: Option<&mut [bool]>,
    ) {
        let emit_left_unmatched = matches!(join_type, JoinType::Left | JoinType::Full);
        for i in range {
            let head = self.head_for(keys, i, left_null[i]);
            match join_type {
                JoinType::Semi => {
                    if head.is_some() {
                        left_out.push(Some(i as u32));
                        right_out.push(None);
                    }
                }
                JoinType::Anti => {
                    if head.is_none() {
                        left_out.push(Some(i as u32));
                        right_out.push(None);
                    }
                }
                _ => match head {
                    Some(mut r) => {
                        // Walk the chain of right rows sharing this key.
                        loop {
                            if let Some(rm) = right_matched.as_deref_mut() {
                                rm[r as usize] = true;
                            }
                            left_out.push(Some(i as u32));
                            right_out.push(Some(r));
                            let nxt = self.next[r as usize];
                            if nxt == u32::MAX {
                                break;
                            }
                            r = nxt;
                        }
                    }
                    None => {
                        if emit_left_unmatched {
                            left_out.push(Some(i as u32));
                            right_out.push(None);
                        }
                    }
                },
            }
        }
    }
}

/// Build a chained hash table over the right (build) side, probe with the left, and
/// emit the index pairs. Shared by every key representation via [`JoinKeys`]; the
/// build and probe loops live in [`JoinTable`] (so the broadcast executor's
/// build-once/probe-many path runs exactly the same code) and are composed here once,
/// so every join path agrees.
#[allow(clippy::too_many_arguments)]
fn build_probe<K: JoinKeys>(
    keys: &K,
    left_rows: usize,
    right_rows: usize,
    left_null: &[bool],
    right_null: &[bool],
    join_type: JoinType,
    use_bloom: bool,
    bloom_fp_rate: f64,
) -> JoinIndices {
    let table = JoinTable::build(keys, right_rows, right_null, use_bloom, bloom_fp_rate);

    // Probe with the left side. Pre-size outputs to the left row count — the lower
    // bound for inner/left; outer and duplicate-key cases grow from there.
    let mut left_out: Vec<Option<u32>> = Vec::with_capacity(left_rows);
    let mut right_out: Vec<Option<u32>> = Vec::with_capacity(left_rows);
    let emit_right_unmatched = matches!(join_type, JoinType::Right | JoinType::Full);
    let mut right_matched = emit_right_unmatched.then(|| vec![false; right_rows]);

    table.probe_range(
        keys,
        0..left_rows,
        left_null,
        join_type,
        &mut left_out,
        &mut right_out,
        right_matched.as_deref_mut(),
    );

    if let Some(right_matched) = right_matched {
        // Every unmatched right row is preserved — including null-key rows, which
        // match nothing (NULL != NULL) but are still part of the right relation.
        for (r, matched) in right_matched.iter().enumerate() {
            if !matched {
                left_out.push(None);
                right_out.push(Some(r as u32));
            }
        }
    }

    JoinIndices {
        left: UInt32Array::from(left_out),
        right: UInt32Array::from(right_out),
    }
}

/// Build the hash table over the right (build) side **once**, then probe the left
/// side in `n_chunks` parallel row-range slices — the broadcast join's core.
///
/// Returns one [`JoinIndices`] per chunk (left/right indices absolute into the full
/// sides), in chunk order, so the caller gathers each chunk's output against the full
/// batches and concatenates. This is what lets a broadcast join probe a large left
/// side across every core *without* rebuilding the (replicated) build table per
/// chunk and *without* shuffling either side by key.
///
/// Only the **left-driven** join types are valid here (Inner/Left/Semi/Anti): each
/// left row lands in exactly one chunk, and no build-side-unmatched rows are emitted
/// (Right/Full must coordinate across all chunks, so the executor runs them
/// single-pass). The produced relation (unioned over chunks) is identical to
/// [`hash_join_indices`] — same build, same probe loop, only the probe is sliced.
pub fn broadcast_hash_join_indices(
    left_keys: &[ArrayRef],
    right_keys: &[ArrayRef],
    join_type: JoinType,
    n_chunks: usize,
    bloom_fp_rate: f64,
    bloom_min_build_rows: usize,
) -> Result<Vec<JoinIndices>, RuntimeError> {
    debug_assert!(
        matches!(
            join_type,
            JoinType::Inner | JoinType::Left | JoinType::Semi | JoinType::Anti
        ),
        "broadcast probe is left-driven only; Right/Full run single-pass"
    );
    let left_rows = left_keys.first().map_or(0, |a| a.len());
    let right_rows = right_keys.first().map_or(0, |a| a.len());
    let left_null = null_mask(left_keys, left_rows);
    let right_null = null_mask(right_keys, right_rows);
    let use_bloom = use_probe_bloom_with(right_rows, left_rows, bloom_min_build_rows);

    // The chunk row-range boundaries (near-equal, contiguous, covering 0..left_rows).
    let chunks = n_chunks.max(1);
    let per = left_rows.div_ceil(chunks).max(1);
    let ranges: Vec<std::ops::Range<usize>> = (0..left_rows)
        .step_by(per)
        .map(|s| s..(s + per).min(left_rows))
        .collect();

    // One key representation, one build, then a parallel sliced probe. The macro keeps
    // the three key shapes (single i64, two i64, row-encoded) on the same code without
    // boxing a `dyn JoinKeys` (the hot probe loop must monomorphize).
    macro_rules! run {
        ($keys:expr) => {{
            let table = JoinTable::build(&$keys, right_rows, &right_null, use_bloom, bloom_fp_rate);
            ranges
                .par_iter()
                .map(|r| {
                    let mut left_out: Vec<Option<u32>> = Vec::with_capacity(r.len());
                    let mut right_out: Vec<Option<u32>> = Vec::with_capacity(r.len());
                    table.probe_range(
                        &$keys,
                        r.clone(),
                        &left_null,
                        join_type,
                        &mut left_out,
                        &mut right_out,
                        None,
                    );
                    JoinIndices {
                        left: UInt32Array::from(left_out),
                        right: UInt32Array::from(right_out),
                    }
                })
                .collect()
        }};
    }

    if let Some(keys) = I64Keys::try_new(left_keys, right_keys) {
        return Ok(run!(keys));
    }
    if let Some(keys) = I64x2Keys::try_new(left_keys, right_keys) {
        return Ok(run!(keys));
    }
    let fields: Vec<SortField> = right_keys
        .iter()
        .map(|a| SortField::new(a.data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields)?;
    let keys = RowKeys {
        right: converter.convert_columns(right_keys)?,
        left: converter.convert_columns(left_keys)?,
    };
    Ok(run!(keys))
}

/// A per-row mask: true where ANY key column is null (such rows never match).
///
/// Combines the key columns' validity bitmaps word-wise via `NullBuffer::union`
/// (which intersects validity — a row stays valid only if valid in every column),
/// rather than a per-row `is_null` call per column. Columns with no null buffer
/// contribute nothing, so the all-non-null case allocates one zeroed mask and does
/// no bit work.
fn null_mask(keys: &[ArrayRef], rows: usize) -> Vec<bool> {
    let mut combined: Option<NullBuffer> = None;
    for key in keys {
        if key.null_count() != 0 {
            combined = NullBuffer::union(combined.as_ref(), key.nulls());
        }
    }
    match combined {
        None => vec![false; rows],
        Some(nulls) => (0..rows).map(|i| nulls.is_null(i)).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow::array::Int64Array;
    use arrow::datatypes::DataType;

    fn keys(v: &[i64]) -> Vec<ArrayRef> {
        vec![Arc::new(Int64Array::from(v.to_vec())) as ArrayRef]
    }

    fn pairs(idx: &JoinIndices) -> Vec<(Option<u32>, Option<u32>)> {
        (0..idx.left.len())
            .map(|i| {
                (
                    idx.left.is_valid(i).then(|| idx.left.value(i)),
                    idx.right.is_valid(i).then(|| idx.right.value(i)),
                )
            })
            .collect()
    }

    #[test]
    fn inner_join_pairs() {
        // left  = [1,2,3], right = [2,3,3]
        let idx = hash_join_indices(&keys(&[1, 2, 3]), &keys(&[2, 3, 3]), JoinType::Inner).unwrap();
        let mut got = pairs(&idx);
        got.sort();
        // 2 matches right#0; 3 matches right#1 and #2.
        assert_eq!(
            got,
            vec![(Some(1), Some(0)), (Some(2), Some(1)), (Some(2), Some(2))]
        );
    }

    #[test]
    fn left_join_keeps_unmatched() {
        let idx = hash_join_indices(&keys(&[1, 2]), &keys(&[2]), JoinType::Left).unwrap();
        let mut got = pairs(&idx);
        got.sort();
        assert_eq!(got, vec![(Some(0), None), (Some(1), Some(0))]);
    }

    #[test]
    fn semi_and_anti() {
        let semi = hash_join_indices(&keys(&[1, 2, 3]), &keys(&[2, 3]), JoinType::Semi).unwrap();
        let mut s: Vec<_> = pairs(&semi).into_iter().map(|(l, _)| l).collect();
        s.sort();
        assert_eq!(s, vec![Some(1), Some(2)]);

        let anti = hash_join_indices(&keys(&[1, 2, 3]), &keys(&[2, 3]), JoinType::Anti).unwrap();
        let a: Vec<_> = pairs(&anti).into_iter().map(|(l, _)| l).collect();
        assert_eq!(a, vec![Some(0)]);
    }

    #[test]
    fn full_join_emits_both_unmatched() {
        let idx = hash_join_indices(&keys(&[1, 2]), &keys(&[2, 3]), JoinType::Full).unwrap();
        let mut got = pairs(&idx);
        got.sort();
        // 1 unmatched (left), 2 matches, 3 unmatched (right).
        assert_eq!(
            got,
            vec![(None, Some(1)), (Some(0), None), (Some(1), Some(0))]
        );
    }

    #[test]
    fn null_keys_never_match() {
        let left: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![Some(1), None]))];
        let right: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![None, Some(1)]))];
        let idx = hash_join_indices(&left, &right, JoinType::Inner).unwrap();
        // Only 1==1 matches; the null rows do not.
        assert_eq!(pairs(&idx), vec![(Some(0), Some(1))]);
    }

    #[test]
    fn duplicate_keys_cross_product() {
        // Both sides repeat key 5: left#0,#2 × right#0,#1,#3 = 2×3 = 6 pairs, plus
        // the lone 7==7 match. Exercises the build-side chain walk on duplicates.
        let left = keys(&[5, 9, 5, 7]);
        let right = keys(&[5, 5, 8, 5, 7]);
        let idx = hash_join_indices(&left, &right, JoinType::Inner).unwrap();
        let mut got = pairs(&idx);
        got.sort();
        let mut want = vec![(Some(3), Some(4))];
        for l in [0u32, 2] {
            for r in [0u32, 1, 3] {
                want.push((Some(l), Some(r)));
            }
        }
        want.sort();
        assert_eq!(got, want);
    }

    #[test]
    fn multi_key_join() {
        // Composite key (a, b): left rows (1,2)/(1,3)/(2,2) vs right (1,2)/(2,2)/(1,9).
        // (1,2)==right#0 and (2,2)==right#1 match; (1,3) and (1,9) share no full key.
        let left: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(vec![1, 1, 2])),
            Arc::new(Int64Array::from(vec![2, 3, 2])),
        ];
        let right: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(vec![1, 2, 1])),
            Arc::new(Int64Array::from(vec![2, 2, 9])),
        ];
        let idx = hash_join_indices(&left, &right, JoinType::Inner).unwrap();
        let mut got = pairs(&idx);
        got.sort();
        assert_eq!(got, vec![(Some(0), Some(0)), (Some(2), Some(1))]);
    }

    fn sorted_pairs(idx: &JoinIndices) -> Vec<(Option<u32>, Option<u32>)> {
        let mut p = pairs(idx);
        p.sort();
        p
    }

    /// The single-`Int64`-key fast path (`I64Keys`) must produce exactly the relation
    /// the row-encoded path (`RowKeys`) does, for every join type — including duplicate
    /// keys (cross products), unmatched rows, and null keys. Driving `build_probe` with
    /// each key implementation over the same inputs pins that equivalence directly.
    #[test]
    fn i64_fast_path_matches_row_encoded() {
        let left: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![
            Some(1),
            Some(2),
            Some(2),
            None,
            Some(7),
        ]))];
        let right: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![
            Some(2),
            Some(2),
            Some(3),
            None,
            Some(1),
        ]))];
        let ln = null_mask(&left, 5);
        let rn = null_mask(&right, 5);
        let i64keys = I64Keys::try_new(&left, &right).expect("both single Int64");
        let fields = vec![SortField::new(DataType::Int64)];
        let conv = RowConverter::new(fields).unwrap();
        let rowkeys = RowKeys {
            right: conv.convert_columns(&right).unwrap(),
            left: conv.convert_columns(&left).unwrap(),
        };
        for jt in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
            JoinType::Semi,
            JoinType::Anti,
        ] {
            // Exercise both with and without the probe bloom (a pure short-circuit).
            for bloom in [false, true] {
                let fast = build_probe(&i64keys, 5, 5, &ln, &rn, jt, bloom, BLOOM_FP_RATE);
                let slow = build_probe(&rowkeys, 5, 5, &ln, &rn, jt, bloom, BLOOM_FP_RATE);
                assert_eq!(
                    sorted_pairs(&fast),
                    sorted_pairs(&slow),
                    "i64 vs row mismatch for {jt:?} bloom={bloom}"
                );
            }
        }
    }

    /// The two-`Int64`-key fast path (`I64x2Keys`) must match the row-encoded oracle
    /// for every join type, including a partial-key match (first component equal,
    /// second differs → no match), duplicate composite keys, and null keys.
    #[test]
    fn i64x2_fast_path_matches_row_encoded() {
        let left: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(vec![
                Some(1),
                Some(1),
                Some(2),
                None,
                Some(7),
            ])),
            Arc::new(Int64Array::from(vec![
                Some(2),
                Some(3),
                Some(2),
                Some(5),
                None,
            ])),
        ];
        let right: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(vec![
                Some(1),
                Some(2),
                Some(1),
                None,
                Some(7),
            ])),
            Arc::new(Int64Array::from(vec![
                Some(2),
                Some(2),
                Some(9),
                Some(5),
                None,
            ])),
        ];
        let ln = null_mask(&left, 5);
        let rn = null_mask(&right, 5);
        let fastkeys = I64x2Keys::try_new(&left, &right).expect("both two Int64");
        let fields = vec![
            SortField::new(DataType::Int64),
            SortField::new(DataType::Int64),
        ];
        let conv = RowConverter::new(fields).unwrap();
        let rowkeys = RowKeys {
            right: conv.convert_columns(&right).unwrap(),
            left: conv.convert_columns(&left).unwrap(),
        };
        for jt in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
            JoinType::Semi,
            JoinType::Anti,
        ] {
            for bloom in [false, true] {
                let fast = build_probe(&fastkeys, 5, 5, &ln, &rn, jt, bloom, BLOOM_FP_RATE);
                let slow = build_probe(&rowkeys, 5, 5, &ln, &rn, jt, bloom, BLOOM_FP_RATE);
                assert_eq!(
                    sorted_pairs(&fast),
                    sorted_pairs(&slow),
                    "i64x2 vs row mismatch for {jt:?} bloom={bloom}"
                );
            }
        }
    }

    /// Sort-merge join must produce the same relation as the hash-join oracle for
    /// every join type — with duplicate keys (cross products) and null keys.
    #[test]
    fn sort_merge_matches_hash_oracle() {
        let left: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![
            Some(2),
            Some(1),
            Some(2),
            None,
            Some(3),
            Some(2),
        ]))];
        let right: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![
            Some(2),
            Some(3),
            Some(2),
            Some(4),
            None,
        ]))];
        for jt in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
            JoinType::Semi,
            JoinType::Anti,
        ] {
            let hash = hash_join_indices(&left, &right, jt).unwrap();
            let smj = sort_merge_join_indices(&left, &right, jt).unwrap();
            assert_eq!(
                sorted_pairs(&hash),
                sorted_pairs(&smj),
                "sort-merge disagrees with hash for {jt:?}"
            );
        }
    }

    /// Already-ascending keys on both sides exercise the no-sort fast path
    /// (`sort_indices_if_unsorted` skips the sort); the result must still equal the
    /// hash oracle for every join type, across duplicate keys.
    #[test]
    fn sort_merge_presorted_fast_path_matches_oracle() {
        let left: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![
            Some(1),
            Some(2),
            Some(2),
            Some(3),
            Some(5),
        ]))];
        let right: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![
            Some(2),
            Some(2),
            Some(4),
            Some(5),
        ]))];
        for jt in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
            JoinType::Semi,
            JoinType::Anti,
        ] {
            let hash = hash_join_indices(&left, &right, jt).unwrap();
            let smj = sort_merge_join_indices(&left, &right, jt).unwrap();
            assert_eq!(
                sorted_pairs(&hash),
                sorted_pairs(&smj),
                "presorted sort-merge disagrees with hash for {jt:?}"
            );
        }
    }

    /// The probe-side bloom is a pure performance short-circuit: forcing it on must
    /// produce the identical join relation as forcing it off, for every join type,
    /// across duplicate and null keys. This is the seq-oracle invariant for the
    /// runtime filter — a bloom can only ever skip a provably-empty chain.
    #[test]
    fn bloom_matches_no_bloom_oracle() {
        let left: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![
            Some(2),
            Some(1),
            Some(2),
            None,
            Some(3),
            Some(2),
            Some(7),
        ]))];
        let right: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![
            Some(2),
            Some(3),
            Some(2),
            Some(4),
            None,
        ]))];
        for jt in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
            JoinType::Semi,
            JoinType::Anti,
        ] {
            let with = hash_join_indices_impl(&left, &right, jt, true, BLOOM_FP_RATE).unwrap();
            let without = hash_join_indices_impl(&left, &right, jt, false, BLOOM_FP_RATE).unwrap();
            assert_eq!(
                sorted_pairs(&with),
                sorted_pairs(&without),
                "bloom-on disagrees with bloom-off for {jt:?}"
            );
        }
    }

    /// With a build side that shares no key with most of the probe side, the bloom
    /// path prunes the bulk of probe rows yet still yields the exact inner join — the
    /// case the filter is built for (many provable misses).
    #[test]
    fn bloom_prunes_disjoint_keys_correctly() {
        let left: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(
            (0..1_000i64).collect::<Vec<_>>(),
        ))];
        let right: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![10, 20, 999, 5000]))];
        let with =
            hash_join_indices_impl(&left, &right, JoinType::Inner, true, BLOOM_FP_RATE).unwrap();
        let without =
            hash_join_indices_impl(&left, &right, JoinType::Inner, false, BLOOM_FP_RATE).unwrap();
        assert_eq!(sorted_pairs(&with), sorted_pairs(&without));
        // 10, 20, 999 are in [0,1000); 5000 is not — three matches.
        assert_eq!(with.left.len(), 3);
    }

    #[test]
    fn probe_bloom_gate_is_conservative() {
        // Tiny / balanced joins skip the bloom; a large selective probe engages it.
        assert!(!use_probe_bloom_with(10, 10, BLOOM_MIN_BUILD_ROWS));
        assert!(!use_probe_bloom_with(
            BLOOM_MIN_BUILD_ROWS,
            BLOOM_MIN_BUILD_ROWS - 1,
            BLOOM_MIN_BUILD_ROWS
        ));
        assert!(use_probe_bloom_with(
            BLOOM_MIN_BUILD_ROWS,
            BLOOM_MIN_BUILD_ROWS * 4,
            BLOOM_MIN_BUILD_ROWS
        ));
    }

    /// The broadcast probe (build the table once, probe the left side in parallel
    /// row-range chunks) must produce exactly the relation the single-pass oracle does
    /// for every left-driven join type — across chunk counts that don't divide the
    /// left evenly, duplicate keys (cross products), and null keys. Unioning the
    /// per-chunk indices and comparing as a multiset pins that equivalence.
    #[test]
    fn broadcast_probe_matches_single_pass_oracle() {
        let left = keys(&[5, 9, 5, 7, 2, 5, 1, 8, 9]);
        let right = keys(&[5, 5, 8, 9, 7]);
        for jt in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Semi,
            JoinType::Anti,
        ] {
            let oracle = hash_join_indices(&left, &right, jt).unwrap();
            // A chunk count that splits 9 left rows unevenly (1, 4, 9 chunks).
            for n_chunks in [1usize, 4, 9] {
                let parts =
                    broadcast_hash_join_indices(&left, &right, jt, n_chunks, BLOOM_FP_RATE, 1)
                        .unwrap();
                let mut got: Vec<_> = parts.iter().flat_map(pairs).collect();
                got.sort();
                assert_eq!(
                    got,
                    sorted_pairs(&oracle),
                    "broadcast disagrees with oracle for {jt:?} n_chunks={n_chunks}"
                );
            }
        }
    }

    /// Broadcast probe over a null-key-bearing input, with the bloom forced on (a
    /// large `min_build_rows` threshold of 1 makes the probe bloom engage), still
    /// equals the oracle — the parallel sliced probe honors NULL ≠ NULL identically.
    #[test]
    fn broadcast_probe_handles_nulls_and_bloom() {
        let left: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![
            Some(1),
            None,
            Some(2),
            Some(2),
            None,
            Some(3),
        ]))];
        let right: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![
            Some(2),
            None,
            Some(3),
            Some(2),
        ]))];
        for jt in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Semi,
            JoinType::Anti,
        ] {
            let oracle = hash_join_indices(&left, &right, jt).unwrap();
            let parts =
                broadcast_hash_join_indices(&left, &right, jt, 3, BLOOM_FP_RATE, 1).unwrap();
            let mut got: Vec<_> = parts.iter().flat_map(pairs).collect();
            got.sort();
            assert_eq!(got, sorted_pairs(&oracle), "broadcast+null mismatch {jt:?}");
        }
    }

    #[test]
    fn sort_merge_handles_empty_sides() {
        let empty: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(Vec::<i64>::new()))];
        let some = keys(&[1, 2, 3]);
        for jt in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
        ] {
            let h = hash_join_indices(&empty, &some, jt).unwrap();
            let s = sort_merge_join_indices(&empty, &some, jt).unwrap();
            assert_eq!(sorted_pairs(&h), sorted_pairs(&s), "empty-left {jt:?}");
            let h2 = hash_join_indices(&some, &empty, jt).unwrap();
            let s2 = sort_merge_join_indices(&some, &empty, jt).unwrap();
            assert_eq!(sorted_pairs(&h2), sorted_pairs(&s2), "empty-right {jt:?}");
        }
    }
}
