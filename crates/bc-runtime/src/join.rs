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
use arrow::row::{OwnedRow, RowConverter, Rows, SortField};
use bc_sketches::BloomFilter;
use hashbrown::hash_table::Entry;
use hashbrown::HashTable;
use indexmap::IndexMap;

use crate::error::RuntimeError;

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

/// Build a chained hash table over the right (build) side, probe with the left, and
/// emit the index pairs. Shared by every key representation via [`JoinKeys`]; the
/// loop — including null handling, bloom pre-filtering, chain walking, and unmatched
/// emission for each join type — is written exactly once, so all paths agree.
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
    // `heads` maps a key to the head of a singly-linked chain of right-row indices
    // sharing that key; `next` threads the rest (`u32::MAX` terminates). Null-key rows
    // are skipped — they never match (NULL ≠ NULL).
    let state = ahash::RandomState::with_seeds(0x9E37, 0x79B9, 0x7F4A, 0x7C15);
    let mut heads: HashTable<u32> = HashTable::with_capacity(right_rows);
    let mut next: Vec<u32> = vec![u32::MAX; right_rows];
    // Probe-side bloom over the build keys (when it pays — see `use_probe_bloom_with`).
    // Built in the same pass that hashes each build row, so it adds no extra hashing.
    let mut bloom = use_bloom.then(|| BloomFilter::with_params(right_rows as u64, bloom_fp_rate));
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

    // Probe with the left side. Pre-size outputs to the left row count — the lower
    // bound for inner/left; outer and duplicate-key cases grow from there.
    let mut left_out: Vec<Option<u32>> = Vec::with_capacity(left_rows);
    let mut right_out: Vec<Option<u32>> = Vec::with_capacity(left_rows);
    let mut right_matched = vec![false; right_rows];

    let emit_right_unmatched = matches!(join_type, JoinType::Right | JoinType::Full);
    let emit_left_unmatched = matches!(join_type, JoinType::Left | JoinType::Full);

    for (i, &is_null) in left_null.iter().enumerate() {
        // Chain head for this left key — `None` for a null key or no match. A present
        // head always points at a real right row, so `is_some()` means "≥1 match".
        let head = if is_null {
            None
        } else {
            let hash = keys.hash_left(&state, i);
            // A bloom miss is definitive (no false negatives): the key is not on the
            // build side, so the chain is provably empty — skip the hash-table lookup.
            if bloom.as_ref().is_some_and(|b| !b.contains_hash(hash)) {
                None
            } else {
                heads
                    .find(hash, |&h| keys.right_eq_left(h as usize, i))
                    .copied()
            }
        };

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
                        right_matched[r as usize] = true;
                        left_out.push(Some(i as u32));
                        right_out.push(Some(r));
                        let nxt = next[r as usize];
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

    if emit_right_unmatched {
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
/// [`JoinIndices`] relation** as [`hash_join_indices`] for every join type (output
/// order differs — these are unordered relations). The win is no hash table: both
/// sides stream in key order, so it suits two large (or already-sorted) inputs the
/// way Spark's default join does. NULL keys never match (`NULL ≠ NULL`).
pub fn sort_merge_join_indices(
    left_keys: &[ArrayRef],
    right_keys: &[ArrayRef],
    join_type: JoinType,
) -> Result<JoinIndices, RuntimeError> {
    use std::cmp::Ordering;

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

    let mut left_out: Vec<Option<u32>> = Vec::new();
    let mut right_out: Vec<Option<u32>> = Vec::new();
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
