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
mod build;
mod radix;
mod sort_merge;
mod stream;

pub use asof::asof_join_indices;
pub use sort_merge::sort_merge_join_indices;
pub use stream::{streaming_supported, BroadcastProbe};

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

/// The sentinel an [`IndexBuf`] stores for a NULL join index.
///
/// A real row index can never reach it: indices are already `u32`, so a relation of
/// `u32::MAX` rows would have overflowed the join long before this. Asserted on push.
const NULL_INDEX: u32 = u32::MAX;

/// One side's join indices under construction — a flat `Vec<u32>` with a NULL sentinel.
///
/// The obvious `Vec<Option<u32>>` is **eight bytes per output row**, and there are two of
/// them: a 60 M-row join writes 960 MB of scratch, then `UInt32Array::from` reads it all
/// back to build a 240 MB values buffer and a bitmap. On a probe loop that is already
/// memory-bandwidth bound, that is most of a gigabyte of traffic spent encoding a null
/// that inner joins never emit.
///
/// Storing `u32` with a sentinel halves the scratch, and `finish` skips the null buffer
/// entirely when nothing null was ever pushed — which is every inner join, the dominant
/// analytical shape. `any_null` is a single branch-free `|=` on the hot path.
#[derive(Default)]
pub(crate) struct IndexBuf {
    idx: Vec<u32>,
    any_null: bool,
}

impl IndexBuf {
    pub(crate) fn with_capacity(rows: usize) -> Self {
        Self {
            idx: Vec::with_capacity(rows),
            any_null: false,
        }
    }

    /// Append a real row index.
    #[inline]
    fn push(&mut self, row: u32) {
        debug_assert_ne!(row, NULL_INDEX, "row index collides with the NULL sentinel");
        self.idx.push(row);
    }

    /// Append a NULL (an unmatched outer row, or the unused side of a semi/anti join).
    #[inline]
    fn push_null(&mut self) {
        self.idx.push(NULL_INDEX);
        self.any_null = true;
    }

    pub(crate) fn len(&self) -> usize {
        self.idx.len()
    }

    /// Append another buffer's indices, preserving its NULL sentinels.
    ///
    /// Used to concatenate per-partition pieces **in partition order**, which is what makes
    /// the parallel radix join emit byte-identical rows to the sequential one.
    fn extend(&mut self, other: IndexBuf) {
        self.idx.extend_from_slice(&other.idx);
        self.any_null |= other.any_null;
    }

    /// Sort the buffered indices ascending.
    ///
    /// Only meaningful for a side whose partner is all-NULL — a semi/anti join — where the
    /// pairing carries nothing to preserve and the row order is the *only* information in the
    /// output. Asserted, so it cannot be reached for a join whose pairs matter.
    fn sort_ascending(&mut self) {
        debug_assert!(
            !self.any_null,
            "sorting a side whose partner carries real indices would break the pairing"
        );
        self.idx.sort_unstable();
    }

    /// The Arrow column. No null buffer is built unless a NULL was actually pushed.
    pub(crate) fn finish(self) -> UInt32Array {
        if !self.any_null {
            return UInt32Array::from(self.idx);
        }
        self.idx
            .into_iter()
            .map(|v| (v != NULL_INDEX).then_some(v))
            .collect()
    }
}

impl JoinIndices {
    pub(crate) fn from_bufs(left: IndexBuf, right: IndexBuf) -> Self {
        Self {
            left: left.finish(),
            right: right.finish(),
        }
    }
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
    // Canonicalize float keys before ANY key handling below.
    //
    // `crate::keys` is documented as the one canonical form every hash path derives key
    // identity from "so they cannot disagree" — and the shuffle, the window, and all three
    // aggregate paths already do. The join did not, so it encoded its keys through
    // `RowConverter` raw: `-0.0` and `0.0` produce different row bytes and never match,
    // even though `=`, `GROUP BY`, and the shuffle all treat them as one value. A join on a
    // float key therefore silently DROPPED matching rows (`0.0 ⋈ -0.0` returned nothing),
    // and NaN keys — which the aggregate path folds to one canonical quiet NaN — likewise
    // failed to match themselves. Canonicalizing here, at the entry, puts the join on the
    // same key identity as every other operator, which is the invariant `keys.rs` exists to
    // hold. A key set with no float column is returned unchanged (`None`), so the integer
    // fast paths below are untouched.
    let l_canon = crate::keys::canonicalize_float_keys(left_keys);
    let r_canon = crate::keys::canonicalize_float_keys(right_keys);
    let left_keys: &[ArrayRef] = l_canon.as_deref().unwrap_or(left_keys);
    let right_keys: &[ArrayRef] = r_canon.as_deref().unwrap_or(right_keys);

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
        if radix_eligible(join_type) && right_rows > RADIX_MIN_BUILD_ROWS {
            return Ok(radix_join_scalar(
                |i| keys.right[i],
                |l| keys.left[l],
                right_rows,
                left_rows,
                &right_null,
                &left_null,
                join_type,
            ));
        }
        return Ok(build_probe_flat(
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
        if radix_eligible(join_type) && right_rows > RADIX_MIN_BUILD_ROWS {
            return Ok(radix_join_scalar(
                |i| (keys.right.0[i], keys.right.1[i]),
                |l| (keys.left.0[l], keys.left.1[l]),
                right_rows,
                left_rows,
                &right_null,
                &left_null,
                join_type,
            ));
        }
        return Ok(build_probe_flat(
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
    Ok(build_probe_flat(
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
    /// Heads sharded by hash (see [`build::shard_of`]) — one shard for a small build, so the
    /// common small table is exactly the flat one it always was. A key's shard is a function of
    /// its hash alone, so a chain never spans shards and the build parallelizes with no
    /// synchronization; `next` stays a single absolute-indexed chain either way.
    heads: Vec<HashTable<u32>>,
    next: Vec<u32>,
    state: ahash::RandomState,
    bloom: Option<BloomFilter>,
}

impl JoinTable {
    /// Build the chained hash table over the right (build) side. The optional probe
    /// bloom is populated in this same pass (no extra hashing) — see
    /// [`use_probe_bloom_with`].
    ///
    /// Past [`build::PARALLEL_BUILD_MIN_ROWS`] the heads are sharded and built across every
    /// core: the build loop was the join's sequential prefix and, on a large build, its
    /// dominant cost (10.7 ms serial against a 6.0 ms parallel probe, measured on TPC-H q5).
    /// The result is bit-identical either way — see [`build`].
    fn build<K: JoinKeys + Sync>(
        keys: &K,
        right_rows: usize,
        right_null: &[bool],
        use_bloom: bool,
        bloom_fp_rate: f64,
    ) -> Self {
        let state = ahash::RandomState::with_seeds(0x9E37, 0x79B9, 0x7F4A, 0x7C15);
        let bloom = use_bloom.then(|| BloomFilter::with_params(right_rows as u64, bloom_fp_rate));
        let shards = build::shard_count(right_rows);
        if shards > 1 {
            let (heads, next, bloom) =
                build::build_sharded(keys, &state, right_rows, right_null, shards, bloom);
            return Self {
                heads,
                next,
                state,
                bloom,
            };
        }

        let mut heads: HashTable<u32> = HashTable::with_capacity(right_rows);
        let mut next: Vec<u32> = vec![u32::MAX; right_rows];
        let mut bloom = bloom;
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
            heads: vec![heads],
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
        // The build put this key in exactly one shard, chosen from its hash — so the probe
        // finds it there without any coordination. One shard (the small-build case) reduces to
        // the flat lookup this always was.
        self.heads[build::shard_of(hash, self.heads.len())]
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
        left_out: &mut IndexBuf,
        right_out: &mut IndexBuf,
        mut right_matched: Option<&mut [bool]>,
    ) {
        let emit_left_unmatched = matches!(join_type, JoinType::Left | JoinType::Full);
        for i in range {
            let head = self.head_for(keys, i, left_null[i]);
            match join_type {
                JoinType::Semi => {
                    if head.is_some() {
                        left_out.push(i as u32);
                        right_out.push_null();
                    }
                }
                JoinType::Anti => {
                    if head.is_none() {
                        left_out.push(i as u32);
                        right_out.push_null();
                    }
                }
                _ => match head {
                    Some(mut r) => {
                        // Walk the chain of right rows sharing this key.
                        loop {
                            if let Some(rm) = right_matched.as_deref_mut() {
                                rm[r as usize] = true;
                            }
                            left_out.push(i as u32);
                            right_out.push(r);
                            let nxt = self.next[r as usize];
                            if nxt == u32::MAX {
                                break;
                            }
                            r = nxt;
                        }
                    }
                    None => {
                        if emit_left_unmatched {
                            left_out.push(i as u32);
                            right_out.push_null();
                        }
                    }
                },
            }
        }
    }
}

/// Whether an integer-keyed join of this build size should take the cache-radix path.
///
/// A build side past this many rows spills its hash table out of cache, so the flat
/// path's per-probe random lookup becomes a cache miss (measured ~4.5× throughput cliff
/// past ~64K build rows). [`radix_join_scalar`] partitions both sides into cache-sized
/// partitions and gathers each partition's keys contiguously, so the probe hits cache.
/// Only the integer fast paths (which can gather a `Copy` key) use it; the row-encoded
/// path stays flat. Left-driven join types only (Inner/Left/Semi/Anti) — Right/Full need
/// cross-partition unmatched bookkeeping the flat oracle already does.
const RADIX_MIN_BUILD_ROWS: usize = 1 << 16;

/// Build-row floor for the **parallel broadcast** radix path. Higher than the
/// single-threaded floor because a broadcast probe runs every core against one shared
/// build table, which stays resident in the ~tens-of-MB shared L3 well past the ~64K that
/// spills a single core's L2 — only once the build exceeds L3 does the per-probe miss
/// dominate the parallel-sliced probe. Below this, the sequential partitioning gather
/// would cost more than the cache it saves (measured: radix regressed small broadcasts).
/// ~2M `i64` rows ≈ a ~34 MB build table + key array, past a typical L3.
const RADIX_MIN_BUILD_ROWS_BROADCAST: usize = 1 << 21;

/// Whether `join_type` is left-driven (every emitted row is keyed by a left/probe row),
/// the shapes [`radix_join_scalar`] supports.
fn radix_eligible(join_type: JoinType) -> bool {
    matches!(
        join_type,
        JoinType::Inner | JoinType::Left | JoinType::Semi | JoinType::Anti
    )
}

/// Target build rows per radix partition — sized so a partition's hash table + chain
/// stays cache-resident, which is the whole point (a probe into it then hits cache).
const RADIX_PART_ROWS: usize = 1 << 15;

/// Cap on radix fan-out: enough partitions to make any realistic build cache-resident
/// without the partition vectors themselves thrashing cache on the scatter.
const RADIX_MAX_PARTS: usize = 1 << 12;

/// Total rows (build + probe) below which joining the radix partitions concurrently costs
/// more than it saves.
///
/// The per-partition join is already cache-resident by construction, so the only thing
/// parallelism buys is core count; below this the rayon fan-out plus the per-partition
/// `IndexBuf` allocations and the final concatenation dominate. Above it the partition loop
/// is pure independent work and scales with cores.
const RADIX_PARALLEL_MIN_ROWS: usize = 1 << 18;

/// Whether [`radix_join_scalar`] should join its partitions concurrently.
///
/// Needs real work (`RADIX_PARALLEL_MIN_ROWS`), more than one partition to spread, and more
/// than one core to spread them over. Nested inside an outer `par_iter` (the materializing
/// executor's partitioned join) this simply finds no idle workers and runs inline, so it
/// cannot oversubscribe.
fn radix_parallel_worthwhile(build_rows: usize, probe_rows: usize, parts: usize) -> bool {
    parts > 1
        && build_rows.saturating_add(probe_rows) >= RADIX_PARALLEL_MIN_ROWS
        && rayon::current_num_threads() > 1
}

/// Cache-radix hash join over a `Copy` key witness (the integer fast paths), left-driven
/// join types only.
///
/// Partitions BOTH sides by the high bits of the key hash into `parts` cache-sized
/// partitions, **gathering each partition's keys contiguously** (`(key, abs_row)` pairs)
/// so the per-partition build table, its chain, and the probe's key comparisons all touch
/// only that small, cache-resident partition — never a random access back into the
/// multi-megabyte source key array (the miss the flat path pays on every probe past
/// cache). Equal keys land in the same partition (same deterministic hash), so the union
/// over partitions is exactly the flat join — `radix_matches_flat` proves it. Null-key
/// rows never match; the unmatched (`Left`) / no-match (`Anti`) ones are emitted last.
///
/// `build_key`/`probe_key` read the source key for a row; they are called once per row in
/// ascending order (a sequential, streaming pass over the source array), then never again.
fn radix_join_scalar<O: Copy + std::hash::Hash + Eq + Send + Sync>(
    build_key: impl Fn(usize) -> O + Sync,
    probe_key: impl Fn(usize) -> O + Sync,
    build_rows: usize,
    probe_rows: usize,
    build_null: &[bool],
    probe_null: &[bool],
    join_type: JoinType,
) -> JoinIndices {
    let (state, build_parts, probe_parts) = radix_partition(
        build_key, probe_key, build_rows, probe_rows, build_null, probe_null,
    );

    let mut left_out = IndexBuf::with_capacity(probe_rows);
    let mut right_out = IndexBuf::with_capacity(probe_rows);

    if radix_parallel_worthwhile(build_rows, probe_rows, build_parts.len()) {
        // Partitions are independent (equal keys co-partition), so each can be joined on its
        // own core. Concatenating the pieces **in partition order** reproduces the sequential
        // loop's appends exactly — same rows, same order — so this is a scheduling change
        // only, and the semi/anti row-order contract below is untouched.
        let pieces: Vec<(IndexBuf, IndexBuf)> = build_parts
            .par_iter()
            .zip(probe_parts.par_iter())
            .map(|(b, probe)| {
                let mut heads: HashTable<u32> = HashTable::with_capacity(b.len());
                let mut l = IndexBuf::with_capacity(probe.len());
                let mut r = IndexBuf::with_capacity(probe.len());
                join_partition_into(b, probe, &state, join_type, &mut heads, &mut l, &mut r);
                (l, r)
            })
            .collect();
        for (l, r) in pieces {
            left_out.extend(l);
            right_out.extend(r);
        }
    } else {
        // One table, reused per partition (each partition's build rows are disjoint).
        let mut heads: HashTable<u32> = HashTable::with_capacity(RADIX_PART_ROWS);
        for (b, probe) in build_parts.iter().zip(&probe_parts) {
            join_partition_into(
                b,
                probe,
                &state,
                join_type,
                &mut heads,
                &mut left_out,
                &mut right_out,
            );
        }
    }
    emit_null_probe_unmatched(probe_null, join_type, &mut left_out, &mut right_out);
    restore_probe_order(join_type, &mut left_out);

    JoinIndices::from_bufs(left_out, right_out)
}

/// Parallel cache-radix join for the broadcast path: one [`JoinIndices`] per partition,
/// joined concurrently across cores. Each partition is independent (equal keys
/// co-partition), so the union over the returned pieces is exactly the flat relation —
/// the broadcast caller gathers and concatenates them just as it does per-chunk pieces.
/// Left-driven only (the broadcast contract); `radix_parallel_matches_flat` proves parity.
fn radix_join_scalar_parallel<O: Copy + std::hash::Hash + Eq + Send + Sync>(
    build_key: impl Fn(usize) -> O + Sync,
    probe_key: impl Fn(usize) -> O + Sync,
    build_rows: usize,
    probe_rows: usize,
    build_null: &[bool],
    probe_null: &[bool],
    join_type: JoinType,
) -> Vec<JoinIndices> {
    let (state, build_parts, probe_parts) = radix_partition(
        build_key, probe_key, build_rows, probe_rows, build_null, probe_null,
    );

    let mut out: Vec<JoinIndices> = build_parts
        .par_iter()
        .zip(probe_parts.par_iter())
        .map(|(b, probe)| {
            let mut heads: HashTable<u32> = HashTable::with_capacity(b.len());
            let mut left_out = IndexBuf::with_capacity(probe.len());
            let mut right_out = IndexBuf::with_capacity(probe.len());
            join_partition_into(
                b,
                probe,
                &state,
                join_type,
                &mut heads,
                &mut left_out,
                &mut right_out,
            );
            JoinIndices::from_bufs(left_out, right_out)
        })
        .collect();
    // Null-key probe rows (Left/Anti keep them as unmatched) — one extra piece.
    let mut left_out = IndexBuf::default();
    let mut right_out = IndexBuf::default();
    emit_null_probe_unmatched(probe_null, join_type, &mut left_out, &mut right_out);
    if left_out.len() > 0 {
        out.push(JoinIndices::from_bufs(left_out, right_out));
    }
    out
}

/// Number of cache-sized partitions for a build of `build_rows`, and the high-bit shift
/// that maps a 64-bit hash to a partition.
fn radix_parts(build_rows: usize) -> (usize, u32) {
    let parts = (build_rows / RADIX_PART_ROWS)
        .next_power_of_two()
        .clamp(2, RADIX_MAX_PARTS);
    (parts, 64 - parts.trailing_zeros())
}

/// Gather both sides' non-null rows into cache-sized partitions, each carrying the key
/// inline as `(key, abs_row)` so no later step touches the source key arrays. Shared by
/// the sequential and parallel radix joins (one source of the partitioning, hence of the
/// co-partitioning invariant). Returns the hash state (so the join reproduces the same
/// hashes) and the per-partition build/probe vectors.
#[allow(clippy::type_complexity)]
fn radix_partition<O: Copy + std::hash::Hash + Eq + Send + Sync>(
    build_key: impl Fn(usize) -> O + Sync,
    probe_key: impl Fn(usize) -> O + Sync,
    build_rows: usize,
    probe_rows: usize,
    build_null: &[bool],
    probe_null: &[bool],
) -> (ahash::RandomState, Vec<Vec<(O, u32)>>, Vec<Vec<(O, u32)>>) {
    let (parts, shift) = radix_parts(build_rows);
    let state = ahash::RandomState::with_seeds(0x9E37, 0x79B9, 0x7F4A, 0x7C15);
    let part_of = |k: &O| (state.hash_one(k) >> shift) as usize;
    // Both scatters run histogram → prefix-sum → parallel write (`join::radix`), which
    // reproduces the serial `push` loop's per-partition row order exactly while spreading
    // the pass — the join's dominant sequential prefix — across every worker.
    let build_parts = radix::partition_side(&build_key, build_null, parts, part_of);
    let probe_parts = radix::partition_side(&probe_key, probe_null, parts, part_of);
    let _ = probe_rows; // row counts arrive via the null masks; kept for a symmetric signature
    (state, build_parts, probe_parts)
}

/// Join one radix partition: build a small (cache-resident) chained table over `b`'s keys
/// with a partition-local `next` chain, probe with `probe`, and append absolute-index
/// pairs. `heads` is cleared and reused. The one place the radix probe loop lives, so the
/// sequential and parallel drivers cannot diverge.
fn join_partition_into<O: Copy + std::hash::Hash + Eq>(
    b: &[(O, u32)],
    probe: &[(O, u32)],
    state: &ahash::RandomState,
    join_type: JoinType,
    heads: &mut HashTable<u32>,
    left_out: &mut IndexBuf,
    right_out: &mut IndexBuf,
) {
    let hash = |k: &O| state.hash_one(k);
    let emit_left_unmatched = matches!(join_type, JoinType::Left);
    let is_semi = matches!(join_type, JoinType::Semi);
    let is_anti = matches!(join_type, JoinType::Anti);
    let mut next_local: Vec<u32> = vec![u32::MAX; b.len()];
    heads.clear();
    for (j, &(k, _)) in b.iter().enumerate() {
        match heads.entry(
            hash(&k),
            |&s| b[s as usize].0 == k,
            |&s| hash(&b[s as usize].0),
        ) {
            Entry::Occupied(mut e) => {
                next_local[j] = *e.get();
                *e.get_mut() = j as u32;
            }
            Entry::Vacant(e) => {
                e.insert(j as u32);
            }
        }
    }
    for &(k, labs) in probe {
        let head = heads.find(hash(&k), |&s| b[s as usize].0 == k).copied();
        match head {
            Some(_) if is_semi => {
                left_out.push(labs);
                right_out.push_null();
            }
            Some(mut s) => {
                if !is_anti {
                    loop {
                        left_out.push(labs);
                        right_out.push(b[s as usize].1);
                        let nxt = next_local[s as usize];
                        if nxt == u32::MAX {
                            break;
                        }
                        s = nxt;
                    }
                }
            }
            None => {
                if emit_left_unmatched || is_anti {
                    left_out.push(labs);
                    right_out.push_null();
                }
            }
        }
    }
}

/// Put a semi/anti join's output back in **probe-row order**.
///
/// A semi/anti join emits a *subset of the probe side* and no build column, so its row order is
/// the only information in the result — and the flat path emits it in probe-row order, because it
/// scans the probe in order. The radix path instead emits partition by partition (and
/// `emit_null_probe_unmatched` appends the null-key rows last), so without this the *same query*
/// answers in a different order once the build crosses `RADIX_MIN_BUILD_ROWS` — `SELECT … WHERE
/// EXISTS (…) LIMIT 10` would return different rows for a bigger build side, which is a data-size
/// dependency no user can see coming.
///
/// Cheap by construction: the output holds at most one index per probe row, and only the semi/anti
/// shapes reach it (an inner/outer join's pairs must keep their emitted order).
fn restore_probe_order(join_type: JoinType, left_out: &mut IndexBuf) {
    if matches!(join_type, JoinType::Semi | JoinType::Anti) {
        left_out.sort_ascending();
    }
}

/// Emit the null-key probe rows a `Left`/`Anti` join keeps as unmatched (they are excluded
/// from the partitions since a null key matches nothing).
fn emit_null_probe_unmatched(
    probe_null: &[bool],
    join_type: JoinType,
    left_out: &mut IndexBuf,
    right_out: &mut IndexBuf,
) {
    if matches!(join_type, JoinType::Left | JoinType::Anti) {
        for (l, &is_null) in probe_null.iter().enumerate() {
            if is_null {
                left_out.push(l as u32);
                right_out.push_null();
            }
        }
    }
}

/// The single-table hash join (no radix): build one chain table over the whole right
/// side and probe the whole left. The correctness oracle and the small-build fast path.
#[allow(clippy::too_many_arguments)]
fn build_probe_flat<K: JoinKeys + Sync>(
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
    let mut left_out = IndexBuf::with_capacity(left_rows);
    let mut right_out = IndexBuf::with_capacity(left_rows);
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
                left_out.push_null();
                right_out.push(r as u32);
            }
        }
    }

    JoinIndices::from_bufs(left_out, right_out)
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
    // Canonicalize float keys before ANY key handling below.
    //
    // `crate::keys` is documented as the one canonical form every hash path derives key
    // identity from "so they cannot disagree" — and the shuffle, the window, and all three
    // aggregate paths already do. The join did not, so it encoded its keys through
    // `RowConverter` raw: `-0.0` and `0.0` produce different row bytes and never match,
    // even though `=`, `GROUP BY`, and the shuffle all treat them as one value. A join on a
    // float key therefore silently DROPPED matching rows (`0.0 ⋈ -0.0` returned nothing),
    // and NaN keys — which the aggregate path folds to one canonical quiet NaN — likewise
    // failed to match themselves. Canonicalizing here, at the entry, puts the join on the
    // same key identity as every other operator, which is the invariant `keys.rs` exists to
    // hold. A key set with no float column is returned unchanged (`None`), so the integer
    // fast paths below are untouched.
    let l_canon = crate::keys::canonicalize_float_keys(left_keys);
    let r_canon = crate::keys::canonicalize_float_keys(right_keys);
    let left_keys: &[ArrayRef] = l_canon.as_deref().unwrap_or(left_keys);
    let right_keys: &[ArrayRef] = r_canon.as_deref().unwrap_or(right_keys);

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
                    let mut left_out = IndexBuf::with_capacity(r.len());
                    let mut right_out = IndexBuf::with_capacity(r.len());
                    table.probe_range(
                        &$keys,
                        r.clone(),
                        &left_null,
                        join_type,
                        &mut left_out,
                        &mut right_out,
                        None,
                    );
                    JoinIndices::from_bufs(left_out, right_out)
                })
                .collect()
        }};
    }

    if let Some(keys) = I64Keys::try_new(left_keys, right_keys) {
        // A build past cache takes the parallel cache-radix path (each partition's table is
        // cache-resident); a small build stays on the single-table parallel-sliced probe
        // (the table already fits cache, so radix's gather is pure overhead).
        if right_rows > RADIX_MIN_BUILD_ROWS_BROADCAST {
            return Ok(radix_join_scalar_parallel(
                |i| keys.right[i],
                |l| keys.left[l],
                right_rows,
                left_rows,
                &right_null,
                &left_null,
                join_type,
            ));
        }
        return Ok(run!(keys));
    }
    if let Some(keys) = I64x2Keys::try_new(left_keys, right_keys) {
        if right_rows > RADIX_MIN_BUILD_ROWS_BROADCAST {
            return Ok(radix_join_scalar_parallel(
                |i| (keys.right.0[i], keys.right.1[i]),
                |l| (keys.left.0[l], keys.left.1[l]),
                right_rows,
                left_rows,
                &right_null,
                &left_null,
                join_type,
            ));
        }
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

    /// An inner join emits no NULL index, so `finish` must build no null buffer — that is
    /// where the halved traffic comes from. And a buffer that *did* see a NULL must
    /// reproduce it exactly, sentinel and all.
    #[test]
    fn index_buf_encodes_nulls_without_paying_for_them() {
        let mut inner = IndexBuf::with_capacity(3);
        inner.push(0);
        inner.push(7);
        inner.push(NULL_INDEX - 1); // the largest index that is not the sentinel
        let arr = inner.finish();
        assert_eq!(arr.null_count(), 0);
        assert!(
            arr.nulls().is_none(),
            "an inner join must build no null buffer"
        );
        assert_eq!(arr.values(), &[0, 7, NULL_INDEX - 1]);

        let mut outer = IndexBuf::default();
        outer.push(5);
        outer.push_null();
        outer.push(6);
        let arr = outer.finish();
        assert_eq!(arr.null_count(), 1);
        assert!(arr.is_null(1));
        assert_eq!((arr.value(0), arr.value(2)), (5, 6));
    }

    /// An empty buffer is an empty column, not a panic.
    #[test]
    fn index_buf_finishes_empty() {
        assert_eq!(IndexBuf::default().finish().len(), 0);
        let mut only_null = IndexBuf::default();
        only_null.push_null();
        let arr = only_null.finish();
        assert_eq!(arr.len(), 1);
        assert!(arr.is_null(0));
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

    // The cache-radix path MUST produce the identical relation as the flat oracle for
    // every left-driven join type, including duplicate keys, misses, and null keys.
    #[test]
    fn radix_matches_flat() {
        fn arr(v: &[i64]) -> Vec<ArrayRef> {
            vec![Arc::new(Int64Array::from(v.to_vec())) as ArrayRef]
        }
        fn nulls(v: &[i64], is_null: &[usize]) -> Vec<ArrayRef> {
            let opts: Vec<Option<i64>> = v
                .iter()
                .enumerate()
                .map(|(i, &x)| if is_null.contains(&i) { None } else { Some(x) })
                .collect();
            vec![Arc::new(Int64Array::from(opts)) as ArrayRef]
        }
        // Build with duplicate keys (3, 3), probe with matches/misses/dupes.
        let cases: Vec<(Vec<ArrayRef>, Vec<ArrayRef>)> = vec![
            (arr(&[1, 5, 2, 8, 3, 9]), arr(&[3, 1, 3, 7, 5, 5, 2, 4])),
            (nulls(&[1, 5, 2, 3, 3], &[2]), nulls(&[3, 1, 3, 7, 5], &[3])),
            (arr(&[]), arr(&[1, 2, 3])),
            (arr(&[1, 2, 3]), arr(&[])),
        ];
        for (li, ri) in &cases {
            let lrows = li[0].len();
            let rrows = ri[0].len();
            let lnull = null_mask(li, lrows);
            let rnull = null_mask(ri, rrows);
            let keys = I64Keys::try_new(li, ri).unwrap();
            for jt in [
                JoinType::Inner,
                JoinType::Left,
                JoinType::Semi,
                JoinType::Anti,
            ] {
                let flat = build_probe_flat(
                    &keys,
                    lrows,
                    rrows,
                    &lnull,
                    &rnull,
                    jt,
                    false,
                    BLOOM_FP_RATE,
                );
                // Force radix even on tiny inputs (parts >= 2 via clamp).
                let radix = radix_join_scalar(
                    |i| keys.right[i],
                    |l| keys.left[l],
                    rrows,
                    lrows,
                    &rnull,
                    &lnull,
                    jt,
                );
                let mut a = pairs(&flat);
                let mut b = pairs(&radix);
                a.sort();
                b.sort();
                assert_eq!(a, b, "radix != flat for {jt:?}");

                // The parallel broadcast radix (union of per-partition pieces) must match too.
                let pieces = radix_join_scalar_parallel(
                    |i| keys.right[i],
                    |l| keys.left[l],
                    rrows,
                    lrows,
                    &rnull,
                    &lnull,
                    jt,
                );
                let mut c: Vec<_> = pieces.iter().flat_map(pairs).collect();
                c.sort();
                assert_eq!(a, c, "parallel radix != flat for {jt:?}");
            }
        }
    }

    // Run with: cargo test -p bc-runtime --release join_timing -- --ignored --nocapture
    #[test]
    #[ignore]
    fn join_timing() {
        use std::time::Instant;
        let probe_n: i64 = 1_200_000;
        for build_n in [4_000i64, 40_000, 288_000, 2_000_000] {
            let build: Vec<ArrayRef> =
                vec![Arc::new(Int64Array::from((0..build_n).collect::<Vec<_>>()))];
            let probe_vals: Vec<i64> = (0..probe_n).map(|i| i % build_n).collect();
            let probe: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(probe_vals))];
            let _ = hash_join_indices(&probe, &build, JoinType::Inner).unwrap();
            let mut best = f64::MAX;
            let mut out_rows = 0;
            for _ in 0..5 {
                let t = Instant::now();
                let idx = hash_join_indices(&probe, &build, JoinType::Inner).unwrap();
                best = best.min(t.elapsed().as_secs_f64() * 1000.0);
                out_rows = idx.left.len();
            }
            println!(
                "build={build_n:>9} probe={probe_n} out={out_rows:>9}: {best:6.2} ms  ({:>4.0} M probe/s)",
                probe_n as f64 / (best / 1000.0) / 1e6
            );
        }
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
                let fast = build_probe_flat(&i64keys, 5, 5, &ln, &rn, jt, bloom, BLOOM_FP_RATE);
                let slow = build_probe_flat(&rowkeys, 5, 5, &ln, &rn, jt, bloom, BLOOM_FP_RATE);
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
                let fast = build_probe_flat(&fastkeys, 5, 5, &ln, &rn, jt, bloom, BLOOM_FP_RATE);
                let slow = build_probe_flat(&rowkeys, 5, 5, &ln, &rn, jt, bloom, BLOOM_FP_RATE);
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

#[cfg(test)]
mod hunt_tests {
    use super::*;
    use std::sync::Arc;

    use arrow::array::{Array, Int64Array};

    fn i64_col(v: &[Option<i64>]) -> ArrayRef {
        Arc::new(Int64Array::from(v.to_vec()))
    }

    /// Reconstruct the join output of a single-i64 join as a sorted multiset of *value*
    /// pairs, so two strategies that pick different rows in a duplicate group still compare
    /// equal iff they emit the same logical relation.
    fn vpairs1(
        idx: &JoinIndices,
        left: &[Option<i64>],
        right: &[Option<i64>],
    ) -> Vec<(Option<i64>, Option<i64>)> {
        let mut out: Vec<_> = (0..idx.left.len())
            .map(|k| {
                let l = idx
                    .left
                    .is_valid(k)
                    .then(|| left[idx.left.value(k) as usize])
                    .flatten();
                let r = idx
                    .right
                    .is_valid(k)
                    .then(|| right[idx.right.value(k) as usize])
                    .flatten();
                (l, r)
            })
            .collect();
        out.sort();
        out
    }

    fn vpairs2(
        idx: &JoinIndices,
        la: &[Option<i64>],
        lb: &[Option<i64>],
        ra: &[Option<i64>],
        rb: &[Option<i64>],
    ) -> Vec<((Option<i64>, Option<i64>), (Option<i64>, Option<i64>))> {
        let mut out: Vec<_> = (0..idx.left.len())
            .map(|k| {
                let l = if idx.left.is_valid(k) {
                    let i = idx.left.value(k) as usize;
                    (la[i], lb[i])
                } else {
                    (None, None)
                };
                let r = if idx.right.is_valid(k) {
                    let i = idx.right.value(k) as usize;
                    (ra[i], rb[i])
                } else {
                    (None, None)
                };
                (l, r)
            })
            .collect();
        out.sort();
        out
    }

    /// The TWO-column i64 key at RADIX scale (build past `RADIX_MIN_BUILD_ROWS`) must agree
    /// with the independent sort-merge strategy for every left-driven join type — the tuple
    /// radix path is only ever exercised on tiny inputs elsewhere.
    #[test]
    fn two_col_radix_at_scale_matches_sort_merge() {
        let build_rows = RADIX_MIN_BUILD_ROWS + 5_000;
        let ra: Vec<Option<i64>> = (0..build_rows as i64)
            .map(|k| Some(if k % 40 == 0 { 3 } else { k % 5000 }))
            .collect();
        let rb: Vec<Option<i64>> = (0..build_rows as i64).map(|k| Some(k % 7)).collect();
        let la: Vec<Option<i64>> = (0..25_000i64)
            .map(|i| match i % 6 {
                0 => None,
                _ => Some((i * 3) % 5000),
            })
            .collect();
        let lb: Vec<Option<i64>> = (0..25_000i64).map(|i| Some(i % 7)).collect();
        let left = vec![i64_col(&la), i64_col(&lb)];
        let right = vec![i64_col(&ra), i64_col(&rb)];
        for jt in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Semi,
            JoinType::Anti,
        ] {
            let h = hash_join_indices(&left, &right, jt).unwrap(); // tuple radix path
            let s = sort_merge_join_indices(&left, &right, jt).unwrap(); // independent
            assert_eq!(
                vpairs2(&h, &la, &lb, &ra, &rb),
                vpairs2(&s, &la, &lb, &ra, &rb),
                "two-col radix != sort-merge for {jt:?}"
            );
        }
    }

    /// Build-side symmetry: Kyber may build either side. `hash_join(L, R, t)` must produce the
    /// same value relation as `hash_join(R, L, swap(t))` with the two index columns swapped,
    /// for every join type. An asymmetry here means the optimizer's build-side choice silently
    /// changes the answer.
    #[test]
    fn build_side_symmetry_all_join_types() {
        let lv: Vec<Option<i64>> = vec![
            Some(1),
            Some(2),
            Some(2),
            None,
            Some(3),
            Some(5),
            Some(2),
            None,
        ];
        let rv: Vec<Option<i64>> = vec![Some(2), Some(2), Some(3), Some(4), None, Some(1), Some(1)];
        let left = vec![i64_col(&lv)];
        let right = vec![i64_col(&rv)];
        let swap = |t: JoinType| match t {
            JoinType::Left => JoinType::Right,
            JoinType::Right => JoinType::Left,
            other => other,
        };
        // Semi/Anti are inherently one-sided (left-only output), so symmetry is defined only
        // for the two-sided flavors.
        for jt in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
        ] {
            let direct = hash_join_indices(&left, &right, jt).unwrap();
            let flipped = hash_join_indices(&right, &left, swap(jt)).unwrap();
            // `flipped` has (left=right-rows, right=left-rows); swap the reconstruction args.
            let a = vpairs1(&direct, &lv, &rv);
            let mut b: Vec<_> = (0..flipped.left.len())
                .map(|k| {
                    let r = flipped
                        .left
                        .is_valid(k)
                        .then(|| rv[flipped.left.value(k) as usize])
                        .flatten();
                    let l = flipped
                        .right
                        .is_valid(k)
                        .then(|| lv[flipped.right.value(k) as usize])
                        .flatten();
                    (l, r)
                })
                .collect();
            b.sort();
            assert_eq!(a, b, "build-side asymmetry for {jt:?}");
        }
    }
}
