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

use arrow::array::{Array, ArrayRef, OffsetSizeTrait, UInt32Array};
use arrow::buffer::NullBuffer;
use arrow::row::{RowConverter, Rows, SortField};
use bc_sketches::BloomFilter;
use hashbrown::hash_table::Entry;
use hashbrown::HashTable;
use rayon::prelude::*;

use crate::error::RuntimeError;

mod asof;
mod build;
mod dense;
mod key_filter;
mod radix;
mod range;
mod sort_merge;
mod stream;

pub use asof::{asof_join_indices, AsofDirection, AsofSpec};
pub use key_filter::KeyFilter;
pub use range::{range_join_indices, RangeOp};
pub use sort_merge::sort_merge_join_indices;
pub use stream::{streaming_shape_supported, streaming_supported, BroadcastProbe};

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
    pub(crate) fn push(&mut self, row: u32) {
        debug_assert_ne!(row, NULL_INDEX, "row index collides with the NULL sentinel");
        self.idx.push(row);
    }

    /// Append a NULL (an unmatched outer row, or the unused side of a semi/anti join).
    #[inline]
    pub(crate) fn push_null(&mut self) {
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
    pub(crate) fn extend(&mut self, other: IndexBuf) {
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
    hash_join_indices_impl(
        left_keys,
        right_keys,
        join_type,
        use_bloom,
        bloom_fp_rate,
        bloom_min_build_rows,
    )
}

/// The hash-join index builder, with the probe-side bloom pre-filter made explicit.
///
/// `use_bloom` is decided by [`use_probe_bloom_with`] on the public path; tests drive it
/// both ways to prove the bloom is a pure performance short-circuit (the produced
/// [`JoinIndices`] relation is identical with the filter on or off).
///
/// `bloom_min_build_rows` is carried in *addition* to `use_bloom` for one reason: the
/// semi/anti swap below exchanges which side is built, so the caller's `use_bloom` — decided
/// for the other orientation — cannot answer whether the *new*, smaller build side deserves a
/// bloom. It is re-decided there from this threshold, which is why the threshold and not just
/// the verdict has to reach this far. Every other path uses `use_bloom` as given.
#[allow(clippy::too_many_arguments)]
/// Run `$body` with a [`BytesKeys`] bound to `$name`, at whichever offset width the key
/// carries; falls through when the key is not a single byte-array column.
///
/// A macro rather than a function because the probe loop is generic over [`JoinKeys`] and
/// **must monomorphize** — boxing a `dyn JoinKeys` would put a virtual call on the per-row
/// hash and comparison, which is the cost this path exists to remove. Every call site
/// returns from `$body`, so at most one arm ever runs.
macro_rules! with_bytes_keys {
    ($left:expr, $right:expr, |$name:ident| $body:block) => {
        if let Some($name) = BytesKeys::<i32>::try_new($left, $right) {
            $body
        }
        if let Some($name) = BytesKeys::<i64>::try_new($left, $right) {
            $body
        }
    };
}

pub(crate) fn hash_join_indices_impl(
    left_keys: &[ArrayRef],
    right_keys: &[ArrayRef],
    join_type: JoinType,
    use_bloom: bool,
    bloom_fp_rate: f64,
    bloom_min_build_rows: usize,
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
    // Decode dictionary keys *before* the float fold, in the order the two demand: a dictionary
    // of floats must be decoded before its floats can be canonicalized. The two sides of a join
    // are reached by different operator chains, so one can carry a dictionary while the other
    // carries decoded values, and `RowConverter` — built from one side's type and fed both —
    // then rejects the join outright. Same "one canonical form" argument as the fold below; see
    // `keys::decode_dict_keys`. `None` when no key is a dictionary, so nothing is allocated.
    let l_dec = crate::keys::decode_dict_keys(left_keys);
    let r_dec = crate::keys::decode_dict_keys(right_keys);
    let left_keys: &[ArrayRef] = l_dec.as_deref().unwrap_or(left_keys);
    let right_keys: &[ArrayRef] = r_dec.as_deref().unwrap_or(right_keys);

    let l_canon = crate::keys::canonicalize_float_keys(left_keys);
    let r_canon = crate::keys::canonicalize_float_keys(right_keys);
    let left_keys: &[ArrayRef] = l_canon.as_deref().unwrap_or(left_keys);
    let right_keys: &[ArrayRef] = r_canon.as_deref().unwrap_or(right_keys);

    let left_rows = left_keys.first().map_or(0, |a| a.len());
    let right_rows = right_keys.first().map_or(0, |a| a.len());
    let left_null = null_mask(left_keys, left_rows);
    let right_null = null_mask(right_keys, right_rows);

    // A semi/anti join returns left rows and uses the right only as a membership test, so
    // when the right is much the larger side the build-right convention builds a hash table
    // over the relation it is about to throw away. Build over the left instead and mark it.
    // Checked before the radix dispatch below, because the point is to not build that table
    // at all — partitioning it into cache-sized pieces first would be optimizing the work
    // this removes. Same relation, same row order; see `semi_anti_swapped`.
    if matches!(join_type, JoinType::Semi | JoinType::Anti)
        && swap_semi_build(left_rows, right_rows)
    {
        return semi_anti_swapped(
            left_keys,
            right_keys,
            left_rows,
            right_rows,
            &left_null,
            &right_null,
            join_type,
            bloom_fp_rate,
            bloom_min_build_rows,
        );
    }

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

    // Three-or-more-`Int64`-key fast path: the same raw-value treatment as the one- and
    // two-column cases above, for the composite surrogate key a star-schema fact-to-fact
    // join carries. See [`I64xNKeys`]. No radix arm: its scalar key must be `Copy`, and an
    // N-column key has no such value.
    if let Some(keys) = I64xNKeys::try_new(left_keys, right_keys) {
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

    // Single string/binary key fast path: hash and compare the raw bytes instead of
    // encoding every row of both sides through the `RowConverter`. See [`BytesKeys`].
    with_bytes_keys!(left_keys, right_keys, |keys| {
        // Parallel cache-radix arm, for a byte key short enough to pack into one `u128`.
        //
        // The flat path below is a *serial* build and probe: on a 96-core box a 4 M-row
        // string-keyed join measured 2.48x parallelism against the integer key's 28.6x on the
        // identical shape, and 1,894 ms against 37 ms. The join was not slow because bytes are
        // expensive to compare -- it was slow because it was the one key type that never
        // reached [`radix_join_scalar`], whose key must be `Copy`.
        //
        // A short byte key *is* such a value once packed ([`pack_byte_key`]), and the packing
        // is injective, so the partitions, the chains and the matches are the same ones the
        // byte comparison produces. No new join algorithm: the same proven radix join the
        // integer paths above use, reached with a different key witness.
        //
        // Fifteen bytes covers what join keys actually are -- ids, codes, SKUs, ISO dates,
        // categoricals. Anything longer keeps the flat path.
        if radix_eligible(join_type)
            && right_rows > RADIX_MIN_BUILD_ROWS
            && byte_keys_packable(&keys.right, right_rows)
            && byte_keys_packable(&keys.left, left_rows)
        {
            return Ok(radix_join_scalar(
                |i| pack_byte_key(keys.right.get(i)),
                |l| pack_byte_key(keys.left.get(l)),
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
    });

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

    /// The raw `(build, probe)` key slices, when the key is exactly one `Int64` column.
    ///
    /// `None` for every other shape, which is what confines the dense direct-map path
    /// ([`dense::DenseHeads`]) to the one key type whose values can index an array. A
    /// multi-column or row-encoded key has no such value, so it keeps the hash table.
    fn dense_keys(&self) -> Option<(&[i64], &[i64])> {
        None
    }
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
    fn dense_keys(&self) -> Option<(&[i64], &[i64])> {
        Some((self.right, self.left))
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

/// N-`Int64`-key fast path (three columns or more): the raw value slices of every key
/// column, per side.
///
/// [`I64Keys`] and [`I64x2Keys`] cover one and two columns; a *third* fell all the way to
/// [`RowConverter`], and that cliff is where the star-schema fact-to-fact join lives. TPC-DS
/// joins `store_sales` to `store_returns` on
/// `(ticket_number, item_sk, customer_sk)` — three surrogate keys — and encoding the 2.75 M-row
/// probe side into arrow's escaped row format cost **127 ms against 9.5 ms for the same join
/// on two of the three keys**, a 13x cliff for one extra column. Nine TPC-DS queries carry a
/// three-or-more-column equi-join.
///
/// Hashing walks the columns per row rather than a packed value, because an `i64` triple does
/// not fit a register and packing would need value ranges nothing here measures. That still
/// leaves the per-row work as N loads and N hasher writes, against the row encoder's fresh
/// allocation plus a byte-slice compare on every chain walk. The same [`build_probe`] loop
/// drives it, so it is bit-identical to the row-encoded oracle — only the key accessor differs.
struct I64xNKeys<'a> {
    right: Vec<&'a [i64]>,
    left: Vec<&'a [i64]>,
}

impl<'a> I64xNKeys<'a> {
    /// `Some` when both sides are the same number (three or more) of `Int64` columns.
    fn try_new(left_keys: &'a [ArrayRef], right_keys: &'a [ArrayRef]) -> Option<Self> {
        use arrow::array::Int64Array;
        use arrow::datatypes::DataType;
        if left_keys.len() < 3 || left_keys.len() != right_keys.len() {
            return None;
        }
        if left_keys
            .iter()
            .chain(right_keys)
            .any(|k| k.data_type() != &DataType::Int64)
        {
            return None;
        }
        let cols = |keys: &'a [ArrayRef]| -> Option<Vec<&'a [i64]>> {
            keys.iter()
                .map(|a| {
                    a.as_any()
                        .downcast_ref::<Int64Array>()
                        .map(|c| c.values().as_ref())
                })
                .collect()
        };
        Some(Self {
            right: cols(right_keys)?,
            left: cols(left_keys)?,
        })
    }

    #[inline]
    fn hash_at(state: &ahash::RandomState, cols: &[&[i64]], i: usize) -> u64 {
        use std::hash::{BuildHasher, Hasher};
        let mut hasher = state.build_hasher();
        for col in cols {
            hasher.write_i64(col[i]);
        }
        hasher.finish()
    }
}

impl JoinKeys for I64xNKeys<'_> {
    fn hash_right(&self, state: &ahash::RandomState, i: usize) -> u64 {
        Self::hash_at(state, &self.right, i)
    }
    fn hash_left(&self, state: &ahash::RandomState, i: usize) -> u64 {
        Self::hash_at(state, &self.left, i)
    }
    fn right_eq_right(&self, a: usize, b: usize) -> bool {
        self.right.iter().all(|c| c[a] == c[b])
    }
    fn right_eq_left(&self, r: usize, l: usize) -> bool {
        self.right
            .iter()
            .zip(&self.left)
            .all(|(rc, lc)| rc[r] == lc[l])
    }
}

/// One side's byte-array key column, addressed as `data[offsets[i]..offsets[i + 1]]`.
///
/// `value_offsets` is already adjusted for a sliced array and indexes `value_data`
/// absolutely, so a morsel that is a slice of a larger batch reads only its own values.
struct ByteCol<'a, O: OffsetSizeTrait> {
    data: &'a [u8],
    offsets: &'a [O],
}

impl<O: OffsetSizeTrait> ByteCol<'_, O> {
    #[inline]
    fn get(&self, i: usize) -> &[u8] {
        &self.data[self.offsets[i].as_usize()..self.offsets[i + 1].as_usize()]
    }
}

/// Raw byte-slice keys (the fast path): a single string/binary key column per side.
///
/// A string join key is the one common shape that had no fast path, so it fell all the way
/// to [`RowConverter`], which encodes **every probe row** into arrow's escaped row format —
/// a 32-byte block plus a continuation token per value, written into a fresh buffer, in one
/// pass before the join starts. On a fact-to-dimension join whose fact side is the probe
/// (H2O `join` q4 joins 10 M rows to 10 k on a string key), that pass encodes the entire
/// large side to gain nothing the raw bytes do not already give: equal non-null values are
/// equal byte slices, so hashing and comparing the slices is exactly the encoded comparison
/// without the encode. Rows with a null key never match and are excluded by `null_mask`
/// before any key is read, so the offsets under a null slot are never dereferenced.
///
/// The same argument `assign_groups_bytes` makes for `GROUP BY <string>`, where it measured
/// ~25% on the low-cardinality keys and removed the encode outright on the high-cardinality
/// ones. Both sides must carry the same byte type, which is what the planner's key coercion
/// already guarantees.
struct BytesKeys<'a, O: OffsetSizeTrait> {
    right: ByteCol<'a, O>,
    left: ByteCol<'a, O>,
}

impl<'a, O: OffsetSizeTrait> BytesKeys<'a, O> {
    /// `Some` when both sides are exactly one byte-array column of the same type, whose
    /// offset width is `O`.
    fn try_new(left_keys: &'a [ArrayRef], right_keys: &'a [ArrayRef]) -> Option<Self> {
        use arrow::array::{GenericBinaryArray, GenericStringArray};
        use arrow::datatypes::DataType;
        if left_keys.len() != 1 || right_keys.len() != 1 {
            return None;
        }
        let dt = left_keys[0].data_type();
        if dt != right_keys[0].data_type() {
            return None;
        }
        // The offset width the caller instantiated must be the one this type carries;
        // otherwise the downcasts below fail and the other instantiation answers.
        match dt {
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => {}
            _ => return None,
        }
        let col = |a: &'a ArrayRef| -> Option<ByteCol<'a, O>> {
            if let Some(s) = a.as_any().downcast_ref::<GenericStringArray<O>>() {
                return Some(ByteCol {
                    data: s.value_data(),
                    offsets: s.value_offsets(),
                });
            }
            let b = a.as_any().downcast_ref::<GenericBinaryArray<O>>()?;
            Some(ByteCol {
                data: b.value_data(),
                offsets: b.value_offsets(),
            })
        };
        Some(Self {
            right: col(&right_keys[0])?,
            left: col(&left_keys[0])?,
        })
    }
}

impl<O: OffsetSizeTrait> JoinKeys for BytesKeys<'_, O> {
    fn hash_right(&self, state: &ahash::RandomState, i: usize) -> u64 {
        state.hash_one(self.right.get(i))
    }
    fn hash_left(&self, state: &ahash::RandomState, i: usize) -> u64 {
        state.hash_one(self.left.get(i))
    }
    fn right_eq_right(&self, a: usize, b: usize) -> bool {
        self.right.get(a) == self.right.get(b)
    }
    fn right_eq_left(&self, r: usize, l: usize) -> bool {
        self.right.get(r) == self.left.get(l)
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
    /// A perfect hash over a small-range single-`Int64` build key, replacing `heads`.
    ///
    /// When present, `heads` is empty and every lookup is one indexed load with no hashing
    /// on either side — see [`dense::DenseHeads`] for why a dimension table's surrogate key
    /// makes this the common case, and for the memory bound that lets it be enabled on the
    /// key's range alone. `next` and `unique` mean exactly what they mean for the hash path,
    /// so the probe loop below is shared.
    dense: Option<dense::DenseHeads>,
    next: Vec<u32>,
    /// Whether no build key repeats — so every chain has length exactly 1.
    ///
    /// True for every join to a primary/unique key, which is most of them (`o_orderkey`,
    /// `p_partkey`, every dimension table). The chain walk in [`Self::probe_range`] then loads
    /// `next[r]` once per *emitted row* purely to read the `u32::MAX` that ends the loop — a
    /// random access into a multi-megabyte array, one per output row, whose answer is known in
    /// advance. Knowing the build is unique lets the probe skip that load entirely, and lets
    /// [`build::build_sharded`] skip allocating `next` at all.
    ///
    /// Result-invariant: a length-1 chain emits the same single `(i, r)` pair either way.
    unique: bool,
    state: ahash::RandomState,
    bloom: Option<BloomFilter>,
    /// Runtime verdict on whether the probe-side bloom is *earning* its lookup.
    ///
    /// The bloom's value is its **rejection rate**, which no planner-side row count can
    /// predict: a foreign-key join where every probe row matches (TPC-H `lineitem ⋈ orders`)
    /// rejects nothing, so the per-row `contains_hash` is a pure random-access cache miss on
    /// top of the hash lookup it was meant to save. [`use_probe_bloom_with`] can only see
    /// build/probe *sizes*, and the streaming executor cannot even see the probe count
    /// (it passes `usize::MAX`), so on that path the bloom is switched on unconditionally.
    ///
    /// So measure it instead: [`Self::probe_range`] counts rejections over the first
    /// [`BLOOM_TRIAL_ROWS`] probe rows and latches the bloom off when the rate is below
    /// [`BLOOM_MIN_REJECT_RATE`]. This never changes a result — a bloom hit is not a match,
    /// only a "maybe", so skipping the filter just runs the authoritative hash lookup that
    /// follows it. Morsel-granular (decided per range, never per row), so the counters cost
    /// two relaxed atomics per morsel rather than two per row, and every tier — sequential,
    /// morsel-parallel, and distributed — adapts independently on what it actually sees.
    bloom_trial: BloomTrial,
}

/// Probe rows to observe before ruling on the bloom. One morsel-ish sample: long enough that
/// the rate is not noise, short enough that a useless bloom is abandoned almost immediately.
const BLOOM_TRIAL_ROWS: u64 = 1 << 16;

/// Rejection rate below which the probe-side bloom costs more than it saves. A rejected row
/// saves one hash-table probe; a passed row pays one bloom lookup for nothing. Both are random
/// accesses of the same order, so the filter needs to reject a real fraction to break even —
/// well under half. Set conservatively: only an obviously-useless bloom is switched off.
const BLOOM_MIN_REJECT_RATE: f64 = 0.10;

/// Rejection accounting for [`JoinTable::bloom_trial`]. Shared across the worker threads that
/// probe one broadcast table, so both counters are atomics; they are touched once per probed
/// range, not per row.
#[derive(Debug, Default)]
struct BloomTrial {
    seen: std::sync::atomic::AtomicU64,
    rejected: std::sync::atomic::AtomicU64,
}

impl BloomTrial {
    /// Whether the bloom is still worth consulting: during the trial always, afterwards only
    /// if it rejected enough of the sample to pay for itself.
    #[inline]
    fn worth_consulting(&self) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let seen = self.seen.load(Relaxed);
        if seen < BLOOM_TRIAL_ROWS {
            return true;
        }
        (self.rejected.load(Relaxed) as f64) >= (seen as f64) * BLOOM_MIN_REJECT_RATE
    }

    /// Fold one probed range's tally in. Called once per range.
    #[inline]
    fn observe(&self, seen: u64, rejected: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.seen.fetch_add(seen, Relaxed);
        self.rejected.fetch_add(rejected, Relaxed);
    }
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

        // A single small-range `Int64` build key is direct-mapped instead of hashed. This is
        // checked first because it subsumes the hash build entirely: no build-side hashing,
        // no probe-side hashing, and no bloom (the "is this key present" question the bloom
        // approximates is answered exactly, by the same indexed load that finds the head).
        if let Some((right, _)) = keys.dense_keys() {
            if let Some(d) = dense::DenseHeads::build(right, right_rows, right_null) {
                return Self {
                    heads: Vec::new(),
                    dense: Some(d.heads),
                    next: build::stitch_chain(d.links, right_rows, d.unique),
                    unique: d.unique,
                    state,
                    bloom: None,
                    bloom_trial: BloomTrial::default(),
                };
            }
        }

        let bloom = use_bloom.then(|| BloomFilter::with_params(right_rows as u64, bloom_fp_rate));
        let shards = build::shard_count(right_rows);
        if shards > 1 {
            let (heads, next, bloom, unique) =
                build::build_sharded(keys, &state, right_rows, right_null, shards, bloom);
            return Self {
                heads,
                dense: None,
                next,
                unique,
                state,
                bloom,
                bloom_trial: BloomTrial::default(),
            };
        }

        let mut heads: HashTable<u32> = HashTable::with_capacity(right_rows);
        let mut next: Vec<u32> = vec![u32::MAX; right_rows];
        let mut bloom = bloom;
        // Set on the first repeated key — the serial mirror of `build_sharded`'s chain check.
        let mut unique = true;
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
                    unique = false;
                }
                Entry::Vacant(e) => {
                    e.insert(i as u32);
                }
            }
        }
        Self {
            heads: vec![heads],
            dense: None,
            next,
            unique,
            state,
            bloom,
            bloom_trial: BloomTrial::default(),
        }
    }

    /// The chain head for probe (left) row `l` — `None` for a null key, a bloom miss,
    /// or no match; otherwise a real right-row index (`is_some()` ⇒ ≥1 match).
    /// The bloom is supplied by the caller rather than read from `self`, so a probe loop can
    /// hoist the "is this bloom worth consulting" decision out of the row loop and tally what
    /// it rejected. `rejected` counts the rows this bloom short-circuited.
    #[inline]
    fn head_for<K: JoinKeys>(
        &self,
        keys: &K,
        l: usize,
        is_null: bool,
        bloom: Option<&BloomFilter>,
        rejected: &mut u64,
    ) -> Option<u32> {
        if is_null {
            return None;
        }
        // The dense direct map answers without hashing either side. The `Option` test is
        // loop-invariant over a probe range, so this is a predicted branch, not a per-row cost.
        if let Some(d) = self.dense.as_ref() {
            // `dense` is only ever set from a key set that reported `dense_keys`, so the
            // probe side reports it too — both come from the same `JoinKeys`.
            let (_, left) = keys.dense_keys().expect("dense table implies i64 keys");
            return d.head(left[l]);
        }
        let hash = keys.hash_left(&self.state, l);
        // A bloom miss is definitive (no false negatives): the key is not on the build
        // side, so the chain is provably empty — skip the hash-table lookup.
        if bloom.is_some_and(|b| !b.contains_hash(hash)) {
            *rejected += 1;
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
        left_null: Option<&[bool]>,
        join_type: JoinType,
        left_out: &mut IndexBuf,
        right_out: &mut IndexBuf,
        mut right_matched: Option<&mut [bool]>,
    ) {
        let emit_left_unmatched = matches!(join_type, JoinType::Left | JoinType::Full);
        // Decide once per range whether to consult the bloom, then tally what it rejected so
        // the next range can re-decide. `None` here is exactly the "no bloom" path.
        let bloom = self
            .bloom
            .as_ref()
            .filter(|_| self.bloom_trial.worth_consulting());
        let mut rejected = 0u64;
        let seen = range.len() as u64;
        for i in range {
            // `None` ⇒ no key column had a null, so no row is null-keyed — the check is skipped
            // entirely and the caller never allocated the mask. The `Option` is loop-invariant,
            // so this is a predicted null-pointer test, not the 16 KB per-morsel mask a foreign-key
            // probe (its key never null) used to allocate and zero for nothing.
            let is_null = left_null.is_some_and(|m| m[i]);
            let head = self.head_for(keys, i, is_null, bloom, &mut rejected);
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
                    // Unique build key ⇒ the chain is exactly one row, so emit it and skip the
                    // `next[r]` load that would only confirm the end. Same `(i, r)` pair, same
                    // order, one fewer random multi-megabyte access per emitted row.
                    Some(r) if self.unique => {
                        if let Some(rm) = right_matched.as_deref_mut() {
                            rm[r as usize] = true;
                        }
                        left_out.push(i as u32);
                        right_out.push(r);
                    }
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
        // Only meaningful while the bloom was actually consulted; once it is latched off the
        // rate is frozen at whatever the trial measured.
        if bloom.is_some() {
            self.bloom_trial.observe(seen, rejected);
        }
    }

    /// Probe rows `range` against the table, recording *which build rows were hit* rather
    /// than emitting index pairs — the semi/anti join's inverted direction.
    ///
    /// [`Self::probe_range`] answers "does this probe row have a match", which is the question
    /// a semi join asks when the table is built on the side it does *not* return. This answers
    /// "was this build row matched by anything", which is the same question asked of a table
    /// built on the side it *does* return. See [`semi_anti_swapped`] for when that is the
    /// cheaper way round and why the emitted relation is unchanged.
    ///
    /// No output buffer grows here: a match sets one byte, so a probe side that fans out
    /// heavily costs the same as one that does not. That is the second win, and it is why
    /// this is not expressed as an inner join whose right indices are then deduplicated.
    ///
    /// `matched` is shared across the ranges [`mark_probe`] runs concurrently. Relaxed is the
    /// right ordering and not a shortcut: a slot only ever moves `false → true`, never back,
    /// so concurrent writers cannot disagree about the value and no writer reads one. The
    /// reader runs after `par_iter().for_each` has joined, and that join is the
    /// happens-before edge that publishes every write.
    fn mark_range<K: JoinKeys>(
        &self,
        keys: &K,
        range: std::ops::Range<usize>,
        probe_null: Option<&[bool]>,
        matched: &[std::sync::atomic::AtomicBool],
    ) {
        use std::sync::atomic::Ordering::Relaxed;
        let bloom = self
            .bloom
            .as_ref()
            .filter(|_| self.bloom_trial.worth_consulting());
        let mut rejected = 0u64;
        let seen = range.len() as u64;
        for i in range {
            let is_null = probe_null.is_some_and(|m| m[i]);
            let Some(mut r) = self.head_for(keys, i, is_null, bloom, &mut rejected) else {
                continue;
            };
            matched[r as usize].store(true, Relaxed);
            // A unique build key ⇒ the chain is exactly one row, so skip the `next[r]` load
            // whose answer (`u32::MAX`) is already known — the same reasoning, and the same
            // random multi-megabyte access avoided, as in `probe_range`.
            if self.unique {
                continue;
            }
            loop {
                let nxt = self.next[r as usize];
                if nxt == u32::MAX {
                    break;
                }
                r = nxt;
                matched[r as usize].store(true, Relaxed);
            }
        }
        if bloom.is_some() {
            self.bloom_trial.observe(seen, rejected);
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

/// The widest byte key that packs losslessly into a `u128`: fifteen value bytes under a
/// length byte. See [`pack_byte_key`].
const MAX_PACKED_KEY_BYTES: usize = 15;

/// Pack a byte key into one `u128`, **injectively**.
///
/// The length goes in the top byte and the value bytes in the fifteen below it, most
/// significant first. Two distinct keys therefore cannot share a word: different lengths
/// differ in the top byte, and equal lengths differ wherever their bytes do. That is the
/// whole correctness argument for the radix arm this feeds — the join over packed keys
/// matches exactly the rows the byte-slice comparison matches, because "packs equal" and
/// "bytes equal" are the same predicate.
///
/// It is not order-preserving and does not need to be: a hash join partitions and compares
/// for equality, never for order. (`ops::byte_sort` packs for *ordering* and pays attention
/// to exactly that difference.)
#[inline]
fn pack_byte_key(b: &[u8]) -> u128 {
    let mut w = (b.len() as u128) << 120;
    for (i, &c) in b.iter().enumerate() {
        w |= (c as u128) << (8 * (14 - i));
    }
    w
}

/// Whether every value in `col` is short enough for [`pack_byte_key`].
///
/// Read from the offsets alone — no value bytes are touched — and in parallel, because on
/// the ten-million-row side this decision sits in front of the join it is deciding about.
fn byte_keys_packable<O: OffsetSizeTrait>(col: &ByteCol<'_, O>, rows: usize) -> bool {
    (0..rows)
        .into_par_iter()
        .all(|i| col.offsets[i + 1].as_usize() - col.offsets[i].as_usize() <= MAX_PACKED_KEY_BYTES)
}

/// Build-row floor for the **parallel broadcast** radix path. Higher than the
/// single-threaded floor because a broadcast probe runs every core against one shared
/// build table, which stays resident in the ~tens-of-MB shared L3 well past the ~64K that
/// spills a single core's L2 — only once the build exceeds L3 does the per-probe miss
/// dominate the parallel-sliced probe. Below this, the sequential partitioning gather
/// would cost more than the cache it saves (measured: radix regressed small broadcasts).
/// ~2M `i64` rows ≈ a ~34 MB build table + key array, past a typical L3.
pub const RADIX_MIN_BUILD_ROWS_BROADCAST: usize = 1 << 21;

/// Whether `join_type` is left-driven (every emitted row is keyed by a left/probe row),
/// the shapes [`radix_join_scalar`] supports.
fn radix_eligible(join_type: JoinType) -> bool {
    matches!(
        join_type,
        JoinType::Inner | JoinType::Left | JoinType::Semi | JoinType::Anti
    )
}

/// Target build rows per radix partition when the key width is unknown — sized so a
/// partition's hash table + chain stays cache-resident, which is the whole point (a probe
/// into it then hits cache).
///
/// The historical fixed value, kept as the fallback and as the reference for
/// [`radix_part_rows`]: 32,768 rows of an `i64` key is ~800 KiB of partition-local state,
/// which is resident on a 1 MiB-L2 part and roughly 1.6x over on a 512 KiB one. That spread
/// is exactly why the live path computes the figure instead of assuming it.
const RADIX_PART_ROWS: usize = 1 << 15;

/// Partition-local bytes one build row costs, for a key of `key_bytes`.
///
/// Everything a probe into a partition touches, and nothing it does not — this is the
/// working set the L2 budget has to cover:
///
/// * the gathered `(key, abs_row)` pair, which is what the probe compares against;
/// * one `u32` of `next_local`, the partition-local collision chain;
/// * the `HashTable<u32>` slot: 4 bytes of index plus 1 control byte, held by hashbrown at a
///   7/8 load factor, so ~5.7 bytes of table per row present.
///
/// The probe-side partition vector is deliberately excluded: it is streamed once in order,
/// so it costs bandwidth rather than residency, and counting it would halve every partition
/// for no cache benefit.
fn radix_row_bytes(key_bytes: usize) -> usize {
    // The gathered pair is `(O, u32)`, laid out with the alignment padding a real struct has.
    let pair = key_bytes.next_multiple_of(4) + 4;
    let chain = 4;
    let table = 5 * 8 / 7 + 1; // 4-byte index + 1 control byte at a 7/8 load factor
    pair + chain + table
}

/// Build rows per radix partition that keep the partition's state resident in **this host's**
/// L2, for a key of `key_bytes`.
///
/// The fixed 32,768 this replaces was measured on one machine and is wrong on the next by the
/// ratio of the two L2s, which spans 4x across the parts the engine runs on (512 KiB on a
/// small ARM core, 1 MiB on Cascade Lake, 2 MiB on Zen 4). Too large and the per-partition
/// probe misses cache, which is the entire cost the radix path exists to remove; too small and
/// the fan-out grows for nothing, paying scatter and TLB pressure to over-partition.
///
/// Clamped to `[4096, RADIX_PART_ROWS]`: never so small that the per-partition table setup
/// dominates, and never larger than the historical value, so on a big-cache host this can only
/// keep the previous behavior rather than widen into an unmeasured regime.
fn radix_part_rows(key_bytes: usize) -> usize {
    bc_arrow::CpuTopology::detect()
        .l2_resident_rows(radix_row_bytes(key_bytes))
        .clamp(1 << 12, RADIX_PART_ROWS)
}

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
        // Sized to the largest partition the split can produce, so the reused table is
        // allocated once here rather than growing (and rehashing) inside the partition loop.
        let mut heads: HashTable<u32> =
            HashTable::with_capacity(build_parts.iter().map(|b| b.len()).max().unwrap_or(0));
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

/// Number of cache-sized partitions for a build of `build_rows` keyed by `O`, and the
/// high-bit shift that maps a 64-bit hash to a partition.
///
/// The target partition size comes from [`radix_part_rows`], which reads the host's real L2
/// rather than assuming one — a wide key (a 16-byte composite) partitions more finely than an
/// `i64` on the same machine, and the same key partitions more finely on a small-cache part
/// than on a large one. Both are the point: the partition has to fit the cache it will be
/// probed in.
fn radix_parts<O>(build_rows: usize) -> (usize, u32) {
    let parts = (build_rows / radix_part_rows(std::mem::size_of::<O>()))
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
    let (parts, shift) = radix_parts::<O>(build_rows);
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
/// Probe rows a semi/anti join's *returned* side must be under before the table is built on
/// it instead of on the other one.
///
/// Below this the whole question is noise — the build being optimized away is a few thousand
/// rows — and the swap's one cost (a `Vec<bool>` over the returned side, plus a pass over it
/// to emit) is not worth reasoning about either way. `1 << 16` is the same figure
/// [`RADIX_MIN_BUILD_ROWS`] uses to decide a build has outgrown cache, for the same reason:
/// it is where a hash table stops being free.
const SEMI_SWAP_MIN_PROBE_ROWS: usize = 1 << 16;

/// How many times larger the discarded side must be before the swap is taken.
///
/// Both orders do ~`|L| + |R|` hash operations, so the swap is not a complexity win — it is a
/// *build* win, and a build row costs several times a probe row (it allocates a table slot,
/// writes a chain link, and is later walked by random access). Requiring a 4x margin means
/// the swap is only taken where that difference dominates the arithmetic either way, so an
/// imprecise estimate of the per-row constants cannot make it a loss. TPC-H q4's
/// `orders SEMI lineitem` sits at 66x (3.79M against 57k), far inside it.
const SEMI_SWAP_MIN_RATIO: usize = 4;

/// Whether a semi/anti join should build its table on the left (returned) side.
///
/// `left_rows` is the side the join returns; `right_rows` is the side it only tests against.
fn swap_semi_build(left_rows: usize, right_rows: usize) -> bool {
    right_rows >= SEMI_SWAP_MIN_PROBE_ROWS
        && right_rows >= left_rows.saturating_mul(SEMI_SWAP_MIN_RATIO)
}

/// `Semi`/`Anti` with the hash table built on the **left** side and the right side scanned
/// to mark it — the opposite of this module's build-right convention, for the case where
/// that convention builds over the larger relation.
///
/// A semi join returns left rows and discards the right entirely, so the right exists only to
/// answer "does this key occur". Building on it is therefore backwards whenever it is the
/// bigger side, and it routinely is: TPC-H q4 is `orders SEMI lineitem`, where the standard
/// order builds a table over **3.79M** filtered `lineitem` rows and probes it with **57k**
/// orders. Semi joins are not commutative, so no plan-level rewrite can fix this — the sides
/// are fixed by the query, and only the *physical* build direction is free. That is what this
/// chooses.
///
/// **The relation is identical, row for row and in the same order.** Both directions emit
/// ascending left indices with a null right index: `probe_range` pushes `i` as it walks the
/// left side in order, and this pushes `i` as it walks `matched` in order. Nulls agree
/// because they are handled in the same two places — a null *probe* key is refused by
/// [`JoinTable::head_for`], and a null *build* key is never inserted into the table
/// ([`radix::partition_side`] skips it) — so a null-keyed left row is unmatched either way,
/// which `Semi` drops and `Anti` keeps, exactly as SQL requires. It is a pure performance
/// short-circuit, and the tests drive both directions over the same inputs to pin that.
#[allow(clippy::too_many_arguments)]
fn semi_anti_swapped(
    left_keys: &[ArrayRef],
    right_keys: &[ArrayRef],
    left_rows: usize,
    right_rows: usize,
    left_null: &[bool],
    right_null: &[bool],
    join_type: JoinType,
    bloom_fp_rate: f64,
    bloom_min_build_rows: usize,
) -> Result<JoinIndices, RuntimeError> {
    // The build side is now the left, so the bloom is sized and admitted on *its* row count.
    let use_bloom = use_probe_bloom_with(left_rows, right_rows, bloom_min_build_rows);
    // Every `JoinKeys` implementation names its two sides "right" (build) and "left" (probe).
    // Constructing one with the arguments exchanged is what puts our left on the build side;
    // nothing below this line has to know the swap happened.
    if let Some(keys) = I64Keys::try_new(right_keys, left_keys) {
        return Ok(mark_probe(
            &keys,
            left_rows,
            right_rows,
            left_null,
            right_null,
            join_type,
            use_bloom,
            bloom_fp_rate,
        ));
    }
    if let Some(keys) = I64x2Keys::try_new(right_keys, left_keys) {
        return Ok(mark_probe(
            &keys,
            left_rows,
            right_rows,
            left_null,
            right_null,
            join_type,
            use_bloom,
            bloom_fp_rate,
        ));
    }
    with_bytes_keys!(right_keys, left_keys, |keys| {
        return Ok(mark_probe(
            &keys,
            left_rows,
            right_rows,
            left_null,
            right_null,
            join_type,
            use_bloom,
            bloom_fp_rate,
        ));
    });
    // The converter is built from the *build* side's types, which is now the left.
    let fields: Vec<SortField> = left_keys
        .iter()
        .map(|a| SortField::new(a.data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields)?;
    let keys = RowKeys {
        right: converter.convert_columns(left_keys)?,
        left: converter.convert_columns(right_keys)?,
    };
    Ok(mark_probe(
        &keys,
        left_rows,
        right_rows,
        left_null,
        right_null,
        join_type,
        use_bloom,
        bloom_fp_rate,
    ))
}

/// Build over the returned side, mark it with the discarded side, then emit in row order.
#[allow(clippy::too_many_arguments)]
fn mark_probe<K: JoinKeys + Sync>(
    keys: &K,
    build_rows: usize,
    probe_rows: usize,
    build_null: &[bool],
    probe_null: &[bool],
    join_type: JoinType,
    use_bloom: bool,
    bloom_fp_rate: f64,
) -> JoinIndices {
    use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

    let table = JoinTable::build(keys, build_rows, build_null, use_bloom, bloom_fp_rate);
    let matched: Vec<AtomicBool> = (0..build_rows).map(|_| AtomicBool::new(false)).collect();

    // The probe side is the *large* one here — that is the entire premise of the swap — so
    // scanning it on one core throws away more than the smaller build saves. Measured on
    // TPC-H q4: a sequential mark made the query 45 ms → 90 ms against the parallel radix
    // path it replaced, which is a 2x regression rather than the win the smaller build
    // predicted. Row ranges are independent (a mark is a write, never a read), so this
    // parallelizes with no coordination beyond the relaxed store.
    let threads = rayon::current_num_threads().max(1);
    // Never smaller than a morsel: below that the range is pure scheduling overhead.
    let chunk = probe_rows
        .div_ceil(threads)
        .max(bc_arrow::DEFAULT_MORSEL_ROWS);
    let ranges: Vec<std::ops::Range<usize>> = (0..probe_rows)
        .step_by(chunk)
        .map(|s| s..(s + chunk).min(probe_rows))
        .collect();
    ranges
        .par_iter()
        .for_each(|r| table.mark_range(keys, r.clone(), Some(probe_null), &matched));

    // `Semi` keeps the rows something matched; `Anti` keeps exactly the others. Ascending
    // build index, which is the order the build-right path emits too.
    let keep = matches!(join_type, JoinType::Semi);
    let mut left_out = IndexBuf::with_capacity(build_rows);
    let mut right_out = IndexBuf::with_capacity(build_rows);
    for (i, hit) in matched.iter().enumerate() {
        if hit.load(Relaxed) == keep {
            left_out.push(i as u32);
            right_out.push_null();
        }
    }
    JoinIndices::from_bufs(left_out, right_out)
}

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
        Some(left_null),
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
    // Decode dictionary keys *before* the float fold, in the order the two demand: a dictionary
    // of floats must be decoded before its floats can be canonicalized. The two sides of a join
    // are reached by different operator chains, so one can carry a dictionary while the other
    // carries decoded values, and `RowConverter` — built from one side's type and fed both —
    // then rejects the join outright. Same "one canonical form" argument as the fold below; see
    // `keys::decode_dict_keys`. `None` when no key is a dictionary, so nothing is allocated.
    let l_dec = crate::keys::decode_dict_keys(left_keys);
    let r_dec = crate::keys::decode_dict_keys(right_keys);
    let left_keys: &[ArrayRef] = l_dec.as_deref().unwrap_or(left_keys);
    let right_keys: &[ArrayRef] = r_dec.as_deref().unwrap_or(right_keys);

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
                        Some(&left_null),
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
    with_bytes_keys!(left_keys, right_keys, |keys| {
        return Ok(run!(keys));
    });
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
        // `logical_nulls`, never `nulls`. A column whose values are *all* null arrives as
        // arrow's `Null` type, which encodes nullity in the type itself and carries no
        // validity buffer at all — so `nulls()` is `None` and `null_count()` is **0** for a
        // column in which every single value is null. Reading it that way made every null
        // key look like a valid, equal key, and an equi-join on such a column produced the
        // full cartesian product where SQL requires no rows at all (`NULL = NULL` is
        // unknown). `logical_nulls` is arrow's answer to exactly this: it materializes the
        // nullity a type implies, so `Null`, dictionary and run-end arrays all report the
        // nulls they logically hold.
        let Some(nulls) = key.logical_nulls() else {
            continue;
        };
        if nulls.null_count() == 0 {
            continue;
        }
        combined = NullBuffer::union(combined.as_ref(), Some(&nulls));
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

    #[test]
    fn radix_row_bytes_counts_only_what_the_probe_touches() {
        // An `i64` key: the (i64, u32) pair costs 12 bytes of payload, the chain 4, the
        // table ~6. A wider key must cost strictly more, or the sizing is ignoring it.
        let i64_row = radix_row_bytes(8);
        assert_eq!(i64_row, 12 + 4 + (5 * 8 / 7 + 1));
        assert!(radix_row_bytes(16) > i64_row);
        assert!(radix_row_bytes(4) < i64_row);
        // Never zero: it is a divisor in the partition-size computation.
        assert!(radix_row_bytes(0) > 0);
    }

    #[test]
    fn radix_part_rows_stay_inside_the_hosts_l2() {
        for key_bytes in [4usize, 8, 16, 32] {
            let rows = radix_part_rows(key_bytes);
            assert!(
                (1 << 12..=RADIX_PART_ROWS).contains(&rows),
                "{key_bytes}-byte key produced {rows} rows, outside the clamp"
            );
            // Unless the clamp bound it, the partition's state must fit the L2 budget the
            // topology reports — that is the entire claim this sizing makes.
            let budget = bc_arrow::CpuTopology::detect().l2_bytes / 2;
            if rows > 1 << 12 {
                assert!(
                    rows * radix_row_bytes(key_bytes) <= budget,
                    "{key_bytes}-byte key: {rows} rows exceeds the {budget}-byte L2 budget"
                );
            }
        }
        // A wider key never partitions more coarsely than a narrow one.
        assert!(radix_part_rows(32) <= radix_part_rows(8));
    }

    #[test]
    fn radix_parts_grow_with_the_build_and_stay_powers_of_two() {
        let (small, small_shift) = radix_parts::<i64>(1_000);
        let (big, big_shift) = radix_parts::<i64>(50_000_000);
        assert!(small.is_power_of_two() && big.is_power_of_two());
        assert_eq!(small, 2, "a tiny build needs the floor, not a fan-out");
        assert!(big > small, "a large build must partition more finely");
        assert!(big <= RADIX_MAX_PARTS);
        // The shift must select exactly `log2(parts)` high bits of the hash.
        assert_eq!(small_shift, 64 - small.trailing_zeros());
        assert_eq!(64 - big_shift, big.trailing_zeros());
        // A wider key partitions at least as finely for the same row count.
        let (wide, _) = radix_parts::<[u8; 32]>(50_000_000);
        assert!(wide >= big);
    }

    /// Build a table over `build` and probe it with `probe` in `chunk`-row ranges — the way a
    /// morsel-at-a-time executor does, which is what drives [`BloomTrial`]'s per-range verdict.
    fn probe_in_chunks(
        build: &[i64],
        probe: &[i64],
        chunk: usize,
    ) -> (Vec<(u32, Option<u32>)>, JoinTable) {
        let bk = keys(build);
        let pk = keys(probe);
        let bnull = vec![false; build.len()];
        let pnull = vec![false; probe.len()];
        let k = I64Keys::try_new(&pk, &bk).expect("i64 keys");
        // `use_bloom = true`: this is about the runtime verdict, not the size heuristic.
        let table = JoinTable::build(
            &I64Keys::try_new(&bk, &bk).expect("i64 keys"),
            build.len(),
            &bnull,
            true,
            BLOOM_FP_RATE,
        );
        let mut pairs = Vec::new();
        for start in (0..probe.len()).step_by(chunk) {
            let end = (start + chunk).min(probe.len());
            let mut l = IndexBuf::with_capacity(end - start);
            let mut r = IndexBuf::with_capacity(end - start);
            table.probe_range(
                &k,
                start..end,
                Some(&pnull),
                JoinType::Inner,
                &mut l,
                &mut r,
                None,
            );
            let (la, ra) = (l.finish(), r.finish());
            for i in 0..la.len() {
                pairs.push((
                    la.value(i),
                    if ra.is_null(i) {
                        None
                    } else {
                        Some(ra.value(i))
                    },
                ));
            }
        }
        (pairs, table)
    }

    /// Spacing between the bloom tests' build keys, wide enough that
    /// [`dense::DenseHeads`] refuses the key set.
    ///
    /// This is load-bearing, not arbitrary. The dense direct map accepts any key whose value
    /// range is within a small multiple of its row count — which a contiguous `0..n` is — and
    /// a dense table needs **no bloom at all** (an out-of-range key is rejected by the index
    /// bound itself). Generating these keys contiguously therefore routes them past the hash
    /// build entirely, leaves `bloom_trial` untouched at its default, and makes both
    /// assertions below vacuously true. Keep the keys spread.
    const BLOOM_TEST_KEY_SPREAD: i64 = 1 << 12;

    /// `n` build keys spread past the dense map's range budget — see
    /// [`BLOOM_TEST_KEY_SPREAD`].
    fn spread_keys(n: i64) -> Vec<i64> {
        (0..n).map(|i| i * BLOOM_TEST_KEY_SPREAD).collect()
    }

    /// The bloom is a pure short-circuit, so latching it off mid-probe must not change a single
    /// emitted row. Probed in chunks well past [`BLOOM_TRIAL_ROWS`] so the verdict actually flips
    /// part-way through — exactly the window a wrong implementation would corrupt.
    #[test]
    fn latching_the_bloom_off_midway_emits_the_same_rows() {
        // Every probe key matches — the shape that makes the bloom useless (TPC-H lineitem⋈orders).
        let build: Vec<i64> = spread_keys(40_000);
        let probe: Vec<i64> = (0..(BLOOM_TRIAL_ROWS as i64 * 3))
            .map(|i| (i % 40_000) * BLOOM_TEST_KEY_SPREAD)
            .collect();

        let (chunked, table) = probe_in_chunks(&build, &probe, 8_192);
        assert!(
            !table.bloom_trial.worth_consulting(),
            "a bloom that rejected nothing over 3x the trial must have been latched off",
        );

        // The oracle: one range, so the verdict never flips and the bloom is consulted throughout.
        let (whole, _) = probe_in_chunks(&build, &probe, probe.len());
        assert_eq!(
            chunked, whole,
            "latching the bloom off must not change the emitted pairs",
        );
        assert_eq!(
            chunked.len(),
            probe.len(),
            "every probe row has exactly one match",
        );
    }

    /// The other direction, and the reason the filter exists: a genuinely selective join must
    /// *keep* its bloom. Without this, "switch the bloom off" would read as a free win and would
    /// silently cost every selective probe its short-circuit.
    #[test]
    fn a_selective_bloom_is_kept() {
        let build: Vec<i64> = spread_keys(40_000);
        // 1 probe row in 50 can match; the rest fall far outside the build's key range (whose
        // maximum is `40_000 * BLOOM_TEST_KEY_SPREAD`, so a billion is clear of every key).
        let probe: Vec<i64> = (0..(BLOOM_TRIAL_ROWS as i64 * 3))
            .map(|i| {
                if i % 50 == 0 {
                    (i % 40_000) * BLOOM_TEST_KEY_SPREAD
                } else {
                    1_000_000_000 + i
                }
            })
            .collect();

        let (_, table) = probe_in_chunks(&build, &probe, 8_192);
        assert!(
            table.bloom_trial.worth_consulting(),
            "a bloom rejecting ~98% of probe rows must stay engaged",
        );
    }

    /// The dense direct map must emit exactly what the hash table emits.
    ///
    /// Scaling every key by a constant is a bijection, so it leaves the join's *structure*
    /// untouched — the same probe rows pair with the same build rows — while moving the key
    /// set from inside the dense map's range budget to outside it. So the same logical join
    /// runs down both paths and the index pairs must match element for element, including
    /// the chain order for duplicated keys and the placement of unmatched rows.
    ///
    /// Without this, the dense path could disagree with the hash path on duplicate-key chain
    /// order and every existing test would still pass, because they all use one path or the
    /// other, never both on the same relation.
    #[test]
    fn the_dense_map_and_the_hash_table_emit_identical_pairs() {
        // Duplicated build keys (so chains exist), gaps, and probe keys that miss.
        let build_raw: Vec<i64> = vec![3, 7, 3, 0, 12, 7, 7, 5, 3];
        let probe_raw: Vec<i64> = vec![7, 3, 99, 0, 12, 5, 7, 100, 3, 0];

        for join_type in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Semi,
            JoinType::Anti,
            JoinType::Right,
            JoinType::Full,
        ] {
            let dense = hash_join_indices(&keys(&probe_raw), &keys(&build_raw), join_type).unwrap();
            // The same relation, keys spread past the dense range budget ⇒ the hash path.
            let scale =
                |v: &[i64]| -> Vec<i64> { v.iter().map(|k| k * BLOOM_TEST_KEY_SPREAD).collect() };
            let hashed = hash_join_indices(
                &keys(&scale(&probe_raw)),
                &keys(&scale(&build_raw)),
                join_type,
            )
            .unwrap();
            assert_eq!(
                dense.left, hashed.left,
                "left indices diverged for {join_type:?}"
            );
            assert_eq!(
                dense.right, hashed.right,
                "right indices diverged for {join_type:?}"
            );
        }
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
    /// An all-null key column is arrow's `Null` type, which carries no validity buffer: it
    /// reports `null_count() == 0` while every value in it is null. Reading nullity that way
    /// made a join match null against null and return the full cartesian product, where SQL
    /// requires no rows at all — and it did so only on the streaming path, so `collect()` was
    /// right and `iter_batches()` was silently wrong on the same query.
    #[test]
    fn an_all_null_key_column_is_masked_even_though_it_carries_no_validity_buffer() {
        use arrow::array::NullArray;

        let keys: Vec<ArrayRef> = vec![Arc::new(NullArray::new(3))];
        assert_eq!(
            keys[0].null_count(),
            0,
            "the trap: arrow reports no nulls here"
        );
        assert_eq!(
            keys[0].logical_null_count(),
            3,
            "while every value is logically null"
        );
        assert_eq!(null_mask(&keys, 3), vec![true, true, true]);
    }

    /// A key with a validity buffer is unaffected, which is what keeps the streaming probe's
    /// fast path (no mask allocated for a never-null foreign key) intact.
    #[test]
    fn a_typed_key_still_masks_exactly_its_nulls() {
        let keys: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![Some(1), None, Some(3)]))];
        assert_eq!(keys[0].logical_null_count(), 1);
        assert_eq!(null_mask(&keys, 3), vec![false, true, false]);
    }

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

    /// The three-column integer fast path must match the row-encoded oracle for every join
    /// type. The cases that matter are the ones a per-column hash could get wrong where the
    /// row encoding cannot: rows agreeing on a *prefix* of the key but not the whole of it,
    /// a permuted key tuple (`(1,2,3)` against `(3,2,1)` — the columns must not commute), a
    /// duplicate key on both sides, and a null in one column only.
    #[test]
    fn i64xn_fast_path_matches_row_encoded() {
        let col = |v: Vec<Option<i64>>| -> ArrayRef { Arc::new(Int64Array::from(v)) };
        let left: Vec<ArrayRef> = vec![
            col(vec![Some(1), Some(1), Some(3), Some(2), None, Some(1)]),
            col(vec![Some(2), Some(2), Some(2), Some(2), Some(2), Some(9)]),
            col(vec![Some(3), Some(4), Some(1), Some(2), Some(3), Some(3)]),
        ];
        let right: Vec<ArrayRef> = vec![
            col(vec![Some(1), Some(3), Some(1), Some(2), Some(1), None]),
            col(vec![Some(2), Some(2), Some(2), Some(2), Some(9), Some(2)]),
            col(vec![Some(3), Some(1), Some(3), Some(9), Some(3), Some(3)]),
        ];
        let ln = null_mask(&left, 6);
        let rn = null_mask(&right, 6);
        let fastkeys = I64xNKeys::try_new(&left, &right).expect("three Int64 columns per side");
        let conv = RowConverter::new(vec![SortField::new(DataType::Int64); 3]).unwrap();
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
                let fast = build_probe_flat(&fastkeys, 6, 6, &ln, &rn, jt, bloom, BLOOM_FP_RATE);
                let slow = build_probe_flat(&rowkeys, 6, 6, &ln, &rn, jt, bloom, BLOOM_FP_RATE);
                assert_eq!(
                    sorted_pairs(&fast),
                    sorted_pairs(&slow),
                    "i64xN vs row mismatch for {jt:?} bloom={bloom}"
                );
            }
        }
    }

    /// The single byte-array fast path (`BytesKeys`) must match the row-encoded oracle for
    /// every join type. The cases that matter are the ones where raw bytes and arrow's
    /// escaped row encoding could conceivably disagree: an empty string (a zero-length
    /// slice, which the row format encodes with its own token), a value that is a strict
    /// prefix of another (`"a"` against `"ab"` — the length must separate them), a
    /// duplicate key on both sides, and a null key.
    #[test]
    fn bytes_fast_path_matches_row_encoded() {
        use arrow::array::StringArray;
        let left: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec![
            Some("a"),
            Some("ab"),
            Some("ab"),
            None,
            Some(""),
            Some("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"),
        ]))];
        let right: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec![
            Some("ab"),
            Some("ab"),
            Some("b"),
            None,
            Some(""),
            Some("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"),
        ]))];
        let ln = null_mask(&left, 6);
        let rn = null_mask(&right, 6);
        let bytekeys = BytesKeys::<i32>::try_new(&left, &right).expect("both single Utf8");
        let conv = RowConverter::new(vec![SortField::new(DataType::Utf8)]).unwrap();
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
                let fast = build_probe_flat(&bytekeys, 6, 6, &ln, &rn, jt, bloom, BLOOM_FP_RATE);
                let slow = build_probe_flat(&rowkeys, 6, 6, &ln, &rn, jt, bloom, BLOOM_FP_RATE);
                assert_eq!(
                    sorted_pairs(&fast),
                    sorted_pairs(&slow),
                    "bytes vs row mismatch for {jt:?} bloom={bloom}"
                );
            }
        }
    }

    /// A **sliced** key column indexes its own rows. `value_offsets` is offset-adjusted but
    /// `value_data` is the whole backing buffer, so a fast path that mixed the two would
    /// silently read another morsel's bytes — and morsels are almost always slices.
    #[test]
    fn bytes_fast_path_reads_only_a_slices_own_rows() {
        use arrow::array::StringArray;
        let whole: ArrayRef = Arc::new(StringArray::from(vec!["aa", "bb", "cc", "dd"]));
        let left: Vec<ArrayRef> = vec![whole.slice(2, 2)]; // "cc", "dd"
        let right: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec!["dd", "aa"]))];
        let ln = null_mask(&left, 2);
        let rn = null_mask(&right, 2);
        let keys = BytesKeys::<i32>::try_new(&left, &right).expect("both single Utf8");
        let out = build_probe_flat(&keys, 2, 2, &ln, &rn, JoinType::Inner, false, BLOOM_FP_RATE);
        // Only "dd" matches: left row 1 against right row 0.
        assert_eq!(sorted_pairs(&out), vec![(Some(1), Some(0))]);
    }

    /// A `LargeUtf8` key takes the `i64`-offset instantiation and the `i32` one declines,
    /// which is what keeps the macro's two arms from both firing.
    #[test]
    fn bytes_fast_path_picks_the_right_offset_width() {
        use arrow::array::{LargeStringArray, StringArray};
        let large: Vec<ArrayRef> = vec![Arc::new(LargeStringArray::from(vec!["a", "b"]))];
        assert!(BytesKeys::<i32>::try_new(&large, &large).is_none());
        assert!(BytesKeys::<i64>::try_new(&large, &large).is_some());
        let small: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec!["a", "b"]))];
        assert!(BytesKeys::<i32>::try_new(&small, &small).is_some());
        assert!(BytesKeys::<i64>::try_new(&small, &small).is_none());
        // Mismatched widths on the two sides decline both arms and keep the oracle.
        assert!(BytesKeys::<i32>::try_new(&small, &large).is_none());
        assert!(BytesKeys::<i64>::try_new(&small, &large).is_none());
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
            let with = hash_join_indices_impl(
                &left,
                &right,
                jt,
                true,
                BLOOM_FP_RATE,
                BLOOM_MIN_BUILD_ROWS,
            )
            .unwrap();
            let without = hash_join_indices_impl(
                &left,
                &right,
                jt,
                false,
                BLOOM_FP_RATE,
                BLOOM_MIN_BUILD_ROWS,
            )
            .unwrap();
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
        let with = hash_join_indices_impl(
            &left,
            &right,
            JoinType::Inner,
            true,
            BLOOM_FP_RATE,
            BLOOM_MIN_BUILD_ROWS,
        )
        .unwrap();
        let without = hash_join_indices_impl(
            &left,
            &right,
            JoinType::Inner,
            false,
            BLOOM_FP_RATE,
            BLOOM_MIN_BUILD_ROWS,
        )
        .unwrap();
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

    /// One matched row as `((left key col a, col b), (right key col a, col b))`, with
    /// `None` standing for the unmatched side of an outer join.
    type KeyPair2 = ((Option<i64>, Option<i64>), (Option<i64>, Option<i64>));

    fn vpairs2(
        idx: &JoinIndices,
        la: &[Option<i64>],
        lb: &[Option<i64>],
        ra: &[Option<i64>],
        rb: &[Option<i64>],
    ) -> Vec<KeyPair2> {
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

#[cfg(test)]
mod semi_swap_tests {
    use super::*;
    use std::sync::Arc;

    use arrow::array::{Int64Array, StringArray};

    fn i64_col(v: &[Option<i64>]) -> ArrayRef {
        Arc::new(Int64Array::from(v.to_vec()))
    }

    fn str_col(v: &[Option<&str>]) -> ArrayRef {
        Arc::new(StringArray::from(v.to_vec()))
    }

    /// `(left_index, right_index)` pairs in emitted order — order included on purpose, since
    /// the claim being tested is that the two directions agree on it and not merely on the set.
    fn pairs(idx: &JoinIndices) -> Vec<(Option<u32>, Option<u32>)> {
        (0..idx.left.len())
            .map(|i| {
                let l = (!idx.left.is_null(i)).then(|| idx.left.value(i));
                let r = (!idx.right.is_null(i)).then(|| idx.right.value(i));
                (l, r)
            })
            .collect()
    }

    /// Run one shape both ways and require the *same relation in the same order*.
    ///
    /// `hash_join_indices` is the build-right oracle: every input here is far below
    /// [`SEMI_SWAP_MIN_PROBE_ROWS`], so the dispatch inside it never takes the swap and it is
    /// genuinely the other implementation. `semi_anti_swapped` is called directly, which is
    /// what lets the equivalence be pinned on small, readable inputs instead of on the 65k
    /// rows the production threshold requires.
    fn assert_same_relation(left: &[ArrayRef], right: &[ArrayRef], what: &str) {
        let left_rows = left[0].len();
        let right_rows = right[0].len();
        for jt in [JoinType::Semi, JoinType::Anti] {
            let oracle = hash_join_indices(left, right, jt).unwrap();
            let l_null = null_mask(left, left_rows);
            let r_null = null_mask(right, right_rows);
            for &bloom in &[false, true] {
                // Drive the bloom both ways: it has no false negatives, so it may only skip a
                // provably-empty chain and the relation must be identical either setting.
                let swapped = semi_anti_swapped(
                    left,
                    right,
                    left_rows,
                    right_rows,
                    &l_null,
                    &r_null,
                    jt,
                    BLOOM_FP_RATE,
                    if bloom { 0 } else { usize::MAX },
                )
                .unwrap();
                assert_eq!(
                    pairs(&oracle),
                    pairs(&swapped),
                    "{what}: {jt:?} diverged (bloom={bloom})"
                );
            }
        }
    }

    #[test]
    fn swapped_semi_anti_matches_the_build_right_oracle() {
        // Plain: some match, some do not.
        assert_same_relation(
            &[i64_col(&[Some(1), Some(2), Some(3), Some(4)])],
            &[i64_col(&[Some(2), Some(4), Some(9)])],
            "plain",
        );
        // Duplicates on the returned side: every copy is judged independently.
        assert_same_relation(
            &[i64_col(&[Some(1), Some(1), Some(2), Some(1)])],
            &[i64_col(&[Some(1)])],
            "duplicate left keys",
        );
        // Duplicates on the discarded side: a chain longer than one, so the walk in
        // `mark_range` is exercised rather than the `unique` short-circuit.
        assert_same_relation(
            &[i64_col(&[Some(1), Some(2)])],
            &[i64_col(&[Some(1), Some(1), Some(1), Some(2)])],
            "duplicate right keys",
        );
        // NULL on the returned side never matches: `Semi` drops it, `Anti` keeps it.
        assert_same_relation(
            &[i64_col(&[Some(1), None, Some(3)])],
            &[i64_col(&[Some(1), Some(3)])],
            "null left key",
        );
        // NULL on the discarded side matches nothing, so it cannot rescue a left row.
        assert_same_relation(
            &[i64_col(&[Some(1), Some(2)])],
            &[i64_col(&[None, Some(2)])],
            "null right key",
        );
        assert_same_relation(
            &[i64_col(&[None, None])],
            &[i64_col(&[None, None])],
            "nulls on both sides",
        );
        // Degenerate sizes.
        assert_same_relation(&[i64_col(&[])], &[i64_col(&[Some(1)])], "empty left");
        assert_same_relation(&[i64_col(&[Some(1)])], &[i64_col(&[])], "empty right");
        assert_same_relation(
            &[i64_col(&[Some(1), Some(2)])],
            &[i64_col(&[Some(7), Some(8)])],
            "no matches at all",
        );
        assert_same_relation(
            &[i64_col(&[Some(1), Some(2)])],
            &[i64_col(&[Some(1), Some(2)])],
            "everything matches",
        );
        // The `RowKeys` (row-encoded) path — a string key has no integer fast path.
        assert_same_relation(
            &[str_col(&[Some("a"), None, Some("c"), Some("a")])],
            &[str_col(&[Some("a"), Some("z"), None])],
            "string key",
        );
        // The `I64x2Keys` (composite integer) path.
        assert_same_relation(
            &[
                i64_col(&[Some(1), Some(1), Some(2)]),
                i64_col(&[Some(10), Some(20), Some(10)]),
            ],
            &[i64_col(&[Some(1), Some(2)]), i64_col(&[Some(20), Some(10)])],
            "two-column integer key",
        );
    }

    /// The threshold must actually route a real semi join through the swapped path, and the
    /// answer must still be the oracle's. Sized past [`SEMI_SWAP_MIN_PROBE_ROWS`] so the
    /// production dispatch — not the direct call above — is what is under test.
    #[test]
    fn the_dispatch_takes_the_swap_and_still_agrees() {
        // TPC-H q4's shape in miniature: a small returned side against a large discarded one.
        let left: Vec<Option<i64>> = (0..2_000).map(Some).collect();
        let right: Vec<Option<i64>> = (0..100_000).map(|i| Some(i % 3_000)).collect();
        assert!(
            swap_semi_build(left.len(), right.len()),
            "this shape must reach the swapped path or the test proves nothing"
        );
        let l = [i64_col(&left)];
        let r = [i64_col(&right)];
        for jt in [JoinType::Semi, JoinType::Anti] {
            // The oracle, with the swap refused by making the ratio unreachable.
            let l_null = null_mask(&l, left.len());
            let r_null = null_mask(&r, right.len());
            let oracle = build_probe_flat(
                &I64Keys::try_new(&l, &r).unwrap(),
                left.len(),
                right.len(),
                &l_null,
                &r_null,
                jt,
                false,
                BLOOM_FP_RATE,
            );
            let dispatched = hash_join_indices(&l, &r, jt).unwrap();
            assert_eq!(pairs(&oracle), pairs(&dispatched), "{jt:?} at scale");
        }
        // Keys 0..2000 all occur in the right side (which cycles 0..3000), so Semi keeps
        // everything and Anti keeps nothing — a check that the relation is the expected one
        // and not merely self-consistent.
        let semi = hash_join_indices(&l, &r, JoinType::Semi).unwrap();
        assert_eq!(semi.left.len(), 2_000);
        let anti = hash_join_indices(&l, &r, JoinType::Anti).unwrap();
        assert_eq!(anti.left.len(), 0);
    }

    /// The packing is the entire correctness argument for the byte radix arm, so it is
    /// tested as the property it has to have: distinct keys never share a word.
    #[test]
    fn pack_byte_key_is_injective() {
        use std::collections::HashMap;
        let mut keys: Vec<Vec<u8>> = vec![
            b"".to_vec(),
            b"a".to_vec(),
            b"ab".to_vec(),
            b"abc".to_vec(),
            // prefixes of one another, which a length-blind packing would collide
            b"id1".to_vec(),
            b"id10".to_vec(),
            b"id100".to_vec(),
            // embedded NUL, which a C-string-style packing would truncate
            b"a\0b".to_vec(),
            b"a\0".to_vec(),
            // the widest packable key, and one differing only in its last byte
            vec![0xFF; 15],
            {
                let mut v = vec![0xFF; 15];
                v[14] = 0xFE;
                v
            },
        ];
        keys.push(b"id1000001".to_vec()); // the shape this arm exists for
        let mut seen: HashMap<u128, Vec<u8>> = HashMap::new();
        for k in &keys {
            let w = pack_byte_key(k);
            if let Some(prev) = seen.insert(w, k.clone()) {
                panic!("collision: {prev:?} and {k:?} both pack to {w:#x}");
            }
        }
        // ...and equal keys must pack equal, which is the other half of "same predicate".
        assert_eq!(pack_byte_key(b"id1000001"), pack_byte_key(b"id1000001"));
    }

    #[test]
    fn byte_keys_packable_boundary() {
        let fits: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec!["123456789012345", "a"]))];
        let over: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec!["1234567890123456", "a"]))];
        let k = BytesKeys::<i32>::try_new(&fits, &fits).unwrap();
        assert!(byte_keys_packable(&k.left, 2), "15 bytes must pack");
        let k = BytesKeys::<i32>::try_new(&over, &over).unwrap();
        assert!(!byte_keys_packable(&k.left, 2), "16 bytes must decline");
    }

    /// The byte-key counterpart of [`radix_matches_flat`]: the packed cache-radix path MUST
    /// produce the identical relation as the flat byte oracle, for every left-driven join
    /// type, over duplicates, misses, nulls, empty sides and keys that are prefixes of one
    /// another.
    #[test]
    fn packed_byte_radix_matches_flat() {
        fn arr(v: &[&str]) -> Vec<ArrayRef> {
            vec![Arc::new(StringArray::from(v.to_vec())) as ArrayRef]
        }
        fn nulls(v: &[&str], is_null: &[usize]) -> Vec<ArrayRef> {
            let opts: Vec<Option<&str>> = v
                .iter()
                .enumerate()
                .map(|(i, &x)| if is_null.contains(&i) { None } else { Some(x) })
                .collect();
            vec![Arc::new(StringArray::from(opts)) as ArrayRef]
        }
        let cases: Vec<(Vec<ArrayRef>, Vec<ArrayRef>)> = vec![
            (
                arr(&["id1", "id5", "id2", "id8", "id3", "id9"]),
                arr(&["id3", "id1", "id3", "id7", "id5", "id5", "id2", "id4"]),
            ),
            // prefixes and the empty string, where a length-blind key would over-match
            (arr(&["", "a", "ab", "abc"]), arr(&["a", "", "abc", "abcd"])),
            (
                nulls(&["id1", "id5", "id2", "id3", "id3"], &[2]),
                nulls(&["id3", "id1", "id3", "id7", "id5"], &[3]),
            ),
            (arr(&[]), arr(&["a", "b", "c"])),
            (arr(&["a", "b", "c"]), arr(&[])),
        ];
        for (li, ri) in &cases {
            let lrows = li[0].len();
            let rrows = ri[0].len();
            let lnull = null_mask(li, lrows);
            let rnull = null_mask(ri, rrows);
            let keys = BytesKeys::<i32>::try_new(li, ri).unwrap();
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
                let radix = radix_join_scalar(
                    |i| pack_byte_key(keys.right.get(i)),
                    |l| pack_byte_key(keys.left.get(l)),
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
                assert_eq!(a, b, "join type {jt:?} diverged on {li:?}");
            }
        }
    }
}
