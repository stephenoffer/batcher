//! Sortable key forms for a range join's axes, and the dense ranking built on them.
//!
//! The sorts are the largest phase of a two-condition range join (`report_range_join_phases`),
//! and almost all of that was comparator overhead rather than the sort itself. This module is
//! the answer: an order-preserving `u64` per value where the type allows one, the encoder
//! where it does not, and a dense rank so the sweep compares `u32`s and never an encoded key.

use arrow::array::{Array, ArrayRef};
use arrow::compute::SortOptions;
use arrow::datatypes::DataType;
use arrow::row::{Row, RowConverter, Rows, SortField};
use rayon::prelude::*;

use crate::error::RuntimeError;

/// Whether a key type can be range-joined through the row encoder.
///
/// The encoder's byte order has to agree with the comparison the planner rewrote away, so
/// only types with a single total order qualify. Nested and structured types are declined:
/// an inequality between two lists or structs is not a comparison this engine defines, so
/// there is no order for the encoder to reproduce.
pub(super) fn supported_key_type(dt: &DataType) -> bool {
    match dt {
        DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float32
        | DataType::Float64
        | DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Binary
        | DataType::LargeBinary
        | DataType::Date32
        | DataType::Date64
        | DataType::Time32(_)
        | DataType::Time64(_)
        | DataType::Timestamp(_, _)
        | DataType::Duration(_)
        | DataType::Decimal128(_, _)
        | DataType::Decimal256(_, _) => true,
        DataType::Dictionary(_, v) => supported_key_type(v),
        _ => false,
    }
}

/// Encode one condition's key columns with a shared converter, so the two sides' bytes are
/// mutually comparable and their byte order *is* the sort order the algorithm wants.
pub(super) fn encode_axis(
    left: &ArrayRef,
    right: &ArrayRef,
    descending: bool,
) -> Result<(Rows, Rows), RuntimeError> {
    let field = SortField::new_with_options(
        left.data_type().clone(),
        SortOptions {
            descending,
            // Null keys never reach the encoder (they are excluded up front), so this only
            // has to be *a* choice, not a meaningful one.
            nulls_first: false,
        },
    );
    let converter = RowConverter::new(vec![field])?;
    let l = converter.convert_columns(std::slice::from_ref(left))?;
    let r = converter.convert_columns(std::slice::from_ref(right))?;
    Ok((l, r))
}

/// An order-preserving `u64` for each value of a primitive key column, or `None` when the
/// type has no such encoding.
///
/// The sorts are ~70% of a two-condition range join's time (`report_range_join_phases`), and
/// almost all of that was comparator overhead rather than the sort itself: comparing two
/// `arrow::row::Row`s is an indirect load plus a `memcmp`, against a register compare here.
/// Null-keyed rows never reach this — they are excluded from the universe up front — so no
/// null byte is needed and the whole key fits in the 64 bits.
///
/// Floats use the standard total-order transform, which lands exactly on this engine's float
/// contract because [`canonicalize_float_keys`](crate::keys::canonicalize_float_keys) has
/// already folded `-0.0` into `0.0` and every NaN into one positive quiet NaN: the transform
/// then ranks that NaN above `+inf`, which is where the rest of the engine puts it.
pub(super) fn u64_order_keys(arr: &ArrayRef, descending: bool) -> Option<Vec<u64>> {
    use arrow::array::PrimitiveArray;
    use arrow::datatypes::*;

    #[inline]
    fn signed(v: i64) -> u64 {
        (v as u64) ^ (1u64 << 63)
    }
    #[inline]
    fn float(bits: u64) -> u64 {
        if bits >> 63 == 1 {
            !bits
        } else {
            bits ^ (1u64 << 63)
        }
    }

    macro_rules! prim {
        ($ty:ty, $f:expr) => {{
            let a = arr.as_any().downcast_ref::<PrimitiveArray<$ty>>()?;
            #[allow(clippy::redundant_closure_call)]
            Some(a.values().iter().map(|&v| $f(v)).collect::<Vec<u64>>())
        }};
    }

    let keys: Option<Vec<u64>> = match arr.data_type() {
        DataType::Int8 => prim!(Int8Type, |v: i8| signed(v as i64)),
        DataType::Int16 => prim!(Int16Type, |v: i16| signed(v as i64)),
        DataType::Int32 => prim!(Int32Type, |v: i32| signed(v as i64)),
        DataType::Int64 => prim!(Int64Type, signed),
        DataType::UInt8 => prim!(UInt8Type, |v: u8| v as u64),
        DataType::UInt16 => prim!(UInt16Type, |v: u16| v as u64),
        DataType::UInt32 => prim!(UInt32Type, |v: u32| v as u64),
        DataType::UInt64 => prim!(UInt64Type, |v: u64| v),
        DataType::Float32 => prim!(Float32Type, |v: f32| float((v as f64).to_bits())),
        DataType::Float64 => prim!(Float64Type, |v: f64| float(v.to_bits())),
        DataType::Date32 => prim!(Date32Type, |v: i32| signed(v as i64)),
        DataType::Date64 => prim!(Date64Type, signed),
        DataType::Time32(TimeUnit::Second) => prim!(Time32SecondType, |v: i32| signed(v as i64)),
        DataType::Time32(TimeUnit::Millisecond) => {
            prim!(Time32MillisecondType, |v: i32| signed(v as i64))
        }
        DataType::Time64(TimeUnit::Microsecond) => prim!(Time64MicrosecondType, signed),
        DataType::Time64(TimeUnit::Nanosecond) => prim!(Time64NanosecondType, signed),
        DataType::Timestamp(TimeUnit::Second, _) => prim!(TimestampSecondType, signed),
        DataType::Timestamp(TimeUnit::Millisecond, _) => prim!(TimestampMillisecondType, signed),
        DataType::Timestamp(TimeUnit::Microsecond, _) => prim!(TimestampMicrosecondType, signed),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => prim!(TimestampNanosecondType, signed),
        DataType::Duration(TimeUnit::Second) => prim!(DurationSecondType, signed),
        DataType::Duration(TimeUnit::Millisecond) => prim!(DurationMillisecondType, signed),
        DataType::Duration(TimeUnit::Microsecond) => prim!(DurationMicrosecondType, signed),
        DataType::Duration(TimeUnit::Nanosecond) => prim!(DurationNanosecondType, signed),
        _ => None,
    };
    keys.map(|mut k| {
        if descending {
            for v in &mut k {
                *v = !*v;
            }
        }
        k
    })
}

/// One axis's sortable form for the whole universe: left rows first, then right rows.
///
/// `Fast` is the `u64` path above; `Rows` keeps the encoder for a type it cannot handle
/// (strings, decimals, booleans), so nothing is declined for lack of a fast path.
pub(super) enum AxisKeys {
    Fast(Vec<u64>),
    Encoded { left: Rows, right: Rows },
}

/// `(key - min) << 32 | entry` for every entry, when the key span fits in 32 bits.
///
/// The sort is the largest phase of the operator, and a sort is memory-bound: halving the
/// element from a padded `(u64, u32)` — 16 bytes — to a bare `u64` halves the traffic and
/// lets the comparison be a single register compare with no tuple destructuring. The entry
/// occupies the low bits, so ordering by the packed value orders by key and breaks ties by
/// entry, which is the reproducible total order the sweep already relied on.
///
/// The span condition is what a real column usually satisfies without anyone arranging it:
/// dates, identifiers, prices and timestamps-within-a-run all sit inside a 4-billion-wide
/// window even when their absolute values do not. `None` falls back to the pair sort, which
/// is correct for any span.
fn packed_keys(keys: &[u64]) -> Option<Vec<u64>> {
    if keys.is_empty() || keys.len() > u32::MAX as usize {
        return None;
    }
    let (mut lo, mut hi) = (u64::MAX, 0u64);
    for &k in keys {
        lo = lo.min(k);
        hi = hi.max(k);
    }
    if hi - lo >= 1u64 << 32 {
        return None;
    }
    Some(
        keys.iter()
            .enumerate()
            .map(|(e, &k)| ((k - lo) << 32) | e as u64)
            .collect(),
    )
}

/// Sort the packed keys and read the order and dense ranks off them in one pass.
fn rank_packed(mut packed: Vec<u64>) -> (Vec<u32>, Vec<u32>) {
    if packed.len() >= PARALLEL_SORT_MIN_ROWS {
        packed.par_sort_unstable();
    } else {
        packed.sort_unstable();
    }
    let mut order = Vec::with_capacity(packed.len());
    let mut ranks = vec![0u32; packed.len()];
    let mut rank = 0u32;
    let mut prev = u64::MAX;
    for (i, &p) in packed.iter().enumerate() {
        let k = p >> 32;
        if i == 0 || k != prev {
            if i != 0 {
                rank += 1;
            }
            prev = k;
        }
        let e = (p & 0xFFFF_FFFF) as u32;
        ranks[e as usize] = rank;
        order.push(e);
    }
    (order, ranks)
}

/// Sort `(key, entry)` pairs, on rayon once the input is large enough.
fn sort_u64_pairs(pairs: &mut [(u64, u32)]) {
    if pairs.len() >= PARALLEL_SORT_MIN_ROWS {
        pairs.par_sort_unstable();
    } else {
        pairs.sort_unstable();
    }
}

impl AxisKeys {
    /// Build the universe's keys for one condition, preferring the `u64` path.
    pub(super) fn build(
        left: &ArrayRef,
        right: &ArrayRef,
        descending: bool,
        lmap: &[u32],
        rmap: &[u32],
    ) -> Result<Self, RuntimeError> {
        if let (Some(l), Some(r)) = (
            u64_order_keys(left, descending),
            u64_order_keys(right, descending),
        ) {
            let mut keys = Vec::with_capacity(lmap.len() + rmap.len());
            keys.extend(lmap.iter().map(|&i| l[i as usize]));
            keys.extend(rmap.iter().map(|&i| r[i as usize]));
            return Ok(AxisKeys::Fast(keys));
        }
        let (left, right) = encode_axis(left, right, descending)?;
        Ok(AxisKeys::Encoded { left, right })
    }

    /// The universe entries in ascending key order **and** their dense ranks, in one pass.
    ///
    /// Ties are broken by entry, so the order — and therefore the output row order — is
    /// reproducible across runs. The two results are produced together because the ranking
    /// pass then reads the sorted keys *contiguously*: computing it afterwards from the
    /// order alone is a random-access gather over the key array, which measured as the
    /// single largest remaining phase after the sorts themselves.
    pub(super) fn sorted_order_and_ranks(
        &self,
        n: usize,
        nl: usize,
        lmap: &[u32],
        rmap: &[u32],
    ) -> (Vec<u32>, Vec<u32>) {
        match self {
            AxisKeys::Fast(keys) => match packed_keys(keys) {
                Some(packed) => rank_packed(packed),
                None => {
                    let mut pairs: Vec<(u64, u32)> = keys.iter().copied().zip(0u32..).collect();
                    sort_u64_pairs(&mut pairs);
                    let mut order = Vec::with_capacity(pairs.len());
                    let mut ranks = vec![0u32; pairs.len()];
                    let mut rank = 0u32;
                    let mut prev = pairs.first().map(|&(k, _)| k);
                    for &(k, e) in &pairs {
                        if prev != Some(k) {
                            rank += 1;
                            prev = Some(k);
                        }
                        ranks[e as usize] = rank;
                        order.push(e);
                    }
                    (order, ranks)
                }
            },
            AxisKeys::Encoded { left, right } => {
                let mut order: Vec<u32> = (0..n as u32).collect();
                sort_by_key(&mut order, |e| key(e, nl, left, right, lmap, rmap));
                let ranks = dense_ranks(&order, self, nl, lmap, rmap);
                (order, ranks)
            }
        }
    }

    /// Whether two universe entries have equal keys — the one thing dense ranking needs.
    #[inline]
    fn eq(&self, a: u32, b: u32, nl: usize, lmap: &[u32], rmap: &[u32]) -> bool {
        match self {
            AxisKeys::Fast(keys) => keys[a as usize] == keys[b as usize],
            AxisKeys::Encoded { left, right } => {
                key(a, nl, left, right, lmap, rmap) == key(b, nl, left, right, lmap, rmap)
            }
        }
    }

    /// Order two universe entries by key. Used by the one-inequality path, which searches
    /// directly rather than ranking (it has only one axis, so ranking would buy nothing).
    #[inline]
    pub(super) fn cmp(
        &self,
        a: u32,
        b: u32,
        nl: usize,
        lmap: &[u32],
        rmap: &[u32],
    ) -> std::cmp::Ordering {
        match self {
            AxisKeys::Fast(keys) => keys[a as usize].cmp(&keys[b as usize]),
            AxisKeys::Encoded { left, right } => {
                key(a, nl, left, right, lmap, rmap).cmp(&key(b, nl, left, right, lmap, rmap))
            }
        }
    }

    /// The right-side universe entries in ascending key order.
    pub(super) fn sorted_right(&self, n: usize, nl: usize, lmap: &[u32], rmap: &[u32]) -> Vec<u32> {
        match self {
            AxisKeys::Fast(keys) => {
                let mut pairs: Vec<(u64, u32)> = ((nl as u32)..(n as u32))
                    .map(|e| (keys[e as usize], e))
                    .collect();
                if pairs.len() >= PARALLEL_SORT_MIN_ROWS {
                    pairs.par_sort_unstable();
                } else {
                    pairs.sort_unstable();
                }
                pairs.into_iter().map(|(_, e)| e).collect()
            }
            AxisKeys::Encoded { left, right } => {
                let mut idx: Vec<u32> = ((nl as u32)..(n as u32)).collect();
                sort_by_key(&mut idx, |e| key(e, nl, left, right, lmap, rmap));
                idx
            }
        }
    }

    /// The flat order-preserving `u64` key per universe entry, when this axis has one.
    ///
    /// `None` for the encoded (row-format) axis, whose keys are variable-width bytes and
    /// cannot be handed out as a slice. Callers that want the fast path must keep a generic
    /// fallback for that case.
    pub(super) fn fast(&self) -> Option<&[u64]> {
        match self {
            AxisKeys::Fast(keys) => Some(keys),
            AxisKeys::Encoded { .. } => None,
        }
    }

    /// The **left**-side universe entries in ascending key order.
    ///
    /// The mirror of [`Self::sorted_right`], and the band join's reason for existing: with
    /// the left rows in key order their bounds into the sorted right side are monotone, so
    /// one merge pass replaces a binary search per row. At five million rows a side those
    /// searches were 11.5 s of a 11.8 s join — 23 random probes into a 40 MB array, five
    /// million times over, which no amount of parallelism makes cache-friendly.
    pub(super) fn sorted_left(&self, nl: usize, lmap: &[u32], rmap: &[u32]) -> Vec<u32> {
        match self {
            AxisKeys::Fast(keys) => {
                let mut pairs: Vec<(u64, u32)> =
                    (0..nl as u32).map(|e| (keys[e as usize], e)).collect();
                if pairs.len() >= PARALLEL_SORT_MIN_ROWS {
                    pairs.par_sort_unstable();
                } else {
                    pairs.sort_unstable();
                }
                pairs.into_iter().map(|(_, e)| e).collect()
            }
            AxisKeys::Encoded { left, right } => {
                let mut idx: Vec<u32> = (0..nl as u32).collect();
                sort_by_key(&mut idx, |e| key(e, nl, left, right, lmap, rmap));
                idx
            }
        }
    }
}

/// Dense ranks over a sorted order: equal keys share a rank, so comparing ranks is exactly
/// comparing keys — ties included.
///
/// This is what lets the sweep run on `u32` compares alone. Without it every cursor step and
/// every binary-search probe re-derives an `arrow::row::Row` and `memcmp`s it, which for a
/// two-condition join over millions of rows is most of the non-sort time.
pub(super) fn dense_ranks(
    order: &[u32],
    keys: &AxisKeys,
    nl: usize,
    lmap: &[u32],
    rmap: &[u32],
) -> Vec<u32> {
    let mut ranks = vec![0u32; order.len()];
    let mut rank = 0u32;
    for i in 0..order.len() {
        if i > 0 && !keys.eq(order[i], order[i - 1], nl, lmap, rmap) {
            rank += 1;
        }
        ranks[order[i] as usize] = rank;
    }
    ranks
}

/// Rows-to-sort floor above which the axis sorts are worth handing to rayon.
///
/// Below it the pool hand-off costs more than the sort saves; a few thousand encoded rows
/// sort in well under the time it takes to schedule them.
const PARALLEL_SORT_MIN_ROWS: usize = 32_768;

/// Sort `idx` by each entry's encoded key, in parallel once the input is large enough.
///
/// The sorts are the `n log n` term of both algorithms here, and the comparator is a pure
/// function of the encoded bytes, so the work fans out with no shared state. Worth 1.6-1.8x
/// on an interval-containment join in the hundred-thousand range; past a few million the
/// mark-array scan dominates instead and this stops being where the time goes.
///
/// The entry index breaks ties, which makes the order **total** rather than merely correct.
/// The algorithms do not need it (equal keys are contiguous, and both the axis-1 binary
/// search and the axis-2 cursor stop on a value boundary, so where a tie group's members sit
/// relative to each other never changes an answer) — but an unstable parallel sort would
/// otherwise let the *output row order* vary between runs of the same query, and a join whose
/// row order is reproducible is worth more than the nanoseconds the tie-break costs.
pub(super) fn sort_by_key<'a, F>(idx: &mut [u32], key_of: F)
where
    F: Fn(u32) -> Row<'a> + Sync,
{
    let cmp = |&a: &u32, &b: &u32| key_of(a).cmp(&key_of(b)).then(a.cmp(&b));
    if idx.len() >= PARALLEL_SORT_MIN_ROWS {
        idx.par_sort_unstable_by(cmp);
    } else {
        idx.sort_unstable_by(cmp);
    }
}

/// The encoded key of universe entry `e`: left rows occupy `[0, nl)`, right rows the rest.
#[inline]
pub(super) fn key<'a>(
    e: u32,
    nl: usize,
    l: &'a Rows,
    r: &'a Rows,
    lmap: &[u32],
    rmap: &[u32],
) -> Row<'a> {
    let e = e as usize;
    if e < nl {
        l.row(lmap[e] as usize)
    } else {
        r.row(rmap[e - nl] as usize)
    }
}
