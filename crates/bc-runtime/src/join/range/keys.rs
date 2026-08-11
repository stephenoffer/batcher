//! Sortable key forms for a range join's axes, and the dense ranking built on them.
//!
//! The sorts are the largest phase of a two-condition range join (`report_range_join_phases`),
//! and almost all of that was comparator overhead rather than the sort itself. This module is
//! the answer: an order-preserving `u64` per value where the type allows one, the encoder
//! where it does not, and a dense rank so the sweep compares `u32`s and never an encoded key.

use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};

use arrow::array::{Array, ArrayRef};
use arrow::compute::SortOptions;
use arrow::datatypes::DataType;
use arrow::row::{Row, RowConverter, Rows, SortField};
use rayon::prelude::*;

use crate::error::RuntimeError;
// The order-preserving `u64` key form and its parallel-map helpers moved to the crate's
// canonical key module: the window frame and ASOF paths need the same typed extraction, and
// a second copy of a key encoding is exactly the divergence `keys.rs` exists to prevent.
use crate::keys::{u64_order_keys, PARALLEL_MAP_MIN};

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
    // min/max is associative, so the fold order cannot change the result.
    let (lo, hi) = if keys.len() >= PARALLEL_MAP_MIN {
        keys.par_iter()
            .fold(|| (u64::MAX, 0u64), |(lo, hi), &k| (lo.min(k), hi.max(k)))
            .reduce(|| (u64::MAX, 0u64), |a, b| (a.0.min(b.0), a.1.max(b.1)))
    } else {
        keys.iter()
            .fold((u64::MAX, 0u64), |(lo, hi), &k| (lo.min(k), hi.max(k)))
    };
    if hi - lo >= 1u64 << 32 {
        return None;
    }
    // Each packed value is a pure function of its key and its own index, so an indexed
    // parallel `collect` writes exactly what the sequential `enumerate` did.
    if keys.len() >= PARALLEL_MAP_MIN {
        Some(
            keys.par_iter()
                .enumerate()
                .map(|(e, &k)| ((k - lo) << 32) | e as u64)
                .collect(),
        )
    } else {
        Some(
            keys.iter()
                .enumerate()
                .map(|(e, &k)| ((k - lo) << 32) | e as u64)
                .collect(),
        )
    }
}

/// Sort the packed keys and read the order and dense ranks off them.
///
/// The ranking looks inherently serial — a running counter bumped on every key change — but it
/// is a prefix sum in disguise, and that is what lets it be split. The rank at sorted position
/// `i` is exactly *the number of key changes in `1..=i`*, so a chunk needs only the count of
/// changes before it in order to compute all of its own ranks independently. Three phases:
/// count the changes per chunk in parallel, exclusive-scan those per-chunk counts (one pass over
/// as many entries as there are chunks, so it stays serial without mattering), then have each
/// chunk walk its own range from its base.
///
/// This was the last serial pass over the universe on either axis, and at five million rows a
/// side there are ten million entries of it, twice — once per axis, inside a phase whose sort
/// half was already parallel.
///
/// Every write is a reindexing of the same values: `order` is written at the index it is read
/// from, and `ranks` is scattered through `order`, which is a permutation, so each slot is
/// written exactly once. The slot type carries that disjointness the way `pos1` does in the
/// IEJoin sweep. The chunk-boundary comparison reads `packed[start - 1]`, i.e. across the chunk
/// edge, which is why the phases index the whole array rather than the chunk slices.
fn rank_packed(mut packed: Vec<u64>) -> (Vec<u32>, Vec<u32>) {
    if packed.len() >= PARALLEL_SORT_MIN_ROWS {
        packed.par_sort_unstable();
    } else {
        packed.sort_unstable();
    }
    let n = packed.len();

    // `i` opens a new dense rank when its key differs from its predecessor's. Position 0 never
    // does, which is what makes the first rank 0 rather than 1.
    let changes = |lo: usize, hi: usize| -> u32 {
        let mut c = 0u32;
        for i in lo.max(1)..hi {
            if packed[i] >> 32 != packed[i - 1] >> 32 {
                c += 1;
            }
        }
        c
    };

    if n < PARALLEL_MAP_MIN {
        let mut order = vec![0u32; n];
        let mut ranks = vec![0u32; n];
        let mut rank = 0u32;
        for i in 0..n {
            if i > 0 && packed[i] >> 32 != packed[i - 1] >> 32 {
                rank += 1;
            }
            let e = (packed[i] & 0xFFFF_FFFF) as u32;
            ranks[e as usize] = rank;
            order[i] = e;
        }
        return (order, ranks);
    }

    // One chunk per worker, floored so a modest universe is not split into slivers whose
    // scheduling costs more than their scan.
    let per = n
        .div_ceil(rayon::current_num_threads().max(1))
        .max(super::SETUP_CHUNK);
    let bounds: Vec<(usize, usize)> = (0..n.div_ceil(per))
        .map(|c| (c * per, ((c + 1) * per).min(n)))
        .collect();

    // Phase 1: how many ranks each chunk opens, counted independently.
    let counts: Vec<u32> = bounds.par_iter().map(|&(lo, hi)| changes(lo, hi)).collect();

    // Phase 2: exclusive scan, so `base[c]` is the number of changes strictly before chunk `c`
    // — which is the rank its first element carries before its own change is counted.
    let mut base = Vec::with_capacity(bounds.len());
    let mut acc = 0u32;
    for &c in &counts {
        base.push(acc);
        acc += c;
    }

    // Phase 3: each chunk walks its own range from its base.
    let order_slots: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
    let rank_slots: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
    bounds
        .par_iter()
        .zip(base.par_iter())
        .for_each(|(&(lo, hi), &b)| {
            let mut rank = b;
            for i in lo..hi {
                if i > 0 && packed[i] >> 32 != packed[i - 1] >> 32 {
                    rank += 1;
                }
                let e = (packed[i] & 0xFFFF_FFFF) as u32;
                rank_slots[e as usize].store(rank, AtomicOrdering::Relaxed);
                order_slots[i].store(e, AtomicOrdering::Relaxed);
            }
        });

    let unwrap = |slots: Vec<AtomicU32>| -> Vec<u32> {
        slots
            .into_iter()
            .map(|s| s.into_inner())
            .collect::<Vec<u32>>()
    };
    (unwrap(order_slots), unwrap(rank_slots))
}

/// Sort `(key, entry)` pairs, on rayon once the input is large enough.
fn sort_u64_pairs(pairs: &mut [(u64, u32)]) {
    if pairs.len() >= PARALLEL_SORT_MIN_ROWS {
        pairs.par_sort_unstable();
    } else {
        pairs.sort_unstable();
    }
}

/// The universe entries `first..last` in ascending key order, ties broken by entry.
///
/// Entries in that half-open range index `keys` directly, so the keys involved are a contiguous
/// subslice and the whole thing is one sort of one array.
///
/// Both one-sided orders — the band's sorted left and sorted right — used to build a
/// `Vec<(u64, u32)>`, sort that, and map the entries back out. Three costs, all avoidable: the
/// tuple pads to **16 bytes** where the information is 12 and fits in 8, and a sort is
/// memory-bound; and the build and extract passes ran on one core. This packs key and entry into
/// a single `u64` exactly as [`packed_keys`] does for the two-sided order, which was already
/// doing it — the one-sided paths simply never got the same treatment.
///
/// The order is **identical**, not merely equivalent: with the key in the high bits and the
/// entry in the low ones, ordering by the packed value orders by key and breaks ties by entry,
/// which is what the pair sort did. Packing needs the key span to fit in 32 bits, and the pair
/// sort remains for the spans that do not.
fn sorted_entries(keys: &[u64], first: u32, last: u32) -> Vec<u32> {
    let sub = &keys[first as usize..last as usize];
    let n = sub.len();

    // min/max is associative, so the fold order cannot change the span.
    let (lo, hi) = if n >= PARALLEL_MAP_MIN {
        sub.par_iter()
            .fold(|| (u64::MAX, 0u64), |(l, h), &k| (l.min(k), h.max(k)))
            .reduce(|| (u64::MAX, 0u64), |a, b| (a.0.min(b.0), a.1.max(b.1)))
    } else {
        sub.iter()
            .fold((u64::MAX, 0u64), |(l, h), &k| (l.min(k), h.max(k)))
    };

    if n == 0 || n > u32::MAX as usize || hi.wrapping_sub(lo) >= 1u64 << 32 {
        let mut pairs: Vec<(u64, u32)> = (first..last).map(|e| (keys[e as usize], e)).collect();
        sort_u64_pairs(&mut pairs);
        return pairs.into_iter().map(|(_, e)| e).collect();
    }

    let mut packed: Vec<u64> = if n >= PARALLEL_MAP_MIN {
        sub.par_iter()
            .enumerate()
            .map(|(i, &k)| ((k - lo) << 32) | (first as u64 + i as u64))
            .collect()
    } else {
        sub.iter()
            .enumerate()
            .map(|(i, &k)| ((k - lo) << 32) | (first as u64 + i as u64))
            .collect()
    };
    if n >= PARALLEL_SORT_MIN_ROWS {
        packed.par_sort_unstable();
    } else {
        packed.sort_unstable();
    }
    if n >= PARALLEL_MAP_MIN {
        packed.par_iter().map(|&p| p as u32).collect()
    } else {
        packed.iter().map(|&p| p as u32).collect()
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
            // The universe is the left rows' keys followed by the right rows', each gathered
            // through its side's row map. Preallocating and splitting lets both halves be
            // gathered in parallel; `extend` could not, and this is a pass over the whole
            // universe on every axis of every range join.
            let mut keys = vec![0u64; lmap.len() + rmap.len()];
            let (kl, kr) = keys.split_at_mut(lmap.len());
            rayon::join(|| gather_u64(kl, lmap, &l), || gather_u64(kr, rmap, &r));
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
            AxisKeys::Fast(keys) => sorted_entries(keys, nl as u32, n as u32),
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
            AxisKeys::Fast(keys) => sorted_entries(keys, 0, nl as u32),
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

/// Gather `src[i]` for each `i` in `idx` into `dst`, on rayon once it is large enough to pay.
///
/// `dst` and `idx` are the same length and are walked in lockstep, so element `j` of `dst`
/// receives `src[idx[j]]` exactly as the sequential loop wrote it.
fn gather_u64(dst: &mut [u64], idx: &[u32], src: &[u64]) {
    debug_assert_eq!(dst.len(), idx.len());
    if dst.len() >= PARALLEL_MAP_MIN {
        dst.par_iter_mut()
            .zip(idx.par_iter())
            .for_each(|(d, &i)| *d = src[i as usize]);
    } else {
        for (d, &i) in dst.iter_mut().zip(idx) {
            *d = src[i as usize];
        }
    }
}

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
