//! ASOF (nearest-match) join: each left row matched to the right row whose `on` key
//! is nearest in a direction within its `by` group. Left-style (every left row
//! emitted; unmatched → null right). Split out of the join module along the
//! algorithm seam; like the equi-join it carries no single-node assumption —
//! partitioning both sides by `by` makes a global ASOF the union of per-partition
//! ASOFs (the distributed seam).

use arrow::array::{Array, ArrayRef, UInt32Array};
use arrow::buffer::NullBuffer;
use arrow::datatypes::DataType;
use arrow::row::{OwnedRow, RowConverter, SortField};
use indexmap::IndexMap;

use super::{null_mask, JoinIndices};
use crate::error::RuntimeError;
use crate::measure::NumericKeys;

/// Which side of the left key an ASOF match may come from.
///
/// `Backward` (the default everywhere: pandas, Polars, DuckDB) takes the last known value
/// at or before the left row — the "what was the price when this trade happened" reading.
/// `Forward` takes the first value at or after it. `Nearest` takes whichever of the two is
/// closer, breaking an exact tie toward the backward one, matching pandas'
/// `direction="nearest"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsofDirection {
    Backward,
    Forward,
    Nearest,
}

/// Everything about an ASOF match beyond the keys themselves.
///
/// Bundled rather than passed as three more positional arguments, because the three
/// interact: `Nearest` and a `tolerance` both need a measurable key, and
/// `allow_exact_matches` moves the boundary both directions search from.
#[derive(Debug, Clone, Copy)]
pub struct AsofSpec {
    pub direction: AsofDirection,
    /// Cap on the distance between the matched keys, in the key's own units and in
    /// microseconds for a temporal key. `None` = uncapped.
    pub tolerance: Option<f64>,
    /// Whether a right row whose key *equals* the left row's may be the match.
    ///
    /// `false` is the strict form pandas spells `allow_exact_matches=False`: a backward
    /// join then takes the last row strictly *before* the left key. It is what keeps a
    /// backtest honest — a quote stamped at the same instant as the trade is information
    /// the trade did not have, and matching it is look-ahead bias that inflates every
    /// result downstream without ever looking like a bug.
    pub allow_exact_matches: bool,
}

impl Default for AsofSpec {
    fn default() -> Self {
        AsofSpec {
            direction: AsofDirection::Backward,
            tolerance: None,
            allow_exact_matches: true,
        }
    }
}

/// Compute ASOF (nearest-match) join indices. Every left row is emitted (left-style);
/// it is matched to the right row whose `on` key is nearest *in `direction`* within
/// the same `by` group (exact `by` equality). Unmatched left rows get a null right
/// index (arrow `take` then yields null), exactly like a left outer join.
///
/// [`AsofDirection`] chooses which side of the left key a match may come from. Keys are
/// arrow row-encoded, so `on` (order-preserving) and `by` (equality) work for any type.
/// Rows with a null `on` never match. As with the equi-join primitive, partitioning both
/// sides by `by` makes a global ASOF equal the union of per-partition ASOFs — the seam
/// the distributed path can use.
///
/// `tolerance` caps how far apart the two keys may be, in the key's own units and in
/// **microseconds** for any temporal key. Beyond it the left row is unmatched rather than
/// matched to a stale value, which is the difference between "the quote at the time of the
/// trade" and "some quote from three days earlier". It requires a numeric or temporal `on`
/// key, as does `Nearest`, because both have to subtract two keys; a non-numeric key with
/// either errors rather than silently ignoring the request.
pub fn asof_join_indices(
    left_on: &ArrayRef,
    right_on: &ArrayRef,
    left_by: &[ArrayRef],
    right_by: &[ArrayRef],
    spec: AsofSpec,
) -> Result<JoinIndices, RuntimeError> {
    if let Some(idx) = asof_int_fast(left_on, right_on, left_by, right_by, spec) {
        return Ok(idx);
    }
    asof_join_indices_general(left_on, right_on, left_by, right_by, spec)
}

/// An integer or temporal key's values widened to `i64`, or `None` for any other type.
///
/// Every type here orders exactly as its `i64` value does, which is what lets the fast path
/// compare plain integers where the general path compares arrow row encodings. Values under
/// a null slot are read and ignored, as arrow's own kernels do; the caller masks them.
fn int_key(a: &ArrayRef) -> Option<std::borrow::Cow<'_, [i64]>> {
    use arrow::array::AsArray;
    use arrow::datatypes::{
        DataType, Date32Type, Date64Type, Int32Type, Int64Type, TimeUnit, TimestampMicrosecondType,
        TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType,
    };
    use rayon::prelude::*;
    use std::borrow::Cow;
    let widen = |v: &[i32]| Cow::Owned(v.par_iter().map(|&x| i64::from(x)).collect());
    Some(match a.data_type() {
        DataType::Int64 => Cow::Borrowed(a.as_primitive::<Int64Type>().values().as_ref()),
        DataType::Date64 => Cow::Borrowed(a.as_primitive::<Date64Type>().values().as_ref()),
        DataType::Timestamp(unit, _) => Cow::Borrowed(match unit {
            TimeUnit::Second => a.as_primitive::<TimestampSecondType>().values().as_ref(),
            TimeUnit::Millisecond => a
                .as_primitive::<TimestampMillisecondType>()
                .values()
                .as_ref(),
            TimeUnit::Microsecond => a
                .as_primitive::<TimestampMicrosecondType>()
                .values()
                .as_ref(),
            TimeUnit::Nanosecond => a
                .as_primitive::<TimestampNanosecondType>()
                .values()
                .as_ref(),
        }),
        DataType::Int32 => widen(a.as_primitive::<Int32Type>().values()),
        DataType::Date32 => widen(a.as_primitive::<Date32Type>().values()),
        _ => return None,
    })
}

/// Whether [`asof_join_indices`] takes its parallel integer path for these key types and spec.
///
/// That path parallelizes inside one call, over the whole input, so a caller that would
/// otherwise hash-partition both sides by `by` to get parallelism -- copying every column of
/// both -- should hand it the whole input instead. Decided from types alone, before any data is
/// touched; it is exactly the precondition the path itself checks.
pub fn asof_is_whole_input(
    left_on: &DataType,
    right_on: &DataType,
    left_by: &[DataType],
    right_by: &[DataType],
    spec: AsofSpec,
) -> bool {
    let int = |t: &DataType| {
        matches!(
            t,
            DataType::Int64
                | DataType::Int32
                | DataType::Date32
                | DataType::Date64
                | DataType::Timestamp(_, _)
        )
    };
    spec.tolerance.is_none()
        && spec.direction != AsofDirection::Nearest
        && left_on == right_on
        && int(left_on)
        && match (left_by, right_by) {
            ([], []) => true,
            ([l], [r]) => l == r && int(l),
            _ => false,
        }
}

/// [`asof_join_indices`] for the shape nearly every ASOF join has: an integer or temporal `on`
/// key, at most one integer `by` key, a backward or forward match, no tolerance. `None` for any
/// other shape, which takes the general path.
///
/// The general path is fully general -- row-encoded keys of any type -- and pays for it per
/// row: a heap-allocated group key and an owned `on` row for every right row, a hash lookup
/// and a key allocation for every left row, all on one thread. Polars joined 1M rows against
/// 1M in 49 ms where that took 1,054 ms. Here the right side is one flat `(by, on, row)`
/// array sorted once, the left side is put in the same order, and the two are merged across
/// the pool.
///
/// The result is the general path's, row for row. Sorting by `(by, on, row)` orders a group
/// by `on` with ties in row order, which is exactly what the general path's *stable* sort of
/// each group by `on` produces; the searches are the same `partition_point`s over it; and the
/// null rules are its own -- a null `on` or `by` on either side matches nothing.
fn asof_int_fast(
    left_on: &ArrayRef,
    right_on: &ArrayRef,
    left_by: &[ArrayRef],
    right_by: &[ArrayRef],
    spec: AsofSpec,
) -> Option<JoinIndices> {
    use arrow::buffer::BooleanBuffer;
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    let types = |a: &[ArrayRef]| a.iter().map(|c| c.data_type().clone()).collect::<Vec<_>>();
    if !asof_is_whole_input(
        left_on.data_type(),
        right_on.data_type(),
        &types(left_by),
        &types(right_by),
        spec,
    ) {
        return None;
    }
    let (lby, rby) = match (left_by, right_by) {
        ([l], [r]) => (Some((l, int_key(l)?)), Some((r, int_key(r)?))),
        _ => (None, None),
    };
    let (lon, ron) = (int_key(left_on)?, int_key(right_on)?);
    let entries = &sorted_tuples(right_on, &ron, rby.as_ref());
    let exact = spec.allow_exact_matches;
    let backward = spec.direction == AsofDirection::Backward;
    // The general path's two boundaries, over `on` within one `by` group: `back` takes the
    // last right row at (or, strictly, before) the left key, `fwd` the first at (or after) it.
    // `passed` says whether right entry `e` lies before that boundary for probe `(k, t)`.
    let passed = |e: &(i64, i64, u32), k: i64, t: i64| {
        let at_or_before = if backward == exact { e.1 <= t } else { e.1 < t };
        e.0 < k || (e.0 == k && at_or_before)
    };
    // Sort the left side on the same `(by, on)` order and merge, rather than binary-searching
    // the right side once per left row. A left side handed in `on` order -- the usual input --
    // visits a different `by` group on almost every row, so a per-row search is a run of cache
    // misses through a right side far larger than cache: 70% of a 6M x 5.5M join's CPU went to
    // it. The counting sort in `sorted_tuples` puts the left side in group order in two linear
    // passes (no comparisons when `on` is already ordered), and the merge then walks both sides
    // forward. Each chunk of the sorted left finds its starting right position once.
    let n_left = u32::try_from(left_on.len()).ok()?;
    // `NONE` marks an unmatched left row; a real right index never reaches it.
    const NONE: u32 = u32::MAX;
    if u32::try_from(right_on.len()).ok()? == NONE {
        return None;
    }
    let probes = sorted_tuples(left_on, &lon, lby.as_ref());
    let chunk = probes
        .len()
        .div_ceil(rayon::current_num_threads().max(1) * 4)
        .max(4096);
    // Each left row appears in exactly one probe, so every slot is written at most once; the
    // atomics only make the scattered parallel writes expressible without `unsafe`.
    let slots: Vec<AtomicU32> = (0..n_left)
        .into_par_iter()
        .map(|_| AtomicU32::new(NONE))
        .collect();
    probes.par_chunks(chunk).for_each(|part| {
        let (k0, t0, _) = part[0];
        let mut p = entries.partition_point(|e| passed(e, k0, t0));
        for &(k, t, i) in part {
            while p < entries.len() && passed(&entries[p], k, t) {
                p += 1;
            }
            let at = if backward { p.checked_sub(1) } else { Some(p) };
            if let Some(e) = at.and_then(|q| entries.get(q)).filter(|e| e.0 == k) {
                slots[i as usize].store(e.2, Ordering::Relaxed);
            }
        }
    });
    let mut right_idx: Vec<u32> = slots.into_par_iter().map(AtomicU32::into_inner).collect();
    let valid = BooleanBuffer::collect_bool(right_idx.len(), |i| right_idx[i] != NONE);
    let right = if valid.count_set_bits() < right_idx.len() {
        // A null slot's value is never read, but zero keeps it a valid index regardless.
        right_idx
            .par_iter_mut()
            .filter(|j| **j == NONE)
            .for_each(|j| *j = 0);
        UInt32Array::new(right_idx.into(), Some(NullBuffer::new(valid)))
    } else {
        UInt32Array::from(right_idx)
    };
    Some(JoinIndices {
        left: UInt32Array::from((0..n_left).collect::<Vec<_>>()),
        right,
    })
}

/// One side's `(by, on, row)` tuples in row order, skipping a row `mask` marks null.
///
/// Built in parallel from the decoded slices, and a side with no nulls is never asked about
/// validity: a per-row `is_valid` through `ArrayRef` was most of the 34 ms this took serially
/// over a million rows.
fn tuples(on_v: &[i64], by_v: Option<&[i64]>, mask: Option<&NullBuffer>) -> Vec<(i64, i64, u32)> {
    use rayon::prelude::*;

    let row = |i: usize| (by_v.map_or(0, |v| v[i]), on_v[i], i as u32);
    match mask {
        None => (0..on_v.len()).into_par_iter().map(row).collect(),
        Some(m) => (0..on_v.len())
            .into_par_iter()
            .filter(|&i| m.is_valid(i))
            .map(row)
            .collect(),
    }
}

/// One side's `(by, on, row)` tuples, a row whose `on` or `by` is null left out, sorted on the
/// whole tuple -- without a comparison sort where the keys allow it.
///
/// Two sorts of a million 24-byte tuples were half of the join once the probe became a merge.
/// When the `by` keys span a range no wider than a few times the row count, a stable counting
/// sort ([`counting_sort`]) groups them in linear passes, and because it is stable each group
/// keeps row order: so when `on` is already non-decreasing -- the input an as-of join is
/// usually handed -- every group is already ordered on `(on, row)` and nothing is compared at
/// all. Otherwise only the groups are sorted, each small and independent. Wide or sparse keys
/// take the plain sort.
fn sorted_tuples(
    on: &ArrayRef,
    on_v: &[i64],
    by: Option<&(&ArrayRef, std::borrow::Cow<'_, [i64]>)>,
) -> Vec<(i64, i64, u32)> {
    use rayon::prelude::*;

    let by_nulls = by.and_then(|(a, _)| a.logical_nulls());
    let mask = NullBuffer::union(on.logical_nulls().as_ref(), by_nulls.as_ref());
    let mask = mask.as_ref();
    let ordered_within_groups = |rows: &[(i64, i64, u32)]| {
        rows.par_windows(2)
            .all(|w| w[0].0 != w[1].0 || w[0].1 <= w[1].1)
    };
    let Some(by_v) = by.map(|(_, v)| v.as_ref()) else {
        let mut rows = tuples(on_v, None, mask);
        if !ordered_within_groups(&rows) {
            rows.par_sort_unstable();
        }
        return rows;
    };
    let kept = |i: &usize| mask.is_none_or(|m| m.is_valid(*i));
    let Some((lo, hi)) = (0..by_v.len())
        .into_par_iter()
        .filter(kept)
        .map(|i| (by_v[i], by_v[i]))
        .reduce_with(|a, b| (a.0.min(b.0), a.1.max(b.1)))
    else {
        return Vec::new();
    };
    let span = hi.abs_diff(lo);
    if span > (by_v.len() as u64).saturating_mul(4).max(1 << 16) {
        let mut rows = tuples(on_v, Some(by_v), mask);
        rows.par_sort_unstable();
        return rows;
    }
    let mut rows = counting_sort(on_v, by_v, lo, span as usize + 1, mask);
    if !ordered_within_groups(&rows) {
        rows.par_chunk_by_mut(|a, b| a.0 == b.0)
            .for_each(|g| g.sort_unstable());
    }
    rows
}

/// The output buffer of [`counting_sort`], shared by its parallel scatter. A method rather than
/// a public field, so a closure captures the whole wrapper (which is `Send`/`Sync`) and not the
/// bare pointer inside it.
struct ScatterTo(*mut (i64, i64, u32));
// SAFETY: see `counting_sort` -- writes through the pointer never alias across threads.
unsafe impl Send for ScatterTo {}
unsafe impl Sync for ScatterTo {}
impl ScatterTo {
    fn at(&self, pos: usize) -> *mut (i64, i64, u32) {
        self.0.wrapping_add(pos)
    }
}

/// A stable parallel counting sort of the kept rows on `by - lo` (`width` buckets), emitting
/// `(by, on, row)` straight from the decoded columns.
///
/// The textbook parallel form: each contiguous row range counts its own rows per bucket, an
/// exclusive prefix sum over `(bucket, range)` reserves every range a disjoint slice of every
/// bucket, and each range then writes its rows into its slices in row order -- so every bucket
/// holds its rows in row order, exactly as a serial stable scatter would. That is two reads of
/// the key column and one write of each tuple. The version it replaced first materialized the
/// tuples, split them by key range into fresh vectors, and counted and scattered those again:
/// about 45 ms a side for six million rows on sixteen cores, a third of the whole join.
///
/// Per-range counts cost `ranges x width` counters, so the range count is capped to keep them
/// within about two per row; at the widest key span the sort allows that is a single range.
fn counting_sort(
    on_v: &[i64],
    by_v: &[i64],
    lo: i64,
    width: usize,
    mask: Option<&NullBuffer>,
) -> Vec<(i64, i64, u32)> {
    use rayon::prelude::*;

    let n = by_v.len();
    let kept = |i: usize| mask.is_none_or(|m| m.is_valid(i));
    let bucket = |i: usize| by_v[i].abs_diff(lo) as usize;
    let n_ranges = rayon::current_num_threads()
        .min((2 * n / width).max(1))
        .min(n.div_ceil(1 << 14))
        .max(1);
    let step = n.div_ceil(n_ranges).max(1);
    let ranges: Vec<std::ops::Range<usize>> =
        (0..n).step_by(step).map(|s| s..(s + step).min(n)).collect();
    let mut cursors: Vec<Vec<u32>> = ranges
        .par_iter()
        .map(|r| {
            let mut counts = vec![0u32; width];
            for i in r.clone().filter(|&i| kept(i)) {
                counts[bucket(i)] += 1;
            }
            counts
        })
        .collect();
    // Exclusive prefix sum, bucket-major and range-minor: range `c`'s rows of bucket `g` land
    // after every earlier range's, which is what keeps the scatter stable. Row indices fit
    // `u32` (the caller's contract), so the running total does too.
    let mut total = 0u32;
    for g in 0..width {
        for counts in &mut cursors {
            let c = counts[g];
            counts[g] = total;
            total += c;
        }
    }
    let total = total as usize;
    let mut out: Vec<(i64, i64, u32)> = Vec::with_capacity(total);
    let dst = ScatterTo(out.as_mut_ptr());
    ranges
        .into_par_iter()
        .zip(cursors.into_par_iter())
        .for_each(|(r, mut next)| {
            for i in r.filter(|&i| kept(i)) {
                let g = bucket(i);
                // SAFETY: `next[g]` walks the slice the prefix sum reserved for this range's
                // rows of bucket `g`, which no other range or bucket addresses and which lies
                // below `total`, the capacity of `out`. Every reserved slot is written exactly
                // once, because this pass visits exactly the rows the counting pass counted.
                unsafe { dst.at(next[g] as usize).write((by_v[i], on_v[i], i as u32)) };
                next[g] += 1;
            }
        });
    // SAFETY: the scatter initialized every one of the `total` slots.
    unsafe { out.set_len(total) };
    out
}

/// [`asof_join_indices`] over arrow row encodings: any key type, every direction, tolerances.
fn asof_join_indices_general(
    left_on: &ArrayRef,
    right_on: &ArrayRef,
    left_by: &[ArrayRef],
    right_by: &[ArrayRef],
    spec: AsofSpec,
) -> Result<JoinIndices, RuntimeError> {
    let AsofSpec {
        direction,
        tolerance,
        allow_exact_matches,
    } = spec;
    let n_left = left_on.len();
    let n_right = right_on.len();

    // Canonicalize signed zero / NaN on the `on` ordering key too. `-0.0` and `0.0` are the
    // *same value* (IEEE equality, and how DuckDB's ASOF inequality treats them), but arrow's
    // row encoding gives them distinct, totally-ordered bytes (`-0.0 < 0.0`). Without folding,
    // a left `on = -0.0` would find no right `on = 0.0` "≤" it (backward), and no right
    // `on = -0.0` "≥" a left `0.0` (forward) — silently missing an exact nearest match that
    // DuckDB emits. Likewise two distinct NaN bit patterns would fail to match, exactly the
    // bug the equi-join fixed by canonicalizing. Folding here (as `canon_f64` does everywhere)
    // is orthogonal to the ordering of *distinct* finite values — it only merges the values
    // that are already equal. An int `on` has no float column, so it is returned unchanged.
    let lon_canon = crate::keys::canonicalize_float_keys(std::slice::from_ref(left_on));
    let ron_canon = crate::keys::canonicalize_float_keys(std::slice::from_ref(right_on));
    let left_on: &ArrayRef = lon_canon.as_deref().map_or(left_on, |c| &c[0]);
    let right_on: &ArrayRef = ron_canon.as_deref().map_or(right_on, |c| &c[0]);

    // A distance is needed only to enforce a tolerance or to choose between the two
    // `Nearest` candidates; a plain backward/forward search reads only the ordering.
    let needs_distance = tolerance.is_some() || direction == AsofDirection::Nearest;
    let (left_num, right_num) = if needs_distance {
        let l = NumericKeys::read(left_on)?;
        let r = NumericKeys::read(right_on)?;
        match (l, r) {
            (Some(l), Some(r)) => (Some(l), Some(r)),
            _ => {
                return Err(RuntimeError::AsofKeyNotMeasurable {
                    dtype: left_on.data_type().to_string(),
                })
            }
        }
    } else {
        (None, None)
    };

    // One shared converter so left/right `on` encodings are mutually order-comparable.
    let on_conv = RowConverter::new(vec![SortField::new(right_on.data_type().clone())])?;
    let left_on_enc = on_conv.convert_columns(std::slice::from_ref(left_on))?;
    let right_on_enc = on_conv.convert_columns(std::slice::from_ref(right_on))?;

    // Canonicalize signed zero on the `by` *equality* keys so `-0.0`/`0.0` group together,
    // matching the hash/sort-merge equi-join paths.
    let lby_canon = crate::keys::canonicalize_float_keys(left_by);
    let rby_canon = crate::keys::canonicalize_float_keys(right_by);
    let left_by: &[ArrayRef] = lby_canon.as_deref().unwrap_or(left_by);
    let right_by: &[ArrayRef] = rby_canon.as_deref().unwrap_or(right_by);

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

    // A null in any `by` column makes the row match nothing — `by` is an *equality* key and
    // `NULL != NULL` (the same SQL rule the equi-join enforces via `null_mask`). Arrow's row
    // encoding gives a null a concrete byte string, so without this a null-`by` right row
    // would form a group that null-`by` left rows would then "match" — matching every left
    // null to a right null, which neither DuckDB nor the equi-join does. Empty `by` (no
    // grouping columns) has no null to mask.
    let left_by_null = left_by_enc.as_ref().map(|_| null_mask(left_by, n_left));
    let right_by_null = right_by_enc.as_ref().map(|_| null_mask(right_by, n_right));

    // Group right rows by `by` key (byte-encoded; empty key when there are no `by`
    // columns), each group sorted ascending by `on` for binary search.
    let mut groups: IndexMap<Vec<u8>, Vec<(OwnedRow, u32)>> = IndexMap::new();
    for j in 0..n_right {
        if right_on.is_null(j) || right_by_null.as_ref().is_some_and(|m| m[j]) {
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
        if left_on.is_null(i) || left_by_null.as_ref().is_some_and(|m| m[i]) {
            right_idx.push(None);
            continue;
        }
        let key = left_by_enc
            .as_ref()
            .map_or_else(Vec::new, |e| e.row(i).as_ref().to_vec());
        let target = left_on_enc.row(i);
        let matched = groups.get(&key).and_then(|g| {
            // The two candidates either side of the left key. `back` is the last row at or
            // before it and `fwd` the first at or after it, so an exact match is *both* —
            // which is why a tie under `Nearest` resolves to the same row either way.
            // `allow_exact_matches` moves the boundary: with it, an equal key is the last
            // row `back` may take and the first `fwd` may take; without it, both must step
            // past the whole run of equal keys.
            let back = {
                let pp = if allow_exact_matches {
                    g.partition_point(|(on, _)| on.row() <= target)
                } else {
                    g.partition_point(|(on, _)| on.row() < target)
                };
                (pp > 0).then(|| g[pp - 1].1)
            };
            let fwd = {
                let pp = if allow_exact_matches {
                    g.partition_point(|(on, _)| on.row() < target)
                } else {
                    g.partition_point(|(on, _)| on.row() <= target)
                };
                (pp < g.len()).then(|| g[pp].1)
            };
            // `dist` is only ever `None` for a key with no distance, which `needs_distance`
            // has already rejected — so an unwrap-shaped default here would be unreachable
            // rather than lenient. Treating it as "infinitely far" keeps that unreachable.
            let dist = |j: u32| -> f64 {
                match (&left_num, &right_num) {
                    (Some(l), Some(r)) => l.distance(i, r, j as usize).unwrap_or(f64::INFINITY),
                    _ => f64::INFINITY,
                }
            };
            let chosen = match direction {
                AsofDirection::Backward => back,
                AsofDirection::Forward => fwd,
                // Ties go backward, matching pandas' `direction="nearest"`.
                AsofDirection::Nearest => match (back, fwd) {
                    (Some(b), Some(f)) => Some(if dist(f) < dist(b) { f } else { b }),
                    (b, f) => b.or(f),
                },
            }?;
            match tolerance {
                Some(tol) if dist(chosen) > tol => None,
                _ => Some(chosen),
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
    use std::sync::Arc;

    use arrow::array::{Int64Array, StringArray};

    use super::*;

    fn i64s(v: Vec<Option<i64>>) -> ArrayRef {
        Arc::new(Int64Array::from(v))
    }
    fn strs(v: Vec<Option<&str>>) -> ArrayRef {
        Arc::new(StringArray::from(v))
    }

    /// A null `by` key matches nothing: `by` is an equality key and `NULL != NULL`. Before the
    /// fix, arrow's row encoder gave a null `by` a concrete byte string, so a null-`by` right
    /// row formed a group and every null-`by` left row "matched" it — disagreeing with DuckDB
    /// and with the equi-join's own `NULL != NULL` rule.
    #[test]
    fn null_by_key_matches_nothing() {
        // left: (sym, ts) — rows 1 and 2 have a null `sym`.
        let left_on = i64s(vec![Some(10), Some(20), Some(30), Some(10)]);
        let left_by = vec![strs(vec![Some("A"), None, None, Some("B")])];
        // right: a null-`sym` quote at ts=5 that must NOT be matched by the null-`sym` left rows.
        let right_on = i64s(vec![Some(5), Some(5), Some(25)]);
        let right_by = vec![strs(vec![None, Some("A"), None])];

        let idx = asof_join_indices(
            &left_on,
            &right_on,
            &left_by,
            &right_by,
            AsofSpec::default(),
        )
        .unwrap();
        let right: Vec<Option<u32>> = (0..idx.right.len())
            .map(|i| idx.right.is_valid(i).then(|| idx.right.value(i)))
            .collect();
        // row0 ("A", 10) -> right#1 ("A", 5); rows 1,2 (null sym) -> None; row3 ("B", ...) -> None.
        assert_eq!(right, vec![Some(1), None, None, None]);
    }

    fn f64s(v: Vec<Option<f64>>) -> ArrayRef {
        Arc::new(arrow::array::Float64Array::from(v))
    }
    fn asof_right(idx: &JoinIndices) -> Vec<Option<u32>> {
        (0..idx.right.len())
            .map(|i| idx.right.is_valid(i).then(|| idx.right.value(i)))
            .collect()
    }

    /// A float `on` key of `-0.0` is the *same value* as `0.0` (IEEE equality; how DuckDB's
    /// ASOF inequality treats it), so a nearest-match search must find it. Before the fix the
    /// `on` column was row-encoded without canonicalizing signed zero, so arrow's total order
    /// (`-0.0 < 0.0`) hid the match: `backward` from `-0.0` found no right `0.0 ≤ -0.0`, and
    /// `forward` from `0.0` found no right `-0.0 ≥ 0.0` — both returned NULL where DuckDB emits
    /// the exact match. Distinct NaN bit patterns had the same defect the equi-join already fixed.
    #[test]
    fn signed_zero_and_nan_on_key_match() {
        // backward: left -0.0 must match right 0.0 (equal → exact nearest).
        let idx = asof_join_indices(
            &f64s(vec![Some(-0.0)]),
            &f64s(vec![Some(0.0)]),
            &[],
            &[],
            AsofSpec::default(),
        )
        .unwrap();
        assert_eq!(
            asof_right(&idx),
            vec![Some(0)],
            "backward: -0.0 must match 0.0"
        );

        // forward: left 0.0 must match right -0.0.
        let idx = asof_join_indices(
            &f64s(vec![Some(0.0)]),
            &f64s(vec![Some(-0.0)]),
            &[],
            &[],
            AsofSpec {
                direction: AsofDirection::Forward,
                ..AsofSpec::default()
            },
        )
        .unwrap();
        assert_eq!(
            asof_right(&idx),
            vec![Some(0)],
            "forward: 0.0 must match -0.0"
        );

        // Two distinct NaN bit patterns are one canonical NaN → an exact match, not a miss.
        let nan2 = f64::from_bits(0x7ff8_0000_0000_0001);
        assert!(nan2.is_nan());
        let idx = asof_join_indices(
            &f64s(vec![Some(f64::NAN)]),
            &f64s(vec![Some(nan2)]),
            &[],
            &[],
            AsofSpec::default(),
        )
        .unwrap();
        assert_eq!(asof_right(&idx), vec![Some(0)], "NaN must match NaN");
    }

    /// Tiny deterministic xorshift RNG.
    struct Rng(u64);
    impl Rng {
        fn below(&mut self, n: u64) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x % n
        }
    }

    /// Independent brute-force ASOF reference. Mirrors the documented rule and tie-break:
    /// backward = the (max on ≤ target, then max original-row) right match within the `by`
    /// group; forward = the (min on ≥ target, then min original-row); nearest = the smallest
    /// |on − target|, preferring the backward candidate on a tie. `tolerance` drops any
    /// candidate further than that from the target. A null `on` or any null `by` (either
    /// side) matches nothing. Returns the chosen right row per left row.
    fn brute_asof(
        left_on: &[Option<i64>],
        right_on: &[Option<i64>],
        left_by: &[Vec<Option<i64>>],
        right_by: &[Vec<Option<i64>>],
        direction: AsofDirection,
        tolerance: Option<u64>,
        allow_exact: bool,
    ) -> Vec<Option<u32>> {
        // Any null in a `by` column means the row has no group at all, so the whole key
        // is `None` rather than a key with a hole in it.
        let by_of = |cols: &[Vec<Option<i64>>], row: usize| -> Option<Vec<i64>> {
            cols.iter().map(|c| c[row]).collect()
        };
        (0..left_on.len())
            .map(|i| {
                let lon = left_on[i]?;
                let lby = by_of(left_by, i)?;
                // Every right row that could match this left row at all.
                let candidates: Vec<(i64, u32)> = (0..right_on.len())
                    .filter_map(|j| {
                        let ron = right_on[j]?;
                        if by_of(right_by, j)? != lby {
                            return None;
                        }
                        if tolerance.is_some_and(|t| (ron - lon).unsigned_abs() > t) {
                            return None;
                        }
                        Some((ron, j as u32))
                    })
                    .collect();
                // The two sides, each under its own documented tie-break: backward takes
                // the largest `on` at or below the target and, among equals, the latest
                // row; forward takes the smallest at or above and, among equals, the
                // earliest. Building them separately keeps the reference a statement of
                // the rule rather than a second copy of the binary search.
                let backward = candidates
                    .iter()
                    .filter(|(on, _)| if allow_exact { *on <= lon } else { *on < lon })
                    .max_by_key(|(on, row)| (*on, *row))
                    .copied();
                let forward = candidates
                    .iter()
                    .filter(|(on, _)| if allow_exact { *on >= lon } else { *on > lon })
                    .min_by_key(|(on, row)| (*on, *row))
                    .copied();
                let chosen = match direction {
                    AsofDirection::Backward => backward,
                    AsofDirection::Forward => forward,
                    // Ties prefer the backward candidate (pandas' `direction="nearest"`).
                    AsofDirection::Nearest => match (backward, forward) {
                        (Some(b), Some(f)) => {
                            Some(if (f.0 - lon).unsigned_abs() < (b.0 - lon).unsigned_abs() {
                                f
                            } else {
                                b
                            })
                        }
                        (b, f) => b.or(f),
                    },
                };
                chosen.map(|(_, j)| j)
            })
            .collect()
    }

    /// Fuzz ASOF against the brute-force reference across random inputs: all three
    /// directions crossed with eight (tolerance, allow-exact) combinations, 0/1/2 `by` columns, nulls in `on` and `by`,
    /// heavy ties on `on`, empty sides, and unsorted input (the impl must sort each group
    /// itself). The reference searches every right row and ranks the candidates, so it
    /// shares no code with the kernel's binary search.
    #[test]
    fn fuzz_asof_matches_brute_force() {
        let mut rng = Rng(0xA50F_1234);
        for _ in 0..2000 {
            let nl = rng.below(8) as usize;
            let nr = rng.below(8) as usize;
            let n_by = rng.below(3) as usize; // 0, 1, or 2 by columns
            let gen_on = |rng: &mut Rng, n: usize| -> Vec<Option<i64>> {
                (0..n)
                    .map(|_| (rng.below(6) != 0).then(|| rng.below(5) as i64 - 2))
                    .collect()
            };
            let gen_by = |rng: &mut Rng, n: usize| -> Vec<Vec<Option<i64>>> {
                (0..n_by)
                    .map(|_| {
                        (0..n)
                            .map(|_| (rng.below(6) != 0).then(|| rng.below(2) as i64))
                            .collect()
                    })
                    .collect()
            };
            let lon = gen_on(&mut rng, nl);
            let ron = gen_on(&mut rng, nr);
            let lby_v = gen_by(&mut rng, nl);
            let rby_v = gen_by(&mut rng, nr);

            let left_on = i64s(lon.clone());
            let right_on = i64s(ron.clone());
            let left_by: Vec<ArrayRef> = lby_v.iter().map(|c| i64s(c.clone())).collect();
            let right_by: Vec<ArrayRef> = rby_v.iter().map(|c| i64s(c.clone())).collect();

            for direction in [
                AsofDirection::Backward,
                AsofDirection::Forward,
                AsofDirection::Nearest,
            ] {
                // `on` values span [-2, 2], so these tolerances cover "nothing matches",
                // the interesting middle, and "the tolerance never binds".
                for (tol, exact) in [
                    (None, true),
                    (Some(0u64), true),
                    (Some(1), true),
                    (Some(2), true),
                    (Some(100), true),
                    (None, false),
                    (Some(1), false),
                    (Some(100), false),
                ] {
                    let idx = asof_join_indices(
                        &left_on,
                        &right_on,
                        &left_by,
                        &right_by,
                        AsofSpec {
                            direction,
                            tolerance: tol.map(|t| t as f64),
                            allow_exact_matches: exact,
                        },
                    )
                    .unwrap();
                    let got: Vec<Option<u32>> = (0..idx.right.len())
                        .map(|i| idx.right.is_valid(i).then(|| idx.right.value(i)))
                        .collect();
                    let want = brute_asof(&lon, &ron, &lby_v, &rby_v, direction, tol, exact);
                    // The chosen right row is unambiguous under our tie-break, so compare exactly.
                    assert_eq!(
                        got, want,
                        "asof mismatch direction={direction:?} tol={tol:?} exact={exact}\n lon={lon:?} ron={ron:?}\n lby={lby_v:?} rby={rby_v:?}"
                    );
                    // Left indices must always be the identity 0..nl (left-style).
                    let lidx: Vec<u32> = (0..idx.left.len()).map(|i| idx.left.value(i)).collect();
                    assert_eq!(lidx, (0..nl as u32).collect::<Vec<_>>());
                }
            }
        }
    }

    /// A multi-column `by` where only one component is null still masks the whole row (any-null
    /// == no match), matching the equi-join's `null_mask` semantics.
    #[test]
    fn partial_null_by_key_matches_nothing() {
        let left_on = i64s(vec![Some(10), Some(10)]);
        let left_by = vec![strs(vec![Some("A"), Some("A")]), i64s(vec![Some(1), None])];
        let right_on = i64s(vec![Some(5), Some(5)]);
        let right_by = vec![strs(vec![Some("A"), Some("A")]), i64s(vec![Some(1), None])];

        let idx = asof_join_indices(
            &left_on,
            &right_on,
            &left_by,
            &right_by,
            AsofSpec::default(),
        )
        .unwrap();
        let right: Vec<Option<u32>> = (0..idx.right.len())
            .map(|i| idx.right.is_valid(i).then(|| idx.right.value(i)))
            .collect();
        // row0 ("A",1) matches right#0; row1 ("A",null) matches nothing.
        assert_eq!(right, vec![Some(0), None]);
    }

    /// The fast path is the general path, row for row, on every shape it accepts: both
    /// directions, exact matches allowed or not, ties within a group, nulls in either key on
    /// either side, no `by` key, and `on`/`by` of each integer-like type it widens.
    #[test]
    fn the_integer_fast_path_matches_the_general_path() {
        use arrow::array::{Date32Array, Int32Array};

        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move |m: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % m
        };
        for round in 0..48 {
            let (n_l, n_r) = (1 + next(300) as usize, next(300) as usize);
            let mut col = |n: usize, range: u64, nulls: bool| -> Vec<Option<i64>> {
                (0..n)
                    .map(|_| (!nulls || next(9) != 0).then(|| next(range) as i64 - 20))
                    .collect()
            };
            let (mut l_on, mut r_on) = (col(n_l, 60, true), col(n_r, 60, true));
            let (mut l_by, mut r_by) = (col(n_l, 7, true), col(n_r, 7, true));
            // `key_order` has three routes -- dense keys over an already ordered `on`, dense
            // keys over an unordered one, and sparse keys -- so the rounds visit all three.
            if round % 5 < 2 {
                for v in [&mut l_on, &mut r_on] {
                    let mut present: Vec<i64> = v.iter().flatten().copied().collect();
                    present.sort_unstable();
                    let mut it = present.into_iter();
                    v.iter_mut().flatten().for_each(|x| *x = it.next().unwrap());
                }
            }
            if round % 6 == 5 {
                for v in [&mut l_by, &mut r_by] {
                    v.iter_mut().flatten().for_each(|x| *x *= 100_000_007);
                }
            }
            let as_type = |v: &[Option<i64>]| -> ArrayRef {
                match round % 3 {
                    0 => i64s(v.to_vec()),
                    1 => Arc::new(Int32Array::from(
                        v.iter().map(|x| x.map(|x| x as i32)).collect::<Vec<_>>(),
                    )),
                    _ => Arc::new(Date32Array::from(
                        v.iter().map(|x| x.map(|x| x as i32)).collect::<Vec<_>>(),
                    )),
                }
            };
            let (lo, ro) = (as_type(&l_on), as_type(&r_on));
            let grouped = round % 4 != 0;
            let (lb, rb) = if grouped {
                (vec![as_type(&l_by)], vec![as_type(&r_by)])
            } else {
                (vec![], vec![])
            };
            for direction in [AsofDirection::Backward, AsofDirection::Forward] {
                for allow_exact_matches in [true, false] {
                    let spec = AsofSpec {
                        direction,
                        tolerance: None,
                        allow_exact_matches,
                    };
                    let fast = asof_int_fast(&lo, &ro, &lb, &rb, spec).expect("fast path shape");
                    let general = asof_join_indices_general(&lo, &ro, &lb, &rb, spec).unwrap();
                    assert_eq!(fast.left, general.left, "round {round}");
                    assert_eq!(fast.right, general.right, "round {round} {direction:?}");
                }
            }
        }
    }

    /// Shapes the fast path does not take go to the general path untouched.
    #[test]
    fn sorted_tuples_matches_a_plain_sort_on_every_route() {
        let ordered: Vec<(i64, i64, u32)> = (0..500u32)
            .map(|i| (i64::from(i % 7) - 3, i64::from(i / 3), i))
            .collect();
        let unordered: Vec<_> = ordered
            .iter()
            .map(|&(k, t, i)| (k, (t * 7919) % 101, i))
            .collect();
        let sparse: Vec<_> = unordered
            .iter()
            .map(|&(k, t, i)| (k * (1 << 40), t, i))
            .collect();
        // Large enough to split into several row ranges, ordered and not; a single key; keys at
        // the top edge of `i64`.
        let big: Vec<_> = (0..200_000u32)
            .map(|i| (i64::from(i % 997) * 3, i64::from(i / 7), i))
            .collect();
        let big_unordered: Vec<_> = (0..200_000u32)
            .map(|i| (i64::from(i % 997) * 3, i64::from((i * 7919) % 1000), i))
            .collect();
        let one_key: Vec<_> = (0..50_000u32)
            .map(|i| (5, i64::from((i * 31) % 777), i))
            .collect();
        let edges: Vec<_> = (0..5_000u32)
            .map(|i| (i64::MAX - i64::from(i % 3), i64::from(i / 2), i))
            .collect();
        let cases = [
            ordered,
            unordered,
            sparse,
            big,
            big_unordered,
            one_key,
            edges,
            vec![],
        ];
        for (c, rows) in cases.into_iter().enumerate() {
            // Every third row of the larger cases gets a null `by` or `on`, which must drop it.
            let null_at = |i: usize, m: usize| c % 2 == 1 && i % 3 == m;
            let by: ArrayRef = i64s(
                rows.iter()
                    .enumerate()
                    .map(|(i, r)| (!null_at(i, 0)).then_some(r.0))
                    .collect(),
            );
            let on: ArrayRef = i64s(
                rows.iter()
                    .enumerate()
                    .map(|(i, r)| (!null_at(i, 1)).then_some(r.1))
                    .collect(),
            );
            let mut want: Vec<_> = rows
                .iter()
                .enumerate()
                .filter(|&(i, _)| !null_at(i, 0) && !null_at(i, 1))
                .map(|(_, r)| *r)
                .collect();
            want.sort_unstable();
            let (on_v, by_v) = (int_key(&on).unwrap(), int_key(&by).unwrap());
            let got = sorted_tuples(&on, &on_v, Some(&(&by, by_v)));
            assert_eq!(got, want, "case {c}");
            // Keyless: every row is one group.
            let mut want_keyless: Vec<_> = rows
                .iter()
                .enumerate()
                .filter(|&(i, _)| !null_at(i, 1))
                .map(|(_, r)| (0, r.1, r.2))
                .collect();
            want_keyless.sort_unstable();
            assert_eq!(sorted_tuples(&on, &on_v, None), want_keyless, "case {c}");
        }
    }

    #[test]
    fn the_fast_path_declines_what_it_does_not_cover() {
        let on = i64s(vec![Some(1), Some(2)]);
        let nearest = AsofSpec {
            direction: AsofDirection::Nearest,
            ..AsofSpec::default()
        };
        assert!(asof_int_fast(&on, &on, &[], &[], nearest).is_none());
        let tolerant = AsofSpec {
            tolerance: Some(1.0),
            ..AsofSpec::default()
        };
        assert!(asof_int_fast(&on, &on, &[], &[], tolerant).is_none());
        let by = vec![strs(vec![Some("a"), Some("b")])];
        assert!(asof_int_fast(&on, &on, &by, &by, AsofSpec::default()).is_none());
        let two = vec![on.clone(), on.clone()];
        assert!(asof_int_fast(&on, &on, &two, &two, AsofSpec::default()).is_none());
    }
}
