//! LSD radix sort for fixed-width integer / temporal / float sort keys.
//!
//! A full sort (no `LIMIT`) on an integer, temporal, or float column is O(n·w) by radix
//! (w = key bytes) versus the comparison sort's O(n log n) — a real win on the wide
//! inputs the external (spilling) sort generates run-by-run, and on the per-range sorts
//! of the parallel sample-sort. This is a *drop-in* permutation builder: it returns the
//! same relation a stable sort would, identical to `arrow::compute::sort_to_indices`.
//! Floats use an order-preserving bit transform matching arrow's `total_cmp`; a column
//! with a `NaN` (no single numeric position), a string/boolean key, a multi-key sort, or
//! a top-N returns `None` and the caller falls back to the comparison sort.

use arrow::array::{
    Array, ArrayRef, Date32Array, Date64Array, Float32Array, Float64Array, Int16Array, Int32Array,
    Int64Array, Int8Array, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray, TimestampSecondArray, UInt16Array, UInt32Array, UInt64Array,
    UInt8Array,
};
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, TimeUnit};

/// Above this row count the float radix declines (its random-scatter key array no longer
/// fits cache and it loses to the comparison sort). Sized to ~L2: a `u64` key array of
/// 2^18 rows is 2 MiB. Large float sorts arrive here only per-range (parallel sample-sort)
/// or per-run (spill) — both below this — so a whole-array serial float sort never radixes.
const FLOAT_RADIX_MAX_ROWS: usize = 1 << 18;

/// Build the sort permutation by LSD radix, or `None` if the key type is unsupported.
///
/// Only called for a full sort (the caller gates on `limit.is_none()`). Nulls are
/// grouped first/last per `opts.nulls_first` in input order; non-null rows are sorted
/// by an order-preserving `u64` transform of the key (sign-flipped for signed types,
/// bit-inverted for descending). The sort is stable, so equal keys keep input order.
pub(crate) fn radix_sort_indices(values: &ArrayRef, opts: SortOptions) -> Option<UInt32Array> {
    let keys = ordered_keys(values)?;
    let n = values.len();

    // Split row indices into null and non-null (both in input order → stable).
    let nulls = values.nulls();
    let mut null_idx: Vec<u32> = Vec::new();
    let mut live_idx: Vec<u32> = Vec::with_capacity(n);
    for i in 0..n {
        if nulls.is_some_and(|nb| nb.is_null(i)) {
            null_idx.push(i as u32);
        } else {
            live_idx.push(i as u32);
        }
    }

    // Already in order — a constant key, a time-ordered scan, a re-sort by the key the data is
    // already clustered on — means the permutation is the identity, because a stable sort leaves
    // an ordered input alone. Checking costs one comparison per row when it holds and, since
    // `all` short-circuits, about two when it does not; the eight counting passes it replaces
    // cost far more than that even on a key whose bytes are constant enough to skip most of them.
    // Restricted to a null-free column so the identity claim covers the whole output rather than
    // the live rows alone.
    if nulls.is_none() && is_ordered(&keys, opts.descending) {
        return Some(UInt32Array::from(live_idx));
    }

    let live_sorted = lsd_radix(live_idx, &keys, opts.descending);

    let mut out: Vec<u32> = Vec::with_capacity(n);
    if opts.nulls_first {
        out.extend_from_slice(&null_idx);
        out.extend_from_slice(&live_sorted);
    } else {
        out.extend_from_slice(&live_sorted);
        out.extend_from_slice(&null_idx);
    }
    Some(UInt32Array::from(out))
}

/// Order-preserving `u64` key per row (ascending order of the original values). Null
/// slots get an arbitrary key (their indices are handled separately). `None` for any
/// type radix does not support, so the caller falls back to the comparison sort.
fn ordered_keys(values: &ArrayRef) -> Option<Vec<u64>> {
    // Signed ints map to order-preserving u64 by flipping the sign bit after widening
    // to i64 (widening preserves order); unsigned widen directly.
    macro_rules! signed {
        ($arr:ty) => {{
            let a = values.as_any().downcast_ref::<$arr>()?;
            (0..a.len())
                .map(|i| ((a.value(i) as i64) as u64) ^ (1u64 << 63))
                .collect()
        }};
    }
    macro_rules! unsigned {
        ($arr:ty) => {{
            let a = values.as_any().downcast_ref::<$arr>()?;
            (0..a.len()).map(|i| a.value(i) as u64).collect()
        }};
    }
    // IEEE-754 floats map to an order-preserving u64 matching arrow's `total_cmp`:
    // negatives bit-invert, non-negatives flip only the sign bit. This places `-0.0`
    // just below `+0.0` exactly as arrow's comparison sort does (so the value sequences
    // agree bit-for-bit). NaN has no single numeric position, so a column containing one
    // bails to the comparison sort (`None`) — keeping the radix path exactly arrow-equal.
    //
    // Float radix wins only on **cache-fitting** inputs: the LSD passes scatter by a
    // random key byte, so once the key array spills L2 it thrashes and loses badly to
    // the comparison sort (a 2M-row serial radix measured ~4× *slower*). It is reached
    // on cache-sized work — the parallel sample-sort's per-range sorts and the spill
    // runs — so above `FLOAT_RADIX_MAX_ROWS` it declines and the caller's comparison
    // sort (or, for a large input, the parallel sample-sort) takes over.
    macro_rules! float {
        ($arr:ty) => {{
            let a = values.as_any().downcast_ref::<$arr>()?;
            if a.len() > FLOAT_RADIX_MAX_ROWS {
                return None;
            }
            let nulls = values.nulls();
            let mut keys = Vec::with_capacity(a.len());
            for i in 0..a.len() {
                let v = a.value(i) as f64;
                if !nulls.is_some_and(|nb| nb.is_null(i)) && v.is_nan() {
                    return None;
                }
                let b = v.to_bits();
                keys.push(if b >> 63 == 1 { !b } else { b | (1u64 << 63) });
            }
            keys
        }};
    }
    let keys: Vec<u64> = match values.data_type() {
        DataType::Float32 => float!(Float32Array),
        DataType::Float64 => float!(Float64Array),
        DataType::Int8 => signed!(Int8Array),
        DataType::Int16 => signed!(Int16Array),
        DataType::Int32 => signed!(Int32Array),
        DataType::Int64 => signed!(Int64Array),
        DataType::UInt8 => unsigned!(UInt8Array),
        DataType::UInt16 => unsigned!(UInt16Array),
        DataType::UInt32 => unsigned!(UInt32Array),
        DataType::UInt64 => unsigned!(UInt64Array),
        // Temporal types are physically signed integers (days / millis / micros …).
        DataType::Date32 => signed!(Date32Array),
        DataType::Date64 => signed!(Date64Array),
        DataType::Timestamp(TimeUnit::Second, _) => signed!(TimestampSecondArray),
        DataType::Timestamp(TimeUnit::Millisecond, _) => signed!(TimestampMillisecondArray),
        DataType::Timestamp(TimeUnit::Microsecond, _) => signed!(TimestampMicrosecondArray),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => signed!(TimestampNanosecondArray),
        _ => return None,
    };
    Some(keys)
}

/// The `k` best **non-null** row indices of a fixed-width key, in sorted order, or `None` for a
/// type with no order-preserving `u64` encoding.
///
/// The same encoding [`ordered_keys`] builds for the radix, fed to a bounded heap instead of a
/// counting sort: ranking is what a `LIMIT` needs and ordering the other `n - k` rows is what it
/// does not. Reads the value buffer once, sequentially, and touches the heap only for a row that
/// beats the worst kept so far.
///
/// Unlike the radix this does **not** decline on a NaN or on a large float column. Both of those
/// limits are properties of the counting sort — an unrepresentable numeric position and a random
/// scatter that leaves cache — and neither applies to a sequential scan against a heap. See
/// [`float_rank`] for why a NaN needs no special case here.
pub(super) fn top_k_live(values: &ArrayRef, descending: bool, k: usize) -> Option<Vec<u32>> {
    let ranks = ranks(values, descending)?;
    Some(super::heap_select_k(values.len(), values.nulls(), k, |i| {
        ranks[i]
    }))
}

/// An order-preserving `u64` per row: ordering these integers orders the rows, exactly as a
/// stable sort under `descending` would. `None` for a type with no such encoding.
///
/// Null slots carry whatever their (unread) payload encodes to; every caller places nulls
/// itself, because null ordering is `nulls_first`'s business rather than the key's.
pub(super) fn ranks(values: &ArrayRef, descending: bool) -> Option<Vec<u64>> {
    let n = values.len();
    // One arm per concrete primitive: this path exists to read a typed values slice
    // sequentially, which a `dyn Array` accessor would give up.
    macro_rules! encode {
        ($arr:ty, $conv:expr) => {{
            let a = values.as_any().downcast_ref::<$arr>()?;
            let v = a.values();
            let conv = $conv;
            let mut out = Vec::with_capacity(n);
            out.extend((0..n).map(|i| {
                let r: u64 = conv(v[i]);
                if descending {
                    !r
                } else {
                    r
                }
            }));
            Some(out)
        }};
    }
    // Signed widen to `i64` then flip the sign bit; unsigned widen directly; floats take the
    // order-preserving bit transform. Identical rankings to [`ordered_keys`], by construction.
    macro_rules! signed {
        ($arr:ty, $t:ty) => {
            encode!($arr, |x: $t| ((x as i64) as u64) ^ (1u64 << 63))
        };
    }
    macro_rules! unsigned {
        ($arr:ty, $t:ty) => {
            encode!($arr, |x: $t| x as u64)
        };
    }
    match values.data_type() {
        DataType::Int8 => signed!(Int8Array, i8),
        DataType::Int16 => signed!(Int16Array, i16),
        DataType::Int32 => signed!(Int32Array, i32),
        DataType::Int64 => signed!(Int64Array, i64),
        DataType::UInt8 => unsigned!(UInt8Array, u8),
        DataType::UInt16 => unsigned!(UInt16Array, u16),
        DataType::UInt32 => unsigned!(UInt32Array, u32),
        DataType::UInt64 => unsigned!(UInt64Array, u64),
        DataType::Float32 => encode!(Float32Array, |x: f32| float_rank(x as f64)),
        DataType::Float64 => encode!(Float64Array, float_rank),
        DataType::Date32 => signed!(Date32Array, i32),
        DataType::Date64 => signed!(Date64Array, i64),
        DataType::Timestamp(TimeUnit::Second, _) => signed!(TimestampSecondArray, i64),
        DataType::Timestamp(TimeUnit::Millisecond, _) => signed!(TimestampMillisecondArray, i64),
        DataType::Timestamp(TimeUnit::Microsecond, _) => signed!(TimestampMicrosecondArray, i64),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => signed!(TimestampNanosecondArray, i64),
        _ => None,
    }
}

/// An order-preserving `u64` for a float: ordering these integers is exactly `f64::total_cmp`.
///
/// Negatives bit-invert, non-negatives flip only the sign bit. That is the standard IEEE-754
/// total-order transform, and it needs no NaN case because it *is* total: a negative NaN inverts
/// below `-∞` and a positive one lands above `+∞`, which is where `total_cmp` puts them and
/// therefore where arrow's comparison sort does. (The counting sort declines on a NaN instead,
/// but for a reason that belongs to the counting sort — see [`ordered_keys`].) Sort keys reaching
/// here have normally been through `bc_arrow::canon_float_array` already, which collapses every
/// NaN to the positive quiet one and `-0.0` to `0.0`; agreeing with `total_cmp` on the raw bits
/// means the ranking is right either way.
#[inline]
fn float_rank(v: f64) -> u64 {
    let b = v.to_bits();
    if b >> 63 == 1 {
        !b
    } else {
        b | (1u64 << 63)
    }
}

/// Columns a composite packed key will consider. Past this the per-column rank passes cost
/// more than the comparison sort they replace, and a key that wide has almost certainly
/// exhausted the bit budget anyway.
const PACKED_MAX_KEYS: usize = 8;

/// Rows below which the composite pack is not worth its passes; the comparison sort answers.
///
/// The parallel sample-sort and the external merge sort call the same entry point over small
/// slices, so this floor is hit often. It is safe at any value because both paths produce the
/// *same* permutation — a stable lexicographic one is unique — so the floor trades speed, never
/// agreement.
const PACKED_MIN_ROWS: usize = 64;

/// The permutation of a **multi-key** sort, built by packing every key into one `u64` sized from
/// the columns' measured value ranges and radix-sorting that, or `None` when the key does not
/// fit.
///
/// ## Why a composite key can be one integer
///
/// A multi-key `ORDER BY` has no fast path here: [`radix_sort_indices`] takes a single column,
/// so `ORDER BY o_orderdate, o_shippriority` falls to the row-encoded comparison sort, which
/// encodes every row into arrow's escaped row format and then pays `O(n log n)` memcmps over it.
/// The two columns hold about 2,400 and 5 distinct values — fifteen bits between them, against
/// the ninety-six their declared types claim and the twelve-plus bytes the row encoder writes.
///
/// DuckDB narrows exactly this way before it materializes a sort payload
/// (`src/optimizer/compressed_materialization/compress_order.cpp`, which rewrites a column to
/// `value - min` at the smallest width its statistics allow). This is the same idea taken one
/// step further: the ranges are **measured** on the rows in hand rather than read from a
/// catalog, so it needs no statistics, is exact on every input, and narrows an intermediate
/// that no catalog describes.
///
/// ## Why the permutation is identical to the comparison sort's
///
/// Column `j` contributes [`ranks`]'s order-preserving `u64` — the encoding the single-key radix
/// already sorts by, which folds `descending` in by inverting — offset to `0` at its measured
/// minimum. Subtracting a constant is monotone, so the field orders the column exactly as the
/// comparison sort does. Fields are laid out **most-significant first in key order** and each is
/// wide enough to hold its own column's range, so comparing two packed keys as integers compares
/// their fields left to right, stopping at the first difference: that is the definition of
/// lexicographic order.
///
/// Nulls are encoded *in the field* rather than partitioned out, because a multi-key sort's nulls
/// are per column and interleaved. A null takes the field's lowest value under `nulls_first` and
/// its highest otherwise, which is where the comparison sort puts it; the field is widened by one
/// value to make room, and only when the column actually has a null.
///
/// The radix is stable, so rows equal on every key keep their input order — the same tie-break
/// the fallback gets from its trailing row-index column. `a_packed_multi_sort_equals_the_row_encoded_one`
/// pins the agreement over integers, temporals, descending keys, nulls at both ends, ties and
/// declines.
///
/// Floats are excluded rather than declined by width: their ranks span the whole `u64`, so a
/// float key could never fit beside another one, and admitting them would put a second statement
/// of Batcher's NaN / `-0.0` ordering here — the thing [`ordered_keys`] refuses for the same
/// reason.
pub(crate) fn packed_multi_sort_indices(
    vals: &[ArrayRef],
    opts: &[SortOptions],
) -> Option<UInt32Array> {
    let n = vals.first()?.len();
    if vals.len() < 2 || vals.len() > PACKED_MAX_KEYS || n < PACKED_MIN_ROWS {
        return None;
    }
    // A prefix of the rows is enough to *reject* a key, and rejecting early is what keeps the
    // decline free. Widths only grow as more rows are seen, so a sample that already exceeds the
    // budget proves the whole column does — while a sample that fits proves nothing and the
    // exact scan below still runs. Without this a pair of full-width `Int64` keys scanned all
    // 8 M rows to learn what its first 4,096 already showed.
    if !prefix_could_fit(vals, opts, n) {
        return None;
    }

    // Measure every column's width *before* materializing any of them. Deciding on a
    // materialized rank array instead is what an earlier version did, and it made the decline
    // the expensive case: two full-width `Int64` keys over 8 M rows built 128 MiB of ranks and
    // then rejected the budget, measured at **1.41x slower** than simply not trying. A width is
    // two values, so it costs a min/max scan and no allocation.
    let mut widths: Vec<FieldWidth> = Vec::with_capacity(vals.len());
    let mut total_bits = 0u32;
    for (v, o) in vals.iter().zip(opts) {
        if v.len() != n || !is_packable_key(v.data_type()) {
            return None;
        }
        let w = FieldWidth::measure(v, *o)?;
        total_bits = total_bits.checked_add(w.bits)?;
        if total_bits > 64 {
            return None;
        }
        widths.push(w);
    }

    // Most-significant first: column 0 owns the top `bits[0]` of the used width, so an integer
    // comparison of two keys is a left-to-right comparison of their fields.
    let mut packed = vec![0u64; n];
    let mut shift = total_bits;
    for (w, (v, o)) in widths.iter().zip(vals.iter().zip(opts)) {
        shift -= w.bits;
        w.write(v, *o, shift, &mut packed)?;
    }

    let idx: Vec<u32> = (0..n as u32).collect();
    if is_ordered(&packed, false) {
        return Some(UInt32Array::from(idx));
    }
    Some(UInt32Array::from(lsd_radix(idx, &packed, false)))
}

/// Rows sampled to reject an over-wide key before the exact width scan reads the whole column.
const PACKED_PROBE_ROWS: usize = 4_096;

/// Whether the first [`PACKED_PROBE_ROWS`] rows leave any chance the whole key fits a `u64`.
///
/// One-sided on purpose: `false` is a proof (a range measured over a subset can only widen), and
/// `true` is only the absence of one. That asymmetry is what makes this safe to consult before
/// the exact measurement rather than instead of it.
fn prefix_could_fit(vals: &[ArrayRef], opts: &[SortOptions], n: usize) -> bool {
    if n <= PACKED_PROBE_ROWS {
        return true;
    }
    let mut bits = 0u32;
    for (v, o) in vals.iter().zip(opts) {
        let head = v.slice(0, PACKED_PROBE_ROWS);
        let Some(w) = FieldWidth::measure(&head, *o) else {
            return false;
        };
        bits += w.bits;
        if bits > 64 {
            return false;
        }
    }
    true
}

/// Key types the composite pack admits: the integers and the temporals, i.e. exactly the arms
/// [`ranks`] encodes without a float's total-order question.
fn is_packable_key(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Date32
            | DataType::Date64
            | DataType::Timestamp(_, _)
    )
}

/// How wide one column's field must be, and where its values and its nulls sit inside it.
///
/// Measured from two rank values — the column's smallest and largest — rather than from a
/// materialized rank array, so a key that turns out not to fit costs a scan and nothing else.
struct FieldWidth {
    /// The rank every non-null value is offset by, so the smallest becomes zero.
    low: u64,
    bits: u32,
    /// Added to every non-null field value, so a `nulls_first` column leaves `0` for its nulls.
    live_offset: u64,
    /// The field value a null takes, or `None` when the column has none.
    null_value: Option<u64>,
}

impl FieldWidth {
    /// Measure `v` under `o`, or `None` if its live range cannot be described in a `u64` field.
    ///
    /// [`ranks`] is monotone in the value, so the extreme *ranks* are the ranks of the extreme
    /// values and one min/max scan of the ranks answers the width. The scan runs over the ranks
    /// rather than the raw values because `descending` is folded into the rank, which keeps this
    /// one statement of the ordering instead of two.
    fn measure(v: &ArrayRef, o: SortOptions) -> Option<Self> {
        let (low, high, has_null) = rank_extremes(v, o.descending)?;
        // An all-null column orders nothing: it contributes no bits, and every row's field is
        // the single value zero.
        let span = match (low, high) {
            (Some(lo), Some(hi)) => (hi - lo) as u128 + 1,
            _ => 0,
        };
        let (live_offset, null_value) = match (has_null, o.nulls_first) {
            (false, _) => (0, None),
            (true, true) => (1, Some(0)),
            (true, false) => (0, Some(span as u64)),
        };
        Some(FieldWidth {
            low: low.unwrap_or(0),
            bits: bits_for(span + u128::from(has_null)),
            live_offset,
            null_value,
        })
    }

    /// Or this field's value for every row into `packed` at `shift`.
    fn write(&self, v: &ArrayRef, o: SortOptions, shift: u32, packed: &mut [u64]) -> Option<()> {
        if self.bits == 0 {
            return Some(());
        }
        let r = ranks(v, o.descending)?;
        match (v.nulls(), self.null_value) {
            (Some(nb), Some(null_value)) => {
                for (i, out) in packed.iter_mut().enumerate() {
                    let f = if nb.is_null(i) {
                        null_value
                    } else {
                        r[i] - self.low + self.live_offset
                    };
                    *out |= f << shift;
                }
            }
            // No null to place: every row takes the live encoding, which lets the loop stream
            // the rank slice with no per-row null check.
            _ => {
                for (out, &rank) in packed.iter_mut().zip(&r) {
                    *out |= (rank - self.low) << shift;
                }
            }
        }
        Some(())
    }
}

/// The smallest and largest rank over `v`'s non-null rows, and whether it has a null.
///
/// `(None, None, _)` means every row is null. **Allocates nothing**, which is the whole point:
/// this is what decides whether the packed key is affordable, so it must not build the thing it
/// is deciding about.
///
/// It gets that by not restating [`ranks`]. The rank of a value is monotone in the value (and
/// anti-monotone under `descending`, which is the same statement inverted), so the extreme
/// *ranks* are the ranks of the extreme *values*: find those two rows with a plain scan, then
/// ask [`ranks`] about a two-row slice. The ordering rule stays in one place and the scan knows
/// nothing about it.
fn rank_extremes(v: &ArrayRef, descending: bool) -> Option<(Option<u64>, Option<u64>, bool)> {
    let has_null = v.nulls().is_some_and(|nb| nb.null_count() > 0);
    let Some((lo_row, hi_row)) = value_extreme_rows(v) else {
        return Some((None, None, has_null));
    };
    let picks = UInt32Array::from(vec![lo_row as u32, hi_row as u32]);
    let two = arrow::compute::take(v.as_ref(), &picks, None).ok()?;
    let r = ranks(&two, descending)?;
    Some((Some(r[0].min(r[1])), Some(r[0].max(r[1])), has_null))
}

/// The rows holding `v`'s smallest and largest non-null values, or `None` when every row is null.
///
/// A plain native comparison per row, with no notion of sort direction or null placement — those
/// belong to [`ranks`], which is applied to the two rows this returns.
fn value_extreme_rows(v: &ArrayRef) -> Option<(usize, usize)> {
    macro_rules! extremes {
        ($arr:ty) => {{
            let a = v.as_any().downcast_ref::<$arr>()?;
            let vals = a.values();
            match v.nulls() {
                // The common case streams the value slice with no per-row null test.
                None => vals
                    .iter()
                    .enumerate()
                    .fold(None, |best, (i, x)| match best {
                        None => Some((i, i)),
                        Some((lo, hi)) => Some((
                            if *x < vals[lo] { i } else { lo },
                            if *x > vals[hi] { i } else { hi },
                        )),
                    }),
                Some(nb) => (0..vals.len())
                    .filter(|i| !nb.is_null(*i))
                    .fold(None, |best, i| match best {
                        None => Some((i, i)),
                        Some((lo, hi)) => Some((
                            if vals[i] < vals[lo] { i } else { lo },
                            if vals[i] > vals[hi] { i } else { hi },
                        )),
                    }),
            }
        }};
    }
    match v.data_type() {
        DataType::Int8 => extremes!(Int8Array),
        DataType::Int16 => extremes!(Int16Array),
        DataType::Int32 => extremes!(Int32Array),
        DataType::Int64 => extremes!(Int64Array),
        DataType::UInt8 => extremes!(UInt8Array),
        DataType::UInt16 => extremes!(UInt16Array),
        DataType::UInt32 => extremes!(UInt32Array),
        DataType::UInt64 => extremes!(UInt64Array),
        DataType::Date32 => extremes!(Date32Array),
        DataType::Date64 => extremes!(Date64Array),
        DataType::Timestamp(TimeUnit::Second, _) => extremes!(TimestampSecondArray),
        DataType::Timestamp(TimeUnit::Millisecond, _) => extremes!(TimestampMillisecondArray),
        DataType::Timestamp(TimeUnit::Microsecond, _) => extremes!(TimestampMicrosecondArray),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => extremes!(TimestampNanosecondArray),
        _ => None,
    }
}

/// The bit width that holds every value in `0 .. card`, i.e. `ceil(log2(card))`.
///
/// Saturates at 65 for a cardinality past `u64`, which the caller's budget then rejects — the
/// point being that it must not report a *small* width for a huge one.
fn bits_for(card: u128) -> u32 {
    if card <= 1 {
        0
    } else {
        (128 - (card - 1).leading_zeros()).min(65)
    }
}

/// Whether the order-preserving keys are already non-decreasing (non-increasing for
/// `descending`), i.e. the sort has nothing to do.
fn is_ordered(keys: &[u64], descending: bool) -> bool {
    keys.windows(2).all(|w| {
        if descending {
            w[0] >= w[1]
        } else {
            w[0] <= w[1]
        }
    })
}

/// Stable least-significant-byte-first radix sort of `idx` by `keys[idx]`. Eight
/// 256-bucket counting-sort passes (one per byte of the u64 key); a pass whose byte
/// is constant across the input is skipped. `descending` inverts the key so an
/// ascending radix yields descending order.
fn lsd_radix(mut idx: Vec<u32>, keys: &[u64], descending: bool) -> Vec<u32> {
    let n = idx.len();
    if n <= 1 {
        return idx;
    }
    let key = |i: u32| {
        let k = keys[i as usize];
        if descending {
            !k
        } else {
            k
        }
    };
    let mut buf = vec![0u32; n];
    for shift in (0..64).step_by(8) {
        let mut count = [0usize; 257];
        for &i in &idx {
            let b = ((key(i) >> shift) & 0xff) as usize;
            count[b + 1] += 1;
        }
        // All keys share this byte → this pass is the identity (stable), skip it.
        if count[1..].contains(&n) {
            continue;
        }
        for k in 0..256 {
            count[k + 1] += count[k];
        }
        for &i in &idx {
            let b = ((key(i) >> shift) & 0xff) as usize;
            buf[count[b]] = i;
            count[b] += 1;
        }
        std::mem::swap(&mut idx, &mut buf);
    }
    idx
}

#[cfg(test)]
mod ordered_shortcut_tests {
    use std::sync::Arc;

    use arrow::compute::{sort_to_indices, take};

    use super::*;

    /// An already-ordered key must radix to the identity, and that has to be checked against
    /// arrow's own sort rather than against `0..n` — the claim is that the permutation is
    /// unchanged, and only the comparison sort can say what the permutation should be.
    #[test]
    fn an_ordered_column_radixes_to_itself() {
        let ascending: ArrayRef = Arc::new(Int64Array::from((0..5_000i64).collect::<Vec<_>>()));
        let constant: ArrayRef = Arc::new(Int64Array::from(vec![7i64; 5_000]));
        let descending_vals: ArrayRef =
            Arc::new(Int64Array::from((0..5_000i64).rev().collect::<Vec<_>>()));
        let mut unordered: Vec<i64> = (0..5_000i64).collect();
        unordered.swap(0, 4_999);
        let unordered: ArrayRef = Arc::new(Int64Array::from(unordered));

        for values in [ascending, constant, descending_vals, unordered] {
            for descending in [false, true] {
                let opts = SortOptions {
                    descending,
                    nulls_first: false,
                };
                let got = radix_sort_indices(&values, opts).expect("Int64 is radix-sortable");
                let want = sort_to_indices(values.as_ref(), Some(opts), None).unwrap();
                let g = take(values.as_ref(), &got, None).unwrap();
                let w = take(values.as_ref(), &want, None).unwrap();
                assert_eq!(g.as_ref(), w.as_ref(), "descending={descending}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int32Array, Int64Array, UInt32Array as U32, UInt64Array};
    use arrow::compute::{sort_to_indices, take};

    use super::*;

    #[test]
    fn matches_arrow_float_with_nulls_signs_and_zeros() {
        // Finite floats spanning negatives, ±0.0, ±inf, ties, and nulls — the radix
        // float key must sort identically to arrow's comparison sort. (NaN bails to the
        // comparison sort and is covered by `nan_present_bails`.)
        let v: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(5.5),
            None,
            Some(-3.25),
            Some(5.5),
            Some(0.0),
            Some(-0.0),
            Some(f64::NEG_INFINITY),
            Some(f64::INFINITY),
            None,
            Some(-3.25),
            Some(1e308),
        ]));
        assert_radix_matches_arrow(v);
        let f32v: ArrayRef = Arc::new(Float32Array::from(vec![
            Some(2.0f32),
            Some(-1.0),
            None,
            Some(0.0),
            Some(-0.0),
            Some(f32::INFINITY),
        ]));
        assert_radix_matches_arrow(f32v);
    }

    #[test]
    fn nan_present_bails_to_comparison_sort() {
        // A column with a NaN is not radix-sortable (no single numeric position), so the
        // builder returns None and the caller uses arrow's comparison sort.
        let v: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(1.0),
            Some(f64::NAN),
            Some(2.0),
        ]));
        assert!(radix_sort_indices(&v, SortOptions::default()).is_none());
    }

    /// Radix and arrow's comparison sort must produce the **same sorted column** for
    /// every option combination (the relation is identical even if a tie permutation
    /// differs — both are valid stable sorts here). Checks the value sequence after
    /// gathering, across signs, nulls, ties, ascending/descending, nulls first/last.
    fn assert_radix_matches_arrow(values: ArrayRef) {
        for descending in [false, true] {
            for nulls_first in [false, true] {
                let opts = SortOptions {
                    descending,
                    nulls_first,
                };
                let radix = radix_sort_indices(&values, opts).expect("supported type");
                let arrow = sort_to_indices(&values, Some(opts), None).unwrap();
                let r = take(values.as_ref(), &radix, None).unwrap();
                let a = take(values.as_ref(), &arrow, None).unwrap();
                assert_eq!(
                    r.as_ref(),
                    a.as_ref(),
                    "desc={descending} nulls_first={nulls_first}"
                );
            }
        }
    }

    #[test]
    fn matches_arrow_signed_with_nulls_and_ties() {
        let v: ArrayRef = Arc::new(Int32Array::from(vec![
            Some(5),
            None,
            Some(-3),
            Some(5),
            Some(0),
            None,
            Some(i32::MIN),
            Some(i32::MAX),
            Some(-3),
        ]));
        assert_radix_matches_arrow(v);
    }

    #[test]
    fn matches_arrow_unsigned() {
        let v: ArrayRef = Arc::new(UInt64Array::from(vec![
            Some(10u64),
            Some(0),
            None,
            Some(u64::MAX),
            Some(10),
            Some(7),
        ]));
        assert_radix_matches_arrow(v);
    }

    #[test]
    fn matches_arrow_int64_full_range() {
        let v: ArrayRef = Arc::new(Int64Array::from(vec![
            Some(0i64),
            Some(-1),
            Some(1),
            Some(i64::MIN),
            Some(i64::MAX),
            None,
            Some(-1),
        ]));
        assert_radix_matches_arrow(v);
    }

    #[test]
    fn matches_arrow_all_nulls_and_empty() {
        assert_radix_matches_arrow(Arc::new(Int32Array::from(vec![None, None, None])) as ArrayRef);
        assert_radix_matches_arrow(
            Arc::new(Int32Array::from(Vec::<Option<i32>>::new())) as ArrayRef
        );
    }

    #[test]
    fn unsupported_type_returns_none() {
        // Strings/booleans have no fixed-width radix key, so the builder declines and the
        // caller uses arrow's comparison sort. (Floats are now supported — see the float
        // tests; a NaN-bearing float column declines via `nan_present_bails`.)
        let s: ArrayRef = Arc::new(arrow::array::StringArray::from(vec!["a", "b"]));
        assert!(radix_sort_indices(&s, SortOptions::default()).is_none());
        let b: ArrayRef = Arc::new(arrow::array::BooleanArray::from(vec![true, false]));
        assert!(radix_sort_indices(&b, SortOptions::default()).is_none());
    }

    #[test]
    fn stable_keeps_input_order_for_ties() {
        // Distinct payload via index lets us see the tie order: equal keys must keep
        // ascending input index (the stable property a stable arrow sort also gives).
        let v: ArrayRef = Arc::new(U32::from(vec![7u32, 7, 7, 7]));
        let idx = radix_sort_indices(&v, SortOptions::default()).unwrap();
        assert_eq!(idx.values(), &[0, 1, 2, 3]);
    }
}

#[cfg(test)]
mod packed_multi_key_tests {
    use super::*;
    use arrow::array::{ArrayRef, Int32Array, Int64Array, UInt32Array};
    use std::sync::Arc;

    /// The permutation the composite pack must reproduce: arrow's own lexicographic sort with
    /// the ascending row-index tie-break appended, which is exactly what `sort_indices_of`
    /// falls back to. Comparing against *that* rather than a hand-written expectation is the
    /// point — it is the path this replaces, and it decides null placement and direction with
    /// code this module shares nothing with.
    fn oracle(vals: &[ArrayRef], opts: &[SortOptions]) -> Vec<u32> {
        let n = vals[0].len();
        let mut columns: Vec<arrow::compute::SortColumn> = vals
            .iter()
            .zip(opts)
            .map(|(values, o)| arrow::compute::SortColumn {
                values: values.clone(),
                options: Some(*o),
            })
            .collect();
        columns.push(arrow::compute::SortColumn {
            values: Arc::new(UInt32Array::from_iter_values(0..n as u32)),
            options: Some(SortOptions {
                descending: false,
                nulls_first: false,
            }),
        });
        arrow::compute::lexsort_to_indices(&columns, None)
            .expect("arrow sorts these key types")
            .values()
            .to_vec()
    }

    fn check(vals: Vec<ArrayRef>, opts: Vec<SortOptions>) {
        let packed = packed_multi_sort_indices(&vals, &opts)
            .expect("this key was built to fit the packed budget");
        assert_eq!(packed.values().to_vec(), oracle(&vals, &opts));
    }

    fn asc() -> SortOptions {
        SortOptions {
            descending: false,
            nulls_first: true,
        }
    }

    /// Deterministic pseudo-random values, so a failure is reproducible.
    fn spread(n: usize, modulus: i64, seed: u64) -> Vec<i64> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((x >> 33) as i64).rem_euclid(modulus)
            })
            .collect()
    }

    // Above `PACKED_PROBE_ROWS`, so every case exercises the prefix probe as well as the
    // exact width scan.
    const N: usize = 6_000;

    #[test]
    fn a_packed_multi_sort_equals_the_row_encoded_one() {
        let a: ArrayRef = Arc::new(Int64Array::from(spread(N, 2_000, 7)));
        let b: ArrayRef = Arc::new(Int64Array::from(spread(N, 1_500, 11)));
        check(vec![a, b], vec![asc(), asc()]);
    }

    /// Ties on the leading key are what make stability observable: every distinct value of `a`
    /// is shared by ~1,000 rows here, and rows equal on both keys must keep input order.
    #[test]
    fn ties_resolve_to_input_order_exactly_as_the_oracle_does() {
        let a: ArrayRef = Arc::new(Int64Array::from(spread(N, 4, 3)));
        let b: ArrayRef = Arc::new(Int64Array::from(spread(N, 3, 5)));
        check(vec![a, b], vec![asc(), asc()]);
    }

    /// Every combination of direction, since `descending` is folded into the rank rather than
    /// applied to the packed key as a whole — mixing the two directions is where that could go
    /// wrong and a whole-key inversion would be caught.
    #[test]
    fn mixed_ascending_and_descending_keys_match() {
        for (d0, d1) in [(false, false), (false, true), (true, false), (true, true)] {
            let a: ArrayRef = Arc::new(Int64Array::from(spread(N, 50, 17)));
            let b: ArrayRef = Arc::new(Int64Array::from(spread(N, 900, 19)));
            check(
                vec![a, b],
                vec![
                    SortOptions {
                        descending: d0,
                        nulls_first: true,
                    },
                    SortOptions {
                        descending: d1,
                        nulls_first: true,
                    },
                ],
            );
        }
    }

    /// Nulls are encoded inside their field rather than partitioned out, at whichever end
    /// `nulls_first` names — per column, and independently of that column's direction.
    #[test]
    fn nulls_land_where_each_column_asks_for_them() {
        for nf0 in [true, false] {
            for nf1 in [true, false] {
                for desc in [false, true] {
                    let a: ArrayRef = Arc::new(Int64Array::from(
                        spread(N, 60, 23)
                            .into_iter()
                            .enumerate()
                            .map(|(i, v)| (i % 7 != 0).then_some(v))
                            .collect::<Vec<_>>(),
                    ));
                    let b: ArrayRef = Arc::new(Int32Array::from(
                        spread(N, 40, 29)
                            .into_iter()
                            .enumerate()
                            .map(|(i, v)| (i % 5 != 0).then_some(v as i32))
                            .collect::<Vec<_>>(),
                    ));
                    check(
                        vec![a, b],
                        vec![
                            SortOptions {
                                descending: desc,
                                nulls_first: nf0,
                            },
                            SortOptions {
                                descending: desc,
                                nulls_first: nf1,
                            },
                        ],
                    );
                }
            }
        }
    }

    /// Mixed widths and an unsigned column: each is offset to its own minimum, so a narrow
    /// column beside a wide one costs only its own bits.
    #[test]
    fn mixed_integer_widths_and_signedness_match() {
        let a: ArrayRef = Arc::new(Int32Array::from(
            spread(N, 300, 31)
                .into_iter()
                .map(|v| v as i32 - 150)
                .collect::<Vec<_>>(),
        ));
        let b: ArrayRef = Arc::new(UInt32Array::from(
            spread(N, 700, 37)
                .into_iter()
                .map(|v| v as u32 + 1_000_000)
                .collect::<Vec<_>>(),
        ));
        let c: ArrayRef = Arc::new(Int64Array::from(spread(N, 9, 41)));
        check(vec![a, b, c], vec![asc(), asc(), asc()]);
    }

    /// A temporal key is a signed integer physically, and is the commonest second sort key
    /// there is (`ORDER BY <date>, <id>`).
    #[test]
    fn a_date_key_beside_an_integer_matches() {
        let a: ArrayRef = Arc::new(arrow::array::Date32Array::from(
            spread(N, 2_400, 43)
                .into_iter()
                .map(|v| 19_000 + v as i32)
                .collect::<Vec<_>>(),
        ));
        let b: ArrayRef = Arc::new(Int64Array::from(spread(N, 5, 47)));
        check(vec![a, b], vec![asc(), asc()]);
    }

    /// A constant column takes no bits and must not shift the others out of place.
    #[test]
    fn a_constant_key_column_contributes_nothing() {
        let a: ArrayRef = Arc::new(Int64Array::from(vec![42i64; N]));
        let b: ArrayRef = Arc::new(Int64Array::from(spread(N, 800, 53)));
        check(vec![a, b], vec![asc(), asc()]);
    }

    /// An all-null column has no live range at all; it orders nothing and must not corrupt
    /// the key.
    #[test]
    fn an_all_null_key_column_orders_nothing() {
        let a: ArrayRef = Arc::new(Int64Array::from(vec![None::<i64>; N]));
        let b: ArrayRef = Arc::new(Int64Array::from(spread(N, 600, 59)));
        check(vec![a, b], vec![asc(), asc()]);
    }

    /// The declines, so a shape outside the budget reaches the comparison sort rather than a
    /// wrong answer: two full-width columns, a float key, a string key, and a short input.
    #[test]
    fn shapes_outside_the_budget_decline() {
        let wide0: ArrayRef = Arc::new(Int64Array::from(
            (0..N as i64)
                .map(|i| i.wrapping_mul(1_000_000_007))
                .collect::<Vec<_>>(),
        ));
        let wide1: ArrayRef = Arc::new(Int64Array::from(
            (0..N as i64)
                .map(|i| i.wrapping_mul(999_999_937))
                .collect::<Vec<_>>(),
        ));
        assert!(packed_multi_sort_indices(&[wide0, wide1], &[asc(), asc()]).is_none());

        let f: ArrayRef = Arc::new(Float64Array::from(
            spread(N, 100, 61)
                .into_iter()
                .map(|v| v as f64)
                .collect::<Vec<_>>(),
        ));
        let i: ArrayRef = Arc::new(Int64Array::from(spread(N, 100, 67)));
        assert!(packed_multi_sort_indices(&[f, i.clone()], &[asc(), asc()]).is_none());

        let s: ArrayRef = Arc::new(arrow::array::StringArray::from(
            (0..N).map(|k| format!("v{k:04}")).collect::<Vec<_>>(),
        ));
        assert!(packed_multi_sort_indices(&[s, i.clone()], &[asc(), asc()]).is_none());

        let short: ArrayRef = Arc::new(Int64Array::from(vec![3i64, 1, 2]));
        assert!(packed_multi_sort_indices(&[short.clone(), short], &[asc(), asc()]).is_none());

        // A single key is the radix path's business, not this one's.
        assert!(packed_multi_sort_indices(&[i], &[asc()]).is_none());
    }

    /// The prefix probe may only *reject*. A key whose first rows are narrow and whose tail is
    /// wide must therefore reach the exact scan and decline there — never pack on the strength
    /// of the sample.
    #[test]
    fn a_narrow_prefix_with_a_wide_tail_still_declines() {
        let a: ArrayRef = Arc::new(Int64Array::from(
            (0..N as i64)
                .map(|i| {
                    if i < 5_000 {
                        i % 8
                    } else {
                        i.wrapping_mul(1_000_000_007)
                    }
                })
                .collect::<Vec<_>>(),
        ));
        let b: ArrayRef = Arc::new(Int64Array::from(
            (0..N as i64)
                .map(|i| {
                    if i < 5_000 {
                        i % 8
                    } else {
                        i.wrapping_mul(999_999_937)
                    }
                })
                .collect::<Vec<_>>(),
        ));
        assert!(prefix_could_fit(
            &[a.clone(), b.clone()],
            &[asc(), asc()],
            N
        ));
        assert!(packed_multi_sort_indices(&[a, b], &[asc(), asc()]).is_none());
    }

    /// And it must reject the shape it exists for, without reading past the sample.
    #[test]
    fn a_wide_prefix_rejects_before_the_exact_scan() {
        let a: ArrayRef = Arc::new(Int64Array::from(
            (0..N as i64)
                .map(|i| i.wrapping_mul(1_000_000_007))
                .collect::<Vec<_>>(),
        ));
        let b: ArrayRef = Arc::new(Int64Array::from(
            (0..N as i64)
                .map(|i| i.wrapping_mul(999_999_937))
                .collect::<Vec<_>>(),
        ));
        assert!(!prefix_could_fit(&[a, b], &[asc(), asc()], N));
    }

    /// Already-ordered input short-circuits to the identity permutation, which must still be
    /// what the oracle produces (a stable sort leaves an ordered relation alone).
    #[test]
    fn an_already_ordered_key_returns_the_identity() {
        let a: ArrayRef = Arc::new(Int64Array::from(
            (0..N as i64).map(|i| i / 100).collect::<Vec<_>>(),
        ));
        let b: ArrayRef = Arc::new(Int64Array::from(
            (0..N as i64).map(|i| i % 100).collect::<Vec<_>>(),
        ));
        let packed = packed_multi_sort_indices(&[a.clone(), b.clone()], &[asc(), asc()]).unwrap();
        assert_eq!(packed.values().to_vec(), (0..N as u32).collect::<Vec<_>>());
        assert_eq!(packed.values().to_vec(), oracle(&[a, b], &[asc(), asc()]));
    }
}
