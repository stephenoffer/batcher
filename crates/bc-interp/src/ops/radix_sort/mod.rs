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

mod packed;

pub(crate) use packed::packed_multi_sort_indices;

/// Above this row count a **float** key stops permuting an index array and sorts `(key, row)`
/// pairs instead ([`pair_sort_indices`]). Sized to ~L2: a `u64` key array of 2^18 rows is 2 MiB.
///
/// Only floats have the problem this switches on. Every counting pass reads `keys[idx[i]]`, so
/// seven of a `u64`'s eight passes are a random gather over the key array — and a float is the
/// one key whose rank spans all 64 bits, so a float is the one key that runs all eight passes.
/// An integer or temporal column's live range is nearly always far narrower than its type
/// (`bits_for` and the constant-digit skip in [`lsd_radix`] cut a 200,000-value `Int64` to three
/// passes), which is why the switch is keyed on the *type* and not on the row count alone.
///
/// **This used to be a decline**, on the reasoning that a large float sort "arrives here only
/// per-range or per-run, both below this by construction". That is true on a 96-core node, where
/// the sample-sort cuts 6 M rows into 64 ranges of 94 K. It is false on a 16-core one, where the
/// same sort cuts into 15 ranges of **400 K** — over the bound, declining, and landing on
/// `lexsort_to_indices`, which `perf` then showed spending 59% of a single-key `ORDER BY <float>`
/// inside arrow's two-column comparison sort. Measured on 15 concurrent 400 K-row ranges of
/// `l_extendedprice`-shaped values: index-scatter radix 89.4 ms, the comparison sort it declined
/// to **67.0 ms**, and the pair sort **37.5 ms**. The bound was right about the radix and wrong
/// about what to do at it.
const FLOAT_SCATTER_MAX_ROWS: usize = 1 << 18;

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

    // Natural runs first: an input that is already partly ordered — an appended log, a union
    // of sorted files, a re-sort by a clustered key — merges its runs instead of radixing all
    // of them, and an input with no runs pays only the strided detection scan. See
    // `super::run_sort` for why that scan is `O(log n)` comparisons rather than `O(n)`.
    let descending = opts.descending;
    // A wide float key sorts pairs; everything else permutes the index array. Both produce the
    // same permutation — see [`pair_sort_indices`] — so this chooses only how the memory is
    // walked, and the run detection above sits in front of either.
    let scattered_float = matches!(values.data_type(), DataType::Float32 | DataType::Float64)
        && n > FLOAT_SCATTER_MAX_ROWS;
    let sort_part = |part: Vec<u32>| {
        if scattered_float {
            pair_sort_indices(part, &keys, descending)
        } else {
            lsd_radix(part, &keys, descending)
        }
    };
    let live_sorted = super::run_sort::run_aware_sort(&live_idx, &keys, descending, sort_part)
        .unwrap_or_else(|| sort_part(live_idx));

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
    // A float's rank spans the whole `u64`, so its counting passes scatter by a random key
    // byte and thrash once the key array leaves cache — a 2 M-row index-scatter radix measured
    // ~4x *slower* than the comparison sort. That is a fact about the **layout**, not about the
    // encoding, so it is answered above `FLOAT_SCATTER_MAX_ROWS` by sorting `(key, row)` pairs
    // rather than by declining the key. See that constant.
    macro_rules! float {
        ($arr:ty) => {{
            let a = values.as_any().downcast_ref::<$arr>()?;
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

/// Digits a `u64` key is sorted by, one byte each.
const RADIX_DIGITS: usize = 8;

/// Stable least-significant-byte-first radix sort of `idx` by `keys[idx]`. Up to eight
/// 256-bucket counting-sort passes (one per byte of the u64 key); a byte that is constant
/// across the input is skipped. `descending` inverts the key so an ascending radix yields
/// descending order.
///
/// **Every digit's histogram is built in one pass**, not one pass per digit. A radix pass
/// permutes `idx`; it never changes what is *in* it, so the multiset of keys each pass counts
/// is the same multiset, and the eight histograms are all determined before the first
/// scatter. Counting per pass re-read `keys[i]` eight times over — and after the first pass
/// `idx` is permuted, so seven of those eight reads were a random gather over the whole key
/// array. `perf` put 68.6% of a 6M-row two-key sort inside this function, against 9.3% in the
/// two `take` kernels that actually move the data.
///
/// Skipping a constant digit is the same decision it always was, and it now costs nothing to
/// discover: it used to be found *by* the counting pass it then skipped, so a key narrower
/// than its type — which is the common case, and the one `packed_multi_sort_indices` builds
/// deliberately — paid full price for the passes it did not need.
/// The permutation that sorts `idx` by `keys`, ordering a contiguous `(key, position)` array
/// instead of permuting `idx` across eight counting passes.
///
/// Same permutation as [`lsd_radix`], different memory access. The radix's passes read
/// `keys[idx[i]]` with `idx` already permuted, so all but the first are a random gather over the
/// whole key array; this reads every key exactly once, sequentially, and then sorts 12-byte
/// records that move as a block. Above [`FLOAT_SCATTER_MAX_ROWS`] that is worth roughly 2.4x on
/// a float, and below it the radix's `O(n·w)` still wins — hence a switch rather than a
/// replacement.
///
/// **Stable for any `idx`, not merely an ascending one.** The tie-break is the row's *position
/// in `idx`*, not the row number, so rows equal on the key keep whatever order `idx` gave them —
/// which is what `lsd_radix` guarantees and what the obvious `(key, row)` spelling would only
/// give when `idx` happens to be ascending. It is ascending at both of today's call sites, and
/// paying one indexed read per row to not depend on that is worth it: the alternative is a
/// silent, input-shaped tie-order divergence between this and the sequential oracle, which is
/// exactly the class of bug the trailing row-index tie-break in `sort_indices_of` exists to
/// prevent.
fn pair_sort_indices(idx: Vec<u32>, keys: &[u64], descending: bool) -> Vec<u32> {
    let mut pairs: Vec<(u64, u32)> = idx
        .iter()
        .enumerate()
        .map(|(pos, &row)| {
            let k = keys[row as usize];
            (if descending { !k } else { k }, pos as u32)
        })
        .collect();
    pairs.sort_unstable();
    pairs
        .into_iter()
        .map(|(_, pos)| idx[pos as usize])
        .collect()
}

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
    // 8 x 256 x 8 bytes = 16 KiB of counters, so every digit's bucket stays L1-resident while
    // the single pass over the keys streams past it.
    let mut counts = [[0usize; 256]; RADIX_DIGITS];
    for &i in &idx {
        let k = key(i);
        for (d, digit) in counts.iter_mut().enumerate() {
            digit[((k >> (d * 8)) & 0xff) as usize] += 1;
        }
    }

    let mut buf = vec![0u32; n];
    for (d, digit) in counts.iter().enumerate() {
        // All keys share this byte → this pass is the identity (stable), skip it.
        if digit.contains(&n) {
            continue;
        }
        let mut offset = [0usize; 256];
        let mut running = 0usize;
        for (o, c) in offset.iter_mut().zip(digit) {
            *o = running;
            running += c;
        }
        let shift = d * 8;
        for &i in &idx {
            let b = ((key(i) >> shift) & 0xff) as usize;
            buf[offset[b]] = i;
            offset[b] += 1;
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

    /// A float column **over** `FLOAT_SCATTER_MAX_ROWS` takes [`pair_sort_indices`] rather than
    /// the index-scatter radix, and must come back with the same relation arrow's comparison
    /// sort gives — the same claim `matches_arrow_float_with_nulls_signs_and_zeros` makes below
    /// the bound, which is the only size that path used to reach.
    ///
    /// Sorting the **values** rather than the permutation is deliberate. Ties are the whole
    /// question here (the pair sort's tie-break is a position, not a row number), and arrow's
    /// `sort_to_indices` is unstable, so two permutations that disagree on which of two equal
    /// rows comes first are both correct. Comparing the sorted values holds the part that is
    /// defined; `ties_keep_input_order_above_the_scatter_bound` holds the part that is ours.
    #[test]
    fn matches_arrow_float_above_the_scatter_bound() {
        let n = FLOAT_SCATTER_MAX_ROWS + 5_000;
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        // `l_extendedprice`-shaped: a wide-but-bounded positive range, plus the signed zeros,
        // the infinities and the nulls the encoding has to place, and enough duplicates that
        // ties are dense rather than incidental.
        let v: ArrayRef = Arc::new(Float64Array::from(
            (0..n)
                .map(|i| match i % 997 {
                    0 => None,
                    1 => Some(0.0),
                    2 => Some(-0.0),
                    3 => Some(f64::NEG_INFINITY),
                    4 => Some(f64::INFINITY),
                    _ => Some(((rnd() % 10_000) as f64) / 100.0 - 25.0),
                })
                .collect::<Vec<_>>(),
        ));
        assert!(
            v.len() > FLOAT_SCATTER_MAX_ROWS,
            "the pair path is what runs"
        );
        for descending in [false, true] {
            for nulls_first in [false, true] {
                let opts = SortOptions {
                    descending,
                    nulls_first,
                };
                let got = radix_sort_indices(&v, opts).expect("Float64 is radix-sortable");
                let want = sort_to_indices(&v, Some(opts), None).unwrap();
                let g = take(v.as_ref(), &got, None).unwrap();
                let w = take(v.as_ref(), &want, None).unwrap();
                assert_eq!(g.as_ref(), w.as_ref(), "desc={descending} nf={nulls_first}");
            }
        }
    }

    /// Rows equal on the key keep their input order on the pair path, ascending and descending
    /// alike — the stability the index-scatter radix gives for free and a comparison sort does
    /// not. Every key here is one of four values over `FLOAT_SCATTER_MAX_ROWS` rows, so every
    /// row is in a tie group of ~65,000 and an unstable sort could not pass by luck.
    #[test]
    fn ties_keep_input_order_above_the_scatter_bound() {
        let n = FLOAT_SCATTER_MAX_ROWS + 4;
        let v: ArrayRef = Arc::new(Float64Array::from(
            (0..n).map(|i| (i % 4) as f64).collect::<Vec<_>>(),
        ));
        for descending in [false, true] {
            let opts = SortOptions {
                descending,
                nulls_first: false,
            };
            let idx = radix_sort_indices(&v, opts).expect("Float64 is radix-sortable");
            let rows: Vec<u32> = idx.values().to_vec();
            for group in rows.chunks(n / 4) {
                assert!(
                    group.windows(2).all(|w| w[0] < w[1]),
                    "a tie group came back out of input order (desc={descending})"
                );
            }
        }
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
    fn a_constant_middle_digit_is_skipped_without_reordering() {
        // Byte 1 is `0x5A` in every value while bytes 0 and 2 vary, so exactly one *interior*
        // pass is skipped and the passes either side of it still have to compose. Every other
        // case here varies its low bytes and pins its high ones, which a skip at the top of
        // the key cannot get wrong: the last pass placed is the most significant, so dropping
        // trailing identity passes is trivially safe and dropping an interior one is not.
        //
        // It is the case the single-pass histogram makes reachable at all. The counts are now
        // taken before any scatter and read back per digit, so a digit's prefix sums have to
        // be built from *that digit's* row of the table — a version that indexed the wrong row
        // would still produce a permutation, and a constant digit is where it would first
        // stop being arrow's.
        let v: ArrayRef = Arc::new(Int64Array::from(
            (0..600i64)
                .map(|i| Some(((i % 23) << 16) | (0x5A << 8) | (i % 251)))
                .chain([None, Some(0x5A00)])
                .collect::<Vec<_>>(),
        ));
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
