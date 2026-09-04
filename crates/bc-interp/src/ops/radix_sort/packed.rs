//! The **composite** packed key: several sort columns narrowed into one integer.
//!
//! A multi-key `ORDER BY` has no single-column fast path — [`super::radix_sort_indices`] takes
//! one array — so it would otherwise fall to a row-encoded or comparator sort. This module
//! measures each key's *live* range, packs the tuple into one word when the ranges fit, and
//! sorts that word. Two budgets: a `u64` the counting sort orders, and a `u128` that carries its
//! own row position and is sorted directly.
//!
//! The ordering itself is not restated here. Every rank comes from [`super::ranks`], and a
//! float's extremes are found by comparing [`super::float_rank`], so this module knows how to
//! *arrange* keys and nothing about how to order them.

use arrow::array::{
    Array, ArrayRef, Date32Array, Date64Array, Float32Array, Float64Array, Int16Array, Int32Array,
    Int64Array, Int8Array, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray, TimestampSecondArray, UInt16Array, UInt32Array, UInt64Array,
    UInt8Array,
};
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, TimeUnit};

use super::{float_rank, is_ordered, ranks};
/// Columns a composite packed key will consider. Past this the per-column rank passes cost
/// more than the comparison sort they replace, and a key that wide has almost certainly
/// exhausted the bit budget anyway.
const PACKED_MAX_KEYS: usize = 8;

/// Bits a packed key may use while its row position still fits beside it in a `u64`.
///
/// Both tiers carry the position *inside* the word, so the only question either asks is which
/// word is wide enough. The counting sort that used to own the `u64` tier is gone from this
/// module: with the position packed in, a `Vec<u64>` sorts directly, and pdqsort over 8-byte
/// words beats `lsd_radix` at every width measured. On 15 concurrent 400 K-row ranges:
///
/// | packed key | pack + counting sort | word sort carrying the position |
/// |---:|---:|---:|
/// | 26 bits | 40.8 ms | **20.1 ms** (`u64`) |
/// | 32 bits | 38.5 ms | **17.0 ms** (`u64`) |
/// | 40 bits | 49.8 ms | **28.3 ms** (`u128`) |
/// | 48 bits | 72.3 ms | **27.4 ms** (`u128`) |
/// | 64 bits | 93.8 ms | **30.5 ms** (`u128`) |
///
/// The gap widens with the key because the counting sort adds a pass per byte while the
/// comparison sort's cost is flat in the width. That is why a key between 33 and 64 bits — which
/// fits a `u64` but leaves no room for a position — goes to the `u128` tier rather than back to
/// the counting sort: the wider word is cheaper than the extra passes.
///
/// `lsd_radix` is untouched and still owns the **single-key** path, where there is no position to
/// carry and the key is the whole word.
const PACKED_U64_WITH_POSITION_BITS: u32 = 64 - ROW_POSITION_BITS;

/// Bits a packed key may use once it needs a `u128` — 128 less the 32 the row position takes.
///
/// The wide tier exists for the one shape the `u64` budget cannot reach and the row encoder is
/// worst at: **a float beside another key**. A float's rank is 56 bits on `l_extendedprice` and
/// 64 in the worst case, so it never fits a `u64` beside anything, and a two-key sort is below
/// `bc_arrow::row_sort::MIN_KEYS_FOR_ROW_ENCODING` as well — so `ORDER BY <int>, <float>` fell
/// all the way to arrow's `lexsort_to_indices` over three columns of comparator dispatch.
/// Measured on 15 concurrent 400 K-row ranges of `l_partkey DESC, l_extendedprice`: that
/// three-column comparison sort **98.3 ms**, the packed `u128` **41.3 ms**.
///
/// The row position rides in the low 32 bits rather than in a `(key, row)` tuple beside it. That
/// is what keeps the record 16 bytes instead of the 24 a `(u128, u32)` pads to, and it makes the
/// tie-break free: sorting the words ascending already resolves rows equal on every key to
/// ascending row number, which is the input order the `u64` tier's stable counting sort gives.
const PACKED_U128_BITS: u32 = 96;

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
        if total_bits > PACKED_U128_BITS {
            return None;
        }
        widths.push(w);
    }
    if total_bits > PACKED_U64_WITH_POSITION_BITS {
        return packed_wide_sort_indices(vals, opts, &widths, total_bits, n);
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
    // The packed key is one `u64` per row, so the composite sort gets natural-run detection on
    // exactly the same terms the single-key radix does — and it is the shape that wants it
    // most, since a multi-key `ORDER BY` whose leading key is the one the data is clustered on
    // is the commonest partly-ordered sort there is. Kept in front of the word sort rather than
    // dropped with the counting sort: pdqsort exploits a run far less than a merge does, and on
    // a 400 K-row range of ten sorted runs the word sort alone measured no faster than on random
    // input (21.1 ms against 19.8 ms), so the detection is still doing the work here.
    let sorted = crate::ops::run_sort::run_aware_sort(&idx, &packed, false, |part| {
        position_word_sort(part, &packed)
    })
    .unwrap_or_else(|| position_word_sort(idx, &packed));
    Some(UInt32Array::from(sorted))
}

/// The permutation that sorts `idx` by `keys`, by ordering one `u64` per row that holds the key
/// in its high bits and the row's **position in `idx`** in its low [`ROW_POSITION_BITS`].
///
/// Requires every key to fit [`PACKED_U64_WITH_POSITION_BITS`], which is what the caller's
/// dispatch guarantees.
///
/// The tie-break is the row's position in `idx`, not its row number, so ties keep whatever order
/// `idx` gave them — the input order the counting sort this replaces gave for free.
///
/// **At today's call sites the two spellings are equivalent, and no test distinguishes them.**
/// `idx` is `0..n` here, and `run_aware_sort` hands the fallback a contiguous *slice* of it
/// (`merge_runs`: `fallback(idx[r.start..r.end].to_vec())`), so a part is ascending too and a
/// row-number tie-break would order ties identically. That was checked rather than assumed:
/// rewriting this to pack the row number leaves all 17 composite tests green. The position form
/// is kept because it makes the function's contract independent of its caller, at one indexed
/// read per row — the same trade [`super::pair_sort_indices`] makes for the same reason.
fn position_word_sort(idx: Vec<u32>, keys: &[u64]) -> Vec<u32> {
    let mut words: Vec<u64> = idx
        .iter()
        .enumerate()
        .map(|(pos, &row)| (keys[row as usize] << ROW_POSITION_BITS) | pos as u64)
        .collect();
    words.sort_unstable();
    words
        .into_iter()
        .map(|w| idx[(w & u64::from(u32::MAX)) as usize])
        .collect()
}

/// The same composite key when it needs more than a `u64`: packed into a `u128` that carries its
/// own row position, and sorted as a plain `Vec<u128>`.
///
/// Called only from [`packed_multi_sort_indices`], with widths that function has already measured
/// and a `total_bits` it has already held to [`PACKED_U128_BITS`].
///
/// Same shape as the narrow tier — key in the high bits, row position in the low
/// [`ROW_POSITION_BITS`], sort the words — in twice the word, for a key that leaves no room for a
/// position in a `u64`.
///
/// One thing differs, and it is a limitation rather than a choice: `run_aware_sort` and
/// [`is_ordered`] both read `&[u64]`, so a `u128` key gets neither. An already-ordered input is
/// still cheap (pdqsort detects it), a partly-ordered one is not merged. Giving those two a
/// `u128` form is the obvious next step if a partly-ordered wide composite sort ever shows up as
/// a cost; nothing has measured one.
fn packed_wide_sort_indices(
    vals: &[ArrayRef],
    opts: &[SortOptions],
    widths: &[FieldWidth],
    total_bits: u32,
    n: usize,
) -> Option<UInt32Array> {
    if n > u32::MAX as usize {
        return None;
    }
    let mut packed: Vec<u128> = (0..n as u128).collect();
    let mut shift = total_bits + ROW_POSITION_BITS;
    for (w, (v, o)) in widths.iter().zip(vals.iter().zip(opts)) {
        shift -= w.bits;
        w.write(v, *o, shift, &mut packed)?;
    }
    packed.sort_unstable();
    Some(UInt32Array::from(
        packed
            .into_iter()
            .map(|p| (p & u128::from(u32::MAX)) as u32)
            .collect::<Vec<u32>>(),
    ))
}

/// Low bits of a wide packed key reserved for the row's own position, making the tie-break free.
const ROW_POSITION_BITS: u32 = 32;

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
        if bits > PACKED_U128_BITS {
            return false;
        }
    }
    true
}

/// Key types the composite pack admits: exactly the arms [`ranks`] encodes.
///
/// Floats used to be excluded, on the argument that "their ranks span the whole `u64`, so a float
/// key could never fit beside another one". The first half is true and the second stopped being
/// true when the budget grew to 128 bits: a float measures 56 bits on `l_extendedprice` and 64 in
/// the worst case, which leaves room for an integer beside it. The other half of that argument —
/// that admitting a float would restate Batcher's NaN and `-0.0` ordering here — is answered by
/// never restating it: this path ranks through [`ranks`], and [`value_extreme_rows`] finds a
/// float's extremes by comparing [`float_rank`] itself rather than by comparing `f64`s.
fn is_packable_key(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Float32
            | DataType::Float64
            | DataType::Int8
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
    ///
    /// Generic over the packed word so one statement of the encoding serves both budgets — the
    /// `u64` the counting sort orders and the `u128` that carries its own row position. A second
    /// copy for the wider word is exactly the duplication that lets two paths drift on a null's
    /// placement or a descending key's complement.
    fn write<W>(&self, v: &ArrayRef, o: SortOptions, shift: u32, packed: &mut [W]) -> Option<()>
    where
        W: Copy + From<u64> + core::ops::Shl<u32, Output = W> + core::ops::BitOrAssign,
    {
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
                    *out |= W::from(f) << shift;
                }
            }
            // No null to place: every row takes the live encoding, which lets the loop stream
            // the rank slice with no per-row null check.
            _ => {
                for (out, &rank) in packed.iter_mut().zip(&r) {
                    *out |= W::from(rank - self.low) << shift;
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
    // A float's extremes are found by comparing [`float_rank`], not by comparing `f64` — `<` on
    // an `f64` is not a total order (every comparison with a NaN is false, so a fold over `<`
    // silently keeps whatever it started with), and reaching for `total_cmp` instead would be a
    // second statement of the ordering sitting next to the one `ranks` uses. Comparing the rank
    // is the same function, so the two cannot disagree.
    macro_rules! float_extremes {
        ($arr:ty) => {{
            let a = v.as_any().downcast_ref::<$arr>()?;
            let vals = a.values();
            let rank_at = |i: usize| float_rank(vals[i] as f64);
            let live = |i: &usize| !v.nulls().is_some_and(|nb| nb.is_null(*i));
            (0..vals.len())
                .filter(live)
                .fold(None, |best, i| match best {
                    None => Some((i, i)),
                    Some((lo, hi)) => Some((
                        if rank_at(i) < rank_at(lo) { i } else { lo },
                        if rank_at(i) > rank_at(hi) { i } else { hi },
                    )),
                })
        }};
    }
    match v.data_type() {
        DataType::Float32 => float_extremes!(Float32Array),
        DataType::Float64 => float_extremes!(Float64Array),
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

    fn desc() -> SortOptions {
        SortOptions {
            descending: true,
            nulls_first: true,
        }
    }

    fn nulls_last() -> SortOptions {
        SortOptions {
            descending: false,
            nulls_first: false,
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

    /// Values spread across the **whole** `i64` range, so one column alone measures ~64 bits and
    /// two of them cannot share even the wide budget. `spread`'s `rem_euclid` caps a column at
    /// its modulus, which is what these tests need to be *inside* the budget; this is what they
    /// need to be outside it.
    fn full_width(n: usize, seed: u64) -> Vec<i64> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                x as i64
            })
            .collect()
    }

    /// A `l_extendedprice`-shaped float: positive, bounded, and two decimal places, whose *rank*
    /// range is ~56 bits — too wide for the `u64` budget beside anything, and the shape the
    /// `u128` tier exists for.
    fn prices(n: usize, seed: u64) -> Vec<f64> {
        spread(n, 10_404_850, seed)
            .into_iter()
            .map(|c| (c + 90_100) as f64 / 100.0)
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

    /// A float beside an integer, in **both** budgets, against the oracle.
    ///
    /// `small` is a float whose whole live range is a hundred integers, so its rank measures ~50
    /// bits and the pair still packs into one `u64` — the counting-sort tier. `wide` is
    /// `l_extendedprice`-shaped at ~56 bits, which no `u64` holds beside anything, so it can only
    /// have come back from the `u128` tier. Both must equal the row-encoded permutation exactly:
    /// a float's ordering is the one thing this path could restate and get subtly wrong, and
    /// `oracle` is arrow's own comparison sort over the same columns.
    #[test]
    fn a_float_beside_an_integer_matches_the_oracle() {
        let i: ArrayRef = Arc::new(Int64Array::from(spread(N, 200_000, 67)));
        let small: ArrayRef = Arc::new(Float64Array::from(
            spread(N, 100, 61)
                .into_iter()
                .map(|v| v as f64)
                .collect::<Vec<_>>(),
        ));
        let wide: ArrayRef = Arc::new(Float64Array::from(prices(N, 61)));
        check(vec![i.clone(), small.clone()], vec![asc(), asc()]);
        check(vec![i.clone(), wide.clone()], vec![asc(), asc()]);
        check(vec![wide.clone(), i.clone()], vec![asc(), asc()]);
        // `op-sort-multikey-wide` itself: a descending integer over an ascending float.
        check(vec![i, wide], vec![desc(), asc()]);
    }

    /// The `u128` tier under everything that makes a packed field hard: nulls at both ends, a
    /// descending key, and negative floats whose ranks bit-invert.
    #[test]
    fn a_key_too_wide_for_one_word_still_matches_the_oracle() {
        let signed: Vec<Option<f64>> = prices(N, 71)
            .into_iter()
            .enumerate()
            .map(|(i, v)| match i % 11 {
                0 => None,
                1 => Some(-v),
                2 => Some(0.0),
                _ => Some(v),
            })
            .collect();
        let f: ArrayRef = Arc::new(Float64Array::from(signed));
        let i: ArrayRef = Arc::new(Int64Array::from(
            spread(N, 300, 73)
                .into_iter()
                .enumerate()
                .map(|(r, v)| if r % 13 == 0 { None } else { Some(v) })
                .collect::<Vec<_>>(),
        ));
        for fo in [asc(), desc(), nulls_last()] {
            for io in [asc(), desc(), nulls_last()] {
                check(vec![i.clone(), f.clone()], vec![io, fo]);
                check(vec![f.clone(), i.clone()], vec![fo, io]);
            }
        }
    }

    /// A **partly ordered** composite key, which is the input `run_aware_sort` sits in front of
    /// and the one the word sort's position tie-break has to be right about: the runs it hands
    /// the sort are not `0..n`, so a `(key, row)` tie-break would order tied rows by row number
    /// while the oracle orders them by their place in the run.
    ///
    /// Ten ascending runs of the leading key with a random second key, so every run is long
    /// enough for the detection to fire and ties inside it are dense.
    #[test]
    fn a_partly_ordered_composite_key_matches_the_oracle() {
        let mut lead: Vec<i64> = Vec::with_capacity(N);
        for r in 0..10 {
            let mut run = spread(N / 10, 40, 79 + r as u64);
            run.sort_unstable();
            lead.extend(run);
        }
        lead.resize(N, 39);
        let a: ArrayRef = Arc::new(Int64Array::from(lead));
        let b: ArrayRef = Arc::new(Int64Array::from(spread(N, 6, 83)));
        for ao in [asc(), desc()] {
            for bo in [asc(), desc()] {
                check(vec![a.clone(), b.clone()], vec![ao, bo]);
            }
        }
    }

    /// The narrow tier ends where the row position stops fitting beside the key in a `u64`, and
    /// both sides of that line must agree with the oracle. 30 bits is inside it; 34 is not and
    /// falls to the `u128` tier, which is the transition most likely to be got wrong because
    /// nothing about the *result* changes across it.
    #[test]
    fn both_sides_of_the_word_boundary_match_the_oracle() {
        let narrow_a: ArrayRef = Arc::new(Int64Array::from(spread(N, 1 << 15, 89)));
        let narrow_b: ArrayRef = Arc::new(Int64Array::from(spread(N, 1 << 15, 97)));
        let wide_a: ArrayRef = Arc::new(Int64Array::from(spread(N, 1 << 17, 101)));
        let wide_b: ArrayRef = Arc::new(Int64Array::from(spread(N, 1 << 17, 103)));
        check(vec![narrow_a, narrow_b], vec![asc(), desc()]);
        check(vec![wide_a, wide_b], vec![asc(), desc()]);
    }

    /// Two keys of ~45 bits each — the shape `shapes_outside_the_budget_decline` used to hold as
    /// a decline, because 90 bits does not fit a `u64`. It fits 96, so it must now come back, and
    /// come back right.
    #[test]
    fn a_pair_between_the_two_budgets_is_admitted_not_declined() {
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
        assert!(
            packed_multi_sort_indices(&[a.clone(), b.clone()], &[asc(), asc()]).is_some(),
            "90 bits is inside the 96-bit budget"
        );
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
    /// wrong answer: two full-width columns, two floats, a string key, and a short input.
    ///
    /// The boundary these hold is [`PACKED_U128_BITS`], not the 64 bits it used to be. Two keys
    /// of ~45 bits each are now *inside* the budget and are covered by
    /// `a_key_too_wide_for_one_word_still_matches_the_oracle`; what is outside it is a pair that
    /// cannot share 96 bits however the widths fall.
    #[test]
    fn shapes_outside_the_budget_decline() {
        let wide0: ArrayRef = Arc::new(Int64Array::from(full_width(N, 1_000_000_007)));
        let wide1: ArrayRef = Arc::new(Int64Array::from(full_width(N, 999_999_937)));
        assert!(packed_multi_sort_indices(&[wide0, wide1], &[asc(), asc()]).is_none());

        // Two floats: ~56 bits each, so they fit neither budget however they are arranged.
        let f0: ArrayRef = Arc::new(Float64Array::from(prices(N, 61)));
        let f1: ArrayRef = Arc::new(Float64Array::from(prices(N, 63)));
        assert!(packed_multi_sort_indices(&[f0, f1], &[asc(), asc()]).is_none());
        let i: ArrayRef = Arc::new(Int64Array::from(spread(N, 100, 67)));

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
        let wide_a = full_width(N, 1_000_000_007);
        let wide_b = full_width(N, 999_999_937);
        let a: ArrayRef = Arc::new(Int64Array::from(
            (0..N as i64)
                .map(|i| if i < 5_000 { i % 8 } else { wide_a[i as usize] })
                .collect::<Vec<_>>(),
        ));
        let b: ArrayRef = Arc::new(Int64Array::from(
            (0..N as i64)
                .map(|i| if i < 5_000 { i % 8 } else { wide_b[i as usize] })
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
        let a: ArrayRef = Arc::new(Int64Array::from(full_width(N, 1_000_000_007)));
        let b: ArrayRef = Arc::new(Int64Array::from(full_width(N, 999_999_937)));
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
