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
