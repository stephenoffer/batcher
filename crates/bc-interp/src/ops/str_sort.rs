//! Stable sort permutation for a `Utf8` / `LargeUtf8` sort key.
//!
//! Arrow's `sort_to_indices` is **not stable** for strings (nor, in general, for any
//! type): rows with equal keys come back in an arbitrary, input-size-dependent order.
//! The fixed-width keys dodge this because a full single-key sort takes the stable LSD
//! radix path (`super::radix_sort`); strings had no such path, so a string `ORDER BY`
//! was the one sort whose tie order was nondeterministic.
//!
//! That nondeterminism is what blocks the parallel sample-sort on a string key: the
//! sample-sort sorts each range independently, and a range's tie order could not be made
//! to agree with the whole-array sort's. This module supplies the missing guarantee —
//! ties resolve to input order — so the serial oracle and the per-range sorts produce the
//! identical relation, and `seq == par` holds for strings as it does for integers.
//!
//! Ties break on the original row index, which is a deterministic total order and is
//! exactly what a stable sort yields, so `sort_unstable_by` is safe (and avoids the
//! merge-sort allocation a `sort_by` would pay).

use arrow::array::{Array, ArrayRef, GenericStringArray, OffsetSizeTrait, UInt32Array};
use arrow::compute::SortOptions;
use arrow::datatypes::DataType;

/// The stable permutation that sorts a string `values` column under `opts`, or `None` if
/// `values` is not a string array (caller falls back to the comparison sort).
///
/// Nulls are grouped first/last per `opts.nulls_first`, in input order; non-null rows sort
/// byte-lexicographically (the ordering arrow itself uses for `Utf8`), descending inverting
/// only the key comparison, never the tie-break — so equal keys always keep input order.
pub(crate) fn stable_sort_indices_str(values: &ArrayRef, opts: SortOptions) -> Option<UInt32Array> {
    match values.data_type() {
        DataType::Utf8 => Some(sort_generic(
            values.as_any().downcast_ref::<GenericStringArray<i32>>()?,
            opts,
        )),
        DataType::LargeUtf8 => Some(sort_generic(
            values.as_any().downcast_ref::<GenericStringArray<i64>>()?,
            opts,
        )),
        _ => None,
    }
}

fn sort_generic<O: OffsetSizeTrait>(arr: &GenericStringArray<O>, opts: SortOptions) -> UInt32Array {
    let n = arr.len();
    let mut null_idx: Vec<u32> = Vec::new();
    let mut live_idx: Vec<u32> = Vec::with_capacity(n);
    for i in 0..n {
        if arr.is_null(i) {
            null_idx.push(i as u32);
        } else {
            live_idx.push(i as u32);
        }
    }

    live_idx.sort_unstable_by(|&a, &b| {
        let (x, y) = (arr.value(a as usize), arr.value(b as usize));
        let ord = if opts.descending { y.cmp(x) } else { x.cmp(y) };
        ord.then_with(|| a.cmp(&b))
    });

    let mut out: Vec<u32> = Vec::with_capacity(n);
    if opts.nulls_first {
        out.extend_from_slice(&null_idx);
        out.extend_from_slice(&live_idx);
    } else {
        out.extend_from_slice(&live_idx);
        out.extend_from_slice(&null_idx);
    }
    UInt32Array::from(out)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::StringArray;

    use super::*;

    fn idx(vals: Vec<Option<&str>>, descending: bool, nulls_first: bool) -> Vec<u32> {
        let a: ArrayRef = Arc::new(StringArray::from(vals));
        stable_sort_indices_str(
            &a,
            SortOptions {
                descending,
                nulls_first,
            },
        )
        .unwrap()
        .values()
        .to_vec()
    }

    #[test]
    fn ties_keep_input_order() {
        // Every key equal -> the permutation must be the identity.
        let v = vec![Some("a"); 64];
        assert_eq!(idx(v, false, false), (0..64u32).collect::<Vec<_>>());
    }

    #[test]
    fn descending_ties_still_keep_input_order() {
        let v = vec![Some("a"); 32];
        assert_eq!(idx(v, true, false), (0..32u32).collect::<Vec<_>>());
    }

    #[test]
    fn orders_bytewise_and_places_nulls() {
        let v = vec![Some("b"), None, Some("a"), Some("c")];
        assert_eq!(idx(v.clone(), false, false), vec![2, 0, 3, 1]);
        assert_eq!(idx(v.clone(), false, true), vec![1, 2, 0, 3]);
        assert_eq!(idx(v, true, false), vec![3, 0, 2, 1]);
    }

    #[test]
    fn interleaved_ties_are_stable() {
        let v: Vec<Option<&str>> = (0..300)
            .map(|i| Some(["aaa", "bbb", "ccc"][i % 3]))
            .collect();
        let got = idx(v, false, false);
        // First 100 are the "aaa" rows: original indices 0,3,6,... ascending.
        assert_eq!(
            got[..100],
            (0..100).map(|i| i as u32 * 3).collect::<Vec<_>>()[..]
        );
    }

    #[test]
    fn non_string_returns_none() {
        let a: ArrayRef = Arc::new(arrow::array::Int64Array::from(vec![1i64, 2]));
        assert!(stable_sort_indices_str(
            &a,
            SortOptions {
                descending: false,
                nulls_first: false
            }
        )
        .is_none());
    }
}
