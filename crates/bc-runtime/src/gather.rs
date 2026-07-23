//! Column gather (`take`) and multi-array `concat`, with fast paths for variable-length
//! string columns.
//!
//! Gathering a permutation is the dominant cost of a sort and of a join's output, and for
//! `Utf8`/`LargeUtf8` arrow's `take` is far slower than the memory it moves: it drives
//! `MutableArrayData::extend` once per row, paying a call and bounds checks to copy a
//! handful of bytes. On a 5 M-row sort, adding one string column cost ~52 ms — an order of
//! magnitude more than the ~50 MB of characters involved.
//!
//! `concat` has exactly the same shape and the same problem, and it sits on a hotter path:
//! a high-cardinality group-by concatenates every worker's partial keys, radix-partitions
//! them, and concatenates the partitions' outputs — three passes over the key column. On
//! ClickBench q33 (`GROUP BY URL`, 1 M rows, 275 k groups) those concats measured **60 ms of
//! a 70 ms combine**, moving 43 MB at 1.45 GB/s while the grouping itself took 7.6 ms.
//!
//! The fast paths do what the shape allows: sum the lengths into the offset buffer, then
//! `copy_from_slice` the bytes — and, for `concat`, copy each input's byte range into its own
//! disjoint slice of the output across cores, since the destination ranges are known up front.
//! Everything else — every other data type, a nullable index array, an offset overflow —
//! delegates to arrow, so these are pure performance short-circuits and never a second
//! semantics.

use arrow::array::{
    Array, ArrayRef, GenericStringArray, LargeStringArray, OffsetSizeTrait, StringArray,
    UInt32Array,
};
use arrow::buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow::compute::take;
use arrow::datatypes::DataType;
use rayon::prelude::*;
use std::sync::Arc;

use crate::error::RuntimeError;

/// Concatenate `arrays` (all of one type) into a single array, matching
/// `arrow::compute::concat` element-for-element.
///
/// Takes a bulk path for `Utf8`/`LargeUtf8`, where arrow's per-row `MutableArrayData::extend`
/// is the bottleneck; everything else delegates. Callers pass a non-empty slice.
pub fn concat_columns(arrays: &[&dyn Array]) -> Result<ArrayRef, RuntimeError> {
    match arrays {
        [] => Err(RuntimeError::from(arrow::error::ArrowError::ComputeError(
            "concat of no arrays".into(),
        ))),
        // One input is already the answer — an `Arc` bump rather than a copy of the data.
        [only] => Ok(arrow::array::make_array(only.to_data())),
        _ => {
            let dt = arrays[0].data_type();
            if arrays.iter().all(|a| a.data_type() == dt) {
                match dt {
                    DataType::Utf8 => {
                        if let Some(out) = concat_strings::<i32>(arrays) {
                            return Ok(Arc::new(out));
                        }
                    }
                    DataType::LargeUtf8 => {
                        if let Some(out) = concat_strings::<i64>(arrays) {
                            return Ok(Arc::new(out));
                        }
                    }
                    _ => {}
                }
            }
            Ok(arrow::compute::concat(arrays)?)
        }
    }
}

/// Bulk `concat` for string columns: one offset pass, then the value bytes copied per input
/// into disjoint output slices across cores.
///
/// `None` when any input is not a `GenericStringArray<O>` or the concatenated characters would
/// overflow the offset type, so the caller falls back to arrow's `concat` (which errors or
/// widens as it sees fit) rather than wrapping.
fn concat_strings<O: OffsetSizeTrait>(arrays: &[&dyn Array]) -> Option<GenericStringArray<O>> {
    let arrs: Vec<&GenericStringArray<O>> = arrays
        .iter()
        .map(|a| a.as_any().downcast_ref::<GenericStringArray<O>>())
        .collect::<Option<_>>()?;

    // Each input contributes the byte window its own offsets describe — which is not the whole
    // value buffer when the array is a slice of a larger one.
    let spans: Vec<(usize, usize)> = arrs
        .iter()
        .map(|a| {
            let o = a.value_offsets();
            (o[0].as_usize(), o[a.len()].as_usize())
        })
        .collect();
    let total_rows: usize = arrs.iter().map(|a| a.len()).sum();
    let total_bytes: usize = spans.iter().map(|(s, e)| e - s).sum();
    // Refuse rather than wrap when the result would not fit this offset width.
    O::from_usize(total_bytes)?;

    // Offsets: each input's are its own shifted by the bytes already written. A serial pass —
    // it is one add per row against a copy of the characters themselves.
    let mut offsets: Vec<O> = Vec::with_capacity(total_rows + 1);
    offsets.push(O::usize_as(0));
    let mut base = 0usize;
    for (a, (start, end)) in arrs.iter().zip(&spans) {
        let src = a.value_offsets();
        for i in 0..a.len() {
            offsets.push(O::usize_as(base + src[i + 1].as_usize() - start));
        }
        base += end - start;
    }

    // Values: the destination range of every input is known from the spans, so carve the output
    // into disjoint slices and let each input copy into its own across cores. `vec![0; n]` is an
    // `alloc_zeroed`, which for a buffer this size is zero pages rather than a write pass.
    let mut values = vec![0u8; total_bytes];
    let mut rest = values.as_mut_slice();
    let mut dsts: Vec<&mut [u8]> = Vec::with_capacity(arrs.len());
    for (start, end) in &spans {
        let (head, tail) = rest.split_at_mut(end - start);
        dsts.push(head);
        rest = tail;
    }
    dsts.into_par_iter()
        .zip(arrs.par_iter().zip(spans.par_iter()))
        .for_each(|(dst, (a, (start, end)))| {
            dst.copy_from_slice(&a.value_data()[*start..*end]);
        });

    // A concatenated row is null exactly when its source row is; a null-free input contributes
    // all-valid. Built only when some input actually has nulls, so the common case allocates
    // no validity buffer at all (matching arrow).
    let nulls = arrs.iter().any(|a| a.null_count() > 0).then(|| {
        NullBuffer::from_iter(
            arrs.iter()
                .flat_map(|a| (0..a.len()).map(|i| a.is_valid(i))),
        )
    });

    Some(GenericStringArray::<O>::new(
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        values.into(),
        nulls,
    ))
}

/// Gather `col`'s rows at `indices`, matching `arrow::compute::take` exactly.
pub fn take_column(col: &dyn Array, indices: &UInt32Array) -> Result<ArrayRef, RuntimeError> {
    // A null index means a null output row; the length/copy loops below assume a value.
    if indices.null_count() == 0 {
        match col.data_type() {
            DataType::Utf8 => {
                if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
                    if let Some(out) = take_strings::<i32>(a, indices) {
                        return Ok(Arc::new(out));
                    }
                }
            }
            DataType::LargeUtf8 => {
                if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
                    if let Some(out) = take_strings::<i64>(a, indices) {
                        return Ok(Arc::new(out));
                    }
                }
            }
            _ => {}
        }
    }
    Ok(take(col, indices, None)?)
}

/// `None` when the gathered characters would overflow the offset type, so the caller falls
/// back to arrow's `take` (which widens or errors as it sees fit) rather than wrapping.
fn take_strings<O: OffsetSizeTrait>(
    arr: &GenericStringArray<O>,
    indices: &UInt32Array,
) -> Option<GenericStringArray<O>> {
    let n = indices.len();
    let src_offsets = arr.value_offsets();
    let src_values = arr.value_data();
    let idx = indices.values();

    // One pass: each row's offset pair is a random read, so read it once and copy the bytes
    // while it is in cache, rather than walking the indices twice (once to sum lengths, once
    // to copy). The value buffer is pre-reserved at the source's average row width, so its
    // growth is amortized and usually never reallocates.
    let reserve = if arr.is_empty() {
        0
    } else {
        src_values.len().saturating_mul(n) / arr.len()
    };
    let mut offsets: Vec<O> = Vec::with_capacity(n + 1);
    let mut values: Vec<u8> = Vec::with_capacity(reserve);
    let mut total: usize = 0;
    offsets.push(O::usize_as(0));
    for &i in idx.iter() {
        let i = i as usize;
        let start = src_offsets[i].as_usize();
        let end = src_offsets[i + 1].as_usize();
        values.extend_from_slice(&src_values[start..end]);
        total += end - start;
        offsets.push(O::from_usize(total)?);
    }

    // A gathered row is null exactly when its source row is. Arrow's `take` leaves a null
    // row's slice empty, which the length pass above already does (start == end).
    let nulls = arr
        .nulls()
        .map(|src| NullBuffer::from_iter(idx.iter().map(|&i| src.is_valid(i as usize))));

    Some(GenericStringArray::<O>::new(
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        values.into(),
        nulls,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx(v: &[u32]) -> UInt32Array {
        UInt32Array::from(v.to_vec())
    }

    /// The fast path must equal `arrow::compute::take` element-for-element.
    fn assert_matches_arrow(col: &dyn Array, indices: &UInt32Array) {
        let want = take(col, indices, None).unwrap();
        let got = take_column(col, indices).unwrap();
        assert_eq!(want.as_ref(), got.as_ref());
    }

    #[test]
    fn gathers_utf8_like_arrow() {
        let a = StringArray::from(vec!["alpha", "b", "", "ccc", "dddd"]);
        assert_matches_arrow(&a, &idx(&[4, 0, 2, 2, 1, 3]));
    }

    #[test]
    fn gathers_utf8_with_nulls_like_arrow() {
        let a = StringArray::from(vec![Some("x"), None, Some("yy"), None, Some("")]);
        assert_matches_arrow(&a, &idx(&[1, 4, 0, 3, 2]));
    }

    #[test]
    fn gathers_large_utf8_like_arrow() {
        let a = LargeStringArray::from(vec![Some("aa"), None, Some("bbb")]);
        assert_matches_arrow(&a, &idx(&[2, 1, 0, 0]));
    }

    #[test]
    fn empty_index_list_like_arrow() {
        let a = StringArray::from(vec!["a", "b"]);
        assert_matches_arrow(&a, &idx(&[]));
    }

    /// A sliced source array must be gathered from its own offset window.
    #[test]
    fn gathers_sliced_source_like_arrow() {
        let a = StringArray::from(vec!["a", "bb", "ccc", "dddd"]);
        let sliced = a.slice(1, 3);
        assert_matches_arrow(&sliced, &idx(&[2, 0, 1]));
    }

    /// A null index falls back to arrow (which emits a null row).
    #[test]
    fn null_indices_fall_back_to_arrow() {
        let a = StringArray::from(vec!["a", "b"]);
        let i = UInt32Array::from(vec![Some(1), None, Some(0)]);
        assert_matches_arrow(&a, &i);
    }

    /// Non-string columns route to arrow unchanged.
    #[test]
    fn non_string_columns_use_arrow() {
        let a = arrow::array::Int64Array::from(vec![Some(5i64), None, Some(7)]);
        assert_matches_arrow(&a, &idx(&[2, 1, 0]));
    }

    /// The concat fast path must equal `arrow::compute::concat` element-for-element.
    fn assert_concat_matches_arrow(arrays: &[&dyn Array]) {
        let want = arrow::compute::concat(arrays).unwrap();
        let got = concat_columns(arrays).unwrap();
        assert_eq!(want.as_ref(), got.as_ref());
    }

    #[test]
    fn concatenates_utf8_like_arrow() {
        let a = StringArray::from(vec!["alpha", "b", ""]);
        let b = StringArray::from(vec!["ccc", "dddd"]);
        let c = StringArray::from(Vec::<&str>::new());
        assert_concat_matches_arrow(&[&a, &b, &c]);
        assert_concat_matches_arrow(&[&c, &a]);
    }

    /// Nulls in *some* inputs: the validity of a null-free input still has to be materialized
    /// as all-valid, or the rows after it shift against their bits.
    #[test]
    fn concatenates_utf8_with_nulls_like_arrow() {
        let a = StringArray::from(vec![Some("x"), None, Some("yy")]);
        let b = StringArray::from(vec!["p", "q"]);
        let c = StringArray::from(vec![None, Some("z")]);
        assert_concat_matches_arrow(&[&a, &b, &c]);
        assert_concat_matches_arrow(&[&b, &a]);
    }

    /// A sliced input contributes only its own byte window, not its parent's buffer — the
    /// shape every `combine` sees, since partials are slices of morsels.
    #[test]
    fn concatenates_sliced_inputs_like_arrow() {
        let a = StringArray::from(vec!["a", "bb", "ccc", "dddd", "e"]);
        let s1 = a.slice(1, 3);
        let s2 = a.slice(0, 2);
        assert_concat_matches_arrow(&[&s1, &s2, &a]);
    }

    #[test]
    fn concatenates_large_utf8_like_arrow() {
        let a = LargeStringArray::from(vec![Some("aa"), None]);
        let b = LargeStringArray::from(vec![Some("bbb")]);
        assert_concat_matches_arrow(&[&a, &b]);
    }

    /// One input is returned as-is, and non-string types delegate — both must still be
    /// element-identical to arrow.
    #[test]
    fn single_input_and_other_types_match_arrow() {
        let a = StringArray::from(vec!["only"]);
        assert_concat_matches_arrow(&[&a]);
        let x = arrow::array::Int64Array::from(vec![Some(1i64), None]);
        let y = arrow::array::Int64Array::from(vec![Some(3i64)]);
        assert_concat_matches_arrow(&[&x, &y]);
    }
}
