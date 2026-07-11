//! Column gather (`take`) with a fast path for variable-length string columns.
//!
//! Gathering a permutation is the dominant cost of a sort and of a join's output, and for
//! `Utf8`/`LargeUtf8` arrow's `take` is far slower than the memory it moves: it drives
//! `MutableArrayData::extend` once per row, paying a call and bounds checks to copy a
//! handful of bytes. On a 5 M-row sort, adding one string column cost ~52 ms — an order of
//! magnitude more than the ~50 MB of characters involved.
//!
//! The fast path here does what the shape allows: one pass to sum the gathered lengths into
//! the offset buffer, one pass to `copy_from_slice` the bytes. Everything else — every
//! other data type, a nullable index array, an offset overflow — delegates to arrow's
//! `take`, so this is a pure performance short-circuit and never a second semantics.

use arrow::array::{
    Array, ArrayRef, GenericStringArray, LargeStringArray, OffsetSizeTrait, StringArray,
    UInt32Array,
};
use arrow::buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow::compute::take;
use arrow::datatypes::DataType;
use std::sync::Arc;

use crate::error::RuntimeError;

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
}
