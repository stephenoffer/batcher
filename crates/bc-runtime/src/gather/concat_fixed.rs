//! The fixed-width arm of [`super::concat_columns`]: one output buffer, filled across cores.

use arrow::array::{Array, ArrayRef};
use rayon::prelude::*;

use crate::error::RuntimeError;

/// Rows past which a fixed-width `concat` copies on every core rather than one.
///
/// Below this the copy is a few hundred microseconds and arrow's sequential `concat` is the
/// cheaper call. Above it the copy is a serial step between a join's build-side read and its
/// probe: TPC-H q10 at sf100 spent ~150 ms here on one core of 64.
pub(super) const PAR_CONCAT_MIN_ROWS: usize = 1 << 20;

/// Bytes each parallel copy task moves at once.
const PAR_CONCAT_CHUNK_BYTES: usize = 1 << 20;

/// Bulk `concat` for a fixed-width primitive type: the value bytes copied into disjoint slices
/// of one output buffer across cores, then the validity bitmaps appended.
///
/// Byte-level, so every fixed-width type (integers, floats, dates, timestamps with their zone,
/// decimals with their precision) takes it unchanged: the output carries `arrays[0]`'s data
/// type, and every input was checked to share it. Element for element the result is arrow's
/// `concat`, which the tests hold it to.
pub(super) fn concat_fixed(arrays: &[&dyn Array], width: usize) -> Result<ArrayRef, RuntimeError> {
    let total: usize = arrays.iter().map(|a| a.len()).sum();
    // Zeroed by the allocator (fresh pages are zero), so this costs no pass of its own; the
    // pages fault in during the parallel copy below, on the cores doing it.
    let mut values = arrow::buffer::MutableBuffer::from_len_zeroed(total * width);
    let mut rest = values.as_slice_mut();
    let mut pieces = Vec::with_capacity(arrays.len());
    for a in arrays {
        let (dst, tail) = rest.split_at_mut(a.len() * width);
        pieces.push((dst, a.to_data()));
        rest = tail;
    }
    pieces.into_par_iter().for_each(|(dst, data)| {
        let start = data.offset() * width;
        let src = &data.buffers()[0].as_slice()[start..start + dst.len()];
        dst.par_chunks_mut(PAR_CONCAT_CHUNK_BYTES)
            .zip(src.par_chunks(PAR_CONCAT_CHUNK_BYTES))
            .for_each(|(d, s)| d.copy_from_slice(s));
    });
    let nulls = if arrays.iter().any(|a| a.null_count() > 0) {
        let mut bits = arrow::array::BooleanBufferBuilder::new(total);
        for a in arrays {
            match a.logical_nulls() {
                Some(n) => bits.append_buffer(n.inner()),
                None => bits.append_n(a.len(), true),
            }
        }
        Some(arrow::buffer::NullBuffer::new(bits.finish()))
    } else {
        None
    };
    let data = arrow::array::ArrayData::builder(arrays[0].data_type().clone())
        .len(total)
        .add_buffer(values.into())
        .nulls(nulls)
        .build()?;
    Ok(arrow::array::make_array(data))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Date32Array, Float64Array, Int64Array, TimestampMicrosecondArray};

    use super::*;

    /// Element for element, type for type, arrow's `concat` — over inputs with and without
    /// nulls, sliced (a non-zero offset), and of a type whose metadata (a zone) must survive.
    #[test]
    fn matches_arrow_concat() {
        let n = PAR_CONCAT_MIN_ROWS / 3 + 7;
        let ints: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from_iter(
                (0..n as i64).map(|i| (i % 5 != 0).then_some(i)),
            )),
            Arc::new(Int64Array::from_iter_values(0..n as i64).slice(3, n - 10)),
            Arc::new(
                Int64Array::from_iter((0..n as i64).map(|i| (i % 3 == 0).then_some(-i)))
                    .slice(1, n - 1),
            ),
            Arc::new(Int64Array::from_iter_values(0..n as i64)),
        ];
        let floats: Vec<ArrayRef> = (0..4)
            .map(|k| {
                Arc::new(Float64Array::from_iter_values(
                    (0..n).map(|i| (i * k) as f64 * 0.5),
                )) as ArrayRef
            })
            .collect();
        let dates: Vec<ArrayRef> = (0..4)
            .map(|k| {
                Arc::new(Date32Array::from_iter_values((0..n as i32).map(|i| i + k))) as ArrayRef
            })
            .collect();
        let stamps: Vec<ArrayRef> = (0..4)
            .map(|k| {
                Arc::new(
                    TimestampMicrosecondArray::from_iter_values((0..n as i64).map(|i| i * k))
                        .with_timezone("Asia/Kolkata"),
                ) as ArrayRef
            })
            .collect();
        for set in [ints, floats, dates, stamps] {
            let refs: Vec<&dyn Array> = set.iter().map(|a| a.as_ref()).collect();
            let width = refs[0].data_type().primitive_width().unwrap();
            let got = concat_fixed(&refs, width).unwrap();
            let want = arrow::compute::concat(&refs).unwrap();
            assert_eq!(got.data_type(), want.data_type());
            assert_eq!(got.to_data(), want.to_data(), "{}", want.data_type());
        }
    }
}
