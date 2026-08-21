//! Gathering a **fixed-width** column: one output slot per row, at a stride the type fixes.
//!
//! The byte-array paths in the parent module have to compute where each output row starts,
//! because a row's offset is the running sum of every preceding row's length. A fixed-width
//! column has no such dependency — row `k` lands at `k * width` — so the whole gather is a
//! scatter of independent copies, and the only thing standing between it and memory bandwidth
//! is the dependent cache miss on `src[idx[k]]`. Every path here answers that the same way: a
//! fixed-distance prefetch, and chunks that fill in parallel.
//!
//! [`take_chunked`] is the fallback for a type with no fill of its own: split the index list,
//! `take` each range with arrow, concatenate the pieces. It writes every value twice, which is
//! why it runs only where the fills decline.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, FixedSizeBinaryArray, UInt32Array};
use arrow::buffer::{NullBuffer, ScalarBuffer};
use arrow::compute::take;
use arrow::datatypes::DataType;
use rayon::prelude::*;

use super::{concat_columns, PARALLEL_TAKE_MIN_ROWS, PREFETCH_OFFSET_DISTANCE, TAKE_CHUNK_ROWS};
use crate::error::RuntimeError;

/// The last-resort parallel gather: split the index list, `take` each range with arrow, and
/// concatenate the pieces in order — or `None` for a type or size that must not take it.
///
/// This is what a type with no single-buffer fill above still gets, so no large gather is left
/// on one core. `take` is positional (`out[i] = source[indices[i]]`), so taking contiguous
/// index ranges and concatenating them in order reproduces the single-shot result exactly;
/// order is preserved by construction rather than by convention, which matters because a
/// `LIMIT` over a join must return the same rows on every executor.
///
/// It writes every value twice — once into a chunk, again into the concatenation — which is
/// why the fill paths above exist and why this runs only where they decline.
///
/// **Only for flat, self-describing types.** A **dictionary** is not one: `take` re-indexes
/// keys against the original dictionary, while `concat` over several dictionary chunks may
/// unify them into a *different* dictionary — logically the same values behind a different
/// encoding. Nested types (list/struct/union/map) and run-end encoding carry the same class of
/// representational question. None of that is worth reasoning about per type for a scheduling
/// choice, so they take arrow's single-shot gather: slower on one column, unambiguously
/// correct.
pub(super) fn take_chunked(
    col: &dyn Array,
    indices: &UInt32Array,
) -> Option<Result<ArrayRef, RuntimeError>> {
    use arrow::datatypes::DataType as D;
    let width = rayon::current_num_threads();
    if indices.len() < PARALLEL_TAKE_MIN_ROWS
        || width < 2
        || !matches!(
            col.data_type(),
            D::Boolean
                | D::Float16
                | D::Binary
                | D::LargeBinary
                | D::Duration(_)
                | D::Interval(_)
                | D::Decimal128(_, _)
                | D::Decimal256(_, _)
        )
    {
        return None;
    }
    let rows = indices.len();
    let chunk = rows.div_ceil(width);
    let pieces: Result<Vec<ArrayRef>, RuntimeError> = (0..rows)
        .step_by(chunk)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|start| {
            let slice = indices.slice(start, chunk.min(rows - start));
            Ok(take(col, &slice, None)?)
        })
        .collect();
    Some(pieces.and_then(|pieces| {
        let refs: Vec<&dyn Array> = pieces.iter().map(|p| p.as_ref()).collect();
        concat_columns(&refs)
    }))
}

/// A large fixed-width gather, spread across cores into one output buffer — or `None` for a
/// type or a size this does not claim, leaving the caller on arrow's `take`.
///
/// Arrow's `take` is single-threaded, and its callers here compensated by splitting the index
/// array into per-thread ranges, gathering each into its own array, and `concat`-ing the
/// pieces back. That reaches every core but writes every value **twice** — once into a chunk
/// and again into the concatenated result — and allocates a buffer per chunk on top. For a
/// fixed-width type none of that is necessary: a row's output position is its index position,
/// known before any value moves, so the output buffer can be allocated once and carved into
/// disjoint chunks that each thread fills in place. Same values, same order, one write each.
///
/// A variable-width column genuinely does need the measure-then-copy dance (a row's output
/// offset is the running sum of the ones before it), which is what [`take_bytes_parallel`]
/// does; this is the fixed-width case where that dependency does not exist.
pub(super) fn take_fixed_width_parallel(
    col: &dyn Array,
    indices: &UInt32Array,
) -> Option<ArrayRef> {
    use arrow::datatypes::{
        Date32Type, Date64Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type,
        Int8Type, Time32MillisecondType, Time32SecondType, Time64MicrosecondType,
        Time64NanosecondType, TimestampMicrosecondType, TimestampMillisecondType,
        TimestampNanosecondType, TimestampSecondType, UInt16Type, UInt32Type, UInt64Type,
        UInt8Type,
    };
    if indices.len() < PARALLEL_TAKE_MIN_ROWS || rayon::current_num_threads() < 2 {
        return None;
    }
    // `FixedSizeBinary` is fixed-width without being a `PrimitiveArray`, so the macro below
    // cannot reach it and arrow's single-shot `take` was the only thing left — on one core.
    // It is the type a fixed-layout record key and payload arrive as.
    //
    // Width zero is excluded because arrow's own `FixedSizeBinaryArray::try_new` computes
    // `values.len() / size` and so divides by zero on it — the fill could not return the array
    // it built even if it built it correctly. Such a column cannot be constructed through
    // arrow's public constructors at all, only through raw `ArrayData`; declining sends it to
    // `take`, which is the one path that does not go through `try_new`.
    if let DataType::FixedSizeBinary(width) = col.data_type() {
        if *width > 0 {
            return Some(Arc::new(take_fixed_size_binary_parallel(
                col.as_any().downcast_ref()?,
                indices,
            )) as ArrayRef);
        }
    }
    macro_rules! dispatch {
        ($($variant:pat => $ty:ty),* $(,)?) => {
            match col.data_type() {
                $($variant => Some(Arc::new(take_primitive_parallel::<$ty>(
                    col.as_any().downcast_ref()?,
                    indices,
                )) as ArrayRef),)*
                _ => None,
            }
        };
    }
    dispatch! {
        DataType::Int8 => Int8Type,
        DataType::Int16 => Int16Type,
        DataType::Int32 => Int32Type,
        DataType::Int64 => Int64Type,
        DataType::UInt8 => UInt8Type,
        DataType::UInt16 => UInt16Type,
        DataType::UInt32 => UInt32Type,
        DataType::UInt64 => UInt64Type,
        DataType::Float32 => Float32Type,
        DataType::Float64 => Float64Type,
        DataType::Date32 => Date32Type,
        DataType::Date64 => Date64Type,
        DataType::Time32(arrow::datatypes::TimeUnit::Second) => Time32SecondType,
        DataType::Time32(arrow::datatypes::TimeUnit::Millisecond) => Time32MillisecondType,
        DataType::Time64(arrow::datatypes::TimeUnit::Microsecond) => Time64MicrosecondType,
        DataType::Time64(arrow::datatypes::TimeUnit::Nanosecond) => Time64NanosecondType,
        DataType::Timestamp(arrow::datatypes::TimeUnit::Second, _) => TimestampSecondType,
        DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, _) => TimestampMillisecondType,
        DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, _) => TimestampMicrosecondType,
        DataType::Timestamp(arrow::datatypes::TimeUnit::Nanosecond, _) => TimestampNanosecondType,
    }
}

/// The fixed-width gather itself: one output buffer, filled in disjoint parallel chunks.
///
/// Every row's destination is known up front, so there is no prefix-sum phase and no scratch
/// array — `out[k] = src[idx[k]]`, with the same fixed-distance prefetch the string path uses,
/// because the source read is the identical dependent cache miss.
///
/// A gathered row is null exactly when its source row is, which is what arrow's `take` also
/// produces for a non-null index array. The timezone/precision-carrying types keep their
/// `DataType` through `with_data_type`, so a `Timestamp(_, Some(tz))` does not come back naive.
fn take_primitive_parallel<T: arrow::datatypes::ArrowPrimitiveType>(
    arr: &arrow::array::PrimitiveArray<T>,
    indices: &UInt32Array,
) -> arrow::array::PrimitiveArray<T>
where
    T::Native: Send + Sync,
{
    let src = arr.values();
    let idx = indices.values();
    // `vec![default; n]` is an `alloc_zeroed` for every native this dispatches (all of them
    // are zero-valued at `Default`), which at this size is zero pages rather than a write pass
    // — the same reasoning `take_bytes_parallel`'s value buffer records.
    let mut out: Vec<T::Native> = vec![T::Native::default(); idx.len()];
    out.par_chunks_mut(TAKE_CHUNK_ROWS)
        .zip(idx.par_chunks(TAKE_CHUNK_ROWS))
        .for_each(|(dst, rows)| {
            for (k, (slot, &i)) in dst.iter_mut().zip(rows).enumerate() {
                if let Some(&far) = rows.get(k + PREFETCH_OFFSET_DISTANCE) {
                    bc_arrow::prefetch_read(&src[far as usize]);
                }
                *slot = src[i as usize];
            }
        });
    let nulls = arr
        .nulls()
        .map(|src| NullBuffer::from_iter(idx.iter().map(|&i| src.is_valid(i as usize))));
    arrow::array::PrimitiveArray::<T>::new(ScalarBuffer::from(out), nulls)
        .with_data_type(arr.data_type().clone())
}

/// Gather a `FixedSizeBinary` column: `out[k] = src[idx[k]]`, each value a `width`-byte copy at
/// a `width`-byte stride.
///
/// The simplest gather there is, and until this existed it was the slowest one the engine had.
/// `FixedSizeBinary` is fixed-width but not a `PrimitiveArray`, so it matched neither the
/// primitive fill above nor [`take_chunked`]'s type list, and fell through to arrow's
/// single-shot `take` — one core, whatever the machine. Measured at 4 M rows of 90-byte values
/// that is 459 ms, against 79 ms for the *same bytes* held as a variable-length `Binary`
/// column — the type with less structure was six times faster. Filled here it is **48.6 ms**,
/// a 9.4x gain at that width and 22.2x at a 10-byte one (142.0 ms -> 6.4 ms), which makes it
/// the fastest of the byte types, as a type with no offsets to chase should be.
/// `report_the_byte_gather` in the parent module prints the table.
///
/// The caller guarantees a non-zero width; see the guard in [`take_fixed_width_parallel`].
///
/// The shape is [`take_primitive_parallel`]'s with a byte width in place of a native type: fill
/// disjoint chunks in parallel, and prefetch a fixed distance ahead because the source read is
/// the identical dependent cache miss. A gathered row is null exactly when its source row is,
/// which is what arrow's `take` also produces for a non-null index array.
fn take_fixed_size_binary_parallel(
    arr: &FixedSizeBinaryArray,
    indices: &UInt32Array,
) -> FixedSizeBinaryArray {
    let width = arr.value_length() as usize;
    let src = arr.value_data();
    // `value_data` is the whole underlying buffer, while `value(i)` reads at
    // `(offset + i) * width` — so a **sliced** array's first row does not start at byte zero.
    // Indexing from zero would silently gather the wrong bytes, which is a wrong answer rather
    // than an error, and every morselized path here hands out slices.
    let base = arr.value_offset(0) as usize;
    let idx = indices.values();
    // `alloc_zeroed`, which at this size is zero pages rather than a write pass — the same
    // reasoning the byte fills record.
    let mut out = vec![0u8; idx.len() * width];
    out.par_chunks_mut(TAKE_CHUNK_ROWS * width)
        .zip(idx.par_chunks(TAKE_CHUNK_ROWS))
        .for_each(|(dst, rows)| {
            for (k, &i) in rows.iter().enumerate() {
                if let Some(&far) = rows.get(k + PREFETCH_OFFSET_DISTANCE) {
                    if let Some(byte) = src.get(base + far as usize * width) {
                        bc_arrow::prefetch_read(byte);
                    }
                }
                let (from, to) = (base + i as usize * width, k * width);
                dst[to..to + width].copy_from_slice(&src[from..from + width]);
            }
        });
    let nulls = arr
        .nulls()
        .map(|src| NullBuffer::from_iter(idx.iter().map(|&i| src.is_valid(i as usize))));
    FixedSizeBinaryArray::new(width as i32, out.into(), nulls)
}
