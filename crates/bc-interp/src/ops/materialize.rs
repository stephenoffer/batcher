//! Concatenating morsels back into one batch — the first step of every pipeline
//! breaker (sort / join / asof / window). On a wide fact table at morsel granularity
//! this full copy dominates the operator, so it is parallelized: independent columns
//! fan across cores, and a null-free fixed-width primitive column copies via a
//! parallel memcpy (each morsel's values slice to its own disjoint output offset),
//! saturating memory bandwidth where arrow's serial per-chunk `concat` (~3 GB/s) does
//! not. The result is byte-identical to `concat_batches`.

use arrow::array::{Array, ArrayRef, RecordBatch};
use arrow::compute::concat_batches;
use rayon::prelude::*;
use std::sync::Arc;

use crate::error::InterpError;

/// Concatenate morsels into one batch. Errors if there are none (no schema).
pub(crate) fn materialize(batches: &[RecordBatch]) -> Result<RecordBatch, InterpError> {
    let first = batches.first().ok_or(InterpError::EmptyJoinInput)?;
    // A single batch (or empty/degenerate input) needs no copy.
    if batches.len() == 1 {
        return Ok(first.clone());
    }
    let schema = first.schema();
    let ncols = schema.fields().len();
    // A column-less batch carries only a row count, which `RecordBatch::try_new` can't
    // infer from zero columns — let `concat_batches` sum it.
    if ncols == 0 {
        return Ok(concat_batches(&schema, batches)?);
    }
    // Independent columns concatenate across rayon workers.
    let columns: Vec<ArrayRef> = (0..ncols)
        .into_par_iter()
        .map(|c| concat_column(batches, c))
        .collect::<Result<_, _>>()?;
    Ok(RecordBatch::try_new(schema, columns)?)
}

/// Concatenate column `c` across `batches`. For a null-free fixed-width primitive column
/// (the dominant `Int64`/`Float64` breaker input) this is a **parallel memcpy** into one
/// pre-sized buffer; any other column (nulls, strings, nested) uses arrow's `concat`,
/// byte-identical.
fn concat_column(batches: &[RecordBatch], c: usize) -> Result<ArrayRef, InterpError> {
    use arrow::datatypes::{DataType, Float64Type, Int64Type};
    let no_nulls = batches.iter().all(|b| b.column(c).null_count() == 0);
    if no_nulls {
        match batches[0].column(c).data_type() {
            DataType::Int64 => return Ok(fast_concat_primitive::<Int64Type>(batches, c)),
            DataType::Float64 => return Ok(fast_concat_primitive::<Float64Type>(batches, c)),
            _ => {}
        }
    }
    let cols: Vec<&dyn Array> = batches.iter().map(|b| b.column(c).as_ref()).collect();
    Ok(arrow::compute::concat(&cols)?)
}

/// A `*mut` wrapper asserting `Send`/`Sync` for the disjoint parallel copies in
/// [`fast_concat_primitive`]. Sound only because each thread writes a disjoint output
/// range (the per-morsel offsets partition `0..total`); private to this module.
#[derive(Clone, Copy)]
struct SendMutPtr<T>(*mut T);
// SAFETY: see [`fast_concat_primitive`] — writes through this pointer never alias across
// threads, so sharing it is race-free.
unsafe impl<T> Send for SendMutPtr<T> {}
unsafe impl<T> Sync for SendMutPtr<T> {}

/// Null-free primitive concat via parallel memcpy (see [`concat_column`]).
fn fast_concat_primitive<T>(batches: &[RecordBatch], c: usize) -> ArrayRef
where
    T: arrow::datatypes::ArrowPrimitiveType,
    T::Native: Copy,
{
    use arrow::array::AsArray;
    let chunks: Vec<&arrow::array::PrimitiveArray<T>> = batches
        .iter()
        .map(|b| b.column(c).as_primitive::<T>())
        .collect();
    let total: usize = chunks.iter().map(|a| a.len()).sum();
    // Per-chunk destination offsets partition `0..total`.
    let mut offsets = Vec::with_capacity(chunks.len());
    let mut off = 0usize;
    for a in &chunks {
        offsets.push(off);
        off += a.len();
    }
    let mut out: Vec<T::Native> = Vec::with_capacity(total);
    let dst = SendMutPtr(out.as_mut_ptr());
    chunks
        .par_iter()
        .zip(offsets.par_iter())
        .for_each(|(a, &o)| {
            let base = dst; // capture the whole `SendMutPtr` (Copy), not its raw field
            let src = a.values();
            // SAFETY: `o..o+src.len()` is within `0..total` and disjoint from every other
            // chunk's range (offsets are the running prefix sums of chunk lengths), so no two
            // threads write the same slot and every slot is written exactly once.
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), base.0.add(o), src.len());
            }
        });
    // SAFETY: every index in `0..total` was written exactly once by the copies above.
    unsafe {
        out.set_len(total);
    }
    Arc::new(arrow::array::PrimitiveArray::<T>::new(out.into(), None))
}
