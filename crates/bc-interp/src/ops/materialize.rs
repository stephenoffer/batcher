//! Concatenating morsels back into one batch — the first step of every pipeline
//! breaker (sort / join / asof / window). On a wide fact table at morsel granularity
//! this full copy dominates the operator, so it is parallelized: independent columns
//! fan across cores, and a null-free fixed-width primitive column copies via a
//! parallel memcpy (each morsel's values slice to its own disjoint output offset),
//! saturating memory bandwidth where arrow's serial per-chunk `concat` (~3 GB/s) does
//! not. The result is byte-identical to `concat_batches`.

use arrow::array::{Array, ArrayRef, RecordBatch};
use arrow::compute::concat_batches;
use arrow::datatypes::{Field, Schema, SchemaRef};
use rayon::prelude::*;
use std::sync::Arc;

use crate::error::InterpError;

/// Concatenate morsels into one batch, or `None` when there are none.
///
/// The bounded-memory breakers all begin by concatenating their input, and all of them
/// have to answer "what if the input is empty?". [`materialize`] answers it with
/// `EmptyJoinInput`, which invited every one of them to spell the empty case as
/// `match materialize(..) { Ok(b) => .., Err(_) => Vec::new() }` — and that discards
/// *every* error, not just the empty one. A schema mismatch between morsels, a `Utf8`
/// column whose concatenation overflows 32-bit offsets into a type the rebuilt schema
/// rejects, an allocation that fails on a large breaker: each of those returned **zero
/// rows and reported success**. Two of the five sites were in `exec_seq`, the Tier-0
/// oracle every other tier is checked against, so the silently-empty answer was the
/// reference answer.
///
/// Separating "there was nothing to concatenate" from "concatenating failed" is what
/// makes the empty case expressible without a catch-all, so callers can propagate the
/// real error with `?`.
pub(crate) fn materialize_opt(batches: &[RecordBatch]) -> Result<Option<RecordBatch>, InterpError> {
    if batches.is_empty() {
        return Ok(None);
    }
    materialize(batches).map(Some)
}

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
    // A column that outgrew 32-bit offsets comes back widened (`Utf8` -> `LargeUtf8`), so
    // the schema is rebuilt from what was actually produced rather than from the input's.
    Ok(RecordBatch::try_new(
        widened_schema(&schema, &columns),
        columns,
    )?)
}

/// `schema` with each field's type replaced by the concatenated column's actual type.
///
/// Only a `Utf8`/`Binary` column that overflowed 32-bit offsets differs (see
/// [`concat_column`]); every other field is returned unchanged, so this is a no-op on the
/// overwhelming majority of batches. Field names, nullability, and metadata are preserved.
fn widened_schema(schema: &SchemaRef, columns: &[ArrayRef]) -> SchemaRef {
    if schema
        .fields()
        .iter()
        .zip(columns)
        .all(|(f, a)| f.data_type() == a.data_type())
    {
        return Arc::clone(schema);
    }
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .zip(columns)
        .map(|(f, a)| f.as_ref().clone().with_data_type(a.data_type().clone()))
        .collect();
    Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()))
}

/// The largest total value-bytes a 32-bit-offset Arrow array (`Utf8`, `Binary`) can hold.
/// The last offset must be representable as an `i32`.
const OFFSET32_BYTE_LIMIT: usize = i32::MAX as usize;

/// Whether concatenating column `c` would overflow a 32-bit offset buffer.
///
/// `arrow::compute::concat` builds the output through a `GenericBytesBuilder`, which adds
/// each value's length to a running `i32` offset and **panics on overflow** rather than
/// returning an error (`attempt to add with overflow`). Because `materialize` runs on data
/// — it is the first step of every pipeline breaker — that abort is reachable from a user
/// query, which the crate's rules forbid. Detecting the overflow here lets the column be
/// widened instead (see `concat_widened`).
///
/// Only `Utf8`/`Binary` are checked: `LargeUtf8`/`LargeBinary` already use 64-bit offsets,
/// and fixed-width columns cannot overflow an offset buffer they do not have.
fn offset32_overflow(batches: &[RecordBatch], c: usize) -> Option<usize> {
    offset32_overflow_with(batches, c, OFFSET32_BYTE_LIMIT)
}

/// [`offset32_overflow`] against an explicit `limit`, so the check is testable without
/// allocating two gigabytes of strings.
fn offset32_overflow_with(batches: &[RecordBatch], c: usize, limit: usize) -> Option<usize> {
    use arrow::array::{BinaryArray, StringArray};
    use arrow::datatypes::DataType;

    let mut bytes: usize = 0;
    for batch in batches {
        let col = batch.column(c);
        let len = match col.data_type() {
            DataType::Utf8 => col
                .as_any()
                .downcast_ref::<StringArray>()
                .map(|a| value_bytes(a.value_offsets())),
            DataType::Binary => col
                .as_any()
                .downcast_ref::<BinaryArray>()
                .map(|a| value_bytes(a.value_offsets())),
            _ => return None, // no 32-bit offset buffer to overflow
        };
        bytes = bytes.saturating_add(len?);
        if bytes > limit {
            return Some(bytes);
        }
    }
    None
}

/// The value bytes an array actually holds: `offsets[len] - offsets[0]`.
///
/// **Not** `value_data().len()`. Every morsel is a zero-copy *slice* of its source batch,
/// and a sliced Arrow array keeps the whole values buffer and merely narrows its offsets —
/// so `value_data()` returns the source's bytes, not the slice's. Summing that across the
/// 3,663 morsels of a 60M-row `lineitem` counts one 264 MB buffer 3,663 times and declares
/// a 966 GB "overflow" on a column that fits comfortably. The offsets are the only honest
/// measure of a slice's own bytes.
fn value_bytes<O: arrow::array::OffsetSizeTrait>(offsets: &[O]) -> usize {
    match (offsets.first(), offsets.last()) {
        (Some(&first), Some(&last)) => last.as_usize() - first.as_usize(),
        _ => 0, // an empty array carries no offsets
    }
}

/// Concatenate column `c` across `batches`. For a null-free fixed-width primitive column
/// (the dominant `Int64`/`Float64` breaker input) this is a **parallel memcpy** into one
/// pre-sized buffer; any other column (nulls, strings, nested) uses arrow's `concat`,
/// byte-identical.
fn concat_column(batches: &[RecordBatch], c: usize) -> Result<ArrayRef, InterpError> {
    use arrow::datatypes::{DataType, Float64Type, Int64Type};
    if let Some(bytes) = offset32_overflow(batches, c) {
        return concat_widened(batches, c, bytes);
    }
    let no_nulls = batches.iter().all(|b| b.column(c).null_count() == 0);
    if no_nulls {
        match batches[0].column(c).data_type() {
            DataType::Int64 => return Ok(fast_concat_primitive::<Int64Type>(batches, c)),
            DataType::Float64 => return Ok(fast_concat_primitive::<Float64Type>(batches, c)),
            _ => {}
        }
    }
    let cols: Vec<&dyn Array> = batches.iter().map(|b| b.column(c).as_ref()).collect();
    // The `Int64`/`Float64` arms above are a parallel memcpy; everything else went to arrow's
    // `concat`, which copies a variable-width column row by row on one core. That left the
    // *string* half of every materialized relation serial — and this function is what
    // concatenates a whole probe side before an un-shardable join. `concat_columns` gives the
    // string columns the same treatment the numeric ones already had, and delegates to arrow for
    // the types it does not fast-path, so the result is byte-identical either way.
    Ok(bc_runtime::gather::concat_columns(&cols)?)
}

/// Concatenate a `Utf8`/`Binary` column that no longer fits 32-bit offsets by widening it
/// to its 64-bit-offset counterpart first.
///
/// Arrow's `concat` builds the output through a `GenericBytesBuilder`, which accumulates
/// each value's length into an `i32` and **panics on overflow** instead of erroring. Since
/// `materialize` is the first step of every pipeline breaker, that abort is reachable from
/// an ordinary query. Widening keeps the data addressable — every value is preserved and
/// the column's *logical* contents are unchanged; only its offset width grows. Refusing the
/// batch instead would make a legal plan unrunnable.
///
/// The widened type propagates in the rebuilt schema (`widened_schema`). Widening is a last
/// resort, not a routine path: it changes the column's Arrow type, and the scalar kernels in
/// `bc-expr` accept `Utf8` only (a `starts_with` or an `==` against a `Utf8` literal fails on
/// a `LargeUtf8` argument). Teaching those kernels both offset widths — or, better, not
/// materializing a whole relation at a breaker at all — is the durable fix.
fn concat_widened(
    batches: &[RecordBatch],
    c: usize,
    bytes: usize,
) -> Result<ArrayRef, InterpError> {
    use arrow::datatypes::DataType;
    let target = match batches[0].column(c).data_type() {
        DataType::Utf8 => DataType::LargeUtf8,
        DataType::Binary => DataType::LargeBinary,
        // `offset32_overflow` only reports the two 32-bit-offset byte types.
        _ => {
            return Err(InterpError::MaterializeOffsetOverflow {
                column: batches[0].schema().field(c).name().clone(),
                bytes,
                limit: OFFSET32_BYTE_LIMIT,
            })
        }
    };
    let widened: Vec<ArrayRef> = batches
        .iter()
        .map(|b| arrow::compute::cast(b.column(c), &target))
        .collect::<Result<_, _>>()?;
    let cols: Vec<&dyn Array> = widened.iter().map(|a| a.as_ref()).collect();
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, StringArray};

    fn str_batch(vals: &[&str]) -> RecordBatch {
        RecordBatch::try_from_iter(vec![(
            "s",
            Arc::new(StringArray::from(vals.to_vec())) as ArrayRef,
        )])
        .unwrap()
    }

    fn int_batch(vals: &[i64]) -> RecordBatch {
        RecordBatch::try_from_iter(vec![(
            "i",
            Arc::new(Int64Array::from(vals.to_vec())) as ArrayRef,
        )])
        .unwrap()
    }

    /// The value bytes accumulate across morsels: no single morsel need exceed the limit
    /// for the concatenated array to.
    #[test]
    fn overflow_is_detected_across_morsels() {
        let batches = [str_batch(&["aaaa", "bbbb"]), str_batch(&["cccc"])];
        assert_eq!(offset32_overflow_with(&batches, 0, 100), None);
        assert_eq!(offset32_overflow_with(&batches, 0, 8), Some(12));
        // Exactly at the limit is representable; one byte past is not.
        assert_eq!(offset32_overflow_with(&batches, 0, 12), None);
        assert_eq!(offset32_overflow_with(&batches, 0, 11), Some(12));
    }

    /// A morsel is a zero-copy slice of its source batch; its byte count must be its own,
    /// not the whole values buffer it still points at. Counting the buffer made a 264 MB
    /// column look like hundreds of gigabytes once summed over its morsels, needlessly
    /// widening it to `LargeUtf8` and breaking every downstream `Utf8` kernel.
    #[test]
    fn a_sliced_morsel_counts_only_its_own_bytes() {
        let source = str_batch(&["aaaa", "bbbb", "cccc", "dddd"]); // 16 bytes of values
        let morsels: Vec<RecordBatch> = (0..4).map(|i| source.slice(i, 1)).collect();
        // Each morsel holds 4 bytes, so the four together hold 16 — not 4 x 16.
        assert_eq!(offset32_overflow_with(&morsels, 0, 16), None);
        assert_eq!(offset32_overflow_with(&morsels, 0, 15), Some(16));
        // A single morsel is 4 bytes, not the source's 16.
        assert_eq!(offset32_overflow_with(&morsels[..1], 0, 4), None);
        assert_eq!(offset32_overflow_with(&morsels[..1], 0, 3), Some(4));
    }

    /// An empty array carries no values and cannot overflow anything.
    #[test]
    fn empty_array_counts_zero_bytes() {
        let empty = str_batch(&[]);
        assert_eq!(offset32_overflow_with(&[empty], 0, 0), None);
    }

    /// Fixed-width columns have no offset buffer, so they can never overflow one.
    #[test]
    fn fixed_width_columns_never_overflow() {
        let batches = [int_batch(&[1, 2, 3]), int_batch(&[4])];
        assert_eq!(offset32_overflow_with(&batches, 0, 0), None);
    }

    /// The real limit admits an ordinary string relation untouched.
    #[test]
    fn ordinary_strings_are_not_flagged() {
        let batches = [str_batch(&["hello", "world"])];
        assert_eq!(offset32_overflow(&batches, 0), None);
        assert!(materialize(&batches).is_ok());
    }

    /// An overflowing byte column is widened to 64-bit offsets rather than refused, and the
    /// values survive the widening unchanged.
    #[test]
    fn overflowing_column_is_widened_not_refused() {
        use arrow::array::LargeStringArray;
        use arrow::datatypes::DataType;

        let batches = [str_batch(&["aaaa", "bbbb"]), str_batch(&["cccc"])];
        // `concat_widened` is what the guard dispatches to; drive it directly, since no
        // test-sized input can reach the real `OFFSET32_BYTE_LIMIT`.
        let widened = concat_widened(&batches, 0, 12).expect("utf8 widens to large_utf8");
        assert_eq!(widened.data_type(), &DataType::LargeUtf8);
        let values = widened
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("large_utf8");
        assert_eq!(
            values.iter().collect::<Vec<_>>(),
            vec![Some("aaaa"), Some("bbbb"), Some("cccc")]
        );
    }

    /// The rebuilt schema follows the columns actually produced, so a widened column does
    /// not fail `RecordBatch::try_new`'s type check.
    #[test]
    fn schema_follows_the_widened_columns() {
        use arrow::array::LargeStringArray;
        use arrow::datatypes::DataType;

        let batches = [str_batch(&["a"])];
        let schema = batches[0].schema();
        let widened: ArrayRef = Arc::new(LargeStringArray::from(vec!["a"]));
        let out = widened_schema(&schema, std::slice::from_ref(&widened));
        assert_eq!(out.field(0).data_type(), &DataType::LargeUtf8);
        assert_eq!(out.field(0).name(), "s");
        // An unchanged column reuses the input schema object.
        let same = widened_schema(&schema, &[Arc::clone(batches[0].column(0))]);
        assert!(Arc::ptr_eq(&schema, &same));
    }

    /// No input is `None`; it is the *only* thing that is.
    #[test]
    fn empty_input_is_none_not_an_error() {
        assert!(materialize_opt(&[]).unwrap().is_none());
        let one = materialize_opt(std::slice::from_ref(&int_batch(&[1, 2]))).unwrap();
        assert_eq!(one.unwrap().num_rows(), 2);
    }

    /// The distinction the breakers depend on: morsels that cannot be concatenated are an
    /// **error**, never an empty relation.
    ///
    /// Every breaker used to spell its empty case as `Err(_) => Vec::new()`, so this input
    /// produced zero rows and reported success — in `exec_seq`, the oracle included. The
    /// mismatch here (`s: Utf8` then `i: Int64`) stands in for the ones a real corpus
    /// produces: a schema that drifted between files, a column concatenated past 32-bit
    /// offsets.
    #[test]
    fn mismatched_morsels_error_rather_than_vanish() {
        let mixed = [str_batch(&["a"]), int_batch(&[1])];
        assert!(materialize_opt(&mixed).is_err());
    }
}
