//! Column gather (`take`) and multi-array `concat`, with fast paths for variable-length
//! **byte** columns: `Utf8`, `LargeUtf8`, `Binary` and `LargeBinary`.
//!
//! Gathering a permutation is the dominant cost of a sort and of a join's output, and for
//! these four arrow's `take` is far slower than the memory it moves: it drives
//! `MutableArrayData::extend` once per row, paying a call and bounds checks to copy a
//! handful of bytes. On a 5 M-row sort, adding one string column cost ~52 ms — an order of
//! magnitude more than the ~50 MB of characters involved.
//!
//! All four are one layout — an offset buffer and a value buffer — so they are one
//! implementation here, generic over `ByteArrayType`. That was not always true: the fast
//! paths were written for `Utf8`/`LargeUtf8`, and `Binary` — the *payload* type of every blob,
//! every serialized value, and every fixed-layout record — fell to [`take_chunked`], which
//! `take`s each chunk with arrow and concatenates them. That is parallel, and it writes every
//! value **twice**.
//!
//! Removing the second write is what the widening buys, and it is a factor (1.04x-1.27x
//! against the chunked path, 10x-11x against arrow's single-shot `take`) rather than the order
//! of magnitude the serial-to-parallel string case saw. The measurement, and two ways its
//! first draft misread itself, are in `report_the_byte_gather`.
//!
//! `concat` has exactly the same shape and the same problem, and it sits on a hotter path:
//! a high-cardinality group-by concatenates every worker's partial keys, radix-partitions
//! them, and concatenates the partitions' outputs — three passes over the key column. On
//! ClickBench q33 (`GROUP BY URL`, 1 M rows, 275 k groups) those concats measured **60 ms of
//! a 70 ms combine**, moving 43 MB at 1.45 GB/s while the grouping itself took 7.6 ms.
//!
//! The fast paths do what the shape allows: sum the lengths into the offset buffer, then
//! `copy_from_slice` the bytes, and carve the output into disjoint slices so the copying runs
//! across cores. For `concat` the destination ranges are known up front, one per input. For
//! `take` they are not — a row's output offset is the running sum of every preceding row's
//! length — so a large gather measures first, prefix-sums the per-chunk totals, and only then
//! copies; that is what turns the same prefix-sum dependency into independent work.
//!
//! Everything else — every other data type, a nullable index array, an offset overflow —
//! delegates to arrow, so these are pure performance short-circuits and never a second
//! semantics.

use arrow::array::{Array, ArrayRef, GenericByteArray, UInt32Array};
use arrow::buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow::compute::take;
use arrow::datatypes::{
    ArrowNativeType, BinaryType, ByteArrayType, DataType, LargeBinaryType, LargeUtf8Type, Utf8Type,
};
use rayon::prelude::*;
use std::sync::Arc;

mod fixed;

use fixed::{take_chunked, take_fixed_width_parallel};

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
                        if let Some(out) = concat_bytes::<Utf8Type>(arrays) {
                            return Ok(Arc::new(out));
                        }
                        // `concat_bytes` returned `None`. Either the downcast failed (an
                        // unexpected layout — arrow's own `concat` handles it) or the result
                        // does not fit 32-bit offsets, and *that* case must not reach arrow:
                        // its builder does `.expect("byte array offset overflow")`, so it
                        // aborts the process rather than returning an error. A panic inside a
                        // rayon worker crosses the FFI as an unrecoverable `PanicException`
                        // and takes the whole query engine with it.
                        //
                        // Reached on the Join Order Benchmark's `q7c`, whose join output holds
                        // more than 2 GiB of `name`/`title` text in one column. DuckDB answers
                        // it in 936 ms; Batcher aborted.
                        byte_span_fits::<Utf8Type>(arrays)?;
                    }
                    DataType::LargeUtf8 => {
                        if let Some(out) = concat_bytes::<LargeUtf8Type>(arrays) {
                            return Ok(Arc::new(out));
                        }
                        byte_span_fits::<LargeUtf8Type>(arrays)?;
                    }
                    // Binary is the same layout and the same bulk copy; only the element type
                    // differs, and it differs in Batcher's favour — a `Binary` result needs no
                    // UTF-8 validation pass where a `Utf8` one does.
                    DataType::Binary => {
                        if let Some(out) = concat_bytes::<BinaryType>(arrays) {
                            return Ok(Arc::new(out));
                        }
                        byte_span_fits::<BinaryType>(arrays)?;
                    }
                    DataType::LargeBinary => {
                        if let Some(out) = concat_bytes::<LargeBinaryType>(arrays) {
                            return Ok(Arc::new(out));
                        }
                        byte_span_fits::<LargeBinaryType>(arrays)?;
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
/// `None` when any input is not a `GenericByteArray<T>` or the concatenated characters would
/// overflow the offset type, so the caller falls back to arrow's `concat` (which errors or
/// widens as it sees fit) rather than wrapping.
/// Refuse a concat whose value bytes cannot be addressed by `T`'s offset width.
///
/// Arrow's `concat` builds through `GenericByteBuilder`, which `.expect()`s on the offset
/// conversion — so an oversized result aborts the process instead of returning an error. This
/// is the guard in front of it: same arithmetic as [`concat_bytes`], reported as a typed error
/// naming the fix. `Ok(())` when the arrays are not the expected layout at all, which is
/// arrow's case to handle rather than this one's.
fn byte_span_fits<T: ByteArrayType>(arrays: &[&dyn Array]) -> Result<(), RuntimeError> {
    let mut total: usize = 0;
    for a in arrays {
        let Some(s) = a.as_any().downcast_ref::<GenericByteArray<T>>() else {
            return Ok(()); // not the layout we model; let arrow decide
        };
        let o = s.value_offsets();
        total += o[s.len()].as_usize() - o[0].as_usize();
    }
    if T::Offset::from_usize(total).is_none() {
        return Err(RuntimeError::ByteOffsetOverflow {
            dtype: arrays[0].data_type().to_string(),
            bytes: total,
        });
    }
    Ok(())
}

fn concat_bytes<T: ByteArrayType>(arrays: &[&dyn Array]) -> Option<GenericByteArray<T>> {
    let arrs: Vec<&GenericByteArray<T>> = arrays
        .iter()
        .map(|a| a.as_any().downcast_ref::<GenericByteArray<T>>())
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
    T::Offset::from_usize(total_bytes)?;

    // Where each input's characters begin in the output — the exclusive prefix sum over the
    // spans. One add per *input*, and the figure both the offset and the value pass need.
    let mut bases: Vec<usize> = Vec::with_capacity(arrs.len());
    let mut base = 0usize;
    for (start, end) in &spans {
        bases.push(base);
        base += end - start;
    }

    // Offsets, carved and filled the same way the values are below. This was a serial pass on
    // the grounds that it is one add per row against a copy of the characters — true per byte,
    // but the copy runs across every core and this did not. Measured at 350 MB over 92 inputs
    // it is worth about 5% (107 ms → 101 ms): the byte copy still dominates, so this removes
    // the last sequential section rather than the bottleneck. Element 0 is the leading zero
    // every Arrow offset buffer carries, so each input's entries are the *ends* and sit one
    // slot to the right.
    let mut offsets: Vec<T::Offset> = vec![T::Offset::usize_as(0); total_rows + 1];
    let mut orest = &mut offsets[1..];
    let mut odsts: Vec<&mut [T::Offset]> = Vec::with_capacity(arrs.len());
    for a in &arrs {
        let (head, tail) = orest.split_at_mut(a.len());
        odsts.push(head);
        orest = tail;
    }
    odsts
        .into_par_iter()
        .zip(arrs.par_iter().zip(spans.par_iter()).zip(bases.par_iter()))
        .for_each(|(dst, ((a, (start, _end)), &base))| {
            let src = a.value_offsets();
            for (i, slot) in dst.iter_mut().enumerate() {
                *slot = T::Offset::usize_as(base + src[i + 1].as_usize() - start);
            }
        });

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

    Some(GenericByteArray::<T>::new(
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        values.into(),
        nulls,
    ))
}

/// Gather `col`'s rows at `indices`, matching `arrow::compute::take` exactly.
pub fn take_column(col: &dyn Array, indices: &UInt32Array) -> Result<ArrayRef, RuntimeError> {
    // A null index means a null output row; the length/copy loops below assume a value.
    if indices.null_count() == 0 {
        // The single-pass byte fill, for the four variable-length byte types that share a
        // layout. It declines when the gathered bytes would not fit the offset width, and that
        // decline must fall to `take_chunked` rather than to arrow: arrow's builder `.expect()`s
        // on the offset conversion, so an oversized `Binary` gather would abort the process
        // instead of returning an error, and gathering it in chunks is what keeps each piece
        // inside the width. `concat_columns` then makes the same check for the join of them.
        let bytes = match col.data_type() {
            DataType::Utf8 => take_byte_array::<Utf8Type>(col, indices),
            DataType::LargeUtf8 => take_byte_array::<LargeUtf8Type>(col, indices),
            DataType::Binary => take_byte_array::<BinaryType>(col, indices),
            DataType::LargeBinary => take_byte_array::<LargeBinaryType>(col, indices),
            _ => None,
        };
        if let Some(out) = bytes {
            return Ok(out);
        }
        if let Some(out) = take_fixed_width_parallel(col, indices) {
            return Ok(out);
        }
        if let Some(out) = take_chunked(col, indices) {
            return out;
        }
    }
    Ok(take(col, indices, None)?)
}

/// [`take_bytes`] behind the downcast, so the dispatch above names a type once rather than
/// naming both its array struct and its `ByteArrayType` witness.
fn take_byte_array<T: ByteArrayType>(col: &dyn Array, indices: &UInt32Array) -> Option<ArrayRef> {
    let arr = col.as_any().downcast_ref::<GenericByteArray<T>>()?;
    Some(Arc::new(take_bytes::<T>(arr, indices)?))
}

/// Gather one output column from *several* byte-array sources addressed by two `u32` planes:
/// output row `k` is `cols[part_of[k]]` at row `row_of[k]`. `None` for anything that is not a
/// uniform `Utf8`/`LargeUtf8`/`Binary`/`LargeBinary` set, or whose bytes overflow the offset
/// width.
///
/// This is `arrow::compute::interleave` for the one type where it costs what a `take` of the
/// same size costs: it builds through `MutableArrayData::extend`, a call and bounds checks per
/// row to copy a handful of bytes, and it wants a materialized `&[(usize, usize)]` — sixteen
/// bytes of index per output row where the planes are eight. The aggregate `combine` already
/// avoids both for primitive columns and had nothing for string keys, which is the *common*
/// high-cardinality group key: `interleave_bytes` was 6% of H2O `groupby` q2 (`sum(v1) BY id1,
/// id2`, two string keys, 10,000 groups over 10M rows) and is on every `GROUP BY <string>`
/// that reaches the parallel merge.
///
/// Same three phases as [`take_bytes_parallel`], and for the same reason — a row's output
/// offset is the running sum of the ones before it, which is a prefix sum and therefore
/// decomposable. The only difference is that a row's bytes are found through a second plane.
pub fn gather_bytes(cols: &[&dyn Array], part_of: &[u32], row_of: &[u32]) -> Option<ArrayRef> {
    let dt = cols.first()?.data_type();
    if !cols.iter().all(|c| c.data_type() == dt) {
        return None;
    }
    match dt {
        DataType::Utf8 => Some(Arc::new(gather_bytes_of::<Utf8Type>(
            cols, part_of, row_of,
        )?)),
        DataType::LargeUtf8 => Some(Arc::new(gather_bytes_of::<LargeUtf8Type>(
            cols, part_of, row_of,
        )?)),
        DataType::Binary => Some(Arc::new(gather_bytes_of::<BinaryType>(
            cols, part_of, row_of,
        )?)),
        DataType::LargeBinary => Some(Arc::new(gather_bytes_of::<LargeBinaryType>(
            cols, part_of, row_of,
        )?)),
        _ => None,
    }
}

fn gather_bytes_of<T: ByteArrayType>(
    cols: &[&dyn Array],
    part_of: &[u32],
    row_of: &[u32],
) -> Option<GenericByteArray<T>> {
    let arrs: Vec<&GenericByteArray<T>> = cols
        .iter()
        .map(|a| a.as_any().downcast_ref::<GenericByteArray<T>>())
        .collect::<Option<_>>()?;
    let n = part_of.len();

    // Phase 1 (parallel): each row's source span, and each chunk's byte total. The span is
    // recorded so phase 3 reads a sequentially-addressed scratch array rather than chasing
    // `offsets[row_of[k]]` through a cold buffer a second time. Which *array* it came from is
    // not recorded — `part_of` is already a sequential read in phase 3.
    let mut spans: Vec<(u32, u32)> = vec![(0, 0); n];
    let chunk_totals: Vec<Option<usize>> = spans
        .par_chunks_mut(TAKE_CHUNK_ROWS)
        .zip(part_of.par_chunks(TAKE_CHUNK_ROWS))
        .zip(row_of.par_chunks(TAKE_CHUNK_ROWS))
        .map(|((out, parts), rows)| {
            let mut total = 0usize;
            for (slot, (&p, &r)) in out.iter_mut().zip(parts.iter().zip(rows)) {
                let offsets = arrs.get(p as usize)?.value_offsets();
                let start = offsets.get(r as usize)?.as_usize();
                let len = offsets.get(r as usize + 1)?.as_usize() - start;
                *slot = (u32::try_from(start).ok()?, u32::try_from(len).ok()?);
                total += len;
            }
            Some(total)
        })
        .collect();
    let chunk_totals: Vec<usize> = chunk_totals.into_iter().collect::<Option<_>>()?;

    // Phase 2: exclusive prefix sum over chunks, and the offset-width check, once per chunk.
    let mut bases: Vec<usize> = Vec::with_capacity(chunk_totals.len());
    let mut running = 0usize;
    for &t in &chunk_totals {
        bases.push(running);
        running += t;
    }
    T::Offset::from_usize(running)?;

    // Phase 3: carve the output into the ranges phase 2 reserved and fill each in parallel.
    let mut values = vec![0u8; running];
    let mut rest = values.as_mut_slice();
    let mut dsts: Vec<&mut [u8]> = Vec::with_capacity(chunk_totals.len());
    for &t in &chunk_totals {
        let (head, tail) = rest.split_at_mut(t);
        dsts.push(head);
        rest = tail;
    }
    let mut offsets: Vec<T::Offset> = vec![T::Offset::usize_as(0); n + 1];
    offsets[1..]
        .par_chunks_mut(TAKE_CHUNK_ROWS)
        .zip(spans.par_chunks(TAKE_CHUNK_ROWS))
        .zip(part_of.par_chunks(TAKE_CHUNK_ROWS))
        .zip(dsts.into_par_iter().zip(bases.par_iter()))
        .for_each(|(((offs, rows), parts), (dst, &base))| {
            let mut at = 0usize;
            for ((slot, &(start, len)), &p) in offs.iter_mut().zip(rows).zip(parts) {
                let (start, len) = (start as usize, len as usize);
                dst[at..at + len]
                    .copy_from_slice(&arrs[p as usize].value_data()[start..start + len]);
                at += len;
                *slot = T::Offset::usize_as(base + at);
            }
        });

    // A gathered row is null exactly when its source row is, which is what `interleave` also
    // produces. Built only when some source actually has nulls, matching arrow.
    let nulls = arrs.iter().any(|a| a.null_count() > 0).then(|| {
        NullBuffer::from_iter(
            part_of
                .iter()
                .zip(row_of)
                .map(|(&p, &r)| arrs[p as usize].is_valid(r as usize)),
        )
    });
    Some(GenericByteArray::<T>::new(
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        values.into(),
        nulls,
    ))
}

/// How far ahead [`take_bytes`] prefetches the *offset* pair for an upcoming row.
///
/// Far enough that the fetch completes before the loop arrives — a last-level miss is on the
/// order of 200-300 cycles and one iteration is a handful — and near enough that the line is
/// not evicted before use.
///
/// Measured (5 M rows, 302 MB of URL-width strings, random permutation, Cascade Lake): the
/// prefetch alone is worth **982 ms → 522 ms, 1.88x** on one core, and the distance itself
/// barely matters — every value swept from 16 to 96 landed inside the run-to-run noise band.
/// So this is not a tuned constant to be preserved, it is a point on a plateau; what matters
/// is that the loads are issued early at all. With [`take_bytes_parallel`] on top, the same
/// gather is **133 ms**, a 7.4x total.
pub(super) const PREFETCH_OFFSET_DISTANCE: usize = 32;

/// How far ahead [`take_bytes`] prefetches the *bytes* of an upcoming row.
///
/// Half the offset distance, because computing a row's byte address requires its offset pair
/// to have landed first. At this point the far prefetch has already brought that pair in, so
/// reading it here is a cache hit rather than the miss it would be at the head of the loop.
pub(super) const PREFETCH_VALUE_DISTANCE: usize = 16;

/// Rows above which the gather is worth spreading across cores.
///
/// Below it the three-phase parallel form loses to the single pass: it allocates a per-row
/// scratch vector and walks the rows twice, which only pays back once the copying is large
/// enough to hide it. One morsel's worth of rows is not; several are.
pub(super) const PARALLEL_TAKE_MIN_ROWS: usize = 1 << 17;

/// Rows per parallel gather chunk.
///
/// Small enough that a 96-core box gets real work-stealing granularity on a few-million-row
/// gather, large enough that the per-chunk bookkeeping is noise against the bytes copied.
pub(super) const TAKE_CHUNK_ROWS: usize = 1 << 14;

/// Gather `arr`'s rows at `indices`, spreading the work across cores once it is large enough.
///
/// `None` when the gathered characters would overflow the offset type, so the caller falls
/// back to arrow's `take` (which widens or errors as it sees fit) rather than wrapping. Both
/// paths return exactly the same relation — the split is scheduling, and
/// `the_parallel_and_serial_gathers_agree` holds them to it.
fn take_bytes<T: ByteArrayType>(
    arr: &GenericByteArray<T>,
    indices: &UInt32Array,
) -> Option<GenericByteArray<T>> {
    if indices.len() >= PARALLEL_TAKE_MIN_ROWS && rayon::current_num_threads() > 1 {
        take_bytes_parallel(arr, indices)
    } else {
        take_bytes_serial(arr, indices)
    }
}

/// The single-pass gather: one loop, one output buffer, no scratch.
///
/// The reference the parallel path is checked against, and the path a morsel-sized gather
/// actually takes — below a few hundred thousand rows the extra measure pass costs more than
/// the cores buy back.
fn take_bytes_serial<T: ByteArrayType>(
    arr: &GenericByteArray<T>,
    indices: &UInt32Array,
) -> Option<GenericByteArray<T>> {
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
    let mut offsets: Vec<T::Offset> = Vec::with_capacity(n + 1);
    let mut values: Vec<u8> = Vec::with_capacity(reserve);
    let mut total: usize = 0;
    offsets.push(T::Offset::usize_as(0));
    for (k, &i) in idx.iter().enumerate() {
        // Without this the loop is a dependent chain of cache misses: the offset load for row
        // `k` cannot start until `idx[k]` is known, and the byte copy cannot start until that
        // offset arrives. On a 5 M-row source the offset buffer alone is 20-40 MB, so every
        // one of those loads misses the last-level cache and the loop runs at memory latency
        // rather than memory bandwidth. Issuing both loads early, for rows the loop has not
        // reached, lets the misses overlap with the copying of the current row.
        //
        // Two distances because the addresses depend on each other. The far one pulls in the
        // offset pair; by the time the near one needs that pair to compute where the *bytes*
        // live, it is already resident, so reading it there costs nothing.
        if let Some(&far) = idx.get(k + PREFETCH_OFFSET_DISTANCE) {
            bc_arrow::prefetch_read(&src_offsets[far as usize]);
        }
        if let Some(&near) = idx.get(k + PREFETCH_VALUE_DISTANCE) {
            // `get`, not indexing: a row's start offset equals the value buffer's length when
            // it is the last row and that row is empty, so the address to prefetch is one past
            // the end. There is nothing to pull in for an empty row anyway.
            if let Some(byte) = src_values.get(src_offsets[near as usize].as_usize()) {
                bc_arrow::prefetch_read(byte);
            }
        }
        let i = i as usize;
        let start = src_offsets[i].as_usize();
        let end = src_offsets[i + 1].as_usize();
        values.extend_from_slice(&src_values[start..end]);
        total += end - start;
        offsets.push(T::Offset::from_usize(total)?);
    }

    // A gathered row is null exactly when its source row is. Arrow's `take` leaves a null
    // row's slice empty, which the length pass above already does (start == end).
    let nulls = arr
        .nulls()
        .map(|src| NullBuffer::from_iter(idx.iter().map(|&i| src.is_valid(i as usize))));

    Some(GenericByteArray::<T>::new(
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        values.into(),
        nulls,
    ))
}

/// [`take_bytes`] across cores, for a gather large enough to pay for the extra pass.
///
/// The serial form is inherently sequential in one place only: a row's output offset is the
/// running sum of every preceding row's length. That is a prefix sum, so it decomposes the
/// same way the radix scatter next door does — measure, prefix-sum the chunk totals, then let
/// every chunk write into the disjoint output range it reserved.
///
/// Three passes, of which two are parallel:
///
/// 1. **Measure** (parallel). Each chunk reads its rows' offset pairs and records `(start,
///    len)` per row plus its own byte total. Recording the pair is what keeps the random read
///    of the source offset buffer to *one* pass — phase 3 then works from a sequentially-read
///    scratch array instead of chasing `src_offsets[idx[k]]` a second time.
/// 2. **Prefix-sum** the chunk totals. One add per chunk, not per row.
/// 3. **Copy** (parallel). The output byte buffer is carved into per-chunk slices with
///    `split_at_mut`, so each chunk copies into memory no other chunk can name — the same
///    safe-by-construction carve `concat_bytes` uses, with no `unsafe` anywhere.
///
/// Returns `None` on offset overflow, exactly like the serial path, so the caller still falls
/// back to arrow rather than wrapping.
fn take_bytes_parallel<T: ByteArrayType>(
    arr: &GenericByteArray<T>,
    indices: &UInt32Array,
) -> Option<GenericByteArray<T>> {
    let n = indices.len();
    let src_offsets = arr.value_offsets();
    let src_values = arr.value_data();
    let idx = indices.values();

    // Phase 1: per-row source span and per-chunk byte total, computed independently.
    // `(u32, u32)` rather than the offset type: a source span is bounded by the *source*
    // value buffer, which an `i32`-offset array caps at 2 GiB, and a `LargeUtf8` source
    // large enough to exceed 4 GiB in one gather is past what this fast path claims — the
    // `checked` conversion below sends it to arrow instead of truncating.
    let mut spans: Vec<(u32, u32)> = vec![(0, 0); n];
    let chunk_totals: Vec<Option<usize>> = spans
        .par_chunks_mut(TAKE_CHUNK_ROWS)
        .zip(idx.par_chunks(TAKE_CHUNK_ROWS))
        .map(|(out, rows)| {
            let mut total = 0usize;
            for (k, (slot, &i)) in out.iter_mut().zip(rows).enumerate() {
                // This pass touches only the offset buffer, so it needs only the offset
                // prefetch — the same dependent miss chain the serial loop has, and the same
                // fix. The byte prefetch belongs in phase 3, where the bytes are read.
                if let Some(&far) = rows.get(k + PREFETCH_OFFSET_DISTANCE) {
                    bc_arrow::prefetch_read(&src_offsets[far as usize]);
                }
                let i = i as usize;
                let start = src_offsets[i].as_usize();
                let len = src_offsets[i + 1].as_usize() - start;
                *slot = (u32::try_from(start).ok()?, u32::try_from(len).ok()?);
                total += len;
            }
            Some(total)
        })
        .collect();
    let chunk_totals: Vec<usize> = chunk_totals.into_iter().collect::<Option<_>>()?;

    // Phase 2: exclusive prefix sum over chunks, and the overflow check the serial path makes
    // per row. One add per chunk.
    let mut bases: Vec<usize> = Vec::with_capacity(chunk_totals.len());
    let mut running = 0usize;
    for &t in &chunk_totals {
        bases.push(running);
        running += t;
    }
    let total_bytes = running;
    // Refuse rather than wrap when the gathered characters exceed this offset width.
    T::Offset::from_usize(total_bytes)?;

    // Phase 3: carve the output into the disjoint ranges phase 2 reserved, then fill each in
    // parallel. `vec![0; n]` is an `alloc_zeroed`, which at this size is zero pages rather
    // than a write pass.
    let mut values = vec![0u8; total_bytes];
    let mut rest = values.as_mut_slice();
    let mut dsts: Vec<&mut [u8]> = Vec::with_capacity(chunk_totals.len());
    for &t in &chunk_totals {
        let (head, tail) = rest.split_at_mut(t);
        dsts.push(head);
        rest = tail;
    }
    // Offsets are carved the same way: chunk `c` owns output rows `[c*CHUNK, ...)`, and its
    // running offsets start from `bases[c]`. Element 0 is the leading zero every Arrow offset
    // buffer carries, so the per-row entries are the *ends* and land one slot to the right.
    let mut offsets: Vec<T::Offset> = vec![T::Offset::usize_as(0); n + 1];
    offsets[1..]
        .par_chunks_mut(TAKE_CHUNK_ROWS)
        .zip(spans.par_chunks(TAKE_CHUNK_ROWS))
        .zip(dsts.into_par_iter().zip(bases.par_iter()))
        .for_each(|((offs, rows), (dst, &base))| {
            let mut at = 0usize;
            for (k, (slot, &(start, len))) in offs.iter_mut().zip(rows).enumerate() {
                // The source span is already known here (phase 1 recorded it, and `rows` is
                // read sequentially), so the only miss left is the source *bytes* — which can
                // therefore be pulled in a fixed distance ahead with no dependent load first.
                if let Some(&(ahead, _)) = rows.get(k + PREFETCH_VALUE_DISTANCE) {
                    if let Some(byte) = src_values.get(ahead as usize) {
                        bc_arrow::prefetch_read(byte);
                    }
                }
                let (start, len) = (start as usize, len as usize);
                dst[at..at + len].copy_from_slice(&src_values[start..start + len]);
                at += len;
                *slot = T::Offset::usize_as(base + at);
            }
        });

    // A gathered row is null exactly when its source row is — identical to the serial path.
    let nulls = arr
        .nulls()
        .map(|src| NullBuffer::from_iter(idx.iter().map(|&i| src.is_valid(i as usize))));

    Some(GenericByteArray::<T>::new(
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        values.into(),
        nulls,
    ))
}

#[cfg(test)]
mod tests {
    use arrow::array::{
        BinaryArray, Decimal128Array, DictionaryArray, Float64Array, Int64Array, LargeBinaryArray,
        LargeStringArray, StringArray, TimestampMicrosecondArray,
    };
    use arrow::datatypes::Int32Type;

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

    /// `Binary` and `LargeBinary` are the same layout as the string types, so they take the
    /// same fast path — and must produce what arrow's `take` produces, byte for byte.
    ///
    /// These are the payload types: a blob, a serialized value, an encoded key, the 90 bytes of
    /// a fixed-layout record. They fell to arrow's per-row builder until the fast paths were
    /// made generic over the layout rather than over the element type.
    #[test]
    fn gathers_binary_like_arrow() {
        let values: Vec<Option<&[u8]>> = vec![
            Some(b"alpha"),
            None,
            Some(b""),
            Some(b"\x00\x01\x02"),
            Some(b"dddd"),
        ];
        assert_matches_arrow(
            &BinaryArray::from(values.clone()),
            &idx(&[4, 0, 2, 2, 1, 3]),
        );
        assert_matches_arrow(&LargeBinaryArray::from(values), &idx(&[4, 0, 2, 2, 1, 3]));
    }

    /// A gather large enough to take the *parallel* byte path, on a binary column, against
    /// arrow — the chunked prefix-sum is where a layout generalization would go wrong, and it
    /// is not exercised by the small cases above.
    #[test]
    fn gathers_binary_in_parallel_like_arrow() {
        let rows = PARALLEL_TAKE_MIN_ROWS + 1_000;
        let source: Vec<Option<Vec<u8>>> = (0..rows)
            .map(|i| (i % 13 != 0).then(|| vec![(i % 251) as u8; i % 17]))
            .collect();
        let a = BinaryArray::from_iter(source.iter().map(|v| v.as_deref()));
        let indices = idx(&(0..rows)
            .map(|i| ((i * 4099) % rows) as u32)
            .collect::<Vec<u32>>());
        assert_matches_arrow(&a, &indices);
    }

    /// `FixedSizeBinary` gathers by a fixed stride, and must equal arrow's `take` on every
    /// shape that can move where a row's bytes begin: nulls, a **slice** of a larger array,
    /// several widths, and a zero-width column.
    ///
    /// The slice is the load-bearing case. `value_data()` is the whole underlying buffer while
    /// `value(i)` reads at `(offset + i) * width`, so a fill that indexed from zero would
    /// gather the wrong bytes from any sliced array — a wrong answer, not an error, on a path
    /// every morselized executor hands slices to.
    #[test]
    fn gathers_fixed_size_binary_like_arrow() {
        use arrow::array::FixedSizeBinaryArray;

        for width in [1usize, 4, 10, 90] {
            let rows = PARALLEL_TAKE_MIN_ROWS + 64;
            let values: Vec<Option<Vec<u8>>> = (0..rows)
                .map(|i| (i % 11 != 0).then(|| vec![(i % 251) as u8; width]))
                .collect();
            let a = FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                values.into_iter(),
                width as i32,
            )
            .expect("uniform width");
            let indices = idx(&(0..rows)
                .map(|i| ((i * 4099) % rows) as u32)
                .collect::<Vec<u32>>());
            assert_matches_arrow(&a, &indices);

            // The same gather over a slice, whose rows start part-way into the value buffer.
            let sliced = a.slice(7, rows - 7);
            let n = sliced.len();
            let sliced_idx = idx(&(0..n)
                .map(|i| ((i * 4099) % n) as u32)
                .collect::<Vec<u32>>());
            assert_matches_arrow(&sliced, &sliced_idx);
        }
    }

    /// A byte gather that declines must reach the **chunked** fallback, not arrow.
    ///
    /// This is a routing assertion, not a size one: the decline that matters is an output too
    /// large for the offset width, and arrow's builder `.expect()`s on that conversion — so a
    /// 2 GiB `Binary` gather routed to arrow aborts the process rather than returning an error.
    /// Gathering in chunks is what keeps each piece inside the width. The size itself cannot be
    /// tested here (it needs 2 GiB), so what is pinned is that `take_chunked` still claims the
    /// two binary types, which is the thing a later cleanup would remove as dead.
    #[test]
    fn a_declining_byte_gather_still_has_a_chunked_fallback() {
        let rows = PARALLEL_TAKE_MIN_ROWS + 16;
        let payload = [7u8; 4];
        let a = BinaryArray::from_iter_values((0..rows).map(|_| &payload[..]));
        let indices = idx(&(0..rows as u32).rev().collect::<Vec<u32>>());
        assert!(
            take_chunked(&a, &indices).is_some(),
            "take_chunked must still claim Binary, or a declining fast path falls to arrow"
        );
        let large = LargeBinaryArray::from_iter_values((0..rows).map(|_| &payload[..]));
        assert!(take_chunked(&large, &indices).is_some(), "and LargeBinary");
        // Both fallbacks produce arrow's answer, which is what makes them usable as one.
        assert_matches_arrow(&a, &indices);
        assert_matches_arrow(&large, &indices);
    }

    /// The `concat` counterpart, including a sliced input and nulls in only some inputs.
    #[test]
    fn concatenates_binary_like_arrow() {
        let a = BinaryArray::from(vec![Some(b"x".as_ref()), None, Some(b"yy".as_ref())]);
        let b = BinaryArray::from(vec![Some(b"\x00\xff".as_ref())]);
        let c = BinaryArray::from(Vec::<Option<&[u8]>>::new());
        assert_concat_matches_arrow(&[&a, &b, &c]);
        assert_concat_matches_arrow(&[&c, &a]);
        assert_concat_matches_arrow(&[&a.slice(1, 2), &b]);

        let la = LargeBinaryArray::from(vec![Some(b"aa".as_ref()), None]);
        let lb = LargeBinaryArray::from(vec![Some(b"bbb".as_ref())]);
        assert_concat_matches_arrow(&[&la, &lb]);
    }

    /// The plane-addressed gather the parallel aggregate combine uses must take binary too,
    /// for the same reason: a binary group key or payload otherwise falls to `interleave`.
    #[test]
    fn the_plane_addressed_binary_gather_equals_interleave() {
        let a = BinaryArray::from(vec![Some(b"x".as_ref()), None, Some(b"yy".as_ref())]);
        let b = BinaryArray::from(vec![Some(b"zzz".as_ref()), Some(b"".as_ref())]);
        let part_of = [0u32, 1, 0, 1, 0];
        let row_of = [2u32, 0, 1, 1, 0];
        let got = gather_bytes(&[&a, &b], &part_of, &row_of).expect("binary is modelled");
        let pairs: Vec<(usize, usize)> = part_of
            .iter()
            .zip(&row_of)
            .map(|(&p, &r)| (p as usize, r as usize))
            .collect();
        let want = arrow::compute::interleave(&[&a, &b], &pairs).unwrap();
        assert_eq!(got.as_ref(), want.as_ref());
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

    /// The parallel gather must equal arrow's `take` exactly, not merely as a multiset.
    ///
    /// Every other test in this module is below `PARALLEL_TAKE_MIN_ROWS`, so none of them
    /// reaches `take_bytes_parallel` at all — the three-phase path could be arbitrarily
    /// wrong and they would all still pass. This is the one that runs it.
    ///
    /// The cases are chosen for where a chunked prefix sum goes wrong: rows whose lengths
    /// vary (so a chunk's byte total is not its row count times a constant), empty rows
    /// (zero-length spans at a chunk boundary), and nulls (whose validity is carried
    /// separately from the offsets).
    #[test]
    fn the_parallel_gather_equals_arrow_row_for_row() {
        let rows = PARALLEL_TAKE_MIN_ROWS + TAKE_CHUNK_ROWS + 7; // not a chunk multiple
        let source: Vec<Option<String>> = (0..rows)
            .map(|i| match i % 11 {
                0 => None,                    // null row
                1 => Some(String::new()),     // empty row
                k => Some("x".repeat(k * 3)), // varying width
            })
            .collect();
        let a = StringArray::from(source);
        // A scattered permutation, so no chunk reads a contiguous source range.
        let indices: Vec<u32> = (0..rows).map(|i| ((i * 7919) % rows) as u32).collect();
        assert_matches_arrow(&a, &idx(&indices));

        // Ascending indices too: the chunk bases must be right even when the gather is a
        // straight copy, which is the shape a sort of already-ordered data produces.
        let ascending: Vec<u32> = (0..rows as u32).collect();
        assert_matches_arrow(&a, &idx(&ascending));

        // Every index the same row: every chunk total is a multiple of one width, and the
        // whole output is one repeated value — the degenerate prefix sum.
        let repeated: Vec<u32> = vec![5u32; rows];
        assert_matches_arrow(&a, &idx(&repeated));
    }

    /// The fixed-width parallel gather must equal arrow's `take` exactly, for every shape a
    /// chunked fill can get wrong.
    ///
    /// This is the only test that reaches [`take_fixed_width_parallel`]: below
    /// `PARALLEL_TAKE_MIN_ROWS` `take_column` hands every primitive straight to arrow, so a
    /// broken chunk carve would be invisible to the rest of this module. The cases are a
    /// scattered permutation (no chunk reads a contiguous source range), a straight ascending
    /// copy, one repeated index, and a nullable source — validity travels beside the values
    /// rather than in them, so it is the half a fill loop can silently drop.
    #[test]
    fn the_parallel_fixed_width_gather_equals_arrow_row_for_row() {
        let rows = PARALLEL_TAKE_MIN_ROWS + TAKE_CHUNK_ROWS + 7; // not a chunk multiple
        let dense = Int64Array::from((0..rows as i64).map(|i| i * 3 - 7).collect::<Vec<_>>());
        let nullable = Int64Array::from(
            (0..rows)
                .map(|i| (i % 9 != 0).then_some(i as i64))
                .collect::<Vec<Option<i64>>>(),
        );
        let floats = Float64Array::from((0..rows).map(|i| i as f64 * 0.5).collect::<Vec<f64>>());
        // A timestamp carries a timezone in its `DataType` that the values do not, so it is
        // the type that proves the gathered array keeps its full type and not just its width.
        let stamped = TimestampMicrosecondArray::from((0..rows as i64).collect::<Vec<_>>())
            .with_timezone("UTC");
        let scattered: Vec<u32> = (0..rows).map(|i| ((i * 7919) % rows) as u32).collect();
        let ascending: Vec<u32> = (0..rows as u32).collect();
        let repeated: Vec<u32> = vec![5u32; rows];
        for indices in [&scattered, &ascending, &repeated] {
            assert_matches_arrow(&dense, &idx(indices));
            assert_matches_arrow(&nullable, &idx(indices));
            assert_matches_arrow(&floats, &idx(indices));
            assert_matches_arrow(&stamped, &idx(indices));
        }
    }

    /// The chunk-and-concat fallback must reproduce arrow's single-shot `take` exactly, and the
    /// types that cannot take it must not.
    ///
    /// Exactness rather than multiset equality is the point: this preserves join and sort output
    /// order, and a `LIMIT` above turns any reordering into wrong rows rather than a slow query.
    /// `Decimal128` is the flat type it claims (and one whose `DataType` carries a precision and
    /// scale that must survive the concat); a **dictionary** is the type it must decline, because
    /// `concat` may unify the chunks' dictionaries into an encoding a single `take` would never
    /// produce. Both are checked the same way — against arrow — so the decline is proved by the
    /// answer rather than asserted about the type list.
    #[test]
    fn the_chunked_fallback_equals_arrow_row_for_row() {
        let rows = PARALLEL_TAKE_MIN_ROWS + TAKE_CHUNK_ROWS + 7;
        let indices = idx(&(0..rows)
            .map(|i| ((i * 7919) % rows) as u32)
            .collect::<Vec<u32>>());

        let decimals = Decimal128Array::from(
            (0..rows)
                .map(|i| (i % 17 != 0).then_some(i as i128 * 101))
                .collect::<Vec<Option<i128>>>(),
        )
        .with_precision_and_scale(20, 4)
        .expect("in-range precision");
        assert_matches_arrow(&decimals, &indices);
        assert_eq!(decimals.data_type(), &DataType::Decimal128(20, 4));

        let words: Vec<String> = (0..23).map(|i| format!("v{i}")).collect();
        let dict: DictionaryArray<Int32Type> = (0..rows).map(|i| words[i % 23].as_str()).collect();
        assert_matches_arrow(&dict, &indices);
    }

    /// The plane-addressed string gather must equal `arrow::compute::interleave` row for row.
    ///
    /// Sized past a chunk boundary and given varying widths, empty rows and nulls, because the
    /// per-chunk prefix sum is what a fixed-width test cannot exercise: with every row the same
    /// length a chunk's byte total is its row count times a constant, and a wrong base would
    /// still land on a value boundary. Sources of *different lengths* matter too — a row's
    /// coordinates name an array and a row inside it, and a single flattened offset would read
    /// the wrong array without ever going out of bounds.
    #[test]
    fn the_plane_addressed_utf8_gather_equals_interleave() {
        let a = StringArray::from(vec![Some("alpha"), None, Some(""), Some("dd")]);
        let b = StringArray::from(vec![Some("z"), Some("yyyyyyyy"), None]);
        let c = StringArray::from(
            (0..TAKE_CHUNK_ROWS + 5)
                .map(|i| (i % 6 != 0).then(|| "q".repeat(i % 23)))
                .collect::<Vec<Option<String>>>(),
        );
        let lens = [a.len(), b.len(), c.len()];
        let rows = TAKE_CHUNK_ROWS * 2 + 3;
        let part_of: Vec<u32> = (0..rows).map(|i| (i % 3) as u32).collect();
        let row_of: Vec<u32> = (0..rows)
            .map(|i| ((i * 7919) % lens[i % 3]) as u32)
            .collect();

        let cols: Vec<&dyn Array> = vec![&a, &b, &c];
        let pairs: Vec<(usize, usize)> = part_of
            .iter()
            .zip(&row_of)
            .map(|(&p, &r)| (p as usize, r as usize))
            .collect();
        let want = arrow::compute::interleave(&cols, &pairs).unwrap();
        let got = gather_bytes(&cols, &part_of, &row_of).expect("the string plane path");
        assert_eq!(want.as_ref(), got.as_ref());
    }

    /// Anything the plane-addressed gather does not model must decline rather than guess.
    #[test]
    fn the_plane_addressed_gather_declines_what_it_does_not_model() {
        let s = StringArray::from(vec!["a", "b"]);
        let i = Int64Array::from(vec![1, 2]);
        let l = LargeStringArray::from(vec!["a", "b"]);
        let planes = ([0u32, 1], [0u32, 1]);
        // A non-string column, and a mixed Utf8/LargeUtf8 set, both fall back to `interleave`.
        assert!(gather_bytes(&[&i, &i], &planes.0, &planes.1).is_none());
        assert!(gather_bytes(&[&s, &l], &planes.0, &planes.1).is_none());
        assert!(gather_bytes(&[], &[], &[]).is_none());
        // The types it does model still answer.
        assert!(gather_bytes(&[&s, &s], &planes.0, &planes.1).is_some());
        assert!(gather_bytes(&[&l, &l], &planes.0, &planes.1).is_some());
    }

    /// The parallel and serial paths must agree with each other, not just each with arrow.
    ///
    /// Both are driven directly on the same input, so a divergence surfaces as a diff between
    /// the two implementations rather than being attributed to arrow. This is the test that
    /// makes the threshold a *scheduling* choice: crossing it must not change the answer.
    #[test]
    fn the_parallel_and_serial_gathers_agree() {
        for rows in [
            PARALLEL_TAKE_MIN_ROWS + 1_000,
            TAKE_CHUNK_ROWS,     // exactly one chunk
            TAKE_CHUNK_ROWS * 3, // an exact chunk multiple
            TAKE_CHUNK_ROWS + 1, // a one-row final chunk
            1,
            0,
        ] {
            let source: Vec<Option<String>> = (0..rows.max(1))
                .map(|i| (i % 13 != 0).then(|| "ab".repeat(i % 17)))
                .collect();
            let a = StringArray::from(source);
            let n = a.len();
            let indices = idx(&(0..rows)
                .map(|i| ((i * 4099) % n) as u32)
                .collect::<Vec<u32>>());
            let parallel = take_bytes_parallel::<Utf8Type>(&a, &indices).expect("parallel path");
            let serial = take_bytes_serial::<Utf8Type>(&a, &indices).expect("serial path");
            assert_eq!(serial, parallel, "{rows} rows");
        }
    }

    /// A gather whose look-ahead lands on an empty *last* row.
    ///
    /// That row's start offset equals the value buffer's length, so computing its byte address
    /// yields a pointer one past the end. Indexing there panics even though nothing is read —
    /// the prefetch is a hint, but the address arithmetic in front of it is not. The index list
    /// has to be longer than the prefetch distance for the look-ahead to fire at all, which is
    /// why this needs more than a handful of rows.
    #[test]
    fn gathers_an_empty_trailing_row_from_inside_the_prefetch_window() {
        let mut vals: Vec<&str> = vec!["abc"; PREFETCH_OFFSET_DISTANCE + 4];
        *vals.last_mut().unwrap() = ""; // empty last row: start offset == values.len()
        let a = StringArray::from(vals.clone());
        let last = (vals.len() - 1) as u32;
        // Every position in the list points at the empty tail, so it is reached at every
        // prefetch distance as well as directly.
        let indices: Vec<u32> = (0..vals.len() as u32).map(|_| last).collect();
        assert_matches_arrow(&a, &idx(&indices));
        // And a mixed list that walks up to the tail.
        let walking: Vec<u32> = (0..vals.len() as u32).collect();
        assert_matches_arrow(&a, &idx(&walking));
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

    /// Report the byte-array gather against arrow's `take`, per element type and size.
    ///
    /// The end-to-end sort A/B could not resolve this change on a shared machine: the same
    /// build measured 0.71x and 1.57x on consecutive runs, and a control column that the
    /// change cannot touch moved as much as the columns it does. This measures the one
    /// function instead, where the signal is not buried under a query.
    ///
    /// `binary` is the row to read. Before the fast paths were made generic over the layout it
    /// fell to [`take_chunked`] — arrow's `take` per chunk, then a `concat` of the chunks —
    /// which is parallel but writes every value **twice**. That is why the gain is a factor
    /// rather than the order of magnitude the serial-to-parallel string case saw.
    ///
    /// The `ratio` column is **chunked over batcher** — the change this measures — with arrow's
    /// single-shot `take` printed beside it for scale. Gathering a full permutation:
    ///
    /// | Rows | Value width | `Binary` | `LargeBinary` |
    /// |---|---|---|---|
    /// | 1,000,000 | 10 B | 1.10x | 1.14x |
    /// | 1,000,000 | 90 B | 1.17x | 1.27x |
    /// | 4,000,000 | 10 B | 0.88x | 1.16x |
    /// | 4,000,000 | 90 B | 1.04x | 1.24x |
    ///
    /// Against arrow's `take` the same rows are 10x to 11x. And a gathered `Binary` column
    /// costs about half what the same bytes cost as `Utf8` (79 ms against 145 ms at 4 M x
    /// 90 B), because constructing the result needs no UTF-8 validation pass.
    ///
    /// Two things this table's first draft got wrong, both worth not repeating. It built the
    /// binary columns nullable and the text columns not, and read the `NullBuffer::from_iter`
    /// that cost as a property of the offset width — reporting `LargeBinary` at 0.52x, a
    /// regression that did not exist. And it averaged its repeats on a machine with forty other
    /// processes on it, where a mean measures the neighbours and a minimum measures the work.
    ///
    /// `cargo test --release -p bc-runtime --lib -- --ignored --nocapture report_the_byte_gather`
    #[test]
    #[ignore = "measurement, not an assertion"]
    fn report_the_byte_gather() {
        use std::time::Instant;

        /// The **fastest** of seven runs, after one warm-up. A shared machine's noise is
        /// one-sided, so a mean measures the neighbours and a minimum measures the work.
        fn time(f: &dyn Fn()) -> f64 {
            f();
            let mut best = f64::INFINITY;
            for _ in 0..7 {
                let start = Instant::now();
                f();
                best = best.min(start.elapsed().as_secs_f64() * 1e3);
            }
            best
        }

        println!(
            "{:>9} {:>6} {:>14} {:>12} {:>12} {:>12} {:>8}",
            "rows", "width", "type", "batcher", "chunked", "arrow", "ratio"
        );
        for rows in [100_000usize, 1_000_000, 4_000_000] {
            for width in [10usize, 90] {
                let payload = vec![0xabu8; width];
                // Built **without** a null buffer, like the text columns below: a present
                // all-valid buffer costs a 4M-bit `NullBuffer::from_iter` in every gather, and
                // an earlier draft of this table compared nullable binary against non-nullable
                // text and read the difference as a property of the offset width.
                let values: Vec<&[u8]> = (0..rows).map(|_| &payload[..]).collect();
                let indices = UInt32Array::from(
                    (0..rows)
                        .map(|i| ((i * 7919) % rows) as u32)
                        .collect::<Vec<u32>>(),
                );
                let text: String = "x".repeat(width);
                let utf8: ArrayRef =
                    Arc::new(arrow::array::StringArray::from(vec![text.as_str(); rows]));
                let columns: Vec<(&str, ArrayRef)> = vec![
                    (
                        "binary",
                        Arc::new(BinaryArray::from_iter_values(values.iter())),
                    ),
                    (
                        "large_binary",
                        Arc::new(LargeBinaryArray::from_iter_values(values.iter())),
                    ),
                    ("utf8", utf8),
                    (
                        "large_utf8",
                        Arc::new(LargeStringArray::from(vec![text.as_str(); rows])),
                    ),
                    (
                        "fixed_size_binary",
                        Arc::new(
                            arrow::array::FixedSizeBinaryArray::try_from_iter(
                                values.iter().copied(),
                            )
                            .expect("uniform width"),
                        ),
                    ),
                ];
                for (name, col) in columns {
                    let ours = time(&|| {
                        std::hint::black_box(take_column(col.as_ref(), &indices).unwrap());
                    });
                    // What a byte column got before the fast paths covered its layout: arrow's
                    // `take` per chunk across cores, then a `concat` of the chunks. Parallel,
                    // and it writes every value twice — which is the cost the single-pass fill
                    // removes, and the honest baseline for this change.
                    let chunked = time(&|| {
                        std::hint::black_box(take_chunked(col.as_ref(), &indices));
                    });
                    let theirs = time(&|| {
                        std::hint::black_box(take(col.as_ref(), &indices, None).unwrap());
                    });
                    // The fast path must be arrow's answer, or the ratio is meaningless.
                    assert_eq!(
                        take_column(col.as_ref(), &indices).unwrap().as_ref(),
                        take(col.as_ref(), &indices, None).unwrap().as_ref(),
                        "{name} rows={rows} width={width}"
                    );
                    println!(
                        "{rows:>9} {width:>6} {name:>14} {ours:>11.1}ms {chunked:>11.1}ms \
                         {theirs:>11.1}ms {:>7.2}x",
                        chunked / ours
                    );
                }
            }
        }
    }
}
