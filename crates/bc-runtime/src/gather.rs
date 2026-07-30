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
//! `copy_from_slice` the bytes, and carve the output into disjoint slices so the copying runs
//! across cores. For `concat` the destination ranges are known up front, one per input. For
//! `take` they are not — a row's output offset is the running sum of every preceding row's
//! length — so a large gather measures first, prefix-sums the per-chunk totals, and only then
//! copies; that is what turns the same prefix-sum dependency into independent work.
//!
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
    let mut offsets: Vec<O> = vec![O::usize_as(0); total_rows + 1];
    let mut orest = &mut offsets[1..];
    let mut odsts: Vec<&mut [O]> = Vec::with_capacity(arrs.len());
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
                *slot = O::usize_as(base + src[i + 1].as_usize() - start);
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

/// How far ahead [`take_strings`] prefetches the *offset* pair for an upcoming row.
///
/// Far enough that the fetch completes before the loop arrives — a last-level miss is on the
/// order of 200-300 cycles and one iteration is a handful — and near enough that the line is
/// not evicted before use.
///
/// Measured (5 M rows, 302 MB of URL-width strings, random permutation, Cascade Lake): the
/// prefetch alone is worth **982 ms → 522 ms, 1.88x** on one core, and the distance itself
/// barely matters — every value swept from 16 to 96 landed inside the run-to-run noise band.
/// So this is not a tuned constant to be preserved, it is a point on a plateau; what matters
/// is that the loads are issued early at all. With [`take_strings_parallel`] on top, the same
/// gather is **133 ms**, a 7.4x total.
const PREFETCH_OFFSET_DISTANCE: usize = 32;

/// How far ahead [`take_strings`] prefetches the *bytes* of an upcoming row.
///
/// Half the offset distance, because computing a row's byte address requires its offset pair
/// to have landed first. At this point the far prefetch has already brought that pair in, so
/// reading it here is a cache hit rather than the miss it would be at the head of the loop.
const PREFETCH_VALUE_DISTANCE: usize = 16;

/// Rows above which the gather is worth spreading across cores.
///
/// Below it the three-phase parallel form loses to the single pass: it allocates a per-row
/// scratch vector and walks the rows twice, which only pays back once the copying is large
/// enough to hide it. One morsel's worth of rows is not; several are.
const PARALLEL_TAKE_MIN_ROWS: usize = 1 << 17;

/// Rows per parallel gather chunk.
///
/// Small enough that a 96-core box gets real work-stealing granularity on a few-million-row
/// gather, large enough that the per-chunk bookkeeping is noise against the bytes copied.
const TAKE_CHUNK_ROWS: usize = 1 << 14;

/// Gather `arr`'s rows at `indices`, spreading the work across cores once it is large enough.
///
/// `None` when the gathered characters would overflow the offset type, so the caller falls
/// back to arrow's `take` (which widens or errors as it sees fit) rather than wrapping. Both
/// paths return exactly the same relation — the split is scheduling, and
/// `the_parallel_and_serial_gathers_agree` holds them to it.
fn take_strings<O: OffsetSizeTrait>(
    arr: &GenericStringArray<O>,
    indices: &UInt32Array,
) -> Option<GenericStringArray<O>> {
    if indices.len() >= PARALLEL_TAKE_MIN_ROWS && rayon::current_num_threads() > 1 {
        take_strings_parallel(arr, indices)
    } else {
        take_strings_serial(arr, indices)
    }
}

/// The single-pass gather: one loop, one output buffer, no scratch.
///
/// The reference the parallel path is checked against, and the path a morsel-sized gather
/// actually takes — below a few hundred thousand rows the extra measure pass costs more than
/// the cores buy back.
fn take_strings_serial<O: OffsetSizeTrait>(
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

/// [`take_strings`] across cores, for a gather large enough to pay for the extra pass.
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
///    safe-by-construction carve `concat_strings` uses, with no `unsafe` anywhere.
///
/// Returns `None` on offset overflow, exactly like the serial path, so the caller still falls
/// back to arrow rather than wrapping.
fn take_strings_parallel<O: OffsetSizeTrait>(
    arr: &GenericStringArray<O>,
    indices: &UInt32Array,
) -> Option<GenericStringArray<O>> {
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
    O::from_usize(total_bytes)?;

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
    let mut offsets: Vec<O> = vec![O::usize_as(0); n + 1];
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
                *slot = O::usize_as(base + at);
            }
        });

    // A gathered row is null exactly when its source row is — identical to the serial path.
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

    /// The parallel gather must equal arrow's `take` exactly, not merely as a multiset.
    ///
    /// Every other test in this module is below `PARALLEL_TAKE_MIN_ROWS`, so none of them
    /// reaches `take_strings_parallel` at all — the three-phase path could be arbitrarily
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
            let parallel = take_strings_parallel::<i32>(&a, &indices).expect("parallel path");
            let serial = take_strings_serial::<i32>(&a, &indices).expect("serial path");
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
}
