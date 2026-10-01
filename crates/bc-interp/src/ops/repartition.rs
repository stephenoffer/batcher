//! Hash-partition a relation held as morsels, gathering each row exactly **once**.
//!
//! The shuffle join used to `materialize` its probe side into one batch and then
//! `partition_by_keys` it, which gathers every row again: two full copies of the query's
//! largest relation, back to back. `interleave` can gather from many source arrays at once,
//! so the concatenation is unnecessary — each bucket is built directly from the morsels.
//!
//! **The result is identical to partitioning the concatenated relation.** A row's bucket is a
//! deterministic function of its key value (`shuffle::bucket_of_rows`), so it lands in the
//! same bucket whichever morsel carries it; and the gather visits morsels in order and rows in
//! order within a morsel, so each bucket holds its rows in the relation's original order —
//! exactly what a single `scatter_into_buckets` over the concatenated batch produces. The
//! per-bucket join and the `seq == par` oracle both depend on that.
//!
//! The buckets stay **contiguous** — one `RecordBatch` each. That is the difference between
//! this and the obvious "partition each morsel independently" approach, which was tried and
//! reverted: it leaves each bucket holding one small piece per morsel (366 pieces of ~170 rows
//! at sf10), and the per-piece overhead of the downstream join swamps the copy it saved.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, ArrowPrimitiveType, GenericByteArray, PrimitiveArray, RecordBatch,
};
use arrow::buffer::OffsetBuffer;
use arrow::compute::interleave;
use arrow::datatypes::{
    ArrowNativeType, BinaryType, ByteArrayType, DataType, Date32Type, Date64Type, Float32Type,
    Float64Type, Int16Type, Int32Type, Int64Type, Int8Type, LargeBinaryType, LargeUtf8Type,
    TimeUnit, TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
    TimestampSecondType, UInt16Type, UInt32Type, UInt64Type, UInt8Type, Utf8Type,
};
use bc_runtime::shuffle;
use rayon::prelude::*;

use crate::error::InterpError;
use crate::ops::columns_by_name;

/// Hash-partition `batches` into `parts` buckets by `keys`, one contiguous batch per bucket.
///
/// A single bucket needs no gather at all — the relation is concatenated as-is, which is what
/// the caller would have done anyway.
pub(crate) fn partition_morsels(
    batches: &[RecordBatch],
    keys: &[String],
    parts: usize,
) -> Result<Vec<RecordBatch>, InterpError> {
    partition_morsels_with(batches, parts, |b| columns_by_name(b, keys), 0)
}

/// [`partition_morsels`] keyed by column *index* — the form the distributed shuffle
/// speaks (`dist::partition_batches` receives key indices, not names).
/// Hash-partition morsels by the key columns at `key_indices`, re-mixing the hash with
/// `salt`.
///
/// `salt == 0` is the cluster-wide bucket assignment, which a shuffle must never perturb. A
/// non-zero salt exists for **re-splitting a bucket that did not fit**, where re-using the
/// unsalted hash is not merely suboptimal but inert: `bucket_of` reads the low bits at a
/// power-of-two bucket count, so re-partitioning a 16-way bucket into 8 sub-buckets sends
/// every row to `bucket & 7` — one sub-bucket, always. The re-partition writes and re-reads
/// the whole bucket and changes nothing.
pub(crate) fn partition_morsels_by_index_salted(
    batches: &[RecordBatch],
    key_indices: &[usize],
    parts: usize,
    salt: u64,
) -> Result<Vec<RecordBatch>, InterpError> {
    partition_morsels_with(
        batches,
        parts,
        |b| Ok(key_indices.iter().map(|&i| b.column(i).clone()).collect()),
        salt,
    )
}

/// The shared body: everything but *how a morsel's key columns are selected* is
/// independent of whether the caller names its keys or indexes them.
fn partition_morsels_with(
    batches: &[RecordBatch],
    parts: usize,
    key_cols_of: impl Fn(&RecordBatch) -> Result<Vec<ArrayRef>, InterpError> + Sync,
    salt: u64,
) -> Result<Vec<RecordBatch>, InterpError> {
    debug_assert!(parts >= 1);
    if parts == 1 || batches.is_empty() {
        return Ok(vec![crate::ops::materialize(batches)?]);
    }

    // One hash pass per morsel, then that morsel's row ids binned by bucket into a flat
    // CSR array. The `Vec<Vec<u32>>` shape would ask for one growing vector per
    // (morsel, bucket) — ~350k of them here — where the whole step is a single pass.
    let per_morsel: Vec<(Vec<u32>, Vec<u32>)> = batches
        .par_iter()
        .map(|batch| {
            let key_cols = key_cols_of(batch)?;
            let part_of = shuffle::bucket_of_rows_salted(&key_cols, batch.num_rows(), parts, salt)?;
            Ok(shuffle::bucket_csr(&part_of, parts))
        })
        .collect::<Result<_, InterpError>>()?;

    let schema = batches[0].schema();
    let ncols = schema.fields().len();
    // `interleave`'s sources are the same for every bucket; build the pointer table once,
    // and decide once per column how it will be moved (`plan_column`).
    let sources: Vec<Vec<&dyn Array>> = (0..ncols)
        .map(|c| batches.iter().map(|b| b.column(c).as_ref()).collect())
        .collect();
    let plans: Vec<ColGather> = sources.iter().map(|s| plan_column(s)).collect();
    let layout = Layout::new(&per_morsel, parts, rayon::current_num_threads());

    // Column-major: each column is scattered into every bucket at once (see `Layout`), so a
    // column's result is one array per bucket. Transposed into one batch per bucket below.
    let by_column: Vec<Vec<ArrayRef>> = plans
        .par_iter()
        .zip(&sources)
        .map(|(plan, src)| match plan {
            ColGather::Fast(cols) => Ok(cols.scatter(&per_morsel, &layout)),
            ColGather::Bytes(cols) => cols.scatter(&per_morsel, &layout),
            ColGather::Interleave => (0..parts)
                .into_par_iter()
                .map(|bucket| {
                    let mut pairs = Vec::with_capacity(layout.totals[bucket]);
                    for (morsel, (rows, off)) in per_morsel.iter().enumerate() {
                        pairs.extend(
                            rows[off[bucket] as usize..off[bucket + 1] as usize]
                                .iter()
                                .map(|&row| (morsel, row as usize)),
                        );
                    }
                    interleave(src, &pairs).map_err(InterpError::from)
                })
                .collect(),
        })
        .collect::<Result<_, InterpError>>()?;

    let mut columns: Vec<std::vec::IntoIter<ArrayRef>> =
        by_column.into_iter().map(Vec::into_iter).collect();
    (0..parts)
        .map(|_| {
            let cols: Vec<ArrayRef> = columns
                .iter_mut()
                .map(|c| c.next().expect("one array per bucket"))
                .collect();
            RecordBatch::try_new(schema.clone(), cols).map_err(InterpError::from)
        })
        .collect()
}

/// Where every scatter task writes: the morsels cut into contiguous **chunks**, and each
/// chunk's row count per bucket.
///
/// The gather used to be *bucket-major* — one task per bucket, pulling that bucket's rows out
/// of every morsel. A bucket owns about one row in `parts`, so each task read every source
/// morsel at a `parts`-row stride: a cache line fetched per 8-byte value, from a relation far
/// larger than any cache. And it copied each string with its own `memmove` call, for values a
/// few bytes long. Measured on H2O `groupby` q10 (10 M rows, six keys, three of them strings,
/// 315 buckets on 15 cores), that gather was **68 % of the query's CPU** — `memmove` alone
/// 33 %, ahead of hashing and the aggregation itself.
///
/// Scattering *chunk-major* turns the reads sequential: a task walks its own morsels in order,
/// each morsel's column resident in cache while its rows are dealt to the buckets, and writes
/// each bucket's rows into a slot reserved for it. The slots are disjoint slices of each
/// bucket's single output buffer, laid out chunk by chunk, so the output is **byte-for-byte
/// what the bucket-major gather produced**: every bucket holds its rows morsels-in-order,
/// rows-in-order within a morsel, contiguous in one batch. The slots are carved with
/// `split_at_mut`, so the parallel writes need no `unsafe` and no synchronisation.
///
/// Chunks rather than morsels, because a slot is one `&mut [T]` per (chunk, bucket): at
/// sf10's 3,663 morsels and 576 buckets per-morsel slots would be two million slice
/// descriptors per column, where a few chunks per worker keep it to tens of thousands. And
/// chunks cut by **rows**, not whole morsels, because the input is not always morsel-sized: a
/// filtered or materialized relation can arrive as a handful of large batches, and a chunk of
/// whole morsels would then leave one task scattering the entire relation while the pool
/// waited. That shape is exactly the shuffle join's (15 buckets over two large batches), and
/// it read 25 % slower than the bucket-major gather until the cut moved inside the batch.
struct Layout {
    /// Contiguous row ranges of the relation, in order — one scatter task each.
    chunks: Vec<Vec<Seg>>,
    /// `rows[c][b]`: rows chunk `c` sends to bucket `b`.
    rows: Vec<Vec<usize>>,
    /// Rows per bucket, over the whole relation.
    totals: Vec<usize>,
}

/// Chunks per worker: enough that an uneven chunk does not leave the pool idle, few enough
/// that the per-(chunk, bucket) slot table stays small.
const CHUNKS_PER_THREAD: usize = 4;

/// Rows below which a chunk is not worth its own task and its own row of slots.
const MIN_CHUNK_ROWS: usize = 4_096;

/// The rows `[lo, hi)` of one morsel: a piece of a chunk.
#[derive(Clone, Copy)]
struct Seg {
    morsel: usize,
    lo: u32,
    hi: u32,
    /// The whole morsel, so a bin needs no trimming to the range.
    whole: bool,
}

impl Layout {
    fn new(per_morsel: &[(Vec<u32>, Vec<u32>)], parts: usize, threads: usize) -> Self {
        // `bucket_csr` keeps one row id per row, so its length is the morsel's row count.
        let total: usize = per_morsel.iter().map(|(rows, _)| rows.len()).sum();
        let want = threads.max(1).saturating_mul(CHUNKS_PER_THREAD);
        let target = total.div_ceil(want).max(MIN_CHUNK_ROWS);
        let mut chunks: Vec<Vec<Seg>> = Vec::new();
        let (mut cur, mut cur_rows) = (Vec::new(), 0usize);
        for (morsel, (rows, _)) in per_morsel.iter().enumerate() {
            let len = rows.len();
            let mut lo = 0usize;
            while lo < len {
                let take = (target - cur_rows).min(len - lo);
                cur.push(Seg {
                    morsel,
                    lo: lo as u32,
                    hi: (lo + take) as u32,
                    whole: take == len,
                });
                (lo, cur_rows) = (lo + take, cur_rows + take);
                if cur_rows >= target {
                    chunks.push(std::mem::take(&mut cur));
                    cur_rows = 0;
                }
            }
        }
        if !cur.is_empty() {
            chunks.push(cur);
        }
        let rows: Vec<Vec<usize>> = chunks
            .par_iter()
            .map(|chunk| {
                let mut counts = vec![0usize; parts];
                for seg in chunk {
                    for (b, count) in counts.iter_mut().enumerate() {
                        *count += bin(per_morsel, seg, b).len();
                    }
                }
                counts
            })
            .collect();
        let totals = column_sums(&rows, parts);
        Layout {
            chunks,
            rows,
            totals,
        }
    }
}

/// Bucket `b`'s rows within `seg`, ascending: the morsel's CSR bin, trimmed to the segment's
/// range when it covers only part of the morsel. A bin is sorted (`bucket_csr` visits rows in
/// order), so the trim is two binary searches, not a scan.
fn bin<'p>(per_morsel: &'p [(Vec<u32>, Vec<u32>)], seg: &Seg, b: usize) -> &'p [u32] {
    let (rows, off) = &per_morsel[seg.morsel];
    let bin = &rows[off[b] as usize..off[b + 1] as usize];
    if seg.whole {
        return bin;
    }
    let start = bin.partition_point(|&r| r < seg.lo);
    let end = bin.partition_point(|&r| r < seg.hi);
    &bin[start..end]
}

/// One buffer of `len + extra` copies of `fill` per bucket, allocated **across the pool**.
///
/// Allocated serially, this was the scatter's one regression: a 15-way shuffle join over 3 M
/// rows asks for fifteen 1.6 MB buffers, and zeroing them (and taking their first-touch page
/// faults) on the calling thread put ~18 ms of serial work in front of a parallel copy that
/// takes less. The bucket-major gather never paid it, because each bucket's task allocated
/// its own output. The fill is overwritten in full by the scatter; it exists only so the
/// slots can be carved as initialized memory, with no `unsafe`.
fn zeroed_buffers<T: Copy + Send + Sync>(lens: &[usize], extra: usize, fill: T) -> Vec<Vec<T>> {
    lens.par_iter().map(|&n| vec![fill; n + extra]).collect()
}

/// `sums[b] = Σ_c table[c][b]`.
fn column_sums(table: &[Vec<usize>], parts: usize) -> Vec<usize> {
    let mut sums = vec![0usize; parts];
    for row in table {
        for (s, &x) in sums.iter_mut().zip(row) {
            *s += x;
        }
    }
    sums
}

/// Carve each bucket's buffer (past its first `skip` elements) into one disjoint slot per
/// chunk, sized `sizes[c][b]` and laid out in chunk order. Returned chunk-major — `[c][b]` —
/// so each scatter task owns exactly its own row of slots.
fn carve<'a, T>(
    bufs: &'a mut [Vec<T>],
    sizes: &[Vec<usize>],
    skip: usize,
) -> Vec<Vec<&'a mut [T]>> {
    let mut out: Vec<Vec<&'a mut [T]>> = (0..sizes.len())
        .map(|_| Vec::with_capacity(bufs.len()))
        .collect();
    for (b, buf) in bufs.iter_mut().enumerate() {
        let mut rest: &'a mut [T] = &mut buf.as_mut_slice()[skip..];
        for (c, slots) in out.iter_mut().enumerate() {
            let (slot, tail) = std::mem::take(&mut rest).split_at_mut(sizes[c][b]);
            slots.push(slot);
            rest = tail;
        }
        debug_assert!(rest.is_empty(), "slots must tile the bucket exactly");
    }
    out
}

/// Scatter one primitive column into every bucket — see [`Layout`].
///
/// `interleave` needs a materialized `&[(usize, usize)]` — **sixteen bytes of index per
/// output row** — where the row ids already exist as `u32` in the CSR bins, so this reads them
/// in place and writes the output directly.
fn scatter_prim<T: ArrowPrimitiveType>(
    cols: &[&PrimitiveArray<T>],
    per_morsel: &[(Vec<u32>, Vec<u32>)],
    layout: &Layout,
) -> Vec<ArrayRef> {
    let parts = layout.totals.len();
    let mut bufs: Vec<Vec<T::Native>> = zeroed_buffers(&layout.totals, 0, T::Native::default());
    carve(&mut bufs, &layout.rows, 0)
        .into_par_iter()
        .zip(layout.chunks.par_iter())
        .for_each(|(mut slots, chunk)| {
            let mut cursor = vec![0usize; parts];
            for seg in chunk {
                let values = cols[seg.morsel].values();
                for (b, (slot, at)) in slots.iter_mut().zip(cursor.iter_mut()).enumerate() {
                    let bin = bin(per_morsel, seg, b);
                    for (dst, &r) in slot[*at..*at + bin.len()].iter_mut().zip(bin) {
                        *dst = values[r as usize];
                    }
                    *at += bin.len();
                }
            }
        });
    bufs.into_iter()
        .map(|v| Arc::new(PrimitiveArray::<T>::new(v.into(), None)) as ArrayRef)
        .collect()
}

/// Values at most this long are copied as one fixed-width move rather than a `memmove`
/// call. Most group and join keys are short, and a call per few-byte string was the single
/// largest cost of the old gather — see [`Layout`].
const SHORT_COPY: usize = 16;

/// Copy `src[s..e]` to `dst[at..]`, over-writing up to [`SHORT_COPY`] bytes past the value
/// when both sides have room. The over-written tail lies inside the caller's own slot and is
/// overwritten by the values that follow it, so only the exact `e - s` bytes survive.
#[inline(always)]
fn copy_value(dst: &mut [u8], at: usize, src: &[u8], s: usize, e: usize) {
    let len = e - s;
    if len <= SHORT_COPY && at + SHORT_COPY <= dst.len() && s + SHORT_COPY <= src.len() {
        dst[at..at + SHORT_COPY].copy_from_slice(&src[s..s + SHORT_COPY]);
    } else {
        dst[at..at + len].copy_from_slice(&src[s..e]);
    }
}

/// Scatter one null-free byte column into every bucket — the [`scatter_prim`] argument at the
/// type where it is worth the most.
///
/// Two passes per chunk over the same (cache-resident) morsels: the first sums each
/// (chunk, bucket)'s value bytes, which fixes every slot of every bucket's value buffer; the
/// second writes the offsets and the bytes together. The output is the array `interleave`
/// produces — same rows, same order. An offset that would outgrow `T::Offset` is an error, not
/// a wrapped value; the bytes are not re-validated as UTF-8, since each is a copy of a value
/// that already was (checked in debug builds).
fn scatter_bytes<T: ByteArrayType>(
    cols: &[&GenericByteArray<T>],
    per_morsel: &[(Vec<u32>, Vec<u32>)],
    layout: &Layout,
) -> Result<Vec<ArrayRef>, InterpError> {
    let parts = layout.totals.len();
    let bytes: Vec<Vec<usize>> = layout
        .chunks
        .par_iter()
        .map(|chunk| {
            let mut counts = vec![0usize; parts];
            for seg in chunk {
                let src = cols[seg.morsel].value_offsets();
                for (b, count) in counts.iter_mut().enumerate() {
                    for &r in bin(per_morsel, seg, b) {
                        *count += src[r as usize + 1].as_usize() - src[r as usize].as_usize();
                    }
                }
            }
            counts
        })
        .collect();
    let byte_totals = column_sums(&bytes, parts);
    // Each slot's first absolute byte position within its bucket's value buffer.
    let mut base: Vec<Vec<usize>> = Vec::with_capacity(bytes.len());
    let mut running = vec![0usize; parts];
    for counts in &bytes {
        base.push(running.clone());
        for (r, &x) in running.iter_mut().zip(counts) {
            *r += x;
        }
    }
    if let Some(&too_big) = byte_totals
        .iter()
        .find(|&&n| T::Offset::from_usize(n).is_none())
    {
        return Err(InterpError::Arrow(
            arrow::error::ArrowError::OffsetOverflowError(too_big),
        ));
    }

    let mut data: Vec<Vec<u8>> = zeroed_buffers(&byte_totals, 0, 0u8);
    let mut offsets: Vec<Vec<T::Offset>> =
        zeroed_buffers(&layout.totals, 1, T::Offset::usize_as(0));
    carve(&mut data, &bytes, 0)
        .into_par_iter()
        .zip(carve(&mut offsets, &layout.rows, 1))
        .zip(layout.chunks.par_iter().zip(&base))
        .for_each(|((mut dslots, mut oslots), (chunk, base))| {
            let mut row_at = vec![0usize; parts];
            let mut byte_at = vec![0usize; parts];
            for seg in chunk {
                let src = cols[seg.morsel].value_offsets();
                let values = cols[seg.morsel].value_data();
                for b in 0..parts {
                    let (dslot, oslot) = (&mut *dslots[b], &mut *oslots[b]);
                    let (mut k, mut at) = (row_at[b], byte_at[b]);
                    for &r in bin(per_morsel, seg, b) {
                        let (s, e) = (src[r as usize].as_usize(), src[r as usize + 1].as_usize());
                        copy_value(dslot, at, values, s, e);
                        at += e - s;
                        oslot[k] = T::Offset::usize_as(base[b] + at);
                        k += 1;
                    }
                    (row_at[b], byte_at[b]) = (k, at);
                }
            }
        });

    data.into_iter()
        .zip(offsets)
        .map(|(data, offsets)| {
            // `OffsetBuffer::new` still checks the offsets are monotone; what is skipped is the
            // UTF-8 re-validation of every gathered byte, which measured ~22% of ClickBench
            // q14/q30/q31 (string group keys through this partition).
            let offsets = OffsetBuffer::new(offsets.into());
            let data: arrow::buffer::Buffer = data.into();
            debug_assert!(
                GenericByteArray::<T>::try_new(offsets.clone(), data.clone(), None).is_ok(),
                "a scattered byte column must be what its sources were"
            );
            // SAFETY: every value in `data` is a byte-exact copy of one value of a source array
            // of this same type `T` (`copy_value` moves exactly `src[s..e]` for a source row),
            // and the offsets are the running sums of those value lengths, so each offset falls
            // on a value boundary. The sources are valid arrays (UTF-8 for a string type), so
            // the result is too; the offset width was checked against `T::Offset` above.
            Ok(
                Arc::new(unsafe { GenericByteArray::<T>::new_unchecked(offsets, data, None) })
                    as ArrayRef,
            )
        })
        .collect()
}

/// One column's source arrays, downcast once for the whole partition.
///
/// The downcast is per (column, morsel) — 3,663 morsels here — and the gather runs per
/// (column, bucket). Doing the downcast *inside* the bucket loop makes it
/// `buckets × columns × morsels`, which is invisible at 96 buckets and costs 20% of TPC-H
/// Q9 at 576. It depends only on the column, so it is hoisted to exactly that.
enum ColGather<'a> {
    Fast(FastCols<'a>),
    /// A null-free string/binary column: scattered from the CSR bins, like [`ColGather::Fast`].
    Bytes(ByteCols<'a>),
    /// A nested type, or any source carrying a null: `interleave` owns it.
    Interleave,
}

macro_rules! byte_cols {
    ($($variant:ident => $ty:ty),* $(,)?) => {
        /// The concrete byte types the CSR scatter handles, downcast once per column.
        enum ByteCols<'a> { $($variant(Vec<&'a GenericByteArray<$ty>>)),* }

        impl ByteCols<'_> {
            fn scatter(
                &self,
                per_morsel: &[(Vec<u32>, Vec<u32>)],
                layout: &Layout,
            ) -> Result<Vec<ArrayRef>, InterpError> {
                match self {
                    $(ByteCols::$variant(cols) => scatter_bytes(cols, per_morsel, layout)),*
                }
            }
        }

        /// Downcast every source of one byte column, or `None` if any is not `$ty`.
        fn downcast_bytes<'a, T: ByteArrayType>(
            sources: &[&'a dyn Array],
        ) -> Option<Vec<&'a GenericByteArray<T>>> {
            sources
                .iter()
                .map(|a| a.as_any().downcast_ref::<GenericByteArray<T>>())
                .collect()
        }
    };
}

byte_cols! {
    Utf8 => Utf8Type, LargeUtf8 => LargeUtf8Type,
    Binary => BinaryType, LargeBinary => LargeBinaryType,
}

macro_rules! fast_cols {
    ($($variant:ident => $ty:ty),* $(,)?) => {
        /// The concrete primitive types the flat scatter handles, downcast once per column.
        enum FastCols<'a> { $($variant(Vec<&'a PrimitiveArray<$ty>>)),* }

        impl FastCols<'_> {
            fn scatter(
                &self,
                per_morsel: &[(Vec<u32>, Vec<u32>)],
                layout: &Layout,
            ) -> Vec<ArrayRef> {
                match self {
                    $(FastCols::$variant(cols) => scatter_prim(cols, per_morsel, layout)),*
                }
            }
        }

        /// Downcast every source of one column, or `None` if any is not `$ty`.
        fn downcast_all<'a, T: ArrowPrimitiveType>(
            sources: &[&'a dyn Array],
        ) -> Option<Vec<&'a PrimitiveArray<T>>> {
            sources
                .iter()
                .map(|a| a.as_any().downcast_ref::<PrimitiveArray<T>>())
                .collect()
        }
    };
}

fast_cols! {
    I8 => Int8Type, I16 => Int16Type, I32 => Int32Type, I64 => Int64Type,
    U8 => UInt8Type, U16 => UInt16Type, U32 => UInt32Type, U64 => UInt64Type,
    F32 => Float32Type, F64 => Float64Type,
    D32 => Date32Type, D64 => Date64Type,
    TsS => TimestampSecondType, TsMs => TimestampMillisecondType,
    TsUs => TimestampMicrosecondType, TsNs => TimestampNanosecondType,
}

/// Decide once, per column, how its rows will be moved into each bucket.
fn plan_column<'a>(sources: &[&'a dyn Array]) -> ColGather<'a> {
    if sources.iter().any(|a| a.null_count() > 0) {
        return ColGather::Interleave; // a null buffer to rebuild: `interleave` owns it
    }
    let Some(dtype) = sources.first().map(|a| a.data_type()) else {
        return ColGather::Interleave;
    };
    macro_rules! fast {
        ($variant:ident, $ty:ty) => {
            match downcast_all::<$ty>(sources) {
                Some(cols) => ColGather::Fast(FastCols::$variant(cols)),
                None => ColGather::Interleave,
            }
        };
    }
    macro_rules! bytes {
        ($variant:ident, $ty:ty) => {
            match downcast_bytes::<$ty>(sources) {
                Some(cols) => ColGather::Bytes(ByteCols::$variant(cols)),
                None => ColGather::Interleave,
            }
        };
    }
    match dtype {
        DataType::Utf8 => bytes!(Utf8, Utf8Type),
        DataType::LargeUtf8 => bytes!(LargeUtf8, LargeUtf8Type),
        DataType::Binary => bytes!(Binary, BinaryType),
        DataType::LargeBinary => bytes!(LargeBinary, LargeBinaryType),
        DataType::Int8 => fast!(I8, Int8Type),
        DataType::Int16 => fast!(I16, Int16Type),
        DataType::Int32 => fast!(I32, Int32Type),
        DataType::Int64 => fast!(I64, Int64Type),
        DataType::UInt8 => fast!(U8, UInt8Type),
        DataType::UInt16 => fast!(U16, UInt16Type),
        DataType::UInt32 => fast!(U32, UInt32Type),
        DataType::UInt64 => fast!(U64, UInt64Type),
        DataType::Float32 => fast!(F32, Float32Type),
        DataType::Float64 => fast!(F64, Float64Type),
        DataType::Date32 => fast!(D32, Date32Type),
        DataType::Date64 => fast!(D64, Date64Type),
        DataType::Timestamp(TimeUnit::Second, None) => fast!(TsS, TimestampSecondType),
        DataType::Timestamp(TimeUnit::Millisecond, None) => fast!(TsMs, TimestampMillisecondType),
        DataType::Timestamp(TimeUnit::Microsecond, None) => fast!(TsUs, TimestampMicrosecondType),
        DataType::Timestamp(TimeUnit::Nanosecond, None) => fast!(TsNs, TimestampNanosecondType),
        _ => ColGather::Interleave,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, Int64Array, StringArray};
    use std::sync::Arc;

    fn batch(keys: &[i64], vals: &[&str]) -> RecordBatch {
        RecordBatch::try_from_iter(vec![
            ("k", Arc::new(Int64Array::from(keys.to_vec())) as ArrayRef),
            ("v", Arc::new(StringArray::from(vals.to_vec())) as ArrayRef),
        ])
        .unwrap()
    }

    fn rows_of(batches: &[RecordBatch]) -> Vec<(i64, String)> {
        let mut out = Vec::new();
        for b in batches {
            let k = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let v = b.column(1).as_any().downcast_ref::<StringArray>().unwrap();
            for i in 0..b.num_rows() {
                out.push((k.value(i), v.value(i).to_string()));
            }
        }
        out
    }

    /// The invariant everything rests on: partitioning morsels is byte-for-byte what
    /// partitioning the concatenated relation produces — same buckets, same order in each.
    #[test]
    fn matches_partitioning_the_concatenated_relation() {
        let morsels = [
            batch(&[1, 2, 3, 4], &["a", "b", "c", "d"]),
            batch(&[5, 1, 6, 2], &["e", "f", "g", "h"]),
            batch(&[3, 7], &["i", "j"]),
        ];
        let parts = 4;
        let whole = crate::ops::materialize(&morsels).unwrap();
        let expected = shuffle::partition_by_keys(&whole, &[0], parts).unwrap();
        let got = partition_morsels(&morsels, &["k".into()], parts).unwrap();

        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(&expected) {
            assert_eq!(
                rows_of(std::slice::from_ref(g)),
                rows_of(std::slice::from_ref(e))
            );
        }
    }

    /// Equal keys co-partition across morsels — the invariant the per-bucket join needs.
    #[test]
    fn equal_keys_share_a_bucket_across_morsels() {
        let morsels = [
            batch(&[7], &["a"]),
            batch(&[7], &["b"]),
            batch(&[9], &["c"]),
        ];
        let got = partition_morsels(&morsels, &["k".into()], 8).unwrap();
        let holding_seven: Vec<usize> = got
            .iter()
            .enumerate()
            .filter(|(_, b)| {
                rows_of(std::slice::from_ref(b))
                    .iter()
                    .any(|(k, _)| *k == 7)
            })
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            holding_seven.len(),
            1,
            "key 7 must land in exactly one bucket"
        );
        let bucket = &got[holding_seven[0]];
        assert_eq!(rows_of(std::slice::from_ref(bucket)).len(), 2);
    }

    /// Every row is placed exactly once; nothing is dropped or duplicated.
    #[test]
    fn every_row_is_placed_exactly_once() {
        let morsels = [
            batch(&[1, 2, 3], &["a", "b", "c"]),
            batch(&[4, 5], &["d", "e"]),
        ];
        let got = partition_morsels(&morsels, &["k".into()], 3).unwrap();
        let mut all = rows_of(&got);
        all.sort();
        assert_eq!(
            all,
            vec![
                (1, "a".into()),
                (2, "b".into()),
                (3, "c".into()),
                (4, "d".into()),
                (5, "e".into())
            ]
        );
    }

    /// One bucket is the degenerate case: no hashing, no gather.
    #[test]
    fn a_single_bucket_is_the_concatenated_relation() {
        let morsels = [batch(&[1, 2], &["a", "b"]), batch(&[3], &["c"])];
        let got = partition_morsels(&morsels, &["k".into()], 1).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(rows_of(&got).len(), 3);
    }

    /// **A sliced morsel indexes its own rows.** `PrimitiveArray::values()` is offset-
    /// adjusted, but a gather that read the whole backing buffer would silently take the
    /// wrong values. Morsels are almost always slices of a larger batch, so this is the
    /// invariant the fast gather lives or dies on.
    #[test]
    fn a_sliced_morsel_gathers_only_its_own_values() {
        let whole = batch(&[10, 11, 12, 13, 14, 15], &["a", "b", "c", "d", "e", "f"]);
        let morsels = [whole.slice(2, 2), whole.slice(4, 2)]; // keys 12,13 then 14,15
        let got = partition_morsels(&morsels, &["k".into()], 4).unwrap();
        let mut all = rows_of(&got);
        all.sort();
        assert_eq!(
            all,
            vec![
                (12, "c".into()),
                (13, "d".into()),
                (14, "e".into()),
                (15, "f".into())
            ]
        );
    }

    /// A nullable column falls back to `interleave`, and the nulls survive the round trip.
    #[test]
    fn a_column_with_nulls_falls_back_and_keeps_them() {
        let mk = |keys: Vec<i64>, vals: Vec<Option<i64>>| {
            RecordBatch::try_from_iter(vec![
                ("k", Arc::new(Int64Array::from(keys)) as ArrayRef),
                ("v", Arc::new(Int64Array::from(vals)) as ArrayRef),
            ])
            .unwrap()
        };
        let morsels = [mk(vec![1, 2], vec![None, Some(7)]), mk(vec![3], vec![None])];
        let got = partition_morsels(&morsels, &["k".into()], 4).unwrap();
        let mut seen: Vec<(i64, Option<i64>)> = Vec::new();
        for b in &got {
            let k = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let v = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
            for i in 0..b.num_rows() {
                seen.push((k.value(i), (!v.is_null(i)).then(|| v.value(i))));
            }
        }
        seen.sort();
        assert_eq!(seen, vec![(1, None), (2, Some(7)), (3, None)]);
    }

    /// Float and date columns take the fast gather; a string column takes `interleave`.
    /// Both must land in the same bucket batch with the same rows.
    #[test]
    fn mixed_fast_and_fallback_columns_agree() {
        use arrow::array::{Date32Array, Float64Array};
        let mk = |k: Vec<i64>, f: Vec<f64>, d: Vec<i32>, s: Vec<&str>| {
            RecordBatch::try_from_iter(vec![
                ("k", Arc::new(Int64Array::from(k)) as ArrayRef),
                ("f", Arc::new(Float64Array::from(f)) as ArrayRef),
                ("d", Arc::new(Date32Array::from(d)) as ArrayRef),
                ("s", Arc::new(StringArray::from(s)) as ArrayRef),
            ])
            .unwrap()
        };
        let morsels = [
            mk(vec![1, 2], vec![1.5, 2.5], vec![100, 200], vec!["a", "b"]),
            mk(vec![3], vec![3.5], vec![300], vec!["c"]),
        ];
        let got = partition_morsels(&morsels, &["k".into()], 4).unwrap();
        let mut seen = Vec::new();
        for b in &got {
            let k = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let f = b.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
            let d = b.column(2).as_any().downcast_ref::<Date32Array>().unwrap();
            let st = b.column(3).as_any().downcast_ref::<StringArray>().unwrap();
            for i in 0..b.num_rows() {
                seen.push((k.value(i), f.value(i), d.value(i), st.value(i).to_string()));
            }
        }
        seen.sort_by_key(|r| r.0);
        assert_eq!(
            seen,
            vec![
                (1, 1.5, 100, "a".into()),
                (2, 2.5, 200, "b".into()),
                (3, 3.5, 300, "c".into())
            ]
        );
    }

    /// The CSR byte gather must produce exactly what `interleave` produces, on the shapes
    /// where a hand-written offset walk could go wrong: values of differing length, an empty
    /// string, a morsel that is a **slice** of a larger batch (so `value_offsets` is rebased
    /// while `value_data` is not), and a bucket that receives nothing.
    #[test]
    fn the_byte_gather_matches_interleave() {
        let whole: ArrayRef = Arc::new(StringArray::from(vec![
            "",
            "a",
            "bcdef",
            "gh",
            "",
            "ijklmnopqrs",
            "t",
            "uv",
        ]));
        let keys: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 2, 3, 4, 5, 6, 7, 8]));
        let schema = |k: ArrayRef, v: ArrayRef| {
            RecordBatch::try_from_iter(vec![("k", k), ("v", v)]).unwrap()
        };
        // Two morsels, both slices of the source arrays.
        let morsels = [
            schema(keys.slice(0, 5), whole.slice(0, 5)),
            schema(keys.slice(5, 3), whole.slice(5, 3)),
        ];

        for parts in [2usize, 4, 8] {
            let got = partition_morsels(&morsels, &["k".into()], parts).unwrap();
            // The same partitioning, with every column forced through `interleave`.
            let part_of: Vec<(Vec<u32>, Vec<u32>)> = morsels
                .iter()
                .map(|b| {
                    let k = vec![b.column(0).clone()];
                    let p = shuffle::bucket_of_rows(&k, b.num_rows(), parts).unwrap();
                    shuffle::bucket_csr(&p, parts)
                })
                .collect();
            let src: Vec<&dyn Array> = morsels.iter().map(|b| b.column(1).as_ref()).collect();
            for (bucket, batch) in got.iter().enumerate() {
                let mut pairs: Vec<(usize, usize)> = Vec::new();
                for (mi, (rows, off)) in part_of.iter().enumerate() {
                    pairs.extend(
                        rows[off[bucket] as usize..off[bucket + 1] as usize]
                            .iter()
                            .map(|&r| (mi, r as usize)),
                    );
                }
                let want = interleave(&src, &pairs).unwrap();
                assert_eq!(
                    batch.column(1),
                    &want,
                    "bucket {bucket} of {parts} differs from interleave"
                );
            }
        }
    }

    /// The chunked scatter against `interleave` on a relation with **more morsels than
    /// chunks** (so a chunk spans several morsels and a slot holds several bins), values on
    /// both sides of the fixed-width short copy (0, 15, 16, 17 and 40 bytes), and every
    /// morsel a slice ending at its source buffer's last byte — where an over-read of the
    /// short copy would run off the end.
    #[test]
    fn the_chunked_scatter_matches_interleave_across_many_morsels() {
        use arrow::array::Float64Array;
        let lens = [0usize, 1, 15, 16, 17, 3, 40, 2, 16, 5];
        let morsels: Vec<RecordBatch> = (0..97)
            .map(|m| {
                let n = 1 + (m * 7) % 23;
                let keys: Vec<i64> = (0..n).map(|i| ((m * 31 + i * 17) % 50) as i64).collect();
                let strs: Vec<String> = (0..n)
                    .map(|i| {
                        let len = lens[(m + i) % lens.len()];
                        (0..len)
                            .map(|j| (b'a' + ((m + i + j) % 26) as u8) as char)
                            .collect()
                    })
                    .collect();
                let f: Vec<f64> = (0..n).map(|i| (m * 100 + i) as f64).collect();
                RecordBatch::try_from_iter(vec![
                    ("k", Arc::new(Int64Array::from(keys)) as ArrayRef),
                    ("s", Arc::new(StringArray::from(strs)) as ArrayRef),
                    ("f", Arc::new(Float64Array::from(f)) as ArrayRef),
                ])
                .unwrap()
            })
            .collect();
        for parts in [2usize, 7, 64] {
            let got = partition_morsels(&morsels, &["k".into()], parts).unwrap();
            assert_eq!(got.len(), parts);
            let bins: Vec<(Vec<u32>, Vec<u32>)> = morsels
                .iter()
                .map(|b| {
                    let k = vec![b.column(0).clone()];
                    let p = shuffle::bucket_of_rows(&k, b.num_rows(), parts).unwrap();
                    shuffle::bucket_csr(&p, parts)
                })
                .collect();
            for (bucket, batch) in got.iter().enumerate() {
                let mut pairs: Vec<(usize, usize)> = Vec::new();
                for (mi, (rows, off)) in bins.iter().enumerate() {
                    pairs.extend(
                        rows[off[bucket] as usize..off[bucket + 1] as usize]
                            .iter()
                            .map(|&r| (mi, r as usize)),
                    );
                }
                for c in 0..3 {
                    let src: Vec<&dyn Array> =
                        morsels.iter().map(|b| b.column(c).as_ref()).collect();
                    let want = interleave(&src, &pairs).unwrap();
                    assert_eq!(
                        batch.column(c),
                        &want,
                        "column {c}, bucket {bucket} of {parts} differs from interleave"
                    );
                }
            }
        }
    }

    /// A relation of a few **large** batches — what a filter or a materialized input hands a
    /// shuffle join — is cut into chunks *inside* each batch, so the scatter fills the pool
    /// instead of running on one task per batch. The cut must not move a row: every bucket
    /// still equals `interleave` over the same bins, with chunks ending mid-batch at points
    /// that need the bin trimming (`bin`) on both sides.
    #[test]
    fn chunks_cut_inside_large_batches_and_still_match_interleave() {
        let mk = |from: usize, n: usize| {
            let keys: Vec<i64> = (from..from + n)
                .map(|i| (i * 7919 % 1_000) as i64)
                .collect();
            let strs: Vec<String> = (from..from + n).map(|i| "x".repeat(i % 23)).collect();
            RecordBatch::try_from_iter(vec![
                ("k", Arc::new(Int64Array::from(keys)) as ArrayRef),
                ("s", Arc::new(StringArray::from(strs)) as ArrayRef),
            ])
            .unwrap()
        };
        let big = mk(0, 60_000);
        // A slice of a larger batch, so the trimmed bins index a rebased `value_offsets`.
        let morsels = [
            big.slice(0, 41_000),
            mk(41_000, 9),
            big.slice(41_000, 19_000),
        ];
        let total: usize = morsels.iter().map(|b| b.num_rows()).sum();
        let layout_chunks = {
            let bins: Vec<(Vec<u32>, Vec<u32>)> = morsels
                .iter()
                .map(|b| {
                    let p =
                        shuffle::bucket_of_rows(&[b.column(0).clone()], b.num_rows(), 5).unwrap();
                    shuffle::bucket_csr(&p, 5)
                })
                .collect();
            Layout::new(&bins, 5, 4).chunks
        };
        assert!(
            layout_chunks.iter().flatten().any(|s| !s.whole),
            "the fixture must make a chunk end inside a batch"
        );
        assert_eq!(
            layout_chunks
                .iter()
                .flatten()
                .map(|s| (s.hi - s.lo) as usize)
                .sum::<usize>(),
            total,
            "the chunks must tile the relation"
        );
        for parts in [3usize, 16] {
            let got = partition_morsels(&morsels, &["k".into()], parts).unwrap();
            let bins: Vec<(Vec<u32>, Vec<u32>)> = morsels
                .iter()
                .map(|b| {
                    let p = shuffle::bucket_of_rows(&[b.column(0).clone()], b.num_rows(), parts)
                        .unwrap();
                    shuffle::bucket_csr(&p, parts)
                })
                .collect();
            for (bucket, batch) in got.iter().enumerate() {
                let mut pairs: Vec<(usize, usize)> = Vec::new();
                for (mi, (rows, off)) in bins.iter().enumerate() {
                    pairs.extend(
                        rows[off[bucket] as usize..off[bucket + 1] as usize]
                            .iter()
                            .map(|&r| (mi, r as usize)),
                    );
                }
                for c in 0..2 {
                    let src: Vec<&dyn Array> =
                        morsels.iter().map(|b| b.column(c).as_ref()).collect();
                    let want = interleave(&src, &pairs).unwrap();
                    assert_eq!(
                        batch.column(c),
                        &want,
                        "column {c}, bucket {bucket}/{parts}"
                    );
                }
            }
        }
    }

    /// A `Binary` column takes the same path, and a `LargeUtf8` one takes the 64-bit-offset
    /// instantiation — the two the `Utf8` case would silently mis-downcast if the dispatch
    /// were keyed on anything but the exact type.
    #[test]
    fn wide_and_binary_byte_columns_round_trip() {
        use arrow::array::{BinaryArray, LargeStringArray};
        let k: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 2, 3, 4]));
        let b: ArrayRef = Arc::new(BinaryArray::from(vec![
            &b""[..],
            &b"\x00\xff"[..],
            &b"xyz"[..],
            &b"\x01"[..],
        ]));
        let l: ArrayRef = Arc::new(LargeStringArray::from(vec!["", "aa", "bbb", "c"]));
        let batch =
            RecordBatch::try_from_iter(vec![("k", k), ("b", b.clone()), ("l", l.clone())]).unwrap();
        let got = partition_morsels(&[batch], &["k".into()], 4).unwrap();
        let mut seen: Vec<(Vec<u8>, String)> = Vec::new();
        for out in &got {
            let bb = out
                .column(1)
                .as_any()
                .downcast_ref::<BinaryArray>()
                .unwrap();
            let ll = out
                .column(2)
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .unwrap();
            for i in 0..out.num_rows() {
                seen.push((bb.value(i).to_vec(), ll.value(i).to_string()));
            }
        }
        seen.sort();
        let mut want: Vec<(Vec<u8>, String)> = vec![
            (b"".to_vec(), String::new()),
            (b"\x00\xff".to_vec(), "aa".into()),
            (b"xyz".to_vec(), "bbb".into()),
            (b"\x01".to_vec(), "c".into()),
        ];
        want.sort();
        assert_eq!(seen, want);
    }

    /// A bucket that receives no rows is still an empty batch with the right schema.
    #[test]
    fn empty_buckets_keep_their_schema() {
        let morsels = [batch(&[42], &["a"])];
        let got = partition_morsels(&morsels, &["k".into()], 8).unwrap();
        assert_eq!(got.len(), 8);
        assert_eq!(got.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
        for b in &got {
            assert_eq!(b.schema(), morsels[0].schema());
        }
    }
}
