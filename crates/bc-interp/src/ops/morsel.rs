//! Morselization: splitting input batches into row- **and** byte-bounded morsels
//! for the parallel scheduler.
//!
//! The scheduler wants morsels that are small enough to load-balance and whose
//! working set fits in cache, but a fixed row count is byte-blind: 16 384 rows of
//! `Int64` is ~128 KiB, while 16 384 rows of multi-MB blobs is gigabytes. So a
//! morsel is "full" at **either** the row target or the byte budget
//! ([`bc_arrow::MorselTarget`]).
//!
//! Byte sizing is *measured*, not estimated — but only when it can matter. For an
//! all-fixed-width batch the per-row width is constant, so the split is O(1). For a
//! variable-width batch whose *average* row is narrow enough that the row target
//! always trips first (`avg_row_bytes × rows ≤ byte_budget` — the common analytical
//! case of a few short string/code columns beside numerics), per-row widths cannot
//! move the boundaries, so it too splits uniformly in O(1). Only when the average row
//! is wide enough to approach the byte budget do we read the offset buffers for each
//! row's true cost and accumulate greedily — so a single row wider than the whole
//! budget becomes its own one-row morsel (a giant cell never co-resides with 16 k
//! others), and intra-batch width variance (999 tiny rows + 1 huge one) is split
//! where it actually is. The average-width guard uses `get_array_memory_size()`, which
//! over-counts a *slice* (shared parent buffer); that only makes the guard
//! conservative — it never skips the per-row walk when the walk is needed.
//!
//! With a row-only target (`MorselTarget::rows`, byte bound = `usize::MAX`) every
//! path short-circuits to the historical row-count morselizer — byte-for-byte
//! identical, no offset walk, zero added cost on the narrow-data fast path.

use arrow::array::OffsetSizeTrait;
use arrow::array::{
    Array, ArrayRef, GenericBinaryArray, GenericListArray, GenericStringArray, RecordBatch,
    StructArray,
};
use arrow::datatypes::DataType;

/// Split `batches` into morsels bounded by `target`'s row and byte limits, coalescing
/// a run of undersized batches into one morsel first.
///
/// Rows are preserved exactly (same multiset, same order); only the batch boundaries
/// change. Empty batches pass through unchanged.
///
/// Two failure modes of a source's own batching are corrected here so the parallel
/// scheduler always sees well-sized morsels:
///   * **Too large** — an over-target batch is split into row/byte-bounded morsels.
///   * **Too small** — a run of under-target batches (a streaming reader, a highly
///     selective upstream, a fine-grained shuffle emitting thousands of tiny batches)
///     is concatenated up to the target before splitting. Without this, each tiny
///     batch becomes its own morsel — one poorly-parallelized task, and one partial
///     state to merge, *per batch* — which measured ~19x slower on a high-cardinality
///     group-by fed 256-row batches than the same rows in morsel-sized batches.
///
/// A batch that already fills a morsel is never buffered: it splits (if over) or passes
/// through, both zero-copy — so well-sized input pays nothing (no concat, no byte walk).
pub(crate) fn morselize(
    batches: &[RecordBatch],
    target: bc_arrow::MorselTarget,
) -> Vec<RecordBatch> {
    let mut out = Vec::with_capacity(batches.len());
    // A run of consecutive undersized batches, held until it reaches the target (or a
    // full/empty batch forces a flush) and then merged into one morsel.
    let mut pending: Vec<RecordBatch> = Vec::new();
    let mut pending_rows = 0usize;
    let mut pending_bytes = 0usize;
    for b in batches {
        let n = b.num_rows();
        if n == 0 {
            // Preserve the historical empty-batch passthrough; flush first so order holds.
            flush_coalesced(
                &mut out,
                &mut pending,
                &mut pending_rows,
                &mut pending_bytes,
                target,
            );
            out.push(b.clone());
            continue;
        }
        if batch_stands_alone(b, n, target) {
            flush_coalesced(
                &mut out,
                &mut pending,
                &mut pending_rows,
                &mut pending_bytes,
                target,
            );
            split_batch(&mut out, b, target);
            continue;
        }
        pending.push(b.clone());
        pending_rows += n;
        if target.byte_bounded() {
            pending_bytes += sliced_batch_bytes(b);
        }
        if pending_rows >= target.rows || (target.byte_bounded() && pending_bytes >= target.bytes) {
            flush_coalesced(
                &mut out,
                &mut pending,
                &mut pending_rows,
                &mut pending_bytes,
                target,
            );
        }
    }
    flush_coalesced(
        &mut out,
        &mut pending,
        &mut pending_rows,
        &mut pending_bytes,
        target,
    );
    out
}

/// Whether `b` is already a well-sized morsel on its own — so it passes straight through
/// (splitting if it overshoots) rather than being buffered to coalesce with its neighbours.
///
/// "Well-sized" is **at least half** the target (rows, or — when byte-bounded — bytes), not
/// the full target. Coalescing only pays when it turns many *tiny* batches into one morsel;
/// concatenating a batch that is already a healthy fraction of a morsel copies its whole
/// payload to gain nothing. The motivating regression: an 88%-selective filter emits
/// ~14 k-row batches from 16 k-row morsels; at the full-target threshold every one was below
/// it, so the whole filtered relation was concatenated and re-split — a full extra copy
/// (~2.4x slower on a broad filter). Half-target lets those near-full batches through
/// untouched while still coalescing a fine-grained source's genuinely small batches (the
/// 256-row-batch pathology is `1/64` of the target, far under the line). The row check
/// short-circuits the byte walk on the common fast path.
fn batch_stands_alone(b: &RecordBatch, n: usize, target: bc_arrow::MorselTarget) -> bool {
    n.saturating_mul(2) >= target.rows
        || (target.byte_bounded() && sliced_batch_bytes(b).saturating_mul(2) >= target.bytes)
}

/// Emit the buffered undersized run: concatenate it into one contiguous batch and split
/// that to the target (the merged batch may slightly overshoot, so it still gets bounded).
/// A single buffered batch needs no concat. On a concat error — impossible for the
/// schema-identical batches of one source, but handled rather than panicked on a data
/// path — each buffered batch is emitted on its own.
fn flush_coalesced(
    out: &mut Vec<RecordBatch>,
    pending: &mut Vec<RecordBatch>,
    pending_rows: &mut usize,
    pending_bytes: &mut usize,
    target: bc_arrow::MorselTarget,
) {
    match pending.len() {
        0 => {}
        1 => split_batch(out, &pending[0], target),
        _ => match arrow::compute::concat_batches(&pending[0].schema(), pending.iter()) {
            Ok(merged) => split_batch(out, &merged, target),
            Err(_) => {
                for b in pending.iter() {
                    split_batch(out, b, target);
                }
            }
        },
    }
    pending.clear();
    *pending_rows = 0;
    *pending_bytes = 0;
}

/// Parallel [`morselize`] for the scan: split each input batch across rayon, preserving
/// order. Byte-bounded splitting of a variable-width (string/list) column reads that
/// column's offsets for every row ([`split_batch`]'s per-row-cost path) — an O(rows) walk
/// the sequential `morselize` runs on **one** core. On a wide, string-heavy table that walk
/// is the scan's dominant cost (measured ~450 ms and `cpu≈1 core` splitting 60 M rows of
/// TPC-H `lineitem` for a `SELECT *` — the whole rest of the query is faster than it). The
/// split of one batch is independent of every other, so fanning it across cores is a pure
/// scheduling win: `batches.len()` was already the parallelism the downstream operators use.
///
/// Correctness: when no batch needs **coalescing** (every batch already stands alone as a
/// morsel), `morselize` reduces to "split each batch, in order" with no cross-batch state —
/// exactly what the parallel map computes, so the morsels are byte-for-byte identical. The
/// coalescing path (a fine-grained source's undersized batches merged across boundaries) is
/// inherently sequential, so if *any* batch is undersized this falls back to `morselize`.
/// Row-only targets never walk offsets, so they gain nothing here and take the cheap
/// sequential path too.
pub(crate) fn morselize_par(
    batches: &[RecordBatch],
    target: bc_arrow::MorselTarget,
) -> Vec<RecordBatch> {
    use rayon::prelude::*;
    // Parallelizing only pays for a byte-bounded target (the offset walk) with enough
    // batches to fan out; and only the no-coalescing case is order-independent.
    let worth_parallel = target.byte_bounded()
        && batches.len() > 1
        && batches
            .iter()
            .all(|b| b.num_rows() == 0 || batch_stands_alone(b, b.num_rows(), target));
    if !worth_parallel {
        return morselize(batches, target);
    }
    batches
        .par_iter()
        .map(|b| {
            let mut out = Vec::new();
            if b.num_rows() == 0 {
                out.push(b.clone());
            } else {
                split_batch(&mut out, b, target);
            }
            out
        })
        .flatten()
        .collect()
}

/// Re-bound already-produced morsels after a width-changing operator.
///
/// The scheduler morselizes at the scan, but a projection that adds a wide column
/// or an unnest/unpivot that multiplies rows can push a 1:1 output morsel past the
/// byte budget even though its input was within it. Re-splitting here keeps every
/// downstream operator's working set bounded.
///
/// In row-only mode (no byte bound) this is a no-op that preserves the historical
/// morsel boundaries exactly. In byte-bounded mode an already-within-budget batch
/// is returned as a cheap `Arc` clone, so narrow data pays nothing; only a batch
/// that actually overshoots is split.
pub(crate) fn remorselize(
    batches: Vec<RecordBatch>,
    target: bc_arrow::MorselTarget,
) -> Vec<RecordBatch> {
    if !target.byte_bounded() {
        return batches;
    }
    // Same parallel split as the scan: a wide join/aggregate output (14 GB of TPC-H
    // `lineitem` for a `SELECT *` join) is re-split here, and the per-row byte walk over its
    // string columns is the same single-threaded O(rows) cost `morselize_par` fans across cores.
    morselize_par(&batches, target)
}

/// Emit the morsels for one batch into `out`.
fn split_batch(out: &mut Vec<RecordBatch>, b: &RecordBatch, target: bc_arrow::MorselTarget) {
    let n = b.num_rows();
    if n == 0 {
        out.push(b.clone());
        return;
    }
    // Row-only target (the historical default): no byte walk at all.
    if !target.byte_bounded() {
        emit_uniform(out, b, n, target.rows);
        return;
    }
    // All-fixed-width batch: per-row width is constant, so chunk size is O(1).
    if let Some(w) = constant_row_width(b) {
        let by_bytes = (target.bytes / w).max(1);
        emit_uniform(out, b, n, target.rows.min(by_bytes));
        return;
    }
    // Variable-width batch, but the *average* row is narrow enough that the row cap
    // always trips before the byte budget: then per-row widths cannot change the
    // boundaries, so slice uniformly by rows and skip the O(rows) byte walk. This is
    // the common analytical case (a few short string/code columns alongside numerics) —
    // q1's `l_returnflag`/`l_linestatus` are single chars, so a morsel of `rows` is far
    // under the byte budget. The size is measured **slice-aware** ([`sliced_batch_bytes`],
    // O(columns)): a plain `get_array_memory_size` counts a slice's whole shared parent
    // buffer, so a 16 k-row slice of a 60 M-row string column looks kilobytes-per-row and
    // wrongly forces the O(rows) offset walk on every such morsel — the dominant cost of a
    // projection/DISTINCT over sliced string columns at scale.
    let avg_row_bytes = sliced_batch_bytes(b) / n;
    if avg_row_bytes.saturating_mul(target.rows) <= target.bytes {
        emit_uniform(out, b, n, target.rows);
        return;
    }
    // Wide/variable rows: accumulate by *measured* per-row bytes so a few large rows
    // (decoded frames, long lists) are isolated into their own morsels.
    let costs = per_row_bytes(b);
    let mut start = 0usize;
    let mut acc = 0usize;
    let mut rows_in = 0usize;
    for (i, &c) in costs.iter().enumerate() {
        // Close the current morsel before adding row `i` if it is already at the
        // row cap or adding `i` would overshoot the byte budget. The `rows_in > 0`
        // guard guarantees forward progress, so a lone row wider than the whole
        // budget is emitted as a one-row morsel rather than looping.
        if rows_in > 0 && (rows_in >= target.rows || acc + c > target.bytes) {
            out.push(b.slice(start, rows_in));
            start = i;
            acc = 0;
            rows_in = 0;
        }
        acc += c;
        rows_in += 1;
    }
    if rows_in > 0 {
        out.push(b.slice(start, rows_in));
    }
}

/// The byte size of `b` accounting for slicing — unlike [`RecordBatch::get_array_memory_size`],
/// which counts a sliced array's whole shared parent buffer. O(columns): fixed-width columns
/// contribute `width × rows`; string/binary columns their slice's data span (last minus first
/// offset) plus the offset slots; other variable-width columns fall back to the memory-size
/// over-count (which only makes the byte guard more conservative for those rarer types).
pub(crate) fn sliced_batch_bytes(b: &RecordBatch) -> usize {
    b.columns().iter().map(sliced_column_bytes).sum()
}

fn sliced_column_bytes(col: &ArrayRef) -> usize {
    let n = col.len();
    if let Some(w) = bc_arrow::fixed_width(col.data_type()) {
        return w * n;
    }
    match col.data_type() {
        DataType::Utf8 => byte_slice_size::<i32>(
            col.as_any()
                .downcast_ref::<GenericStringArray<i32>>()
                .map(|a| a.value_offsets()),
        ),
        DataType::LargeUtf8 => byte_slice_size::<i64>(
            col.as_any()
                .downcast_ref::<GenericStringArray<i64>>()
                .map(|a| a.value_offsets()),
        ),
        DataType::Binary => byte_slice_size::<i32>(
            col.as_any()
                .downcast_ref::<GenericBinaryArray<i32>>()
                .map(|a| a.value_offsets()),
        ),
        DataType::LargeBinary => byte_slice_size::<i64>(
            col.as_any()
                .downcast_ref::<GenericBinaryArray<i64>>()
                .map(|a| a.value_offsets()),
        ),
        // List/Struct/etc.: the memory-size over-count on a slice is acceptable here (only
        // shifts the guard toward the safe per-row walk for these less common columns).
        _ => col.get_array_memory_size(),
    }
}

/// Slice-aware byte size of a string/binary column from its offset buffer: the data span
/// (`offsets[last] − offsets[first]`) plus the per-row offset slots. `None` (a failed
/// downcast) can't happen given the caller's type match; it maps to 0 defensively.
fn byte_slice_size<O: OffsetSizeTrait>(offsets: Option<&[O]>) -> usize {
    match offsets {
        Some(o) => {
            let span = match (o.first(), o.last()) {
                (Some(&f), Some(&l)) => (l - f).as_usize(),
                _ => 0,
            };
            span + std::mem::size_of_val(o)
        }
        None => 0,
    }
}

/// Emit `b` in fixed `chunk`-row slices (the historical row morselizer).
fn emit_uniform(out: &mut Vec<RecordBatch>, b: &RecordBatch, n: usize, chunk: usize) {
    if n <= chunk {
        out.push(b.clone());
        return;
    }
    let mut off = 0;
    while off < n {
        let len = (n - off).min(chunk);
        out.push(b.slice(off, len));
        off += len;
    }
}

/// The constant per-row byte width of a batch whose every column is fixed-width,
/// or `None` if any column is variable-width (then the offset walk is needed).
/// Always ≥ 1 so it can divide the byte budget.
fn constant_row_width(b: &RecordBatch) -> Option<usize> {
    let mut w = 0usize;
    for f in b.schema().fields() {
        w += bc_arrow::fixed_width(f.data_type())?;
    }
    Some(w.max(1))
}

/// The true byte cost of each row, summed across columns. Fixed-width columns add
/// a constant; string/binary columns add their per-row payload from the offset
/// buffer (plus the offset slot); other variable-width columns (List/Struct/…)
/// amortize their total Arrow bytes over the rows.
fn per_row_bytes(b: &RecordBatch) -> Vec<usize> {
    let mut costs = vec![0usize; b.num_rows()];
    for col in b.columns() {
        add_column_bytes(&mut costs, col);
    }
    costs
}

fn add_column_bytes(costs: &mut [usize], col: &ArrayRef) {
    if let Some(w) = bc_arrow::fixed_width(col.data_type()) {
        for c in costs.iter_mut() {
            *c += w;
        }
        return;
    }
    match col.data_type() {
        DataType::Utf8 => add_string_bytes::<i32>(costs, col),
        DataType::LargeUtf8 => add_string_bytes::<i64>(costs, col),
        DataType::Binary => add_binary_bytes::<i32>(costs, col),
        DataType::LargeBinary => add_binary_bytes::<i64>(costs, col),
        // Variable-length list/struct: walk per-row so a few huge rows (decoded
        // video frames, long `List<float32>` waveforms) are isolated into their own
        // morsels instead of being smeared over a batch average.
        DataType::List(_) => add_list_bytes::<i32>(costs, col),
        DataType::LargeList(_) => add_list_bytes::<i64>(costs, col),
        DataType::Struct(_) => add_struct_bytes(costs, col),
        // Other variable-width (Map/Union/FixedSizeList-of-variable/…): amortize the
        // total bytes over the rows. Coarser than a per-row walk, but these are the
        // rare cases and the result still tracks the column's real footprint.
        _ => add_amortized_bytes(costs, col),
    }
}

/// Amortize a column's total Arrow footprint evenly over its rows — the fallback
/// for variable-width types without a cheap per-row walk.
fn add_amortized_bytes(costs: &mut [usize], col: &ArrayRef) {
    let per = (col.get_array_memory_size() / costs.len().max(1)).max(1);
    for c in costs.iter_mut() {
        *c += per;
    }
}

/// Add each row's list payload: `(elements in row) × child width + offset slot`,
/// recursing into a variable-width child so a long list of wide elements is costed
/// per row, not averaged.
fn add_list_bytes<O: OffsetSizeTrait>(costs: &mut [usize], col: &ArrayRef) {
    let a = col
        .as_any()
        .downcast_ref::<GenericListArray<O>>()
        .expect("data type checked by caller");
    let offsets = a.value_offsets();
    let values = a.values();
    let off_w = std::mem::size_of::<O>();
    if let Some(w) = bc_arrow::fixed_width(values.data_type()) {
        for (i, c) in costs.iter_mut().enumerate() {
            let n = offsets[i + 1].as_usize() - offsets[i].as_usize();
            *c += n * w + off_w;
        }
        return;
    }
    // Variable-width child: cost each child element once, then sum each row's slice.
    let mut child_costs = vec![0usize; values.len()];
    add_column_bytes(&mut child_costs, values);
    for (i, c) in costs.iter_mut().enumerate() {
        let lo = offsets[i].as_usize();
        let hi = offsets[i + 1].as_usize();
        *c += child_costs[lo..hi].iter().sum::<usize>() + off_w;
    }
}

/// Add each row's struct payload by recursing into the fields. Each field array is
/// row-aligned with the struct, so per-row field costs accumulate directly; a
/// sliced/offset struct (children longer than the logical rows) falls back to
/// amortization to stay correct.
fn add_struct_bytes(costs: &mut [usize], col: &ArrayRef) {
    let a = col
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("data type checked by caller");
    for child in a.columns() {
        if child.len() == costs.len() {
            add_column_bytes(costs, child);
        } else {
            add_amortized_bytes(costs, child);
        }
    }
}

fn add_string_bytes<O: OffsetSizeTrait>(costs: &mut [usize], col: &ArrayRef) {
    let a = col
        .as_any()
        .downcast_ref::<GenericStringArray<O>>()
        .expect("data type checked by caller");
    add_offset_bytes(costs, a.value_offsets(), std::mem::size_of::<O>());
}

fn add_binary_bytes<O: OffsetSizeTrait>(costs: &mut [usize], col: &ArrayRef) {
    let a = col
        .as_any()
        .downcast_ref::<GenericBinaryArray<O>>()
        .expect("data type checked by caller");
    add_offset_bytes(costs, a.value_offsets(), std::mem::size_of::<O>());
}

/// Add each row's variable payload (`offsets[i+1] - offsets[i]`) plus the
/// per-row offset slot (`offset_width`) to `costs`.
fn add_offset_bytes<O: OffsetSizeTrait>(costs: &mut [usize], offsets: &[O], offset_width: usize) {
    for (i, c) in costs.iter_mut().enumerate() {
        let payload = offsets[i + 1].as_usize() - offsets[i].as_usize();
        *c += payload + offset_width;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn str_batch(vals: &[&str]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, false)]));
        let arr = Arc::new(StringArray::from(vals.to_vec())) as ArrayRef;
        RecordBatch::try_new(schema, vec![arr]).unwrap()
    }

    fn int_batch(n: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int64, false)]));
        let arr = Arc::new(Int64Array::from((0..n as i64).collect::<Vec<_>>())) as ArrayRef;
        RecordBatch::try_new(schema, vec![arr]).unwrap()
    }

    fn total_rows(batches: &[RecordBatch]) -> usize {
        batches.iter().map(|b| b.num_rows()).sum()
    }

    /// A single giant cell amid many tiny rows is isolated into its own one-row
    /// morsel — the batch-average heuristic would have hidden it in a fat morsel.
    #[test]
    fn giant_row_isolated_from_tiny_rows() {
        let giant = "y".repeat(100_000);
        let mut vals: Vec<&str> = vec!["x"; 100];
        vals.push(&giant);
        vals.extend(std::iter::repeat_n("z", 100));
        let b = str_batch(&vals);

        let target = bc_arrow::MorselTarget::new(16_384, 1024);
        let out = morselize(&[b], target);

        assert_eq!(total_rows(&out), 201, "rows must be preserved exactly");
        // The morsel holding the giant value is a one-row morsel.
        let giant_morsel = out
            .iter()
            .find(|m| {
                let s = m.column(0).as_any().downcast_ref::<StringArray>().unwrap();
                (0..s.len()).any(|i| s.value(i).len() == 100_000)
            })
            .expect("giant value survives");
        assert_eq!(giant_morsel.num_rows(), 1, "giant cell must stand alone");
    }

    /// A `List<Float32>` batch with a few very long rows (e.g. decoded waveforms)
    /// among many tiny ones is split per row, not by the batch average — the long
    /// rows are isolated instead of smeared. Regression for the amortized fallback.
    #[test]
    fn long_list_rows_isolated_from_tiny_rows() {
        use arrow::array::{Float32Array, ListArray};
        use arrow::buffer::OffsetBuffer;
        use arrow::datatypes::Field as ArrowField;

        // 100 single-element rows, then one 8 000-element row, then 100 more tiny.
        let mut values: Vec<f32> = Vec::new();
        let mut offsets: Vec<i32> = vec![0];
        for _ in 0..100 {
            values.push(1.0);
            offsets.push(values.len() as i32);
        }
        values.extend(std::iter::repeat_n(2.0, 8_000));
        offsets.push(values.len() as i32);
        for _ in 0..100 {
            values.push(3.0);
            offsets.push(values.len() as i32);
        }
        let child = Arc::new(Float32Array::from(values));
        let field = Arc::new(ArrowField::new("item", DataType::Float32, false));
        let list = ListArray::new(field, OffsetBuffer::new(offsets.into()), child, None);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "w",
            list.data_type().clone(),
            false,
        )]));
        let b = RecordBatch::try_new(schema, vec![Arc::new(list) as ArrayRef]).unwrap();

        // Budget ≈ 1 KiB; the 8 000-elem row (~32 KB) far exceeds it and must stand
        // alone, while the tiny rows pack together.
        let target = bc_arrow::MorselTarget::new(16_384, 1024);
        let out = morselize(&[b], target);
        assert_eq!(total_rows(&out), 201, "rows must be preserved exactly");
        let big = out
            .iter()
            .find(|m| {
                let l = m.column(0).as_any().downcast_ref::<ListArray>().unwrap();
                (0..l.len()).any(|i| l.value(i).len() == 8_000)
            })
            .expect("the long list row survives");
        assert_eq!(big.num_rows(), 1, "the long list row must stand alone");
    }

    /// An all-fixed-width batch splits by the constant per-row width (O(1) path)
    /// and preserves every row.
    #[test]
    fn fixed_width_constant_chunking() {
        let b = int_batch(1000);
        // 8 bytes/row → a 256-byte budget yields ~32-row morsels.
        let target = bc_arrow::MorselTarget::new(16_384, 256);
        let out = morselize(&[b], target);
        assert_eq!(total_rows(&out), 1000);
        assert!(
            out.len() > 1,
            "tight byte budget should split fixed-width data"
        );
    }

    /// A variable-width batch whose average row is narrow (short strings) splits by the
    /// row cap with no per-row byte walk — the O(1) fast path — and still isolates a
    /// genuinely huge row when one appears (the walk re-engages).
    #[test]
    fn narrow_strings_split_uniformly_by_rows() {
        // 1000 short strings; average row ~ a few bytes, far under any sane byte budget.
        let vals: Vec<String> = (0..1000).map(|i| format!("c{}", i % 5)).collect();
        let refs: Vec<&str> = vals.iter().map(String::as_str).collect();
        let b = str_batch(&refs);
        // Row cap 128, generous byte budget: avg_row_bytes × 128 ≪ budget, so every
        // morsel is exactly the row cap (uniform) — the boundaries the per-row walk
        // would also produce, reached without it.
        let target = bc_arrow::MorselTarget::new(128, 1 << 20);
        let out = morselize(&[b], target);
        assert_eq!(total_rows(&out), 1000);
        assert!(out.iter().all(|m| m.num_rows() <= 128));
        assert_eq!(out.len(), 1000_usize.div_ceil(128));

        // A batch with one giant string still isolates it (avg width is pulled high, so
        // the per-row walk runs and stands the giant row alone).
        let giant = "y".repeat(2 << 20);
        let mixed = str_batch(&["a", giant.as_str(), "b"]);
        let out = morselize(&[mixed], bc_arrow::MorselTarget::new(128, 1 << 20));
        let big = out
            .iter()
            .find(|m| {
                let s = m.column(0).as_any().downcast_ref::<StringArray>().unwrap();
                (0..s.len()).any(|i| s.value(i).len() == 2 << 20)
            })
            .expect("the giant string row survives");
        assert_eq!(big.num_rows(), 1, "the giant string row must stand alone");
    }

    /// A row-only target never walks bytes: a wide batch under the row cap is one
    /// morsel, byte-for-byte the historical behavior.
    #[test]
    fn row_only_target_is_identity() {
        let giant = "y".repeat(100_000);
        let b = str_batch(&[giant.as_str(), "x", "z"]);
        let out = morselize(&[b], bc_arrow::MorselTarget::rows(16_384));
        assert_eq!(out.len(), 1, "row-only target must not byte-split");
        assert_eq!(total_rows(&out), 3);
    }

    /// A run of many tiny batches is coalesced into full morsels — the fix for the
    /// per-batch-task pathology on fine-grained sources. Rows are preserved and every
    /// morsel (bar the trailing remainder) reaches the row target.
    #[test]
    fn tiny_batches_are_coalesced_to_the_target() {
        let tiny: Vec<RecordBatch> = (0..500).map(|_| int_batch(64)).collect(); // 64 rows each
        let target = bc_arrow::MorselTarget::rows(16_384);
        let out = morselize(&tiny, target);
        assert_eq!(total_rows(&out), 500 * 64, "rows preserved");
        assert!(
            out.len() < 500 / 10,
            "coalesced to far fewer morsels than input batches, got {}",
            out.len()
        );
        // Every full morsel is at the target; only the last may be a smaller remainder.
        for m in &out[..out.len() - 1] {
            assert_eq!(m.num_rows(), 16_384);
        }
    }

    /// Coalescing preserves row order across the run (the concatenation is in order).
    #[test]
    fn coalescing_preserves_row_order() {
        // Three tiny batches of distinct value ranges, in order.
        let mk = |lo: i64, hi: i64| {
            let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int64, false)]));
            let arr = Arc::new(Int64Array::from((lo..hi).collect::<Vec<_>>())) as ArrayRef;
            RecordBatch::try_new(schema, vec![arr]).unwrap()
        };
        let batches = vec![mk(0, 100), mk(100, 250), mk(250, 300)];
        let out = morselize(&batches, bc_arrow::MorselTarget::rows(16_384));
        let stitched: Vec<i64> = out
            .iter()
            .flat_map(|m| {
                let a = m.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
                (0..a.len()).map(|i| a.value(i)).collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(stitched, (0..300).collect::<Vec<_>>(), "order preserved");
    }

    /// An already-full batch is never buffered/copied: a run of morsel-sized batches
    /// passes through as the same Arc-backed batches (zero coalescing on the fast path).
    #[test]
    fn full_batches_pass_through_without_coalescing() {
        let full: Vec<RecordBatch> = (0..3).map(|_| int_batch(16_384)).collect();
        let target = bc_arrow::MorselTarget::rows(16_384);
        let out = morselize(&full, target);
        assert_eq!(out.len(), 3, "each full batch stays its own morsel");
        for (a, b) in out.iter().zip(full.iter()) {
            assert_eq!(a.num_rows(), b.num_rows());
        }
    }

    /// Near-full batches (an 88%-selective filter's output) stand alone: they pass through
    /// as their own morsels with no concat/re-split copy — the fix for the broad-filter
    /// regression. Each ~14 k-row batch is above the half-target line, so none is buffered.
    #[test]
    fn near_full_batches_are_not_coalesced() {
        let nearly: Vec<RecordBatch> = (0..8).map(|_| int_batch(14_400)).collect();
        let target = bc_arrow::MorselTarget::rows(16_384);
        let out = morselize(&nearly, target);
        assert_eq!(
            out.len(),
            8,
            "each near-full batch stays its own morsel, no coalescing"
        );
        assert_eq!(total_rows(&out), 8 * 14_400, "rows preserved");
        for m in &out {
            assert_eq!(
                m.num_rows(),
                14_400,
                "batches passed through unchanged (no re-split)"
            );
        }
    }

    /// `morselize_par` must produce byte-for-byte the same morsels as `morselize` on every
    /// shape — it only changes *which core* splits each batch, never the boundaries. Covers
    /// the parallel fast path (full wide string batches, byte-bounded) and the fall-throughs
    /// (row-only target, undersized/coalescing batches, empties).
    #[test]
    fn morselize_par_matches_sequential() {
        let long = "z".repeat(300); // wide enough to trip the per-row byte walk
        let wide: Vec<RecordBatch> = (0..40)
            .map(|_| str_batch(&[long.as_str(); 8_000]))
            .collect();
        let cases: Vec<(Vec<RecordBatch>, bc_arrow::MorselTarget)> = vec![
            // Parallel fast path: many full wide batches, byte-bounded.
            (wide.clone(), bc_arrow::MorselTarget::new(16_384, 64 * 1024)),
            // Row-only target: falls back to sequential (identity split).
            (wide, bc_arrow::MorselTarget::rows(16_384)),
            // Undersized batches: coalescing path, must fall back to sequential.
            (
                (0..50).map(|_| int_batch(64)).collect(),
                bc_arrow::MorselTarget::new(16_384, 1 << 20),
            ),
            // Mixed with an empty batch interleaved.
            (
                vec![str_batch(&["a", "b"]), str_batch(&[]), int_batch(20_000)],
                bc_arrow::MorselTarget::new(16_384, 1 << 20),
            ),
        ];
        for (batches, target) in cases {
            let seq = morselize(&batches, target);
            let par = morselize_par(&batches, target);
            assert_eq!(par.len(), seq.len(), "morsel count differs");
            for (p, s) in par.iter().zip(&seq) {
                assert_eq!(p.num_rows(), s.num_rows(), "morsel row count differs");
            }
            assert_eq!(total_rows(&par), total_rows(&seq), "total rows differ");
        }
    }

    /// A tiny run followed by a big batch: the run flushes before the big one splits,
    /// so global row order holds across the boundary.
    #[test]
    fn tiny_run_then_large_batch_keeps_order() {
        let mk = |lo: i64, hi: i64| {
            let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int64, false)]));
            let arr = Arc::new(Int64Array::from((lo..hi).collect::<Vec<_>>())) as ArrayRef;
            RecordBatch::try_new(schema, vec![arr]).unwrap()
        };
        let batches = vec![mk(0, 50), mk(50, 100), mk(100, 100_000)];
        let out = morselize(&batches, bc_arrow::MorselTarget::rows(16_384));
        assert_eq!(total_rows(&out), 100_000);
        let first = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(first.value(0), 0, "the tiny run leads");
    }

    /// `remorselize` is a no-op in row-only mode and re-splits over-budget output
    /// in byte-bounded mode.
    #[test]
    fn remorselize_respects_mode() {
        let giant = "y".repeat(100_000);
        let wide = vec![str_batch(&[giant.as_str(); 8])];

        let noop = remorselize(wide.clone(), bc_arrow::MorselTarget::rows(16_384));
        assert_eq!(noop.len(), 1, "row-only remorselize keeps boundaries");

        let split = remorselize(wide, bc_arrow::MorselTarget::new(16_384, 1024));
        assert_eq!(total_rows(&split), 8);
        assert!(
            split.len() > 1,
            "byte-bounded remorselize splits wide output"
        );
    }
}
