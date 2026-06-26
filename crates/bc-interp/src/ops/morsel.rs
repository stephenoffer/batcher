//! Morselization: splitting input batches into row- **and** byte-bounded morsels
//! for the parallel scheduler.
//!
//! The scheduler wants morsels that are small enough to load-balance and whose
//! working set fits in cache, but a fixed row count is byte-blind: 16 384 rows of
//! `Int64` is ~128 KiB, while 16 384 rows of multi-MB blobs is gigabytes. So a
//! morsel is "full" at **either** the row target or the byte budget
//! ([`bc_arrow::MorselTarget`]).
//!
//! Byte sizing is *measured*, not estimated. For an all-fixed-width batch the
//! per-row width is constant, so the split is O(1). For a batch with
//! variable-width columns (strings/blobs/embeddings) we read the offset buffers to
//! get each row's true byte cost and accumulate greedily — so a single row wider
//! than the whole budget becomes its own one-row morsel (a giant cell never
//! co-resides with 16 k others), and intra-batch width variance (999 tiny rows +
//! 1 huge one) is split where it actually is, not where a batch-average would
//! guess. Reading `get_array_memory_size()` on a *slice* cannot do this: an Arrow
//! slice shares the parent buffer and reports its full capacity, so it would
//! over-count every slice.
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

/// Split `batches` into morsels bounded by `target`'s row and byte limits.
///
/// Rows are preserved exactly (same multiset, same order); only the batch
/// boundaries change. Empty batches pass through unchanged.
pub(crate) fn morselize(
    batches: &[RecordBatch],
    target: bc_arrow::MorselTarget,
) -> Vec<RecordBatch> {
    let mut out = Vec::new();
    for b in batches {
        split_batch(&mut out, b, target);
    }
    out
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
    morselize(&batches, target)
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
    // Variable-width batch: accumulate rows by *measured* per-row bytes.
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
