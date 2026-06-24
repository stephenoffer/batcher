//! Row-reshaping per-batch primitives: `unnest`/`explode`, `unpivot`/`melt`, and
//! content-hash `sample`. Split out of `ops` to keep that module under the size
//! limit; these share the "change the row count of one batch, statelessly" shape and
//! are reused unchanged by the sequential, parallel, and distributed executors.

use std::collections::BinaryHeap;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, GenericListArray, Int64Array, LargeListArray, ListArray,
    OffsetSizeTrait, RecordBatch, StringArray, UInt32Array,
};
use arrow::compute::{concat, filter_record_batch, take};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::row::{OwnedRow, RowConverter, SortField};

use crate::error::InterpError;

/// Append a sequential row-index column (`alias`) starting at `offset`, numbered in
/// batch-arrival order across the whole input (Polars `with_row_index`). A single
/// counter runs over the batches, so the result matches on the sequential and
/// parallel paths whenever the upstream preserves row order.
pub(crate) fn add_row_ids(
    batches: &[RecordBatch],
    alias: &str,
    offset: i64,
) -> Result<Vec<RecordBatch>, InterpError> {
    let mut next = offset;
    let mut out = Vec::with_capacity(batches.len());
    for b in batches {
        let n = b.num_rows();
        let ids: Int64Array = (next..next + n as i64).collect();
        next += n as i64;
        // Prepend the index column (Polars `with_row_index` convention).
        let mut fields: Vec<Arc<Field>> = vec![Arc::new(Field::new(alias, DataType::Int64, false))];
        fields.extend(b.schema().fields().iter().cloned());
        let mut columns: Vec<ArrayRef> = vec![Arc::new(ids)];
        columns.extend(b.columns().iter().cloned());
        out.push(RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            columns,
        )?);
    }
    Ok(out)
}

/// Explode the list/array column `column` into one row per element, binding the
/// element values to `alias`. The named column is replaced in place; every other
/// column is gathered (repeated) once per element. Null/empty lists yield no rows
/// (DuckDB `UNNEST` semantics), so an all-empty batch produces an empty batch with
/// the post-unnest schema. Stateless and per-batch — the parallel and distributed
/// paths reuse this unchanged.
pub(crate) fn unnest_batch(
    batch: &RecordBatch,
    column: &str,
    alias: &str,
) -> Result<RecordBatch, InterpError> {
    let col = batch
        .column_by_name(column)
        .ok_or_else(|| InterpError::UnnestUnknownColumn(column.to_string()))?;
    let (parent_idx, exploded) = match col.data_type() {
        DataType::List(_) => explode_list(col.as_any().downcast_ref::<ListArray>().unwrap()),
        DataType::LargeList(_) => {
            explode_list(col.as_any().downcast_ref::<LargeListArray>().unwrap())
        }
        other => {
            return Err(InterpError::UnnestNotList {
                column: column.to_string(),
                got: other.to_string(),
            })
        }
    }?;
    let parent_indices = UInt32Array::from(parent_idx);

    // Output preserves input column order, replacing the exploded column in place
    // with its element values (renamed to `alias`); other columns are gathered by
    // the parent index so each repeats once per element.
    let schema = batch.schema();
    let mut fields = Vec::with_capacity(batch.num_columns());
    let mut columns = Vec::with_capacity(batch.num_columns());
    for (i, field) in schema.fields().iter().enumerate() {
        if field.name() == column {
            fields.push(Field::new(alias, exploded.data_type().clone(), true));
            columns.push(exploded.clone());
        } else {
            let gathered = take(batch.column(i).as_ref(), &parent_indices, None)?;
            fields.push(Field::new(
                field.name(),
                gathered.data_type().clone(),
                field.is_nullable(),
            ));
            columns.push(gathered);
        }
    }
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

/// Build the (parent-row-index, exploded-values) pair for a list array of either
/// offset width. A null list entry contributes no rows regardless of its offsets.
fn explode_list<O: OffsetSizeTrait>(
    list: &GenericListArray<O>,
) -> Result<(Vec<u32>, ArrayRef), InterpError> {
    let offsets = list.value_offsets();
    let mut parent_idx: Vec<u32> = Vec::new();
    let mut child_idx: Vec<u32> = Vec::new();
    for i in 0..list.len() {
        if list.is_null(i) {
            continue;
        }
        let start = offsets[i].as_usize();
        let end = offsets[i + 1].as_usize();
        for j in start..end {
            parent_idx.push(i as u32);
            child_idx.push(j as u32);
        }
    }
    let child_indices = UInt32Array::from(child_idx);
    let exploded = take(list.values().as_ref(), &child_indices, None)?;
    Ok((parent_idx, exploded))
}

/// Reshape one batch wide → long (SQL `UNPIVOT` / `melt`). For `n` input rows and
/// `k` `on` columns, emits `n * k` rows: the `index` columns repeat (tiled), a
/// `variable_name` Utf8 column names the source column, and `value_name` stacks the
/// `on` columns' values (which must share a type — `concat` enforces it).
pub(crate) fn unpivot_batch(
    batch: &RecordBatch,
    index: &[String],
    on: &[String],
    variable_name: &str,
    value_name: &str,
) -> Result<RecordBatch, InterpError> {
    let n = batch.num_rows();
    let k = on.len();
    let lookup = |name: &str| {
        batch
            .column_by_name(name)
            .ok_or_else(|| InterpError::UnpivotUnknownColumn(name.to_string()))
    };

    // Parent index tiles 0..n once per `on` column, so each index column repeats
    // and lines up with the stacked values below (column-major row order).
    let mut parent: Vec<u32> = Vec::with_capacity(n * k);
    for _ in 0..k {
        parent.extend(0..n as u32);
    }
    let parent_indices = UInt32Array::from(parent);

    let mut fields: Vec<Field> = Vec::with_capacity(index.len() + 2);
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(index.len() + 2);

    for name in index {
        let gathered = take(lookup(name)?.as_ref(), &parent_indices, None)?;
        fields.push(Field::new(name, gathered.data_type().clone(), true));
        columns.push(gathered);
    }

    // The `variable` column: each `on` name repeated `n` times, in `on` order.
    let mut var: Vec<&str> = Vec::with_capacity(n * k);
    for name in on {
        for _ in 0..n {
            var.push(name);
        }
    }
    fields.push(Field::new(variable_name, DataType::Utf8, false));
    columns.push(Arc::new(StringArray::from(var)));

    // The `value` column: the `on` columns concatenated in order (same type).
    let value_arrays: Vec<ArrayRef> = on
        .iter()
        .map(|name| lookup(name).cloned())
        .collect::<Result<_, _>>()?;
    let refs: Vec<&dyn Array> = value_arrays.iter().map(|a| a.as_ref()).collect();
    let value = concat(&refs)?;
    fields.push(Field::new(value_name, value.data_type().clone(), true));
    columns.push(value);

    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

/// Keep a `fraction` of rows by a stable per-row hash seeded with `seed`. Encoding
/// each row to comparable bytes (the same `RowConverter` the sort path uses) and
/// hashing those means the keep/drop decision depends only on row *content* and the
/// seed — never on batch boundaries or worker count — so the sample is deterministic
/// and identical single-node or distributed.
pub(crate) fn sample_batch(
    batch: &RecordBatch,
    fraction: f64,
    seed: u64,
) -> Result<RecordBatch, InterpError> {
    if fraction >= 1.0 {
        return Ok(batch.clone());
    }
    let fields: Vec<SortField> = batch
        .schema()
        .fields()
        .iter()
        .map(|f| SortField::new(f.data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields)?;
    let rows = converter.convert_columns(batch.columns())?;
    // Threshold scales the keep-probability over the full u64 range.
    let threshold = (fraction.clamp(0.0, 1.0) * (u64::MAX as f64)) as u64;
    let keep: BooleanArray = (0..batch.num_rows())
        .map(|i| Some(fnv1a_seeded(rows.row(i).as_ref(), seed) <= threshold))
        .collect();
    Ok(filter_record_batch(batch, &keep)?)
}

/// Keep the `n` rows with the smallest per-row hash (a fixed-count sample). The
/// global n-smallest hashes are the same regardless of how the input is chunked or
/// partitioned, so this is **deterministic and partition-independent**, and it
/// merges (each partition's n-smallest, then the global n-smallest). A breaker:
/// it must see all rows. Memory is bounded to a size-`n` heap of row encodings, not
/// the whole input. Hash ties break by row content (so identical-content rows are
/// interchangeable and the output multiset is deterministic).
pub(crate) fn sample_n_batches(
    batches: &[RecordBatch],
    n: usize,
    seed: u64,
) -> Result<Vec<RecordBatch>, InterpError> {
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    if n == 0 {
        return Ok(Vec::new());
    }
    if total <= n {
        return Ok(batches.to_vec()); // keep everything
    }
    let schema = batches[0].schema();
    let fields: Vec<SortField> = schema
        .fields()
        .iter()
        .map(|f| SortField::new(f.data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields)?;

    // Max-heap of the n smallest `(hash, row, batch, row_idx)` seen so far: the heap
    // top is the largest kept entry, evicted when a smaller one arrives.
    let mut heap: BinaryHeap<(u64, OwnedRow, usize, usize)> = BinaryHeap::with_capacity(n + 1);
    for (bi, b) in batches.iter().enumerate() {
        let rows = converter.convert_columns(b.columns())?;
        for ri in 0..b.num_rows() {
            let r = rows.row(ri);
            let entry = (fnv1a_seeded(r.as_ref(), seed), r.owned(), bi, ri);
            if heap.len() < n {
                heap.push(entry);
            } else if entry < *heap.peek().expect("heap is full") {
                heap.pop();
                heap.push(entry);
            }
        }
    }

    // Gather the kept row indices per batch (sorted, so each output batch keeps the
    // input's relative row order).
    let mut per_batch: Vec<Vec<u32>> = vec![Vec::new(); batches.len()];
    for (_, _, bi, ri) in heap {
        per_batch[bi].push(ri as u32);
    }
    let mut out = Vec::new();
    for (bi, b) in batches.iter().enumerate() {
        if per_batch[bi].is_empty() {
            continue;
        }
        per_batch[bi].sort_unstable();
        let idx = arrow::array::UInt32Array::from(std::mem::take(&mut per_batch[bi]));
        let cols = b
            .columns()
            .iter()
            .map(|c| take(c.as_ref(), &idx, None))
            .collect::<Result<Vec<_>, _>>()?;
        out.push(RecordBatch::try_new(b.schema(), cols)?);
    }
    Ok(out)
}

/// A fixed, version-stable per-row hash: FNV-1a over `bytes` (seeded), then a
/// splitmix64 avalanche finalizer. The finalizer is essential — plain FNV's high
/// bits barely move for small fixed-width row encodings (e.g. a single Int64
/// column), which would skew the sample fraction; the avalanche makes every output
/// bit depend on all input bits, so a threshold on the result honors `fraction`.
fn fnv1a_seeded(bytes: &[u8], seed: u64) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ seed;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // splitmix64 finalizer (strong avalanche).
    hash ^= hash >> 30;
    hash = hash.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    hash ^= hash >> 27;
    hash = hash.wrapping_mul(0x94d0_49bb_1331_11eb);
    hash ^= hash >> 31;
    hash
}
