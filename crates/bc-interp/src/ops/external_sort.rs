//! Out-of-core sort: spill sorted runs and merge them with bounded fan-in.
//!
//! Split out of `ops/mod.rs` along the external-sort seam. The public entry points
//! are [`external_merge_sort`] (returns the sorted batches) and
//! [`external_sort_to_final_store`] (returns the final spilled run so a caller can
//! stream it more than once — the spilling quantile/median path). Everything else is
//! the streaming k-way merge machinery, private to this module. Disk spill uses the
//! Arrow-IPC [`bc_runtime::agg::spill::DiskSpillStore`].

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use arrow::array::{Array, ArrayRef, RecordBatch};
use arrow::compute::{interleave, SortOptions};
use arrow::datatypes::SchemaRef;
use arrow::row::{OwnedRow, RowConverter, Rows, SortField};
use bc_ir::SortKey;

use super::sort_batch;
use crate::error::InterpError;

/// Out-of-core sort: sort each input morsel into a run and spill it (dropping the
/// input batch as we go), then merge the runs with a **bounded-fan-in, streaming**
/// k-way merge. Peak memory is O(`sort_merge_fanin` morsels) regardless of input size
/// — only one batch per run in the active merge group is resident, and the output is
/// streamed back to disk between passes. The result equals a single in-memory
/// `sort_batch` over the whole input. Disk spill uses Arrow-IPC [`DiskSpillStore`].
pub(crate) fn external_merge_sort(
    parts: Vec<RecordBatch>,
    keys: &[SortKey],
    dir: &std::path::Path,
    sort_merge_fanin: usize,
    codec: bc_runtime::agg::spill::SpillCodec,
) -> Result<Vec<RecordBatch>, InterpError> {
    let Some(mut store) = external_sort_to_final_store(parts, keys, dir, sort_merge_fanin, codec)?
    else {
        return Ok(Vec::new());
    };
    // The final run holds the globally sorted result; stream its morsels out.
    let mut out = Vec::new();
    if let Some(reader) = store.open_reader(0).map_err(InterpError::from)? {
        for batch in reader {
            let batch = batch?;
            if batch.num_rows() > 0 {
                out.push(batch);
            }
        }
    }
    Ok(out)
}

/// Spill + bounded multi-pass merge, returning the final [`DiskSpillStore`] whose
/// partition 0 holds the globally sorted run (or `None` for empty input). The store
/// is returned so a caller can stream the sorted output via `open_reader(0)` more
/// than once (e.g. the two-pass spilling quantile) without materializing it. Memory
/// is bounded throughout (one batch per run in flight); see [`external_merge_sort`].
pub(crate) fn external_sort_to_final_store(
    parts: Vec<RecordBatch>,
    keys: &[SortKey],
    dir: &std::path::Path,
    sort_merge_fanin: usize,
    codec: bc_runtime::agg::spill::SpillCodec,
) -> Result<Option<bc_runtime::agg::spill::DiskSpillStore>, InterpError> {
    use bc_runtime::agg::spill::{DiskSpillStore, SpillStore};

    // Pass 0: sort each input morsel into a run and spill it, dropping each input
    // batch as it is consumed so the sorted runs never co-reside with the full input.
    let mut store = DiskSpillStore::with_codec(dir.to_path_buf(), parts.len().max(1), codec)
        .map_err(InterpError::from)?;
    let mut n_runs = 0usize;
    for b in parts.into_iter() {
        if b.num_rows() == 0 {
            continue;
        }
        let run = sort_batch(&b, keys, None)?;
        store.append(n_runs, &run).map_err(InterpError::from)?;
        n_runs += 1;
        // `b` and `run` drop here — the input morsel's memory is released.
    }
    if n_runs == 0 {
        return Ok(None);
    }

    // Merge passes: each merges groups of <= `fanin` runs into one larger (spilled)
    // run, streaming so only one batch per run is resident. Repeats until a single run
    // remains. Fan-in bounds the resident working set independent of the run count; it
    // is a perf-only knob (default 16, or the control plane's tuning), not the result.
    let fanin = sort_merge_fanin.max(2);
    while n_runs > 1 {
        let n_groups = n_runs.div_ceil(fanin);
        let mut next = DiskSpillStore::with_codec(dir.to_path_buf(), n_groups, codec)
            .map_err(InterpError::from)?;
        for g in 0..n_groups {
            let lo = g * fanin;
            let hi = (lo + fanin).min(n_runs);
            let mut readers = Vec::with_capacity(hi - lo);
            for i in lo..hi {
                if let Some(r) = store.open_reader(i).map_err(InterpError::from)? {
                    readers.push(r);
                }
            }
            stream_merge_group(readers, keys, &mut next, g)?;
        }
        store = next;
        n_runs = n_groups;
    }
    Ok(Some(store))
}

/// A streaming reader over one spilled run's batches.
type RunReader = arrow::ipc::reader::StreamReader<std::io::BufReader<std::fs::File>>;

/// Build the key-row converter for a run group from a sample batch, baking each
/// key's asc/desc/nulls options into the encoding so encoded rows compare in order.
fn build_key_converter(batch: &RecordBatch, keys: &[SortKey]) -> Result<RowConverter, InterpError> {
    let key_cols = eval_sort_keys(batch, keys)?;
    let fields: Vec<SortField> = key_cols
        .iter()
        .zip(keys)
        .map(|(arr, k)| {
            SortField::new_with_options(
                arr.data_type().clone(),
                SortOptions {
                    descending: k.descending,
                    nulls_first: k.nulls_first,
                },
            )
        })
        .collect();
    Ok(RowConverter::new(fields)?)
}

/// Advance reader `ri` to its next non-empty batch, encoding that batch's key rows.
/// Sets `cur[ri]`/`cur_rows[ri]` to `None` when the reader is exhausted. Builds the
/// shared `converter`/`schema` from the first batch seen across the group.
#[allow(clippy::too_many_arguments)]
fn load_next_run_batch(
    ri: usize,
    readers: &mut [RunReader],
    cur: &mut [Option<RecordBatch>],
    cur_rows: &mut [Option<Rows>],
    idx: &mut [usize],
    converter: &mut Option<RowConverter>,
    schema: &mut Option<SchemaRef>,
    keys: &[SortKey],
) -> Result<(), InterpError> {
    loop {
        match readers[ri].next() {
            Some(batch) => {
                let batch = batch?;
                if batch.num_rows() == 0 {
                    continue;
                }
                if schema.is_none() {
                    *schema = Some(batch.schema());
                }
                if converter.is_none() {
                    *converter = Some(build_key_converter(&batch, keys)?);
                }
                let key_cols = eval_sort_keys(&batch, keys)?;
                let rows = converter
                    .as_ref()
                    .expect("converter built above")
                    .convert_columns(&key_cols)?;
                cur[ri] = Some(batch);
                cur_rows[ri] = Some(rows);
                idx[ri] = 0;
                return Ok(());
            }
            None => {
                cur[ri] = None;
                cur_rows[ri] = None;
                return Ok(());
            }
        }
    }
}

/// Flush the accumulated `(slot, row)` selections into one output batch via
/// `interleave` and append it to `store`'s `out_partition`. Exhausted (`None`) slots
/// get a type-correct empty placeholder; they are never indexed by `sel` because a
/// flush always precedes loading a slot's next batch.
fn flush_selection(
    sel: &mut Vec<(usize, usize)>,
    cur: &[Option<RecordBatch>],
    schema: &SchemaRef,
    store: &mut dyn bc_runtime::agg::spill::SpillStore,
    out_partition: usize,
) -> Result<(), InterpError> {
    if sel.is_empty() {
        return Ok(());
    }
    let mut cols: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
    for (c, field) in schema.fields().iter().enumerate() {
        let owned: Vec<ArrayRef> = cur
            .iter()
            .map(|b| match b {
                Some(batch) => batch.column(c).clone(),
                None => arrow::array::new_empty_array(field.data_type()),
            })
            .collect();
        let refs: Vec<&dyn Array> = owned.iter().map(|a| a.as_ref()).collect();
        cols.push(interleave(&refs, sel)?);
    }
    let batch = RecordBatch::try_new(schema.clone(), cols)?;
    store
        .append(out_partition, &batch)
        .map_err(InterpError::from)?;
    sel.clear();
    Ok(())
}

/// Streaming k-way merge of `readers` (each a sorted run) into `store`'s
/// `out_partition`. Holds at most one batch per reader plus one output morsel of
/// `(slot, row)` selections, so memory is bounded by the fan-in — not the run sizes.
fn stream_merge_group(
    mut readers: Vec<RunReader>,
    keys: &[SortKey],
    store: &mut dyn bc_runtime::agg::spill::SpillStore,
    out_partition: usize,
) -> Result<(), InterpError> {
    let k = readers.len();
    if k == 0 {
        return Ok(());
    }
    let mut cur: Vec<Option<RecordBatch>> = (0..k).map(|_| None).collect();
    let mut cur_rows: Vec<Option<Rows>> = (0..k).map(|_| None).collect();
    let mut idx: Vec<usize> = vec![0; k];
    let mut converter: Option<RowConverter> = None;
    let mut schema: Option<SchemaRef> = None;
    // Min-heap over the current head key of each live reader (owned, so it survives
    // the reader advancing to its next batch).
    let mut heap: BinaryHeap<Reverse<(OwnedRow, usize)>> = BinaryHeap::new();

    for ri in 0..k {
        load_next_run_batch(
            ri,
            &mut readers,
            &mut cur,
            &mut cur_rows,
            &mut idx,
            &mut converter,
            &mut schema,
            keys,
        )?;
        if let Some(rows) = &cur_rows[ri] {
            heap.push(Reverse((rows.row(0).owned(), ri)));
        }
    }
    // The output schema is fixed once the first batch is seen; `schema` (the Option)
    // stays threaded through later `load_next_run_batch` calls (a no-op once set).
    let Some(out_schema) = schema.clone() else {
        return Ok(()); // every reader was empty
    };

    let target = bc_arrow::DEFAULT_MORSEL_ROWS;
    let mut sel: Vec<(usize, usize)> = Vec::with_capacity(target);

    while let Some(Reverse((_key, ri))) = heap.pop() {
        sel.push((ri, idx[ri]));
        idx[ri] += 1;
        let n = cur[ri].as_ref().map_or(0, |b| b.num_rows());
        if idx[ri] < n {
            heap.push(Reverse((
                cur_rows[ri]
                    .as_ref()
                    .expect("live cursor")
                    .row(idx[ri])
                    .owned(),
                ri,
            )));
        } else {
            // Reader `ri` exhausted its current batch. The pending selections still
            // reference the current batches, so flush before swapping `ri`'s batch.
            flush_selection(&mut sel, &cur, &out_schema, store, out_partition)?;
            load_next_run_batch(
                ri,
                &mut readers,
                &mut cur,
                &mut cur_rows,
                &mut idx,
                &mut converter,
                &mut schema,
                keys,
            )?;
            if let Some(rows) = &cur_rows[ri] {
                heap.push(Reverse((rows.row(0).owned(), ri)));
            }
        }
        if sel.len() >= target {
            flush_selection(&mut sel, &cur, &out_schema, store, out_partition)?;
        }
    }
    flush_selection(&mut sel, &cur, &out_schema, store, out_partition)
}

/// Evaluate the sort-key expressions of `batch` into their key columns.
fn eval_sort_keys(batch: &RecordBatch, keys: &[SortKey]) -> Result<Vec<ArrayRef>, InterpError> {
    keys.iter()
        .map(|k| k.expr.eval(batch).map_err(InterpError::from))
        .collect()
}
