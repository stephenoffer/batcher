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

/// Out-of-core sort: sort the input into `run_target_bytes`-sized runs and spill them
/// (dropping each input batch as we go), then merge the runs with a **bounded-fan-in,
/// streaming** k-way merge. Peak memory is O(`sort_merge_fanin` morsels) regardless of input
/// size: only one batch per run in the active merge group is resident, and the output is
/// streamed back to disk between passes. The result equals a single in-memory
/// `sort_batch` over the whole input. Disk spill uses Arrow-IPC [`DiskSpillStore`].
pub(crate) fn external_merge_sort(
    parts: Vec<RecordBatch>,
    keys: &[SortKey],
    dir: &std::path::Path,
    sort_merge_fanin: usize,
    run_target_bytes: u64,
    codec: bc_runtime::agg::spill::SpillCodec,
    cancel: Option<&bc_resource::CancelToken>,
) -> Result<(Vec<RecordBatch>, u64), InterpError> {
    let Some((mut store, spill_bytes)) = external_sort_to_final_store(
        parts,
        keys,
        dir,
        sort_merge_fanin,
        run_target_bytes,
        codec,
        cancel,
    )?
    else {
        return Ok((Vec::new(), 0));
    };
    // The final run holds the globally sorted result; stream its morsels out.
    let mut out = Vec::new();
    let mut rows = 0u64;
    if let Some(reader) = store.open_reader(0).map_err(InterpError::from)? {
        for batch in reader {
            let batch = batch?;
            rows += batch.num_rows() as u64;
            if batch.num_rows() > 0 {
                out.push(batch);
            }
        }
    }
    // `open_reader` streams, so it cannot make the check `read`/`drain` make for themselves —
    // and a truncated IPC stream reads back as a shorter *valid* one, which here would mean
    // silently returning a sorted prefix of the relation. Making the same comparison the
    // store would have made is what keeps that an error rather than a wrong answer.
    let expected = store.partition_rows(0);
    if rows < expected {
        return Err(InterpError::from(
            bc_runtime::RuntimeError::SpillTruncated {
                dir: dir.display().to_string(),
                partition: 0,
                expected_rows: expected,
                got_rows: rows,
                missing: expected - rows,
            },
        ));
    }
    Ok((out, spill_bytes))
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
    run_target_bytes: u64,
    codec: bc_runtime::agg::spill::SpillCodec,
    cancel: Option<&bc_resource::CancelToken>,
) -> Result<Option<(bc_runtime::agg::spill::DiskSpillStore, u64)>, InterpError> {
    use bc_runtime::agg::spill::{DiskSpillStore, SpillStore};

    // Pass 0: sort the input into runs and spill them, dropping each input batch as it is
    // consumed so the sorted runs never co-reside with the full input.
    //
    // Runs are grown to `run_target_bytes` rather than made one-per-morsel, and that is the
    // dominant cost in a large sort. The merge is multi-pass with fan-in `f`, so it rewrites
    // the entire dataset ceil(log_f(runs)) times: at a 2 MB morsel a 10 GB sort produces
    // ~5,000 single-morsel runs and pays four full passes, where 64 MB runs give ~160 and
    // pay two. Halving the pass count halves the spill I/O, and the input is already
    // resident here, so coalescing costs only the transient sort scratch for one run.
    let mut store = DiskSpillStore::with_codec(
        dir.to_path_buf(),
        run_slots(&parts, run_target_bytes),
        codec,
    )
    .map_err(InterpError::from)?;
    let mut n_runs = 0usize;
    let mut group: Vec<RecordBatch> = Vec::new();
    let mut group_bytes = 0u64;
    // Rows in, so the merge's rows out can be checked against them (see below).
    let mut rows_in = 0u64;
    for b in parts.into_iter() {
        if b.num_rows() == 0 {
            continue;
        }
        group_bytes += b.get_array_memory_size() as u64;
        rows_in += b.num_rows() as u64;
        group.push(b);
        if group_bytes >= run_target_bytes {
            spill_run(&mut group, keys, &mut store, n_runs)?;
            n_runs += 1;
            group_bytes = 0;
        }
    }
    if !group.is_empty() {
        spill_run(&mut group, keys, &mut store, n_runs)?;
        n_runs += 1;
    }
    if n_runs == 0 {
        return Ok(None);
    }
    // Pass-0 wrote the whole input to sorted runs; that is the representative spill volume
    // (merge passes re-spill subsets of it). Captured before the merge loop reassigns `store`.
    let spill_bytes = store.spilled_bytes();

    // Merge passes: each merges groups of <= `fanin` runs into one larger (spilled)
    // run, streaming so only one batch per run is resident. Repeats until a single run
    // remains. Fan-in bounds the resident working set independent of the run count; it
    // is a perf-only knob (default 16, or the control plane's tuning), not the result.
    let fanin = sort_merge_fanin.max(2);
    while n_runs > 1 {
        // A merge pass over a large sort runs for minutes; without this the operator
        // boundary in `par::exec` is the next chance to notice a cancel.
        if cancel.is_some_and(bc_resource::CancelToken::is_cancelled) {
            return Err(InterpError::Cancelled);
        }
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
            // Same reason as pass 0: a merge pass fills output group `g` and never returns
            // to it, so closing here keeps the pass's open descriptors at one output plus
            // the fan-in's readers, rather than one per group.
            next.close_partition(g).map_err(InterpError::from)?;
        }
        store = next;
        n_runs = n_groups;
    }
    // A sort must return every row it was given, and the merge is the only place they could
    // go missing: each pass writes runs and reads them back, so a spill file that lost its
    // tail turns into a *sorted prefix* of the relation rather than an error — an IPC stream
    // truncated at a message boundary reads back as a shorter valid stream. The store already
    // counts rows per partition, so comparing the final run against the input costs nothing:
    // no extra I/O, and it covers every merge pass at once. It also catches the merge simply
    // dropping rows, which no result comparison in a spilling test would notice, because the
    // spilled path is the only one being run.
    let rows_out = store.partition_rows(0);
    if rows_out < rows_in {
        return Err(InterpError::from(
            bc_runtime::RuntimeError::SpillTruncated {
                dir: dir.display().to_string(),
                partition: 0,
                expected_rows: rows_in,
                got_rows: rows_out,
                missing: rows_in - rows_out,
            },
        ));
    }
    Ok(Some((store, spill_bytes)))
}

/// A streaming reader over one spilled run's batches.
type RunReader = arrow::ipc::reader::StreamReader<std::io::BufReader<std::fs::File>>;

/// Default target size of a pass-0 sorted run when the caller has no operator envelope to
/// derive one from (the quantile/median spill paths).
///
/// Deliberately modest: the run is built by concatenating the accumulated morsels and
/// sorting the result, so the transient scratch is about twice this on top of the still
/// resident input. 64 MiB is large enough to collapse the run count by one to two orders of
/// magnitude against per-morsel runs — which is where the merge-pass saving comes from —
/// and small enough that the scratch is noise next to any envelope big enough to have
/// spilled in the first place.
pub(crate) const DEFAULT_RUN_TARGET_BYTES: u64 = 64 << 20;

/// Upper bound on the number of runs pass 0 can produce, used to size the store's partition
/// vector up front. Exact would require summing every batch's size twice; this over-counts
/// harmlessly (an unused partition is an unwritten file) and never under-counts, because a
/// run is only closed once it has reached `run_target_bytes` — except the final partial run,
/// which the `+ 1` covers.
fn run_slots(parts: &[RecordBatch], run_target_bytes: u64) -> usize {
    let total: u64 = parts.iter().map(|b| b.get_array_memory_size() as u64).sum();
    (total / run_target_bytes.max(1)) as usize + 1
}

/// Sort the accumulated morsels as one run, append it to `store` as partition `slot`, and
/// release both the morsels and the run.
///
/// The writer is closed as soon as the run is written: pass 0 fills a run once and never
/// returns to it, so leaving it open would hold one file descriptor per run until the merge
/// reached it. A sort large enough to spill can have thousands of runs, so that reaches
/// `EMFILE` on precisely the inputs spilling exists to serve, while the disk is nowhere near
/// full. Closing here bounds pass 0's open descriptors at one.
fn spill_run(
    group: &mut Vec<RecordBatch>,
    keys: &[SortKey],
    store: &mut bc_runtime::agg::spill::DiskSpillStore,
    slot: usize,
) -> Result<(), InterpError> {
    use bc_runtime::agg::spill::SpillStore;

    let run = if group.len() == 1 {
        sort_batch(&group[0], keys, None)?
    } else {
        let combined = super::materialize(group)?;
        // Release the source morsels before sorting, so the run's peak is the concatenated
        // copy plus its sorted output rather than three copies of the run.
        group.clear();
        sort_batch(&combined, keys, None)?
    };
    group.clear();
    store.append(slot, &run).map_err(InterpError::from)?;
    store.close_partition(slot).map_err(InterpError::from)?;
    Ok(())
}

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
///
/// A `Null`-typed (all-null) key is coerced to a constant column so arrow's `RowConverter`
/// can encode it — the same substitution the in-memory sort applies (see
/// [`super::normalize_sort_key`]) — so the spilling merge orders rows identically to the
/// serial oracle. Both the converter's [`SortField`]s (built from these arrays' types) and
/// the per-batch row conversion go through here, so they stay aligned.
fn eval_sort_keys(batch: &RecordBatch, keys: &[SortKey]) -> Result<Vec<ArrayRef>, InterpError> {
    keys.iter()
        .map(|k| {
            k.expr
                .eval(batch)
                .map(super::normalize_sort_key)
                .map_err(InterpError::from)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use bc_runtime::agg::spill::SpillCodec;
    use std::sync::Arc;

    /// A `(id: Int64, f: Float64)` batch.
    fn fbatch(ids: &[i64], fs: &[f64]) -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("f", DataType::Float64, true),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int64Array::from(ids.to_vec())),
                Arc::new(Float64Array::from(fs.to_vec())),
            ],
        )
        .unwrap()
    }

    /// The exact (id, f-bits) sequence a set of batches produces, in row order — so a
    /// sort can be compared order-dependently (a multiset compare cannot see a sort bug).
    fn seq(batches: &[RecordBatch]) -> Vec<(i64, u64)> {
        let mut out = Vec::new();
        for b in batches {
            let ids = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let fs = b.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
            for i in 0..b.num_rows() {
                out.push((ids.value(i), fs.value(i).to_bits()));
            }
        }
        out
    }

    /// The external (spilling) merge sort must equal the in-memory `sort_batch` oracle
    /// **row-for-row** on a NaN/-0.0/0.0-bearing float key with ties — the exact shape
    /// CLAUDE.md flags (a float sort key under spill shipping unsorted). The in-run
    /// per-morsel sort uses arrow `lexsort`; the cross-run merge uses the arrow row
    /// format — they must agree on where NaN and -0.0 sit, or the merge de-sorts.
    fn assert_external_matches_inmemory(descending: bool) {
        let nan = f64::NAN;
        // Ties on `f` (repeated 2.0, repeated 0.0/-0.0) exercise stability; NaN and the
        // signed zeros exercise the total-order edges.
        let ids: Vec<i64> = (0..12).collect();
        let fs: Vec<f64> = vec![
            2.0, nan, -0.0, 5.0, 2.0, 0.0, -3.0, nan, 1.0, 2.0, 0.0, -0.0,
        ];
        let whole = fbatch(&ids, &fs);
        let keys = vec![SortKey {
            expr: bc_expr::Expr::Col { name: "f".into() },
            descending,
            nulls_first: false,
        }];
        let oracle = super::super::sort_batch(&whole, &keys, None).unwrap();

        // Split into 5 runs of non-uniform size (a count that is NOT a multiple of the
        // run size) so several merge passes run at fan-in 2.
        let parts: Vec<RecordBatch> = vec![
            fbatch(&ids[0..3], &fs[0..3]),
            fbatch(&ids[3..5], &fs[3..5]),
            fbatch(&ids[5..8], &fs[5..8]),
            fbatch(&ids[8..9], &fs[8..9]),
            fbatch(&ids[9..12], &fs[9..12]),
        ];
        let dir = std::env::temp_dir().join(format!(
            "bc_extsort_float_{}_{}",
            descending,
            std::process::id()
        ));
        let (sorted, _) =
            external_merge_sort(parts, &keys, &dir, 2, 1, SpillCodec::None, None).unwrap();
        assert_eq!(
            seq(std::slice::from_ref(&oracle)),
            seq(&sorted),
            "external merge sort diverged from in-memory sort (descending={descending})"
        );
    }

    #[test]
    fn external_sort_float_nan_signed_zero_ties_ascending() {
        assert_external_matches_inmemory(false);
    }

    #[test]
    fn external_sort_float_nan_signed_zero_ties_descending() {
        assert_external_matches_inmemory(true);
    }

    /// A `Null`-typed leading sort key must not crash the spilling merge — arrow's
    /// `RowConverter` rejects the `Null` type just as its sort kernels do. The key is coerced
    /// to a constant (all-equal), so the merge orders by the real secondary key `id`, matching
    /// the in-memory oracle row-for-row.
    /// Coalescing morsels into larger pass-0 runs must not change the result. It is a
    /// different code path from the one-run-per-morsel case — the morsels are concatenated
    /// and sorted as a unit — so both the merge-free case (every morsel in one run) and the
    /// several-runs case are checked against the in-memory oracle row for row.
    #[test]
    fn external_sort_run_coalescing_matches_inmemory() {
        let nan = f64::NAN;
        let ids: Vec<i64> = (0..12).collect();
        let fs: Vec<f64> = vec![
            2.0, nan, -0.0, 5.0, 2.0, 0.0, -3.0, nan, 1.0, 2.0, 0.0, -0.0,
        ];
        let whole = fbatch(&ids, &fs);
        let keys = vec![SortKey {
            expr: bc_expr::Expr::Col { name: "f".into() },
            descending: false,
            nulls_first: false,
        }];
        let oracle = super::super::sort_batch(&whole, &keys, None).unwrap();
        let parts: Vec<RecordBatch> = vec![
            fbatch(&ids[0..3], &fs[0..3]),
            fbatch(&ids[3..5], &fs[3..5]),
            fbatch(&ids[5..8], &fs[5..8]),
            fbatch(&ids[8..9], &fs[8..9]),
            fbatch(&ids[9..12], &fs[9..12]),
        ];
        // A target far above the whole input folds every morsel into one run, so no merge
        // pass runs at all; a mid-sized one gives a handful of multi-morsel runs.
        for target in [u64::from(u32::MAX), 1024] {
            let dir = std::env::temp_dir().join(format!(
                "bc_extsort_coalesce_{target}_{}",
                std::process::id()
            ));
            let (sorted, _) = external_merge_sort(
                parts.clone(),
                &keys,
                &dir,
                2,
                target,
                SpillCodec::None,
                None,
            )
            .unwrap();
            assert_eq!(
                seq(std::slice::from_ref(&oracle)),
                seq(&sorted),
                "coalesced pass-0 runs (target={target}) diverged from the in-memory sort"
            );
        }
    }

    #[test]
    fn external_sort_null_typed_leading_key() {
        use arrow::array::NullArray;
        use arrow::datatypes::{DataType, Field, Schema};

        let schema = Arc::new(Schema::new(vec![
            Field::new("n", DataType::Null, true),
            Field::new("id", DataType::Int64, false),
        ]));
        let mk = |ids: &[i64]| {
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(NullArray::new(ids.len())) as ArrayRef,
                    Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef,
                ],
            )
            .unwrap()
        };
        // ORDER BY n, id — n is all-null (all-equal), so this orders by id ascending.
        let keys = vec![
            SortKey {
                expr: bc_expr::Expr::Col { name: "n".into() },
                descending: false,
                nulls_first: false,
            },
            SortKey {
                expr: bc_expr::Expr::Col { name: "id".into() },
                descending: false,
                nulls_first: false,
            },
        ];
        let parts = vec![mk(&[5, 2, 9]), mk(&[1, 7]), mk(&[3, 8, 4, 6])];
        let dir = std::env::temp_dir().join(format!("bc_extsort_null_{}", std::process::id()));
        let (sorted, _) =
            external_merge_sort(parts, &keys, &dir, 2, 1, SpillCodec::None, None).unwrap();
        let ids: Vec<i64> = sorted
            .iter()
            .flat_map(|b| {
                b.column(1)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        assert_eq!(ids, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
}
