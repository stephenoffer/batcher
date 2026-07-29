//! Distributed-execution primitives.
//!
//! These are the building blocks the (Python) distributed orchestrator composes
//! across Ray workers. They are exactly the mergeable pieces proven in
//! `bc-runtime`, surfaced at a granularity the orchestrator can map over
//! partitions:
//!
//! * [`partial_aggregate`] — a map task runs this on its partition, emitting
//!   *partial state* (group keys + per-aggregate state columns) as one batch.
//! * [`partition_batches`] — hash-shuffle a batch into one bucket per reducer.
//! * [`combine_finalize`] — a reduce task merges the partial states routed to it
//!   and finalizes them into output rows.
//!
//! `combine_finalize(partition(partial(pₖ)))` over all partitions equals a
//! single-node aggregation — the same property the `bc-runtime` tests assert,
//! now spanning machines.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch, RecordBatchOptions};
use arrow::datatypes::{Field, Schema};
use arrow::error::ArrowError;
use arrow::ipc::reader::StreamReader;
use bc_ir::{AggregateItem, ProjectionItem};
use bc_runtime::agg::spill::{
    combine_finalize_spilling as rt_combine_finalize_spilling, DiskSpillStore, SpillCodec,
    SpillStore,
};
use bc_runtime::{agg, shuffle};
use rayon::prelude::*;

use crate::error::InterpError;
use crate::ops;

/// Map step: aggregate one partition into partial state.
///
/// The output batch is `[group_key_columns..., state_columns...]`; state column
/// names are synthetic (`__s{agg}_{col}`) and decoded by [`combine_finalize`]
/// using the aggregate list (only `mean` has two state columns).
///
/// Parallel across the morsels of the input (rayon): a distributed map worker folds
/// tens of millions of rows here, and a single-threaded partial would pin it to one
/// core while the read (now ~16-way concurrent) finishes in a fraction of the time —
/// leaving the fold the whole bottleneck. Partial-aggregate per morsel and `combine`
/// (the same mergeable path the parallel executor uses): the combine of per-morsel
/// partials equals one partial over the whole input, so the result is bit-identical to
/// the sequential fold — only the core count changes. A single morsel stays sequential.
pub fn partial_aggregate(
    group_keys: &[ProjectionItem],
    aggregates: &[AggregateItem],
    batches: &[RecordBatch],
) -> Result<RecordBatch, InterpError> {
    // The input is already morsel-sized (the map prefix's output), so partial-aggregate
    // each batch in parallel and `combine` — no `materialize` concat (it would serialize
    // the whole partition through one core, defeating the point). One batch stays
    // sequential. Runs on a dedicated pool sized to the worker's cores: Ray actors can
    // leave the *global* rayon pool sized to 1 (it is built before the cgroup affinity
    // lands), so `par_iter` on it would run single-threaded — the explicit pool is what
    // actually spreads the fold across all cores.
    let non_empty: Vec<&RecordBatch> = batches.iter().filter(|b| b.num_rows() > 0).collect();
    if non_empty.is_empty() {
        let combined = ops::materialize(batches).map_err(|_| InterpError::EmptyAggregateInput)?;
        let partial = ops::eval_partial(&combined, group_keys, aggregates)?;
        return partial_to_batch(group_keys, &partial);
    }
    if non_empty.len() == 1 {
        let partial = ops::eval_partial(non_empty[0], group_keys, aggregates)?;
        return partial_to_batch(group_keys, &partial);
    }
    let funcs = ops::agg_funcs(aggregates);
    let agg_jit = ops::compile_agg(group_keys, aggregates, non_empty[0]);
    // Share the executor's width-sized pool (NOT rayon's global pool, which a Ray worker
    // leaves at 1 thread — see `par::execute_parallel_with_metrics`), so the fold spreads
    // across every core. `usable_cores` reads the actor's applied CPU affinity *and* the
    // cgroup CPU quota, which is what a Ray/K8s worker is actually limited by.
    let width = bc_arrow::usable_cores();
    let partials: Vec<agg::Partial> = crate::par::pool_for(width)?.install(|| {
        non_empty
            .par_iter()
            .map(|b| ops::eval_partial_jit(b, group_keys, aggregates, &agg_jit))
            .collect::<Result<_, InterpError>>()
    })?;
    let merged = agg::combine(&partials, &funcs)?;
    partial_to_batch(group_keys, &merged)
}

/// Per-aggregate partial-state column count (mean keeps sum+count; var/stddev keep
/// count+mean+M2; everything else a single accumulator).
fn agg_widths(aggregates: &[AggregateItem]) -> Vec<usize> {
    // Reuse the runtime's `state_arity` (the single source of truth) rather than a
    // duplicate table here — so a new multi-column aggregate (e.g. arg_min/arg_max)
    // works in the distributed path automatically.
    ops::agg_funcs(aggregates)
        .iter()
        .map(|f| f.state_arity())
        .collect()
}

/// Serialize a `Partial` into the wire batch `partial_aggregate` emits:
/// `[group_key_columns..., state_columns...]` with synthetic state names.
fn partial_to_batch(
    group_keys: &[ProjectionItem],
    partial: &agg::Partial,
) -> Result<RecordBatch, InterpError> {
    // Exact wire width: one column per group key plus every aggregate's state columns.
    let ncols = group_keys.len() + partial.states.iter().map(|s| s.len()).sum::<usize>();
    let mut fields = Vec::with_capacity(ncols);
    let mut columns = Vec::with_capacity(ncols);
    for (k, c) in group_keys.iter().zip(&partial.group_columns) {
        fields.push(Field::new(&k.alias, c.data_type().clone(), true));
        columns.push(c.clone());
    }
    for (a, state) in partial.states.iter().enumerate() {
        for (c, col) in state.iter().enumerate() {
            fields.push(Field::new(
                format!("__s{a}_{c}"),
                col.data_type().clone(),
                true,
            ));
            columns.push(col.clone());
        }
    }
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

/// Decode partial-state batches back into `Partial`s, splitting the synthetic
/// state columns by each aggregate's width.
///
/// The batches arrive from other Ray workers, so their column count is validated
/// against the wire format (`n_keys + Σ widths`) before any column is indexed: a
/// version-skewed or corrupt partial yields a typed [`InterpError::MalformedPartial`]
/// the orchestrator can treat as a failed task (recompute) rather than panicking the
/// reducer on an out-of-bounds access.
fn batches_to_partials(
    n_keys: usize,
    widths: &[usize],
    partial_batches: &[RecordBatch],
) -> Result<Vec<agg::Partial>, InterpError> {
    let state: usize = widths.iter().sum();
    let expected = n_keys + state;
    let mut partials = Vec::with_capacity(partial_batches.len());
    for batch in partial_batches {
        if batch.num_columns() != expected {
            return Err(InterpError::MalformedPartial {
                expected,
                n_keys,
                state,
                got: batch.num_columns(),
            });
        }
        let group_columns: Vec<ArrayRef> = (0..n_keys).map(|i| batch.column(i).clone()).collect();
        let mut states = Vec::with_capacity(widths.len());
        let mut off = n_keys;
        for &w in widths {
            states.push((0..w).map(|c| batch.column(off + c).clone()).collect());
            off += w;
        }
        partials.push(agg::Partial {
            group_columns,
            states,
        });
    }
    Ok(partials)
}

/// Combine step (no finalize): merge partial-state batches into a single partial
/// batch in the *same* wire format. This lets a streaming/incremental driver keep
/// one running state, bounded by the number of groups, instead of accumulating
/// every micro-batch's partials before a final `combine_finalize`.
pub fn combine(
    group_keys: &[ProjectionItem],
    aggregates: &[AggregateItem],
    partial_batches: &[RecordBatch],
) -> Result<RecordBatch, InterpError> {
    let widths = agg_widths(aggregates);
    let partials = batches_to_partials(group_keys.len(), &widths, partial_batches)?;
    if partials.is_empty() {
        return Err(InterpError::EmptyAggregateInput);
    }
    let funcs = ops::agg_funcs(aggregates);
    let merged = in_worker_pool(|| agg::combine(&partials, &funcs))??;
    partial_to_batch(group_keys, &merged)
}

/// Reduce step: merge the partial-state batches routed to one reducer and
/// finalize them into the output schema (group aliases + aggregate aliases).
pub fn combine_finalize(
    group_keys: &[ProjectionItem],
    aggregates: &[AggregateItem],
    partial_batches: &[RecordBatch],
) -> Result<RecordBatch, InterpError> {
    let widths = agg_widths(aggregates);
    let partials = batches_to_partials(group_keys.len(), &widths, partial_batches)?;
    if partials.is_empty() {
        return Err(InterpError::EmptyAggregateInput);
    }

    let funcs = ops::agg_funcs(aggregates);
    let merged = in_worker_pool(|| agg::combine(&partials, &funcs))??;
    let agg_cols = agg::finalize(&funcs, &merged)?;
    ops::build_agg_batch(group_keys, aggregates, &merged.group_columns, &agg_cols)
}

/// Spilling reduce step: the out-of-core sibling of [`combine_finalize`].
///
/// A reducer must merge every partial routed to its key slice and finalize the result. When
/// that slice's group cardinality is large — a high-cardinality `GROUP BY`, a `DISTINCT`, a
/// `COUNT(DISTINCT)` — the merged state can exceed one worker's RAM, and the in-memory
/// [`combine_finalize`] OOMs. This reads the reducer's partials from their on-disk shuffle
/// files **one at a time**, grace-partitions them to disk by the group key, and merges one
/// hash partition at a time, so the reducer's peak memory is bounded to a single file plus a
/// single partition — independent of how large this reducer's slice of the dataset is. It is
/// the distributed arm of the single-node spilling aggregate: it reuses the *same* recursive
/// [`bc_runtime::agg::spill::combine_finalize_spilling`], so the result is identical to
/// [`combine_finalize`] over the same partials (group order differs — these are unordered
/// relations) and to the single-node aggregate. That is the mergeable-algebra invariant,
/// now holding out-of-core across machines.
///
/// `input_paths` are the Arrow-IPC *stream* files the shuffle wrote (one per mapper for this
/// reducer). `budget_bytes` is the reducer's memory envelope; `spill_dir` is scratch for the
/// grace partitions; `spill_compression` selects the IPC codec (see
/// [`SpillCodec::from_config_str`]). Reading the files here — rather than handing the reducer
/// a fully-materialized batch list — is what makes the bound hold end to end.
pub fn combine_finalize_spilling(
    group_keys: &[ProjectionItem],
    aggregates: &[AggregateItem],
    input_paths: &[PathBuf],
    budget_bytes: usize,
    spill_dir: &Path,
    spill_compression: Option<&str>,
) -> Result<RecordBatch, InterpError> {
    let widths = agg_widths(aggregates);
    let funcs = ops::agg_funcs(aggregates);
    let n_keys = group_keys.len();

    let partitions = reduce_grace_partitions(input_paths, budget_bytes);
    let codec = SpillCodec::from_config_str(spill_compression);
    let dir = spill_dir.join(format!("reduce-{partitions}p"));
    let mut store = DiskSpillStore::with_codec(dir, partitions, codec)?;

    // A lazy, one-file-at-a-time stream of partials. The spill phase of
    // `combine_finalize_spilling` routes each partial to a hash partition and drops it, so
    // only one shuffle file's partials are ever resident. The iterator cannot yield a
    // `Result`, so a read/decode error is stashed and surfaced *after* the fold — a stashed
    // error stops the iterator early, the spiller finalizes whatever it routed, and we
    // discard that partial result and return the error.
    let mut read_err: Option<InterpError> = None;
    let res = {
        let iter = PartialFiles {
            paths: input_paths.iter(),
            buf: Vec::new().into_iter(),
            n_keys,
            widths: &widths,
            err: &mut read_err,
        };
        rt_combine_finalize_spilling(iter, &funcs, &mut store, budget_bytes)?
    };
    if let Some(e) = read_err {
        return Err(e);
    }
    // Touch the store's measured spill volume so a future metrics side-channel can read it;
    // for now it only anchors `store`'s lifetime past the fold. Correctness never depends on it.
    let _spilled = store.spilled_bytes();
    if res.group_columns.is_empty() && res.agg_columns.is_empty() {
        // The reducer's whole slice was empty (it received only empty keyed shards). Report a
        // zero-row result — the reducer's own schema is discarded downstream (the driver
        // supplies the result schema), and this matches the in-memory reduce's zero-row output
        // for the same input. A 0-column batch needs an explicit row count in Arrow.
        return Ok(RecordBatch::try_new_with_options(
            Arc::new(Schema::empty()),
            Vec::new(),
            &RecordBatchOptions::new().with_row_count(Some(0)),
        )?);
    }
    ops::build_agg_batch(group_keys, aggregates, &res.group_columns, &res.agg_columns)
}

/// Initial grace fan-out for the reducer, estimated from the partials' *on-disk* bytes
/// (cheap — a `stat` per file, no read). On-disk bytes are a floor for the in-memory state
/// (IPC never expands), so an under-estimate is safe: `combine_finalize_spilling` re-partitions
/// any partition still over budget once it sees the true in-memory size. At least 2 (spilling
/// with 1 partition saves no memory); capped so a pathological ratio cannot open a huge fan-out.
fn reduce_grace_partitions(paths: &[PathBuf], budget_bytes: usize) -> usize {
    let total: u64 = paths
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .sum();
    let budget = budget_bytes.max(1) as u64;
    crate::spill_split::grace_bucket_count(total as usize, budget as usize)
}

/// Read one Arrow-IPC stream file (a shuffle bucket for this reducer) into its partials.
/// Bounded by a single file — the reducer never holds more than one at a time.
fn read_partials_file(
    path: &Path,
    n_keys: usize,
    widths: &[usize],
) -> Result<Vec<agg::Partial>, InterpError> {
    let file = std::fs::File::open(path).map_err(ArrowError::from)?;
    let reader = StreamReader::try_new(std::io::BufReader::new(file), None)?;
    let batches = reader.collect::<Result<Vec<RecordBatch>, ArrowError>>()?;
    let partials = batches_to_partials(n_keys, widths, &batches)?;
    // Drop *keyed* zero-row partials (empty shuffle shards): they carry no groups and no
    // state, so routing them is a no-op — but if a reducer receives *only* such shards, the
    // spiller must see no partials at all and return the empty result, rather than routing
    // zero rows into every partition and then `concat`-ing an empty set of arrays (which
    // errors). A *global* (n_keys == 0) partial always carries one state row (COUNT over
    // nothing is 0), so it is never dropped.
    Ok(partials
        .into_iter()
        .filter(|p| n_keys == 0 || p.group_columns.first().is_some_and(|c| !c.is_empty()))
        .collect())
}

/// A lazy iterator over the partials in a list of shuffle files, opened one file at a time.
/// Yields `Partial`s for `combine_finalize_spilling` to route+drop; a read error is recorded
/// in `err` and ends iteration (see [`combine_finalize_spilling`]).
struct PartialFiles<'a> {
    paths: std::slice::Iter<'a, PathBuf>,
    buf: std::vec::IntoIter<agg::Partial>,
    n_keys: usize,
    widths: &'a [usize],
    err: &'a mut Option<InterpError>,
}

impl Iterator for PartialFiles<'_> {
    type Item = agg::Partial;

    fn next(&mut self) -> Option<agg::Partial> {
        loop {
            if let Some(p) = self.buf.next() {
                return Some(p);
            }
            if self.err.is_some() {
                return None;
            }
            let path = self.paths.next()?;
            match read_partials_file(path, self.n_keys, self.widths) {
                Ok(v) => self.buf = v.into_iter(),
                Err(e) => {
                    *self.err = Some(e);
                    return None;
                }
            }
        }
    }
}

/// Run a rayon-parallel data-plane step inside the worker's **width-sized** pool rather
/// than rayon's global pool. A Ray map/reduce actor leaves the global pool sized to one
/// thread — it is built before the actor's cgroup CPU affinity lands — so the
/// rayon-parallel kernels in `bc_runtime` (the high-cardinality combine's regroup/merge,
/// the shuffle's hash + scatter) would pin a worker that processes millions of rows to a
/// single core. The width-sized pool (the same fix `partial_aggregate` applies to the map
/// fold) spreads them across every core the actor owns. Result-identical; scheduling only.
fn in_worker_pool<T: Send>(f: impl FnOnce() -> T + Send) -> Result<T, InterpError> {
    let width = bc_arrow::usable_cores();
    Ok(crate::par::pool_for(width)?.install(f))
}

/// Hash-shuffle `batches` into `num_partitions` buckets by the given key columns.
/// Returns one (single-batch) relation per bucket — the unit a reducer consumes.
///
/// Buckets straight from the mapper's morsels, gathering each row **once**
/// (`ops::partition_morsels_by_index`). Concatenating first (`materialize`) and then
/// `partition_by_keys`-ing the result gathers every row a second time — two full copies
/// of the mapper's entire output, on every shuffle of every distributed query. The
/// buckets, their contents, and the row order within each are identical either way (a
/// row's bucket is a deterministic function of its key, and the gather visits morsels
/// and rows in order), which is what the single-node shuffle join already relies on.
pub fn partition_batches(
    batches: &[RecordBatch],
    key_indices: &[usize],
    num_partitions: usize,
) -> Result<Vec<Vec<RecordBatch>>, InterpError> {
    partition_batches_salted(batches, key_indices, num_partitions, 0)
}

/// [`partition_batches`] with an independent re-mix of the key hash.
///
/// `salt == 0` is exactly [`partition_batches`], which is the cluster-wide bucket
/// assignment and must not be perturbed. A non-zero salt is for the *local* decision of how
/// to re-split a bucket that did not fit in memory, where the unsalted hash is not merely a
/// poor choice but an inert one: bucket assignment reads the low bits at a power-of-two
/// count, so re-partitioning a 16-way bucket into 8 sub-buckets sends every row to
/// `bucket & 7` — one sub-bucket, always, at every level of the recursion.
///
/// Equal keys still co-locate, because the salt is a function of the recursion depth and
/// never of the row. That is what keeps each sub-bucket an independent instance of the same
/// reduce whose union is the same relation.
pub fn partition_batches_salted(
    batches: &[RecordBatch],
    key_indices: &[usize],
    num_partitions: usize,
    salt: u64,
) -> Result<Vec<Vec<RecordBatch>>, InterpError> {
    let parts = in_worker_pool(|| {
        ops::partition_morsels_by_index_salted(batches, key_indices, num_partitions, salt)
    })??;
    Ok(parts.into_iter().map(|b| vec![b]).collect())
}

/// Range-shuffle `batches` into `n_buckets` globally-ordered buckets by the leading
/// sort key at `key_index` and the ascending `boundaries` — the distributed-sort
/// counterpart of the hash [`partition_batches`]. Bucket order is the sort order, so a
/// reducer sorts its bucket and the driver concatenates buckets (reversed when
/// `descending`) with no merge. Returns one (single-batch) relation per bucket.
pub fn range_partition_batches(
    batches: &[RecordBatch],
    key_index: usize,
    boundaries: &[f64],
    n_buckets: usize,
    nulls_first: bool,
    descending: bool,
) -> Result<Vec<Vec<RecordBatch>>, InterpError> {
    let combined = ops::materialize(batches)?;
    let parts = in_worker_pool(|| {
        shuffle::range_partition_by_key(
            &combined,
            key_index,
            boundaries,
            n_buckets,
            nulls_first,
            descending,
        )
    })??;
    Ok(parts.into_iter().map(|b| vec![b]).collect())
}

/// Skew-aware shuffle for a single-key distributed join: like [`partition_batches`],
/// but a *hot* key's rows are salted across reducers instead of overloading one.
/// `replicate=false` (probe side) fans each hot row to one salted bucket;
/// `replicate=true` (build side) replicates each hot row to all salted buckets, so
/// every salted probe bucket can match it. Cold keys hash exactly as the unsalted
/// shuffle, so the joined relation is unchanged — only the hot key's work moves off
/// a single reducer. See [`shuffle::salted_partition_by_keys`].
pub fn salted_partition_batches(
    batches: &[RecordBatch],
    key_indices: &[usize],
    num_partitions: usize,
    hot_keys: &std::collections::HashSet<String>,
    salt_count: u32,
    replicate: bool,
) -> Result<Vec<Vec<RecordBatch>>, InterpError> {
    let combined = ops::materialize(batches)?;
    let parts = in_worker_pool(|| {
        shuffle::salted_partition_by_keys(
            &combined,
            key_indices,
            num_partitions,
            hot_keys,
            salt_count,
            replicate,
        )
    })??;
    Ok(parts.into_iter().map(|b| vec![b]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use arrow::datatypes::DataType;

    /// One input chunk as `(key column, value column)`, nulls included. The partition
    /// tests build relations chunk by chunk to prove that how a relation happens to be
    /// morselized never changes which reducer a key lands on.
    type FloatChunk = (Vec<Option<f64>>, Vec<Option<f64>>);
    type IntChunk = (Vec<Option<i64>>, Vec<Option<i64>>);

    fn batch(n_cols: usize) -> RecordBatch {
        let fields: Vec<Field> = (0..n_cols)
            .map(|i| Field::new(format!("c{i}"), DataType::Int64, true))
            .collect();
        let cols: Vec<ArrayRef> = (0..n_cols)
            .map(|_| Arc::new(Int64Array::from(vec![1i64])) as ArrayRef)
            .collect();
        RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap()
    }

    /// A version-skewed/corrupt partial (wrong column count) from another worker is
    /// rejected with a typed error before any column is indexed — never an
    /// out-of-bounds panic on the reducer.
    #[test]
    fn malformed_partial_is_typed_error_not_panic() {
        // Expect n_keys (1) + widths (2 + 1 = 3) = 4 columns; give 2.
        match batches_to_partials(1, &[2, 1], &[batch(2)]) {
            Err(InterpError::MalformedPartial {
                expected,
                n_keys,
                state,
                got,
            }) => assert_eq!((expected, n_keys, state, got), (4, 1, 3, 2)),
            _ => panic!("expected Err(MalformedPartial)"),
        }
    }

    /// A correctly-shaped batch decodes into one partial with the right arity split.
    #[test]
    fn well_formed_partial_decodes() {
        let partials = batches_to_partials(1, &[2, 1], &[batch(4)]).unwrap();
        assert_eq!(partials.len(), 1);
        assert_eq!(partials[0].group_columns.len(), 1);
        let widths: Vec<usize> = partials[0].states.iter().map(|s| s.len()).collect();
        assert_eq!(widths, vec![2, 1]);
    }

    // ----------------------------------------------------------------------------------
    // Full mergeable-invariant composition tests: the distributed map/shuffle/reduce
    // pipeline (`partial_aggregate` -> `partition_batches` -> `combine_finalize`) over N
    // partitions MUST equal the single-node aggregate, for every aggregate and every edge
    // key (-0.0/0.0, NaN, NULL, dup keys spanning morsels/partitions). Composed directly,
    // no Ray.
    // ----------------------------------------------------------------------------------
    use arrow::array::{BooleanArray, Float64Array};
    use bc_ir::{AggFunc, AggregateItem, ProjectionItem};
    use std::collections::BTreeMap;

    fn col(name: &str) -> bc_expr::Expr {
        bc_expr::Expr::Col { name: name.into() }
    }

    fn gk(name: &str) -> Vec<ProjectionItem> {
        vec![ProjectionItem {
            expr: col(name),
            alias: name.into(),
        }]
    }

    /// One aggregate item over `input` (and optional `input2` ordering key / second input).
    fn agg(
        func: AggFunc,
        input: Option<&str>,
        input2: Option<&str>,
        param: Option<f64>,
        alias: &str,
    ) -> AggregateItem {
        AggregateItem {
            func,
            input: input.map(col),
            input2: input2.map(col),
            param,
            alias: alias.into(),
        }
    }

    /// A test relation split into morsels: an Int64 group key `k` (dups spanning morsels,
    /// nulls), a Float64 value `v` (with a null), an Int64 order key `o`, and a Boolean `b`.
    fn agg_morsels() -> Vec<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("v", DataType::Float64, true),
            Field::new("o", DataType::Int64, true),
            Field::new("b", DataType::Boolean, true),
        ]));
        // (k, v, o, b) columns for one morsel.
        type Chunk = (
            Vec<Option<i64>>,
            Vec<Option<f64>>,
            Vec<Option<i64>>,
            Vec<Option<bool>>,
        );
        let chunks: Vec<Chunk> = vec![
            (
                vec![Some(1), Some(2), None, Some(3), Some(2)],
                vec![Some(10.0), Some(20.0), Some(30.0), None, Some(50.0)],
                vec![Some(5), Some(2), Some(9), Some(1), Some(7)],
                vec![Some(true), Some(false), Some(true), None, Some(true)],
            ),
            (
                vec![Some(3), Some(1), None, Some(2)],
                vec![Some(60.0), Some(70.0), Some(80.0), Some(90.0)],
                vec![Some(3), Some(8), Some(4), Some(6)],
                vec![Some(false), Some(true), Some(false), Some(true)],
            ),
            (
                vec![Some(1), Some(3), Some(3), Some(2), Some(4)],
                vec![Some(11.0), Some(21.0), Some(31.0), Some(41.0), Some(51.0)],
                vec![Some(0), Some(2), Some(2), Some(9), Some(3)],
                vec![Some(true), Some(true), Some(false), Some(false), Some(true)],
            ),
        ];
        chunks
            .into_iter()
            .map(|(k, v, o, b)| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int64Array::from(k)) as ArrayRef,
                        Arc::new(Float64Array::from(v)) as ArrayRef,
                        Arc::new(Int64Array::from(o)) as ArrayRef,
                        Arc::new(BooleanArray::from(b)) as ArrayRef,
                    ],
                )
                .unwrap()
            })
            .collect()
    }

    /// The single-node oracle: partial over the whole relation, finalize.
    fn single_node(
        group_keys: &[ProjectionItem],
        aggregates: &[AggregateItem],
        input: &[RecordBatch],
    ) -> RecordBatch {
        let whole = ops::materialize(input).unwrap();
        let partial = ops::eval_partial(&whole, group_keys, aggregates).unwrap();
        let funcs = ops::agg_funcs(aggregates);
        let cols = agg::finalize(&funcs, &partial).unwrap();
        ops::build_agg_batch(group_keys, aggregates, &partial.group_columns, &cols).unwrap()
    }

    /// The distributed pipeline over `map_partitions` map tasks and `n` reducers.
    fn distributed(
        group_keys: &[ProjectionItem],
        aggregates: &[AggregateItem],
        map_partitions: &[Vec<RecordBatch>],
        n: usize,
    ) -> Vec<RecordBatch> {
        // Map: each partition -> partial state batch.
        let map_partials: Vec<RecordBatch> = map_partitions
            .iter()
            .map(|p| partial_aggregate(group_keys, aggregates, p).unwrap())
            .collect();
        // Shuffle: route each map partial's rows to reducers by the group-key columns.
        let key_idx: Vec<usize> = (0..group_keys.len()).collect();
        let mut reduce_inputs: Vec<Vec<RecordBatch>> = vec![Vec::new(); n];
        for mp in &map_partials {
            let buckets = partition_batches(std::slice::from_ref(mp), &key_idx, n).unwrap();
            for (i, bucket) in buckets.into_iter().enumerate() {
                // Keep EVERY shard, including the empty ones: a real reducer receives one
                // (possibly zero-row) partial per map task, so `combine_finalize` must
                // tolerate a mix of empty and non-empty partial-state batches.
                reduce_inputs[i].extend(bucket);
            }
        }
        // Reduce: combine+finalize each reducer (its input is never an empty *list* — one
        // shard per map task — but may be all zero-row shards, which finalize to zero rows).
        reduce_inputs
            .into_iter()
            .filter(|inp| !inp.is_empty())
            .map(|inp| combine_finalize(group_keys, aggregates, &inp).unwrap())
            .collect()
    }

    /// Map from group-key string to the remaining (aggregate) column strings, over any
    /// number of output batches. `arrow`'s formatter stringifies every type (incl. list/map).
    fn result_map(batches: &[RecordBatch]) -> BTreeMap<String, Vec<String>> {
        use arrow::util::display::{ArrayFormatter, FormatOptions};
        let opts = FormatOptions::default();
        let mut out = BTreeMap::new();
        for b in batches {
            let fmts: Vec<ArrayFormatter> = b
                .columns()
                .iter()
                .map(|c| ArrayFormatter::try_new(c, &opts).unwrap())
                .collect();
            for r in 0..b.num_rows() {
                let key = fmts[0].value(r).to_string();
                let vals: Vec<String> = fmts[1..].iter().map(|f| f.value(r).to_string()).collect();
                assert!(
                    out.insert(key.clone(), vals).is_none(),
                    "duplicate group key {key} in output — a group was split across reducers"
                );
            }
        }
        out
    }

    /// The multiset of a formatted list/map's elements, sorted — so an order difference
    /// between single-node (relation order) and distributed (per-partition order) is not a
    /// false mismatch, but a *dropped or duplicated* element still is.
    fn as_sorted_tokens(s: &str) -> Option<Vec<String>> {
        let inner = s.strip_prefix('[').and_then(|x| x.strip_suffix(']'));
        let inner = inner.or_else(|| s.strip_prefix('{').and_then(|x| x.strip_suffix('}')))?;
        let mut toks: Vec<String> = if inner.trim().is_empty() {
            Vec::new()
        } else {
            inner.split(',').map(|t| t.trim().to_string()).collect()
        };
        toks.sort();
        Some(toks)
    }

    /// Compare two stringified values: element-multiset for list/map collections, numeric
    /// (tolerant) when both parse as f64, else exactly. Tolerates float summation-order and
    /// collection-order differences between combine orders — but not lost/duplicated data.
    fn values_match(a: &str, b: &str) -> bool {
        if let (Some(ta), Some(tb)) = (as_sorted_tokens(a), as_sorted_tokens(b)) {
            return ta == tb;
        }
        match (a.parse::<f64>(), b.parse::<f64>()) {
            (Ok(x), Ok(y)) => {
                if x.is_nan() && y.is_nan() {
                    return true;
                }
                (x - y).abs() <= 1e-9 * x.abs().max(y.abs()).max(1.0)
            }
            _ => a == b,
        }
    }

    fn assert_dist_matches_single_node(
        group_keys: &[ProjectionItem],
        aggregates: &[AggregateItem],
        label: &str,
    ) {
        let morsels = agg_morsels();
        let want = result_map(&[single_node(group_keys, aggregates, &morsels)]);
        // A few map-partition splits x reducer counts, including a count that does not
        // divide the group count and a single reducer.
        let map_splits: Vec<Vec<Vec<RecordBatch>>> = vec![
            vec![morsels.clone()],                              // 1 map task
            vec![morsels[..1].to_vec(), morsels[1..].to_vec()], // 2 map tasks
            morsels.iter().map(|m| vec![m.clone()]).collect(),  // 1 map task per morsel
        ];
        for (mi, maps) in map_splits.iter().enumerate() {
            for n in [1usize, 2, 3, 7, 64] {
                let got = result_map(&distributed(group_keys, aggregates, maps, n));
                assert_eq!(
                    got.keys().collect::<Vec<_>>(),
                    want.keys().collect::<Vec<_>>(),
                    "{label}: group-key set differs (map_split={mi}, reducers={n})"
                );
                for (k, wv) in &want {
                    let gv = &got[k];
                    assert_eq!(wv.len(), gv.len(), "{label}: arity for key {k}");
                    for (wc, gc) in wv.iter().zip(gv) {
                        assert!(
                            values_match(wc, gc),
                            "{label}: key {k} value {gc:?} != single-node {wc:?} \
                             (map_split={mi}, reducers={n})"
                        );
                    }
                }
            }
        }
    }

    /// Every scalar aggregate: the distributed pipeline equals the single-node oracle across
    /// {1,2,3,7,64} reducers and several map-partition splits. This exercises the wire-format
    /// state arity (`agg_widths`/`state_arity`), the shuffle disjointness, and combine.
    #[test]
    fn every_aggregate_survives_the_distributed_pipeline() {
        let g = gk("k");
        let cases: Vec<(&str, Vec<AggregateItem>)> = vec![
            (
                "count_star",
                vec![agg(AggFunc::CountStar, None, None, None, "a")],
            ),
            (
                "count",
                vec![agg(AggFunc::Count, Some("v"), None, None, "a")],
            ),
            (
                "count_distinct",
                vec![agg(AggFunc::CountDistinct, Some("k"), None, None, "a")],
            ),
            ("sum", vec![agg(AggFunc::Sum, Some("v"), None, None, "a")]),
            ("min", vec![agg(AggFunc::Min, Some("v"), None, None, "a")]),
            ("max", vec![agg(AggFunc::Max, Some("v"), None, None, "a")]),
            ("mean", vec![agg(AggFunc::Mean, Some("v"), None, None, "a")]),
            ("var", vec![agg(AggFunc::Var, Some("v"), None, None, "a")]),
            (
                "stddev",
                vec![agg(AggFunc::Stddev, Some("v"), None, None, "a")],
            ),
            (
                "median",
                vec![agg(AggFunc::Median, Some("v"), None, None, "a")],
            ),
            (
                "quantile",
                vec![agg(AggFunc::Quantile, Some("v"), None, Some(0.25), "a")],
            ),
            (
                "bool_and",
                vec![agg(AggFunc::BoolAnd, Some("b"), None, None, "a")],
            ),
            (
                "bool_or",
                vec![agg(AggFunc::BoolOr, Some("b"), None, None, "a")],
            ),
            ("mode", vec![agg(AggFunc::Mode, Some("o"), None, None, "a")]),
            (
                "arg_min",
                vec![agg(AggFunc::ArgMin, Some("v"), Some("o"), None, "a")],
            ),
            (
                "arg_max",
                vec![agg(AggFunc::ArgMax, Some("v"), Some("o"), None, "a")],
            ),
            (
                "product",
                vec![agg(AggFunc::Product, Some("v"), None, None, "a")],
            ),
            (
                "bit_and",
                vec![agg(AggFunc::BitAnd, Some("o"), None, None, "a")],
            ),
            (
                "bit_or",
                vec![agg(AggFunc::BitOr, Some("o"), None, None, "a")],
            ),
            (
                "bit_xor",
                vec![agg(AggFunc::BitXor, Some("o"), None, None, "a")],
            ),
            (
                "covar_pop",
                vec![agg(AggFunc::CovarPop, Some("v"), Some("o"), None, "a")],
            ),
            (
                "covar_samp",
                vec![agg(AggFunc::CovarSamp, Some("v"), Some("o"), None, "a")],
            ),
            (
                "corr",
                vec![agg(AggFunc::Corr, Some("v"), Some("o"), None, "a")],
            ),
            (
                "skewness",
                vec![agg(AggFunc::Skewness, Some("v"), None, None, "a")],
            ),
            (
                "kurtosis",
                vec![agg(AggFunc::Kurtosis, Some("v"), None, None, "a")],
            ),
            // Order-sensitive collection outputs: distributed may reorder, but must never
            // drop or duplicate an element (compared as a sorted multiset).
            (
                "list_agg",
                vec![agg(AggFunc::ListAgg, Some("o"), None, None, "a")],
            ),
            (
                "histogram",
                vec![agg(AggFunc::Histogram, Some("o"), None, None, "a")],
            ),
            (
                "multi",
                vec![
                    agg(AggFunc::Sum, Some("v"), None, None, "s"),
                    agg(AggFunc::Mean, Some("v"), None, None, "m"),
                    agg(AggFunc::Var, Some("v"), None, None, "vr"),
                    agg(AggFunc::ArgMax, Some("v"), Some("o"), None, "am"),
                    agg(AggFunc::Corr, Some("v"), Some("o"), None, "cr"),
                    agg(AggFunc::CountStar, None, None, None, "c"),
                ],
            ),
        ];
        for (label, aggs) in &cases {
            assert_dist_matches_single_node(&g, aggs, label);
        }
    }

    /// Float group keys that group-equal but bit-differ (`-0.0`/`0.0`), all-NaN, and NULL
    /// keys must land in one group each — the shuffle must route them exactly as single-node
    /// grouping does. A split here returns more groups distributed than single-node.
    #[test]
    fn edge_float_group_keys_route_identically() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Float64, true),
            Field::new("v", DataType::Float64, true),
        ]));
        let nan = f64::NAN;
        let chunks: Vec<FloatChunk> = vec![
            (
                vec![Some(0.0), Some(-0.0), Some(nan), None, Some(1.0)],
                vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)],
            ),
            (
                vec![Some(-0.0), Some(0.0), Some(nan), None, Some(1.0)],
                vec![Some(6.0), Some(7.0), Some(8.0), Some(9.0), Some(10.0)],
            ),
        ];
        let morsels: Vec<RecordBatch> = chunks
            .into_iter()
            .map(|(k, v)| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Float64Array::from(k)) as ArrayRef,
                        Arc::new(Float64Array::from(v)) as ArrayRef,
                    ],
                )
                .unwrap()
            })
            .collect();
        let g = gk("k");
        let aggs = vec![
            agg(AggFunc::Sum, Some("v"), None, None, "s"),
            agg(AggFunc::CountStar, None, None, None, "c"),
        ];
        let want = result_map(&[single_node(&g, &aggs, &morsels)]);
        // 0.0/-0.0 collapse to one group, all NaN to one, all NULL to one, plus 1.0.
        assert_eq!(
            want.len(),
            4,
            "single-node group count over edge float keys"
        );
        for n in [1usize, 2, 3, 7, 64] {
            let maps = vec![morsels[..1].to_vec(), morsels[1..].to_vec()];
            let got = result_map(&distributed(&g, &aggs, &maps, n));
            assert_eq!(
                got.len(),
                want.len(),
                "reducers={n}: distributed split an edge-key group the single-node oracle merged"
            );
            for (k, wv) in &want {
                for (wc, gc) in wv.iter().zip(&got[k]) {
                    assert!(
                        values_match(wc, gc),
                        "reducers={n} key {k}: {gc:?} != {wc:?}"
                    );
                }
            }
        }
    }

    /// A keyed relation split across several morsels.
    fn keyed_morsels() -> Vec<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("v", DataType::Int64, true),
        ]));
        // Repeating keys, nulls, and uneven morsel sizes — the shapes a shuffle must
        // route identically no matter how the relation happens to be chunked.
        let chunks: Vec<IntChunk> = vec![
            (
                vec![Some(1), Some(2), None, Some(3), Some(2)],
                vec![Some(10), Some(20), Some(30), Some(40), Some(50)],
            ),
            (vec![Some(3), Some(1)], vec![Some(60), Some(70)]),
            (
                vec![Some(7), None, Some(2), Some(9)],
                vec![Some(80), Some(90), Some(100), Some(110)],
            ),
        ];
        chunks
            .into_iter()
            .map(|(k, v)| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int64Array::from(k)) as ArrayRef,
                        Arc::new(Int64Array::from(v)) as ArrayRef,
                    ],
                )
                .unwrap()
            })
            .collect()
    }

    /// `partition_batches` buckets straight from the morsels (gathering each row once)
    /// instead of concatenating the relation and gathering it again. The two must agree
    /// **exactly** — same buckets, same contents, same row order within each bucket —
    /// because a reducer's result depends on it. This pins that equivalence against the
    /// materialize-then-`partition_by_keys` form it replaced.
    #[test]
    fn partition_batches_matches_materialize_then_partition() {
        let morsels = keyed_morsels();
        for parts in [1usize, 2, 3, 8] {
            let got = partition_batches(&morsels, &[0], parts).unwrap();

            let combined = ops::materialize(&morsels).unwrap();
            let want = shuffle::partition_by_keys(&combined, &[0], parts).unwrap();

            assert_eq!(got.len(), want.len(), "bucket count (parts={parts})");
            for (bucket, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.len(), 1, "one batch per bucket");
                assert_eq!(
                    &g[0], w,
                    "bucket {bucket} of {parts} differs from the concatenated shuffle"
                );
            }
        }
    }

    // ----------------------------------------------------------------------------------
    // The distributed *spilling* reduce (`combine_finalize_spilling`, reading partials from
    // on-disk shuffle files under a byte budget) MUST equal the in-memory reduce and the
    // single-node oracle — the mergeable invariant holding out-of-core. A tiny budget forces
    // real grace partitioning (and its recursion) rather than an in-memory shortcut.
    // ----------------------------------------------------------------------------------
    use std::sync::atomic::{AtomicU64, Ordering};

    static SPILL_TEST_SEQ: AtomicU64 = AtomicU64::new(0);

    /// A private scratch dir for one spill-reduce test, removed on drop.
    struct ScratchDir(PathBuf);
    impl ScratchDir {
        fn new() -> Self {
            let seq = SPILL_TEST_SEQ.fetch_add(1, Ordering::Relaxed);
            let dir =
                std::env::temp_dir().join(format!("bc-dist-spill-{}-{seq}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            ScratchDir(dir)
        }
    }
    impl Drop for ScratchDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Write partial-state batches to an Arrow-IPC stream file (the format the shuffle uses).
    fn write_ipc_stream(path: &Path, batches: &[RecordBatch]) {
        use arrow::ipc::writer::StreamWriter;
        let file = std::fs::File::create(path).unwrap();
        let mut w = StreamWriter::try_new(file, &batches[0].schema()).unwrap();
        for b in batches {
            w.write(b).unwrap();
        }
        w.finish().unwrap();
    }

    /// The distributed pipeline over `map_partitions` map tasks and `n` reducers, but each
    /// reducer's partials are written to on-disk IPC files and reduced by the *spilling*
    /// `combine_finalize_spilling` under `budget`. Mirrors [`distributed`].
    fn distributed_spilling(
        group_keys: &[ProjectionItem],
        aggregates: &[AggregateItem],
        map_partitions: &[Vec<RecordBatch>],
        n: usize,
        budget: usize,
        scratch: &Path,
    ) -> Vec<RecordBatch> {
        let map_partials: Vec<RecordBatch> = map_partitions
            .iter()
            .map(|p| partial_aggregate(group_keys, aggregates, p).unwrap())
            .collect();
        let key_idx: Vec<usize> = (0..group_keys.len()).collect();
        // reducer -> list of on-disk file paths (one per mapper).
        let mut reduce_paths: Vec<Vec<PathBuf>> = vec![Vec::new(); n];
        for (mi, mp) in map_partials.iter().enumerate() {
            let buckets = partition_batches(std::slice::from_ref(mp), &key_idx, n).unwrap();
            for (ri, bucket) in buckets.into_iter().enumerate() {
                let path = scratch.join(format!("m{mi}_r{ri}.arrow"));
                write_ipc_stream(&path, &bucket);
                reduce_paths[ri].push(path);
            }
        }
        reduce_paths
            .into_iter()
            .filter_map(|paths| {
                let out = combine_finalize_spilling(
                    group_keys, aggregates, &paths, budget, scratch, None,
                )
                .unwrap();
                (out.num_rows() > 0).then_some(out)
            })
            .collect()
    }

    #[test]
    fn spilling_reduce_matches_single_node() {
        let g = gk("k");
        // A representative mix: constant-state, value-list (median/quantile/mode), a
        // collection output, and a multi-aggregate row — each must merge identically
        // out-of-core.
        let cases: Vec<(&str, Vec<AggregateItem>)> = vec![
            ("sum", vec![agg(AggFunc::Sum, Some("v"), None, None, "a")]),
            (
                "count_star",
                vec![agg(AggFunc::CountStar, None, None, None, "a")],
            ),
            (
                "count_distinct",
                vec![agg(AggFunc::CountDistinct, Some("k"), None, None, "a")],
            ),
            ("mean", vec![agg(AggFunc::Mean, Some("v"), None, None, "a")]),
            ("var", vec![agg(AggFunc::Var, Some("v"), None, None, "a")]),
            (
                "median",
                vec![agg(AggFunc::Median, Some("v"), None, None, "a")],
            ),
            (
                "quantile",
                vec![agg(AggFunc::Quantile, Some("v"), None, Some(0.25), "a")],
            ),
            ("mode", vec![agg(AggFunc::Mode, Some("o"), None, None, "a")]),
            (
                "list_agg",
                vec![agg(AggFunc::ListAgg, Some("o"), None, None, "a")],
            ),
            (
                "multi",
                vec![
                    agg(AggFunc::Sum, Some("v"), None, None, "s"),
                    agg(AggFunc::Mean, Some("v"), None, None, "m"),
                    agg(AggFunc::Median, Some("v"), None, None, "md"),
                    agg(AggFunc::CountStar, None, None, None, "c"),
                ],
            ),
        ];
        let morsels = agg_morsels();
        let map_splits: Vec<Vec<Vec<RecordBatch>>> = vec![
            vec![morsels.clone()],
            vec![morsels[..1].to_vec(), morsels[1..].to_vec()],
            morsels.iter().map(|m| vec![m.clone()]).collect(),
        ];
        for (label, aggs) in &cases {
            let want = result_map(&[single_node(&g, aggs, &morsels)]);
            for maps in &map_splits {
                for n in [1usize, 2, 3, 7] {
                    // Budget of 1 byte forces the deepest grace partitioning + recursion; a
                    // larger one exercises the single-partition merge. Both must agree.
                    for budget in [1usize, 64, 1 << 20] {
                        let scratch = ScratchDir::new();
                        let got = result_map(&distributed_spilling(
                            &g, aggs, maps, n, budget, &scratch.0,
                        ));
                        assert_eq!(
                            got.keys().collect::<Vec<_>>(),
                            want.keys().collect::<Vec<_>>(),
                            "{label}: group-key set differs (reducers={n}, budget={budget})"
                        );
                        for (k, wv) in &want {
                            let gv = &got[k];
                            assert_eq!(wv.len(), gv.len(), "{label}: arity for key {k}");
                            for (wc, gc) in wv.iter().zip(gv) {
                                assert!(
                                    values_match(wc, gc),
                                    "{label}: key {k} value {gc:?} != single-node {wc:?} \
                                     (reducers={n}, budget={budget})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// An empty reducer (all-empty shuffle files) spills to a zero-row result, not an error —
    /// the disk reducer must tolerate the all-empty-shard case the same as the in-memory fold.
    #[test]
    fn spilling_reduce_all_empty_is_zero_rows() {
        let g = gk("k");
        let aggs = vec![agg(AggFunc::Sum, Some("v"), None, None, "a")];
        let scratch = ScratchDir::new();
        // One empty partial (zero-row) written to disk.
        let empty = agg_morsels()[0].slice(0, 0);
        let partial = partial_aggregate(&g, &aggs, &[empty]).unwrap();
        let path = scratch.0.join("empty.arrow");
        write_ipc_stream(&path, &[partial]);
        let out = combine_finalize_spilling(&g, &aggs, &[path], 1, &scratch.0, None).unwrap();
        assert_eq!(out.num_rows(), 0, "an all-empty reducer yields zero rows");
    }
}
