//! Spilling (grace) hash aggregation — bounded-memory `combine` + `finalize`.
//!
//! The in-memory aggregate ([`super::combine`] + [`super::finalize`]) holds every
//! group's state at once: peak memory is the full group cardinality. When that
//! exceeds the operator's memory envelope, this module computes the *same result*
//! with memory bounded to a single hash partition.
//!
//! The mechanism is the mergeable algebra applied locally. Per-morsel partials
//! (the output of [`super::partial`]) are routed to one of `P` partitions by a
//! hash of their group key and written to a [`SpillStore`]. Because a given group
//! key always hashes to the same partition, every partial row for a group lands
//! together — so `combine`+`finalize` run **one partition at a time** equals the
//! global aggregate (`combine` is associative+commutative; partitions are
//! disjoint by key). This is exactly the distributive-equivalence property the
//! distributed path relies on, reused to bound single-node memory.
//!
//! `SpillStore` has two implementations: [`MemSpillStore`] (the partitions stay in
//! memory — used to prove the grace algebra matches the oracle) and
//! [`DiskSpillStore`] (partitions stream to Arrow IPC files, the path that
//! actually bounds resident memory under pressure).

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, RecordBatch, UInt32Array};
use arrow::compute::take;
use arrow::datatypes::{Field, Schema};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use arrow::row::{RowConverter, SortField};

use super::{combine, finalize, AggFunc, GroupAggResult, Partial};
use crate::error::RuntimeError;

/// A partitioned, append-only store of partial-state batches.
///
/// The aggregator appends routed partials during the spill phase and reads each
/// partition back (exactly once) during the merge phase. Implementations decide
/// whether partitions live in memory or on disk; the algorithm is identical.
pub trait SpillStore {
    /// Number of hash partitions this store was created with (`P`).
    fn num_partitions(&self) -> usize;
    /// Append one partial-state batch to `partition`.
    fn append(&mut self, partition: usize, batch: &RecordBatch) -> Result<(), RuntimeError>;
    /// Drain every batch previously appended to `partition`. Called once per
    /// partition; a store may free the partition's backing storage afterward.
    fn read(&mut self, partition: usize) -> Result<Vec<RecordBatch>, RuntimeError>;
}

/// In-memory partitions. Does not reduce resident memory — it exists to test the
/// grace algebra against the non-spilling oracle without touching the filesystem.
pub struct MemSpillStore {
    parts: Vec<Vec<RecordBatch>>,
}

impl MemSpillStore {
    pub fn new(partitions: usize) -> Self {
        let n = partitions.max(1);
        Self {
            parts: (0..n).map(|_| Vec::new()).collect(),
        }
    }
}

impl SpillStore for MemSpillStore {
    fn num_partitions(&self) -> usize {
        self.parts.len()
    }
    fn append(&mut self, partition: usize, batch: &RecordBatch) -> Result<(), RuntimeError> {
        self.parts[partition].push(batch.clone());
        Ok(())
    }
    fn read(&mut self, partition: usize) -> Result<Vec<RecordBatch>, RuntimeError> {
        Ok(std::mem::take(&mut self.parts[partition]))
    }
}

/// Monotonic per-process counter that makes each spill store's scratch directory
/// unique. Without it, every store names its files `part-{i}.arrow` under the same
/// shared spill root, so concurrent stores — sibling spilling operators in one plan,
/// or several distributed worker processes sharing one spill dir — would clobber
/// each other's partitions (and one store's drop would `remove_dir_all` the shared
/// root out from under the others). A process id + this counter isolates them.
static SPILL_SEQ: AtomicU64 = AtomicU64::new(0);

/// Disk-backed partitions: each partition streams to its own Arrow IPC file, so
/// only the partition currently being merged is resident. Each store owns a private
/// subdirectory under the given root, removed on drop (best-effort).
pub struct DiskSpillStore {
    dir: PathBuf,
    paths: Vec<PathBuf>,
    writers: Vec<Option<StreamWriter<File>>>,
}

impl DiskSpillStore {
    /// Create `partitions` empty spill files under a private subdirectory of `root`.
    ///
    /// The store carves out its own `bc-spill-{pid}-{seq}` directory under `root`
    /// (created if absent) so its `part-*.arrow` files never collide with — and its
    /// drop never deletes — another concurrent store's files. This is what lets the
    /// distributed reducers spill safely when many worker processes share one spill
    /// root, and lets a single plan run two spilling breakers at once.
    pub fn new(root: PathBuf, partitions: usize) -> Result<Self, RuntimeError> {
        let seq = SPILL_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = root.join(format!("bc-spill-{}-{seq}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let n = partitions.max(1);
        let paths = (0..n)
            .map(|i| dir.join(format!("part-{i}.arrow")))
            .collect();
        Ok(Self {
            dir,
            paths,
            writers: (0..n).map(|_| None).collect(),
        })
    }

    /// Finish `partition`'s writer (if still open) and return a *streaming* reader
    /// over its batches — yielding one `RecordBatch` at a time, the bounded-memory
    /// counterpart to [`SpillStore::read`] (which returns the whole partition). A
    /// k-way merge uses this so only one batch per run is resident at a time.
    /// `None` when the partition was never written.
    pub fn open_reader(
        &mut self,
        partition: usize,
    ) -> Result<Option<StreamReader<BufReader<File>>>, RuntimeError> {
        if let Some(mut w) = self.writers[partition].take() {
            w.finish()?;
        }
        if !self.paths[partition].exists() {
            return Ok(None);
        }
        let file = File::open(&self.paths[partition])?;
        Ok(Some(StreamReader::try_new(BufReader::new(file), None)?))
    }
}

impl SpillStore for DiskSpillStore {
    fn num_partitions(&self) -> usize {
        self.paths.len()
    }

    fn append(&mut self, partition: usize, batch: &RecordBatch) -> Result<(), RuntimeError> {
        if self.writers[partition].is_none() {
            let file = File::create(&self.paths[partition])?;
            self.writers[partition] = Some(StreamWriter::try_new(file, &batch.schema())?);
        }
        self.writers[partition]
            .as_mut()
            .expect("writer just created")
            .write(batch)?;
        Ok(())
    }

    fn read(&mut self, partition: usize) -> Result<Vec<RecordBatch>, RuntimeError> {
        // Finish (flush + close) the writer so the IPC stream is complete before
        // we read it back. A partition with no appends yields nothing.
        match self.writers[partition].take() {
            Some(mut w) => w.finish()?,
            None => return Ok(Vec::new()),
        }
        let file = File::open(&self.paths[partition])?;
        let reader = StreamReader::try_new(BufReader::new(file), None)?;
        reader
            .collect::<Result<Vec<_>, _>>()
            .map_err(RuntimeError::from)
    }
}

impl Drop for DiskSpillStore {
    fn drop(&mut self) {
        // Best-effort cleanup of the temporary spill directory.
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Spilling equivalent of `finalize(combine(chunk_partials))`.
///
/// Routes each chunk's partial state to a hash partition in `store`, then merges
/// and finalizes one partition at a time. The result equals
/// [`super::group_aggregate`] over the concatenated input (group order differs —
/// these are unordered relations). `funcs` must match the aggregates used to
/// build the partials; for an all-columns distinct grouping pass `&[]`.
pub fn combine_finalize_spilling(
    chunk_partials: impl IntoIterator<Item = Partial>,
    funcs: &[AggFunc],
    store: &mut dyn SpillStore,
) -> Result<GroupAggResult, RuntimeError> {
    let partitions = store.num_partitions().max(1);

    // --- spill phase: route every partial's groups to a hash partition ---------
    let mut n_keys = 0usize;
    let mut any = false;
    for partial in chunk_partials {
        any = true;
        n_keys = partial.group_columns.len();
        let packed = pack_partial(&partial)?;
        for (pi, sub) in route(&packed, n_keys, partitions)? {
            store.append(pi, &sub)?;
        }
    }
    if !any {
        return Ok(GroupAggResult {
            group_columns: Vec::new(),
            agg_columns: Vec::new(),
        });
    }

    // --- merge phase: combine + finalize one partition at a time ---------------
    let mut group_parts: Vec<Vec<ArrayRef>> = Vec::new();
    let mut agg_parts: Vec<Vec<ArrayRef>> = Vec::new();
    for pi in 0..partitions {
        let batches = store.read(pi)?;
        if batches.is_empty() {
            continue;
        }
        let partials: Vec<Partial> = batches
            .iter()
            .map(|b| unpack_partial(b, n_keys, funcs))
            .collect();
        let merged = combine(&partials, funcs)?;
        let aggs = finalize(funcs, &merged)?;
        group_parts.push(merged.group_columns);
        agg_parts.push(aggs);
    }

    // Concatenate the per-partition output chunks column by column.
    let group_columns = (0..n_keys)
        .map(|c| concat_cols(group_parts.iter().map(|g| &g[c])))
        .collect::<Result<_, _>>()?;
    let agg_columns = (0..funcs.len())
        .map(|c| concat_cols(agg_parts.iter().map(|a| &a[c])))
        .collect::<Result<_, _>>()?;
    Ok(GroupAggResult {
        group_columns,
        agg_columns,
    })
}

/// Flatten a [`Partial`] into one batch: group columns first (`g0..`), then each
/// aggregate's state columns (`s{agg}_{col}`). The inverse is [`unpack_partial`].
fn pack_partial(p: &Partial) -> Result<RecordBatch, RuntimeError> {
    let mut fields = Vec::new();
    let mut cols = Vec::new();
    for (i, c) in p.group_columns.iter().enumerate() {
        fields.push(Field::new(format!("g{i}"), c.data_type().clone(), true));
        cols.push(c.clone());
    }
    for (a, state) in p.states.iter().enumerate() {
        for (ci, c) in state.iter().enumerate() {
            fields.push(Field::new(
                format!("s{a}_{ci}"),
                c.data_type().clone(),
                true,
            ));
            cols.push(c.clone());
        }
    }
    Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), cols)?)
}

/// Rebuild a [`Partial`] from a [`pack_partial`] batch using the key arity and the
/// per-aggregate state arity (which `funcs` determines).
fn unpack_partial(b: &RecordBatch, n_keys: usize, funcs: &[AggFunc]) -> Partial {
    let cols = b.columns();
    let group_columns = cols[..n_keys].to_vec();
    let mut states = Vec::with_capacity(funcs.len());
    let mut idx = n_keys;
    for &f in funcs {
        let arity = f.state_arity();
        states.push(cols[idx..idx + arity].to_vec());
        idx += arity;
    }
    Partial {
        group_columns,
        states,
    }
}

/// Partition a packed partial's rows by a stable hash of its group-key columns.
/// A global aggregate (no keys) or a single partition routes everything to 0.
fn route(
    packed: &RecordBatch,
    n_keys: usize,
    partitions: usize,
) -> Result<Vec<(usize, RecordBatch)>, RuntimeError> {
    if n_keys == 0 || partitions <= 1 {
        return Ok(vec![(0, packed.clone())]);
    }
    let group_cols = &packed.columns()[..n_keys];
    let fields: Vec<SortField> = group_cols
        .iter()
        .map(|a| SortField::new(a.data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields)?;
    let rows = converter.convert_columns(group_cols)?;

    // Fixed seeds so the same key routes identically across every chunk.
    let state = ahash::RandomState::with_seeds(0x9E37, 0x79B9, 0x7F4A, 0x7C15);
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); partitions];
    for i in 0..packed.num_rows() {
        let h = state.hash_one(rows.row(i));
        buckets[(h % partitions as u64) as usize].push(i as u32);
    }

    let mut out = Vec::new();
    for (pi, idxs) in buckets.into_iter().enumerate() {
        if idxs.is_empty() {
            continue;
        }
        let indices = UInt32Array::from(idxs);
        let cols = packed
            .columns()
            .iter()
            .map(|c| take(c.as_ref(), &indices, None).map_err(RuntimeError::from))
            .collect::<Result<Vec<_>, _>>()?;
        out.push((pi, RecordBatch::try_new(packed.schema(), cols)?));
    }
    Ok(out)
}

fn concat_cols<'a>(it: impl Iterator<Item = &'a ArrayRef>) -> Result<ArrayRef, RuntimeError> {
    let cols: Vec<&dyn Array> = it.map(|a| a.as_ref()).collect();
    Ok(arrow::compute::concat(&cols)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agg::{group_aggregate, partial, AggCall};
    use arrow::array::{Float64Array, Int64Array, StringArray};
    use std::collections::BTreeMap;

    fn strs(v: &[&str]) -> ArrayRef {
        Arc::new(StringArray::from(v.to_vec()))
    }
    fn i64s(v: &[i64]) -> ArrayRef {
        Arc::new(Int64Array::from(v.to_vec()))
    }

    const FUNCS: [AggFunc; 6] = [
        AggFunc::Sum,
        AggFunc::CountStar,
        AggFunc::Mean,
        AggFunc::Min,
        AggFunc::Max,
        AggFunc::Median,
    ];

    fn calls(v: &ArrayRef) -> Vec<AggCall> {
        FUNCS
            .iter()
            .map(|&func| {
                AggCall::new(
                    func,
                    match func {
                        AggFunc::CountStar => None,
                        _ => Some(v.clone()),
                    },
                )
            })
            .collect()
    }

    /// Render an aggregation result to a key -> [agg cells] map, order-independent.
    fn to_map(keys: &ArrayRef, aggs: &[ArrayRef]) -> BTreeMap<String, Vec<String>> {
        let keys = keys.as_any().downcast_ref::<StringArray>().unwrap();
        let mut m = BTreeMap::new();
        for i in 0..keys.len() {
            let row: Vec<String> = aggs.iter().map(|a| cell(a, i)).collect();
            m.insert(keys.value(i).to_string(), row);
        }
        m
    }

    fn cell(a: &ArrayRef, i: usize) -> String {
        if let Some(x) = a.as_any().downcast_ref::<Int64Array>() {
            return if x.is_null(i) {
                "∅".into()
            } else {
                x.value(i).to_string()
            };
        }
        if let Some(x) = a.as_any().downcast_ref::<Float64Array>() {
            return if x.is_null(i) {
                "∅".into()
            } else {
                format!("{:.4}", x.value(i))
            };
        }
        "?".into()
    }

    /// Split `(keys, vals)` into `chunks` partials and run the spilling path.
    fn spilled(
        keys: &ArrayRef,
        vals: &ArrayRef,
        chunks: usize,
        store: &mut dyn SpillStore,
    ) -> GroupAggResult {
        let n = keys.len();
        let per = n.div_ceil(chunks);
        let mut partials = Vec::new();
        let mut off = 0;
        while off < n {
            let len = per.min(n - off);
            let k = keys.slice(off, len);
            let v = vals.slice(off, len);
            partials.push(partial(std::slice::from_ref(&k), &calls(&v), len).unwrap());
            off += len;
        }
        combine_finalize_spilling(partials, &FUNCS, store).unwrap()
    }

    #[test]
    fn mem_spill_equals_oracle() {
        let keys = strs(&["a", "b", "a", "c", "b", "a", "d", "c", "b", "a"]);
        let vals = i64s(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let oracle =
            group_aggregate(std::slice::from_ref(&keys), &calls(&vals), keys.len()).unwrap();
        let want = to_map(&oracle.group_columns[0], &oracle.agg_columns);

        // Many partitions + many chunks forces routing/merge to do real work.
        let mut store = MemSpillStore::new(4);
        let got = spilled(&keys, &vals, 5, &mut store);
        assert_eq!(want, to_map(&got.group_columns[0], &got.agg_columns));
    }

    #[test]
    fn disk_spill_equals_oracle() {
        let keys = strs(&["a", "b", "a", "c", "b", "a", "d", "c", "b", "a", "e", "a"]);
        let vals = i64s(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

        let oracle =
            group_aggregate(std::slice::from_ref(&keys), &calls(&vals), keys.len()).unwrap();
        let want = to_map(&oracle.group_columns[0], &oracle.agg_columns);

        let dir = std::env::temp_dir().join(format!("bc_spill_test_{}", std::process::id()));
        let mut store = DiskSpillStore::new(dir, 8).unwrap();
        let got = spilled(&keys, &vals, 6, &mut store);
        assert_eq!(want, to_map(&got.group_columns[0], &got.agg_columns));
    }

    #[test]
    fn concurrent_disk_stores_under_one_root_are_isolated() {
        // Two stores sharing one spill root must not collide on `part-*.arrow`, and
        // one store's drop must not delete the other's files. Regression for the
        // distributed-reducer clobber bug (many worker processes, one spill dir):
        // interleave appends across both, drop the first, then the second still reads
        // its own data back correctly.
        let keys_a = strs(&["a", "b", "a", "c", "b", "a"]);
        let vals_a = i64s(&[1, 2, 3, 4, 5, 6]);
        let keys_b = strs(&["x", "y", "x", "z", "y", "x"]);
        let vals_b = i64s(&[10, 20, 30, 40, 50, 60]);

        let want_a = to_map(
            &group_aggregate(std::slice::from_ref(&keys_a), &calls(&vals_a), keys_a.len())
                .unwrap()
                .group_columns[0],
            &group_aggregate(std::slice::from_ref(&keys_a), &calls(&vals_a), keys_a.len())
                .unwrap()
                .agg_columns,
        );
        let want_b = to_map(
            &group_aggregate(std::slice::from_ref(&keys_b), &calls(&vals_b), keys_b.len())
                .unwrap()
                .group_columns[0],
            &group_aggregate(std::slice::from_ref(&keys_b), &calls(&vals_b), keys_b.len())
                .unwrap()
                .agg_columns,
        );

        let root = std::env::temp_dir().join(format!("bc_spill_shared_{}", std::process::id()));
        let mut store_a = DiskSpillStore::new(root.clone(), 8).unwrap();
        let mut store_b = DiskSpillStore::new(root.clone(), 8).unwrap();
        // Distinct private subdirectories — proving the file namespaces don't alias.
        assert_ne!(store_a.dir, store_b.dir);

        let got_a = spilled(&keys_a, &vals_a, 3, &mut store_a);
        drop(store_a); // wipes only store_a's private subdir
        let got_b = spilled(&keys_b, &vals_b, 3, &mut store_b);

        assert_eq!(want_a, to_map(&got_a.group_columns[0], &got_a.agg_columns));
        assert_eq!(want_b, to_map(&got_b.group_columns[0], &got_b.agg_columns));
    }

    #[test]
    fn single_partition_equals_oracle() {
        // P=1 degenerates to plain combine+finalize — a useful sanity floor.
        let keys = strs(&["x", "y", "x", "y", "z"]);
        let vals = i64s(&[5, 6, 7, 8, 9]);
        let oracle =
            group_aggregate(std::slice::from_ref(&keys), &calls(&vals), keys.len()).unwrap();
        let want = to_map(&oracle.group_columns[0], &oracle.agg_columns);

        let mut store = MemSpillStore::new(1);
        let got = spilled(&keys, &vals, 3, &mut store);
        assert_eq!(want, to_map(&got.group_columns[0], &got.agg_columns));
    }
}
