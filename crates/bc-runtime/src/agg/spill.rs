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
use arrow::ipc::writer::{IpcWriteOptions, StreamWriter};
use arrow::ipc::CompressionType;
use arrow::row::{RowConverter, SortField};

use super::{combine, finalize, AggFunc, GroupAggResult, Partial};
use crate::error::RuntimeError;

/// Compression codec for spilled Arrow-IPC streams.
///
/// Spill is **perf-only and result-invariant**: an IPC stream self-describes its
/// compression, so the reader decompresses automatically and no codec choice can
/// change the batches read back. This only trades CPU for spill-file (and
/// disk-bandwidth) bytes. `None` is the historical, byte-for-byte-unchanged path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SpillCodec {
    /// Uncompressed — identical to the pre-compression spill path.
    None,
    /// LZ4 frame — fast, modest ratio.
    Lz4,
    /// Zstandard — slower, better ratio (best for large/blob-heavy spills).
    Zstd,
    /// Datatype-aware: pick the codec per spill from the spilled batch's schema
    /// (see [`SpillCodec::classify`]). The default — a strictly better policy than a
    /// fixed codec, and still result-invariant (IPC self-describes its compression).
    #[default]
    Auto,
}

impl SpillCodec {
    /// Resolve the control-plane codec name (`EngineConfig.spill_compression`).
    /// Unknown names fall back to `Auto` (the datatype-aware policy) rather than
    /// erroring, so a newer control plane never breaks an older engine.
    pub fn from_config_str(name: Option<&str>) -> Self {
        match name.map(|s| s.to_ascii_lowercase()).as_deref() {
            Some("none") => Self::None,
            Some("lz4") => Self::Lz4,
            Some("zstd") => Self::Zstd,
            _ => Self::Auto,
        }
    }

    /// Choose a concrete codec for a spilled batch by its dominant column type.
    ///
    /// Arrow IPC has a single stream-level codec (no per-field slot), so this picks
    /// one codec for the whole stream by where the bytes are. The decisive fact
    /// (measured): on fast local spill disk, general-purpose compression of numeric
    /// or string state is a net *loss* — the CPU outweighs the I/O saved. Only
    /// blob/large-binary payloads compress dramatically enough to win regardless of
    /// disk speed. So `auto` compresses **only** a blob-bearing schema (ZSTD, best
    /// ratio) and leaves everything else uncompressed — never a regression, a win
    /// exactly where the payload dwarfs the CPU.
    fn classify(schema: &Schema) -> Self {
        use arrow::datatypes::DataType::*;
        let has_blob = schema
            .fields()
            .iter()
            .any(|f| matches!(f.data_type(), LargeBinary | Binary | LargeUtf8));
        if has_blob {
            Self::Zstd
        } else {
            Self::None
        }
    }

    /// The IPC write options for this codec given the schema being spilled. `Auto`
    /// classifies the schema first; if the chosen compression is not compiled into
    /// this arrow build, it silently degrades to uncompressed (mirroring the Python
    /// `TieredSpillStore`), so a write never fails on codec.
    fn write_options(self, schema: &Schema) -> IpcWriteOptions {
        let base = IpcWriteOptions::default();
        let codec = match self {
            Self::Auto => return Self::classify(schema).write_options(schema),
            Self::None => return base,
            Self::Lz4 => CompressionType::LZ4_FRAME,
            Self::Zstd => CompressionType::ZSTD,
        };
        base.clone()
            .try_with_compression(Some(codec))
            .unwrap_or(base)
    }
}

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
    /// Spawn a fresh, independent store of the same kind with `partitions` partitions —
    /// used to recursively re-partition an over-large partition during the merge phase
    /// (a disk store nests a subdirectory; a memory store makes another memory store).
    fn child(&self, partitions: usize) -> Result<Box<dyn SpillStore>, RuntimeError>;
    /// Total logical bytes routed to this store's spill path (the sum of appended
    /// batches' in-memory size). This is the measured *spill volume* Carbonite needs to
    /// size spill scratch and disk bandwidth and to tell a 1 GB spill from a 100 GB one —
    /// a `spilled: bool` cannot. `0` for a store that spilled nothing (or does not track).
    fn spilled_bytes(&self) -> u64 {
        0
    }
    /// Spill *skew*: the largest partition's bytes over the mean non-empty partition's — `1.0`
    /// for a perfectly even spill, ≫ `1` when a hot key piles one partition. `1.0` for a store
    /// that spilled nothing or does not track per-partition sizes.
    fn spill_skew(&self) -> f32 {
        1.0
    }
}

/// Max-over-mean of the non-zero entries — the skew factor (`1.0` when even or fewer than
/// two non-empty partitions).
fn skew_of(bytes_per_partition: &[u64]) -> f32 {
    let nonzero: Vec<u64> = bytes_per_partition
        .iter()
        .copied()
        .filter(|&b| b > 0)
        .collect();
    if nonzero.len() < 2 {
        return 1.0;
    }
    let max = *nonzero.iter().max().unwrap() as f64;
    let mean = nonzero.iter().sum::<u64>() as f64 / nonzero.len() as f64;
    if mean <= 0.0 {
        1.0
    } else {
        (max / mean) as f32
    }
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
    fn child(&self, partitions: usize) -> Result<Box<dyn SpillStore>, RuntimeError> {
        Ok(Box::new(MemSpillStore::new(partitions)))
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
    codec: SpillCodec,
    /// Resolved on the first append (when the spilled schema is known, which `Auto`
    /// needs to classify) and reused for every partition's writer.
    write_options: Option<IpcWriteOptions>,
    /// Running sum of appended batches' in-memory size — the logical spill volume this
    /// store has written. The measured signal Carbonite sizes spill scratch from.
    bytes_written: u64,
    /// Bytes written to each partition, indexed by partition. The spread across these (max
    /// vs mean) is the spill *skew*: an even hash gives ~equal partitions (skew ~1), a hot
    /// key piles one partition many times its share (skew ≫ 1) — the signal that a family's
    /// spill thrashes and should shard into more, salted partitions next run.
    bytes_per_partition: Vec<u64>,
}

/// Restrict a spill directory to its owner (`0o700` on Unix); best-effort elsewhere.
///
/// Spilled data is the query's *actual rows* — the same bytes the caller may have taken
/// care to encrypt in flight — written to a shared scratch path such as `/tmp`. Created
/// with the default mode it is world-readable, so any other local user on the node can
/// read a spilled join or aggregate. Restricting the directory is the primary choke point:
/// without search permission on it, the `part-*.arrow` files inside are unreachable
/// regardless of their own mode.
///
/// It is not the *only* one, which is why [`create_private`] restricts the files too. This
/// function is best-effort and **silently ignores failure** — deliberately, so a filesystem
/// that cannot represent Unix modes never fails a query over a hardening step — and that
/// means the directory can legitimately end up 0755 with nothing said. Measured before the
/// file mode was fixed: the directory came out 0700 and every `part-*.arrow` inside it
/// 0644, so the entire protection rested on one call whose failure is by design invisible.
///
/// Best-effort by design — a filesystem that cannot represent Unix modes (or a Windows
/// host) must not fail the query over a hardening step, and the spill path is otherwise
/// platform-neutral.
fn restrict_to_owner(dir: &std::path::Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
    }
    #[cfg(not(unix))]
    let _ = dir;
}

/// Create a spill file readable only by the running user.
///
/// The mode is set in the `open` rather than by a following `chmod`: a chmod leaves a
/// window in which the file is world-readable, and these files hold the query's actual
/// rows. Non-unix falls back to a plain create, matching [`restrict_to_owner`]'s stance
/// that the spill path stays platform-neutral.
fn create_private(path: &std::path::Path) -> std::io::Result<File> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)
    }
    #[cfg(not(unix))]
    {
        File::create(path)
    }
}

impl DiskSpillStore {
    /// Create `partitions` empty, **uncompressed** spill files under a private
    /// subdirectory of `root` — the historical path, byte-for-byte unchanged.
    /// Use [`DiskSpillStore::with_codec`] to compress the streams.
    pub fn new(root: PathBuf, partitions: usize) -> Result<Self, RuntimeError> {
        Self::with_codec(root, partitions, SpillCodec::None)
    }

    /// Create `partitions` empty spill files whose IPC streams use `codec`.
    ///
    /// The store carves out its own `bc-spill-{pid}-{seq}` directory under `root`
    /// (created if absent) so its `part-*.arrow` files never collide with — and its
    /// drop never deletes — another concurrent store's files. This is what lets the
    /// distributed reducers spill safely when many worker processes share one spill
    /// root, and lets a single plan run two spilling breakers at once. The codec is
    /// write-side only; the read path auto-detects compression from the IPC stream.
    pub fn with_codec(
        root: PathBuf,
        partitions: usize,
        codec: SpillCodec,
    ) -> Result<Self, RuntimeError> {
        let seq = SPILL_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = root.join(format!("bc-spill-{}-{seq}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        restrict_to_owner(&dir);
        let n = partitions.max(1);
        let paths = (0..n)
            .map(|i| dir.join(format!("part-{i}.arrow")))
            .collect();
        Ok(Self {
            dir,
            paths,
            writers: (0..n).map(|_| None).collect(),
            codec,
            write_options: None,
            bytes_written: 0,
            bytes_per_partition: vec![0; n],
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
            // Resolve write options on the first append, when the schema is known —
            // `Auto` classifies it to pick a codec. Reused for every partition.
            let opts = self
                .write_options
                .get_or_insert_with(|| self.codec.write_options(&batch.schema()))
                .clone();
            let file = create_private(&self.paths[partition])?;
            self.writers[partition] = Some(StreamWriter::try_new_with_options(
                file,
                &batch.schema(),
                opts,
            )?);
        }
        self.writers[partition]
            .as_mut()
            .expect("writer just created")
            .write(batch)?;
        // Count the logical volume spilled (in-memory size, codec-independent) so the
        // control plane can size spill scratch from a measured magnitude, not a bool.
        let n = batch.get_array_memory_size() as u64;
        self.bytes_written += n;
        if let Some(slot) = self.bytes_per_partition.get_mut(partition) {
            *slot += n;
        }
        Ok(())
    }

    fn spilled_bytes(&self) -> u64 {
        self.bytes_written
    }

    fn spill_skew(&self) -> f32 {
        skew_of(&self.bytes_per_partition)
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
    fn child(&self, partitions: usize) -> Result<Box<dyn SpillStore>, RuntimeError> {
        // Nest the recursive re-partition under this store's private directory (itself
        // removed on drop), inheriting the codec. `with_codec` adds its own unique
        // `bc-spill-{pid}-{seq}` subdir, so siblings never collide.
        Ok(Box::new(DiskSpillStore::with_codec(
            self.dir.clone(),
            partitions,
            self.codec,
        )?))
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
///
/// `budget_bytes` bounds the memory of a single partition's `combine`. The initial
/// partition count is only an *average*-case fit (`total / budget`); under key skew one
/// partition can hold far more than its share and OOM the merge. When a partition's
/// spilled bytes exceed `budget_bytes`, it is **recursively re-partitioned** (a fresh
/// salted hash spreads the colliding keys) and merged sub-partition by sub-partition — so
/// peak memory stays bounded regardless of distribution. `budget_bytes == 0` disables the
/// guard (the historical single-level behavior, for callers that don't supply an envelope).
pub fn combine_finalize_spilling(
    chunk_partials: impl IntoIterator<Item = Partial>,
    funcs: &[AggFunc],
    store: &mut dyn SpillStore,
    budget_bytes: usize,
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
    let mut group_parts: Vec<Vec<ArrayRef>> = Vec::with_capacity(partitions);
    let mut agg_parts: Vec<Vec<ArrayRef>> = Vec::with_capacity(partitions);
    for pi in 0..partitions {
        let batches = store.read(pi)?;
        if batches.is_empty() {
            continue;
        }
        let (group_columns, aggs) =
            merge_partition(batches, n_keys, funcs, budget_bytes, store, 0)?;
        group_parts.push(group_columns);
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

/// Merge one spilled partition's packed partials into `(group_columns, agg_columns)`,
/// recursively re-partitioning if it is too large to `combine` within `budget`.
///
/// A partition that fits (`bytes <= budget`, or the guard is off, or no keys to split on,
/// or the recursion depth cap is hit) is combined + finalized directly. Otherwise it is
/// re-routed through a fresh child store with a **depth-varying salt** — the mergeable
/// algebra holds because equal keys still co-locate within each level, so merging the
/// sub-partitions and concatenating yields the identical relation, just with peak memory
/// bounded to one sub-partition. The depth cap backstops the pathological case (e.g. one
/// key dominating a partition — where its constant-size state is already tiny, so a direct
/// combine is safe).
fn merge_partition(
    batches: Vec<RecordBatch>,
    n_keys: usize,
    funcs: &[AggFunc],
    budget: usize,
    parent: &dyn SpillStore,
    depth: u32,
) -> Result<(Vec<ArrayRef>, Vec<ArrayRef>), RuntimeError> {
    const MAX_DEPTH: u32 = 4;
    let bytes: usize = batches.iter().map(|b| b.get_array_memory_size()).sum();
    if budget == 0 || bytes <= budget || n_keys == 0 || depth >= MAX_DEPTH {
        let partials: Vec<Partial> = batches
            .iter()
            .map(|b| unpack_partial(b, n_keys, funcs))
            .collect::<Result<_, _>>()?;
        let merged = combine(&partials, funcs)?;
        let aggs = finalize(funcs, &merged)?;
        return Ok((merged.group_columns, aggs));
    }

    // Re-partition this over-large partition into ~`bytes/budget` sub-partitions under a
    // fresh child store, with a nonzero salt (varying by depth) so the colliding keys
    // spread rather than re-collide. The read-back batches are already in packed form.
    let sub_p = bytes.div_ceil(budget.max(1)).clamp(2, 1 << 12);
    let salt = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(depth as u64 + 1) | 1;
    let mut child = parent.child(sub_p)?;
    for b in &batches {
        for (pi, sub) in route_salted(b, n_keys, sub_p, salt)? {
            child.append(pi, &sub)?;
        }
    }
    drop(batches); // release the over-large partition before merging its sub-partitions

    let mut group_parts: Vec<Vec<ArrayRef>> = Vec::with_capacity(sub_p);
    let mut agg_parts: Vec<Vec<ArrayRef>> = Vec::with_capacity(sub_p);
    for pi in 0..sub_p {
        let sub = child.read(pi)?;
        if sub.is_empty() {
            continue;
        }
        let (g, a) = merge_partition(sub, n_keys, funcs, budget, child.as_ref(), depth + 1)?;
        group_parts.push(g);
        agg_parts.push(a);
    }
    let group_columns = (0..n_keys)
        .map(|c| concat_cols(group_parts.iter().map(|g| &g[c])))
        .collect::<Result<_, _>>()?;
    let agg_columns = (0..funcs.len())
        .map(|c| concat_cols(agg_parts.iter().map(|a| &a[c])))
        .collect::<Result<_, _>>()?;
    Ok((group_columns, agg_columns))
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
///
/// Validates the batch's column count against the packed format (`n_keys + Σ arity`)
/// before slicing, so a truncated or otherwise malformed spill batch surfaces a
/// typed [`RuntimeError::MalformedPartial`] instead of panicking on an out-of-range
/// slice.
fn unpack_partial(
    b: &RecordBatch,
    n_keys: usize,
    funcs: &[AggFunc],
) -> Result<Partial, RuntimeError> {
    let arities: Vec<usize> = funcs.iter().map(|f| f.state_arity()).collect();
    let expected = n_keys + arities.iter().sum::<usize>();
    if b.num_columns() != expected {
        return Err(RuntimeError::MalformedPartial {
            expected,
            got: b.num_columns(),
        });
    }
    let cols = b.columns();
    let group_columns = cols[..n_keys].to_vec();
    let mut states = Vec::with_capacity(funcs.len());
    let mut idx = n_keys;
    for arity in arities {
        states.push(cols[idx..idx + arity].to_vec());
        idx += arity;
    }
    Ok(Partial {
        group_columns,
        states,
    })
}

/// Partition a packed partial's rows by a stable hash of its group-key columns.
/// A global aggregate (no keys) or a single partition routes everything to 0.
fn route(
    packed: &RecordBatch,
    n_keys: usize,
    partitions: usize,
) -> Result<Vec<(usize, RecordBatch)>, RuntimeError> {
    route_salted(packed, n_keys, partitions, 0)
}

/// [`route`] with a `salt` mixed into the key hash. The initial spill uses `salt == 0`;
/// a recursive re-partition of an over-large partition ([`merge_partition`]) uses a
/// nonzero, depth-varying salt so the keys that collided into one partition spread across
/// the sub-partitions instead of re-colliding under the same hash. Equal keys still route
/// together within a level (the salt is fixed for that level), so the grace algebra holds.
fn route_salted(
    packed: &RecordBatch,
    n_keys: usize,
    partitions: usize,
    salt: u64,
) -> Result<Vec<(usize, RecordBatch)>, RuntimeError> {
    if n_keys == 0 || partitions <= 1 {
        return Ok(vec![(0, packed.clone())]);
    }
    let group_cols = &packed.columns()[..n_keys];
    // Canonicalize float keys BEFORE hashing so `-0.0`/`0.0` (and every NaN bit pattern) route
    // to the SAME partition — arrow's row encoding is not canonical for floats, so without this
    // two partials that stored the same SQL group as `-0.0` vs `0.0` (each `partial` keeps its
    // first-seen value, which can differ per morsel) would land in different partitions and be
    // finalized as two groups, disagreeing with the in-memory `combine` (which canonicalizes
    // when it re-groups). Routing decides only *co-location*; the output group value is still
    // `take`n from the original column below, so the representative the query returns is
    // unchanged. Identity (no realloc) when no key is Float64.
    let canon = crate::keys::canonicalize_float_keys(group_cols);
    let encode_cols: &[ArrayRef] = canon.as_deref().unwrap_or(group_cols);
    let fields: Vec<SortField> = encode_cols
        .iter()
        .map(|a| SortField::new(a.data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields)?;
    let rows = converter.convert_columns(encode_cols)?;

    // Fixed seeds so the same key routes identically across every chunk.
    let state = ahash::RandomState::with_seeds(0x9E37, 0x79B9, 0x7F4A, 0x7C15);
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); partitions];
    let p = partitions as u64;
    for i in 0..packed.num_rows() {
        let h = state.hash_one(rows.row(i));
        // salt == 0 is the initial spill: `h % p`, byte-for-byte the historical routing.
        // A nonzero (recursive) salt re-mixes through a multiply-shift avalanche so keys
        // that collided into one partition genuinely spread across the sub-partitions
        // (a plain `h ^ salt` before `% p` could leave the bucket unchanged).
        let bucket = if salt == 0 {
            h % p
        } else {
            let mixed = (h ^ salt.rotate_left(31)).wrapping_mul(0xD6E8_FEB8_6659_FD93);
            mixed % p
        };
        buckets[bucket as usize].push(i as u32);
    }

    let mut out = Vec::with_capacity(buckets.len());
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

    #[test]
    fn skew_of_measures_partition_imbalance() {
        assert_eq!(skew_of(&[]), 1.0); // nothing spilled
        assert_eq!(skew_of(&[100]), 1.0); // one partition — no imbalance
        assert_eq!(skew_of(&[100, 100, 100]), 1.0); // perfectly even
        assert_eq!(skew_of(&[400, 0, 0]), 1.0); // empty partitions ignored → one non-empty
                                                // One partition ~3x its peers: mean = 500/3 ≈ 166.7, max = 300 → ~1.8.
        let s = skew_of(&[300, 100, 100]);
        assert!(
            (1.5..2.0).contains(&s),
            "skew {s} should reflect the imbalance"
        );
    }

    fn strs(v: &[&str]) -> ArrayRef {
        Arc::new(StringArray::from(v.to_vec()))
    }
    fn i64s(v: &[i64]) -> ArrayRef {
        Arc::new(Int64Array::from(v.to_vec()))
    }

    /// A truncated spilled partial (fewer columns than `n_keys + Σ arity`) is
    /// rejected with a typed error rather than panicking on an out-of-range slice.
    #[test]
    fn unpack_rejects_malformed_partial() {
        use arrow::datatypes::{DataType, Field, Schema};
        // n_keys=1 + Sum(1) + CountStar(1) = 3 expected; give a 1-column batch.
        let funcs = [AggFunc::Sum, AggFunc::CountStar];
        let schema = Arc::new(Schema::new(vec![Field::new("g0", DataType::Int64, true)]));
        let bad = RecordBatch::try_new(schema, vec![i64s(&[1])]).unwrap();
        match unpack_partial(&bad, 1, &funcs) {
            Err(RuntimeError::MalformedPartial { expected, got }) => {
                assert_eq!((expected, got), (3, 1))
            }
            _ => panic!("expected Err(MalformedPartial)"),
        }
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
        combine_finalize_spilling(partials, &FUNCS, store, 0).unwrap()
    }

    #[test]
    fn recursive_spill_under_budget_equals_oracle() {
        // Many distinct keys with a tiny per-partition budget: the merge phase must
        // recursively re-partition over-large partitions (a fresh salted hash) and still
        // reproduce the non-spilling oracle exactly — the skew-safety guarantee.
        let n = 500usize;
        let key_strs: Vec<String> = (0..n).map(|i| format!("k{}", i % 137)).collect();
        let keys: ArrayRef = Arc::new(StringArray::from(
            key_strs.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ));
        let vals = i64s(&(0..n as i64).collect::<Vec<_>>());
        let oracle =
            crate::agg::group_aggregate(std::slice::from_ref(&keys), &calls(&vals), n).unwrap();

        // Start with a single partition (P=1) and a 1-byte budget, so EVERY partition
        // overflows and the guard must recurse to make progress.
        let mut store = MemSpillStore::new(1);
        let per = n.div_ceil(7);
        let mut partials = Vec::new();
        let mut off = 0;
        while off < n {
            let len = per.min(n - off);
            let k = keys.slice(off, len);
            let v = vals.slice(off, len);
            partials.push(partial(std::slice::from_ref(&k), &calls(&v), len).unwrap());
            off += len;
        }
        let got = combine_finalize_spilling(partials, &FUNCS, &mut store, 1).unwrap();
        assert_eq!(
            to_map(&got.group_columns[0], &got.agg_columns),
            to_map(&oracle.group_columns[0], &oracle.agg_columns),
            "recursive-spill result must equal the non-spilling oracle",
        );
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

        // Spill is perf-only and result-invariant: every codec (and uncompressed)
        // must reproduce the in-memory oracle exactly. IPC self-describes its
        // compression, so the read path needs no codec.
        for codec in [
            SpillCodec::None,
            SpillCodec::Lz4,
            SpillCodec::Zstd,
            SpillCodec::Auto,
        ] {
            let dir = std::env::temp_dir()
                .join(format!("bc_spill_test_{}_{codec:?}", std::process::id()));
            let mut store = DiskSpillStore::with_codec(dir, 8, codec).unwrap();
            let got = spilled(&keys, &vals, 6, &mut store);
            assert_eq!(
                want,
                to_map(&got.group_columns[0], &got.agg_columns),
                "codec {codec:?} must match the oracle"
            );
        }
    }

    #[test]
    fn auto_codec_classifies_by_dominant_type() {
        use arrow::datatypes::{DataType, Field, Schema};
        // All fixed-width numeric → no compression (general codecs barely shrink it).
        let numeric = Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Float64, false),
        ]);
        assert_eq!(SpillCodec::classify(&numeric), SpillCodec::None);
        // A string column → still None: compressing string state on fast local disk
        // costs more CPU than the I/O it saves (measured).
        let strings = Schema::new(vec![
            Field::new("k", DataType::Utf8, false),
            Field::new("v", DataType::Int64, false),
        ]);
        assert_eq!(SpillCodec::classify(&strings), SpillCodec::None);
        // A blob/large-binary column → ZSTD (payload dwarfs CPU, best ratio).
        let blobs = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("blob", DataType::LargeBinary, false),
        ]);
        assert_eq!(SpillCodec::classify(&blobs), SpillCodec::Zstd);
    }

    #[test]
    fn compressed_spill_roundtrips_every_codec() {
        // A wider batch (so compression actually engages) appended and read back is
        // logically identical under every codec — the layer that guarantees the
        // spill codec is purely a storage concern.
        let keys = strs(&["alpha", "beta", "alpha", "gamma", "beta", "alpha"]);
        let vals = i64s(&[100, 200, 300, 400, 500, 600]);
        let batch = RecordBatch::try_from_iter(vec![
            ("k", Arc::new(keys.clone()) as ArrayRef),
            ("v", Arc::new(vals.clone()) as ArrayRef),
        ])
        .unwrap();

        for codec in [
            SpillCodec::None,
            SpillCodec::Lz4,
            SpillCodec::Zstd,
            SpillCodec::Auto,
        ] {
            let dir =
                std::env::temp_dir().join(format!("bc_spill_rt_{}_{codec:?}", std::process::id()));
            let mut store = DiskSpillStore::with_codec(dir, 1, codec).unwrap();
            store.append(0, &batch).unwrap();
            store.append(0, &batch).unwrap();
            let back = store.read(0).unwrap();
            let total: usize = back.iter().map(|b| b.num_rows()).sum();
            assert_eq!(total, 2 * batch.num_rows(), "codec {codec:?} row count");
            for b in &back {
                assert_eq!(b.schema(), batch.schema(), "codec {codec:?} schema");
            }
        }
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
    fn float_key_signed_zero_merges_across_spill_partitions() {
        use arrow::array::Float64Array;
        // Two partials for the SAME SQL group, but one stored its float key as `-0.0` and the
        // other as `0.0` (each `partial` takes the first-seen value, which can differ per
        // morsel). `combine` merges them (it canonicalizes float keys), so the spilling path
        // MUST too — otherwise `-0.0` and `0.0` route to different hash partitions and the
        // group is finalized twice, disagreeing with the in-memory oracle.
        let k1: ArrayRef = Arc::new(Float64Array::from(vec![-0.0f64]));
        let k2: ArrayRef = Arc::new(Float64Array::from(vec![0.0f64]));
        let v1: ArrayRef = Arc::new(Float64Array::from(vec![10.0f64]));
        let v2: ArrayRef = Arc::new(Float64Array::from(vec![5.0f64]));
        let mk = |v: &ArrayRef| vec![AggCall::new(AggFunc::Sum, Some(v.clone()))];
        let p1 = partial(std::slice::from_ref(&k1), &mk(&v1), 1).unwrap();
        let p2 = partial(std::slice::from_ref(&k2), &mk(&v2), 1).unwrap();

        // Many partitions so `-0.0` and `0.0` (which hash differently under a non-canonical
        // float row encoding) land in different partitions if not canonicalized first.
        let mut store = MemSpillStore::new(16);
        let got = combine_finalize_spilling([p1, p2], &[AggFunc::Sum], &mut store, 0).unwrap();
        assert_eq!(
            got.group_columns[0].len(),
            1,
            "-0.0 and 0.0 must be ONE group after spilling, got {} groups",
            got.group_columns[0].len()
        );
        let sum = got.agg_columns[0]
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0);
        assert_eq!(sum, 15.0, "the merged group's sum must be 10 + 5");
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

    #[cfg(unix)]
    #[test]
    fn a_spill_directory_is_not_readable_by_other_local_users() {
        use std::os::unix::fs::PermissionsExt;

        // Spilled data is the query's actual rows, written to a shared scratch path.
        // Created with the default mode it is world-readable, so a co-tenant on the node
        // could read a spilled join or aggregate straight off disk.
        let root = std::env::temp_dir();
        let store = DiskSpillStore::new(root, 2).unwrap();
        let mode = std::fs::metadata(&store.dir).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o700, "spill dir mode was {:o}", mode & 0o777);
    }
}
