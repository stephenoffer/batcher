//! The two spill stores and the codec that writes them.
//!
//! `SpillStore` is a partitioned, append-only staging area: the grace aggregate routes
//! partials into it by a hash of the group key and reads each partition back once. Two
//! implementations, and the difference is the whole point of the trait —
//! [`MemSpillStore`] keeps partitions in memory so the grace algebra can be proven against
//! the non-spilling oracle without touching a filesystem, and [`DiskSpillStore`] streams
//! them to Arrow IPC files, which is the path that actually bounds resident memory.
//!
//! Split out of the merge algorithm because they answer different questions: this file is
//! about *where the bytes live and who may read them*, `super` is about *how the mergeable
//! algebra reproduces the global aggregate one partition at a time*.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::{IpcWriteOptions, StreamWriter};
use arrow::ipc::CompressionType;

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
    pub(super) fn classify(schema: &Schema) -> Self {
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

/// Total write-buffer memory one spill store may hold across **all** its partitions.
///
/// Arrow's `StreamWriter` issues a separate `write` per IPC message *and per buffer within
/// it* — a batch with `k` columns costs on the order of `2k` syscalls, each typically a few
/// KB of validity/offset data. Written straight to a `File` that is one syscall per buffer;
/// a spill of a few thousand morsels over a dozen columns is hundreds of thousands of
/// syscalls spent on data that would coalesce into a handful of large writes. Buffering is
/// invisible to the reader (the IPC bytes are identical), so it is pure throughput.
///
/// It is budgeted in total rather than per file because the partition count is not small
/// and not bounded by the caller: a skewed grace aggregate re-partitions up to 4,096 ways,
/// so a flat 1 MiB per writer would be 4 GiB of buffers — spill's entire purpose is to *stop*
/// using memory, and a fixed per-file buffer would have made the store's own overhead scale
/// with the skew it exists to absorb.
const SPILL_WRITE_BUF_TOTAL: usize = 32 << 20;

/// Per-partition write buffer, as [`SPILL_WRITE_BUF_TOTAL`] shared among `partitions`.
///
/// Clamped below at the 8 KiB `BufWriter` default, so a very wide fan-out is never *worse*
/// than the unbuffered path it replaced, and above at 1 MiB, because the unit being written
/// is a morsel: at 16,384 rows a single numeric column already exceeds 128 KiB, and past a
/// megabyte the syscall saving flattens.
fn write_buf_capacity(partitions: usize) -> usize {
    (SPILL_WRITE_BUF_TOTAL / partitions.max(1)).clamp(8 << 10, 1 << 20)
}

/// Read buffer in front of each spill file (256 KiB).
///
/// The mirror of [`write_buf_capacity`] on the read side: the IPC `StreamReader` reads a
/// length prefix, then a metadata block, then the body, so the default 8 KiB buffer turns a
/// sequential scan into three syscalls per message plus one per body page. Sized smaller
/// than the write buffer because a bounded-fan-in merge holds one of these open *per run*
/// (16 by default), so this is multiplied by the fan-in while the write buffer is not.
const SPILL_READ_BUF: usize = 256 << 10;

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
    /// Finish `partition`'s writer, releasing whatever resource holds it open, while
    /// keeping the data readable.
    ///
    /// Append-then-read stores hold a writer open per partition until the partition is
    /// read back, which is fine when partitions are few and fixed. It is not fine for the
    /// external sort's pass 0, where a *run* is a partition: it writes each run once and
    /// never returns to it, so without this it holds one open file per input morsel and
    /// dies on `EMFILE` long before it runs out of disk — precisely on the large sorts
    /// spilling exists to serve. Calling this after the last append to a partition bounds
    /// open descriptors at O(1) instead of O(runs).
    ///
    /// Idempotent, and a no-op for a store that holds nothing open (the default).
    fn close_partition(&mut self, partition: usize) -> Result<(), RuntimeError> {
        let _ = partition;
        Ok(())
    }
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

    /// Bytes appended to `partition`, or `0` when the store does not track them.
    ///
    /// Asked *before* the partition is read, which is the whole point: it is what lets the
    /// merge decide to split a skewed partition without first pulling it into memory.
    fn partition_bytes(&self, partition: usize) -> u64 {
        let _ = partition;
        0
    }

    /// Hand `partition`'s batches to `sink` one at a time, then release the partition.
    ///
    /// The bounded-memory counterpart of [`SpillStore::read`], which returns the partition
    /// whole. The default reads it whole and feeds it through — correct for a store with no
    /// streaming reader, and no worse than what the caller would have done anyway.
    fn drain(
        &mut self,
        partition: usize,
        sink: &mut dyn FnMut(&RecordBatch) -> Result<(), RuntimeError>,
    ) -> Result<(), RuntimeError> {
        for b in self.read(partition)? {
            sink(&b)?;
        }
        Ok(())
    }
}

/// Max-over-mean of the non-zero entries — the skew factor (`1.0` when even or fewer than
/// two non-empty partitions).
pub(super) fn skew_of(bytes_per_partition: &[u64]) -> f32 {
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
    fn partition_bytes(&self, partition: usize) -> u64 {
        self.parts
            .get(partition)
            .map(|batches| {
                batches
                    .iter()
                    .map(|b| b.get_array_memory_size() as u64)
                    .sum()
            })
            .unwrap_or(0)
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
    writers: Vec<Option<StreamWriter<BufWriter<File>>>>,
    codec: SpillCodec,
    /// Resolved on the first append (when the spilled schema is known, which `Auto`
    /// needs to classify) and reused for every partition's writer.
    write_options: Option<IpcWriteOptions>,
    /// Running sum of appended batches' in-memory size — the logical spill volume this
    /// store has written. The measured signal Carbonite sizes spill scratch from.
    bytes_written: u64,
    /// Rows written to each partition — the count `read`/`drain` check what they got against.
    ///
    /// An Arrow IPC stream truncated at a message boundary is indistinguishable from a
    /// shorter valid stream: the reader returns the batches it finds and reports success. So
    /// the only thing that can tell a complete partition from a truncated one is a count
    /// taken on the way in. Without it a spill file that lost its tail — a short write a
    /// filesystem reported as success, a file that outlived the process writing it — makes
    /// the query return a wrong answer instead of failing.
    rows_per_partition: Vec<u64>,
    /// Bytes written to each partition, indexed by partition. The spread across these (max
    /// vs mean) is the spill *skew*: an even hash gives ~equal partitions (skew ~1), a hot
    /// key piles one partition many times its share (skew ≫ 1) — the signal that a family's
    /// spill thrashes and should shard into more, salted partitions next run.
    bytes_per_partition: Vec<u64>,
}

/// Spill roots this process has already swept for orphaned scratch, so the sweep costs one
/// `read_dir` per root rather than one per store.
static SWEPT_ROOTS: std::sync::Mutex<Option<std::collections::HashSet<PathBuf>>> =
    std::sync::Mutex::new(None);

/// Delete spill scratch left behind by processes that are no longer running.
///
/// A store removes its own directory on drop, which covers every ordinary end — success,
/// error, panic. It does not cover the one that matters most here: **`SIGKILL`**, and the
/// process most likely to be `SIGKILL`ed is the one spilling, because that is the process
/// the kernel OOM killer picks. Nothing runs on that path, so the scratch survives, and it
/// survives on the spill filesystem — so the next query has less room, spills harder, and is
/// likelier to be killed in turn. Left alone this ratchets a node into a state where every
/// large query fails for space while the data that filled the disk belongs to no one.
///
/// A directory is only removed when the pid embedded in its name is not a live process, so a
/// concurrently spilling sibling — the case the pid is in the name for — is never touched. A
/// reused pid makes the check say "alive" and the directory is kept, which is the safe way
/// to be wrong. Best-effort throughout: this is cleanup, and no failure of it may fail a
/// query.
fn sweep_orphaned_scratch(root: &std::path::Path) {
    {
        let mut swept = match SWEPT_ROOTS.lock() {
            Ok(g) => g,
            // A poisoned lock means another thread panicked mid-sweep. Skipping cleanup is
            // strictly better than propagating that into a query.
            Err(_) => return,
        };
        if !swept
            .get_or_insert_with(std::collections::HashSet::new)
            .insert(root.to_path_buf())
        {
            return;
        }
    }
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(pid) = orphan_pid(name) else {
            continue;
        };
        if !process_is_alive(pid) {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

/// The pid embedded in a `bc-spill-{pid}-{seq}` directory name, or `None` for any other
/// name — the sweep must never touch a directory it did not create.
fn orphan_pid(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("bc-spill-")?;
    let (pid, seq) = rest.split_once('-')?;
    // Both halves must parse, so a directory merely *starting* with the prefix is skipped.
    seq.parse::<u64>().ok()?;
    pid.parse().ok()
}

/// Whether `pid` is a live process on this host.
///
/// Read from `/proc`, which is exact on Linux — where the OOM killer this exists for lives —
/// and unavailable elsewhere. Anything that cannot answer says "alive", so a platform
/// without `/proc` simply never sweeps rather than deleting scratch that is in use.
fn process_is_alive(pid: u32) -> bool {
    if cfg!(target_os = "linux") {
        std::path::Path::new(&format!("/proc/{pid}")).exists()
    } else {
        true
    }
}

/// Turn a spill write failure into [`RuntimeError::SpillOutOfSpace`] when the filesystem (or
/// the quota) is what refused it, and leave every other I/O error alone.
///
/// "No space left on device" on its own is close to useless to whoever has to fix it: spill
/// scratch defaults to the system temp directory, which in a container is routinely a small
/// overlay or a tmpfs sized far below the query's spill volume while the large volume the
/// user believes is in use sits somewhere else entirely. Naming the directory and the volume
/// already written is what makes it actionable.
fn classify_spill_io(e: std::io::Error, dir: &str, written_bytes: u64) -> RuntimeError {
    // `StorageFull` is ENOSPC; `QuotaExceeded` is EDQUOT, which is the same situation for a
    // user on a quota'd filesystem and needs the same answer.
    if matches!(
        e.kind(),
        std::io::ErrorKind::StorageFull | std::io::ErrorKind::QuotaExceeded
    ) {
        return RuntimeError::SpillOutOfSpace {
            dir: dir.to_string(),
            written_bytes,
            source: e,
        };
    }
    RuntimeError::Io(e)
}

/// [`classify_spill_io`] for a failure that reached us through arrow.
///
/// The IPC writer wraps the underlying `io::Error` in an `ArrowError`, so a full disk arrives
/// as `ArrowError::IoError` and would otherwise be reported as a generic arrow failure — the
/// least informative form of the most common spill failure.
fn classify_spill_arrow(
    e: arrow::error::ArrowError,
    dir: &str,
    written_bytes: u64,
) -> RuntimeError {
    if let arrow::error::ArrowError::IoError(_, io) = e {
        return classify_spill_io(io, dir, written_bytes);
    }
    RuntimeError::Arrow(e)
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
    /// Create this store's private scratch directory under `root` and return its path.
    ///
    /// Uses `create_dir` — which fails if the name already exists — rather than
    /// `create_dir_all`, and that distinction is the security property. `create_dir_all`
    /// *succeeds* when the leaf is an attacker-planted symlink into a directory they can
    /// read: every spilled row (the query's actual data, the same bytes the caller may have
    /// taken care to encrypt in flight) is then written through it, with the owner-only
    /// mode applied to a path the attacker still controls. Any node-local user who can
    /// write the shared spill root can do this, and it leaves no trace in the query.
    ///
    /// Failing closed on a name clash would be a reliability regression on its own — pid
    /// reuse can leave a stale `bc-spill-{pid}-{seq}` from a killed process — so a clash
    /// advances the counter and retries instead. The counter is process-wide and monotonic,
    /// so this terminates immediately in practice; the bound is only there so a root that
    /// rejects every create (a full or read-only filesystem) surfaces its real error rather
    /// than spinning.
    fn claim_scratch_dir(root: &std::path::Path) -> Result<PathBuf, RuntimeError> {
        const ATTEMPTS: usize = 64;
        let mut last_err = None;
        for _ in 0..ATTEMPTS {
            let seq = SPILL_SEQ.fetch_add(1, Ordering::Relaxed);
            let dir = root.join(format!("bc-spill-{}-{seq}", std::process::id()));
            match std::fs::create_dir(&dir) {
                Ok(()) => return Ok(dir),
                // Someone (or a previous incarnation of this pid) already holds the name.
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => last_err = Some(e),
                Err(e) => return Err(e.into()),
            }
        }
        Err(last_err
            .unwrap_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    "spill scratch directory names exhausted",
                )
            })
            .into())
    }

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
        std::fs::create_dir_all(&root)?;
        // Once per root per process, before claiming space on it: reclaim scratch abandoned
        // by processes the OOM killer took. See `sweep_orphaned_scratch`.
        sweep_orphaned_scratch(&root);
        let dir = Self::claim_scratch_dir(&root)?;
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
            rows_per_partition: vec![0; n],
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
        self.close_partition(partition)?;
        if !self.paths[partition].exists() {
            return Ok(None);
        }
        let file = File::open(&self.paths[partition])?;
        Ok(Some(StreamReader::try_new(
            BufReader::with_capacity(SPILL_READ_BUF, file),
            None,
        )?))
    }
}

impl DiskSpillStore {
    /// Rows written to `partition`, so a caller streaming it through
    /// [`DiskSpillStore::open_reader`] can make the same check `read` and `drain` make.
    pub fn partition_rows(&self, partition: usize) -> u64 {
        self.rows_per_partition.get(partition).copied().unwrap_or(0)
    }

    /// Fail if a partition read back fewer rows than were written to it.
    ///
    /// The check exists because the failure it catches is *silent*. An Arrow IPC stream
    /// truncated at a message boundary — the last complete batch present, the end-of-stream
    /// marker gone — is byte-for-byte a shorter valid stream, so the reader returns what it
    /// finds and reports success. The query then produces a wrong answer with nothing
    /// anywhere saying so.
    ///
    /// Only a *short* read is an error. Reading more than was written cannot happen from
    /// truncation and would mean the count itself is wrong, so it is not worth failing a
    /// query over.
    fn verify_rows(&self, partition: usize, got: u64) -> Result<(), RuntimeError> {
        let expected = self.partition_rows(partition);
        if got >= expected {
            return Ok(());
        }
        Err(RuntimeError::SpillTruncated {
            dir: self.dir.display().to_string(),
            partition,
            expected_rows: expected,
            got_rows: got,
            missing: expected - got,
        })
    }

    /// This store's private scratch directory — the one removed on drop.
    ///
    /// Named rather than left as a bare field so the merge module's tests can assert the
    /// two properties that matter about it (concurrent stores get distinct directories,
    /// and the directory is owner-only) without the field itself becoming reachable.
    #[cfg(test)]
    pub(super) fn scratch_dir(&self) -> &std::path::Path {
        &self.dir
    }
}

impl SpillStore for DiskSpillStore {
    fn num_partitions(&self) -> usize {
        self.paths.len()
    }

    fn append(&mut self, partition: usize, batch: &RecordBatch) -> Result<(), RuntimeError> {
        let bytes_written = self.bytes_written;
        let dir = || self.dir.display().to_string();
        if self.writers[partition].is_none() {
            // Resolve write options on the first append, when the schema is known —
            // `Auto` classifies it to pick a codec. Reused for every partition.
            let opts = self
                .write_options
                .get_or_insert_with(|| self.codec.write_options(&batch.schema()))
                .clone();
            let file = create_private(&self.paths[partition])
                .map_err(|e| classify_spill_io(e, &dir(), bytes_written))?;
            let cap = write_buf_capacity(self.paths.len());
            self.writers[partition] = Some(StreamWriter::try_new_with_options(
                BufWriter::with_capacity(cap, file),
                &batch.schema(),
                opts,
            )?);
        }
        self.writers[partition]
            .as_mut()
            .expect("writer just created")
            .write(batch)
            .map_err(|e| classify_spill_arrow(e, &dir(), bytes_written))?;
        // Count the logical volume spilled (in-memory size, codec-independent) so the
        // control plane can size spill scratch from a measured magnitude, not a bool.
        let n = batch.get_array_memory_size() as u64;
        self.bytes_written += n;
        if let Some(slot) = self.bytes_per_partition.get_mut(partition) {
            *slot += n;
        }
        if let Some(slot) = self.rows_per_partition.get_mut(partition) {
            *slot += batch.num_rows() as u64;
        }
        Ok(())
    }

    fn spilled_bytes(&self) -> u64 {
        self.bytes_written
    }

    fn spill_skew(&self) -> f32 {
        skew_of(&self.bytes_per_partition)
    }

    /// Finish (flush + close) the partition's writer so the IPC stream is complete and its
    /// descriptor is released. Idempotent: a partition already closed, or never written,
    /// succeeds without doing anything.
    fn close_partition(&mut self, partition: usize) -> Result<(), RuntimeError> {
        if let Some(mut w) = self.writers.get_mut(partition).and_then(Option::take) {
            // `finish` writes the end-of-stream marker and flushes the `BufWriter`, so the
            // file is complete and readable once this returns.
            w.finish()?;
        }
        Ok(())
    }

    fn read(&mut self, partition: usize) -> Result<Vec<RecordBatch>, RuntimeError> {
        // Finish (flush + close) the writer so the IPC stream is complete before we read it
        // back. A partition with no live writer may still have been written and closed
        // early (`close_partition`), so existence — not the writer — decides whether there
        // is anything to read.
        self.close_partition(partition)?;
        if !self.paths[partition].exists() {
            return Ok(Vec::new());
        }
        let file = File::open(&self.paths[partition])?;
        let reader = StreamReader::try_new(BufReader::with_capacity(SPILL_READ_BUF, file), None)?;
        let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>()?;
        let got: u64 = batches.iter().map(|b| b.num_rows() as u64).sum();
        self.verify_rows(partition, got)?;
        Ok(batches)
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

    fn partition_bytes(&self, partition: usize) -> u64 {
        self.bytes_per_partition
            .get(partition)
            .copied()
            .unwrap_or(0)
    }

    /// Stream the partition off disk one IPC batch at a time — the reader
    /// [`DiskSpillStore::open_reader`] already provided and the merge could not reach
    /// through the trait.
    fn drain(
        &mut self,
        partition: usize,
        sink: &mut dyn FnMut(&RecordBatch) -> Result<(), RuntimeError>,
    ) -> Result<(), RuntimeError> {
        let Some(reader) = self.open_reader(partition)? else {
            // Nothing was written, so nothing is missing — but a partition that *was* written
            // and whose file has since vanished must not read as empty.
            return self.verify_rows(partition, 0);
        };
        let mut got = 0u64;
        for batch in reader {
            let batch = batch?;
            got += batch.num_rows() as u64;
            sink(&batch)?;
        }
        self.verify_rows(partition, got)
    }
}

impl Drop for DiskSpillStore {
    fn drop(&mut self) {
        // Best-effort cleanup of the temporary spill directory.
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The write buffering must be a *fixed* budget shared across partitions, not a fixed
    /// size per partition.
    ///
    /// A skewed grace aggregate re-partitions up to 4,096 ways and the grace join fans out
    /// 256 ways on each of two stores. At a flat 1 MiB per writer that is gigabytes of
    /// buffers — held by the subsystem whose entire purpose is to *stop* using memory, and
    /// growing with exactly the skew it exists to absorb. The floor matters as much: a wide
    /// fan-out must never end up with a smaller buffer than the `BufWriter` default it
    /// replaced, or the change would be a regression at the wide end.
    #[test]
    fn write_buffering_is_a_shared_budget_not_a_per_file_size() {
        for partitions in [1usize, 2, 16, 64, 256, 1024, 4096, 100_000] {
            let cap = write_buf_capacity(partitions);
            assert!(
                cap >= 8 << 10,
                "{partitions} partitions got a {cap}-byte buffer, below the BufWriter default"
            );
            assert!(
                cap <= 1 << 20,
                "{partitions} partitions got an oversized {cap}-byte buffer"
            );
        }
        // Few partitions get the full per-file buffer; many share the budget.
        assert_eq!(write_buf_capacity(1), 1 << 20);
        assert_eq!(write_buf_capacity(32), 1 << 20);
        assert!(write_buf_capacity(4096) < write_buf_capacity(64));
        // And the total stays bounded wherever the fan-out lands.
        for partitions in [16usize, 256, 4096] {
            let total = write_buf_capacity(partitions) * partitions;
            assert!(
                total <= SPILL_WRITE_BUF_TOTAL.max(32 << 20),
                "{partitions} partitions would hold {total} bytes of write buffers"
            );
        }
    }

    /// A full spill filesystem must be reported as *that*, not as a generic I/O or arrow
    /// failure. "No space left on device" alone does not say which device, and spill scratch
    /// defaults to the system temp directory — routinely a small container overlay rather
    /// than the large volume the user believes is in use.
    #[test]
    fn a_full_spill_filesystem_is_reported_as_such_through_both_error_paths() {
        let full = || std::io::Error::from(std::io::ErrorKind::StorageFull);

        match classify_spill_io(full(), "/scratch/spill", 4096) {
            RuntimeError::SpillOutOfSpace {
                dir, written_bytes, ..
            } => {
                assert_eq!(dir, "/scratch/spill");
                assert_eq!(written_bytes, 4096);
            }
            other => panic!("a full disk was reported as {other}"),
        }

        // The IPC writer wraps it in an `ArrowError`, which is how it actually arrives.
        let wrapped = arrow::error::ArrowError::IoError("write failed".into(), full());
        assert!(
            matches!(
                classify_spill_arrow(wrapped, "/scratch/spill", 1),
                RuntimeError::SpillOutOfSpace { .. }
            ),
            "a full disk reaching us through arrow was reported as a generic arrow error"
        );

        // A quota is the same situation for the user and needs the same answer.
        assert!(matches!(
            classify_spill_io(
                std::io::Error::from(std::io::ErrorKind::QuotaExceeded),
                "/scratch/spill",
                0
            ),
            RuntimeError::SpillOutOfSpace { .. }
        ));

        // Everything else keeps its own identity.
        assert!(matches!(
            classify_spill_io(
                std::io::Error::from(std::io::ErrorKind::PermissionDenied),
                "/scratch/spill",
                0
            ),
            RuntimeError::Io(_)
        ));
    }

    /// The orphan sweep deletes directories, so the name it accepts is a safety boundary: it
    /// must recognize exactly what it creates and nothing that merely resembles it.
    #[test]
    fn only_this_stores_own_scratch_names_are_sweepable() {
        assert_eq!(orphan_pid("bc-spill-1234-0"), Some(1234));
        assert_eq!(orphan_pid("bc-spill-1-99999"), Some(1));

        for name in [
            "someone-elses-data",
            "bc-spill",
            "bc-spill-",
            "bc-spill-1234",         // no counter
            "bc-spill-abc-0",        // pid is not a number
            "bc-spill-1234-abc",     // counter is not a number
            "bc-spill-1234-0-extra", // trailing junk
            "not-bc-spill-1234-0",   // prefix is not at the start
            "../escape",
        ] {
            assert_eq!(orphan_pid(name), None, "{name} would have been swept");
        }
    }
}
