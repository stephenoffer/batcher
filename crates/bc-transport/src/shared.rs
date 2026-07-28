//! Same-node, cross-process partition transfer via memory-mapped Arrow IPC.
//!
//! `DIRECT_MEMORY` serves a partition to a reducer in the *same process* straight from
//! the in-memory store. Two processes on the *same node* (the common case: many Ray
//! actors per host) cannot share that heap, so today they fall back to Flight — a gRPC
//! round-trip over loopback TCP, serializing the batches even though the bytes never
//! leave the machine. This module is the fast path between them: a mapper writes each
//! bucket as an Arrow IPC stream to a file under a shared directory (Linux tmpfs
//! `/dev/shm` when available, else a temp dir — a regular-file mmap is shared across
//! processes via the page cache on macOS too), and a same-node reducer `mmap`s and
//! reads it. No gRPC, no socket — the plasma-class same-node transfer.
//!
//! It is **best-effort**: a missing file (an empty bucket, a peer that didn't write
//! shm, or shm disabled) returns `None` so the caller falls back to Flight, which
//! stays correct. Files are keyed by the producer's advertised address + ticket, so a
//! reducer derives the exact path from `(source_addr, ticket)` it already holds.

use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::ptr::NonNull;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::buffer::Buffer;
use arrow::ipc::convert::fb_to_schema;
use arrow::ipc::reader::{read_footer_length, FileDecoder};
use arrow::ipc::root_as_footer;
use arrow::ipc::writer::FileWriter;
use memmap2::Mmap;

/// Create `dir` (and parents) **owner-only**, tightening it if it already exists.
///
/// The mode is both requested and asserted, because `create_dir_all` honours the process
/// umask (so it may get less than it asks for) and silently leaves an *existing*
/// directory's mode alone — which is the common case here, since the shm root outlives any
/// one query and an earlier run may have created it 0755.
///
/// A directory this process does not own cannot be tightened; that is tolerated rather
/// than fatal, because the files written inside are created 0600 regardless.
#[cfg(unix)]
pub(crate) fn create_private_dir(dir: &std::path::Path) -> std::io::Result<()> {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(dir)?;
    // Best-effort tightening of a pre-existing directory this process may not own.
    let _ = fs::set_permissions(dir, fs::Permissions::from_mode(0o700));
    Ok(())
}

#[cfg(not(unix))]
pub(crate) fn create_private_dir(dir: &std::path::Path) -> std::io::Result<()> {
    fs::create_dir_all(dir)
}

/// The directory same-node workers exchange shm partitions through, or `None` when no
/// writable shared location exists. Prefers Linux tmpfs (`/dev/shm`, RAM-backed) and
/// falls back to the OS temp dir (still cross-process via the page cache).
///
/// **Owner-only, and that is load-bearing.** A published bucket is the query's actual
/// rows, and `/dev/shm` and `/tmp` are world-writable: at the default 0755/0644 any local
/// user could read a shuffle's data, and — worse — *plant* a well-formed file under a
/// ticket a reducer is about to fetch. The decode below is hardened against a corrupt
/// file, but a planted file that decodes cleanly is read as authoritative shuffle data and
/// silently changes the answer. This is the same exposure the tiered spill store closes
/// with `private_dir`/`open_private`, and it is sharper here: the Flight path this
/// replaces is token-authenticated and optionally mTLS, so the same-node fast path must
/// not be the unauthenticated way in. Peers are the same OS user by construction (one Ray
/// cluster, one worker account), so owner-only costs nothing that was ever intended to work.
fn shm_root() -> Option<PathBuf> {
    for base in ["/dev/shm", "/tmp"] {
        let p = std::path::Path::new(base);
        if p.is_dir() {
            let root = p.join("batcher_shm");
            if create_private_dir(&root).is_ok() {
                return Some(root);
            }
        }
    }
    let root = std::env::temp_dir().join("batcher_shm");
    create_private_dir(&root).ok().map(|()| root)
}

/// Whether a shared-memory transfer directory is usable on this host.
pub fn shm_available() -> bool {
    shm_root().is_some()
}

/// Make `addr` safe to use as a path segment (`host:port` → `host_port`).
fn sanitize(addr: &str) -> String {
    addr.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// The file a partition published by `addr` under `ticket` lives at (producer and
/// consumer derive the *same* path from data they both hold).
fn shm_path(addr: &str, ticket: &str) -> Option<PathBuf> {
    let dir = shm_root()?.join(sanitize(addr));
    create_private_dir(&dir).ok()?;
    mark_owner(&dir);
    Some(dir.join(format!("{}.arrow", sanitize(ticket))))
}

/// Create `path` for writing with owner-only permissions, truncating any existing file.
pub(crate) fn create_private_file(path: &std::path::Path) -> std::io::Result<File> {
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    opts.open(path)
}

/// The marker naming the process that owns a peer directory.
const OWNER_MARKER: &str = ".owner";

/// Record this process as the owner of `dir`, so a later run can tell whether the
/// directory belongs to a live worker. Best-effort; a missing marker only means the
/// directory is never reaped.
fn mark_owner(dir: &std::path::Path) {
    let marker = dir.join(OWNER_MARKER);
    if marker.exists() {
        return;
    }
    if let Ok(mut f) = create_private_file(&marker) {
        let _ = write!(f, "{}", std::process::id());
    }
}

/// Whether process `pid` is alive on this host.
#[cfg(unix)]
fn pid_alive(pid: u32) -> bool {
    // `/proc` rather than `kill(pid, 0)`: no unsafe, no signal semantics to get wrong, and
    // the answer is the same one the kernel would give. A non-Linux unix without `/proc`
    // reads as "alive", which is the conservative direction — it never reaps a live peer.
    if std::path::Path::new("/proc").is_dir() {
        return std::path::Path::new(&format!("/proc/{pid}")).exists();
    }
    true
}

#[cfg(not(unix))]
fn pid_alive(_pid: u32) -> bool {
    true
}

/// Delete shm directories left behind by workers that are no longer running.
///
/// The leak this closes is permanent memory. `clear_shared` frees a peer's buckets at
/// teardown, but only if the process *reaches* teardown — and the cases Carbonite's
/// resilience machinery exists for are exactly the ones where it does not: a SIGKILL, an
/// OOM kill, a spot reclamation. tmpfs is RAM, so every such exit strands its buckets in
/// `/dev/shm` until someone deletes them or the node reboots. A worker's advertised
/// address carries an *ephemeral* port, so each restart takes a fresh directory and the
/// dead ones accumulate: on a churning node that is an unbounded RAM leak whose only
/// symptom is that the box has less memory than it used to.
///
/// Reaping is safe because shm is same-node by construction — every directory here was
/// written by a process on *this* host, so its liveness is a local question. Only a
/// directory whose owner marker names a dead pid is removed; one with no marker (written
/// by an older build) or a live owner is left alone, so the check can never take a running
/// peer's buckets. Best-effort throughout: this is a fast path Flight always backs.
pub fn reap_stale_shm() {
    let Some(root) = shm_root() else { return };
    let Ok(entries) = fs::read_dir(&root) else {
        return;
    };
    for entry in entries.flatten() {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        let Ok(owner) = fs::read_to_string(dir.join(OWNER_MARKER)) else {
            continue; // no marker: an older build's directory, or a race — leave it
        };
        let Ok(pid) = owner.trim().parse::<u32>() else {
            continue;
        };
        if pid != std::process::id() && !pid_alive(pid) {
            let _ = fs::remove_dir_all(&dir);
        }
    }
}

/// Write `batches` as an Arrow IPC stream for a same-node reducer to mmap. The write
/// is atomic (write to a temp sibling, then rename) so a reader never sees a partial
/// file. Empty input writes nothing (the reducer falls back to Flight, which resolves
/// the empty bucket). Best-effort: any I/O error is reported for the caller to ignore.
pub fn publish_shared(addr: &str, ticket: &str, batches: &[RecordBatch]) -> std::io::Result<()> {
    if batches.is_empty() {
        return Ok(());
    }
    let path = shm_path(addr, ticket).ok_or_else(|| std::io::Error::other("no shm directory"))?;
    write_ipc_file(&path, batches)
}

/// Write `batches` to `path` as a 64-byte-aligned Arrow IPC **file**, atomically.
///
/// The file format (not the stream format) because it carries a footer of per-batch block
/// offsets, which is what lets [`read_ipc_file`] mmap it and decode each block zero-copy.
/// The write goes to a temp sibling and is renamed, so a reader never observes a partial
/// file. Owner-only at create — the destinations are world-writable directories and the
/// bytes are the query's rows.
pub(crate) fn write_ipc_file(
    path: &std::path::Path,
    batches: &[RecordBatch],
) -> std::io::Result<()> {
    let tmp = path.with_extension("arrow.tmp");
    {
        // Arrow IPC **file** format (with a footer of per-batch block offsets), 64-byte aligned,
        // so a same-node reader mmaps it and decodes each block ZERO-COPY — the arrays point
        // straight into the mmap instead of the reader copying every buffer out (`fetch_shared`).
        // 0600 at create, not by a later chmod: a chmod leaves a window in which the
        // query's rows are world-readable, and the bucket is fully written inside it.
        let file = create_private_file(&tmp)?;
        let opts = arrow::ipc::writer::IpcWriteOptions::try_new(
            64,
            false,
            arrow::ipc::MetadataVersion::V5,
        )
        .map_err(std::io::Error::other)?;
        let mut writer = FileWriter::try_new_with_options(file, &batches[0].schema(), opts)
            .map_err(std::io::Error::other)?;
        for b in batches {
            writer.write(b).map_err(std::io::Error::other)?;
        }
        writer.finish().map_err(std::io::Error::other)?;
        let mut file = writer.into_inner().map_err(std::io::Error::other)?;
        file.flush()?;
    }
    fs::rename(&tmp, path)
}

/// Read an Arrow IPC file written by [`write_ipc_file`], zero-copy over an mmap.
///
/// `Ok(None)` for a missing file; a corrupt or truncated one is also `Ok(None)` so a
/// best-effort caller can fall back rather than fail. Only a genuine I/O fault is `Err`.
pub(crate) fn read_ipc_file(path: &std::path::Path) -> std::io::Result<Option<Vec<RecordBatch>>> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    // SAFETY: the file is published atomically (write-temp-then-rename) and never mutated
    // in place, so the mapping's bytes do not change under us while we read.
    let mmap = unsafe { Mmap::map(&file)? };
    if mmap.len() < 10 {
        return Ok(None);
    }
    Ok(read_mmap_zero_copy(mmap).ok())
}

/// Read the batches a same-node peer published under `(addr, ticket)`, or `None` if no
/// usable file exists (an empty bucket, an un-shm'd peer, shm disabled, or a
/// corrupt/truncated file — every case falls back to Flight). The file is memory-mapped,
/// so the read is served from the page cache with no socket or gRPC decode.
///
/// This is **best-effort**: a decode failure on the world-writable shm path (a corrupt,
/// truncated, or hostile file) resolves to `None` — a miss the caller answers over Flight
/// — never an error or a panic, so a bad file can never fail or crash a healthy reducer.
/// Only a genuine I/O fault (a read error on the mapping) surfaces as `Err`.
pub fn fetch_shared(addr: &str, ticket: &str) -> std::io::Result<Option<Vec<RecordBatch>>> {
    let Some(path) = shm_path(addr, ticket) else {
        return Ok(None);
    };
    read_ipc_file(&path)
}

/// Decode the IPC-file `mmap` into batches whose buffers point INTO the mmap (zero-copy).
///
/// The mmap is wrapped as an Arrow [`Buffer`] whose owner is the `Mmap` itself (via `Arc`), so
/// the mapping stays alive exactly as long as any decoded array references it — the batches can
/// outlive this call and the file handle safely. Each footer block is decoded in place; a
/// 64-byte-aligned writer (`publish_shared`) keeps numeric buffers zero-copy (the decoder only
/// copies a buffer that is mis-aligned for its type, which does not occur here).
fn read_mmap_zero_copy(mmap: Mmap) -> Result<Vec<RecordBatch>, arrow::error::ArrowError> {
    let len = mmap.len();
    let ptr = NonNull::new(mmap.as_ptr() as *mut u8)
        .ok_or_else(|| arrow::error::ArrowError::IoError("null mmap".into(), null_io()))?;
    // SAFETY: `ptr`/`len` describe the live mapping; the `Mmap` is moved into the Buffer (as its
    // allocation owner) and dropped only when the last referencing array is dropped, so the
    // memory outlives every decoded batch. `Arc::new(mmap)` coerces to `Arc<dyn Allocation>`.
    let buffer = unsafe { Buffer::from_custom_allocation(ptr, len, Arc::new(mmap)) };

    // A truncated / corrupt / hostile file (the shm dir is world-writable) must NOT panic the
    // reducer — this same-node fast path is best-effort. Every malformed shape returns an
    // `ArrowError` the caller turns into a miss (falling back to Flight), never an `unwrap` panic.
    if len < 10 {
        return Err(arrow::error::ArrowError::IpcError(
            "shm file shorter than the 10-byte IPC trailer".into(),
        ));
    }
    let trailer_start = len - 10;
    let trailer: [u8; 10] = buffer[trailer_start..]
        .try_into()
        .map_err(|_| arrow::error::ArrowError::IpcError("bad shm trailer".into()))?;
    let footer_len = read_footer_length(trailer)?;
    if footer_len > trailer_start {
        return Err(arrow::error::ArrowError::IpcError(
            "shm footer length exceeds file size".into(),
        ));
    }
    let footer = root_as_footer(&buffer[trailer_start - footer_len..trailer_start])
        .map_err(|e| arrow::error::ArrowError::IpcError(format!("bad shm footer: {e}")))?;
    let footer_schema = footer
        .schema()
        .ok_or_else(|| arrow::error::ArrowError::IpcError("shm footer has no schema".into()))?;
    let schema = Arc::new(fb_to_schema(footer_schema));
    let mut decoder = FileDecoder::new(schema, footer.version());
    for block in footer.dictionaries().iter().flatten() {
        let data = block_slice(&buffer, block)?;
        decoder.read_dictionary(block, &data)?;
    }
    let mut out = Vec::with_capacity(footer.recordBatches().map_or(0, |r| r.len()));
    if let Some(rbs) = footer.recordBatches() {
        for i in 0..rbs.len() {
            let block = rbs.get(i);
            let data = block_slice(&buffer, block)?;
            if let Some(b) = decoder.read_record_batch(block, &data)? {
                out.push(b);
            }
        }
    }
    Ok(out)
}

/// Slice a footer `block`'s `[offset, offset+meta+body)` region out of `buffer`, after
/// validating it lies fully within the buffer.
///
/// The block coordinates come from the file's footer, which on the world-writable shm
/// path is untrusted: a corrupt, truncated, or hostile file can carry a *valid*
/// flatbuffer footer whose block offset/length point past the (short) file, or even a
/// negative `i64`/`i32` that wraps huge under `as usize`. Passing those straight to
/// [`Buffer::slice_with_length`] panics (`offset + length > len`), crashing the reducer —
/// exactly what `read_mmap_zero_copy` promises never to do. Validate here and return an
/// `ArrowError` (⇒ a miss ⇒ fall back to Flight) instead.
fn block_slice(
    buffer: &Buffer,
    block: &arrow::ipc::Block,
) -> Result<Buffer, arrow::error::ArrowError> {
    let bad = |what: &str| {
        arrow::error::ArrowError::IpcError(format!(
            "shm block out of range: {what} (offset={}, meta={}, body={}, file_len={})",
            block.offset(),
            block.metaDataLength(),
            block.bodyLength(),
            buffer.len()
        ))
    };
    // Reject negative coordinates before the sign-flipping `as usize` cast.
    if block.offset() < 0 || block.metaDataLength() < 0 || block.bodyLength() < 0 {
        return Err(bad("negative coordinate"));
    }
    let offset = block.offset() as usize;
    let block_len = (block.metaDataLength() as usize)
        .checked_add(block.bodyLength() as usize)
        .ok_or_else(|| bad("length overflow"))?;
    let end = offset
        .checked_add(block_len)
        .ok_or_else(|| bad("end overflow"))?;
    if end > buffer.len() {
        return Err(bad("exceeds file"));
    }
    Ok(buffer.slice_with_length(offset, block_len))
}

fn null_io() -> std::io::Error {
    std::io::Error::from(std::io::ErrorKind::InvalidData)
}

/// Remove every shm file a worker published under `addr` (called at plan teardown so a
/// long-lived worker's shm dir doesn't accumulate every stage's buckets).
pub fn clear_shared(addr: &str) {
    if let Some(root) = shm_root() {
        let _ = fs::remove_dir_all(root.join(sanitize(addr)));
    }
}

/// Remove only the shm files `addr` published for `plan_id`, leaving other plans' alone.
///
/// The plan-scoped counterpart to [`clear_shared`], and the missing half of the in-memory
/// [`crate::PartitionStore::remove_prefix`]. A session-scoped worker fleet serves many
/// queries, so tearing down one query must not evict another's live buckets — but leaving
/// them costs *twice*: the `PartitionStore` entry and this file. tmpfs is RAM-backed, so
/// an un-cleared shm directory is a second memory leak on the same node, and nothing freed
/// it at all before this existed.
///
/// Best-effort by design, exactly like `publish_shared`: shm is a fast path that Flight
/// always backs, so a failure to evict wastes memory but cannot produce a wrong answer.
pub fn clear_plan_shared(addr: &str, plan_id: u64) {
    let Some(root) = shm_root() else { return };
    let dir = root.join(sanitize(addr));
    // Tickets render as `plan/stage/src/dst/epoch` and `sanitize` maps every non-alphanumeric
    // byte to `_`, so a plan's files are exactly those named `{plan_id}_…`. The trailing
    // separator matters: without it, clearing plan 1 would also take plan 10's files.
    let prefix = format!("{plan_id}_");
    let Ok(entries) = fs::read_dir(&dir) else {
        return;
    };
    for entry in entries.flatten() {
        if entry.file_name().to_string_lossy().starts_with(&prefix) {
            let _ = fs::remove_file(entry.path());
        }
    }
}

#[cfg(test)]
mod tests {
    /// `clear_plan_shared` must evict exactly one plan's files.
    ///
    /// It infers the filename prefix from the ticket format (`plan/stage/...` sanitized to
    /// `plan_stage_...`). That inference is the fragile part: if the ticket rendering ever
    /// changes, this eviction silently stops matching anything and the shm leak comes back
    /// with no failure anywhere. So the assumption is pinned here rather than left implicit.
    #[test]
    fn clear_plan_shared_evicts_one_plan_only() {
        use arrow::array::{Int64Array, RecordBatch};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        if super::shm_root().is_none() {
            return; // no tmpfs on this platform; the shm path is inert
        }
        let addr = "clear-plan-shared-test:1";
        super::clear_shared(addr);

        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1_i64]))]).unwrap();
        // Ticket strings exactly as `ShuffleTicket::__str__` renders them.
        for ticket in ["1/0/0/0/0", "1/0/1/0/0", "2/0/0/0/0", "10/0/0/0/0"] {
            super::publish_shared(addr, ticket, std::slice::from_ref(&batch)).unwrap();
        }
        assert!(super::fetch_shared(addr, "1/0/0/0/0").unwrap().is_some());

        super::clear_plan_shared(addr, 1);

        assert!(super::fetch_shared(addr, "1/0/0/0/0").unwrap().is_none());
        assert!(super::fetch_shared(addr, "1/0/1/0/0").unwrap().is_none());
        // Another plan's buckets survive — a session fleet serves many queries at once.
        assert!(super::fetch_shared(addr, "2/0/0/0/0").unwrap().is_some());
        // And plan 10 is not a prefix match for plan 1. Without the trailing separator in
        // the prefix this is the assertion that fails.
        assert!(super::fetch_shared(addr, "10/0/0/0/0").unwrap().is_some());

        super::clear_shared(addr);
    }

    use std::sync::Arc;

    use arrow::array::{
        Array, Float64Array, Int64Array, RecordBatch, StringArray, StringDictionaryBuilder,
    };
    use arrow::datatypes::{DataType, Field, Int32Type, Schema};

    use super::*;

    fn batch(vals: &[i64]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vals.to_vec()))]).unwrap()
    }

    #[test]
    fn publish_then_fetch_roundtrips_the_batches() {
        let addr = "host_1:55501";
        let ticket = "1/0/2/3/0";
        let batches = vec![batch(&[1, 2, 3]), batch(&[4, 5])];
        publish_shared(addr, ticket, &batches).unwrap();
        let got = fetch_shared(addr, ticket).unwrap().expect("published");
        assert_eq!(got.len(), 2);
        let total: usize = got.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 5);
        clear_shared(addr);
    }

    #[test]
    fn fetch_missing_is_none_not_error() {
        // An empty bucket / un-shm'd peer ⇒ None ⇒ the caller falls back to Flight.
        assert!(fetch_shared("nobody_99999:1", "9/9/9/9/9")
            .unwrap()
            .is_none());
        let addr = "host_2:55502";
        publish_shared(addr, "t", &[]).unwrap(); // empty ⇒ writes nothing
        assert!(fetch_shared(addr, "t").unwrap().is_none());
    }

    #[test]
    fn clear_removes_published_files() {
        let addr = "host_3:55503";
        publish_shared(addr, "t", &[batch(&[7])]).unwrap();
        assert!(fetch_shared(addr, "t").unwrap().is_some());
        clear_shared(addr);
        assert!(fetch_shared(addr, "t").unwrap().is_none());
    }

    /// The zero-copy decode must reproduce non-numeric layouts exactly: variable-length
    /// (utf8 offsets + data), validity bitmaps (nulls), and multiple columns — not just the
    /// single fixed-width buffer the primary round-trip covers. This is the correctness bar
    /// for enabling shared memory by default.
    #[test]
    fn roundtrips_strings_nulls_and_mixed_columns() {
        let addr = "host_4:55504";
        let ticket = "9/0/1/2/0";
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Utf8, true),
            Field::new("n", DataType::Int64, true),
            Field::new("f", DataType::Float64, false),
        ]));
        let s = StringArray::from(vec![Some("alpha"), None, Some(""), Some("δ-wide")]);
        let n = Int64Array::from(vec![Some(1), Some(-2), None, Some(4)]);
        let f = Float64Array::from(vec![1.5, 2.5, 3.5, 4.5]);
        let want =
            RecordBatch::try_new(schema, vec![Arc::new(s), Arc::new(n), Arc::new(f)]).unwrap();

        publish_shared(addr, ticket, std::slice::from_ref(&want)).unwrap();
        let got = fetch_shared(addr, ticket).unwrap().expect("published");
        assert_eq!(got, vec![want]);
        clear_shared(addr);
    }

    /// A corrupt / hostile file whose footer is a *valid* flatbuffer but whose block
    /// offsets/lengths point past the (truncated) file must resolve to a miss, never a
    /// panic — the shm dir is world-writable, so a co-tenant or a half-written file from
    /// a crashed non-batcher writer must not crash the reducer. We reproduce it by
    /// publishing a valid file, then rewriting the same path with only its footer+trailer
    /// (body dropped): the footer's blocks now reference offsets far beyond the tiny file.
    #[test]
    fn corrupt_footer_with_out_of_range_blocks_is_a_miss_not_a_panic() {
        let addr = "host_6:55506";
        let ticket = "9/0/3/3/0";
        // A valid multi-batch file so the footer references non-trivial block offsets.
        publish_shared(addr, ticket, &[batch(&[1, 2, 3]), batch(&[4, 5, 6])]).unwrap();
        let path = shm_path(addr, ticket).unwrap();
        let v = fs::read(&path).unwrap();
        let len = v.len();
        // IPC-file trailer: <i32 footer_len LE><"ARROW1"> = 10 bytes at end.
        let footer_len = i32::from_le_bytes(v[len - 10..len - 6].try_into().unwrap()) as usize;
        let footer_start = len - 10 - footer_len;
        // Keep the valid footer + trailer, drop the schema + all batch bodies. The footer's
        // block offsets (absolute, into the original body region) now exceed this file.
        let mut corrupt = Vec::new();
        corrupt.extend_from_slice(&v[footer_start..len - 10]); // footer
        corrupt.extend_from_slice(&v[len - 10..]); // trailer
        fs::write(&path, &corrupt).unwrap();

        // Must not panic: a corrupt file resolves to a best-effort miss (`Ok(None)` ⇒ the
        // caller falls back to Flight), never a process-killing index-out-of-bounds in
        // `Buffer::slice_with_length`.
        let got = fetch_shared(addr, ticket);
        assert!(
            matches!(got, Ok(None)),
            "corrupt shm file must be a miss, got {got:?}"
        );
        clear_shared(addr);
    }

    /// Dictionary-encoded columns exercise the `read_dictionary` footer path (dictionary
    /// blocks decoded before the record batches) — a distinct code path from plain arrays.
    #[test]
    fn roundtrips_dictionary_encoded_column() {
        let addr = "host_5:55505";
        let ticket = "9/0/2/2/0";
        let mut b = StringDictionaryBuilder::<Int32Type>::new();
        for v in ["red", "green", "red", "blue", "green", "red"] {
            b.append_value(v);
        }
        let dict = b.finish();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "c",
            dict.data_type().clone(),
            false,
        )]));
        let want = RecordBatch::try_new(schema, vec![Arc::new(dict)]).unwrap();

        publish_shared(addr, ticket, std::slice::from_ref(&want)).unwrap();
        let got = fetch_shared(addr, ticket).unwrap().expect("published");
        assert_eq!(got, vec![want]);
        clear_shared(addr);
    }

    /// The shm path holds the query's actual rows on a world-writable filesystem
    /// (`/dev/shm`, `/tmp`). At the default 0755/0644 any local user could read a
    /// shuffle's data, or plant a well-formed file under a ticket a reducer is about to
    /// fetch — and a planted file that decodes cleanly is read as authoritative shuffle
    /// data, silently changing the answer. The Flight path this replaces is
    /// token-authenticated, so the same-node fast path must not be the way around that.
    #[cfg(unix)]
    #[test]
    fn published_buckets_and_their_directories_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let addr = "127.0.0.1:59991";
        let ticket = "perm/0/0/0";
        publish_shared(addr, ticket, &[batch(&[1])]).expect("publish");

        let path = shm_path(addr, ticket).expect("a shm path");
        let file_mode = fs::metadata(&path)
            .expect("published file")
            .permissions()
            .mode();
        assert_eq!(
            file_mode & 0o777,
            0o600,
            "the bucket is readable by other local users"
        );

        let dir_mode = fs::metadata(path.parent().expect("peer dir"))
            .expect("peer dir")
            .permissions()
            .mode();
        assert_eq!(
            dir_mode & 0o777,
            0o700,
            "the peer directory is world-traversable"
        );

        let root_mode = fs::metadata(shm_root().expect("root"))
            .expect("root")
            .permissions()
            .mode();
        assert_eq!(root_mode & 0o777, 0o700, "the shm root is world-writable");

        let _ = fs::remove_file(&path);
    }

    /// Tightening must also apply to a root some earlier run left world-writable, since
    /// the directory outlives any one query and `create_dir_all` leaves an existing mode alone.
    #[cfg(unix)]
    #[test]
    fn an_existing_loose_root_is_tightened() {
        use std::os::unix::fs::PermissionsExt;

        let root = shm_root().expect("a shm root");
        fs::set_permissions(&root, fs::Permissions::from_mode(0o755)).expect("loosen");
        let tightened = shm_root().expect("a shm root");
        let mode = fs::metadata(&tightened).expect("root").permissions().mode();
        assert_eq!(
            mode & 0o777,
            0o700,
            "a pre-existing loose root was left loose"
        );
    }

    /// tmpfs is RAM, and `clear_shared` only runs if the process reaches teardown — which
    /// a SIGKILL, an OOM kill, or a spot reclamation never does. Each restart advertises a
    /// fresh ephemeral port, so dead directories accumulate rather than being reused. The
    /// reaper must take those and must never take a live peer's.
    #[cfg(unix)]
    #[test]
    fn stale_peer_directories_are_reaped_and_live_ones_are_not() {
        let root = shm_root().expect("a shm root");

        // A directory owned by a pid that cannot exist (pid_max is well under this).
        let dead = root.join("reap_dead_peer");
        create_private_dir(&dead).expect("dead dir");
        fs::write(dead.join(OWNER_MARKER), "4294967290").expect("dead marker");
        fs::write(dead.join("bucket.arrow"), b"x").expect("dead bucket");

        // One owned by this very process, and one with no marker at all (an older build).
        let live = root.join("reap_live_peer");
        create_private_dir(&live).expect("live dir");
        mark_owner(&live);
        let unmarked = root.join("reap_unmarked_peer");
        create_private_dir(&unmarked).expect("unmarked dir");

        reap_stale_shm();

        assert!(
            !dead.exists(),
            "a dead worker's shm directory was left in RAM"
        );
        assert!(live.exists(), "the reaper took a live peer's buckets");
        assert!(
            unmarked.exists(),
            "an unmarked directory was reaped on no evidence"
        );

        let _ = fs::remove_dir_all(&live);
        let _ = fs::remove_dir_all(&unmarked);
    }

    /// The marker must not itself be readable by other local users, and must name us.
    #[cfg(unix)]
    #[test]
    fn the_owner_marker_is_private_and_names_this_process() {
        use std::os::unix::fs::PermissionsExt;

        let addr = "127.0.0.1:59992";
        publish_shared(addr, "own/0/0/0", &[batch(&[1])]).expect("publish");
        let dir = shm_root().expect("root").join(sanitize(addr));
        let marker = dir.join(OWNER_MARKER);

        let owner = fs::read_to_string(&marker).expect("marker");
        assert_eq!(owner.trim().parse::<u32>().ok(), Some(std::process::id()));
        let mode = fs::metadata(&marker).expect("marker").permissions().mode();
        assert_eq!(mode & 0o777, 0o600);

        let _ = fs::remove_dir_all(&dir);
    }
}
