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

/// The directory same-node workers exchange shm partitions through, or `None` when no
/// writable shared location exists. Prefers Linux tmpfs (`/dev/shm`, RAM-backed) and
/// falls back to the OS temp dir (still cross-process via the page cache).
fn shm_root() -> Option<PathBuf> {
    for base in ["/dev/shm", "/tmp"] {
        let p = std::path::Path::new(base);
        if p.is_dir() {
            let root = p.join("batcher_shm");
            if fs::create_dir_all(&root).is_ok() {
                return Some(root);
            }
        }
    }
    let root = std::env::temp_dir().join("batcher_shm");
    fs::create_dir_all(&root).ok().map(|()| root)
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
    fs::create_dir_all(&dir).ok()?;
    Some(dir.join(format!("{}.arrow", sanitize(ticket))))
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
    let tmp = path.with_extension("arrow.tmp");
    {
        // Arrow IPC **file** format (with a footer of per-batch block offsets), 64-byte aligned,
        // so a same-node reader mmaps it and decodes each block ZERO-COPY — the arrays point
        // straight into the mmap instead of the reader copying every buffer out (`fetch_shared`).
        let file = File::create(&tmp)?;
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
    fs::rename(&tmp, &path)
}

/// Read the batches a same-node peer published under `(addr, ticket)`, or `None` if no
/// file exists (an empty bucket, an un-shm'd peer, or shm disabled — fall back to
/// Flight). The file is memory-mapped, so the read is served from the page cache with
/// no socket or gRPC decode.
pub fn fetch_shared(addr: &str, ticket: &str) -> std::io::Result<Option<Vec<RecordBatch>>> {
    let Some(path) = shm_path(addr, ticket) else {
        return Ok(None);
    };
    let file = match File::open(&path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    // SAFETY: the file is published atomically (write-temp-then-rename) and never
    // mutated in place, so the mapping's bytes don't change under us while we read.
    let mmap = unsafe { Mmap::map(&file)? };
    if mmap.len() < 10 {
        return Ok(None); // too short to hold an IPC-file trailer — treat as a miss
    }
    read_mmap_zero_copy(mmap)
        .map(Some)
        .map_err(std::io::Error::other)
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

    let trailer_start = len - 10;
    let footer_len = read_footer_length(buffer[trailer_start..].try_into().unwrap())?;
    let footer = root_as_footer(&buffer[trailer_start - footer_len..trailer_start])
        .map_err(|e| arrow::error::ArrowError::IpcError(format!("bad shm footer: {e}")))?;
    let schema = Arc::new(fb_to_schema(footer.schema().unwrap()));
    let mut decoder = FileDecoder::new(schema, footer.version());
    for block in footer.dictionaries().iter().flatten() {
        let block_len = block.bodyLength() as usize + block.metaDataLength() as usize;
        let data = buffer.slice_with_length(block.offset() as usize, block_len);
        decoder.read_dictionary(block, &data)?;
    }
    let mut out = Vec::new();
    if let Some(rbs) = footer.recordBatches() {
        for i in 0..rbs.len() {
            let block = rbs.get(i);
            let block_len = block.bodyLength() as usize + block.metaDataLength() as usize;
            let data = buffer.slice_with_length(block.offset() as usize, block_len);
            if let Some(b) = decoder.read_record_batch(block, &data)? {
                out.push(b);
            }
        }
    }
    Ok(out)
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

#[cfg(test)]
mod tests {
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
}
