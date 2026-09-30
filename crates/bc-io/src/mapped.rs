//! Serve a local Parquet file's column chunks straight out of a shared memory map.
//!
//! A local read through `object_store`'s `LocalFileSystem` costs three things the decoder
//! never needed: a hop onto tokio's blocking pool, a zero-fill of a freshly allocated buffer,
//! and a kernel copy of the page cache into it. Profiled on TPC-H q6 over sf10 `lineitem`
//! (10 files, four columns), those were `memset` 3.4% and the kernel's `rep_movs` 6.3% of the
//! whole query's CPU, with snappy's own decompression at 41%: work spent moving bytes the
//! decoder then reads once. A mapping hands the decoder the page cache itself: a range is a
//! [`Bytes`] view of the mapping, with no allocation and no copy.
//!
//! ## What a mapping assumes
//!
//! A mapped file that is **truncated** while mapped raises `SIGBUS` on the next read of a lost
//! page, where a `read` would have returned a short count or an error. Parquet files are
//! written once and never modified in place: every writer here (and every lakehouse format
//! Batcher reads) replaces a file rather than rewriting it, and a replaced file keeps the old
//! inode alive for as long as it is mapped, so the reader sees a consistent old version. The
//! residual case is an external process truncating a Parquet file in place mid-scan, which
//! corrupts the read on any path; `BATCHER_IO_MMAP=0` switches mapping off for a deployment
//! that must survive it. Every other failure (no such file, a special file, a mapping the OS
//! refuses) falls back to the ordinary reader rather than failing the query.
//!
//! A mapping is shared per file and revalidated on every open against the file's length and
//! modification time, so a file replaced between queries is mapped afresh rather than served
//! from the old one.

use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

use bytes::Bytes;
use parquet::errors::{ParquetError, Result as ParquetResult};

/// One mapped file, shared by every reader of it.
pub(crate) struct MappedFile {
    map: memmap2::Mmap,
}

/// A range of a [`MappedFile`], kept alive by the `Bytes` handed to the decoder.
struct MapRange {
    file: Arc<MappedFile>,
    start: usize,
    end: usize,
}

impl AsRef<[u8]> for MapRange {
    fn as_ref(&self) -> &[u8] {
        &self.file.map[self.start..self.end]
    }
}

impl MappedFile {
    /// The bytes of `range`, as a zero-copy view of the mapping.
    pub(crate) fn slice(self: &Arc<Self>, range: Range<u64>) -> ParquetResult<Bytes> {
        let (start, end) = (range.start as usize, range.end as usize);
        if start > end || end > self.map.len() {
            return Err(ParquetError::EOF(format!(
                "range {start}..{end} is outside a {}-byte file",
                self.map.len()
            )));
        }
        Ok(Bytes::from_owner(MapRange {
            file: Arc::clone(self),
            start,
            end,
        }))
    }
}

/// Whether local files may be mapped (`BATCHER_IO_MMAP=0` turns it off).
fn enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("BATCHER_IO_MMAP").map_or(true, |v| v != "0"))
}

/// The most mappings the cache keeps between reads. A scan of more files than this still maps
/// each one; it only stops the cache from holding every file a long-lived process ever read.
const MAX_CACHED_FILES: usize = 4096;

/// A file's `(length, modification time)`, which a cached mapping must still match.
type Stamp = (u64, Option<SystemTime>);
/// Every mapping currently shared, by absolute path.
type MapCache = Mutex<HashMap<PathBuf, (Stamp, Arc<MappedFile>)>>;

fn cache() -> &'static MapCache {
    static C: OnceLock<MapCache> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The shared mapping of `path`, or `None` when it cannot or should not be mapped.
///
/// `size` is the length the caller's metadata says the file has; a mapping of any other
/// length is not the file the caller planned against, so it is refused.
pub(crate) fn open(path: &Path, size: u64) -> Option<Arc<MappedFile>> {
    if !enabled() {
        return None;
    }
    let meta = std::fs::metadata(path).ok()?;
    if !meta.is_file() || meta.len() != size || size == 0 {
        return None;
    }
    let stamp: Stamp = (meta.len(), meta.modified().ok());
    let mut guard = cache().lock().ok()?;
    if let Some((seen, file)) = guard.get(path) {
        if *seen == stamp {
            return Some(Arc::clone(file));
        }
    }
    let handle = std::fs::File::open(path).ok()?;
    // SAFETY: the mapping is read-only and never written through. Its one hazard is the file
    // being truncated while mapped, which the module documentation covers: Parquet files are
    // replaced rather than rewritten in place, and `BATCHER_IO_MMAP=0` disables this path for
    // a deployment that cannot rule out an in-place truncation.
    let map = unsafe { memmap2::Mmap::map(&handle) }.ok()?;
    if map.len() as u64 != size {
        return None;
    }
    let file = Arc::new(MappedFile { map });
    if guard.len() >= MAX_CACHED_FILES {
        // A mapping pins its file's pages, and a deleted file's disk space, for as long as it
        // is held. Readers keep their own `Arc`s, so dropping the cache's copies unmaps only
        // the files nobody is reading; the next open of any of them maps it again.
        guard.clear();
    }
    guard.insert(path.to_path_buf(), (stamp, Arc::clone(&file)));
    Some(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_file(bytes: &[u8]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("bc-io-mapped-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("f{}.bin", bytes.len()));
        std::fs::File::create(&path)
            .unwrap()
            .write_all(bytes)
            .unwrap();
        path
    }

    #[test]
    fn a_range_reads_the_files_bytes() {
        let data: Vec<u8> = (0..=255u8).cycle().take(10_000).collect();
        let path = temp_file(&data);
        let file = open(&path, data.len() as u64).expect("maps");
        assert_eq!(file.slice(100..356).unwrap().as_ref(), &data[100..356]);
        assert_eq!(file.slice(0..0).unwrap().len(), 0);
    }

    #[test]
    fn a_range_past_the_end_is_an_error_not_a_panic() {
        let path = temp_file(&[7u8; 64]);
        let file = open(&path, 64).expect("maps");
        assert!(file.slice(60..65).is_err());
    }

    #[test]
    fn a_length_other_than_the_planned_one_is_not_mapped() {
        let path = temp_file(&[1u8; 32]);
        assert!(open(&path, 31).is_none());
    }

    #[test]
    fn a_replaced_file_is_mapped_again_and_the_old_view_is_unchanged() {
        // Replace-by-rename, the way every Parquet writer publishes a file: the old mapping
        // keeps the old inode, and the next open sees the new one.
        let path = temp_file(&[3u8; 128]);
        let first = open(&path, 128).expect("maps");
        let staged = path.with_extension("staged");
        std::fs::File::create(&staged)
            .unwrap()
            .write_all(&[4u8; 256])
            .unwrap();
        std::fs::rename(&staged, &path).unwrap();
        let second = open(&path, 256).expect("maps the new file");
        assert_eq!(second.slice(0..1).unwrap().as_ref(), &[4u8]);
        assert_eq!(first.slice(0..1).unwrap().as_ref(), &[3u8]);
    }
}
