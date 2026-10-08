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

/// The most mappings the cache keeps between reads: a quarter of the kernel's per-process limit
/// on mappings (`vm.max_map_count`, 65,530 by default, so 16,382), and never fewer than 1,024.
///
/// A scan of more files than this still maps each one; the cap only stops the cache holding
/// every file a long-lived process ever read. It was a flat 4,096, which TPC-H sf1000 crosses
/// in one query -- 1,000 files in each of six tables, so q8 opens 5,002 -- and every crossing
/// cleared the whole cache under its lock (see [`open`]).
fn max_cached_files() -> usize {
    static C: OnceLock<usize> = OnceLock::new();
    *C.get_or_init(|| {
        std::fs::read_to_string("/proc/sys/vm/max_map_count")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .map_or(16_382, |n| n / 4)
            .max(1024)
    })
}

/// A file's `(length, modification time)`, which a cached mapping must still match.
type Stamp = (u64, Option<SystemTime>);

/// Every mapping currently shared, by absolute path, with the tick of its last open.
#[derive(Default)]
struct MapCache {
    files: HashMap<PathBuf, (Stamp, Arc<MappedFile>, u64)>,
    tick: u64,
}

impl MapCache {
    /// The cached mapping of `path` if it still matches `stamp`, marked as just used.
    fn get(&mut self, path: &Path, stamp: Stamp) -> Option<Arc<MappedFile>> {
        self.tick += 1;
        let tick = self.tick;
        let (seen, file, used) = self.files.get_mut(path)?;
        if *seen != stamp {
            return None;
        }
        *used = tick;
        Some(Arc::clone(file))
    }

    /// Cache `file` for `path`, keeping at most `cap` entries, and return what it displaced.
    ///
    /// A full cache gives up its least recently opened quarter, not everything: the files a
    /// running query is reading stay mapped, where clearing the lot made every reader of them
    /// map them again. The displaced mappings are handed back rather than dropped here, so
    /// their unmapping -- which for a large, populated mapping is the kernel tearing down every
    /// page-table entry -- runs after the caller has released the lock (see [`open`]).
    fn insert(
        &mut self,
        path: &Path,
        stamp: Stamp,
        file: &Arc<MappedFile>,
        cap: usize,
    ) -> Vec<Arc<MappedFile>> {
        let mut displaced = Vec::new();
        self.tick += 1;
        let entry = (stamp, Arc::clone(file), self.tick);
        if let Some((_, old, _)) = self.files.insert(path.to_path_buf(), entry) {
            displaced.push(old);
        }
        if self.files.len() > cap {
            let mut by_age: Vec<(u64, PathBuf)> = self
                .files
                .iter()
                .map(|(p, (_, _, used))| (*used, p.clone()))
                .collect();
            by_age.sort_unstable_by_key(|(used, _)| *used);
            let evict = self.files.len() - cap * 3 / 4;
            for (_, p) in by_age.into_iter().take(evict) {
                if let Some((_, f, _)) = self.files.remove(&p) {
                    displaced.push(f);
                }
            }
        }
        displaced
    }
}

fn cache() -> &'static Mutex<MapCache> {
    static C: OnceLock<Mutex<MapCache>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(MapCache::default()))
}

/// The shared mapping of `path`, or `None` when it cannot or should not be mapped.
///
/// `size` is the length the caller's metadata says the file has; a mapping of any other
/// length is not the file the caller planned against, so it is refused.
///
/// The cache's lock is held only to look up and to insert. Mapping a file and unmapping the
/// ones an insert displaces both happen outside it: an unmap of a large, populated mapping is
/// the kernel tearing down every page-table entry, and run under one global lock it serialized
/// every reader on the machine -- TPC-H sf1000 q8 at 64 cores sat at 2% occupancy for 1.6 s,
/// almost every sample in `zap_pte_range` under this function.
pub(crate) fn open(path: &Path, size: u64) -> Option<Arc<MappedFile>> {
    open_in(cache(), path, size, max_cached_files())
}

fn open_in(cache: &Mutex<MapCache>, path: &Path, size: u64, cap: usize) -> Option<Arc<MappedFile>> {
    if !enabled() {
        return None;
    }
    let meta = std::fs::metadata(path).ok()?;
    if !meta.is_file() || meta.len() != size || size == 0 {
        return None;
    }
    let stamp: Stamp = (meta.len(), meta.modified().ok());
    if let Some(file) = cache.lock().ok()?.get(path, stamp) {
        return Some(file);
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
    let mapped = Arc::new(MappedFile { map });
    let (file, displaced) = {
        let mut guard = cache.lock().ok()?;
        // Another reader may have mapped the same file meanwhile: share theirs, drop ours.
        match guard.get(path, stamp) {
            Some(theirs) => (theirs, vec![mapped]),
            None => {
                let displaced = guard.insert(path, stamp, &mapped, cap);
                (mapped, displaced)
            }
        }
    };
    // Unmapped here, outside the lock; a mapping a reader still holds lives on until it is done.
    drop(displaced);
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
    fn a_full_cache_evicts_the_least_recently_opened_and_keeps_the_rest() {
        // Twelve files through a cache of eight: the cache never holds more than eight, the
        // files opened most recently stay shared, and a mapping a reader still holds keeps
        // reading after the cache has let it go.
        let dir = std::env::temp_dir().join(format!("bc-io-mapped-evict-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cache = Mutex::new(MapCache::default());
        let paths: Vec<PathBuf> = (0..12u8)
            .map(|i| {
                let p = dir.join(format!("f{i}.bin"));
                std::fs::File::create(&p)
                    .unwrap()
                    .write_all(&[i; 100])
                    .unwrap();
                p
            })
            .collect();
        let held = open_in(&cache, &paths[0], 100, 8).unwrap();
        let first: Vec<Arc<MappedFile>> = paths[1..8]
            .iter()
            .map(|p| open_in(&cache, p, 100, 8).unwrap())
            .collect();
        // Reopen file 1 so it is recent; file 0 (held) and 2.. are older.
        let again = open_in(&cache, &paths[1], 100, 8).unwrap();
        assert!(
            Arc::ptr_eq(&again, &first[0]),
            "a cached file is shared, not remapped"
        );
        for p in &paths[8..] {
            open_in(&cache, p, 100, 8).unwrap();
            assert!(cache.lock().unwrap().files.len() <= 8);
        }
        let guard = cache.lock().unwrap();
        assert!(
            !guard.files.contains_key(&paths[0]),
            "the oldest file is evicted"
        );
        for p in [&paths[1], &paths[9], &paths[10], &paths[11]] {
            assert!(
                guard.files.contains_key(p),
                "a recent file stays cached: {p:?}"
            );
        }
        drop(guard);
        assert_eq!(held.slice(0..1).unwrap().as_ref(), &[0u8]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_cap_follows_the_kernel_mapping_limit() {
        assert!(max_cached_files() >= 1024);
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
