//! Telling the kernel how a spill file is about to be used.
//!
//! Every out-of-core operator in the engine — the spilling aggregation, the external sort's
//! run merge, the window spill — writes a file once and then reads it back exactly once, front
//! to back. That is the most predictable access pattern there is, and by default the kernel
//! does not know it: `read()` on a fresh file starts with a small readahead window (128 KiB on
//! a stock Linux) and grows it only after observing several sequential reads. A k-way merge
//! interleaving sixteen run files defeats even that heuristic, because from the block layer's
//! point of view the reads arrive interleaved rather than sequential per file.
//!
//! `posix_fadvise` closes the gap in one syscall per file. `SEQUENTIAL` doubles the readahead
//! window up front, so the merge is fetching the next chunk of each run while it processes the
//! current one instead of stalling on it. `DONTNEED`, issued once a run is exhausted, drops
//! those pages instead of leaving a spilled multi-gigabyte file resident in page cache — which
//! matters precisely when it is worst, since the engine only spilled because memory was tight.
//!
//! Every function here is **advisory and best-effort**: a failure means the hint was not taken,
//! never that the read or write is wrong. They are no-ops on non-Linux targets, where the
//! equivalents are either absent or spelled differently enough not to be worth a shim.

#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;

/// Tell the kernel this file will be read front to back, so it should read ahead aggressively.
///
/// Issue it once, right after opening a spill file for reading. The win is largest exactly
/// where the engine hurts most: a k-way merge over many run files, where per-file sequential
/// access is invisible to the block layer because the requests interleave.
///
/// Best-effort — a failure leaves the default readahead window in place.
#[cfg(target_os = "linux")]
pub fn advise_sequential(file: &std::fs::File) {
    // SAFETY: `posix_fadvise` takes a file descriptor this `&File` keeps open for the duration
    // of the call, a byte range, and an advice constant. It changes no program state and
    // cannot fail in a way that affects the read; the return code is a hint's acceptance.
    unsafe {
        libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_SEQUENTIAL);
    }
}

/// Tell the kernel this file will be read in a scattered order, so readahead is wasted work.
///
/// The counterpart to [`advise_sequential`], for a spill file probed by row index rather than
/// scanned. Readahead on a random pattern is not neutral: it fetches pages that are then
/// evicted unread, spending both bandwidth and cache the actual reads wanted.
#[cfg(target_os = "linux")]
pub fn advise_random(file: &std::fs::File) {
    // SAFETY: as `advise_sequential` — an advisory call over a live descriptor.
    unsafe {
        libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM);
    }
}

/// Start pulling `len` bytes from `offset` into page cache now, without waiting.
///
/// Use it when the next read is known well before it happens — a merge that has just decided
/// which run it will draw from next, or a partition list where the following partition's file
/// is already identified. `len == 0` means "to the end of the file".
///
/// Asynchronous: it queues the I/O and returns, so it hides latency rather than adding it.
#[cfg(target_os = "linux")]
pub fn advise_willneed(file: &std::fs::File, offset: u64, len: u64) {
    // SAFETY: as `advise_sequential`. An offset or length past the end of the file is
    // defined behavior for `posix_fadvise` — the kernel clamps to the file size.
    unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            offset as libc::off_t,
            len as libc::off_t,
            libc::POSIX_FADV_WILLNEED,
        );
    }
}

/// Drop this file's pages from page cache.
///
/// Call it when a spill file has been fully consumed. The engine spilled because memory was
/// tight, so leaving a consumed multi-gigabyte run resident is pressure on the exact resource
/// that forced the spill — and the pages will never be read again, since a run is consumed
/// once.
///
/// Only clean pages are dropped, so this is safe to call on a file that has been written:
/// dirty pages are left for writeback rather than discarded. Flush before calling if the
/// intent is to free written data.
#[cfg(target_os = "linux")]
pub fn advise_dontneed(file: &std::fs::File) {
    // SAFETY: as `advise_sequential`. `DONTNEED` cannot lose data — the kernel skips dirty
    // pages rather than discarding them.
    unsafe {
        libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
    }
}

/// Reserve `len` bytes of disk for this file up front.
///
/// A spill writer that grows a file by appending makes the filesystem allocate extents
/// incrementally, which fragments the run and turns the later sequential read into a seek per
/// extent. Reserving the whole run in one call gives the allocator a chance to place it
/// contiguously, and surfaces an out-of-space condition at reserve time rather than halfway
/// through writing a hash table.
///
/// Best-effort: on a filesystem without `fallocate` support this simply does nothing, and the
/// write proceeds exactly as it did before.
#[cfg(target_os = "linux")]
pub fn preallocate(file: &std::fs::File, len: u64) {
    if len == 0 {
        return;
    }
    // SAFETY: `posix_fallocate` takes a live descriptor and a byte range. It only reserves
    // blocks; it does not move the file offset or change any bytes the engine reads back.
    unsafe {
        libc::posix_fallocate(file.as_raw_fd(), 0, len as libc::off_t);
    }
}

#[cfg(not(target_os = "linux"))]
pub fn advise_sequential(_file: &std::fs::File) {}
#[cfg(not(target_os = "linux"))]
pub fn advise_random(_file: &std::fs::File) {}
#[cfg(not(target_os = "linux"))]
pub fn advise_willneed(_file: &std::fs::File, _offset: u64, _len: u64) {}
#[cfg(not(target_os = "linux"))]
pub fn advise_dontneed(_file: &std::fs::File) {}
#[cfg(not(target_os = "linux"))]
pub fn preallocate(_file: &std::fs::File, _len: u64) {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Seek, Write};

    fn temp_file_with(bytes: &[u8]) -> (tempdir::Dir, std::fs::File) {
        let dir = tempdir::Dir::new();
        let path = dir.path().join("spill.arrow");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(bytes).unwrap();
        f.sync_all().unwrap();
        (dir, std::fs::File::open(&path).unwrap())
    }

    /// A minimal scoped temp directory — the crate has no dev-dependency on `tempfile` and
    /// these tests need one file, not a fixture framework.
    mod tempdir {
        use std::path::{Path, PathBuf};
        use std::sync::atomic::{AtomicU64, Ordering};

        pub struct Dir(PathBuf);

        impl Dir {
            pub fn new() -> Self {
                static N: AtomicU64 = AtomicU64::new(0);
                let p = std::env::temp_dir().join(format!(
                    "bc-arrow-page-cache-{}-{}",
                    std::process::id(),
                    N.fetch_add(1, Ordering::Relaxed)
                ));
                std::fs::create_dir_all(&p).unwrap();
                Self(p)
            }
            pub fn path(&self) -> &Path {
                &self.0
            }
        }

        impl Drop for Dir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }

    /// The whole contract: an advice call changes what the kernel *prefetches*, never what a
    /// read returns. Applying every hint to one file and then reading it must yield the bytes
    /// that were written, byte for byte.
    #[test]
    fn advice_never_changes_the_bytes_read_back() {
        let payload: Vec<u8> = (0..64_000u32).map(|i| (i % 251) as u8).collect();
        let (_dir, mut f) = temp_file_with(&payload);
        advise_sequential(&f);
        advise_willneed(&f, 0, 0);
        let mut got = Vec::new();
        f.read_to_end(&mut got).unwrap();
        assert_eq!(got, payload);

        // Re-read after DONTNEED: the pages are gone from cache, the data is not.
        advise_dontneed(&f);
        f.rewind().unwrap();
        let mut again = Vec::new();
        f.read_to_end(&mut again).unwrap();
        assert_eq!(again, payload);

        advise_random(&f);
        f.rewind().unwrap();
        let mut third = Vec::new();
        f.read_to_end(&mut third).unwrap();
        assert_eq!(third, payload);
    }

    #[test]
    fn hints_on_an_empty_file_are_harmless() {
        // A spill partition that received no rows is a real case — the writer is opened and
        // then finished empty. Every hint must tolerate it.
        let (_dir, f) = temp_file_with(&[]);
        advise_sequential(&f);
        advise_random(&f);
        advise_dontneed(&f);
        advise_willneed(&f, 0, 0);
        // Past the end of an empty file: the kernel clamps rather than erroring.
        advise_willneed(&f, 4096, 4096);
        assert_eq!(f.metadata().unwrap().len(), 0);
    }

    #[test]
    fn preallocate_reserves_without_changing_the_visible_length() {
        // `posix_fallocate` reserves blocks *and* extends the file length — which is why a
        // spill writer must preallocate before writing, never after. This pins that the
        // reserved region reads back as zeros rather than as garbage.
        let dir = tempdir::Dir::new();
        let path = dir.path().join("run.arrow");
        let f = std::fs::File::create(&path).unwrap();
        preallocate(&f, 1 << 16);
        let len = f.metadata().unwrap().len();
        // On a filesystem without fallocate support the call is a no-op and the file stays
        // empty; either outcome is correct, but nothing in between is.
        assert!(len == 0 || len == 1 << 16, "unexpected length {len}");
        if len > 0 {
            let mut buf = Vec::new();
            std::fs::File::open(&path)
                .unwrap()
                .read_to_end(&mut buf)
                .unwrap();
            assert!(
                buf.iter().all(|b| *b == 0),
                "reserved space must read as zeros"
            );
        }
        // A zero-length reservation is the "unknown size" case and must do nothing at all.
        let f2 = std::fs::File::create(dir.path().join("empty.arrow")).unwrap();
        preallocate(&f2, 0);
        assert_eq!(f2.metadata().unwrap().len(), 0);
    }
}
