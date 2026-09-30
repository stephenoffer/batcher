//! A process-wide cache of remote object bytes, kept in fixed-size blocks: the warm path.
//!
//! **A repeated scan over an object store is bound by the network, and the bytes it moves
//! are the same every time.** TPC-H at SF1000 on eight 16-core nodes reads `lineitem` at
//! ~780 MB/s a node with the CPUs half idle (q6: 7.8 s), and every query of a sweep reads
//! most of the same column chunks again. A warehouse engine keeps those bytes near the
//! cores after the first read -- Databricks' disk cache, which its published TPC-H numbers
//! are measured with, is exactly this, on local SSD -- and this module is that cache, in
//! memory, for every remote Parquet read the native reader makes.
//!
//! **What is cached is the object's bytes, not a decode.** A block is `BLOCK` bytes of the
//! file at a `BLOCK`-aligned offset, so any read of any projection is served from whichever
//! blocks cover it: a query reading four columns warms them for another reading two, and a
//! column chunk costs its compressed size, a third to a fifth of its decoded one. A read
//! assembles its range from the cached blocks and fetches only the blocks it lacks,
//! coalesced into runs and split into concurrent GETs as any other remote read is
//! ([`crate::split_read`]).
//!
//! **Identity is `(uri, size)`,** the identity the footer cache already serves by: a warm
//! footer is only returned after a `HEAD` confirms the size, and every block read of a file
//! follows its footer read. An object rewritten in place at the same size is the case
//! neither cache can see, which is why this is off unless it is asked for.
//!
//! **Off by default.** `BATCHER_OBJECT_CACHE_BYTES` sets the budget in bytes *for this
//! process*. The cache lives in the process that reads, so a node running several reading
//! processes holds one per process; it is meant for the long-lived executors that read on a
//! node's behalf (the aligned and shuffle fleets run one actor per node), and the budget is
//! that process's share of the node, taken out of what execution may use.
//!
//! **The bytes returned are the bytes of the object.** A cached block is a verbatim slice of
//! a GET of the same object at the same offset, and a range is reassembled from blocks in
//! order, so a warm read is byte-identical to a cold one: this changes where bytes come
//! from, never which bytes they are.

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use bytes::{BufMut, Bytes, BytesMut};
use parquet::errors::{ParquetError, Result as ParquetResult};

/// Bytes per cached block.
///
/// Large enough that a block's bookkeeping is noise against its bytes and that a run of
/// missing blocks is a full-throughput GET on its own; small enough that the edges a read
/// shares with its neighbours waste little. TPC-H's column chunks are tens of MiB, so the
/// one partial block at each end of a chunk is a few percent of it.
pub(crate) const BLOCK: u64 = 4 << 20;

/// One cached block: `object` names the file and its size, `block` the aligned offset / BLOCK.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Key {
    object: Arc<str>,
    block: u64,
}

struct Entry {
    bytes: Bytes,
    /// Set on every hit and cleared as the clock hand passes: a block read since the hand
    /// last came round is spared once. This is CLOCK, the usual approximation of LRU that
    /// needs no reordering on a hit, so a hit takes the lock only to flip a flag.
    referenced: bool,
}

#[derive(Default)]
struct Inner {
    map: HashMap<Key, Entry>,
    /// The clock: every cached key once, oldest insertion first.
    order: VecDeque<Key>,
    used: u64,
}

/// The cache, sized once per process.
pub(crate) struct BlockCache {
    capacity: u64,
    inner: Mutex<Inner>,
    hit_bytes: AtomicU64,
    fetched_bytes: AtomicU64,
}

/// Counters of the process's cache, for diagnostics and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    /// Bytes of requested ranges served from cached blocks.
    pub hit_bytes: u64,
    /// Bytes fetched from the store to fill missing blocks.
    pub fetched_bytes: u64,
    /// Bytes the cache holds now.
    pub resident_bytes: u64,
}

/// The process's cache, or `None` when `BATCHER_OBJECT_CACHE_BYTES` is unset or zero.
pub(crate) fn global() -> Option<&'static BlockCache> {
    static CACHE: OnceLock<Option<BlockCache>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            std::env::var("BATCHER_OBJECT_CACHE_BYTES")
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .filter(|&n| n > 0)
                .map(BlockCache::new)
        })
        .as_ref()
}

/// The process cache's counters; all zero when the cache is off.
pub fn stats() -> CacheStats {
    global().map_or(
        CacheStats {
            hit_bytes: 0,
            fetched_bytes: 0,
            resident_bytes: 0,
        },
        BlockCache::stats,
    )
}

/// The identity a file's blocks are cached under: its URI and its size.
pub(crate) fn object_id(uri: &str, size: u64) -> Arc<str> {
    Arc::from(format!("{uri}#{size}"))
}

impl BlockCache {
    pub(crate) fn new(capacity: u64) -> Self {
        BlockCache {
            capacity,
            inner: Mutex::new(Inner::default()),
            hit_bytes: AtomicU64::new(0),
            fetched_bytes: AtomicU64::new(0),
        }
    }

    pub(crate) fn stats(&self) -> CacheStats {
        CacheStats {
            hit_bytes: self.hit_bytes.load(Ordering::Relaxed),
            fetched_bytes: self.fetched_bytes.load(Ordering::Relaxed),
            resident_bytes: self.inner.lock().map_or(0, |g| g.used),
        }
    }

    fn get(&self, object: &Arc<str>, block: u64) -> Option<Bytes> {
        let mut guard = self.inner.lock().ok()?;
        let entry = guard.map.get_mut(&Key {
            object: Arc::clone(object),
            block,
        })?;
        entry.referenced = true;
        Some(entry.bytes.clone())
    }

    fn put(&self, object: &Arc<str>, block: u64, bytes: Bytes) {
        let Ok(mut guard) = self.inner.lock() else {
            return;
        };
        let key = Key {
            object: Arc::clone(object),
            block,
        };
        if guard.map.contains_key(&key) {
            return; // a concurrent reader filled it first; the bytes are the same
        }
        guard.used += bytes.len() as u64;
        guard.order.push_back(key.clone());
        guard.map.insert(
            key,
            Entry {
                bytes,
                referenced: false,
            },
        );
        // Evict until back under budget. Each referenced block is spared once per pass of
        // the hand; bounded by two passes, since the first clears every flag it meets.
        let mut budget = 2 * guard.order.len();
        while guard.used > self.capacity && budget > 0 {
            budget -= 1;
            let Some(victim) = guard.order.pop_front() else {
                break;
            };
            let spare = match guard.map.get_mut(&victim) {
                Some(entry) if entry.referenced => {
                    entry.referenced = false;
                    true
                }
                _ => false,
            };
            if spare {
                guard.order.push_back(victim);
            } else if let Some(gone) = guard.map.remove(&victim) {
                guard.used -= gone.bytes.len() as u64;
            }
        }
    }

    /// Serve `ranges` of an object of `size` bytes from cached blocks, fetching the missing
    /// blocks with `fetch` (which must return one `Bytes` per range it is given, in order).
    ///
    /// Returns one `Bytes` per requested range, in the order asked for. A range inside one
    /// block is a zero-copy slice of it; one spanning blocks is copied together.
    pub(crate) async fn read<F, Fut>(
        &self,
        object: &Arc<str>,
        size: u64,
        ranges: &[Range<u64>],
        fetch: F,
    ) -> ParquetResult<Vec<Bytes>>
    where
        F: FnOnce(Vec<Range<u64>>) -> Fut,
        Fut: Future<Output = ParquetResult<Vec<Bytes>>>,
    {
        let mut needed: Vec<u64> = ranges
            .iter()
            .filter(|r| r.end > r.start)
            .flat_map(|r| r.start / BLOCK..r.end.div_ceil(BLOCK))
            .collect();
        needed.sort_unstable();
        needed.dedup();

        let mut have: HashMap<u64, Bytes> = HashMap::with_capacity(needed.len());
        let mut missing: Vec<u64> = Vec::new();
        for &b in &needed {
            match self.get(object, b) {
                Some(bytes) => {
                    have.insert(b, bytes);
                }
                None => missing.push(b),
            }
        }

        if !missing.is_empty() {
            let runs = runs_of(&missing);
            let byte_runs: Vec<Range<u64>> = runs
                .iter()
                .map(|r| r.start * BLOCK..(r.end * BLOCK).min(size))
                .collect();
            let fetched = fetch(byte_runs.clone()).await?;
            if fetched.len() != runs.len() {
                return Err(ParquetError::General(format!(
                    "block cache: asked for {} runs, got {}",
                    runs.len(),
                    fetched.len()
                )));
            }
            for ((blocks, span), bytes) in runs.iter().zip(&byte_runs).zip(fetched) {
                let want = span.end - span.start;
                if bytes.len() as u64 != want {
                    return Err(ParquetError::General(format!(
                        "block cache: a {want}-byte run came back as {} bytes",
                        bytes.len()
                    )));
                }
                self.fetched_bytes.fetch_add(want, Ordering::Relaxed);
                for b in blocks.clone() {
                    let lo = (b * BLOCK - span.start) as usize;
                    let hi = (((b + 1) * BLOCK).min(size) - span.start) as usize;
                    let block = bytes.slice(lo..hi);
                    self.put(object, b, block.clone());
                    have.insert(b, block);
                }
            }
        }

        let mut out = Vec::with_capacity(ranges.len());
        let mut warm = 0u64;
        for r in ranges {
            out.push(assemble(r, &have)?);
            warm += overlap_outside(r, &missing);
        }
        self.hit_bytes.fetch_add(warm, Ordering::Relaxed);
        Ok(out)
    }
}

/// Bytes of `range` that fall in blocks *not* listed in `missing` (sorted): the part served
/// warm.
fn overlap_outside(range: &Range<u64>, missing: &[u64]) -> u64 {
    if range.end <= range.start {
        return 0;
    }
    let mut total = range.end - range.start;
    for b in range.start / BLOCK..range.end.div_ceil(BLOCK) {
        if missing.binary_search(&b).is_ok() {
            let lo = range.start.max(b * BLOCK);
            let hi = range.end.min((b + 1) * BLOCK);
            total -= hi - lo;
        }
    }
    total
}

/// Contiguous runs of block indices, from a sorted, deduplicated list.
fn runs_of(blocks: &[u64]) -> Vec<Range<u64>> {
    let mut runs: Vec<Range<u64>> = Vec::new();
    for &b in blocks {
        match runs.last_mut() {
            Some(run) if run.end == b => run.end = b + 1,
            _ => runs.push(b..b + 1),
        }
    }
    runs
}

/// `range` cut out of the blocks in `have`, which must cover it.
fn assemble(range: &Range<u64>, have: &HashMap<u64, Bytes>) -> ParquetResult<Bytes> {
    if range.end <= range.start {
        return Ok(Bytes::new());
    }
    let first = range.start / BLOCK;
    let last = (range.end - 1) / BLOCK;
    let piece = |b: u64| -> ParquetResult<Bytes> {
        let block = have
            .get(&b)
            .ok_or_else(|| ParquetError::General(format!("block cache: block {b} missing")))?;
        let base = b * BLOCK;
        let lo = (range.start.max(base) - base) as usize;
        let hi = (range.end.min(base + BLOCK) - base) as usize;
        if hi > block.len() {
            return Err(ParquetError::General(format!(
                "block cache: range {range:?} runs past the object's end"
            )));
        }
        Ok(block.slice(lo..hi))
    };
    if first == last {
        return piece(first);
    }
    let mut buf = BytesMut::with_capacity((range.end - range.start) as usize);
    for b in first..=last {
        buf.put_slice(&piece(b)?);
    }
    Ok(buf.freeze())
}

#[cfg(test)]
// A read is a list of byte ranges, and one range is the common case these tests exercise.
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// An object whose byte at offset `i` is `i % 251`, so every slice is checkable.
    fn object(size: u64) -> Bytes {
        Bytes::from((0..size).map(|i| (i % 251) as u8).collect::<Vec<u8>>())
    }

    fn fetcher(
        data: Bytes,
        calls: Arc<AtomicUsize>,
        fetched: Arc<Mutex<Vec<Range<u64>>>>,
    ) -> impl FnOnce(Vec<Range<u64>>) -> futures::future::Ready<ParquetResult<Vec<Bytes>>> {
        move |runs| {
            calls.fetch_add(1, Ordering::SeqCst);
            fetched.lock().unwrap().extend(runs.iter().cloned());
            futures::future::ready(Ok(runs
                .into_iter()
                .map(|r| data.slice(r.start as usize..r.end as usize))
                .collect()))
        }
    }

    fn read(
        cache: &BlockCache,
        data: &Bytes,
        ranges: &[Range<u64>],
    ) -> (Vec<Bytes>, usize, Vec<Range<u64>>) {
        let id = object_id("s3://b/k", data.len() as u64);
        let calls = Arc::new(AtomicUsize::new(0));
        let fetched = Arc::new(Mutex::new(Vec::new()));
        let got = futures::executor::block_on(cache.read(
            &id,
            data.len() as u64,
            ranges,
            fetcher(data.clone(), calls.clone(), fetched.clone()),
        ))
        .unwrap();
        let runs = fetched.lock().unwrap().clone();
        (got, calls.load(Ordering::SeqCst), runs)
    }

    #[test]
    fn every_range_comes_back_as_the_objects_own_bytes() {
        let size = 3 * BLOCK + 12_345; // a short last block
        let data = object(size);
        let cache = BlockCache::new(1 << 40);
        let ranges = [
            0..10,
            BLOCK - 7..BLOCK + 9, // straddles a boundary
            BLOCK..2 * BLOCK,     // exactly one block
            5..3 * BLOCK + 100,   // spans every block into the short one
            size - 3..size,       // the object's tail
            42..42,               // empty
        ];
        let (got, calls, _) = read(&cache, &data, &ranges);
        assert_eq!(calls, 1, "every missing block is fetched in one call");
        for (r, b) in ranges.iter().zip(&got) {
            assert_eq!(b, &data.slice(r.start as usize..r.end as usize), "{r:?}");
        }
    }

    #[test]
    fn a_warm_read_fetches_nothing_and_a_partly_warm_one_only_what_it_lacks() {
        let size = 5 * BLOCK;
        let data = object(size);
        let cache = BlockCache::new(1 << 40);
        read(&cache, &data, &[BLOCK..2 * BLOCK + 1]); // warms blocks 1 and 2
        let (got, calls, _) = read(&cache, &data, &[BLOCK + 3..2 * BLOCK]);
        assert_eq!(calls, 0);
        assert_eq!(cache.stats().hit_bytes, BLOCK - 3);
        assert_eq!(
            got[0],
            data.slice((BLOCK + 3) as usize..(2 * BLOCK) as usize)
        );

        let (got, calls, runs) = read(&cache, &data, &[0..5 * BLOCK]);
        assert_eq!(calls, 1);
        // Blocks 0, 3 and 4 were missing: two runs, and the warm middle is not re-fetched.
        assert_eq!(runs, vec![0..BLOCK, 3 * BLOCK..5 * BLOCK]);
        assert_eq!(got[0], data);
        assert_eq!(cache.stats().fetched_bytes, 5 * BLOCK);
    }

    #[test]
    fn the_cache_stays_within_its_budget() {
        let size = 10 * BLOCK;
        let data = object(size);
        let cache = BlockCache::new(3 * BLOCK);
        let (got, _, _) = read(&cache, &data, &[0..size]);
        assert_eq!(
            got[0], data,
            "a read larger than the cache is still answered whole"
        );
        assert!(cache.stats().resident_bytes <= 3 * BLOCK);
    }

    #[test]
    fn a_block_read_again_outlives_one_that_was_not() {
        let data = object(4 * BLOCK);
        let cache = BlockCache::new(3 * BLOCK);
        read(&cache, &data, &[0..3 * BLOCK]); // blocks 0, 1, 2
        read(&cache, &data, &[0..1]); // block 0 is hit, so it is spared once
        read(&cache, &data, &[3 * BLOCK..3 * BLOCK + 1]); // block 3 must evict one
        let (_, calls, _) = read(&cache, &data, &[0..1]);
        assert_eq!(calls, 0, "the block read again was kept");
        let (_, calls, _) = read(&cache, &data, &[BLOCK..BLOCK + 1]);
        assert_eq!(
            calls, 1,
            "the oldest block nobody re-read was the one evicted"
        );
    }

    #[test]
    fn a_short_answer_from_the_store_is_an_error_not_a_wrong_read() {
        let data = object(2 * BLOCK);
        let cache = BlockCache::new(1 << 40);
        let id = object_id("s3://b/k", data.len() as u64);
        let short = |runs: Vec<Range<u64>>| {
            futures::future::ready(Ok(runs
                .into_iter()
                .map(|r| data.slice(r.start as usize..(r.end - 1) as usize))
                .collect()))
        };
        let err = futures::executor::block_on(cache.read(&id, 2 * BLOCK, &[0..10], short));
        assert!(err.is_err());
        assert_eq!(cache.stats().resident_bytes, 0);
    }

    #[test]
    fn runs_group_contiguous_blocks() {
        assert_eq!(runs_of(&[0, 1, 2, 5, 7, 8]), vec![0..3, 5..6, 7..9]);
        assert!(runs_of(&[]).is_empty());
    }
}
