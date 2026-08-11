//! Split an oversized object-store read into several concurrent range GETs.
//!
//! **One GET gets one connection's bandwidth, however large the range is.** That is the whole
//! of it, and it is why a Parquet reader that looks optimal can still run at a tenth of the
//! link. Measured against `s3://ray-benchmark-data` from a 96-core node: a single 134 MiB
//! `GET` returns at **94 MB/s**, while the *same bytes* requested as sixteen concurrent
//! 8.4 MiB `GET`s return at **643 MB/s** — 6.8x, for identical bytes over an identical link.
//!
//! Nothing above this module was asking for one big request; `object_store` produces them on
//! its own. [`ObjectStore::get_ranges`] merges ranges separated by less than
//! `OBJECT_STORE_COALESCE_DEFAULT` (1 MiB) into one, and that merge is **unbounded in size**.
//! A Parquet row group stores its column chunks contiguously, so a 16-column projection of a
//! 134 MiB row group is 16 adjacent ranges — which coalesce into exactly one 134 MiB request.
//! The reader then had 8 requests in flight for a 1 GiB file (one per row group) and sat at
//! ~420 MB/s while DuckDB, which splits, reached ~1.9 GB/s on the same file.
//!
//! So this wrapper does the opposite of coalescing, *after* it: a range wider than
//! [`SPLIT_THRESHOLD`] is cut into pieces and the pieces are fetched concurrently. The pieces
//! are issued as separate [`ObjectStore::get_range`] calls rather than one `get_ranges`,
//! because `get_ranges` would merge these perfectly-adjacent pieces straight back into the
//! single request they were cut from.
//!
//! **Bytes are unchanged.** Every piece is a contiguous slice of the requested range and they
//! are reassembled in order, so this is a scheduling change: the same bytes, in the same
//! order, over more connections. It applies only to remote stores ([`store::Resolved::remote`])
//! — a local read is served from the page cache, where splitting buys nothing and costs
//! syscalls.

use std::ops::Range;
use std::sync::{Arc, OnceLock};

use bytes::{BufMut, Bytes, BytesMut};
use futures::future::BoxFuture;
use futures::{FutureExt, StreamExt, TryStreamExt};
use object_store::path::Path;
use object_store::ObjectStore;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::async_reader::{AsyncFileReader, ParquetObjectReader};
use parquet::errors::{ParquetError, Result as ParquetResult};
use parquet::file::metadata::ParquetMetaData;

/// Byte span above which one read is cut into concurrent pieces.
///
/// Sized to be comfortably past the point where a single connection's ramp-up is amortized
/// (so a split piece is a full-throughput request in its own right) while still small enough
/// that a row-group-sized read becomes many pieces rather than two. Below this a read is
/// issued unchanged, so the small-file and metadata paths are untouched.
const SPLIT_THRESHOLD: u64 = 8 * 1024 * 1024;

/// Target size of each piece an oversized read is cut into.
const PIECE_BYTES: u64 = 8 * 1024 * 1024;

/// Concurrent range GETs in flight for one reader call.
///
/// This bounds *requests*, not memory-per-request: the pieces of a read are all in flight
/// together and all retained, so the peak is the read's own size either way. Env-overridable
/// for a link that rate-limits by request count rather than bandwidth.
fn fetch_concurrency() -> usize {
    static C: OnceLock<usize> = OnceLock::new();
    *C.get_or_init(|| {
        std::env::var("BATCHER_OBJECT_STORE_SPLIT_CONCURRENCY")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(16)
    })
}

/// Process-wide ceiling on split range GETs in flight, across every file and row group.
///
/// The per-call budget above is **nested inside two other fan-outs** — `read_parquet_many`
/// runs up to `BATCHER_PARQUET_FILE_CONCURRENCY` files at once and each file runs up to
/// `BATCHER_PARQUET_RG_CONCURRENCY` row groups at once — so multiplying them out is the
/// number of sockets a wide scan would ask for, and it reaches five figures without this.
/// Past the point where the link is saturated more requests buy nothing and start losing to
/// the store's own rate limiting (S3 answers a slow-down with `503`), so the ceiling is a
/// throughput guard, not only a politeness one.
///
/// It bounds *only the split pieces*. An unsplit read never takes a permit, so the
/// small-file and metadata paths cannot be made to wait behind a large scan.
///
/// **Set as a backstop, not as a tuner.** A ceiling low enough to bind is a throttle wearing
/// a safety label: at 256 the 16-column read of a 1 GiB file measured 814 ms against 661 ms
/// with room to spare, because one file's eight row groups alone want 136 requests. 1,024 is
/// an order of magnitude below what the nesting can ask for and above what any single read
/// here reaches, which is the shape a guard should have.
fn global_permits() -> &'static tokio::sync::Semaphore {
    static S: OnceLock<tokio::sync::Semaphore> = OnceLock::new();
    S.get_or_init(|| {
        let n = std::env::var("BATCHER_OBJECT_STORE_SPLIT_MAX_INFLIGHT")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(1024);
        tokio::sync::Semaphore::new(n)
    })
}

/// Cut `range` into [`PIECE_BYTES`]-sized contiguous pieces, or return it whole when it is
/// not worth splitting.
///
/// The pieces tile the range exactly — no gap, no overlap, first starts at `range.start`,
/// last ends at `range.end` — which is what makes reassembly a plain concatenation.
fn pieces_of(range: &Range<u64>) -> Vec<Range<u64>> {
    let len = range.end.saturating_sub(range.start);
    if len <= SPLIT_THRESHOLD {
        return vec![range.clone()];
    }
    let n = len.div_ceil(PIECE_BYTES);
    let size = len.div_ceil(n);
    let mut out = Vec::with_capacity(n as usize);
    let mut at = range.start;
    while at < range.end {
        let end = (at + size).min(range.end);
        out.push(at..end);
        at = end;
    }
    out
}

/// Fetch every piece of `plan` concurrently, in one bounded fan-out.
///
/// `plan` is the flattened piece list for *all* the caller's ranges, so the concurrency
/// budget is spent across the whole request rather than per range — a projection of sixteen
/// column chunks issues sixteen chunks' worth of pieces together instead of serializing the
/// chunks and splitting only within each.
async fn fetch_pieces(
    store: &Arc<dyn ObjectStore>,
    path: &Path,
    plan: Vec<Range<u64>>,
) -> ParquetResult<Vec<Bytes>> {
    futures::stream::iter(plan.into_iter().map(|piece| {
        let store = Arc::clone(store);
        let path = path.clone();
        async move {
            let _permit = global_permits()
                .acquire()
                .await
                .map_err(|e| ParquetError::External(Box::new(e)))?;
            store
                .get_range(&path, piece)
                .await
                .map_err(|e| ParquetError::External(Box::new(e)))
        }
    }))
    .buffered(fetch_concurrency())
    .try_collect()
    .await
}

/// Join `parts` into one buffer, or hand back the single part untouched.
fn join(parts: Vec<Bytes>) -> Bytes {
    if parts.len() == 1 {
        return parts.into_iter().next().expect("len 1");
    }
    let total: usize = parts.iter().map(Bytes::len).sum();
    let mut buf = BytesMut::with_capacity(total);
    for part in parts {
        buf.put_slice(&part);
    }
    buf.freeze()
}

/// A Parquet reader that either splits its large reads or does not.
///
/// One concrete type because `ParquetRecordBatchStreamBuilder` takes a reader by value: the
/// local and remote paths have to be the same type at the call site, and an enum keeps the
/// choice explicit (and the dispatch a branch) where a `Box<dyn AsyncFileReader>` would hide
/// it behind a vtable on every fetch.
pub(crate) enum MaybeSplitReader {
    /// Local: the page cache has no per-request bandwidth limit, so read as-is.
    Plain(ParquetObjectReader),
    /// Remote: cut oversized reads into concurrent range GETs.
    Split {
        inner: ParquetObjectReader,
        store: Arc<dyn ObjectStore>,
        path: Path,
    },
}

/// Wrap `inner` in the splitting reader when reads go over the network, else hand it back.
pub(crate) fn maybe_split(
    inner: ParquetObjectReader,
    store: &Arc<dyn ObjectStore>,
    path: &Path,
    remote: bool,
) -> MaybeSplitReader {
    if remote {
        MaybeSplitReader::Split {
            inner,
            store: Arc::clone(store),
            path: path.clone(),
        }
    } else {
        MaybeSplitReader::Plain(inner)
    }
}

impl AsyncFileReader for MaybeSplitReader {
    fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, ParquetResult<Bytes>> {
        let (inner, store, path) = match self {
            MaybeSplitReader::Plain(inner) => return inner.get_bytes(range),
            MaybeSplitReader::Split { inner, store, path } => (inner, store, path),
        };
        let plan = pieces_of(&range);
        if plan.len() == 1 {
            return inner.get_bytes(range);
        }
        let store = Arc::clone(store);
        let path = path.clone();
        async move { Ok(join(fetch_pieces(&store, &path, plan).await?)) }.boxed()
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, ParquetResult<Vec<Bytes>>> {
        let (inner, store, path) = match self {
            MaybeSplitReader::Plain(inner) => return inner.get_byte_ranges(ranges),
            MaybeSplitReader::Split { inner, store, path } => (inner, store, path),
        };
        // Piece counts per range, so the flat result folds back into one `Bytes` per requested
        // range, in the order the caller asked for them.
        let per_range: Vec<Vec<Range<u64>>> = ranges.iter().map(pieces_of).collect();
        if per_range.iter().all(|p| p.len() == 1) {
            // Nothing oversized: let the inner reader coalesce as it always has. Splitting is
            // for reads too big for one connection; merging adjacent small chunks into one
            // request is the right move for the rest, and is what saves a small-file read a
            // request per column.
            return inner.get_byte_ranges(ranges);
        }
        let counts: Vec<usize> = per_range.iter().map(Vec::len).collect();
        let plan: Vec<Range<u64>> = per_range.into_iter().flatten().collect();
        let store = Arc::clone(store);
        let path = path.clone();
        async move {
            let mut parts = fetch_pieces(&store, &path, plan).await?.into_iter();
            let mut out = Vec::with_capacity(counts.len());
            for n in counts {
                out.push(join(parts.by_ref().take(n).collect()));
            }
            Ok(out)
        }
        .boxed()
    }

    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, ParquetResult<Arc<ParquetMetaData>>> {
        // Footers are small, already one round trip, and cached a layer up
        // (`load_metadata_cached`): nothing to split.
        match self {
            MaybeSplitReader::Plain(inner) => inner.get_metadata(options),
            MaybeSplitReader::Split { inner, .. } => inner.get_metadata(options),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_small_range_is_not_split() {
        assert_eq!(pieces_of(&(0..1024)), vec![0..1024]);
        assert_eq!(
            pieces_of(&(0..SPLIT_THRESHOLD)),
            vec![0..SPLIT_THRESHOLD],
            "exactly at the threshold is still one request"
        );
    }

    /// The pieces must tile the range exactly: concatenating them is the reassembly, so a gap
    /// would drop bytes and an overlap would duplicate them — either one a wrong read rather
    /// than a slow one.
    #[test]
    fn pieces_tile_the_range_exactly() {
        for range in [
            0u64..134 * 1024 * 1024,
            17..(17 + 40 * 1024 * 1024),
            0..(SPLIT_THRESHOLD + 1),
        ] {
            let pieces = pieces_of(&range);
            assert!(pieces.len() > 1, "{range:?} should split");
            assert_eq!(pieces[0].start, range.start);
            assert_eq!(pieces.last().unwrap().end, range.end);
            for pair in pieces.windows(2) {
                assert_eq!(pair[0].end, pair[1].start, "no gap and no overlap");
            }
            let covered: u64 = pieces.iter().map(|p| p.end - p.start).sum();
            assert_eq!(covered, range.end - range.start);
            assert!(pieces.iter().all(|p| p.end > p.start), "no empty piece");
        }
    }

    /// A 134 MiB row-group read — the shape that motivated the module — must become enough
    /// pieces to actually use several connections.
    #[test]
    fn a_row_group_sized_read_splits_widely() {
        let pieces = pieces_of(&(0..134 * 1024 * 1024));
        assert_eq!(pieces.len(), 17);
        assert!(pieces.iter().all(|p| p.end - p.start <= PIECE_BYTES));
    }

    #[test]
    fn joining_one_part_returns_it_unchanged() {
        let only = Bytes::from_static(b"abc");
        assert_eq!(join(vec![only.clone()]), only);
    }

    #[test]
    fn joining_parts_concatenates_in_order() {
        let parts = vec![
            Bytes::from_static(b"ab"),
            Bytes::from_static(b"cd"),
            Bytes::from_static(b"e"),
        ];
        assert_eq!(join(parts), Bytes::from_static(b"abcde"));
    }
}
