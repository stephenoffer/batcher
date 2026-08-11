//! Native Rust format readers (Parquet over object storage; Avro OCF to Arrow).
//!
//! The distributed scan's dominant cost is object-store read throughput. Reading a
//! row-group split through PyArrow issues a chain of latency-bound column-chunk GETs;
//! this crate decodes Parquet natively with the `parquet` crate's async reader, which
//! fetches the projected column chunks of the requested row-groups **concurrently**
//! straight from storage (`object_store`) and streams Arrow `RecordBatch`es out. One
//! reader serves every backend (S3 / GCS / Azure / HTTP / local) via [`store`].
//!
//! It is a pure-Rust leaf crate (depends only on `arrow`): `cargo test`-able with no
//! Python, and exposed to the control plane through `bc-py`. The Python IO layer calls
//! it per row-group split, falling back to PyArrow if a scheme/feature is unsupported,
//! so the result is byte-identical either way.

use std::sync::{Arc, OnceLock};

use arrow::record_batch::RecordBatch;
use futures::{StreamExt, TryStreamExt};
use object_store::ObjectStore;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use parquet::arrow::async_reader::ParquetObjectReader;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::file::metadata::ParquetMetaData;

mod avro;
mod bloom;
mod footer_stats;
mod page_index;
mod predicate;
mod projection;
mod row_filter;
mod split_read;
mod store;

/// Below this many candidate rows a read is short enough that the row-filter probe would cost
/// a larger share of it than the filter could save, so neither runs.
const ROW_FILTER_MIN_ROWS: usize = 200_000;

/// How many rows the row-filter selectivity probe reads before deciding. Enough for a stable
/// selected-fraction, small enough that a decision *not* to filter costs ~1 ms.
const ROW_FILTER_PROBE_ROWS: usize = 8_192;

/// Whether row-level filter pushdown is enabled (`BATCHER_PARQUET_ROW_FILTER=0` disables).
///
/// An escape hatch in the shape the rest of this module already uses, and the A/B switch the
/// feature was measured with: comparing two *builds* on a shared machine could not separate
/// the effect from the noise, whereas one binary run both ways can.
fn row_filter_enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("BATCHER_PARQUET_ROW_FILTER").as_deref() != Ok("0"))
}

pub use avro::read_avro_bytes;
pub use footer_stats::{
    parquet_file_manifest, parquet_footer_stats, ColumnFooterStats, FooterStats,
};

/// How many row-groups to fetch+decode concurrently. The single-stream reader processes
/// row-groups one at a time, so a worker reading a many-row-group file waited on each
/// row-group's GETs in series — far below the object store's achievable throughput.
/// Reading this many row-groups at once overlaps their I/O (it plateaus once the network
/// is saturated). Env-overridable for wider rows / tighter RAM.
fn rg_concurrency() -> usize {
    static C: OnceLock<usize> = OnceLock::new();
    *C.get_or_init(|| {
        std::env::var("BATCHER_PARQUET_RG_CONCURRENCY")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(16)
    })
}

/// How many trailing bytes to speculatively read when fetching a file's footer.
///
/// The metadata reader's default prefetch is the 8-byte footer *length*, which makes every
/// cold footer load two round trips: read the length, then come back for the metadata it
/// points at. One speculative read of this size collapses that to one whenever the footer
/// fits, which it does for ordinary schemas — a 16-column file's footer plus its page index
/// is a few KB. Overshooting costs only the over-read bytes; undershooting degrades to the
/// two-trip behavior.
///
/// **Only safe because the file size is known.** This hint is a size, not a range, and it
/// becomes a bounded read only against a known end. Read as a bare *suffix*, the page-index
/// offsets are computed against a buffer whose start was guessed, and
/// `ParquetMetaDataReader::load_page_index_with_remainder` asserts on the mismatch
/// (`assert!(end <= remainder.len())`) rather than re-fetching — so the process aborts. Five
/// `bc-io` tests panicked inside the parquet crate that way, and a *smaller* hint failed ten.
/// `load_metadata_cached` gets the size from the same suffix `GET` that fetches these bytes,
/// which is why the guarantee now costs no round trip of its own.
fn footer_prefetch() -> usize {
    static P: OnceLock<usize> = OnceLock::new();
    *P.get_or_init(|| {
        std::env::var("BATCHER_PARQUET_FOOTER_PREFETCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(64 * 1024)
    })
}

/// Errors reading Parquet from object storage. Each variant is actionable and string-
/// backed so it can cross the FFI boundary as a plain message (the Python side falls
/// back to PyArrow on any error, so these never abort a query).
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    #[error("invalid URI: {0}")]
    Uri(String),
    #[error("object store error: {0}")]
    Store(String),
    #[error("parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("object store io: {0}")]
    ObjectStore(#[from] object_store::Error),
    #[error("avro error: {0}")]
    Avro(#[from] arrow::error::ArrowError),
    /// An Arrow-side failure that is not an Avro decode — building the footer-statistics
    /// batch, concatenating bounds. String-backed rather than `#[from]`, since
    /// `ArrowError` is already claimed by [`IoError::Avro`].
    #[error("arrow error: {0}")]
    Arrow(String),
}

/// One shared multi-threaded Tokio runtime for all reads in the process. The async
/// parquet reader needs an executor; sharing one runtime lets concurrent split reads
/// (one per worker thread) overlap their object-store I/O on a common thread pool
/// instead of each spinning up its own.
///
/// Sized to [`bc_arrow::usable_cores`] rather than tokio's default. The default is
/// `available_parallelism`, which honors the CPU *affinity mask* but not the cgroup CFS
/// *bandwidth* quota — and Kubernetes' `cpu` limit is the latter. A pod limited to 15 cores
/// on a 16-core node therefore got 16 decode workers, and exceeding the quota does not merely
/// waste a thread: it gets the whole cgroup throttled for the rest of the CFS period, so the
/// extra worker buys stalls for every other thread in the process. Parquet decode is
/// CPU-bound, so this pool is exactly where that shows up.
fn runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(bc_arrow::usable_cores())
            .thread_name("bc-io")
            .enable_all()
            .build()
            .expect("build tokio runtime")
    })
}

/// Read selected row-groups of one Parquet object into Arrow batches.
///
/// `row_groups` empty = all row-groups; `columns` `None` = all columns (else a
/// leaf-column projection by name, pushed into the decode so only those chunks are
/// fetched). `batch_size` is the output `RecordBatch` row count. Blocks on the shared
/// runtime, so it is callable from an ordinary (sync) worker thread.
pub fn read_parquet(
    uri: &str,
    row_groups: &[usize],
    columns: Option<&[String]>,
    batch_size: usize,
) -> Result<Vec<RecordBatch>, IoError> {
    runtime().block_on(read_parquet_async(
        uri, row_groups, columns, batch_size, None,
    ))
}

/// Read one Parquet object with a pushed predicate applied as row-group pruning.
///
/// Identical to [`read_parquet`] except `predicate` (the compact JSON `to_native_predicate`
/// emits; see [`predicate`]) is used to skip work before and during decode: row-group pruning
/// by footer statistics, page pruning by the column index, bloom pruning, and — when the
/// predicate's types allow it and a probe measures it worth doing — a [`row_filter`] applied
/// *during* the decode.
///
/// The first three are superset-safe: they only skip provably-empty blocks. The row filter is
/// not — it removes individual rows — which is sound because `to_native_predicate` is
/// all-or-nothing, so a predicate that arrives here is a *complete* translation of the
/// `Filter` above the scan rather than a weakening of it. The result is therefore anywhere
/// between the exact matching rows and every requested row-group, and the engine keeps its
/// `Filter` either way (`core/scan_only.py` refuses its shortcut whenever a predicate was
/// pushed). An unparseable or non-pushable predicate simply reads every requested row-group.
pub fn read_parquet_filtered(
    uri: &str,
    row_groups: &[usize],
    columns: Option<&[String]>,
    batch_size: usize,
    predicate: &str,
) -> Result<Vec<RecordBatch>, IoError> {
    runtime().block_on(read_parquet_async(
        uri,
        row_groups,
        columns,
        batch_size,
        Some(predicate),
    ))
}

/// How many whole files to read concurrently in a batched multi-file read. A
/// many-small-files scan is latency-bound on per-file footer+chunk GETs, so overlapping
/// files (on top of each file's own row-group concurrency) is the throughput lever.
///
/// **This is a latency budget, not a CPU one, and a flat 64 was leaving most of it unspent.**
/// Each file costs about two sequential round trips (footer, then its column chunks) and
/// almost no CPU, so the useful concurrency is set by how many requests are needed to cover
/// the round-trip time — far more than the core count. Measured reading the 1,024-file
/// `small-parquet/1GiB` corpus from S3 on a 96-core node, one column: **803 ms at 64, 411 ms
/// at 256, 167 ms at 512** — 4.8x for a number, on the layout the scan benchmark measures as
/// Batcher's largest gap.
///
/// Scaled by the core count rather than pinned at the measured best, because the figure that
/// is right here is a property of this link and this host size, and a small pod raising 64 to
/// 512 would spend memory and sockets it does not have on requests its bandwidth cannot
/// carry. The floor keeps every host at least as concurrent as it was.
fn file_concurrency() -> usize {
    static C: OnceLock<usize> = OnceLock::new();
    *C.get_or_init(|| {
        std::env::var("BATCHER_PARQUET_FILE_CONCURRENCY")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| bc_arrow::usable_cores().saturating_mul(4).clamp(64, 512))
    })
}

/// Read many whole Parquet objects in ONE runtime pass, returning per-file batches in URI
/// order. This is the many-small-files throughput path: calling [`read_parquet`] once per
/// file pays a `block_on` + GIL round trip and a fresh future per file, serializing at the
/// FFI boundary; here all files' footer + column-chunk GETs overlap under one global
/// concurrency budget (`BATCHER_PARQUET_FILE_CONCURRENCY`), decoded on the shared runtime.
/// Each file reads all row-groups with the projection pushed into the decode.
pub fn read_parquet_many(
    uris: &[String],
    columns: Option<&[String]>,
    batch_size: usize,
) -> Result<Vec<Vec<RecordBatch>>, IoError> {
    runtime().block_on(async {
        // Each file is `tokio::spawn`ed onto the runtime's worker pool — NOT merely
        // `buffered`, which polls futures cooperatively on the calling thread and would
        // serialize the CPU-bound Parquet decode (only overlapping the I/O). A shared
        // semaphore caps how many run at once; results are joined back in URI order.
        let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(file_concurrency()));
        let handles: Vec<_> = uris
            .iter()
            .map(|uri| {
                let uri = uri.clone();
                let cols = columns.map(<[String]>::to_vec);
                let sem = sem.clone();
                tokio::spawn(async move {
                    let _permit = sem.acquire_owned().await;
                    read_parquet_async(&uri, &[], cols.as_deref(), batch_size, None).await
                })
            })
            .collect();
        let mut out = Vec::with_capacity(handles.len());
        for h in handles {
            out.push(h.await.map_err(|e| IoError::Store(e.to_string()))??);
        }
        Ok(out)
    })
}

/// Process-wide cache of parsed Parquet footers, keyed by URI: `(file_size, metadata)`.
///
/// This is the "never read the same metadata twice" guarantee: multiple splits of one
/// file, and repeated queries over warm (session-fleet) workers, fetch + parse the footer
/// ONCE instead of per read. `ArrowReaderMetadata` is `Arc`-backed, so a hit is a cheap
/// clone.
///
/// It used to be justified by "Parquet files are write-once, so the footer is immutable".
/// That holds for an immutable lake and not for a pipeline re-run, which overwrites its
/// own output under the same deterministic name. The stored size is therefore *checked*
/// rather than merely recorded: a hit whose file has changed size is treated as a miss.
///
/// The failure this prevents is not a stale-looking answer — it is reading the new bytes
/// with the **old row-group offsets**, which surfaces as a corrupt-file error
/// (`Column cannot have more than one dictionary`) on a perfectly valid file.
fn meta_cache(
) -> &'static std::sync::Mutex<std::collections::HashMap<String, (u64, ArrowReaderMetadata)>> {
    static C: OnceLock<
        std::sync::Mutex<std::collections::HashMap<String, (u64, ArrowReaderMetadata)>>,
    > = OnceLock::new();
    C.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Load many files' Parquet footers in ONE runtime pass, best-effort, in URI order.
///
/// The metadata counterpart to [`read_parquet_many`]: footer loads are latency-bound
/// round trips, so they overlap under the same file-concurrency budget instead of running
/// one at a time. Every load goes through [`load_metadata_cached`], so a file the reader
/// has already touched this process costs one HEAD rather than a fetch and parse.
///
/// A file whose footer cannot be read maps to `None` rather than failing the batch — a
/// statistics pass over a directory must survive one unreadable object, and the caller
/// counts how many it actually got so it can decline to report an exact row count.
pub(crate) fn load_metadata_many(
    uris: &[String],
) -> Result<Vec<Option<ArrowReaderMetadata>>, IoError> {
    runtime().block_on(async {
        let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(file_concurrency()));
        let handles: Vec<_> = uris
            .iter()
            .map(|uri| {
                let uri = uri.clone();
                let sem = sem.clone();
                tokio::spawn(async move {
                    let _permit = sem.acquire_owned().await;
                    let resolved = store::resolve(&uri)?;
                    load_metadata_cached(&uri, &resolved).await.map(|(_, m)| m)
                })
            })
            .collect();
        let mut out = Vec::with_capacity(handles.len());
        for h in handles {
            // A panicked task is the only fatal case; a failed *read* is just a skipped file.
            match h.await {
                Ok(Ok(md)) => out.push(Some(md)),
                Ok(Err(_)) => out.push(None),
                Err(e) => return Err(IoError::Store(e.to_string())),
            }
        }
        Ok(out)
    })
}

/// A parquet reader whose first request has already happened.
///
/// The footer load needs two facts about a file: how big it is, and the bytes at its end.
/// Asking for them separately costs two round trips per file — a `head()` for the size, then
/// a ranged `GET` for the footer — and on an object store that is the dominant cost of
/// opening a small file. A **suffix** `GET` answers both at once: `ObjectMeta.size` comes
/// back with the response, so one request yields the size *and* the tail.
///
/// This serves any range that falls inside that prefetched tail from memory and forwards
/// anything else to the store. In practice the footer, and usually the page index with it,
/// are entirely inside it — so opening a file is one request, against two before and three
/// with no size hint at all (an 8-byte length read, then the footer).
///
/// It matters most exactly where it looks smallest. The benchmark's `many_small` layout is
/// ~1.2 MiB files, so a 1 GiB read opens ~850 of them and every saved round trip is paid 850
/// times; that suite measured 10-12x DuckDB before this.
struct PrefetchedFooter {
    store: Arc<dyn ObjectStore>,
    path: object_store::path::Path,
    /// Where `tail` starts within the file.
    tail_start: u64,
    tail: bytes::Bytes,
}

impl parquet::arrow::async_reader::AsyncFileReader for PrefetchedFooter {
    fn get_bytes(
        &mut self,
        range: std::ops::Range<u64>,
    ) -> futures::future::BoxFuture<'_, parquet::errors::Result<bytes::Bytes>> {
        use futures::FutureExt;
        if range.start >= self.tail_start && range.end <= self.tail_start + self.tail.len() as u64 {
            let from = (range.start - self.tail_start) as usize;
            let to = (range.end - self.tail_start) as usize;
            let slice = self.tail.slice(from..to);
            return async move { Ok(slice) }.boxed();
        }
        let store = self.store.clone();
        let path = self.path.clone();
        async move {
            store
                .get_range(&path, range)
                .await
                .map_err(|e| parquet::errors::ParquetError::External(Box::new(e)))
        }
        .boxed()
    }

    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> futures::future::BoxFuture<'a, parquet::errors::Result<Arc<ParquetMetaData>>> {
        use futures::FutureExt;
        // The file size is known exactly — the suffix `GET` that filled `tail` reported it —
        // so the metadata reader gets a *bounded* read and never has to guess where the file
        // ends. That is the property `footer_prefetch` documents as load-bearing: without it
        // the page-index offsets are computed against a buffer whose start was guessed, and
        // the parquet crate asserts rather than re-fetching.
        let file_size = self.tail_start + self.tail.len() as u64;
        let prefetch = self.tail.len();
        let page_index = options.map(|o| o.page_index()).unwrap_or(false);
        async move {
            let reader = parquet::file::metadata::ParquetMetaDataReader::new()
                // `Optional`, never `Required`: a file written without a page index is
                // ordinary, and `bool::into` would map this to `Required` and fail the read
                // outright rather than decoding every page as it did before.
                .with_page_index_policy(if page_index {
                    parquet::file::metadata::PageIndexPolicy::Optional
                } else {
                    parquet::file::metadata::PageIndexPolicy::Skip
                })
                .with_prefetch_hint(Some(prefetch));
            Ok(Arc::new(
                reader.load_and_finish(&mut *self, file_size).await?,
            ))
        }
        .boxed()
    }
}

async fn load_metadata_cached(
    uri: &str,
    resolved: &store::Resolved,
) -> Result<(u64, ArrowReaderMetadata), IoError> {
    // Warm: one HEAD confirms the file is the one the entry describes, and the expensive
    // half — the ranged footer GET and the parse — is served from memory. Serving an
    // unvalidated hit costs correctness instead: the failure is not a stale-looking answer
    // but reading *new* bytes at the *old* row-group offsets.
    let cached = meta_cache().lock().unwrap().get(uri).cloned();
    if let Some((size, amd)) = cached {
        if resolved.store.head(&resolved.path).await?.size == size {
            return Ok((size, amd));
        }
    }
    // Cold (or changed): **one** request. A suffix `GET` returns the trailing bytes *and*
    // `ObjectMeta.size`, which is the whole trick — the two things a footer load needs, and
    // the two things this used to spend two round trips acquiring.
    //
    // Both of the obvious one-trip shapes are wrong, and the file has worn each of them:
    //
    // - `head()` then a ranged `GET` is correct but costs two trips. On the benchmark's
    //   `many_small` layout (~1.2 MiB objects) a 1 GiB read opens ~850 files, so the extra
    //   trip is paid 850 times and dominates the read.
    // - Dropping the `head()` and letting `ParquetObjectReader` read a *suffix* is one trip
    //   but unsound: without a known file size it slices the page index out of a buffer
    //   whose start it guessed, and `load_page_index_with_remainder` asserts on the mismatch
    //   rather than re-fetching. Five `bc-io` tests aborted inside that assert.
    //
    // `PrefetchedFooter` is the shape that is both: the size comes back with the bytes, so
    // every offset is computed against a known end, and the metadata reader's fetches land
    // in the tail already in memory. A footer larger than the hint still works — the reader
    // asks for the range it needs and gets the one extra request it would have made anyway.
    //
    // The ColumnIndex/OffsetIndex load alongside the footer, since they live in the region
    // the suffix already covers and are cached with it: a read that never prunes pays a
    // parse of a few KB, one that does skips whole pages instead of decoding them.
    let want = footer_prefetch() as u64;
    let get = resolved
        .store
        .get_opts(
            &resolved.path,
            object_store::GetOptions {
                range: Some(object_store::GetRange::Suffix(want)),
                ..Default::default()
            },
        )
        .await?;
    let size = get.meta.size;
    let tail = get.bytes().await?;
    let mut probe = PrefetchedFooter {
        store: resolved.store.clone(),
        path: resolved.path.clone(),
        tail_start: size.saturating_sub(tail.len() as u64),
        tail,
    };
    // The page index sits just below the footer, so it is normally inside the tail above and
    // costs nothing extra; when it is not, `PrefetchedFooter` fetches it and the file simply
    // takes the second request it would have taken anyway.
    let amd = ArrowReaderMetadata::load_async(
        &mut probe,
        ArrowReaderOptions::new().with_page_index(true),
    )
    .await?;
    meta_cache()
        .lock()
        .unwrap()
        .insert(uri.to_string(), (size, amd.clone()));
    Ok((size, amd))
}

async fn read_parquet_async(
    uri: &str,
    row_groups: &[usize],
    columns: Option<&[String]>,
    batch_size: usize,
    predicate: Option<&str>,
) -> Result<Vec<RecordBatch>, IoError> {
    let resolved = store::resolve(uri)?;
    let (size, arrow_meta) = load_metadata_cached(uri, &resolved).await?;

    // Which row-groups: the requested subset, else all of them.
    let all: Vec<usize> = (0..arrow_meta.metadata().num_row_groups()).collect();
    let mut targets: Vec<usize> = if row_groups.is_empty() {
        all
    } else {
        row_groups.to_vec()
    };

    // Predicate pushdown: drop the row-groups whose footer statistics prove no row can
    // match (a whole group of column-chunk GETs + decode skipped). Superset-safe — the
    // engine keeps the `Filter`, so a group we cannot prune is simply read and re-filtered.
    let parsed = predicate.and_then(predicate::parse);
    if let Some(pred) = parsed.as_ref() {
        targets = predicate::surviving_row_groups(arrow_meta.metadata(), pred, &targets);
    }
    if targets.is_empty() {
        return Ok(Vec::new()); // every row-group pruned → provably empty
    }

    // Leaf-column projection by name, pushed into the decode (computed once, shared).
    // A `ProjectionMask` is a *set* of leaf indices — it selects columns but does NOT
    // reorder them, so the decoded batch keeps the file's column order. PyArrow's
    // `read_table(columns=[...])` (the fallback this reader must be byte-identical to)
    // returns the columns in the *requested* order, so a reordered projection
    // (`["c","a"]` on a file laid out `a,b,c`) would otherwise come back as `[a,c]`
    // here but `[c,a]` from PyArrow — a silent column-order divergence. We reorder the
    // output below to the requested order to honor that contract.
    let projection = columns.map(|cols| {
        projection::exact_columns(arrow_meta.parquet_schema(), cols.iter().map(|s| s.as_str()))
    });

    // Decide, by measurement, whether a row-level filter pays on this read.
    //
    // A `RowFilter` is the only pruning step here that can *lose*: below `MAX_SELECTIVITY` it
    // saves decoding the non-predicate columns of every rejected row, and above it the
    // fragmented row selection costs more than the decode it skips (measured 1.45x faster at
    // ~2 % selected, 1.57x *slower* at ~95 %). Nothing in the footer can distinguish the two —
    // a scattered predicate leaves every row group's [min, max] spanning the domain whether it
    // selects 2 % or 95 % — so the only honest input is a measurement.
    //
    // The probe decodes just the predicate columns of the first surviving row group, which is
    // a single narrow column chunk, and applies the verdict to the whole read. Selectivity can
    // of course vary between row groups; getting that wrong costs only speed, never rows.
    //
    // Indexing is deliberately non-panicking. `ParquetMetaData::row_group(i)` panics on an
    // out-of-bounds index, and an out-of-range split index reaches here as a plain candidate:
    // probing one would turn a bad index into a process-aborting panic across the FFI, where
    // the un-probed path reports it as a clean `row group N out of bounds` error. So the row
    // counts come from `.get()`, and an out-of-range first target skips the probe entirely and
    // leaves the index for the decoder to report exactly as it does today.
    let row_groups = arrow_meta.metadata().row_groups();
    let probe_rows: usize = targets
        .iter()
        .filter_map(|&rg| row_groups.get(rg))
        .map(|rg| rg.num_rows() as usize)
        .sum();
    let mut row_filter_cols: Option<Vec<String>> = None;
    if let Some(pred) = parsed.as_ref() {
        // Under this size the whole read is already short and the probe would be a larger
        // share of it than anything the filter could save.
        if row_filter_enabled()
            && probe_rows >= ROW_FILTER_MIN_ROWS
            && row_groups.get(targets[0]).is_some()
        {
            // Free pre-check before the probe: if the zone maps already say the predicate keeps
            // most rows, there is nothing for the filter to save and the probe itself would be
            // the only cost anyone measured. The estimate is only ever allowed to *decline*
            // (see `row_filter::estimate`) — installing stays a measured decision, because the
            // interpolation behind it is wrong on skewed data and a wrong install is a
            // slowdown while a wrong decline is only a missed speed-up.
            let permissive = {
                let (mut weighted, mut rows) = (0.0f64, 0.0f64);
                let mut usable = true;
                // Column positions resolved once for the file; the loop below asks for the
                // same columns in every row group.
                let col_index = predicate::ColumnIndex::build(arrow_meta.metadata());
                for &rg in &targets {
                    let Some(meta) = row_groups.get(rg) else {
                        continue;
                    };
                    match row_filter::estimate(pred, meta, &col_index) {
                        Some(f) => {
                            weighted += f * meta.num_rows() as f64;
                            rows += meta.num_rows() as f64;
                        }
                        None => {
                            usable = false;
                            break;
                        }
                    }
                }
                usable && rows > 0.0 && !row_filter::worth_it_frac(weighted / rows)
            };
            if !permissive {
                if let Some(cols) = row_filter::plan(pred, arrow_meta.schema()) {
                    let reader =
                        ParquetObjectReader::new(resolved.store.clone(), resolved.path.clone())
                            .with_file_size(size);
                    let mask = projection::exact_columns(
                        arrow_meta.parquet_schema(),
                        cols.iter().map(String::as_str),
                    );
                    let mut probe = ParquetRecordBatchStreamBuilder::new_with_metadata(
                        reader,
                        arrow_meta.clone(),
                    )
                    .with_batch_size(batch_size.max(1))
                    .with_row_groups(vec![targets[0]])
                    .with_projection(mask)
                    .build()?;
                    // Stop as soon as the estimate is good enough. Decoding the *whole* first row
                    // group to measure it cost more than the filter saved on a permissive
                    // predicate (~18 ms, turning a 179 ms read into 198 ms); a few thousand rows
                    // answer "is this selective?" just as well and cost ~1 ms. The estimate is a
                    // sample, so a clustered column can mislead it — which changes only speed.
                    let (mut selected, mut total) = (0usize, 0usize);
                    while total < ROW_FILTER_PROBE_ROWS {
                        let Some(batch) = probe.try_next().await? else {
                            break;
                        };
                        let m = row_filter::mask_of(pred, &batch);
                        total += m.len();
                        selected += m.true_count();
                    }
                    if row_filter::worth_it(selected, total) {
                        row_filter_cols = Some(cols);
                    }
                }
            }
        }
    }

    // Read row-groups CONCURRENTLY: each as its own short stream over a cloned reader
    // (which shares the Arc'd store + connection pool and the already-parsed metadata).
    // Each row-group future is `tokio::spawn`ed onto the runtime's worker pool — NOT merely
    // `buffered`, which polls the futures cooperatively on the calling task's single thread
    // and would serialize the CPU-bound Parquet decode (overlapping only the I/O). Spawning
    // spreads the decode of a many-row-group file across cores (the win on one large file —
    // TPC-H `lineitem` is a single 16 GB, ~600 M-row file); `buffered` still bounds the
    // in-flight count to `rg_concurrency()` (unchanged memory) and preserves file order.
    // This mirrors the per-file spawn `read_parquet_many` already does across files.
    let store = resolved.store;
    let loc = resolved.path;
    let remote = resolved.remote;
    let batch_size = batch_size.max(1);
    let per_rg = targets.into_iter().map(|rg| {
        // Over the network a row group's contiguous column chunks coalesce into one enormous
        // GET, which one connection then serves at a fraction of the link — see `split_read`.
        // Local reads keep the plain reader: the page cache has no such limit.
        let base = ParquetObjectReader::new(store.clone(), loc.clone()).with_file_size(size);
        let reader = split_read::maybe_split(base, &store, &loc, remote);
        let amd = arrow_meta.clone();
        let proj = projection.clone();
        // Page-level pruning *within* a surviving row group. Computed here, on the metadata
        // this task already holds, and scoped to this one row group — which is what makes
        // the selection's row numbering trivially correct, since the builder below reads
        // exactly this group and nothing else.
        let selection = parsed
            .as_ref()
            .and_then(|pred| page_index::row_selection(arrow_meta.metadata(), pred, rg));
        let bloom_pred = parsed.clone();
        let bloom_meta = arrow_meta.clone();
        let rf_cols = row_filter_cols.clone();
        tokio::spawn(async move {
            let mut b = ParquetRecordBatchStreamBuilder::new_with_metadata(reader, amd)
                .with_batch_size(batch_size)
                .with_row_groups(vec![rg]);
            if let Some(p) = proj {
                b = b.with_projection(p);
            }
            if let Some(s) = selection {
                b = b.with_row_selection(s);
            }
            // Last, and only for equality predicates: the bloom. It is the one pruning step
            // that costs a round trip, so it runs after the free ones have narrowed things,
            // and it answers the case they cannot — an equality on a high-cardinality
            // unordered column, where every page's [min, max] spans the domain.
            if let Some(pred) = bloom_pred.as_ref() {
                if bloom::provably_absent(&mut b, bloom_meta.metadata(), pred, rg).await {
                    return Ok(Vec::new());
                }
            }
            // Last: row-level pushdown *into* the decode. Every step above prunes whole
            // blocks — row group, page, bloom — and none of them can help a predicate whose
            // matches are scattered, which leaves every block alive and every column fully
            // decoded before the engine's `Filter` discards most of it. A `RowFilter` decodes
            // the predicate columns first and decodes the rest for surviving rows only, so
            // the saving is the width of the table times the rows rejected. Installed only
            // when the probe above measured it worth doing, and only for a predicate proved to
            // carry the engine's own comparison semantics — unlike the pruning above, this
            // step *removes rows* rather than skipping provably-empty work.
            if let (Some(pred), Some(cols)) = (bloom_pred.as_ref(), rf_cols.as_ref()) {
                b = b.with_row_filter(row_filter::build(pred, cols, bloom_meta.parquet_schema()));
            }
            let stream = b.build()?;
            stream.try_collect::<Vec<RecordBatch>>().await
        })
    });

    let per_rg_batches: Vec<Vec<RecordBatch>> = futures::stream::iter(per_rg)
        .buffered(rg_concurrency())
        .map(|joined| {
            joined
                .map_err(|e| IoError::Store(e.to_string()))?
                .map_err(IoError::from)
        })
        .try_collect()
        .await?;
    let mut batches: Vec<RecordBatch> = per_rg_batches.into_iter().flatten().collect();
    // Match PyArrow: return columns in the requested projection order, not file order.
    if let Some(cols) = columns {
        reorder_to_projection(&mut batches, cols);
    }
    Ok(batches)
}

/// Reorder each batch's columns to the requested projection order (PyArrow parity).
///
/// The decoder emits columns in the file's schema order regardless of the order the
/// caller asked for. When the requested names map one-to-one onto the batch's top-level
/// fields, reorder to the requested order so the result is identical to PyArrow's
/// `read_table(columns=[...])`. When they do not form a clean bijection (a nested/leaf
/// projection, a duplicated or absent name), leave the batch untouched — the reorder is
/// only defined for the flat top-level projections the engine actually issues, and
/// touching the exotic cases would risk mangling them.
fn reorder_to_projection(batches: &mut [RecordBatch], columns: &[String]) {
    let Some(first) = batches.first() else {
        return;
    };
    let schema = first.schema();
    // A clean bijection: same count, and every requested name resolves to a distinct field.
    if columns.len() != schema.fields().len() {
        return;
    }
    let mut order = Vec::with_capacity(columns.len());
    for name in columns {
        match schema.index_of(name) {
            Ok(idx) if !order.contains(&idx) => order.push(idx),
            _ => return, // absent or duplicate name → not a clean reorder, leave as-is
        }
    }
    if order.iter().enumerate().all(|(i, &idx)| i == idx) {
        return; // already in requested order (the common case) — no work
    }
    for b in batches.iter_mut() {
        let cols: Vec<_> = order.iter().map(|&i| b.column(i).clone()).collect();
        let fields: Vec<_> = order.iter().map(|&i| b.schema().field(i).clone()).collect();
        let new_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(fields));
        // Reindexing existing columns of a valid batch cannot fail; keep the original on
        // the impossible error rather than dropping data.
        if let Ok(nb) = RecordBatch::try_new(new_schema, cols) {
            *b = nb;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    use super::*;

    fn write_parquet(path: &std::path::Path, batches: &[RecordBatch], rows_per_group: usize) {
        let file = std::fs::File::create(path).unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_size(rows_per_group)
            .build();
        let mut w = ArrowWriter::try_new(file, batches[0].schema(), Some(props)).unwrap();
        for b in batches {
            w.write(b).unwrap();
        }
        w.close().unwrap();
    }

    fn sample(n: i64) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Float64, false),
        ]));
        let a = Int64Array::from((0..n).collect::<Vec<_>>());
        let b = Float64Array::from((0..n).map(|x| x as f64 * 0.5).collect::<Vec<_>>());
        RecordBatch::try_new(schema, vec![Arc::new(a), Arc::new(b)]).unwrap()
    }

    #[test]
    fn reads_local_parquet_all() {
        let dir = std::env::temp_dir().join(format!("bcio_all_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.parquet");
        write_parquet(&p, &[sample(1000)], 256); // ~4 row-groups
        let out = read_parquet(p.to_str().unwrap(), &[], None, 512).unwrap();
        let rows: usize = out.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 1000);
        assert_eq!(out[0].num_columns(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn many_row_groups_stay_in_order() {
        // A single file with many small row-groups exercises the spawned per-row-group
        // decode: `buffered` must reassemble them in file order regardless of which
        // spawned task finishes first. Column `a` is 0..N, so any reordering shows up.
        let dir = std::env::temp_dir().join(format!("bcio_order_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.parquet");
        write_parquet(&p, &[sample(4000)], 32); // 125 row-groups
        let out = read_parquet(p.to_str().unwrap(), &[], None, 97).unwrap();
        let a: Vec<i64> = out
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        assert_eq!(a, (0..4000).collect::<Vec<_>>());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn predicate_prunes_row_groups_superset_safe() {
        // 4 row-groups of 250 rows: column `a` ranges 0..999 across them (0-249, 250-499,
        // 500-749, 750-999). `a >= 500` can match only row-groups 2 and 3 → the pruned read
        // returns exactly their rows (a superset of the true matches; the engine's Filter
        // finishes). A predicate no group can satisfy returns empty.
        let dir = std::env::temp_dir().join(format!("bcio_pred_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.parquet");
        write_parquet(&p, &[sample(1000)], 250);
        let path = p.to_str().unwrap();

        let ge500 = r#"{"node":"cmp","col":"a","op":"ge","lit":500}"#;
        let out = read_parquet_filtered(path, &[], None, 4096, ge500).unwrap();
        let rows: usize = out.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 500, "kept exactly row-groups 2+3");

        // Every value is < 5000, so `a > 100000` prunes all row-groups → empty.
        let none = r#"{"node":"cmp","col":"a","op":"gt","lit":100000}"#;
        let empty = read_parquet_filtered(path, &[], None, 4096, none).unwrap();
        assert_eq!(empty.iter().map(|b| b.num_rows()).sum::<usize>(), 0);

        // A malformed / non-pushable predicate must read everything (never fail/underread).
        let out_all = read_parquet_filtered(path, &[], None, 4096, "not json").unwrap();
        assert_eq!(out_all.iter().map(|b| b.num_rows()).sum::<usize>(), 1000);
        std::fs::remove_dir_all(&dir).ok();
    }

    fn sample3(n: i64) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Float64, false),
            Field::new("c", DataType::Int64, false),
        ]));
        let a = Int64Array::from((0..n).collect::<Vec<_>>());
        let b = Float64Array::from((0..n).map(|x| x as f64 * 0.5).collect::<Vec<_>>());
        let c = Int64Array::from((0..n).map(|x| x * 100).collect::<Vec<_>>());
        RecordBatch::try_new(schema, vec![Arc::new(a), Arc::new(b), Arc::new(c)]).unwrap()
    }

    #[test]
    fn reordered_projection_returns_requested_order() {
        // File is laid out [a, b, c]. Requesting ["c", "a"] must return columns in the
        // REQUESTED order (matching PyArrow's `read_table(columns=[...])`, the fallback
        // this reader is contracted to be byte-identical to) — not the file order [a, c].
        let dir = std::env::temp_dir().join(format!("bcio_reorder_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.parquet");
        write_parquet(&p, &[sample3(300)], 1000);
        let cols = vec!["c".to_string(), "a".to_string()];
        let out = read_parquet(p.to_str().unwrap(), &[], Some(&cols), 4096).unwrap();
        let schema = out[0].schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["c", "a"], "columns must follow requested order");
        // And the data must travel with its name: column 0 is `c` (values x*100).
        let c0 = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(c0.value(0), 0);
        assert_eq!(c0.value(1), 100);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `ColumnIndex` must resolve exactly the columns the scan it replaced resolved.
    ///
    /// It exists to stop re-deriving one answer per row group, so the risk is that the
    /// cached answer is a *different* answer. Two properties carry the correctness: a flat
    /// top-level column resolves to statistics, and a nested leaf whose final name collides
    /// with a top-level column does **not** — matching that leaf is what once let `s.a`
    /// shadow `a` and prune away every matching row.
    #[test]
    fn the_column_index_resolves_flat_columns_and_never_a_nested_leaf() {
        use arrow::array::{ArrayRef, StructArray};
        use arrow::datatypes::Fields;

        let dir = std::env::temp_dir().join(format!("bcio_colidx_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.parquet");
        // `s{a}` collides on the leaf name with the top-level `a`, and `b` is flat.
        let inner_a = Arc::new(Int64Array::from(vec![0i64, 1])) as ArrayRef;
        let struct_fields: Fields = vec![Field::new("a", DataType::Int64, false)].into();
        let s = StructArray::new(struct_fields.clone(), vec![inner_a], None);
        let top_a = Arc::new(Int64Array::from(vec![500i64, 600])) as ArrayRef;
        let b = Arc::new(Int64Array::from(vec![7i64, 8])) as ArrayRef;
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Struct(struct_fields), false),
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(s), top_a, b]).unwrap();
        write_parquet(&p, &[batch], 1000);

        let file = std::fs::File::open(&p).unwrap();
        let md = ArrowReaderMetadata::load(&file, ArrowReaderOptions::new()).unwrap();
        let idx = predicate::ColumnIndex::build(md.metadata());
        let rg = md.metadata().row_group(0);

        // The top-level `a` resolves, and to its own statistics (500..600), not `s.a`'s 0..1.
        let (stats, unsigned) = idx.stats(rg, "a").expect("top-level `a` resolves");
        assert!(!unsigned);
        match stats {
            parquet::file::statistics::Statistics::Int64(v) => {
                assert_eq!(
                    *v.min_opt().unwrap(),
                    500,
                    "resolved the nested `s.a` instead"
                );
                assert_eq!(*v.max_opt().unwrap(), 600);
            }
            other => panic!("unexpected statistics type: {other:?}"),
        }
        assert!(
            idx.stats(rg, "b").is_some(),
            "a second flat column resolves"
        );
        // The nested leaf is reachable by neither its leaf name (that is `a`, already taken
        // by the top-level column) nor its dotted path — a predicate on it finds nothing and
        // conservatively keeps the group.
        assert!(idx.stats(rg, "s.a").is_none());
        assert!(
            idx.stats(rg, "s").is_none(),
            "a struct itself has no leaf stats"
        );
        assert!(idx.stats(rg, "absent").is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn predicate_ignores_nested_field_with_colliding_leaf_name() {
        // A struct column `s{a}` shares the leaf name `a` with a top-level column `a`.
        // A predicate on the top-level `a` must prune using the TOP-LEVEL column's stats,
        // never the nested field's — otherwise the wrong (nested) min/max drops rows that
        // actually match. Here the nested `s.a` is 0..3 while the top-level `a` is 500..800;
        // `a >= 500` must keep all rows, but leaf-name matching would prune the group to 0.
        use arrow::array::{ArrayRef, StructArray};
        use arrow::datatypes::Fields;

        let dir = std::env::temp_dir().join(format!("bcio_nest_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.parquet");
        let inner_a = Arc::new(Int64Array::from(vec![0i64, 1, 2, 3])) as ArrayRef;
        let struct_fields: Fields = vec![Field::new("a", DataType::Int64, false)].into();
        let s = StructArray::new(struct_fields.clone(), vec![inner_a], None);
        let top_a = Arc::new(Int64Array::from(vec![500i64, 600, 700, 800])) as ArrayRef;
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Struct(struct_fields), false),
            Field::new("a", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(s), top_a]).unwrap();
        write_parquet(&p, &[batch], 1000);
        let pred = r#"{"node":"cmp","col":"a","op":"ge","lit":500}"#;
        let out = read_parquet_filtered(p.to_str().unwrap(), &[], None, 4096, pred).unwrap();
        let rows: usize = out.iter().map(|b| b.num_rows()).sum();
        assert_eq!(
            rows, 4,
            "top-level `a` matched all rows; nested `s.a` must not prune"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn out_of_range_row_group_with_predicate_errs_not_panics() {
        // An out-of-range split index must produce the SAME clean error with a predicate
        // as without one — predicate pushdown must never turn a bad index into a panic
        // (`meta.row_group(rg)` panics on OOB; the fix keeps the index for the decoder).
        let dir = std::env::temp_dir().join(format!("bcio_oob_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.parquet");
        write_parquet(&p, &[sample(100)], 1000); // one row-group (index 0 only)
        let path = p.to_str().unwrap();
        let pred = r#"{"node":"cmp","col":"a","op":"ge","lit":0}"#;
        // Must be a clean Err, not a panic (would abort the process across FFI).
        let filtered = read_parquet_filtered(path, &[5], None, 4096, pred);
        assert!(filtered.is_err(), "OOB row group must error, not panic");
        // Identical to the no-predicate path's behavior.
        assert!(read_parquet(path, &[5], None, 4096).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn unsigned_column_predicate_does_not_drop_matching_rows() {
        // Parquet stores an unsigned column (UInt32/UInt64) in a *signed* physical
        // INT32/INT64 whose footer min/max are computed by unsigned order. A large value
        // (3e9 in a UInt32 → -1_294_967_296 as i32) therefore reads back negative; taking
        // the stat as signed makes `u >= 2e9` prune the whole group and silently drop the
        // 3e9 row that actually matches — a violation of the superset-safe contract.
        use arrow::array::{UInt32Array, UInt64Array};
        let dir = std::env::temp_dir().join(format!("bcio_unsigned_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // UInt32 with a value above i32::MAX.
        let p32 = dir.join("u32.parquet");
        let schema32 = Arc::new(Schema::new(vec![Field::new("u", DataType::UInt32, false)]));
        let u32b = RecordBatch::try_new(
            schema32,
            vec![Arc::new(UInt32Array::from(vec![10u32, 3_000_000_000u32]))],
        )
        .unwrap();
        write_parquet(&p32, &[u32b], 1000);
        let ge = r#"{"node":"cmp","col":"u","op":"ge","lit":2000000000}"#;
        let out = read_parquet_filtered(p32.to_str().unwrap(), &[], None, 4096, ge).unwrap();
        // The invariant is the one this test is named for: the *matching* row survives. It is
        // asserted by value rather than by row count because `row_filter` now also evaluates
        // the predicate per row, so the non-matching `10` is legitimately gone. The hazard
        // being guarded is unchanged and still caught: if the unsigned max were read as a
        // signed `-1_294_967_296`, the group would be pruned whole and `3e9` would vanish.
        let vals: Vec<u32> = out
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        assert!(
            vals.contains(&3_000_000_000u32),
            "unsigned UInt32 group must be kept (3e9 >= 2e9), got {vals:?}"
        );
        // A predicate the unsigned range truly excludes still prunes.
        let gt = r#"{"node":"cmp","col":"u","op":"gt","lit":4000000000}"#;
        let none = read_parquet_filtered(p32.to_str().unwrap(), &[], None, 4096, gt).unwrap();
        assert_eq!(none.iter().map(|b| b.num_rows()).sum::<usize>(), 0);

        // UInt64 with a value above i64::MAX.
        let p64 = dir.join("u64.parquet");
        let schema64 = Arc::new(Schema::new(vec![Field::new("u", DataType::UInt64, false)]));
        let big = 10_000_000_000_000_000_000u64; // > i64::MAX
        let u64b =
            RecordBatch::try_new(schema64, vec![Arc::new(UInt64Array::from(vec![5u64, big]))])
                .unwrap();
        write_parquet(&p64, &[u64b], 1000);
        // `u >= 9e18` (< i64::MAX) must keep the group — `big` matches.
        let ge64 = r#"{"node":"cmp","col":"u","op":"ge","lit":9000000000000000000}"#;
        let out64 = read_parquet_filtered(p64.to_str().unwrap(), &[], None, 4096, ge64).unwrap();
        // As above: assert the matching value survives, not the unfiltered row count.
        let vals64: Vec<u64> = out64
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        assert!(
            vals64.contains(&big),
            "unsigned UInt64 group must be kept, got {vals64:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn row_filter_returns_exactly_the_matching_rows() {
        // The row filter is the one pruning step here that *removes rows*, so its output must
        // equal the predicate applied by hand — not merely a superset of it. Sized past
        // `ROW_FILTER_MIN_ROWS` so the filter actually engages, and shaped so the predicate is
        // selective enough to survive the selectivity gate.
        let dir = std::env::temp_dir().join(format!("bcio_rowfilter_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.parquet");
        let n: i64 = 300_000;
        // Deliberately **scattered**, not sorted. On sorted data the footer zone maps prune
        // every non-matching row group first, the survivors fall under `ROW_FILTER_MIN_ROWS`,
        // and the row filter never engages — so a sorted fixture would silently test nothing.
        // A stride co-prime with `n` permutes 0..n and leaves every row group's [min, max]
        // spanning the domain, which is exactly the shape only a row filter can prune.
        let scattered: Vec<i64> = (0..n).map(|i| (i * 7919) % n).collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(scattered)),
                Arc::new(Float64Array::from(
                    (0..n).map(|x| x as f64 * 0.5).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap();
        write_parquet(&p, &[batch], 25_000);
        let path = p.to_str().unwrap();

        // `a < 10000` over 0..300000 selects 10,000 rows — 3.3 %, comfortably selective.
        let pred = r#"{"node":"cmp","col":"a","op":"lt","lit":10000}"#;
        let out = read_parquet_filtered(path, &[], None, 8192, pred).unwrap();
        let mut got: Vec<i64> = out
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        got.sort_unstable(); // the file order is a permutation; compare as a set
        let want: Vec<i64> = (0..10_000).collect();
        assert_eq!(
            got.len(),
            want.len(),
            "row count must be exact, not a superset"
        );
        assert_eq!(got, want, "values must be exactly the matching rows");

        // An AND of two ranges, to exercise the Kleene combination path.
        let both = r#"{"node":"and","left":{"node":"cmp","col":"a","op":"ge","lit":100},"right":{"node":"cmp","col":"a","op":"lt","lit":5000}}"#;
        let out2 = read_parquet_filtered(path, &[], None, 8192, both).unwrap();
        let mut got2: Vec<i64> = out2
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        got2.sort_unstable();
        assert_eq!(got2, (100..5000).collect::<Vec<i64>>());

        // A permissive predicate is declined by the selectivity gate, so it returns the
        // un-filtered superset — still correct, because the engine keeps its own `Filter`.
        // Asserting the *matching* rows are all present is what matters either way.
        let wide = r#"{"node":"cmp","col":"a","op":"lt","lit":299000}"#;
        let out3 = read_parquet_filtered(path, &[], None, 8192, wide).unwrap();
        let rows3: usize = out3.iter().map(|b| b.num_rows()).sum();
        assert!(
            rows3 >= 299_000,
            "a declined filter must not drop matching rows, got {rows3}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn selectivity_estimate_tracks_the_zone_map() {
        // The estimator is what lets a permissive predicate decline the row filter for free,
        // without paying for a probe. It only has to be right about *which side of the
        // threshold* it is on, so that is what this asserts.
        let dir = std::env::temp_dir().join(format!("bcio_est_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.parquet");
        write_parquet(&p, &[sample(10_000)], 10_000); // one row group, a = 0..10000
        let file = std::fs::File::open(&p).unwrap();
        let md = ArrowReaderMetadata::load(&file, ArrowReaderOptions::new()).unwrap();
        let rg = md.metadata().row_group(0);
        let idx = predicate::ColumnIndex::build(md.metadata());

        let est = |json: &str| {
            let pred = predicate::parse(json).unwrap();
            row_filter::estimate(&pred, rg, &idx).unwrap()
        };

        // 2 % of the [0, 9999] span sits below 200.
        let selective = est(r#"{"node":"cmp","col":"a","op":"lt","lit":200}"#);
        assert!(selective < 0.05, "expected ~0.02, got {selective}");
        assert!(row_filter::worth_it_frac(selective));

        // 95 % sits below 9500 — the case that must decline.
        let permissive = est(r#"{"node":"cmp","col":"a","op":"lt","lit":9500}"#);
        assert!(permissive > 0.9, "expected ~0.95, got {permissive}");
        assert!(!row_filter::worth_it_frac(permissive));

        // `>` is the complement of `<` over the same bound.
        let gt = est(r#"{"node":"cmp","col":"a","op":"gt","lit":9500}"#);
        assert!((gt + permissive - 1.0).abs() < 1e-9);

        // AND multiplies, so two selective terms stay selective.
        let both = est(
            r#"{"node":"and","left":{"node":"cmp","col":"a","op":"lt","lit":200},"right":{"node":"cmp","col":"a","op":"lt","lit":200}}"#,
        );
        assert!(both < selective);

        // A non-numeric literal has no meaningful span, so the estimator abstains and the
        // caller falls through to the measured probe rather than inventing a number.
        let s = predicate::parse(r#"{"node":"cmp","col":"a","op":"eq","lit":"x"}"#).unwrap();
        assert!(row_filter::estimate(&s, rg, &idx).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn projection_and_row_group_selection() {
        let dir = std::env::temp_dir().join(format!("bcio_proj_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.parquet");
        write_parquet(&p, &[sample(1000)], 250); // exactly 4 row-groups of 250
                                                 // Only column "b", only row-groups 0 and 2 → 500 rows, 1 column.
        let cols = vec!["b".to_string()];
        let out = read_parquet(p.to_str().unwrap(), &[0, 2], Some(&cols), 4096).unwrap();
        let rows: usize = out.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 500);
        assert_eq!(out[0].num_columns(), 1);
        assert_eq!(out[0].schema().field(0).name(), "b");
        std::fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(test)]
mod bloom_tests {
    use std::sync::Arc;

    use arrow::array::Int64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    use super::*;

    /// Write a file whose `k` column carries a bloom filter.
    ///
    /// Batcher's own writer cannot emit blooms (the pinned pyarrow has no option for it),
    /// so this fixture is what makes the read path testable at all. Without it the feature
    /// would be unverifiable — and an unverifiable pruner is exactly how the page-index
    /// work first shipped as a silent no-op.
    fn write_with_bloom(path: &std::path::Path, keys: &[i64], rows_per_group: usize) {
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(keys.to_vec()))],
        )
        .unwrap();
        let props = WriterProperties::builder()
            .set_bloom_filter_enabled(true)
            .set_max_row_group_size(rows_per_group)
            .build();
        let file = std::fs::File::create(path).unwrap();
        let mut w = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    fn dir_for(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("bcio_bloom_{name}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn eq_pred(value: i64) -> String {
        format!(r#"{{"node":"cmp","col":"k","op":"eq","lit":{value}}}"#)
    }

    /// The gap this closes: an equality on a high-cardinality *unordered* column, where
    /// every group's `[min, max]` spans the domain so range pruning achieves nothing.
    fn scattered(n: i64) -> Vec<i64> {
        // A fixed stride over a large domain: each row group's min/max still spans nearly
        // everything, so only a bloom can decide. Deterministic (no rng in a unit test).
        (0..n).map(|i| (i * 7919) % 1_000_003).collect()
    }

    #[test]
    fn a_value_that_is_absent_prunes_every_row_group() {
        let dir = dir_for("absent");
        let p = dir.join("t.parquet");
        let keys = scattered(4000);
        write_with_bloom(&p, &keys, 1000);

        // 1_000_003 is the modulus, so no key can equal it — and it sits inside the
        // min/max range, which is what makes range pruning useless here.
        let out = read_parquet_filtered(p.to_str().unwrap(), &[], None, 512, &eq_pred(1_000_002))
            .unwrap();
        let rows: usize = out.iter().map(|b| b.num_rows()).sum();

        assert!(!keys.contains(&1_000_002));
        assert_eq!(rows, 0, "bloom did not prune an absent value");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_value_that_is_present_is_never_pruned() {
        // The safety direction. A bloom has no false negatives, so a present value must
        // always survive — losing it would be silent data loss.
        let dir = dir_for("present");
        let p = dir.join("t.parquet");
        let keys = scattered(4000);
        write_with_bloom(&p, &keys, 1000);

        for probe in [keys[0], keys[1500], keys[3999]] {
            let out = read_parquet_filtered(p.to_str().unwrap(), &[], None, 512, &eq_pred(probe))
                .unwrap();
            let found: Vec<i64> = out
                .iter()
                .flat_map(|b| {
                    b.column(0)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .values()
                        .to_vec()
                })
                .collect();
            assert!(
                found.contains(&probe),
                "bloom pruned a value that is present"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_file_without_blooms_is_unaffected() {
        let dir = dir_for("noindex");
        let p = dir.join("t.parquet");
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from((0..1000i64).collect::<Vec<_>>()))],
        )
        .unwrap();
        let file = std::fs::File::create(&p).unwrap();
        let mut w = ArrowWriter::try_new(file, schema, None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();

        // No bloom to consult; the row group is read and the engine's Filter does the work.
        let out =
            read_parquet_filtered(p.to_str().unwrap(), &[], None, 512, &eq_pred(500)).unwrap();
        let rows: usize = out.iter().map(|b| b.num_rows()).sum();

        assert!(rows > 0, "a file without blooms must still be read");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_disjunction_needs_both_sides_absent_to_prune() {
        // The lattice, in the direction that loses rows if inverted: `a OR b` is empty only
        // when BOTH are. Treating one absent side as proof would drop the other's matches.
        let dir = dir_for("disjunction");
        let p = dir.join("t.parquet");
        let keys = scattered(4000);
        write_with_bloom(&p, &keys, 1000);
        let present = keys[42];
        let pred = format!(
            r#"{{"node":"or","left":{},"right":{}}}"#,
            eq_pred(1_000_002),
            eq_pred(present)
        );

        let out = read_parquet_filtered(p.to_str().unwrap(), &[], None, 512, &pred).unwrap();
        let found: Vec<i64> = out
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();

        assert!(
            found.contains(&present),
            "an OR was pruned by one absent side"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
