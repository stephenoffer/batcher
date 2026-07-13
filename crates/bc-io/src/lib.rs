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

use std::sync::OnceLock;

use arrow::record_batch::RecordBatch;
use futures::{StreamExt, TryStreamExt};
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use parquet::arrow::async_reader::ParquetObjectReader;
use parquet::arrow::{ParquetRecordBatchStreamBuilder, ProjectionMask};

mod avro;
mod predicate;
mod store;

pub use avro::read_avro_bytes;

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
}

/// One shared multi-threaded Tokio runtime for all reads in the process. The async
/// parquet reader needs an executor; sharing one runtime lets concurrent split reads
/// (one per worker thread) overlap their object-store I/O on a common thread pool
/// instead of each spinning up its own.
fn runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
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
/// emits; see [`predicate`]) prunes the requested row-groups by their footer statistics
/// before decode. Pruning is superset-safe (the engine keeps the `Filter`), so an
/// unparseable or non-pushable predicate simply reads every requested row-group.
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
fn file_concurrency() -> usize {
    static C: OnceLock<usize> = OnceLock::new();
    *C.get_or_init(|| {
        std::env::var("BATCHER_PARQUET_FILE_CONCURRENCY")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(64)
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
/// Parquet files are write-once, so the footer is immutable and safe to cache. This is
/// the "never read the same metadata twice" guarantee: multiple splits of one file, and
/// repeated queries over warm (session-fleet) workers, parse + fetch the footer ONCE
/// instead of per read. `ArrowReaderMetadata` is `Arc`-backed, so a hit is a cheap clone.
fn meta_cache(
) -> &'static std::sync::Mutex<std::collections::HashMap<String, (u64, ArrowReaderMetadata)>> {
    static C: OnceLock<
        std::sync::Mutex<std::collections::HashMap<String, (u64, ArrowReaderMetadata)>>,
    > = OnceLock::new();
    C.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

async fn load_metadata_cached(
    uri: &str,
    resolved: &store::Resolved,
) -> Result<(u64, ArrowReaderMetadata), IoError> {
    if let Some(hit) = meta_cache().lock().unwrap().get(uri) {
        return Ok(hit.clone());
    }
    // Cold: one HEAD for the size, then one ranged GET for the footer (no probing), parse
    // once. Stored so no later read of this file re-reads or re-parses the footer.
    let meta = resolved.store.head(&resolved.path).await?;
    let mut probe = ParquetObjectReader::new(resolved.store.clone(), resolved.path.clone())
        .with_file_size(meta.size);
    let amd = ArrowReaderMetadata::load_async(&mut probe, ArrowReaderOptions::new()).await?;
    meta_cache()
        .lock()
        .unwrap()
        .insert(uri.to_string(), (meta.size, amd.clone()));
    Ok((meta.size, amd))
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
    if let Some(json) = predicate {
        if let Some(pred) = predicate::parse(json) {
            targets = predicate::surviving_row_groups(arrow_meta.metadata(), &pred, &targets);
        }
    }
    if targets.is_empty() {
        return Ok(Vec::new()); // every row-group pruned → provably empty
    }

    // Leaf-column projection by name, pushed into the decode (computed once, shared).
    let projection = columns.map(|cols| {
        ProjectionMask::columns(arrow_meta.parquet_schema(), cols.iter().map(|s| s.as_str()))
    });

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
    let batch_size = batch_size.max(1);
    let per_rg = targets.into_iter().map(|rg| {
        let reader = ParquetObjectReader::new(store.clone(), loc.clone()).with_file_size(size);
        let amd = arrow_meta.clone();
        let proj = projection.clone();
        tokio::spawn(async move {
            let mut b = ParquetRecordBatchStreamBuilder::new_with_metadata(reader, amd)
                .with_batch_size(batch_size)
                .with_row_groups(vec![rg]);
            if let Some(p) = proj {
                b = b.with_projection(p);
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
    Ok(per_rg_batches.into_iter().flatten().collect())
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
