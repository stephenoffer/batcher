//! Native Rust Parquet reader over uniform object storage.
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

mod store;

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
    runtime().block_on(read_parquet_async(uri, row_groups, columns, batch_size))
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
) -> Result<Vec<RecordBatch>, IoError> {
    let resolved = store::resolve(uri)?;
    let (size, arrow_meta) = load_metadata_cached(uri, &resolved).await?;

    // Which row-groups: the requested subset, else all of them.
    let all: Vec<usize> = (0..arrow_meta.metadata().num_row_groups()).collect();
    let targets: Vec<usize> = if row_groups.is_empty() {
        all
    } else {
        row_groups.to_vec()
    };

    // Leaf-column projection by name, pushed into the decode (computed once, shared).
    let projection = columns.map(|cols| {
        ProjectionMask::columns(arrow_meta.parquet_schema(), cols.iter().map(|s| s.as_str()))
    });

    // Read row-groups CONCURRENTLY: each as its own short stream over a cloned reader
    // (which shares the Arc'd store + connection pool and the already-parsed metadata).
    // `buffered` keeps file order while overlapping up to `rg_concurrency()` row-groups'
    // object-store GETs — the throughput fix over the sequential single stream.
    let store = resolved.store;
    let loc = resolved.path;
    let batch_size = batch_size.max(1);
    let per_rg = targets.into_iter().map(|rg| {
        let reader = ParquetObjectReader::new(store.clone(), loc.clone()).with_file_size(size);
        let amd = arrow_meta.clone();
        let proj = projection.clone();
        async move {
            let mut b = ParquetRecordBatchStreamBuilder::new_with_metadata(reader, amd)
                .with_batch_size(batch_size)
                .with_row_groups(vec![rg]);
            if let Some(p) = proj {
                b = b.with_projection(p);
            }
            let stream = b.build()?;
            stream.try_collect::<Vec<RecordBatch>>().await
        }
    });

    let per_rg_batches: Vec<Vec<RecordBatch>> = futures::stream::iter(per_rg)
        .buffered(rg_concurrency())
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
