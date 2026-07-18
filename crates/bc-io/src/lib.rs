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
    // A `ProjectionMask` is a *set* of leaf indices — it selects columns but does NOT
    // reorder them, so the decoded batch keeps the file's column order. PyArrow's
    // `read_table(columns=[...])` (the fallback this reader must be byte-identical to)
    // returns the columns in the *requested* order, so a reordered projection
    // (`["c","a"]` on a file laid out `a,b,c`) would otherwise come back as `[a,c]`
    // here but `[c,a]` from PyArrow — a silent column-order divergence. We reorder the
    // output below to the requested order to honor that contract.
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
        let rows: usize = out.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 2, "unsigned UInt32 group must be kept (3e9 >= 2e9)");
        // A predicate the unsigned range truly excludes still prunes.
        let gt = r#"{"node":"cmp","col":"u","op":"gt","lit":4000000000}"#;
        let none = read_parquet_filtered(p32.to_str().unwrap(), &[], None, 4096, gt).unwrap();
        assert_eq!(none.iter().map(|b| b.num_rows()).sum::<usize>(), 0);

        // UInt64 with a value above i64::MAX.
        let p64 = dir.join("u64.parquet");
        let schema64 = Arc::new(Schema::new(vec![Field::new("u", DataType::UInt64, false)]));
        let big = 10_000_000_000_000_000_000u64; // > i64::MAX
        let u64b = RecordBatch::try_new(
            schema64,
            vec![Arc::new(UInt64Array::from(vec![5u64, big]))],
        )
        .unwrap();
        write_parquet(&p64, &[u64b], 1000);
        // `u >= 9e18` (< i64::MAX) must keep the group — `big` matches.
        let ge64 = r#"{"node":"cmp","col":"u","op":"ge","lit":9000000000000000000}"#;
        let out64 = read_parquet_filtered(p64.to_str().unwrap(), &[], None, 4096, ge64).unwrap();
        assert_eq!(
            out64.iter().map(|b| b.num_rows()).sum::<usize>(),
            2,
            "unsigned UInt64 group must be kept"
        );
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
