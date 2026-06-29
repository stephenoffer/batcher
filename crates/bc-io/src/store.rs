//! Resolve a URI to an `object_store` backend + in-store path, for every scheme the
//! engine reads: `s3://` (and on-prem S3 like MinIO/Ceph via an endpoint override),
//! `gs://`/`gcs://`, `az://`/`abfs://`/`abfss://`, `http(s)://`, and a bare local path.
//!
//! One façade (`resolve`) so the parquet reader is storage-agnostic — the same code
//! decodes a row-group whether it lives on S3 or local disk. Credentials, region, and
//! endpoint come from the URI query string (`?region=…&endpoint=…&anonymous=true`)
//! merged over the process environment (`AWS_REGION`, `AWS_ENDPOINT_URL`, …), so the
//! Rust reader honors the same configuration the Python filesystem façade does.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use object_store::ObjectStore;
use url::Url;

use crate::IoError;

/// A resolved backend: the object store plus the object's path within it.
pub(crate) struct Resolved {
    pub store: Arc<dyn ObjectStore>,
    pub path: object_store::path::Path,
}

/// Process-wide cache of built object stores, keyed by `(scheme, host, sorted-options)`.
/// Building an S3 store resolves the credential chain — for an instance role that is an
/// HTTP round-trip to the metadata service — and opens a connection pool, so doing it on
/// every read (one per row-group split) dominated the read time. The store is `Send +
/// Sync` and pools connections internally, so caching + sharing it is both correct and
/// the throughput fix: subsequent reads reuse the warm client and its connections.
fn store_cache() -> &'static Mutex<HashMap<String, Arc<dyn ObjectStore>>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<dyn ObjectStore>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Build the object store + path for `uri`. A bare path (no `scheme://`) is local.
pub(crate) fn resolve(uri: &str) -> Result<Resolved, IoError> {
    if !uri.contains("://") {
        // Local filesystem: object_store's LocalFileSystem keys off an absolute path.
        let abs = std::fs::canonicalize(uri).unwrap_or_else(|_| std::path::PathBuf::from(uri));
        let path = object_store::path::Path::from_filesystem_path(&abs)
            .map_err(|e| IoError::Store(e.to_string()))?;
        let store = cached_store("file::local", || {
            Ok(Arc::new(object_store::local::LocalFileSystem::new()) as Arc<dyn ObjectStore>)
        })?;
        return Ok(Resolved { store, path });
    }

    let url = Url::parse(uri).map_err(|e| IoError::Uri(format!("{uri}: {e}")))?;
    let opts = store_options(&url);
    // The path within the store is per-object (not cached); the store (connection pool +
    // resolved credentials) is keyed by scheme+host+options so it is built once and reused.
    let key = format!(
        "{}::{}::{:?}",
        url.scheme(),
        url.host_str().unwrap_or(""),
        opts
    );
    let url2 = url.clone();
    let store = cached_store(&key, move || {
        // `parse_url_opts` dispatches on the scheme and threads region/endpoint/credential
        // options into the builder. We discard its path (we re-derive it per call below).
        let (store, _path) = object_store::parse_url_opts(&url2, opts.clone())
            .map_err(|e| IoError::Store(format!("{url2}: {e}")))?;
        Ok(Arc::from(store))
    })?;
    // Re-derive just the object path (cheap, no I/O) from the URL the cached store covers.
    let (_s, path) =
        object_store::parse_url(&url).map_err(|e| IoError::Store(format!("{uri}: {e}")))?;
    Ok(Resolved { store, path })
}

/// Get a cached store for `key`, building it with `build` on first sight.
fn cached_store(
    key: &str,
    build: impl FnOnce() -> Result<Arc<dyn ObjectStore>, IoError>,
) -> Result<Arc<dyn ObjectStore>, IoError> {
    if let Some(s) = store_cache().lock().unwrap().get(key) {
        return Ok(Arc::clone(s));
    }
    let store = build()?;
    store_cache()
        .lock()
        .unwrap()
        .insert(key.to_string(), Arc::clone(&store));
    Ok(store)
}

/// Config options (key, value) for the object-store builder, drawn from the URI query
/// string and the environment. Keys are object_store's generic config keys (e.g.
/// `aws_region`, `aws_endpoint`, `aws_skip_signature`); unknown keys are ignored by the
/// builder, so passing a superset is safe.
fn store_options(url: &Url) -> Vec<(String, String)> {
    let mut opts: Vec<(String, String)> = Vec::new();
    let scheme = url.scheme();

    // Environment defaults (object_store's `parse_url_opts` does not read env itself).
    if matches!(scheme, "s3" | "s3a") {
        if let Ok(r) = std::env::var("AWS_REGION").or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
        {
            opts.push(("aws_region".into(), r));
        }
        if let Ok(e) = std::env::var("AWS_ENDPOINT_URL").or_else(|_| std::env::var("AWS_ENDPOINT"))
        {
            opts.push(("aws_endpoint".into(), e));
        }
        // Allow virtual-hosted-style off for path-style endpoints (MinIO/Ceph).
        if std::env::var("AWS_ALLOW_HTTP").is_ok() {
            opts.push(("aws_allow_http".into(), "true".into()));
        }
    }

    // Query-string overrides win over env. Map the friendly names the Python façade
    // accepts (`endpoint_override`, `region`, `anonymous`) onto object_store keys.
    for (k, v) in url.query_pairs() {
        let (k, v) = (k.to_string(), v.to_string());
        match k.as_str() {
            "region" => opts.push(("aws_region".into(), v)),
            "endpoint" | "endpoint_override" => {
                opts.push(("aws_endpoint".into(), v.clone()));
                opts.push(("aws_allow_http".into(), "true".into()));
            }
            "anonymous" | "skip_signature" => opts.push(("aws_skip_signature".into(), v)),
            "allow_http" => opts.push(("aws_allow_http".into(), v)),
            other => opts.push((other.to_string(), v)),
        }
    }
    opts
}
