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
    /// Whether reads go over the network (`s3://`, `gs://`, `az://`, `http(s)://`) rather
    /// than to a local file.
    ///
    /// The distinction is a *throughput* one, and it is what [`crate::split_read`] keys off.
    /// A remote GET is limited to roughly one connection's bandwidth however large the range
    /// is, so a big read has to be split across several to go fast; a local read is served by
    /// the page cache at memory speed and splitting it only adds syscalls.
    pub remote: bool,
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
        return Ok(Resolved {
            store,
            path,
            remote: false,
        });
    }

    let url = Url::parse(uri).map_err(|e| IoError::Uri(format!("{uri}: {e}")))?;
    let mut opts = store_options(&url);
    resolve_s3_region(&url, &mut opts);
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
    // Re-derive just the object path from the URL the cached store covers.
    //
    // This asks `ObjectStoreScheme::parse` and NOT `object_store::parse_url`, even though the
    // latter reads as the obvious counterpart to the `parse_url_opts` above and returns the
    // same `Path`. `parse_url` **builds an entire second store** and drops it — for `s3://`
    // that means an `AmazonS3Builder::build()`, whose `reqwest` client loads and parses the
    // system root-certificate bundle. Measured on this bucket: **~83 ms per call**, against
    // ~3 us for the path alone. Paid once per file read, it was ~85 s of CPU across a
    // 1,024-file scan — the single largest term in it, and invisible as anything but "the
    // reader is slow", because the store it built was never used for anything.
    //
    // `parse_url_opts` derives its `Path` by calling exactly this function (then a
    // `Path::parse` that is idempotent over an already-parsed path), so the result is
    // identical for every scheme — including the bucket-stripping the HTTPS forms of S3 and
    // Azure need, which a plain `Path::from_url_path(url.path())` would get wrong.
    let (scheme, path) = object_store::ObjectStoreScheme::parse(&url)
        .map_err(|e| IoError::Store(format!("{uri}: {e}")))?;
    // `file://` reaches here as a URL but is still a local read, so it must not be split.
    let remote = !matches!(scheme, object_store::ObjectStoreScheme::Local);
    Ok(Resolved {
        store,
        path,
        remote,
    })
}

/// The bucket whose region needs looking up, or `None` when it must not be asked for.
///
/// Pure, and separated from the lookup so the guards can be tested without a network: which
/// URLs skip the `HeadBucket` is the whole safety argument, and every one of them (a non-S3
/// scheme, an explicit region, a custom endpoint) is a case where the call is either wrong or
/// pointless.
fn region_lookup_bucket<'a>(url: &'a Url, opts: &[(String, String)]) -> Option<&'a str> {
    if url.scheme() != "s3" && url.scheme() != "s3a" {
        return None;
    }
    let has = |k: &str| opts.iter().any(|(name, _)| name == k);
    // An explicit region needs no lookup. A custom endpoint (MinIO, Ceph, a test double) is
    // not AWS: `HeadBucket` against `<bucket>.s3.amazonaws.com` would query a bucket that is
    // not the one being read, and the cross-region redirect it exists to avoid cannot happen.
    if has("aws_region") || has("aws_endpoint") {
        return None;
    }
    url.host_str()
}

/// Fill in an S3 bucket's region when nothing configured one, by asking S3.
///
/// **This was the single largest term in the scan benchmark, and it is not an engine cost at
/// all.** With no region, `object_store` signs for `us-east-1`; a bucket living anywhere else
/// then answers every single request with a redirect, so each GET costs two round trips and a
/// re-sign. Measured on `scan-sum1-many_small` (1,024 objects in `us-west-2`): **5,369 ms with
/// no region set against 281.7 ms with `AWS_REGION=us-west-2`** — the same code, a 19x
/// difference, and the reason Batcher looked 7-9x behind DuckDB on that suite. DuckDB's own
/// adapter issues `SET s3_region=...`, and PyArrow's `S3FileSystem` resolves the bucket's
/// region for itself, so both were quietly reading a correctly-addressed bucket while Batcher
/// redirected on every object.
///
/// `resolve_bucket_region` is one `HeadBucket` call, cached here per bucket for the life of the
/// process, so a whole scan pays it once. It runs only when the region is genuinely absent —
/// an explicit `?region=`, `AWS_REGION`, or a custom `endpoint` (MinIO/Ceph, where the call
/// would be meaningless and the redirect cannot happen) all skip it. A failure is ignored
/// rather than raised: the previous behaviour was to sign for `us-east-1` and it must remain
/// the fallback, since a bucket that really is there is better served slowly than not at all.
fn resolve_s3_region(url: &Url, opts: &mut Vec<(String, String)>) {
    let Some(bucket) = region_lookup_bucket(url, opts) else {
        return;
    };
    let bucket = bucket.to_string();
    let bucket = bucket.as_str();
    static CACHE: OnceLock<Mutex<HashMap<String, Option<String>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    // Held across the lookup for the same single-flight reason `cached_store` is: the callers
    // fan out hundreds of tasks at once, and without it every one of them would issue its own
    // `HeadBucket` for the same bucket.
    let mut guard = cache.lock().unwrap();
    let region = guard
        .entry(bucket.to_string())
        .or_insert_with(|| head_bucket_region(bucket));
    if let Some(region) = region {
        opts.push(("aws_region".into(), region.clone()));
    }
}

/// One `HeadBucket` for `bucket`'s region, on a thread of its own.
///
/// Its own thread, and its own single-thread runtime, because `resolve` is reached from
/// *inside* the shared runtime's worker tasks (`read_parquet_async`, `load_metadata_many`) and
/// `Runtime::block_on` panics when called from within a runtime context. Handing the await to
/// a plain OS thread and joining it is the one form that is correct from both a sync caller
/// and an async one. It runs at most once per bucket per process.
fn head_bucket_region(bucket: &str) -> Option<String> {
    let bucket = bucket.to_string();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .ok()?;
        rt.block_on(object_store::aws::resolve_bucket_region(
            &bucket,
            &Default::default(),
        ))
        .ok()
    })
    .join()
    .ok()
    .flatten()
}

/// Get a cached store for `key`, building it with `build` **once** on first sight.
///
/// The build runs under the lock, which is the whole point: it is single-flight, not merely
/// cached. Releasing the lock to build let every task that missed a cold cache build its own
/// store, and the callers here miss it *simultaneously* — `load_metadata_many` and
/// `read_parquet_many` both fan out `file_concurrency()` tasks at once, so a cold read of a
/// directory built hundreds of S3 clients where it needed one. Each build resolves the
/// credential chain and loads the system root-certificate bundle (~83 ms; see `resolve`), so
/// the herd cost seconds and — the tell — got *worse* as concurrency rose. Measured cold over
/// 1,024 S3 footers: 3.4 s at 8-way, 5.9 s at 64-way, 5.5 s at 384-way, against a warm sweep
/// that scales the right way (12.0 s -> 158 ms over the same range).
///
/// Serializing the build costs nothing real. There is one store per `(scheme, host, options)`
/// and a process has a handful, so the lock is uncontended after the first read of each; what
/// used to be N redundant builds is now one build and N-1 waits on its result.
fn cached_store(
    key: &str,
    build: impl FnOnce() -> Result<Arc<dyn ObjectStore>, IoError>,
) -> Result<Arc<dyn ObjectStore>, IoError> {
    let mut cache = store_cache().lock().unwrap();
    if let Some(s) = cache.get(key) {
        return Ok(Arc::clone(s));
    }
    // Held across `build` deliberately (see above). `build` is `object_store::parse_url_opts`
    // — synchronous, and needing neither this cache nor the async runtime — so it cannot
    // re-enter and cannot deadlock.
    let store = build()?;
    cache.insert(key.to_string(), Arc::clone(&store));
    Ok(store)
}

/// Config options (key, value) for the object-store builder, drawn from the URI query
/// string and the environment. Keys are object_store's generic config keys (e.g.
/// `aws_region`, `aws_endpoint`, `aws_skip_signature`); unknown keys are ignored by the
/// builder, so passing a superset is safe.
fn store_options(url: &Url) -> Vec<(String, String)> {
    let mut opts: Vec<(String, String)> = Vec::new();
    let scheme = url.scheme();
    // Set from `AWS_ALLOW_HTTP` by *value*, not by presence: `AWS_ALLOW_HTTP=false` had been
    // switching plain HTTP on, which is the opposite of what the operator wrote and the kind
    // of setting nobody re-reads once it is set. Applied after the borrow of `opts` the
    // environment reader holds.
    let mut allow_http = false;

    // Environment defaults (object_store's `parse_url_opts` does not read env itself).
    //
    // Instance/workload identity — IRSA, IMDS, ECS task roles, GCE and Azure metadata —
    // is resolved inside the builder and needs nothing here. STATIC credentials in the
    // environment are the gap: object_store's builders start from `new()`, not
    // `from_env()`, so without this a MinIO/Ceph deployment with `AWS_ACCESS_KEY_ID` set
    // (the on-prem norm, where there is no metadata service to fall back on) fails to
    // authenticate. The scan then silently degrades to the slower pyarrow path — correct
    // results, quietly worse, with no diagnostic — which is exactly the kind of failure
    // that never gets reported as a bug.
    let mut env_opt = |key: &str, vars: &[&str]| {
        for var in vars {
            if let Ok(v) = std::env::var(var) {
                if !v.is_empty() {
                    opts.push((key.into(), v));
                    return;
                }
            }
        }
    };
    match scheme {
        "s3" | "s3a" => {
            env_opt("aws_region", &["AWS_REGION", "AWS_DEFAULT_REGION"]);
            env_opt("aws_endpoint", &["AWS_ENDPOINT_URL", "AWS_ENDPOINT"]);
            env_opt("aws_access_key_id", &["AWS_ACCESS_KEY_ID"]);
            env_opt("aws_secret_access_key", &["AWS_SECRET_ACCESS_KEY"]);
            env_opt("aws_session_token", &["AWS_SESSION_TOKEN"]);
            // Plain HTTP for an on-prem endpoint. Read by *value*, not by presence:
            // `AWS_ALLOW_HTTP=false` had been switching it on, which is the opposite of what
            // the operator wrote and the kind of setting nobody re-reads once it is set.
            // Path-style addressing, which most on-prem and GPU-cloud S3 endpoints require.
            env_opt(
                "aws_virtual_hosted_style_request",
                &["AWS_VIRTUAL_HOSTED_STYLE_REQUEST"],
            );
            allow_http = std::env::var("AWS_ALLOW_HTTP")
                .map(|v| is_truthy(&v))
                .unwrap_or_else(|_| {
                    // An `http://` endpoint from the environment implies plain HTTP, the same
                    // way one written into the query string already does. Without this the
                    // asymmetry bites exactly the deployment the endpoint variable exists for:
                    // a MinIO or Ceph gateway on an internal network, reached over HTTP
                    // because there is no certificate for `minio.internal`. `object_store`
                    // refuses the connection, the reader is unusable, and the scan degrades
                    // to the slower PyArrow path with correct results and no diagnostic.
                    //
                    // Only when the operator has not said otherwise: an explicit
                    // `AWS_ALLOW_HTTP=false` against an `http://` endpoint is a contradiction,
                    // and the explicit half wins.
                    std::env::var("AWS_ENDPOINT_URL")
                        .or_else(|_| std::env::var("AWS_ENDPOINT"))
                        .map(|e| e.trim_start().to_ascii_lowercase().starts_with("http://"))
                        .unwrap_or(false)
                });
        }
        "gs" | "gcs" => {
            env_opt(
                "google_service_account",
                &["GOOGLE_SERVICE_ACCOUNT", "GOOGLE_APPLICATION_CREDENTIALS"],
            );
            env_opt(
                "google_service_account_key",
                &["GOOGLE_SERVICE_ACCOUNT_KEY"],
            );
        }
        "az" | "azure" | "abfs" | "abfss" | "wasb" | "wasbs" => {
            env_opt(
                "azure_storage_account_name",
                &["AZURE_STORAGE_ACCOUNT_NAME"],
            );
            env_opt("azure_storage_account_key", &["AZURE_STORAGE_ACCOUNT_KEY"]);
            env_opt("azure_storage_sas_key", &["AZURE_STORAGE_SAS_TOKEN"]);
            env_opt("azure_storage_client_id", &["AZURE_CLIENT_ID"]);
            env_opt("azure_storage_client_secret", &["AZURE_CLIENT_SECRET"]);
            env_opt("azure_storage_tenant_id", &["AZURE_TENANT_ID"]);
        }
        _ => {}
    }

    if allow_http {
        opts.push(("aws_allow_http".into(), "true".into()));
    }

    // Query-string overrides win over env, and every friendly name the Python façade accepts
    // is translated here.
    //
    // **A name this does not translate is silently dropped**, because `object_store`'s
    // builders ignore config keys they do not recognize. That is the divergence worth naming:
    // the same URI then authenticates through PyArrow and not through this reader, or reaches
    // one endpoint through one and another through the other — and since an unusable native
    // reader falls back to PyArrow, the symptom is a scan that is quietly slower rather than
    // one that fails. Credentials (`access_key`), addressing style
    // (`force_virtual_addressing`) and the timeouts were all being dropped that way.
    // Only an S3-family URL gets the `aws_*` translations. The env reader above is already
    // per-scheme and its test says why: an `aws_*` key on a GCS store is not merely useless,
    // it makes the cache key differ for no reason, so two identical stores are built and
    // credential resolution runs twice. The query path had been translating regardless.
    let s3_family = matches!(scheme, "s3" | "s3a");
    for (k, v) in url.query_pairs() {
        let (k, v) = (k.to_string(), v.to_string());
        if !s3_family {
            // A non-S3 scheme keeps whatever the caller wrote; `object_store` ignores a key
            // it does not know, which is the same outcome as translating it wrongly minus the
            // spurious cache entry.
            opts.push((k, v));
            continue;
        }
        match k.as_str() {
            "region" => opts.push(("aws_region".into(), v)),
            "endpoint" | "endpoint_override" => {
                opts.push(("aws_endpoint".into(), v.clone()));
                opts.push(("aws_allow_http".into(), "true".into()));
            }
            "anonymous" | "skip_signature" => opts.push(("aws_skip_signature".into(), v)),
            "allow_http" => opts.push(("aws_allow_http".into(), v)),
            // PyArrow's spelling of the credential triple.
            "access_key" => opts.push(("aws_access_key_id".into(), v)),
            "secret_key" => opts.push(("aws_secret_access_key".into(), v)),
            "session_token" => opts.push(("aws_session_token".into(), v)),
            // PyArrow inverts the sense: `force_virtual_addressing=false` means path style.
            "force_virtual_addressing" => opts.push((
                "aws_virtual_hosted_style_request".into(),
                if is_truthy(&v) {
                    "true".into()
                } else {
                    "false".into()
                },
            )),
            // PyArrow takes whole seconds; `object_store` parses a humantime duration, so a
            // bare number would be rejected and the option lost.
            "connect_timeout" => opts.push(("connect_timeout".into(), as_duration(&v))),
            "request_timeout" | "timeout" => opts.push(("timeout".into(), as_duration(&v))),
            other => opts.push((other.to_string(), v)),
        }
    }
    opts
}

/// Whether a configuration string means "on". Accepts the spellings an operator writes in a
/// URI or an environment variable, so `1`, `true`, `yes` and `on` all agree.
fn is_truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// A timeout as `object_store` parses it. A bare number is read as seconds — the unit PyArrow
/// uses — and anything already carrying a unit suffix is passed through untouched.
fn as_duration(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.chars().all(|c| c.is_ascii_digit()) && !trimmed.is_empty() {
        format!("{trimmed}s")
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts_for(uri: &str) -> HashMap<String, String> {
        store_options(&Url::parse(uri).unwrap())
            .into_iter()
            .collect()
    }

    /// Serializes the tests that write process-global environment variables.
    ///
    /// The environment is shared by every thread cargo runs a test on, so two of these
    /// interleave: one sets `AWS_ENDPOINT_URL` and the other removes it in its cleanup, and
    /// the first then reads an option that is not there. That is a *flake*, which is worse
    /// than a failure — it passes on a rerun, so it teaches whoever hits it to rerun.
    ///
    /// A plain `Mutex` rather than a crate: this is the only place in the crate that needs
    /// one. A poisoned lock is recovered from rather than propagated, since a panic in one
    /// env test has already been reported and must not turn every later one into a second
    /// failure pointing at the wrong test.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The region lookup must fire for a plain `s3://` URL and for nothing else.
    ///
    /// The `HeadBucket` is one request against `<bucket>.s3.amazonaws.com`, so the cases that
    /// must skip it are the ones where that host is the wrong thing to ask: a non-AWS scheme,
    /// and above all a **custom endpoint** — MinIO, Ceph, or a test double, where the answer
    /// would describe a real AWS bucket of the same name rather than the one being read, and
    /// where the cross-region redirect the lookup exists to avoid cannot occur anyway.
    #[test]
    fn the_region_lookup_fires_only_where_it_is_meaningful() {
        let bucket_of = |uri: &str| {
            let url = Url::parse(uri).unwrap();
            let opts = store_options(&url);
            region_lookup_bucket(&url, &opts).map(str::to_string)
        };
        let _g = env_guard();
        // Guard against an AWS_REGION in the ambient environment answering for the plain case.
        let saved = std::env::var("AWS_REGION").ok();
        let saved_default = std::env::var("AWS_DEFAULT_REGION").ok();
        unsafe {
            std::env::remove_var("AWS_REGION");
            std::env::remove_var("AWS_DEFAULT_REGION");
        }

        assert_eq!(
            bucket_of("s3://my-bucket/key.parquet").as_deref(),
            Some("my-bucket"),
            "a plain s3:// URL with no region anywhere is exactly the case that redirects"
        );
        assert_eq!(
            bucket_of("s3://my-bucket/key.parquet?region=eu-west-1"),
            None,
            "an explicit region needs no lookup"
        );
        assert_eq!(
            bucket_of("s3://my-bucket/key.parquet?endpoint=http://minio.local:9000"),
            None,
            "a custom endpoint is not AWS: asking s3.amazonaws.com would be a wrong answer"
        );
        assert_eq!(
            bucket_of("gs://my-bucket/key.parquet"),
            None,
            "only the S3 schemes"
        );
        assert_eq!(bucket_of("https://example.com/key.parquet"), None);

        unsafe {
            if let Some(v) = saved {
                std::env::set_var("AWS_REGION", v);
            }
            if let Some(v) = saved_default {
                std::env::set_var("AWS_DEFAULT_REGION", v);
            }
        }
    }

    /// An `AWS_REGION` in the environment must suppress the lookup too — it is the ordinary
    /// way a deployment configures this, and paying a `HeadBucket` per process on top of it
    /// would be pure waste.
    #[test]
    fn an_environment_region_suppresses_the_lookup() {
        let _g = env_guard();
        let saved = std::env::var("AWS_REGION").ok();
        unsafe { std::env::set_var("AWS_REGION", "ap-south-1") };
        let url = Url::parse("s3://my-bucket/key.parquet").unwrap();
        let opts = store_options(&url);
        assert_eq!(region_lookup_bucket(&url, &opts), None);
        unsafe {
            match saved {
                Some(v) => std::env::set_var("AWS_REGION", v),
                None => std::env::remove_var("AWS_REGION"),
            }
        }
    }

    /// A local path must resolve as *not* remote, in both spellings.
    ///
    /// `split_read` keys its request splitting off this flag, and a local file misclassified
    /// as remote would be cut into concurrent range reads the page cache gains nothing from.
    /// The `file://` form is the one worth pinning: it arrives as a URL and so takes the
    /// scheme branch, where every other scheme is remote.
    #[test]
    fn a_local_path_is_not_remote() {
        let dir = std::env::temp_dir();
        let bare = resolve(dir.to_str().unwrap()).expect("bare path resolves");
        assert!(!bare.remote, "a bare filesystem path is local");
        let url = format!("file://{}", dir.to_str().unwrap());
        let via_url = resolve(&url).expect("file:// resolves");
        assert!(!via_url.remote, "file:// is local, not remote");
    }

    /// Every network scheme the reader accepts must classify as remote, so its large reads
    /// are split across connections.
    #[test]
    fn network_schemes_are_remote() {
        let _g = env_guard();
        for uri in [
            "s3://bucket/key.parquet?region=us-east-1&anonymous=true",
            "gs://bucket/key.parquet",
            "https://example.com/key.parquet",
        ] {
            let resolved = resolve(uri).unwrap_or_else(|e| panic!("{uri}: {e}"));
            assert!(resolved.remote, "{uri} should be remote");
        }
    }

    /// Static credentials in the environment must reach the builder for every cloud, not
    /// just AWS. Without this the on-prem case (MinIO/Ceph with `AWS_ACCESS_KEY_ID`, where
    /// there is no metadata service to fall back on) fails to authenticate and the scan
    /// silently degrades to the slower pyarrow path instead of reporting anything.
    ///
    /// One test rather than several: these env vars are process-global, so splitting this
    /// across functions would race under cargo's parallel test threads.
    #[test]
    fn env_credentials_reach_every_scheme() {
        let _guard = env_guard();
        std::env::set_var("AWS_ACCESS_KEY_ID", "AK");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "SK");
        std::env::set_var("AWS_ENDPOINT_URL", "http://minio.internal:9000");
        std::env::set_var("AZURE_STORAGE_ACCOUNT_KEY", "AZKEY");
        std::env::set_var("GOOGLE_SERVICE_ACCOUNT", "/creds/sa.json");

        let s3 = opts_for("s3://bucket/key.parquet");
        assert_eq!(s3.get("aws_access_key_id").map(String::as_str), Some("AK"));
        assert_eq!(
            s3.get("aws_secret_access_key").map(String::as_str),
            Some("SK")
        );
        assert_eq!(
            s3.get("aws_endpoint").map(String::as_str),
            Some("http://minio.internal:9000")
        );

        // Each scheme gets only its own vendor's keys — an `aws_*` key on a GCS store is
        // not merely useless, it makes the cache key differ for no reason.
        let gs = opts_for("gs://bucket/key.parquet");
        assert_eq!(
            gs.get("google_service_account").map(String::as_str),
            Some("/creds/sa.json")
        );
        assert!(!gs.contains_key("aws_access_key_id"));

        let az = opts_for("abfs://c@a.dfs.core.windows.net/key.parquet");
        assert_eq!(
            az.get("azure_storage_account_key").map(String::as_str),
            Some("AZKEY")
        );
        assert!(!az.contains_key("aws_access_key_id"));

        // A query-string option still wins over the environment.
        let overridden = opts_for("s3://bucket/key.parquet?region=eu-west-1");
        assert_eq!(
            overridden.get("aws_region").map(String::as_str),
            Some("eu-west-1")
        );

        // An `http://` endpoint from the environment implies plain HTTP, the same way one in
        // the query string does. Without it `object_store` refuses the connection to an
        // internal MinIO or Ceph gateway, the reader is unusable, and the scan degrades to
        // PyArrow with correct results and nothing to say why.
        assert_eq!(s3.get("aws_allow_http").map(String::as_str), Some("true"));

        // An https endpoint implies nothing, because it does not need to.
        std::env::set_var("AWS_ENDPOINT_URL", "https://object.example.net");
        assert!(!opts_for("s3://bucket/key.parquet").contains_key("aws_allow_http"));

        // And an operator who said no outranks the inference: an explicit `false` against an
        // `http://` endpoint is a contradiction, and the explicit half wins.
        std::env::set_var("AWS_ENDPOINT_URL", "http://minio.internal:9000");
        std::env::set_var("AWS_ALLOW_HTTP", "false");
        assert!(!opts_for("s3://bucket/key.parquet").contains_key("aws_allow_http"));
        std::env::remove_var("AWS_ALLOW_HTTP");

        for v in [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_ENDPOINT_URL",
            "AWS_ALLOW_HTTP",
            "AZURE_STORAGE_ACCOUNT_KEY",
            "GOOGLE_SERVICE_ACCOUNT",
        ] {
            std::env::remove_var(v);
        }
    }
    /// Every friendly name the Python façade accepts must reach the builder here too.
    ///
    /// `object_store` ignores a config key it does not recognize, so a name this reader fails
    /// to translate is *silently dropped* — and because an unusable native reader falls back
    /// to PyArrow, the symptom is a scan that is quietly slower, or one that reaches a
    /// different endpoint through each of the two readers. Credentials, addressing style and
    /// the timeouts were all being dropped that way.
    #[test]
    fn the_python_facade_option_names_are_translated() {
        let opts = opts_for(
            "s3://bucket/key.parquet?access_key=AK&secret_key=SK&session_token=TOK\
             &force_virtual_addressing=false&connect_timeout=5&request_timeout=90",
        );
        assert_eq!(
            opts.get("aws_access_key_id").map(String::as_str),
            Some("AK")
        );
        assert_eq!(
            opts.get("aws_secret_access_key").map(String::as_str),
            Some("SK")
        );
        assert_eq!(
            opts.get("aws_session_token").map(String::as_str),
            Some("TOK")
        );
        // PyArrow inverts the sense of this one: "not virtual hosted" is path style.
        assert_eq!(
            opts.get("aws_virtual_hosted_style_request")
                .map(String::as_str),
            Some("false")
        );
        // PyArrow takes whole seconds; object_store parses a humantime duration, so a bare
        // number would be rejected and the option lost.
        assert_eq!(opts.get("connect_timeout").map(String::as_str), Some("5s"));
        assert_eq!(opts.get("timeout").map(String::as_str), Some("90s"));
    }

    /// An `aws_*` key on a GCS or Azure store is not merely useless: it changes the cache key,
    /// so two identical stores are built and the credential chain resolves twice. The
    /// environment reader was already per-scheme; the query-string path was not.
    #[test]
    fn the_aws_translations_do_not_leak_onto_other_schemes() {
        let gs = opts_for("gs://bucket/key.parquet?region=eu-west-1&access_key=AK");
        assert!(!gs.contains_key("aws_region"));
        assert!(!gs.contains_key("aws_access_key_id"));
        // The caller's own spelling survives, and object_store ignores what it does not know.
        assert_eq!(gs.get("region").map(String::as_str), Some("eu-west-1"));

        let s3 = opts_for("s3://bucket/key.parquet?region=eu-west-1");
        assert_eq!(s3.get("aws_region").map(String::as_str), Some("eu-west-1"));
    }

    /// A duration that already carries a unit is passed through rather than re-suffixed.
    #[test]
    fn a_duration_with_a_unit_survives() {
        let opts = opts_for("s3://bucket/key.parquet?request_timeout=500ms");
        assert_eq!(opts.get("timeout").map(String::as_str), Some("500ms"));
    }

    /// Addressing style is a three-way answer: path, virtual-hosted, or unstated.
    #[test]
    fn addressing_style_is_only_set_when_asked_for() {
        assert_eq!(
            opts_for("s3://b/k?force_virtual_addressing=true")
                .get("aws_virtual_hosted_style_request")
                .map(String::as_str),
            Some("true")
        );
        assert!(!opts_for("s3://b/k").contains_key("aws_virtual_hosted_style_request"));
    }

    /// The in-store path must be exactly what `object_store::parse_url` would have returned.
    ///
    /// `resolve` derives it with `ObjectStoreScheme::parse` instead, because `parse_url`
    /// builds and discards a whole store to do it (~83 ms per call on S3, once per file read).
    /// The two agree by construction — `parse_url_opts` calls this same function — and this
    /// test is what keeps "make the path derivation cheap" from quietly becoming "make it
    /// wrong". The bucket-stripping cases are the ones a naive `url.path()` gets wrong: they
    /// address S3 and Azure through an HTTPS host, where the first path segment is the
    /// *bucket* and belongs to the store, not to the object.
    #[test]
    fn the_object_path_matches_what_parse_url_derives() {
        for (uri, expected) in [
            ("s3://bucket/a/b.parquet", "a/b.parquet"),
            ("s3a://bucket/a/b.parquet", "a/b.parquet"),
            ("gs://bucket/a/b.parquet", "a/b.parquet"),
            ("abfss://fs@acct.dfs.core.windows.net/a/b", "a/b"),
            ("az://container/a/b.parquet", "b.parquet"), // container is stripped
            ("http://host/a/b.parquet", "a/b.parquet"),
            ("https://host.example/a/b.parquet", "a/b.parquet"),
            // Path-style S3 over HTTPS: the leading segment is the bucket.
            (
                "https://s3.us-east-1.amazonaws.com/bkt/a/b.parquet",
                "a/b.parquet",
            ),
            // Virtual-hosted S3 over HTTPS: the bucket is in the host, so nothing is stripped.
            (
                "https://bkt.s3.us-east-1.amazonaws.com/a/b.parquet",
                "a/b.parquet",
            ),
            // A space and a percent-escape must decode to one unescaped object name.
            ("s3://bucket/a%20b/c.parquet", "a b/c.parquet"),
        ] {
            let url = Url::parse(uri).unwrap();
            let (_scheme, path) = object_store::ObjectStoreScheme::parse(&url).unwrap();
            assert_eq!(path.as_ref(), expected, "path for {uri}");
        }
    }

    /// `AWS_ALLOW_HTTP=false` must not switch plain HTTP *on*. It had, because the variable
    /// was read by presence — the opposite of what the operator wrote, in a setting nobody
    /// re-reads once it is set.
    #[test]
    fn allow_http_is_read_by_value_not_by_presence() {
        let _guard = env_guard();
        std::env::set_var("AWS_ALLOW_HTTP", "false");
        assert!(!opts_for("s3://bucket/key.parquet").contains_key("aws_allow_http"));
        std::env::set_var("AWS_ALLOW_HTTP", "1");
        assert_eq!(
            opts_for("s3://bucket/key.parquet")
                .get("aws_allow_http")
                .map(String::as_str),
            Some("true")
        );
        std::env::remove_var("AWS_ALLOW_HTTP");
    }
}
