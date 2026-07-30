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
                .unwrap_or(false);
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

    /// Static credentials in the environment must reach the builder for every cloud, not
    /// just AWS. Without this the on-prem case (MinIO/Ceph with `AWS_ACCESS_KEY_ID`, where
    /// there is no metadata service to fall back on) fails to authenticate and the scan
    /// silently degrades to the slower pyarrow path instead of reporting anything.
    ///
    /// One test rather than several: these env vars are process-global, so splitting this
    /// across functions would race under cargo's parallel test threads.
    #[test]
    fn env_credentials_reach_every_scheme() {
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

        for v in [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_ENDPOINT_URL",
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

    /// `AWS_ALLOW_HTTP=false` must not switch plain HTTP *on*. It had, because the variable
    /// was read by presence — the opposite of what the operator wrote, in a setting nobody
    /// re-reads once it is set.
    #[test]
    fn allow_http_is_read_by_value_not_by_presence() {
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
