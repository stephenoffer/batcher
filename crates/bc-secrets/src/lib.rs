//! Resolving a secret *reference* to secret material, on the machine that needs it.
//!
//! A reference — `env:NAME`, `file:PATH`, `cmd:NAME` — travels in the plan IR in place of
//! the secret, and is resolved here, in the data plane, against the executing machine's
//! own environment, mounted files, or secret-fetching helper. That is what lets a
//! distributed query use a key without ever putting the key on the wire.
//!
//! # Reaching an external key store
//!
//! There is deliberately **no AWS/GCP/Azure/Vault SDK in this crate**, and that is a
//! design decision rather than a gap. This crate sits below `bc-expr`, which every
//! consumer of the expression layer links; a cloud SDK would drag an async runtime and a
//! TLS stack into builds that never resolve a secret. Two dependency-free routes cover
//! the same ground:
//!
//! * **A file, via the platform's own secret delivery.** Vault Agent, the External
//!   Secrets Operator, and the Kubernetes secrets-store CSI driver all materialize a
//!   secret as a file. `file:/run/secrets/key` is then the whole integration, and the
//!   platform owns rotation, auth, and audit — which is where an operator wants them.
//! * **`cmd:NAME`, via a helper program.** Runs the operator-configured
//!   `BATCHER_SECRET_COMMAND` with `NAME` as its argument and takes stdout as the secret.
//!   One knob reaches `vault kv get`, `aws secretsmanager get-secret-value`,
//!   `gcloud secrets versions access`, `az keyvault secret show`, or a bespoke fetcher.
//!
//! **`cmd:` is inert unless the operator sets `BATCHER_SECRET_COMMAND`,** and the
//! reference supplies only the *argument*, never the program. That asymmetry is the whole
//! security story: a plan is data, it may arrive from somewhere less trusted than the
//! cluster, and letting it name a program to execute would turn a secret reference into
//! arbitrary code execution. The operator chooses the program; the plan chooses which
//! secret to ask that program for.
//!
//! A host that genuinely wants an in-process SDK can [`register_backend`] its own
//! [`SecretBackend`] under a new scheme without this crate taking the dependency.
//!
//! # Caching
//!
//! Resolution is cached with a TTL (default 300s, `BATCHER_SECRET_TTL_SECONDS`) because
//! callers resolve **per batch**: an uncached `cmd:` reference would fork a process per
//! array, and even `file:` would re-read the filesystem on every batch of a scan. The TTL
//! bounds how long a rotated secret stays stale; set it to `0` to disable caching.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, Instant};

/// Failure to resolve a secret reference.
///
/// Every variant names the **reference**, never the resolved value: the reference is not
/// secret and is exactly what an operator needs in order to fix a misconfiguration, while
/// the secret must never reach a log or an error message.
#[derive(Debug, thiserror::Error)]
pub enum SecretError {
    #[error("secret reference {reference} could not be resolved: {reason}")]
    Unresolved { reference: String, reason: String },
    #[error(
        "secret reference {reference} uses the `cmd:` scheme, but BATCHER_SECRET_COMMAND \
         is not set; an operator must configure the secret-fetching program (the reference \
         supplies only its argument, never the program itself)"
    )]
    CommandNotConfigured { reference: String },
}

/// A source of secret material for one reference scheme (`env`, `file`, `cmd`, ...).
///
/// Implement this to reach a store the built-ins do not cover, then [`register_backend`]
/// it. `fetch` receives the part of the reference after the `scheme:` prefix and returns
/// the secret; it must not log or otherwise emit the value it returns.
pub trait SecretBackend: Send + Sync {
    /// The scheme this backend answers for, without the colon (e.g. `"vault"`).
    fn scheme(&self) -> &str;
    /// Fetch the secret named by `target`, the reference minus its `scheme:` prefix.
    fn fetch(&self, target: &str) -> Result<String, SecretError>;
}

struct EnvBackend;
impl SecretBackend for EnvBackend {
    fn scheme(&self) -> &str {
        "env"
    }
    fn fetch(&self, target: &str) -> Result<String, SecretError> {
        std::env::var(target).map_err(|_| SecretError::Unresolved {
            reference: format!("env:{target}"),
            reason: "environment variable is not set".into(),
        })
    }
}

struct FileBackend;
impl SecretBackend for FileBackend {
    fn scheme(&self) -> &str {
        "file"
    }
    fn fetch(&self, target: &str) -> Result<String, SecretError> {
        // Trimmed: a mounted secret file almost always ends with a newline, and sending
        // that newline as part of a key is a baffling authentication failure.
        std::fs::read_to_string(target)
            .map(|s| s.trim().to_string())
            .map_err(|e| SecretError::Unresolved {
                reference: format!("file:{target}"),
                reason: e.to_string(),
            })
    }
}

/// Env var naming the operator-approved secret-fetching program for the `cmd:` scheme.
pub const SECRET_COMMAND_VAR: &str = "BATCHER_SECRET_COMMAND";

struct CommandBackend;
impl SecretBackend for CommandBackend {
    fn scheme(&self) -> &str {
        "cmd"
    }
    fn fetch(&self, target: &str) -> Result<String, SecretError> {
        let reference = format!("cmd:{target}");
        let program = std::env::var(SECRET_COMMAND_VAR)
            .ok()
            .filter(|p| !p.trim().is_empty())
            .ok_or_else(|| SecretError::CommandNotConfigured {
                reference: reference.clone(),
            })?;
        // The program comes from the operator's environment and the argument from the
        // reference — never a shell string built from both, so a reference cannot inject
        // a second command, a pipe, or a redirect.
        let out = std::process::Command::new(&program)
            .arg(target)
            .output()
            .map_err(|e| SecretError::Unresolved {
                reference: reference.clone(),
                reason: format!("could not run {program}: {e}"),
            })?;
        if !out.status.success() {
            // stderr is the helper's diagnostic channel; stdout is the secret and is
            // never quoted back.
            return Err(SecretError::Unresolved {
                reference,
                reason: format!(
                    "{program} exited with {}: {}",
                    out.status,
                    String::from_utf8_lossy(&out.stderr).trim()
                ),
            });
        }
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    }
}

fn registry() -> &'static RwLock<Vec<Arc<dyn SecretBackend>>> {
    static REGISTRY: OnceLock<RwLock<Vec<Arc<dyn SecretBackend>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        RwLock::new(vec![
            Arc::new(EnvBackend) as Arc<dyn SecretBackend>,
            Arc::new(FileBackend),
            Arc::new(CommandBackend),
        ])
    })
}

/// Register a backend for a new scheme (or override a built-in for the same scheme).
///
/// Later registrations win, so a host may replace `file:` with its own implementation.
pub fn register_backend(backend: Arc<dyn SecretBackend>) {
    if let Ok(mut backends) = registry().write() {
        backends.push(backend);
    }
}

struct Cached {
    value: String,
    at: Instant,
}

fn cache() -> &'static RwLock<HashMap<String, Cached>> {
    static CACHE: OnceLock<RwLock<HashMap<String, Cached>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// How long a resolved secret is reused before being fetched again.
fn ttl() -> Duration {
    std::env::var("BATCHER_SECRET_TTL_SECONDS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map_or(Duration::from_secs(300), Duration::from_secs)
}

/// Clear the resolution cache, so the next `resolve` re-fetches every reference.
///
/// A test hook, and the only caller today is this crate's own test module: the cache is
/// process-global with a TTL, so a test that changes a backend has to reset it or read a
/// stale value. It is `pub` because a host embedding this crate needs the same escape
/// hatch to make a credential rotation visible before the TTL expires.
pub fn clear_cache() {
    if let Ok(mut c) = cache().write() {
        c.clear();
    }
}

/// Split `reference` into `(scheme, target)`, or `None` when it names no known scheme.
///
/// A bare value is not a reference — it is an inline literal, which the caller passes
/// through unchanged. A `scheme:` prefix that matches no registered backend is likewise
/// *not* treated as a reference: `s3://bucket/key` and a password that merely contains a
/// colon must not be mistaken for one.
fn split_reference(reference: &str) -> Option<(Arc<dyn SecretBackend>, &str)> {
    let (scheme, target) = reference.split_once(':')?;
    let backends = registry().read().ok()?;
    // Reverse order so a later registration overrides an earlier one for the same scheme.
    let backend = backends.iter().rev().find(|b| b.scheme() == scheme)?;
    Some((Arc::clone(backend), target))
}

/// Whether `reference` names a registered secret scheme.
pub fn is_reference(reference: &str) -> bool {
    split_reference(reference).is_some()
}

/// Resolve a secret reference; return an inline literal unchanged.
///
/// Cached for [`ttl`]; see the module docs for why that matters on a per-batch call path.
pub fn resolve(reference: &str) -> Result<String, SecretError> {
    let Some((backend, target)) = split_reference(reference) else {
        return Ok(reference.to_string()); // an inline literal
    };
    let lifetime = ttl();
    if !lifetime.is_zero() {
        if let Ok(c) = cache().read() {
            if let Some(hit) = c.get(reference) {
                if hit.at.elapsed() < lifetime {
                    return Ok(hit.value.clone());
                }
            }
        }
    }
    let value = backend.fetch(target)?;
    if !lifetime.is_zero() {
        if let Ok(mut c) = cache().write() {
            c.insert(
                reference.to_string(),
                Cached {
                    value: value.clone(),
                    at: Instant::now(),
                },
            );
        }
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes the tests that mutate process-global environment state.
    ///
    /// Cargo runs a binary's tests on parallel threads, and several of these set
    /// `BATCHER_SECRET_COMMAND` or the TTL — process-wide, not per-test. Without this they
    /// race and fail intermittently, which is worse than failing outright because it looks
    /// like flakiness in the code under test. Correct under any `--test-threads`, rather
    /// than relying on an invocation flag nobody will remember to pass.
    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn set_var(k: &str, v: &str) {
        // SAFETY: callers hold `env_guard`, so no other test thread is reading or writing
        // the environment concurrently.
        unsafe { std::env::set_var(k, v) };
    }

    #[test]
    fn an_inline_literal_passes_through() {
        let _guard = env_guard();
        clear_cache();
        assert_eq!(resolve("just-a-password").unwrap(), "just-a-password");
        assert!(!is_reference("just-a-password"));
    }

    #[test]
    fn an_unknown_scheme_is_a_literal_not_a_reference() {
        let _guard = env_guard();
        // A password containing a colon, or an s3:// URL, must not be mistaken for a ref.
        clear_cache();
        assert!(!is_reference("s3://bucket/key"));
        assert_eq!(resolve("pa:ssword").unwrap(), "pa:ssword");
    }

    #[test]
    fn env_reference_resolves() {
        let _guard = env_guard();
        clear_cache();
        set_var("BC_SECRETS_TEST_ENV", "resolved-env");
        assert!(is_reference("env:BC_SECRETS_TEST_ENV"));
        assert_eq!(resolve("env:BC_SECRETS_TEST_ENV").unwrap(), "resolved-env");
    }

    #[test]
    fn a_missing_env_var_names_the_reference_not_the_value() {
        let _guard = env_guard();
        clear_cache();
        let err = resolve("env:BC_SECRETS_TEST_ABSENT").unwrap_err();
        assert!(err.to_string().contains("env:BC_SECRETS_TEST_ABSENT"));
    }

    #[test]
    fn file_reference_reads_and_trims() {
        let _guard = env_guard();
        clear_cache();
        let path = std::env::temp_dir().join("bc_secrets_test_file.txt");
        std::fs::write(&path, "file-secret\n").unwrap();
        let reference = format!("file:{}", path.display());
        assert_eq!(resolve(&reference).unwrap(), "file-secret");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn cmd_is_inert_until_an_operator_configures_the_program() {
        let _guard = env_guard();
        clear_cache();
        set_var(SECRET_COMMAND_VAR, "");
        let err = resolve("cmd:some-secret").unwrap_err();
        assert!(matches!(err, SecretError::CommandNotConfigured { .. }));
        // The error must explain the operator's half of the contract.
        assert!(err.to_string().contains(SECRET_COMMAND_VAR));
    }

    #[test]
    fn cmd_runs_the_operator_program_with_the_reference_as_its_argument() {
        let _guard = env_guard();
        clear_cache();
        // `echo` stands in for `vault kv get` / `aws secretsmanager get-secret-value`.
        set_var(SECRET_COMMAND_VAR, "echo");
        assert_eq!(resolve("cmd:my-secret-name").unwrap(), "my-secret-name");
        set_var(SECRET_COMMAND_VAR, "");
    }

    #[test]
    fn a_reference_cannot_smuggle_a_second_command() {
        let _guard = env_guard();
        // The argument is passed to `Command::arg`, never through a shell, so shell
        // metacharacters in a reference are inert data.
        clear_cache();
        set_var(SECRET_COMMAND_VAR, "echo");
        assert_eq!(
            resolve("cmd:x; touch /tmp/bc_secrets_pwned").unwrap(),
            "x; touch /tmp/bc_secrets_pwned"
        );
        assert!(!std::path::Path::new("/tmp/bc_secrets_pwned").exists());
        set_var(SECRET_COMMAND_VAR, "");
    }

    #[test]
    fn a_failing_helper_reports_stderr_and_never_stdout() {
        let _guard = env_guard();
        clear_cache();
        set_var(SECRET_COMMAND_VAR, "false"); // exits non-zero, prints nothing
        let err = resolve("cmd:whatever").unwrap_err();
        assert!(err.to_string().contains("cmd:whatever"));
        set_var(SECRET_COMMAND_VAR, "");
    }

    #[test]
    fn resolution_is_cached_within_the_ttl() {
        let _guard = env_guard();
        clear_cache();
        set_var("BATCHER_SECRET_TTL_SECONDS", "300");
        set_var("BC_SECRETS_TEST_CACHED", "first");
        assert_eq!(resolve("env:BC_SECRETS_TEST_CACHED").unwrap(), "first");
        set_var("BC_SECRETS_TEST_CACHED", "second");
        // Still the cached value: this is what keeps a per-batch call path from forking a
        // process (or stat-ing the filesystem) once per array.
        assert_eq!(resolve("env:BC_SECRETS_TEST_CACHED").unwrap(), "first");
        clear_cache();
        assert_eq!(resolve("env:BC_SECRETS_TEST_CACHED").unwrap(), "second");
    }

    #[test]
    fn a_zero_ttl_disables_caching() {
        let _guard = env_guard();
        clear_cache();
        set_var("BATCHER_SECRET_TTL_SECONDS", "0");
        set_var("BC_SECRETS_TEST_UNCACHED", "first");
        assert_eq!(resolve("env:BC_SECRETS_TEST_UNCACHED").unwrap(), "first");
        set_var("BC_SECRETS_TEST_UNCACHED", "second");
        assert_eq!(resolve("env:BC_SECRETS_TEST_UNCACHED").unwrap(), "second");
        set_var("BATCHER_SECRET_TTL_SECONDS", "300");
    }

    #[test]
    fn a_host_can_register_its_own_backend() {
        let _guard = env_guard();
        struct Fake;
        impl SecretBackend for Fake {
            fn scheme(&self) -> &str {
                "fakevault"
            }
            fn fetch(&self, target: &str) -> Result<String, SecretError> {
                Ok(format!("from-vault:{target}"))
            }
        }
        clear_cache();
        register_backend(Arc::new(Fake));
        assert_eq!(resolve("fakevault:db/pw").unwrap(), "from-vault:db/pw");
    }
}
