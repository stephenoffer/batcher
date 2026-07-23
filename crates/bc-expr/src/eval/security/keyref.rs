//! Resolving a crypto key *reference* to the key material, at evaluation time.
//!
//! An enterprise never wants a secret written into a query. The Python API therefore
//! lets `aes_encrypt`/`aes_decrypt`/`hmac_sha256` carry a *reference* — `env:NAME` or
//! `file:PATH` — instead of the raw key, and only that reference travels in the plan IR
//! (and thus in plan logs, the profile, `explain()`, and across the FFI boundary). The
//! actual key is read here, in the data plane, on the machine that runs the query — from
//! the environment or a mounted secret file the platform provides.
//!
//! Resolving in Rust rather than in Python is deliberate: the reference travels with the
//! plan to wherever it executes (including a remote Ray worker) and is resolved *there*,
//! against *that* machine's secrets, so a distributed query never ships a resolved key
//! over the wire. A bare value with no scheme prefix is treated as an inline literal —
//! the dev-convenience path the Python layer warns about.
//!
//! The scheme table itself lives in `bc-secrets`, which adds a TTL cache (this is a
//! per-batch call path — an uncached lookup re-read the environment or the filesystem for
//! every array) and the `cmd:` backend that reaches Vault / AWS Secrets Manager / GCP
//! Secret Manager / Azure Key Vault through an operator-configured helper program, with
//! no cloud SDK linked into the engine. This function keeps ownership of the error type
//! so the message a user sees still names the function they called.

use std::borrow::Cow;

use crate::ExprError;

/// Resolve a key reference to the key material.
///
/// * `env:NAME` — the value of environment variable `NAME`.
/// * `file:PATH` — the contents of `PATH`, with surrounding whitespace trimmed (so a
///   trailing newline in a mounted secret file is not part of the key).
/// * `cmd:NAME` — stdout of the operator-configured `BATCHER_SECRET_COMMAND` run with
///   `NAME`; the bridge to Vault / KMS / Secret Manager without an SDK in the engine.
/// * anything else — returned unchanged (an inline literal).
///
/// The error names the *reference* (`env:NAME` / `file:PATH`), never the resolved value:
/// the reference is not secret and is exactly what an operator needs to debug a
/// misconfiguration, whereas the key must never reach a log.
pub(super) fn resolve_key<'a>(func: &'static str, raw: &'a str) -> Result<Cow<'a, str>, ExprError> {
    if !bc_secrets::is_reference(raw) {
        return Ok(Cow::Borrowed(raw)); // an inline literal
    }
    bc_secrets::resolve(raw)
        .map(Cow::Owned)
        .map_err(|_| ExprError::KeyRefUnresolved {
            func,
            reference: raw.to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_value_is_an_inline_literal() {
        assert_eq!(resolve_key("aes_encrypt", "abc123").unwrap(), "abc123");
    }

    #[test]
    fn env_reference_reads_the_variable() {
        // SAFETY: single-threaded test; the var name is unique to this test.
        unsafe { std::env::set_var("BC_TEST_KEYREF_ENV", "resolved-secret") };
        assert_eq!(
            resolve_key("aes_encrypt", "env:BC_TEST_KEYREF_ENV").unwrap(),
            "resolved-secret"
        );
    }

    #[test]
    fn a_missing_env_var_errors_with_the_reference_not_the_value() {
        let err = resolve_key("aes_encrypt", "env:BC_TEST_KEYREF_ABSENT").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("env:BC_TEST_KEYREF_ABSENT"), "{msg}");
    }

    #[test]
    fn file_reference_reads_and_trims() {
        let dir = std::env::temp_dir();
        let path = dir.join("bc_test_keyref.pem");
        std::fs::write(&path, "file-secret\n  \n").unwrap();
        let raw = format!("file:{}", path.display());
        assert_eq!(resolve_key("aes_encrypt", &raw).unwrap(), "file-secret");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_missing_file_errors_with_the_path() {
        let err = resolve_key("aes_decrypt", "file:/no/such/secret").unwrap_err();
        assert!(err.to_string().contains("file:/no/such/secret"));
    }
}
