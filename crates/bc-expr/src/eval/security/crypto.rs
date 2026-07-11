//! Keyed cryptographic primitives: HMAC-SHA-256 pseudonymization and AES-256-GCM-SIV
//! column encryption.
//!
//! Both are **deterministic** — the same `(key, plaintext)` always yields the same
//! output. That is a required property here, not an oversight. `Expr::eval` must be a
//! pure function of its inputs, because the sequential interpreter is the correctness
//! oracle the parallel executor and the JIT are checked against; a randomized nonce
//! would make the three disagree row-for-row and would make the expression
//! un-cacheable and un-constant-foldable.
//!
//! Determinism is also what makes an encrypted column *useful* to a query engine:
//! equal plaintexts encrypt to equal ciphertexts, so a ciphertext column still joins,
//! groups, and equality-filters. The cost is that equality is observable — an
//! encrypted column leaks its value-frequency distribution. AES-GCM-SIV is chosen
//! precisely because it is the nonce-misuse-resistant AEAD: reusing our fixed nonce
//! degrades the scheme to exactly that equality leak and nothing worse. Where equality
//! leakage is unacceptable, do not encrypt the column — mask or tokenize it (see
//! `super::mask`) or leave it out of the projection entirely.

use aes_gcm_siv::aead::{Aead, KeyInit};
use aes_gcm_siv::{Aes256GcmSiv, Nonce};
use base64::Engine as _;
use hmac::{Hmac, Mac};
use sha2::Sha256;

use crate::ExprError;

/// The fixed all-zero nonce. Safe *only* because the cipher is GCM-**SIV**, whose
/// synthetic-IV construction derives the real counter from the plaintext; see the
/// module docs for the equality-leak trade this buys.
const SIV_NONCE: [u8; 12] = [0u8; 12];

/// AES-256 key length in bytes.
const KEY_LEN: usize = 32;

fn base64_engine() -> base64::engine::general_purpose::GeneralPurpose {
    base64::engine::general_purpose::STANDARD
}

/// Decode a 32-byte key from 64 hex characters or standard base64.
///
/// The error deliberately never echoes `key` — the value is secret material, and an
/// error message is the one place in the engine that reliably reaches a log file.
fn decode_key(func: &'static str, key: &str) -> Result<[u8; KEY_LEN], ExprError> {
    let bytes = if key.len() == KEY_LEN * 2 && key.bytes().all(|b| b.is_ascii_hexdigit()) {
        (0..KEY_LEN)
            .map(|i| u8::from_str_radix(&key[i * 2..i * 2 + 2], 16))
            .collect::<Result<Vec<u8>, _>>()
            .map_err(|_| ExprError::InvalidKey { func })?
    } else {
        base64_engine()
            .decode(key)
            .map_err(|_| ExprError::InvalidKey { func })?
    };
    bytes.try_into().map_err(|_| ExprError::InvalidKey { func })
}

/// Build the AES-256-GCM-SIV cipher once per array (key schedule is not per-row work).
pub(super) fn cipher_from_key(func: &'static str, key: &str) -> Result<Aes256GcmSiv, ExprError> {
    let key = decode_key(func, key)?;
    Aes256GcmSiv::new_from_slice(&key).map_err(|_| ExprError::InvalidKey { func })
}

/// Encrypt one value → base64 of `ciphertext || tag`.
///
/// Encryption of a valid plaintext under a valid key cannot fail, so this is total.
pub(super) fn encrypt(cipher: &Aes256GcmSiv, plaintext: &str) -> String {
    let sealed = cipher
        .encrypt(Nonce::from_slice(&SIV_NONCE), plaintext.as_bytes())
        .expect("AES-GCM-SIV encryption of an in-memory plaintext cannot fail");
    base64_engine().encode(sealed)
}

/// Decrypt one value, or `None` if it is not base64, fails authentication (wrong key,
/// or a tampered/truncated ciphertext), or does not decrypt to UTF-8.
///
/// Failure is a NULL rather than a query-killing error: a decrypt runs per row, and one
/// unreadable row must not abort a scan over a billion. This mirrors the existing
/// `from_base64`/`unhex` contract. Round-tripping under the correct key is total, so a
/// column of all-NULLs is the unambiguous signal of a wrong key.
pub(super) fn decrypt(cipher: &Aes256GcmSiv, ciphertext: &str) -> Option<String> {
    let sealed = base64_engine().decode(ciphertext).ok()?;
    let plain = cipher
        .decrypt(Nonce::from_slice(&SIV_NONCE), sealed.as_slice())
        .ok()?;
    String::from_utf8(plain).ok()
}

/// HMAC-SHA-256 of `data` keyed by the raw bytes of `key`, as lowercase hex.
///
/// The pseudonymization primitive: deterministic (so pseudonyms join across tables),
/// irreversible, and — unlike a bare `sha256` of a low-entropy value such as an email
/// or an SSN — not recoverable by brute-forcing the input domain, because the attacker
/// lacks the key.
pub(super) fn hmac_sha256_hex(key: &[u8], data: &[u8]) -> String {
    let mut mac =
        <Hmac<Sha256> as Mac>::new_from_slice(key).expect("HMAC accepts a key of any length");
    mac.update(data);
    crate::eval::str::hex_lower(&mac.finalize().into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEX_KEY: &str = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";

    #[test]
    fn hex_and_base64_keys_are_the_same_key() {
        let raw: [u8; KEY_LEN] = std::array::from_fn(|i| i as u8);
        let b64 = base64_engine().encode(raw);
        assert_eq!(decode_key("aes_encrypt", HEX_KEY).unwrap(), raw);
        assert_eq!(decode_key("aes_encrypt", &b64).unwrap(), raw);
    }

    #[test]
    fn short_or_garbage_keys_are_rejected() {
        for bad in ["", "abc", "00010203", &"a".repeat(63)] {
            assert!(decode_key("aes_encrypt", bad).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn encrypt_is_deterministic_and_round_trips() {
        let c = cipher_from_key("aes_encrypt", HEX_KEY).unwrap();
        let a = encrypt(&c, "alice@example.com");
        let b = encrypt(&c, "alice@example.com");
        assert_eq!(a, b, "equal plaintexts must encrypt equally (joinable)");
        assert_ne!(a, encrypt(&c, "bob@example.com"));
        assert_eq!(decrypt(&c, &a).as_deref(), Some("alice@example.com"));
        assert_eq!(decrypt(&c, &encrypt(&c, "")).as_deref(), Some(""));
    }

    #[test]
    fn decrypt_under_a_wrong_key_or_tampered_text_is_null() {
        let c = cipher_from_key("aes_encrypt", HEX_KEY).unwrap();
        let other = cipher_from_key("aes_encrypt", &"f".repeat(64)).unwrap();
        let ct = encrypt(&c, "secret");
        assert_eq!(decrypt(&other, &ct), None, "wrong key must not decrypt");
        assert_eq!(decrypt(&c, "not base64 at all!"), None);
        assert_eq!(decrypt(&c, &ct[..ct.len() - 4]), None, "truncated → null");
    }

    /// RFC 4231 test case 2 — pins HMAC-SHA-256 to the standard, so a dependency bump
    /// that changed the digest could not pass silently.
    #[test]
    fn hmac_sha256_matches_rfc_4231() {
        assert_eq!(
            hmac_sha256_hex(b"Jefe", b"what do ya want for nothing?"),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }

    #[test]
    fn hmac_is_keyed() {
        assert_ne!(hmac_sha256_hex(b"k1", b"x"), hmac_sha256_hex(b"k2", b"x"));
        assert_eq!(hmac_sha256_hex(b"k1", b"x"), hmac_sha256_hex(b"k1", b"x"));
    }
}
