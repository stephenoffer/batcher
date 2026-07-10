//! Data-protection string functions: `hmac_sha256`, `aes_encrypt`, `aes_decrypt`, `mask`.
//!
//! These are the `StrFunc` arms that let a column be pseudonymized, encrypted, or
//! redacted *inside the data plane* — the primitives the Python governance layer's
//! column-masking policies lower to. They live here rather than in `str.rs` because
//! they share a keying/erroring discipline that the ordinary string functions do not:
//! a key argument that must never appear in an error message, and a per-array key
//! schedule that must not be rebuilt per row.
//!
//! Contracts common to all four:
//!
//! * **Null in, null out**, like every other string function.
//! * **Deterministic**: a pure function of `(value, key, params)`. The interpreter is
//!   the oracle; a randomized output would put it at odds with the parallel executor.
//! * **Key material never reaches an error string** — see `crypto::decode_key`.

mod crypto;
mod mask;

use std::sync::Arc;

use arrow::array::{ArrayRef, StringArray};

use crate::{ExprError, StrFunc};

/// Evaluate one data-protection function over a Utf8 array, preserving nulls.
///
/// `pattern` carries the key (for the keyed functions) or the mask character; `start`
/// and `length` carry `mask`'s reveal counts. That reuse of `Expr::Str`'s existing
/// slots is what keeps these functions off the wire contract's critical path — no new
/// `Expr` variant, so the Python `to_ir()` shape is unchanged.
pub(crate) fn eval_security(
    func: StrFunc,
    s: &StringArray,
    pattern: Option<&str>,
    start: Option<i64>,
    length: Option<i64>,
) -> Result<ArrayRef, ExprError> {
    match func {
        StrFunc::HmacSha256 => {
            let key = require_key(pattern, "hmac_sha256")?;
            Ok(Arc::new(
                s.iter()
                    .map(|o| o.map(|v| crypto::hmac_sha256_hex(key.as_bytes(), v.as_bytes())))
                    .collect::<StringArray>(),
            ))
        }
        StrFunc::AesEncrypt => {
            let cipher =
                crypto::cipher_from_key("aes_encrypt", require_key(pattern, "aes_encrypt")?)?;
            Ok(Arc::new(
                s.iter()
                    .map(|o| o.map(|v| crypto::encrypt(&cipher, v)))
                    .collect::<StringArray>(),
            ))
        }
        StrFunc::AesDecrypt => {
            let cipher =
                crypto::cipher_from_key("aes_decrypt", require_key(pattern, "aes_decrypt")?)?;
            // Double `and_then`: a null stays null, and an undecryptable value becomes
            // one (wrong key / tampered ciphertext) rather than aborting the scan.
            Ok(Arc::new(
                s.iter()
                    .map(|o| o.and_then(|v| crypto::decrypt(&cipher, v)))
                    .collect::<StringArray>(),
            ))
        }
        StrFunc::Mask => {
            let ch = mask_char(pattern)?;
            let (first, last) = (reveal_count(start), reveal_count(length));
            Ok(Arc::new(
                s.iter()
                    .map(|o| o.map(|v| mask::mask(v, ch, first, last)))
                    .collect::<StringArray>(),
            ))
        }
        _ => unreachable!("eval_security called with the non-security StrFunc {func:?}"),
    }
}

fn require_key<'a>(pattern: Option<&'a str>, func: &'static str) -> Result<&'a str, ExprError> {
    pattern.ok_or(ExprError::MissingArgument {
        func: func.to_string(),
        arg: "key",
    })
}

/// `mask`'s replacement character, defaulting to `X`. A multi-character `pattern` is a
/// caller bug (the result would no longer be length-preserving), so it is rejected
/// rather than silently truncated.
fn mask_char(pattern: Option<&str>) -> Result<char, ExprError> {
    match pattern {
        None => Ok('X'),
        Some(p) => {
            let mut chars = p.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) => Ok(c),
                _ => Err(ExprError::MissingArgument {
                    func: "mask".to_string(),
                    arg: "a single-character mask",
                }),
            }
        }
    }
}

/// A reveal count: absent or negative means "reveal nothing".
fn reveal_count(n: Option<i64>) -> usize {
    n.unwrap_or(0).max(0) as usize
}

#[cfg(test)]
mod tests {
    use arrow::array::Array;

    use super::*;

    fn utf8(values: [Option<&str>; 2]) -> StringArray {
        values.into_iter().collect()
    }

    fn eval(func: StrFunc, pattern: Option<&str>) -> Vec<Option<String>> {
        let arr = utf8([Some("alice"), None]);
        let out = eval_security(func, &arr, pattern, None, None).unwrap();
        let out = out.as_any().downcast_ref::<StringArray>().unwrap();
        out.iter().map(|o| o.map(str::to_string)).collect()
    }

    const HEX_KEY: &str = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";

    #[test]
    fn every_function_propagates_nulls() {
        for (func, pattern) in [
            (StrFunc::HmacSha256, Some("k")),
            (StrFunc::AesEncrypt, Some(HEX_KEY)),
            (StrFunc::Mask, None),
        ] {
            let out = eval(func, pattern);
            assert!(out[0].is_some(), "{func:?} dropped a value");
            assert_eq!(out[1], None, "{func:?} did not propagate a null");
        }
    }

    #[test]
    fn aes_round_trips_through_the_array_path() {
        let arr = utf8([Some("alice"), None]);
        let ct = eval_security(StrFunc::AesEncrypt, &arr, Some(HEX_KEY), None, None).unwrap();
        let ct = ct.as_any().downcast_ref::<StringArray>().unwrap();
        let pt = eval_security(StrFunc::AesDecrypt, ct, Some(HEX_KEY), None, None).unwrap();
        let pt = pt.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(pt.value(0), "alice");
        assert!(pt.is_null(1));
    }

    #[test]
    fn a_wrong_key_nulls_the_column_rather_than_erroring() {
        let arr = utf8([Some("alice"), None]);
        let ct = eval_security(StrFunc::AesEncrypt, &arr, Some(HEX_KEY), None, None).unwrap();
        let ct = ct.as_any().downcast_ref::<StringArray>().unwrap();
        let pt = eval_security(StrFunc::AesDecrypt, ct, Some(&"f".repeat(64)), None, None).unwrap();
        assert_eq!(pt.null_count(), pt.len());
    }

    #[test]
    fn a_missing_or_invalid_key_is_an_error_not_a_silent_default() {
        let arr = utf8([Some("alice"), None]);
        assert!(eval_security(StrFunc::AesEncrypt, &arr, None, None, None).is_err());
        assert!(eval_security(StrFunc::HmacSha256, &arr, None, None, None).is_err());
        assert!(eval_security(StrFunc::AesEncrypt, &arr, Some("short"), None, None).is_err());
    }

    /// The key is secret material and errors reach logs — assert it never appears.
    #[test]
    fn an_invalid_key_error_does_not_echo_the_key() {
        let arr = utf8([Some("alice"), None]);
        let err = eval_security(
            StrFunc::AesEncrypt,
            &arr,
            Some("sekrit-but-invalid"),
            None,
            None,
        )
        .unwrap_err();
        assert!(!err.to_string().contains("sekrit"), "{err}");
    }

    #[test]
    fn mask_reveal_counts_come_from_start_and_length() {
        let arr = utf8([Some("4111111111111234"), None]);
        let out = eval_security(StrFunc::Mask, &arr, Some("*"), Some(0), Some(4)).unwrap();
        let out = out.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(out.value(0), "************1234");
    }

    #[test]
    fn a_negative_reveal_count_reveals_nothing() {
        let arr = utf8([Some("abc"), None]);
        let out = eval_security(StrFunc::Mask, &arr, None, Some(-5), Some(-1)).unwrap();
        let out = out.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(out.value(0), "XXX");
    }

    #[test]
    fn a_multi_character_mask_is_rejected() {
        let arr = utf8([Some("abc"), None]);
        assert!(eval_security(StrFunc::Mask, &arr, Some("ab"), None, None).is_err());
    }
}
