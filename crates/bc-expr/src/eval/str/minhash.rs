//! `StrFunc::MinHash` — a MinHash signature of a document → `List<Int64>`.
//!
//! Fuzzy deduplication is the defining preprocessing step of an LLM pretraining corpus:
//! exact dedup misses the boilerplate-and-a-comma near-copies that dominate a web crawl.
//! MinHash makes it tractable. The signature is `num_perm` values, one per hash
//! permutation, each the minimum over the document's character *shingles*. The fraction
//! of positions at which two signatures agree is an unbiased estimator of the documents'
//! Jaccard similarity, with standard error `1/sqrt(num_perm)` — so 128 permutations
//! estimate Jaccard to about ±0.09, and comparing two 128-value signatures replaces
//! comparing two documents.
//!
//! Signature values are bounded to 32 bits. That is not an accident: it lets a signature
//! round-trip exactly through the `Float64` element type the list-binary kernels use, so
//! `.list.jaccard()` counts agreements exactly rather than approximately.
//!
//! Everything is written out and seeded from constants, because two documents on two
//! machines must produce comparable signatures: a permutation that differed by build
//! would make every cross-shard duplicate invisible.

use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Builder, ListBuilder, StringArray};

use crate::{ExprError, StrFunc};

/// The Mersenne prime 2^61 - 1: permutations are affine maps modulo this, which admits
/// the fold-and-subtract reduction below instead of a 128-bit division per shingle.
const MERSENNE_61: u64 = (1 << 61) - 1;
/// Signature values live in `[0, 2^32)` so they are exact as `f64` (see module docs).
const MAX_HASH: u64 = (1 << 32) - 1;

/// SplitMix64's finalizer — derives the permutation coefficients deterministically.
fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// FNV-1a over the shingle's bytes, folded into 32 bits.
fn shingle_hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    (h ^ (h >> 32)) & MAX_HASH
}

/// `x mod (2^61 - 1)` for `x < 2^93`. Folding the high bits back is exact for a Mersenne
/// modulus, and the sum is under `2p`, so a single conditional subtract finishes it.
fn mod_mersenne(x: u128) -> u64 {
    let p = u128::from(MERSENNE_61);
    let mut r = (x & p) + (x >> 61);
    if r >= p {
        r -= p;
    }
    r as u64
}

/// The `i`-th permutation's coefficients `(a, b)`; `a` is non-zero modulo the prime.
fn coefficients(i: usize) -> (u64, u64) {
    let a = mix64(i as u64 * 2 + 1) % (MERSENNE_61 - 1) + 1;
    let b = mix64(i as u64 * 2 + 2) % MERSENNE_61;
    (a, b)
}

/// Evaluate `minhash`: `length` is `num_perm`, `start` is the shingle width in characters.
pub(crate) fn eval_minhash(
    s: &StringArray,
    start: Option<i64>,
    length: Option<i64>,
) -> Result<ArrayRef, ExprError> {
    let num_perm = length.ok_or_else(|| ExprError::MissingArgument {
        func: format!("{:?}", StrFunc::MinHash),
        arg: "length",
    })?;
    let ngram = start.unwrap_or(5);
    if num_perm < 1 || ngram < 1 {
        return Err(ExprError::InvalidArgument {
            func: format!("{:?}", StrFunc::MinHash),
            reason: format!(
                "num_perm and ngram must be >= 1, got num_perm={num_perm} ngram={ngram}"
            ),
        });
    }
    let (num_perm, ngram) = (num_perm as usize, ngram as usize);
    let coeffs: Vec<(u64, u64)> = (0..num_perm).map(coefficients).collect();

    let mut builder = ListBuilder::new(Int64Builder::new());
    let mut signature = vec![0u64; num_perm];
    let mut offsets: Vec<usize> = Vec::new();
    for opt in s.iter() {
        match opt {
            Some(text) => {
                signature.fill(MAX_HASH);
                for hv in shingle_hashes(text, ngram, &mut offsets) {
                    for (slot, (a, b)) in signature.iter_mut().zip(&coeffs) {
                        let v = mod_mersenne(u128::from(*a) * u128::from(hv) + u128::from(*b))
                            & MAX_HASH;
                        if v < *slot {
                            *slot = v;
                        }
                    }
                }
                for v in &signature {
                    builder.values().append_value(*v as i64);
                }
                builder.append(true);
            }
            None => builder.append(false),
        }
    }
    Ok(Arc::new(builder.finish()))
}

/// The hashes of `text`'s character `ngram`-shingles.
///
/// A document shorter than one shingle contributes its whole text as a single shingle,
/// so a short string still has a signature (rather than an empty, incomparable one).
/// Shingling is by character, not byte, so a multi-byte codepoint is never split — two
/// documents differing only in encoding would otherwise share no shingle at all.
fn shingle_hashes(text: &str, ngram: usize, offsets: &mut Vec<usize>) -> Vec<u64> {
    if text.is_ascii() {
        let bytes = text.as_bytes();
        if bytes.len() <= ngram {
            return vec![shingle_hash(bytes)];
        }
        return (0..=bytes.len() - ngram)
            .map(|i| shingle_hash(&bytes[i..i + ngram]))
            .collect();
    }
    offsets.clear();
    offsets.extend(text.char_indices().map(|(i, _)| i));
    offsets.push(text.len());
    let chars = offsets.len() - 1;
    if chars <= ngram {
        return vec![shingle_hash(text.as_bytes())];
    }
    (0..=chars - ngram)
        .map(|i| shingle_hash(&text.as_bytes()[offsets[i]..offsets[i + ngram]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, AsArray};
    use arrow::datatypes::Int64Type;

    fn sig(text: &str, num_perm: i64, ngram: i64) -> Vec<i64> {
        let arr = StringArray::from(vec![Some(text)]);
        let out = eval_minhash(&arr, Some(ngram), Some(num_perm)).unwrap();
        let list = out.as_list::<i32>();
        let vals = list.value(0);
        let vals = vals.as_primitive::<Int64Type>();
        (0..vals.len()).map(|i| vals.value(i)).collect()
    }

    /// The estimator: the fraction of agreeing positions.
    fn estimate(a: &[i64], b: &[i64]) -> f64 {
        let same = a.iter().zip(b).filter(|(x, y)| x == y).count();
        same as f64 / a.len() as f64
    }

    /// Exact Jaccard over the character shingle sets — the quantity being estimated.
    fn exact_jaccard(x: &str, y: &str, w: usize) -> f64 {
        use std::collections::HashSet;
        let shingles = |t: &str| -> HashSet<String> {
            let c: Vec<char> = t.chars().collect();
            if c.len() <= w {
                return HashSet::from([t.to_string()]);
            }
            (0..=c.len() - w)
                .map(|i| c[i..i + w].iter().collect())
                .collect()
        };
        let (a, b) = (shingles(x), shingles(y));
        a.intersection(&b).count() as f64 / a.union(&b).count() as f64
    }

    #[test]
    fn signature_has_the_requested_length_and_is_bounded_to_32_bits() {
        let s = sig("the quick brown fox", 64, 5);
        assert_eq!(s.len(), 64);
        assert!(s.iter().all(|v| *v >= 0 && *v <= MAX_HASH as i64));
    }

    #[test]
    fn identical_documents_have_identical_signatures() {
        assert_eq!(sig("hello world", 32, 4), sig("hello world", 32, 4));
    }

    #[test]
    fn different_documents_have_different_signatures() {
        assert_ne!(sig("hello world", 32, 4), sig("goodbye moon", 32, 4));
    }

    #[test]
    fn identical_documents_estimate_jaccard_one() {
        let a = sig("the quick brown fox jumps over the lazy dog", 128, 5);
        assert_eq!(estimate(&a, &a), 1.0);
    }

    /// The whole point: the agreement rate tracks the true Jaccard. With 128
    /// permutations the standard error is ~0.09, so 0.15 is a wide but real bound.
    #[test]
    fn the_estimate_tracks_the_true_jaccard() {
        let cases = [
            (
                "the quick brown fox jumps over the lazy dog",
                "the quick brown fox jumps over the lazy dog!",
            ),
            (
                "the quick brown fox jumps over the lazy dog",
                "the quick brown cat jumps over the lazy dog",
            ),
            (
                "the quick brown fox jumps over the lazy dog",
                "completely unrelated text about databases",
            ),
        ];
        for (x, y) in cases {
            let (a, b) = (sig(x, 128, 5), sig(y, 128, 5));
            let (est, exact) = (estimate(&a, &b), exact_jaccard(x, y, 5));
            assert!(
                (est - exact).abs() < 0.15,
                "estimate {est:.3} vs exact {exact:.3} for {x:?} / {y:?}"
            );
        }
    }

    #[test]
    fn dissimilar_documents_agree_on_almost_no_positions() {
        let a = sig("the quick brown fox jumps over the lazy dog", 128, 5);
        let b = sig("completely unrelated text about databases", 128, 5);
        assert!(estimate(&a, &b) < 0.1);
    }

    #[test]
    fn a_document_shorter_than_a_shingle_still_has_a_signature() {
        let s = sig("ab", 16, 5);
        assert_eq!(s.len(), 16);
        assert_eq!(s, sig("ab", 16, 5));
        assert_ne!(s, sig("cd", 16, 5));
    }

    #[test]
    fn unicode_shingles_on_character_boundaries() {
        let a = sig("héllo wörld → unicode", 32, 4);
        assert_eq!(a.len(), 32);
        assert_eq!(a, sig("héllo wörld → unicode", 32, 4));
    }

    #[test]
    fn nulls_yield_null_signatures() {
        let arr = StringArray::from(vec![Some("a"), None]);
        let out = eval_minhash(&arr, Some(3), Some(8)).unwrap();
        let list = out.as_list::<i32>();
        assert!(!list.is_null(0));
        assert!(list.is_null(1));
    }

    #[test]
    fn degenerate_parameters_are_rejected() {
        let arr = StringArray::from(vec![Some("abc")]);
        assert!(eval_minhash(&arr, Some(5), Some(0)).is_err());
        assert!(eval_minhash(&arr, Some(0), Some(8)).is_err());
        assert!(eval_minhash(&arr, Some(5), None).is_err());
    }

    /// A permutation that varied by build would make every cross-shard duplicate
    /// invisible, so the coefficients are pinned.
    #[test]
    fn permutation_coefficients_are_stable() {
        assert_eq!(
            coefficients(0),
            (1_227_844_342_346_046_666, 1_682_153_688_901_572_306)
        );
        assert_eq!(
            coefficients(1),
            (2_092_789_425_003_139_054, 1_041_426_021_413_522_125)
        );
    }
}
