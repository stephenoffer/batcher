//! Character-distribution entropy — deliberately *not* one of Gopher's rules.
//!
//! It lives in its own module so that distinction survives: the measures in `gopher` carry
//! published thresholds a pipeline can cite, and this one does not. Choose its threshold from
//! a histogram of your own corpus.

use arrow::array::{ArrayRef, StringArray};

use super::builders::float_column;

/// `char_entropy()` → the Shannon entropy of the character distribution, in bits
/// (→ Float64).
///
/// The gibberish and encoded-blob detector, and the one measure here that is not from Gopher.
/// Natural language in any script lands in a narrow band — roughly 4 to 5 bits per character
/// for English prose. A base64 or hex blob sits above it (near-uniform over its alphabet), and
/// a run of one repeated character sits at zero. Both survive every ratio above, because
/// neither has unusual words, lines, or n-grams — they have unusual *characters*.
///
/// Null for the empty string, which has no distribution.
pub(crate) fn char_entropy(s: &StringArray) -> ArrayRef {
    float_column(s, |t| {
        let mut counts: std::collections::HashMap<char, usize, ahash::RandomState> =
            std::collections::HashMap::default();
        let mut total = 0usize;
        for c in t.chars() {
            *counts.entry(c).or_insert(0) += 1;
            total += 1;
        }
        (total > 0).then(|| {
            let sum: f64 = counts
                .values()
                .map(|&c| {
                    let p = c as f64 / total as f64;
                    p * p.log2()
                })
                .sum();
            // A document of one repeated character sums to exactly 0.0, and negating that
            // yields **-0.0** — so the entropy rendered as "-0". It compares equal to 0.0, so
            // `assert_eq!(h, 0.0)` cannot see it; only a caller reading the number does. Same
            // wart, same cause, as the one in `seq::expected_errors`.
            if sum == 0.0 {
                0.0
            } else {
                -sum
            }
        })
    })
}
