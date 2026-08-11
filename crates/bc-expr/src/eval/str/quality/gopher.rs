//! Gopher, C4, and RefinedWeb's published document-quality rules.
//!
//! Every threshold cited in these docs is from Gopher (Rae et al. 2021, appendix A.1.1)
//! unless stated otherwise, so a pipeline built on them can be compared against the corpora
//! those papers describe.

use arrow::array::{ArrayRef, StringArray};

use super::builders::{float_column, int_column};
use crate::eval::FastSet;

/// Split a document into whitespace-separated words, the unit every Gopher rule counts in.
#[inline]
fn words(text: &str) -> impl Iterator<Item = &str> {
    text.split_whitespace()
}

/// The characters that mark a bulleted line, checked after leading whitespace.
///
/// The hyphen and asterisk are here because Markdown and plain-text listings use them, and a
/// page that is 90% bullets is a navigation menu rather than prose whichever glyph it chose.
const BULLETS: [char; 6] = ['\u{2022}', '\u{2023}', '\u{2043}', '\u{2219}', '-', '*'];

/// Gopher's stop-word set. A document containing fewer than two of these is almost never
/// English prose — it is a keyword list, a table of figures, or a directory index.
const STOPWORDS: [&str; 8] = ["the", "be", "to", "of", "and", "that", "have", "with"];

/// `word_count()` → the number of whitespace-separated words (→ Int64).
pub(crate) fn word_count(s: &StringArray) -> ArrayRef {
    int_column(s, |t| words(t).count() as i64)
}

// A non-empty *line count* is deliberately not exported. `.str.line_count()` already exists
// with different, established semantics (newlines plus one, so blank lines count), and adding
// a second spelling of "how many lines" would be exactly the ambiguity the API avoids. The
// non-empty count the ratios below divide by stays internal to `line_ratio`.

/// `mean_word_length()` → the mean word length in characters (→ Float64).
///
/// Gopher drops a document outside `[3, 10]`. Below three is a token list or a table of
/// numbers; above ten is usually a base64 blob, a minified script, or a URL dump.
pub(crate) fn mean_word_length(s: &StringArray) -> ArrayRef {
    float_column(s, |t| {
        let (total, n) = words(t).fold((0usize, 0usize), |(sum, n), w| {
            (sum + w.chars().count(), n + 1)
        });
        (n > 0).then(|| total as f64 / n as f64)
    })
}

/// `symbol_ratio()` → `(# + …) / words` (→ Float64).
///
/// Gopher drops a document above 0.1. A high hash count is a stripped-out heading structure
/// or a code fragment; a high ellipsis count is a listing page whose entries were truncated
/// for display, which is text that reads as prose and finishes no sentence.
pub(crate) fn symbol_ratio(s: &StringArray) -> ArrayRef {
    float_column(s, |t| {
        let n = words(t).count();
        if n == 0 {
            return None;
        }
        // `…` and the three-dot spelling both count, because which one a page uses is a
        // function of its encoder rather than of its content.
        let hashes = t.matches('#').count();
        let ellipses = t.matches('\u{2026}').count() + t.matches("...").count();
        Some((hashes + ellipses) as f64 / n as f64)
    })
}

/// `alpha_word_ratio()` → the fraction of words containing at least one letter (→ Float64).
///
/// Gopher drops a document below 0.8. The words without letters are prices, timestamps,
/// part numbers and bare punctuation, and a document that is mostly those is a table that
/// lost its structure.
pub(crate) fn alpha_word_ratio(s: &StringArray) -> ArrayRef {
    float_column(s, |t| {
        let (alpha, n) = words(t).fold((0usize, 0usize), |(a, n), w| {
            (a + usize::from(w.chars().any(char::is_alphabetic)), n + 1)
        });
        (n > 0).then(|| alpha as f64 / n as f64)
    })
}

/// `stopword_count()` → how many of Gopher's eight stop words appear (→ Int64).
///
/// Counted as *distinct* words present, not occurrences: the rule is "does this read as
/// English", and a page repeating "the" two hundred times is not better evidence than one
/// using "the" and "of" once each.
pub(crate) fn stopword_count(s: &StringArray) -> ArrayRef {
    int_column(s, |t| {
        // A bitmask over the eight stop words rather than a `Vec<String>` of every word in
        // the document. The obvious spelling lower-cased and collected each word — one heap
        // allocation per word, then a linear scan per stop word — and measured 81 MB/s
        // against `mean_word_length`'s 235 on the same corpus. Nothing here allocates: each
        // word is compared in place, case-insensitively, and the first match sets its bit.
        //
        // Counting *distinct* stop words is what the bitmask structurally is, so that rule no
        // longer depends on a `Vec` being scanned for membership.
        let mut seen: u8 = 0;
        for word in words(t) {
            let trimmed = word.trim_matches(|c: char| !c.is_alphanumeric());
            if trimmed.is_empty() {
                continue;
            }
            for (bit, sw) in STOPWORDS.iter().enumerate() {
                if seen & (1 << bit) == 0 && trimmed.eq_ignore_ascii_case(sw) {
                    seen |= 1 << bit;
                    break;
                }
            }
            if seen == u8::MAX {
                // Every stop word is already present; no later word can add one.
                break;
            }
        }
        i64::from(seen.count_ones())
    })
}

/// Ratio of non-empty lines satisfying `pred`.
fn line_ratio<F>(s: &StringArray, pred: F) -> ArrayRef
where
    F: Fn(&str) -> bool,
{
    float_column(s, |t| {
        let (hit, n) = t
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .fold((0usize, 0usize), |(h, n), l| {
                (h + usize::from(pred(l)), n + 1)
            });
        (n > 0).then(|| hit as f64 / n as f64)
    })
}

/// `bullet_line_ratio()` → the fraction of lines beginning with a bullet (→ Float64).
///
/// Gopher drops a document above 0.9 — that is a navigation menu or a sitemap, not prose.
pub(crate) fn bullet_line_ratio(s: &StringArray) -> ArrayRef {
    line_ratio(s, |l| l.starts_with(BULLETS))
}

/// `ellipsis_line_ratio()` → the fraction of lines ending in an ellipsis (→ Float64).
///
/// Gopher drops a document above 0.3. This is the signature of a listing page: every entry
/// is a teaser cut off at a fixed width, so the text reads like prose and says nothing.
pub(crate) fn ellipsis_line_ratio(s: &StringArray) -> ArrayRef {
    line_ratio(s, |l| l.ends_with('\u{2026}') || l.ends_with("..."))
}

/// Fraction of *characters* that sit in a duplicated unit, where units are the pieces `split`
/// yields.
///
/// Characters rather than units, following Gopher: a page whose one duplicated line is its
/// 500-character footer is a different document from one whose fifty duplicated lines are
/// single words, and a unit count cannot tell them apart. Every occurrence beyond the first
/// counts, so text repeated three times contributes two copies' worth.
fn duplicate_char_ratio<'a, I: Iterator<Item = &'a str>>(units: I) -> Option<f64> {
    let mut seen: FastSet<&str> = FastSet::default();
    let (mut dup, mut total) = (0usize, 0usize);
    for unit in units {
        let len = unit.chars().count();
        if len == 0 {
            continue;
        }
        total += len;
        if !seen.insert(unit) {
            dup += len;
        }
    }
    (total > 0).then(|| dup as f64 / total as f64)
}

/// `duplicate_line_ratio()` → the fraction of characters in repeated lines (→ Float64).
pub(crate) fn duplicate_line_ratio(s: &StringArray) -> ArrayRef {
    float_column(s, |t| duplicate_char_ratio(t.lines().map(str::trim)))
}

/// `duplicate_paragraph_ratio()` → the fraction of characters in repeated paragraphs
/// (→ Float64). A paragraph is a run of text between blank lines.
pub(crate) fn duplicate_paragraph_ratio(s: &StringArray) -> ArrayRef {
    float_column(s, |t| duplicate_char_ratio(t.split("\n\n").map(str::trim)))
}

/// The word n-grams of a document, as `(start, end)` byte offsets into it.
///
/// Offsets rather than owned strings so the counting below borrows: a long document has
/// O(words) n-grams and allocating each one dominates the measure otherwise.
fn word_ngrams(text: &str, n: usize) -> Vec<(usize, usize)> {
    let bounds: Vec<(usize, usize)> = text
        .split_whitespace()
        .map(|w| {
            let start = w.as_ptr() as usize - text.as_ptr() as usize;
            (start, start + w.len())
        })
        .collect();
    if bounds.len() < n {
        return Vec::new();
    }
    (0..=bounds.len() - n)
        .map(|i| (bounds[i].0, bounds[i + n - 1].1))
        .collect()
}

/// `top_ngram_ratio(n)` → the fraction of characters covered by the single most frequent
/// word n-gram (→ Float64).
///
/// Gopher applies this for n of 2 to 4, dropping a document above 0.20 to 0.15. It catches
/// keyword-stuffed SEO pages, where one phrase is repeated to game a ranking, and templated
/// product pages, where one clause appears once per item.
pub(crate) fn top_ngram_ratio(s: &StringArray, n: i64) -> ArrayRef {
    let n = n.max(1) as usize;
    float_column(s, move |t| {
        let grams = word_ngrams(t, n);
        if grams.is_empty() {
            return None;
        }
        let total: usize = t.chars().count();
        if total == 0 {
            return None;
        }
        let mut counts: std::collections::HashMap<&str, usize, ahash::RandomState> =
            std::collections::HashMap::default();
        for &(a, b) in &grams {
            *counts.entry(&t[a..b]).or_insert(0) += 1;
        }
        // The most frequent n-gram's total footprint: how many characters it covers, counting
        // every occurrence. Ties are broken by whichever the iteration reaches first, which is
        // immaterial — the measure is the count, not the identity.
        let best = counts
            .iter()
            .map(|(gram, count)| gram.chars().count() * count)
            .max()
            .unwrap_or(0);
        // Clamped, because word n-grams *overlap*: in "a a a a" the 2-gram "a a" occurs three
        // times over seven characters, so the raw footprint exceeds the document. Gopher
        // defines this as a fraction and thresholds it as one, so a value above 1 would break
        // every comparison built on it; the clamp keeps the contract the name states.
        Some((best as f64 / total as f64).min(1.0))
    })
}

/// `duplicate_ngram_ratio(n)` → the fraction of characters covered by n-grams that appear
/// more than once (→ Float64).
///
/// Gopher applies this for n of 5 to 10, with thresholds falling from 0.15 to 0.10. Unlike
/// [`top_ngram_ratio`] it sums over *every* repeated n-gram, which is what catches a document
/// assembled from a dozen boilerplate blocks rather than one repeated phrase.
///
/// Each duplicated n-gram is counted once regardless of how many times it recurs, matching
/// the published definition — the question is how much of the document is boilerplate, not
/// how many times the boilerplate appears.
pub(crate) fn duplicate_ngram_ratio(s: &StringArray, n: i64) -> ArrayRef {
    let n = n.max(1) as usize;
    float_column(s, move |t| {
        let grams = word_ngrams(t, n);
        if grams.is_empty() {
            return None;
        }
        let total: usize = t.chars().count();
        if total == 0 {
            return None;
        }
        let mut counts: std::collections::HashMap<&str, usize, ahash::RandomState> =
            std::collections::HashMap::default();
        for &(a, b) in &grams {
            *counts.entry(&t[a..b]).or_insert(0) += 1;
        }
        let dup: usize = counts
            .iter()
            .filter(|(_, &c)| c > 1)
            .map(|(gram, _)| gram.chars().count())
            .sum();
        Some((dup as f64 / total as f64).min(1.0))
    })
}
