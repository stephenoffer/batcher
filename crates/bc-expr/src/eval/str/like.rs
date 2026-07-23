//! Fast SQL `LIKE` / substring matching.
//!
//! A `LIKE` predicate searches the **same** pattern against every row of a column, yet the
//! naive path (`map_bool(s, |v| v.contains(pat))` for `contains`, or a desugared
//! `regex::is_match` for `LIKE`) pays a per-row cost the pattern does not require:
//!
//! * `str::contains(needle)` rebuilds its Two-Way searcher on every call — ~80 ns/row where a
//!   prebuilt [`memchr::memmem::Finder`] reused across the column is DuckDB-class ~7 ns/row.
//! * A desugared `LIKE '%a%b%'` runs a full regex automaton where an *ordered sequence of
//!   substring searches* answers the same question far faster (`^.*a.*b.*$` on a 1.5M-row
//!   column measured ~750 ms; the segment scan is tens of ms).
//!
//! [`LikeMatcher`] classifies the pattern **once** into the cheapest shape that answers it and
//! reuses the prebuilt finders across the whole column. It is a throughput-only change: every
//! variant is bit-for-bit equal to the anchored regex `^…$` the interpreter used before, so the
//! interpreter stays the oracle. Patterns it cannot answer without a real automaton — a `_`
//! single-char wildcard, or the Unicode case-folding of `ILIKE` — fall back to that same cached
//! regex, so nothing regresses.

use std::sync::Arc;

use arrow::array::{Array, BooleanArray, StringArray};
use arrow::buffer::BooleanBuffer;
use memchr::memmem::Finder;
use regex::Regex;

/// A compiled `LIKE`/substring predicate, built once per morsel and reused across every row.
pub(crate) enum LikeMatcher {
    /// `%` / `%%` — matches every (non-null) row.
    All,
    /// No wildcards: exact equality (the anchored regex `^p$`).
    Exact(String),
    /// `%needle%` — a single required substring anywhere. Boxed because a `memmem::Finder`
    /// carries a ~288-byte two-way searcher + SIMD prefilter; inline it would bloat every
    /// `LikeMatcher` (and the hot `is_match` match) to that size.
    Contains(Box<Finder<'static>>),
    /// `needle%` — a required prefix.
    StartsWith(String),
    /// `%needle` — a required suffix.
    EndsWith(String),
    /// `pre%mid1%…%suf` — a required prefix and suffix with ordered middle substrings.
    Segments {
        prefix: String,
        suffix: String,
        middles: Vec<Finder<'static>>,
    },
    /// Fallback for anything the fast paths cannot answer exactly (a `_` wildcard, or the
    /// Unicode case-folding of `ILIKE`): the cached anchored regex the desugarer produced.
    Regex(Arc<Regex>),
}

impl LikeMatcher {
    /// Build a matcher for a case-sensitive `LIKE` pattern with **no `_` wildcard** (the caller
    /// checks and routes `_`/`ILIKE` to [`LikeMatcher::Regex`] instead). `%` is any run
    /// (including empty); every other character is a literal — matching the desugarer in
    /// `super::like_regex`.
    pub(crate) fn classify(pattern: &str) -> Self {
        let parts: Vec<&str> = pattern.split('%').collect();
        // No `%` at all: the whole pattern is a literal → exact equality.
        if parts.len() == 1 {
            return LikeMatcher::Exact(pattern.to_string());
        }
        let prefix = parts[0];
        let suffix = parts[parts.len() - 1];
        // Adjacent `%%` yields an empty middle; `.*.*` == `.*`, so an empty middle constrains
        // nothing and is dropped.
        let middles: Vec<&str> = parts[1..parts.len() - 1]
            .iter()
            .copied()
            .filter(|s| !s.is_empty())
            .collect();
        match (prefix.is_empty(), suffix.is_empty(), middles.as_slice()) {
            (true, true, []) => LikeMatcher::All,
            (true, true, [m]) => LikeMatcher::Contains(Box::new(owned_finder(m))),
            (false, true, []) => LikeMatcher::StartsWith(prefix.to_string()),
            (true, false, []) => LikeMatcher::EndsWith(suffix.to_string()),
            _ => LikeMatcher::Segments {
                prefix: prefix.to_string(),
                suffix: suffix.to_string(),
                middles: middles.iter().map(|m| owned_finder(m)).collect(),
            },
        }
    }

    /// A bare `contains(col, needle)` (SQL has no anchors here): a single substring anywhere.
    pub(crate) fn contains(needle: &str) -> Self {
        LikeMatcher::Contains(Box::new(owned_finder(needle)))
    }

    /// A bare `starts_with(col, needle)`.
    pub(crate) fn starts_with(needle: &str) -> Self {
        LikeMatcher::StartsWith(needle.to_string())
    }

    /// A bare `ends_with(col, needle)`.
    pub(crate) fn ends_with(needle: &str) -> Self {
        LikeMatcher::EndsWith(needle.to_string())
    }

    /// Whether one string matches. Every arm is equal to the anchored regex the desugarer
    /// produced for the same pattern.
    #[inline(always)]
    pub(crate) fn is_match(&self, s: &str) -> bool {
        match self {
            LikeMatcher::All => true,
            LikeMatcher::Exact(p) => s == p,
            LikeMatcher::Contains(f) => f.find(s.as_bytes()).is_some(),
            LikeMatcher::StartsWith(p) => s.as_bytes().starts_with(p.as_bytes()),
            LikeMatcher::EndsWith(p) => s.as_bytes().ends_with(p.as_bytes()),
            LikeMatcher::Segments {
                prefix,
                suffix,
                middles,
            } => segment_match(s, prefix, suffix, middles),
            LikeMatcher::Regex(re) => re.is_match(s),
        }
    }

    /// Apply to a whole column, producing the match bitmask. Nulls stay null (the predicate is
    /// evaluated over the — possibly empty — slice a null slot points at, then masked away by the
    /// preserved null buffer). Uses the packed [`BooleanBuffer::collect_bool`] rather than a
    /// per-element `Option<bool>` iterator, so a trivial predicate is bit-packing bound, not
    /// iterator-adapter bound.
    pub(crate) fn eval(&self, s: &StringArray) -> BooleanArray {
        let values = BooleanBuffer::collect_bool(s.len(), |i| self.is_match(s.value(i)));
        BooleanArray::new(values, s.nulls().cloned())
    }
}

/// A prefix, then ordered middle substrings, then a suffix — all within the region the anchors
/// leave free. Byte-oriented throughout: the haystack is valid UTF-8 and every needle is a whole
/// UTF-8 substring, so a byte match can only land on a char boundary (UTF-8 self-synchronization),
/// and no result differs from the char-oriented regex.
///
/// `inline(always)`: this is the body of a per-row closure over the whole column; left as a
/// plain call it costs a function prologue and arg-marshalling per row (measured ~5× the bare
/// `Finder::find` even when the first segment short-circuits), so it must fold into the hot loop.
#[inline(always)]
fn segment_match(s: &str, prefix: &str, suffix: &str, middles: &[Finder<'static>]) -> bool {
    let bytes = s.as_bytes();
    // The prefix and suffix occupy disjoint regions; if together they overrun the string it
    // cannot match (`^ab.*ab$` needs ≥ "abab" for prefix="ab", suffix="ab").
    if bytes.len() < prefix.len() + suffix.len() {
        return false;
    }
    if !bytes.starts_with(prefix.as_bytes()) || !bytes.ends_with(suffix.as_bytes()) {
        return false;
    }
    // Middles are searched only between the anchored regions, in order, each after the last.
    let mut pos = prefix.len();
    let end = bytes.len() - suffix.len();
    for f in middles {
        match f.find(&bytes[pos..end]) {
            Some(i) => pos += i + f.needle().len(),
            None => return false,
        }
    }
    true
}

/// A `Finder` that owns its needle so the matcher can outlive the pattern string.
fn owned_finder(needle: &str) -> Finder<'static> {
    Finder::new(needle.as_bytes()).into_owned()
}

#[cfg(test)]
mod tests {
    use super::LikeMatcher;
    use crate::eval::str::like_regex;

    /// The fast matcher must agree with the anchored regex on every case-sensitive, `_`-free
    /// pattern — the whole correctness argument for the fast path.
    #[test]
    fn matcher_agrees_with_regex_over_many_cases() {
        let patterns = [
            "",
            "%",
            "%%",
            "abc",
            "abc%",
            "%abc",
            "%abc%",
            "a%c",
            "%special%requests%",
            "foo%bar",
            "foo%bar%",
            "%a%b%c%",
            "%%abc%%",
            "x",
            "%x",
            "x%",
            "ab%ab",
            "a%a",
            ".*",
            "a.c%",
            "100%",
        ];
        let inputs = [
            "",
            "abc",
            "abcd",
            "xabc",
            "aXc",
            "a special set of requests here",
            "specialrequests",
            "requests special",
            "fooZZbar",
            "fooZZbarYY",
            "aXbYc",
            "abcabc",
            "ab",
            "aa",
            "a",
            "x",
            "yx",
            "xy",
            "a.c and more",
            "100%",
            "100",
            "abcABC",
            "AXC",
        ];
        for p in patterns {
            let re = like_regex(p, false).expect("valid like pattern");
            let m = LikeMatcher::classify(p);
            for s in inputs {
                assert_eq!(
                    m.is_match(s),
                    re.is_match(s),
                    "LIKE '{p}' on {s:?}: fast matcher disagrees with regex"
                );
            }
        }
    }

    #[test]
    fn contains_starts_ends_match_std() {
        let m = LikeMatcher::contains("req");
        assert!(m.is_match("requests") && m.is_match("prereq") && !m.is_match("quest"));
        let m = LikeMatcher::starts_with("re");
        assert!(m.is_match("requests") && !m.is_match("prereq"));
        let m = LikeMatcher::ends_with("ts");
        assert!(m.is_match("requests") && !m.is_match("request"));
    }

    #[test]
    fn eval_preserves_nulls() {
        use arrow::array::{Array, StringArray};
        let s = StringArray::from(vec![Some("special requests"), None, Some("plain")]);
        let out = LikeMatcher::classify("%special%").eval(&s);
        assert!(out.value(0));
        assert!(out.is_null(1));
        assert!(!out.value(2));
    }
}
