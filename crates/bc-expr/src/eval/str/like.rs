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
        // The three byte-anchored variants dispatch **once**, here, and then run a loop that
        // compares raw bytes. Left inside the per-row closure, the `match self` in `is_match`
        // plus `value(i)`'s `str` wrapper cost 6.5 ns a row for a three-byte prefix, against
        // 0.42 ns to touch every row's bytes and do nothing — so the predicate was an order of
        // magnitude cheaper than the machinery around it. `Contains` and `Segments` keep the
        // generic path: their per-row work is a `memmem` search that dwarfs the dispatch. The two
        // that require a substring take one search over the whole buffer instead (`scanned`).
        let values = match self {
            LikeMatcher::StartsWith(p) => {
                let k = p.as_bytes();
                crate::eval::cmp::starts_with_short(s.value_offsets(), s.value_data(), k)
                    .unwrap_or_else(|| {
                        anchored(s, |d, a, b| b - a >= k.len() && &d[a..a + k.len()] == k)
                    })
            }
            LikeMatcher::EndsWith(p) => {
                let k = p.as_bytes();
                anchored(s, |d, a, b| b - a >= k.len() && &d[b - k.len()..b] == k)
            }
            LikeMatcher::Exact(p) => {
                let k = p.as_bytes();
                anchored(s, |d, a, b| b - a == k.len() && &d[a..b] == k)
            }
            // An empty needle matches everywhere, an empty row included, so it has no position to
            // find and keeps the per-row path (`contains(s, '')`).
            LikeMatcher::Contains(f) if !f.needle().is_empty() => scanned(s, f, |_, _, _| true),
            LikeMatcher::Segments {
                prefix,
                suffix,
                middles,
            } if middles.first().is_some_and(|m| !m.needle().is_empty()) => {
                scanned(s, &middles[0], |d, a, b| {
                    // `d[a..b]` is one row of a `StringArray`, so the conversion cannot fail.
                    std::str::from_utf8(&d[a..b])
                        .is_ok_and(|row| segment_match(row, prefix, suffix, middles))
                })
            }
            _ => BooleanBuffer::collect_bool(s.len(), |i| self.is_match(s.value(i))),
        };
        BooleanArray::new(values, s.nulls().cloned())
    }
}

/// The match mask for a needle anchored at one end of each row, over the raw buffers.
///
/// `hit` receives the row's byte range and answers for it, and the caller passes a *different*
/// closure per variant rather than a shared one branching on an enum: with the branch inside the
/// loop the row cost was 5.9 ns against 3.1 ns for the identical loop written out, because the
/// discriminant test is per row and does not hoist. `value(i)`'s `str` wrapper was the other
/// half — the same predicate through it costs 6.5 ns.
///
/// Byte-oriented and therefore identical to the `str` comparison it replaces: the haystack is
/// valid UTF-8, the needle is a whole UTF-8 substring, and a byte match can only land on a char
/// boundary (UTF-8 self-synchronization) — the same argument `segment_match` rests on.
#[inline(always)]
fn anchored(s: &StringArray, hit: impl Fn(&[u8], usize, usize) -> bool) -> BooleanBuffer {
    let data = s.value_data();
    let offsets = s.value_offsets();
    BooleanBuffer::collect_bool(s.len(), |i| {
        hit(data, offsets[i] as usize, offsets[i + 1] as usize)
    })
}

/// The match mask for a pattern that requires substring `first` somewhere in the row, found by
/// searching the column's value buffer once instead of every row separately.
///
/// Every row that can match contains `first`, so rows are visited only where the search lands:
/// each hit is mapped to its row through the offsets, a hit that runs past its row's end is a
/// false join of two neighbours and the search resumes one byte on, and a hit inside a row hands
/// that row to `rest`, which decides it exactly and is the only thing that ever marks a row. So
/// the mask equals `rest` applied to every row containing `first`, and a row without it is false,
/// which is what the per-row matcher returns for it.
///
/// Why: the per-row search set its SIMD searcher up for every ~50-byte row. TPC-H q13's
/// `o_comment NOT LIKE '%special%requests%'` over 15M rows measured 101 ms against DuckDB's 29 ms
/// in memory; one pass over the buffer is bandwidth-bound, and here about 1% of rows reach `rest`.
fn scanned(
    s: &StringArray,
    first: &Finder<'_>,
    rest: impl Fn(&[u8], usize, usize) -> bool,
) -> BooleanBuffer {
    let n = s.len();
    let data = s.value_data();
    let offsets = s.value_offsets();
    let mut bits = arrow::array::BooleanBufferBuilder::new(n);
    bits.append_n(n, false);
    if n == 0 {
        return bits.finish();
    }
    let (mut pos, end) = (offsets[0] as usize, offsets[n] as usize);
    let width = first.needle().len();
    let mut row = 0usize;
    while pos < end {
        let Some(found) = first.find(&data[pos..end]) else {
            break;
        };
        let hit = pos + found;
        while row < n && offsets[row + 1] as usize <= hit {
            row += 1;
        }
        if row == n {
            break;
        }
        let (a, b) = (offsets[row] as usize, offsets[row + 1] as usize);
        if hit + width <= b {
            if rest(data, a, b) {
                bits.set_bit(row, true);
            }
            pos = b;
            row += 1;
        } else {
            pos = hit + 1;
        }
    }
    bits.finish()
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

    /// The single-pass column scan returns exactly what the per-row matcher returns: needles that
    /// straddle two rows (which a buffer search finds and must reject), a needle repeated within
    /// and across rows, empty rows, nulls, and a sliced array whose offsets do not start at zero.
    #[test]
    fn the_buffer_scan_agrees_with_the_per_row_matcher() {
        use arrow::array::{Array, StringArray};
        let rows: Vec<Option<&str>> = vec![
            Some("xx spe"),
            Some("cial requests"),
            Some("special requests"),
            None,
            Some(""),
            Some("specialspecial requestsrequests"),
            Some("requests special"),
            Some("special"),
            Some(" requests"),
            Some("a special b requests c"),
            Some("specia"),
            Some("l"),
        ];
        let full = StringArray::from(rows);
        let matchers = [
            "%special%",
            "%special%requests%",
            "%requests%",
            "sp%requests%",
            "%l",
        ]
        .iter()
        .map(|p| ((*p).to_string(), LikeMatcher::classify(p)))
        .chain(std::iter::once((
            "contains ''".to_string(),
            LikeMatcher::contains(""),
        )));
        for (pattern, m) in matchers {
            for (offset, len) in [(0, full.len()), (1, full.len() - 1), (2, 7), (5, 0)] {
                let arr = full.slice(offset, len);
                let got = m.eval(&arr);
                for i in 0..arr.len() {
                    assert_eq!(got.is_null(i), arr.is_null(i), "{pattern} @{offset}+{i}");
                    if !arr.is_null(i) {
                        assert_eq!(
                            got.value(i),
                            m.is_match(arr.value(i)),
                            "{pattern} on {:?}",
                            arr.value(i)
                        );
                    }
                }
            }
        }
    }
}
