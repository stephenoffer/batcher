//! Unicode-aware string kernels: normalization, case folding, and grapheme clusters.
//!
//! Each of these answers a question the code-point kernels in `mod.rs` cannot. Two strings
//! that render identically can differ in code points (`é` precomposed against `e` plus a
//! combining acute), so a join or a `group_by` on raw text splits one key in two;
//! [`normalize`] is the fix. `lower` maps case per character and keeps `ß`, so it is not a
//! caseless key; [`casefold`] is. And a ZWJ family emoji is five code points but one
//! user-perceived character, which is what [`grapheme_len`] and [`grapheme_substr`] count.
//!
//! Every function here is locale-independent: the tables are the Unicode Character
//! Database's, never the machine's.

use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, StringArray};
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

use super::{map_str, map_str_borrow};
use crate::{ExprError, StrFunc};

/// The normalization forms `Normalize` accepts in its `pattern` slot. Mirrored by
/// `_NORMAL_FORMS` in `plan/expr_ir/namespaces/strings.py`, which rejects a typo at
/// plan-build time.
pub(crate) const FORMS: [&str; 4] = ["NFC", "NFD", "NFKC", "NFKD"];

/// Normalize every value into `form` (`None` is NFC, DuckDB's `nfc_normalize`).
///
/// An ASCII value is already in all four forms, so it is copied rather than run through the
/// decomposition iterator — the common case on analytic text, and exact.
pub(crate) fn normalize(s: &StringArray, form: Option<&str>) -> Result<ArrayRef, ExprError> {
    let form = form.unwrap_or("NFC");
    let f: fn(&str) -> String = match form {
        "NFC" => |v| v.nfc().collect(),
        "NFD" => |v| v.nfd().collect(),
        "NFKC" => |v| v.nfkc().collect(),
        "NFKD" => |v| v.nfkd().collect(),
        other => {
            return Err(ExprError::InvalidArgument {
                func: format!("{:?}", StrFunc::Normalize),
                reason: format!(
                    "unknown normalization form {other:?}; expected one of {}",
                    FORMS.join(", ")
                ),
            })
        }
    };
    Ok(Arc::new(map_str(s, |v| {
        if v.is_ascii() {
            v.to_string()
        } else {
            f(v)
        }
    })))
}

/// Full Unicode case folding of every value (`caseless::default_case_fold_str`).
pub(crate) fn casefold(s: &StringArray) -> ArrayRef {
    Arc::new(map_str(s, |v| {
        if v.is_ascii() {
            v.to_ascii_lowercase()
        } else {
            caseless::default_case_fold_str(v)
        }
    }))
}

/// The number of extended grapheme clusters in every value (DuckDB `length_grapheme`).
pub(crate) fn grapheme_len(s: &StringArray) -> ArrayRef {
    Arc::new(
        s.iter()
            .map(|o| {
                o.map(|v| {
                    if v.is_ascii() {
                        // `\r\n` is the one ASCII grapheme of two bytes.
                        (v.len() - v.matches("\r\n").count()) as i64
                    } else {
                        v.graphemes(true).count() as i64
                    }
                })
            })
            .collect::<Int64Array>(),
    )
}

/// `substring_grapheme(s, start, length)` over every value.
pub(crate) fn grapheme_substr(s: &StringArray, start: i64, length: Option<i64>) -> ArrayRef {
    Arc::new(map_str_borrow(s, |v| grapheme_slice(v, start, length)))
}

/// One value's grapheme window, as a borrowed slice of it.
///
/// The window is `Substr`'s ([`super::substr_window`]) counted in graphemes, with DuckDB's
/// one difference: a negative `start` that reaches before the first grapheme starts at the
/// first one, where `Substr` would shorten the window by the overshoot. Verified against
/// DuckDB 1.5: `substring_grapheme('abcde', -6, 3)` is `'abc'` and `substring` gives
/// `'ab'`.
fn grapheme_slice(v: &str, start: i64, length: Option<i64>) -> &str {
    // Byte offset of every grapheme boundary, the end included.
    let bounds: Vec<usize> = v
        .grapheme_indices(true)
        .map(|(i, _)| i)
        .chain(std::iter::once(v.len()))
        .collect();
    let n = (bounds.len() - 1) as i64;
    let start = if start < 0 {
        n.saturating_add(start).saturating_add(1).max(1)
    } else {
        start
    };
    let (lo, hi) = super::substr_window(n, start, length);
    if hi <= lo {
        return "";
    }
    &v[bounds[lo]..bounds[hi]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, AsArray};

    fn strs(values: &[Option<&str>]) -> StringArray {
        StringArray::from(values.to_vec())
    }

    fn texts(out: &ArrayRef) -> Vec<Option<String>> {
        out.as_string::<i32>()
            .iter()
            .map(|v| v.map(str::to_string))
            .collect()
    }

    #[test]
    fn normalization_composes_and_decomposes() {
        let decomposed = "e\u{301}";
        let s = strs(&[Some(decomposed), Some("\u{e9}"), Some("ﬁ"), None]);
        let nfc = texts(&normalize(&s, None).unwrap());
        assert_eq!(nfc[0].as_deref(), Some("\u{e9}"));
        assert_eq!(nfc[1].as_deref(), Some("\u{e9}"));
        // A compatibility ligature survives the canonical forms, not the compatibility ones.
        assert_eq!(nfc[2].as_deref(), Some("ﬁ"));
        assert!(nfc[3].is_none());
        let nfd = texts(&normalize(&s, Some("NFD")).unwrap());
        assert_eq!(nfd[1].as_deref(), Some(decomposed));
        let nfkc = texts(&normalize(&s, Some("NFKC")).unwrap());
        assert_eq!(nfkc[2].as_deref(), Some("fi"));
        let nfkd = texts(&normalize(&s, Some("NFKD")).unwrap());
        assert_eq!(nfkd[1].as_deref(), Some(decomposed));
        assert!(normalize(&s, Some("nfc")).is_err());
    }

    #[test]
    fn casefold_is_full_folding_not_lowercase() {
        let s = strs(&[Some("Straße"), Some("ΣΑΣ"), Some("ABC"), None]);
        let out = texts(&casefold(&s));
        assert_eq!(out[0].as_deref(), Some("strasse"));
        // Folding has no final-sigma rule: every sigma folds to `σ`.
        assert_eq!(out[1].as_deref(), Some("σασ"));
        assert_eq!(out[2].as_deref(), Some("abc"));
        assert!(out[3].is_none());
    }

    #[test]
    fn graphemes_count_what_a_reader_sees() {
        let family = "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}";
        let s = strs(&[
            Some(family),
            Some("a\u{301}b"),
            Some("a\r\nb"),
            Some(""),
            None,
        ]);
        let out = grapheme_len(&s);
        let out = out.as_primitive::<arrow::datatypes::Int64Type>();
        assert_eq!(out.value(0), 1);
        assert_eq!(out.value(1), 2);
        assert_eq!(out.value(2), 3);
        assert_eq!(out.value(3), 0);
        assert!(out.is_null(4));
    }

    #[test]
    fn grapheme_substring_matches_duckdb() {
        // Each expectation read from DuckDB 1.5's `substring_grapheme`.
        let v = "abcde";
        for (start, len, want) in [
            (-10, Some(3), "abc"),
            (-6, Some(3), "abc"),
            (-5, Some(3), "abc"),
            (0, Some(2), "a"),
            (0, Some(1), ""),
            (-1, Some(2), "e"),
            (2, Some(-1), "a"),
            (3, Some(-2), "ab"),
            (6, Some(1), ""),
            (1, None, "abcde"),
            (-2, None, "de"),
            (9, None, ""),
        ] {
            assert_eq!(grapheme_slice(v, start, len), want, "({start}, {len:?})");
        }
        let accented = "a\u{301}bc";
        assert_eq!(grapheme_slice(accented, 1, Some(2)), "a\u{301}b");
    }
}
