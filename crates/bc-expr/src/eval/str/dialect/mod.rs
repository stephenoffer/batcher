//! The other engines' reading of a string function Batcher already has.
//!
//! Batcher's string functions follow DuckDB, the differential oracle. Spark, Polars and
//! Daft ship functions under the same names that answer differently on ordinary input:
//! Spark's `initcap` starts a word only after a space, Java's URL codec writes a space as
//! `+`, Polars and Daft expand `$1` in a regex replacement where RE2 inserts it
//! literally, and Daft's Damerau-Levenshtein is the restricted variant. A port that maps
//! the name across gets a plausible wrong answer with no error.
//!
//! Each kernel here is that other reading, reached through its own `StrFunc` tag, which
//! the Python accessor selects from a parameter on the one Batcher spelling
//! (`to_titlecase(boundary="space")`, `url_encode(form=True)`, …). A tag rather than a new
//! field on `Expr::Str` is deliberate: a consumer that translates the IR elsewhere, the
//! device tier first among them, dispatches on the tag, so an unfamiliar one declines to
//! the CPU engine where an unfamiliar field would be read past and ignored.
//!
//! The kernels are pure functions of one value, so the dictionary fast path, the
//! per-row-parameter path and the parallel executor all reach them through `eval_str`
//! unchanged.

use arrow::array::StringArray;

use super::uri_path::url_decode;

/// Spark `initcap`: lowercase everything, then uppercase the first character of the
/// string and each character after an ASCII space.
///
/// Spark lowercases with the whole string as context, which is what turns a word-final
/// `Σ` into `ς`, so each space-delimited segment is lowercased as a unit and only its
/// first character is replaced. Rust has no titlecase mapping, so a word-initial digraph
/// such as `ǆ` becomes `Ǆ` where Spark's ICU mapping gives `ǅ`; no other character has a
/// titlecase form distinct from its uppercase one.
pub(super) fn initcap_space(v: &str) -> String {
    let mut out = String::with_capacity(v.len());
    for (i, segment) in v.split(' ').enumerate() {
        if i > 0 {
            out.push(' ');
        }
        let mut chars = segment.chars();
        let Some(first) = chars.next() else {
            continue;
        };
        out.extend(first.to_uppercase());
        // `str::to_lowercase` maps character by character except for a final sigma, and a
        // segment's first character is never final unless it is the whole segment, so the
        // lowercased first character is exactly the prefix to skip.
        let lowered = segment.to_lowercase();
        let first_lowered_len: usize = first.to_lowercase().map(char::len_utf8).sum();
        out.push_str(&lowered[first_lowered_len..]);
    }
    out
}

/// Java `URLEncoder.encode(s, UTF_8)`, which Spark's `url_encode` calls: letters, digits
/// and `.-*_` stay, a space becomes `+`, and every other UTF-8 byte is `%XX` in uppercase.
pub(super) fn url_encode_form(s: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'*' | b'_') {
            out.push(b as char);
        } else if b == b' ' {
            out.push('+');
        } else {
            out.push('%');
            out.push(HEX[usize::from(b >> 4)] as char);
            out.push(HEX[usize::from(b & 0xf)] as char);
        }
    }
    out
}

/// Java `URLDecoder.decode(s, UTF_8)`, which Spark's `url_decode` calls: `+` is a space,
/// then the `%XX` escapes decode.
///
/// The `+` goes first so an encoded plus (`%2B`) still decodes to `+`. Where Java raises
/// on a malformed escape this leaves it as written, which is the rule the RFC 3986
/// decoder beside it already follows.
pub(super) fn url_decode_form(s: &str) -> String {
    if s.contains('+') {
        url_decode(&s.replace('+', " "))
    } else {
        url_decode(s)
    }
}

/// Optimal String Alignment distance between `a` and `b` over UTF-8 bytes: Levenshtein
/// plus the transposition of two adjacent bytes, with no substring edited twice.
///
/// Bytes, not characters, because the unrestricted sibling counts bytes to match DuckDB
/// and one function should not change its unit with a flag. On non-ASCII text that is a
/// larger count than Daft's, which counts characters.
pub(super) fn osa_distance(a: &str, b: &str) -> usize {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let (n, m) = (a.len(), b.len());
    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }
    let w = m + 1;
    let mut d = vec![0usize; (n + 1) * w];
    for i in 0..=n {
        d[i * w] = i;
    }
    for (j, cell) in d.iter_mut().enumerate().take(m + 1) {
        *cell = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            let mut best = (d[(i - 1) * w + j] + 1)
                .min(d[i * w + j - 1] + 1)
                .min(d[(i - 1) * w + j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                best = best.min(d[(i - 2) * w + j - 2] + 1);
            }
            d[i * w + j] = best;
        }
    }
    d[n * w + m]
}

/// Regex replacement with the `regex` crate's own `$1` / `${name}` / `$$` template syntax
/// — Polars `replace`/`replace_all` and Daft `regexp_replace` call exactly this — where
/// the default kernel reads RE2's `\1`. `global` picks every match over the first.
pub(super) fn replace_dollar(
    s: &StringArray,
    re: &regex::Regex,
    rep: &str,
    global: bool,
) -> StringArray {
    s.iter()
        .map(|o| {
            o.map(|v| {
                if global {
                    re.replace_all(v, rep)
                } else {
                    re.replace(v, rep)
                }
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use arrow::array::Array;

    use super::*;

    #[test]
    fn initcap_space_starts_a_word_only_after_a_space() {
        // Spark's own example (`stringExpressions.scala`, `InitCap`): `sPark sql` → `Spark Sql`.
        assert_eq!(initcap_space("sPark sql"), "Spark Sql");
        assert_eq!(initcap_space("hello-world"), "Hello-world");
        assert_eq!(initcap_space("a\tb"), "A\tb");
        assert_eq!(initcap_space("  two  spaces "), "  Two  Spaces ");
        assert_eq!(initcap_space(""), "");
        assert_eq!(initcap_space("ÉCOLE élève"), "École Élève");
        // A final sigma keeps its context: lowercased with the segment, not alone.
        assert_eq!(
            initcap_space("\u{39f}\u{394}\u{39f}\u{3a3} \u{3a3}\u{39f}"),
            "\u{39f}\u{3b4}\u{3bf}\u{3c2} \u{3a3}\u{3bf}"
        );
    }

    #[test]
    fn form_encoding_is_java_urlencoder() {
        // `URLEncoder.encode("a b+c*~é", UTF_8)` is `a+b%2Bc*%7E%C3%A9`.
        assert_eq!(url_encode_form("a b+c*~é"), "a+b%2Bc*%7E%C3%A9");
        assert_eq!(
            url_encode_form("https://spark.apache.org"),
            "https%3A%2F%2Fspark.apache.org"
        );
        assert_eq!(url_encode_form(""), "");
    }

    #[test]
    fn form_decoding_reads_a_plus_as_a_space_but_not_an_encoded_one() {
        assert_eq!(url_decode_form("a+b%2Bc%20d"), "a b+c d");
        assert_eq!(
            url_decode_form("https%3A%2F%2Fspark.apache.org"),
            "https://spark.apache.org"
        );
        assert_eq!(url_decode_form("100%"), "100%");
        for s in ["a b+c*~é", "", "%+%"] {
            assert_eq!(url_decode_form(&url_encode_form(s)), s);
        }
    }

    #[test]
    fn osa_forbids_editing_a_transposed_pair_again() {
        assert_eq!(osa_distance("ca", "abc"), 3);
        assert_eq!(osa_distance("teh", "the"), 1);
        assert_eq!(osa_distance("", "abc"), 3);
        assert_eq!(osa_distance("abc", ""), 3);
        assert_eq!(osa_distance("kitten", "sitting"), 3);
        assert_eq!(osa_distance("same", "same"), 0);
    }

    #[test]
    fn dollar_templates_expand_groups_and_keep_nulls() {
        let re = regex::Regex::new("(l)").unwrap();
        let s: StringArray = vec![Some("hello"), None, Some("")].into_iter().collect();
        let all = replace_dollar(&s, &re, "[$1]", true);
        assert_eq!(all.value(0), "he[l][l]o");
        assert!(all.is_null(1));
        assert_eq!(all.value(2), "");
        let first = replace_dollar(&s, &re, "[$1]$$", false);
        assert_eq!(first.value(0), "he[l]$lo");
    }
}
