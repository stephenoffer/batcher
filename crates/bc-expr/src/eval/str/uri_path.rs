//! URL escaping, filesystem-path decomposition, binary text, and the two string
//! distances DuckDB spells `hamming`/`mismatches` and `jaccard`.
//!
//! These are grouped because they are the string functions whose *definition* is an
//! external specification rather than an operation on characters — RFC 3986 for the URL
//! pair, POSIX path syntax for the `parse_*` family, and DuckDB's own choices for the
//! rest. Each was checked against DuckDB rather than derived, because the interesting
//! part of every one of them is an edge case:
//!
//! * `url_decode('a%2')` is `'a%2'` in DuckDB — a malformed escape passes through
//!   unchanged, it does not raise and does not null the row.
//! * `parse_dirname` is the *first* path component (`/` for an absolute path), not the
//!   directory; that is `parse_dirpath`. DuckDB has both, and they disagree on every
//!   path deeper than one level.
//! * `hamming` on unequal-length strings is an error in DuckDB, not a silent truncation.
//! * `jaccard` is over the two strings' character *sets*, so `jaccard('aab', 'ab')` is 1.

/// The RFC 3986 unreserved set: these survive `url_encode` as themselves. Everything
/// else — including `/`, `+` and `&` — is percent-encoded, because this encodes a URL
/// *component*, not a whole URL.
#[inline]
fn is_unreserved(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~')
}

/// Percent-encode `s` (DuckDB `url_encode`), over its UTF-8 bytes.
pub(super) fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        if is_unreserved(b) {
            out.push(b as char);
        } else {
            out.push('%');
            out.push(
                char::from_digit((b >> 4) as u32, 16)
                    .unwrap()
                    .to_ascii_uppercase(),
            );
            out.push(
                char::from_digit((b & 0xf) as u32, 16)
                    .unwrap()
                    .to_ascii_uppercase(),
            );
        }
    }
    out
}

/// Percent-decode `s` (DuckDB `url_decode`).
///
/// A `%` not followed by two hex digits is emitted verbatim, and if the decoded bytes are
/// not valid UTF-8 the input is returned unchanged — both are DuckDB's behaviour, and
/// both are why this cannot simply fail.
pub(super) fn url_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = (bytes[i + 1] as char).to_digit(16);
            let lo = (bytes[i + 2] as char).to_digit(16);
            if let (Some(hi), Some(lo)) = (hi, lo) {
                out.push((hi * 16 + lo) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| s.to_string())
}

/// Escape the regex metacharacters in `s` (DuckDB `regexp_escape`).
///
/// This is RE2's `QuoteMeta` rule, which DuckDB inherits, and **not** the `regex`
/// crate's `escape`: RE2 backslashes every ASCII byte outside `[A-Za-z0-9_]`, where
/// `regex::escape` escapes only what its own syntax treats as special. Both produce a
/// pattern that matches the literal, so the difference is invisible if you immediately
/// match with it — but the *returned string* is the result a user sees, stores and
/// compares, and `regex::escape` returned `'a b'` where DuckDB returns `'a\ b'`.
///
/// Escaping the wider set is safe for the engine's own matcher: the `regex` crate
/// accepts a backslash before any punctuation as that literal character (verified for
/// space, `%`, `/` and `-`, which `regex::escape` leaves bare).
///
/// Bytes at or above `0x80` are left alone, so a UTF-8 sequence is never split by a
/// backslash — the same carve-out RE2 makes.
pub(super) fn regexp_escape(s: &str) -> String {
    let mut out: Vec<u8> = Vec::with_capacity(s.len() * 2);
    for &b in s.as_bytes() {
        if b < 0x80 && !(b.is_ascii_alphanumeric() || b == b'_') {
            out.push(b'\\');
        }
        out.push(b);
    }
    // Only ASCII backslashes were inserted, and never inside a multi-byte sequence
    // (those bytes are all >= 0x80), so the result is still valid UTF-8.
    String::from_utf8(out).expect("escaping only inserts ASCII backslashes outside sequences")
}

/// The final component of a path (DuckDB `parse_filename`).
pub(super) fn parse_filename(s: &str) -> &str {
    match s.rfind(is_separator) {
        Some(i) => &s[i + 1..],
        None => s,
    }
}

/// The *first* component of a path (DuckDB `parse_dirname`): `/` for an absolute POSIX
/// path, the leading directory otherwise, and the empty string when there is no
/// separator at all.
pub(super) fn parse_dirname(s: &str) -> &str {
    match s.find(is_separator) {
        Some(0) => &s[..1],
        Some(i) => &s[..i],
        None => "",
    }
}

/// Everything before the last separator (DuckDB `parse_dirpath`); empty when there is
/// none.
///
/// The root is the one case that is not "everything before the last separator": `/`
/// would give the empty string, and DuckDB returns `/` — the root is its own directory.
/// `/single` really is `''`, and `//` really is `/`, so the carve-out is exactly the
/// one-character path, not "a leading separator".
pub(super) fn parse_dirpath(s: &str) -> &str {
    match s.rfind(is_separator) {
        Some(0) if s.len() == 1 => s,
        Some(i) => &s[..i],
        None => "",
    }
}

/// A path split into components (DuckDB `parse_path`), keeping a leading separator as
/// its own first element so an absolute path is distinguishable from a relative one.
pub(super) fn parse_path(s: &str) -> Vec<&str> {
    if s.is_empty() {
        return Vec::new();
    }
    let (lead, rest) = if s.starts_with(is_separator) {
        (Some(&s[..1]), &s[1..])
    } else {
        (None, s)
    };
    let mut out: Vec<&str> = lead.into_iter().collect();
    out.extend(rest.split(is_separator).filter(|p| !p.is_empty()));
    out
}

/// Both separators are accepted, as DuckDB does, so a Windows path parses too.
#[inline]
fn is_separator(c: char) -> bool {
    c == '/' || c == '\\'
}

/// The UTF-8 bytes of `s` as `0`/`1` characters, 8 per byte, MSB first (DuckDB
/// `to_binary`).
pub(super) fn to_binary(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 8);
    for &b in s.as_bytes() {
        for shift in (0..8).rev() {
            out.push(if (b >> shift) & 1 == 1 { '1' } else { '0' });
        }
    }
    out
}

/// Inverse of [`to_binary`]; `None` when the input is not a whole number of 8 `0`/`1`
/// characters or the bytes are not UTF-8 (the row becomes null, as `unhex` does).
pub(super) fn from_binary(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    if bytes.is_empty() || bytes.len() % 8 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks(8) {
        let mut byte = 0u8;
        for &c in chunk {
            byte = match c {
                b'0' => byte << 1,
                b'1' => (byte << 1) | 1,
                _ => return None,
            };
        }
        out.push(byte);
    }
    String::from_utf8(out).ok()
}

/// Hamming distance — the number of positions at which `a` and `b` differ.
///
/// `None` when the two have different character counts; DuckDB raises there rather than
/// comparing a prefix, and silently comparing one would turn a caller's bug into a
/// plausible number. Counted in Unicode scalar values, as `Len`/`Substr` count.
pub(super) fn hamming(a: &str, b: &str) -> Option<i64> {
    let (mut ca, mut cb) = (a.chars(), b.chars());
    let mut distance = 0i64;
    loop {
        match (ca.next(), cb.next()) {
            (None, None) => return Some(distance),
            (Some(x), Some(y)) => distance += i64::from(x != y),
            _ => return None, // different lengths
        }
    }
}

/// Jaccard similarity of the two strings' character *sets* — |A ∩ B| / |A ∪ B|.
///
/// Two empty strings are 1.0 (identical), matching DuckDB rather than producing 0/0.
pub(super) fn jaccard(a: &str, b: &str) -> f64 {
    use std::collections::BTreeSet;
    let sa: BTreeSet<char> = a.chars().collect();
    let sb: BTreeSet<char> = b.chars().collect();
    let union = sa.union(&sb).count();
    if union == 0 {
        return 1.0;
    }
    sa.intersection(&sb).count() as f64 / union as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every expectation here is DuckDB's answer, read from a live DuckDB rather than
    /// derived from the specification — the two differ in exactly the places that matter.
    #[test]
    fn matches_duckdbs_answers_on_the_edge_cases() {
        // A URL *component*: `/` and `+` are escaped, the unreserved set is not.
        assert_eq!(url_encode("a b/c+d~e_f.g"), "a%20b%2Fc%2Bd~e_f.g");
        assert_eq!(url_decode("a%20b%2Fc"), "a b/c");
        // A truncated escape passes through instead of raising or nulling.
        assert_eq!(url_decode("a%2"), "a%2");
        assert_eq!(url_decode("100%"), "100%");

        // RE2's QuoteMeta escapes every ASCII byte outside [A-Za-z0-9_], which is wider
        // than the `regex` crate's own escaper — the space and `/` here are the tell.
        assert_eq!(regexp_escape("a.b*c[d]"), r"a\.b\*c\[d\]");
        assert_eq!(regexp_escape("a b"), r"a\ b");
        assert_eq!(regexp_escape("a/b"), r"a\/b");
        assert_eq!(regexp_escape("a%b"), r"a\%b");
        // Bytes >= 0x80 are left alone so a UTF-8 sequence is never split.
        assert_eq!(regexp_escape("é"), "é");

        // `parse_dirname` is the first component, `parse_dirpath` everything before the
        // filename — they agree only on a one-level relative path.
        assert_eq!(parse_dirname("/x/y/z.csv"), "/");
        assert_eq!(parse_dirpath("/x/y/z.csv"), "/x/y");
        assert_eq!(parse_filename("/x/y/z.csv"), "z.csv");
        assert_eq!(parse_path("/x/y/z.csv"), vec!["/", "x", "y", "z.csv"]);
        assert_eq!(parse_dirname("a/b/c.txt"), "a");
        assert_eq!(parse_dirpath("a/b/c.txt"), "a/b");
        assert_eq!(parse_path("a/b/c.txt"), vec!["a", "b", "c.txt"]);
        // No separator: the whole value is the filename and there is no directory.
        assert_eq!(parse_filename("noslash"), "noslash");
        assert_eq!(parse_dirname("noslash"), "");
        assert_eq!(parse_dirpath("noslash"), "");
        // The root is its own directory; a one-level absolute path has none. Both are
        // DuckDB's answers and the two disagree, which is why the carve-out is the
        // one-character path rather than "starts with a separator".
        assert_eq!(parse_dirpath("/"), "/");
        assert_eq!(parse_dirpath("/single"), "");
        assert_eq!(parse_dirpath("//"), "/");
        assert_eq!(parse_dirpath("//a"), "/");
        assert_eq!(parse_dirpath("a/b/"), "a/b");
        assert_eq!(parse_path("/"), vec!["/"]);
        assert_eq!(parse_path("//a"), vec!["/", "a"]);
        assert!(parse_path("").is_empty());

        assert_eq!(to_binary("a"), "01100001");
        assert_eq!(from_binary("01100001").as_deref(), Some("a"));
        // Not a whole number of bytes, and not a binary digit → null, not an error.
        assert_eq!(from_binary("0110000"), None);
        assert_eq!(from_binary("0110000x"), None);

        assert_eq!(hamming("abc", "abd"), Some(1));
        assert_eq!(hamming("abc", "abc"), Some(0));
        assert_eq!(hamming("abc", "ab"), None); // DuckDB raises on unequal lengths

        // Character *sets*: the repeated 'a' does not change the answer.
        assert_eq!(jaccard("abc", "abd"), 0.5);
        assert_eq!(jaccard("aab", "ab"), 1.0);
        assert_eq!(jaccard("", ""), 1.0);
    }

    /// A round trip is the property that pins both halves of each pair at once, over
    /// input the hand-written cases would not think to include.
    #[test]
    fn the_encodings_round_trip() {
        for s in [
            "",
            "plain",
            "a b/c",
            "100%",
            "ünïcodé",
            "🙂 emoji",
            "tab\tnewline\n",
            "a&b=c?d#e",
        ] {
            assert_eq!(url_decode(&url_encode(s)), s, "url round trip for {s:?}");
            assert_eq!(
                from_binary(&to_binary(s)).as_deref(),
                (!s.is_empty()).then_some(s)
            );
        }
    }
}
