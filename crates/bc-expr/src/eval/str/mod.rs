//! String-function evaluation for `Expr::Str` (split out of `lib.rs`).

use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanArray, Int64Array, StringArray};
use arrow::compute::cast;
use arrow::datatypes::DataType;

use crate::{ExprError, StrFunc};

mod chunk;
mod html;
mod json;
mod minhash;
mod regex_cache;

/// Evaluate a string function over a Utf8 array (preserving nulls).
pub(crate) fn eval_str(
    func: StrFunc,
    arr: &ArrayRef,
    pattern: Option<&str>,
    replacement: Option<&str>,
    start: Option<i64>,
    length: Option<i64>,
) -> Result<ArrayRef, ExprError> {
    // A `Binary`-typed column (how ClickBench's `hits` string columns arrive — a
    // `BYTE_ARRAY` with no UTF-8 logical annotation) is coerced to `Utf8` so string
    // functions apply, matching DuckDB's VARCHAR treatment. `Binary -> Utf8` validates
    // UTF-8 and errors on genuinely non-textual bytes (correct — a string function over
    // real binary has no defined result).
    let coerced;
    let arr = match arr.data_type() {
        DataType::Binary | DataType::LargeBinary => {
            coerced = cast(arr, &DataType::Utf8)?;
            &coerced
        }
        _ => arr,
    };
    let s =
        arr.as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| ExprError::ExpectedString {
                func: format!("{func:?}"),
                got: arr.data_type().to_string(),
            })?;

    let out: ArrayRef = match func {
        StrFunc::Upper => Arc::new(map_str(s, |v| v.to_uppercase())),
        StrFunc::Lower => Arc::new(map_str(s, |v| v.to_lowercase())),
        StrFunc::Len => Arc::new(
            s.iter()
                .map(|o| o.map(|v| v.chars().count() as i64))
                .collect::<Int64Array>(),
        ),
        StrFunc::Contains => {
            let pat = require_pattern(pattern, func)?;
            Arc::new(map_bool(s, |v| v.contains(pat)))
        }
        StrFunc::StartsWith => {
            let pat = require_pattern(pattern, func)?;
            Arc::new(map_bool(s, |v| v.starts_with(pat)))
        }
        StrFunc::EndsWith => {
            let pat = require_pattern(pattern, func)?;
            Arc::new(map_bool(s, |v| v.ends_with(pat)))
        }
        StrFunc::Substr => {
            // SQL semantics: 1-based start; `length` optional (to end of string).
            // Borrowing variant: the result is always a slice of the input, so it
            // avoids the per-row `Vec<char>` + `String` allocation the generic
            // `map_str` closure would force (the array builder copies the slices once).
            let start = start.unwrap_or(1);
            Arc::new(map_str_borrow(s, |v| substr_slice(v, start, length)))
        }
        StrFunc::Replace => {
            let pat = require_pattern(pattern, func)?;
            let rep = replacement.ok_or_else(|| ExprError::MissingArgument {
                func: format!("{func:?}"),
                arg: "replacement",
            })?;
            // An empty search string matches nothing (DuckDB `replace(s, '', r)` = `s`).
            // Rust's `str::replace("")` instead splices `rep` between every character.
            if pat.is_empty() {
                Arc::new(map_str_borrow(s, |v| v))
            } else {
                Arc::new(map_str(s, |v| v.replace(pat, rep)))
            }
        }
        // With a `pattern`, trim that set of characters (DuckDB `trim(s, chars)` /
        // Polars `strip_chars`); without one, trim ASCII/Unicode whitespace.
        StrFunc::Trim => match pattern {
            Some(chars) => {
                let set: Vec<char> = chars.chars().collect();
                Arc::new(map_str(s, |v| {
                    v.trim_matches(|c| set.contains(&c)).to_string()
                }))
            }
            None => Arc::new(map_str(s, |v| {
                v.trim_matches(is_space_separator).to_string()
            })),
        },
        StrFunc::LTrim => match pattern {
            Some(chars) => {
                let set: Vec<char> = chars.chars().collect();
                Arc::new(map_str(s, |v| {
                    v.trim_start_matches(|c| set.contains(&c)).to_string()
                }))
            }
            None => Arc::new(map_str(s, |v| {
                v.trim_start_matches(is_space_separator).to_string()
            })),
        },
        StrFunc::RTrim => match pattern {
            Some(chars) => {
                let set: Vec<char> = chars.chars().collect();
                Arc::new(map_str(s, |v| {
                    v.trim_end_matches(|c| set.contains(&c)).to_string()
                }))
            }
            None => Arc::new(map_str(s, |v| {
                v.trim_end_matches(is_space_separator).to_string()
            })),
        },
        StrFunc::Reverse => Arc::new(map_str(s, |v| v.chars().rev().collect())),
        StrFunc::Repeat => {
            let n = start.unwrap_or(0).max(0) as usize;
            // Guard against `repeat(s, 1e9)` overflowing Arrow's 32-bit string offsets:
            // a clean error, not an allocator abort / offset-overflow panic.
            Arc::new(map_str_checked(s, func, |v| {
                v.len()
                    .checked_mul(n)
                    .filter(|&t| t <= MAX_STR_BYTES)
                    .map(|_| v.repeat(n))
            })?)
        }
        StrFunc::Lpad => {
            let width = start.unwrap_or(0).max(0) as usize;
            let fill = pattern.unwrap_or(" ");
            Arc::new(map_str_checked(s, func, |v| {
                pad_checked(v, width, fill, true)
            })?)
        }
        StrFunc::Rpad => {
            let width = start.unwrap_or(0).max(0) as usize;
            let fill = pattern.unwrap_or(" ");
            Arc::new(map_str_checked(s, func, |v| {
                pad_checked(v, width, fill, false)
            })?)
        }
        StrFunc::Position => {
            let pat = require_pattern(pattern, func)?;
            Arc::new(
                s.iter()
                    .map(|o| o.map(|v| char_position(v, pat)))
                    .collect::<Int64Array>(),
            )
        }
        StrFunc::Right => {
            // SQL `right(s, n)`: the last `n` characters for `n >= 0`; for `n < 0`,
            // everything *except* the first `|n|` characters (DuckDB semantics), e.g.
            // `right('abcdef', -2) = 'cdef'`. A naive `.max(0)` dropped the negative
            // case to the empty string.
            let n = start.unwrap_or(0);
            Arc::new(map_str(s, |v| {
                let chars: Vec<char> = v.chars().collect();
                let len = chars.len();
                let begin = if n < 0 {
                    (n.unsigned_abs() as usize).min(len)
                } else {
                    len.saturating_sub(n as usize)
                };
                chars[begin..].iter().collect()
            }))
        }
        StrFunc::Ascii => Arc::new(
            s.iter()
                .map(|o| o.map(|v| v.chars().next().map_or(0i64, |c| c as i64)))
                .collect::<Int64Array>(),
        ),
        StrFunc::RegexpMatches => {
            let re = compile_regex(pattern, func)?;
            Arc::new(map_bool(s, |v| re.is_match(v)))
        }
        StrFunc::Like | StrFunc::Ilike => {
            let pat = require_pattern(pattern, func)?;
            let re = like_regex(pat, matches!(func, StrFunc::Ilike))?;
            Arc::new(map_bool(s, |v| re.is_match(v)))
        }
        StrFunc::RegexpReplace => {
            let re = compile_regex(pattern, func)?;
            let rep = replacement.ok_or_else(|| ExprError::MissingArgument {
                func: format!("{func:?}"),
                arg: "replacement",
            })?;
            Arc::new(regexp_replace_with(s, &re, rep, false))
        }
        StrFunc::RegexpReplaceAll => {
            let re = compile_regex(pattern, func)?;
            let rep = replacement.ok_or_else(|| ExprError::MissingArgument {
                func: format!("{func:?}"),
                arg: "replacement",
            })?;
            Arc::new(regexp_replace_with(s, &re, rep, true))
        }
        StrFunc::SplitPart => {
            let delim = require_pattern(pattern, func)?;
            let n = start.unwrap_or(1);
            Arc::new(map_str_borrow(s, |v| split_part(v, delim, n)))
        }
        StrFunc::RegexpExtract => {
            let re = compile_regex(pattern, func)?;
            let group = start.unwrap_or(0).max(0) as usize;
            Arc::new(map_str(s, |v| {
                re.captures(v)
                    .and_then(|c| c.get(group))
                    .map_or(String::new(), |m| m.as_str().to_string())
            }))
        }
        StrFunc::JsonExtractString => {
            let path = json::parse_path(require_pattern(pattern, func)?);
            // Nullable result: null where input is not valid JSON or path is absent.
            Arc::new(
                s.iter()
                    .map(|o| o.and_then(|v| json::extract_string(v, &path)))
                    .collect::<StringArray>(),
            )
        }
        StrFunc::JsonExtractInt => {
            let path = json::parse_path(require_pattern(pattern, func)?);
            Arc::new(
                s.iter()
                    .map(|o| o.and_then(|v| json::extract_int(v, &path)))
                    .collect::<Int64Array>(),
            )
        }
        StrFunc::JsonExtractFloat => {
            use arrow::array::Float64Array;
            let path = json::parse_path(require_pattern(pattern, func)?);
            Arc::new(
                s.iter()
                    .map(|o| o.and_then(|v| json::extract_float(v, &path)))
                    .collect::<Float64Array>(),
            )
        }
        StrFunc::JsonExtractBool => {
            let path = json::parse_path(require_pattern(pattern, func)?);
            Arc::new(
                s.iter()
                    .map(|o| o.and_then(|v| json::extract_bool(v, &path)))
                    .collect::<BooleanArray>(),
            )
        }
        StrFunc::Hash64 => Arc::new(
            s.iter()
                .map(|o| o.map(|v| fnv1a64(v.as_bytes()) as i64))
                .collect::<Int64Array>(),
        ),
        StrFunc::Initcap => Arc::new(map_str(s, initcap)),
        StrFunc::OctetLength => Arc::new(
            s.iter()
                .map(|o| o.map(|v| v.len() as i64))
                .collect::<Int64Array>(),
        ),
        StrFunc::BitLength => Arc::new(
            s.iter()
                .map(|o| o.map(|v| (v.len() as i64) * 8))
                .collect::<Int64Array>(),
        ),
        // Data protection (keyed hash / encryption / redaction) — the arms that need a
        // per-array key schedule and a key that never reaches an error message.
        StrFunc::HmacSha256 | StrFunc::AesEncrypt | StrFunc::AesDecrypt | StrFunc::Mask => {
            super::security::eval_security(func, s, pattern, start, length)?
        }
        StrFunc::Hex => Arc::new(map_str(s, hex_encode)),
        StrFunc::Md5 => {
            use md5::{Digest, Md5};
            Arc::new(map_str(s, |v| {
                hex_lower(Md5::digest(v.as_bytes()).as_slice())
            }))
        }
        StrFunc::Sha1 => {
            use sha1::{Digest, Sha1};
            Arc::new(map_str(s, |v| {
                hex_lower(Sha1::digest(v.as_bytes()).as_slice())
            }))
        }
        StrFunc::Sha256 => {
            use sha2::{Digest, Sha256};
            Arc::new(map_str(s, |v| {
                hex_lower(Sha256::digest(v.as_bytes()).as_slice())
            }))
        }
        StrFunc::Crc32 => Arc::new(
            s.iter()
                .map(|o| o.map(|v| crc32fast::hash(v.as_bytes()) as i64))
                .collect::<Int64Array>(),
        ),
        StrFunc::XxHash64 => Arc::new(
            s.iter()
                .map(|o| o.map(|v| xxhash64(v.as_bytes()) as i64))
                .collect::<Int64Array>(),
        ),
        StrFunc::Base64 => {
            use base64::Engine as _;
            Arc::new(map_str(s, |v| {
                base64::engine::general_purpose::STANDARD.encode(v.as_bytes())
            }))
        }
        StrFunc::FromBase64 => {
            use base64::Engine as _;
            // Nullable: invalid base64 or non-UTF-8 decoded bytes → null.
            Arc::new(
                s.iter()
                    .map(|o| {
                        o.and_then(|v| {
                            base64::engine::general_purpose::STANDARD
                                .decode(v)
                                .ok()
                                .and_then(|b| String::from_utf8(b).ok())
                        })
                    })
                    .collect::<StringArray>(),
            )
        }
        StrFunc::Unhex => {
            // Nullable: odd length, non-hex, or non-UTF-8 decoded bytes → null.
            Arc::new(
                s.iter()
                    .map(|o| o.and_then(|v| hex_decode(v).and_then(|b| String::from_utf8(b).ok())))
                    .collect::<StringArray>(),
            )
        }
        StrFunc::Translate => {
            let from = require_pattern(pattern, func)?;
            let to = replacement.ok_or_else(|| ExprError::MissingArgument {
                func: format!("{func:?}"),
                arg: "replacement",
            })?;
            // Build from-char → Option<to-char> (None = delete). First mapping for a
            // given source char wins (matches DuckDB).
            let to_chars: Vec<char> = to.chars().collect();
            let mut map: std::collections::HashMap<char, Option<char>> =
                std::collections::HashMap::new();
            for (i, fc) in from.chars().enumerate() {
                map.entry(fc).or_insert_with(|| to_chars.get(i).copied());
            }
            Arc::new(map_str(s, |v| {
                v.chars()
                    .filter_map(|c| match map.get(&c) {
                        Some(Some(rc)) => Some(*rc),
                        Some(None) => None,
                        None => Some(c),
                    })
                    .collect()
            }))
        }
        StrFunc::Split => {
            use arrow::array::{Array, ListBuilder, StringBuilder};
            let delim = require_pattern(pattern, func)?;
            // One list per row; the parts together hold ~the input bytes (minus delimiters),
            // so pre-size both the offset buffer and the value bytes to skip builder regrowth.
            let mut builder = ListBuilder::with_capacity(
                StringBuilder::with_capacity(s.len(), s.value_data().len()),
                s.len(),
            );
            for o in s.iter() {
                match o {
                    Some(v) => {
                        // An empty delimiter splits into individual characters (DuckDB
                        // `string_split(s, '')`), not the `["", …, ""]` that Rust's
                        // `str::split("")` yields. An empty string stays a single `[""]`.
                        if delim.is_empty() {
                            if v.is_empty() {
                                builder.values().append_value("");
                            } else {
                                let mut buf = [0u8; 4];
                                for c in v.chars() {
                                    builder.values().append_value(c.encode_utf8(&mut buf));
                                }
                            }
                        } else {
                            for part in v.split(delim) {
                                builder.values().append_value(part);
                            }
                        }
                        builder.append(true);
                    }
                    None => builder.append(false),
                }
            }
            Arc::new(builder.finish())
        }
        StrFunc::Chunk => chunk::eval_chunk(s, start, length)?,
        StrFunc::MinHash => minhash::eval_minhash(s, start, length)?,
        StrFunc::StripHtml => Arc::new(map_str(s, html::strip_html_text)),
        StrFunc::SubstringIndex => {
            let delim = require_pattern(pattern, func)?;
            let count = start.unwrap_or(0);
            Arc::new(map_str(s, |v| substring_index(v, delim, count)))
        }
        StrFunc::Overlay => {
            let rep = replacement.ok_or_else(|| ExprError::MissingArgument {
                func: format!("{func:?}"),
                arg: "replacement",
            })?;
            let pos = start.unwrap_or(1);
            Arc::new(map_str(s, |v| overlay(v, rep, pos, length)))
        }
        StrFunc::RegexpExtractAll => {
            use arrow::array::{Array, ListBuilder, StringBuilder};
            let re = compile_regex(pattern, func)?;
            // One list per row; match volume per row is unknown, so pre-size only the
            // outer offset buffer and let the inner value builder grow as matches land.
            let mut builder = ListBuilder::with_capacity(StringBuilder::new(), s.len());
            for o in s.iter() {
                match o {
                    Some(v) => {
                        for m in re.find_iter(v) {
                            builder.values().append_value(m.as_str());
                        }
                        builder.append(true);
                    }
                    None => builder.append(false),
                }
            }
            Arc::new(builder.finish())
        }
        StrFunc::RegexpCount => {
            let re = compile_regex(pattern, func)?;
            Arc::new(
                s.iter()
                    .map(|o| o.map(|v| re.find_iter(v).count() as i64))
                    .collect::<Int64Array>(),
            )
        }
        StrFunc::Levenshtein => {
            let target = require_pattern(pattern, func)?;
            Arc::new(
                s.iter()
                    .map(|o| o.map(|v| levenshtein(v, target) as i64))
                    .collect::<Int64Array>(),
            )
        }
        StrFunc::Soundex => Arc::new(map_str(s, soundex)),
    };
    Ok(out)
}

/// Whether `c` is a Unicode space separator (general category `Zs`).
///
/// DuckDB's argument-less `trim`/`ltrim`/`rtrim` strip exactly the `Zs` category —
/// the space *separators* (ASCII space, NBSP, the em/en spaces, ideographic space,
/// …) — and NOT the C0 control whitespace (tab, newline, carriage return, vertical
/// tab, form feed) that Rust's `str::trim` (Unicode `White_Space`) also removes.
/// Matching DuckDB means, e.g., `trim("\t\n")` keeps the tab and newline.
fn is_space_separator(c: char) -> bool {
    matches!(
        c,
        '\u{0020}' | '\u{00A0}' | '\u{1680}' | '\u{2000}'
            ..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}'
    )
}

/// `substring_index(s, delim, count)` — the part of `s` before the `count`-th
/// occurrence of `delim`. Positive `count` counts delimiters from the left,
/// negative from the right; `0` yields the empty string (Spark semantics).
fn substring_index(s: &str, delim: &str, count: i64) -> String {
    if count == 0 || delim.is_empty() {
        return String::new();
    }
    let parts: Vec<&str> = s.split(delim).collect();
    let n = parts.len();
    if count > 0 {
        // `count as usize` is safe (count > 0); clamp to the number of parts. The old
        // `-count` overflowed for `count == i64::MIN`, panicking on the slice index.
        let take = (count as usize).min(n);
        parts[..take].join(delim)
    } else {
        let take = (count.unsigned_abs() as usize).min(n);
        parts[n - take..].join(delim)
    }
}

/// `split_part(s, delim, n)` — the `n`-th field (1-based) of `s` split on `delim`.
/// A negative `n` counts fields from the right (DuckDB/Spark). An empty `delim` splits
/// into characters. Out-of-range or `n == 0` yields `""`.
fn split_part<'a>(s: &'a str, delim: &str, n: i64) -> &'a str {
    if n == 0 {
        return "";
    }
    if delim.is_empty() {
        let count = s.chars().count() as i64;
        let idx = if n < 0 { count + n } else { n - 1 };
        if idx < 0 || idx >= count {
            return "";
        }
        let idx = idx as usize;
        let lo = s.char_indices().nth(idx).map_or(s.len(), |(b, _)| b);
        let hi = s.char_indices().nth(idx + 1).map_or(s.len(), |(b, _)| b);
        return &s[lo..hi];
    }
    let parts: Vec<&str> = s.split(delim).collect();
    let len = parts.len() as i64;
    let idx = if n < 0 { len + n } else { n - 1 };
    if idx < 0 || idx >= len {
        ""
    } else {
        parts[idx as usize]
    }
}

/// SQL `OVERLAY` — replace `length` chars of `s` starting at 1-based `pos` with
/// `rep`. When `length` is absent it defaults to the replacement's char length.
/// `pos` and `length` are measured in Unicode scalar values.
fn overlay(s: &str, rep: &str, pos: i64, length: Option<i64>) -> String {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len() as i64;
    // Saturating arithmetic throughout: `pos`/`length` are user-supplied i64s, so
    // `pos - 1` and `start + len` overflowed on the extremes (`i64::MIN`, `i64::MAX`),
    // panicking. Clamped to `[0, n]` the result is identical for in-range inputs.
    let start = pos.saturating_sub(1).clamp(0, n) as usize;
    let len = length.unwrap_or(rep.chars().count() as i64).max(0);
    let end = (start as i64).saturating_add(len).clamp(0, n) as usize;
    let mut out: String = chars[..start].iter().collect();
    out.push_str(rep);
    out.extend(chars[end..].iter());
    out
}

/// Levenshtein edit distance between `a` and `b` (insert/delete/substitute = 1),
/// over UTF-8 **bytes**. DuckDB (and PostgreSQL's `fuzzystrmatch`) define `levenshtein`
/// on octets, so a multi-byte codepoint counts as several edits — e.g.
/// `levenshtein('héllo', 'abc') = 6`, not the 5 a codepoint-wise distance gives. Matching
/// the oracle byte-for-byte is what keeps a differential test against DuckDB green (and it
/// is faster: no per-string `Vec<char>` materialization). Classic two-row DP kernel.
fn levenshtein(a: &str, b: &str) -> usize {
    let b = b.as_bytes();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];
    for (i, ca) in a.bytes().enumerate() {
        curr[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

/// American Soundex — a 4-character phonetic key (first letter + 3 consonant
/// digits), matching DuckDB's `soundex`. Non-alphabetic input yields "0000".
fn soundex(v: &str) -> String {
    fn code(c: char) -> Option<u8> {
        match c.to_ascii_uppercase() {
            'B' | 'F' | 'P' | 'V' => Some(b'1'),
            'C' | 'G' | 'J' | 'K' | 'Q' | 'S' | 'X' | 'Z' => Some(b'2'),
            'D' | 'T' => Some(b'3'),
            'L' => Some(b'4'),
            'M' | 'N' => Some(b'5'),
            'R' => Some(b'6'),
            _ => None,
        }
    }
    let first = match v.chars().find(|c| c.is_ascii_alphabetic()) {
        Some(c) => c.to_ascii_uppercase(),
        None => return "0000".to_string(),
    };
    let mut out = vec![first as u8];
    let mut last = code(first);
    for c in v.chars().skip_while(|c| !c.is_ascii_alphabetic()).skip(1) {
        if !c.is_ascii_alphabetic() {
            continue;
        }
        let d = code(c);
        // 'H'/'W' are transparent (don't reset the previous code); vowels reset it.
        if matches!(c.to_ascii_uppercase(), 'H' | 'W') {
            continue;
        }
        if let Some(dig) = d {
            if Some(dig) != last {
                out.push(dig);
                if out.len() == 4 {
                    break;
                }
            }
        }
        last = d;
    }
    while out.len() < 4 {
        out.push(b'0');
    }
    String::from_utf8(out).unwrap()
}

/// Capitalize the first alphanumeric of each word, lowercasing the rest. A word
/// is a maximal run of alphanumerics; any non-alphanumeric resets the boundary
/// (DuckDB `initcap`).
fn initcap(v: &str) -> String {
    let mut out = String::with_capacity(v.len());
    let mut start_of_word = true;
    for c in v.chars() {
        if c.is_alphanumeric() {
            if start_of_word {
                out.extend(c.to_uppercase());
            } else {
                out.extend(c.to_lowercase());
            }
            start_of_word = false;
        } else {
            out.push(c);
            start_of_word = true;
        }
    }
    out
}

/// Uppercase hexadecimal of the UTF-8 bytes (DuckDB `hex`).
fn hex_encode(v: &str) -> String {
    let mut out = String::with_capacity(v.len() * 2);
    for b in v.as_bytes() {
        out.push(
            char::from_digit((b >> 4) as u32, 16)
                .unwrap_or('0')
                .to_ascii_uppercase(),
        );
        out.push(
            char::from_digit((b & 0x0f) as u32, 16)
                .unwrap_or('0')
                .to_ascii_uppercase(),
        );
    }
    out
}

/// Lowercase hex of arbitrary bytes — the digest encoding DuckDB's `md5`/`sha1`/
/// `sha256` emit (distinct from `hex_encode`, which uppercases UTF-8 text bytes).
/// Shared with `eval::security`, whose HMAC digest uses the same encoding.
pub(crate) fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(char::from_digit((b >> 4) as u32, 16).unwrap_or('0'));
        out.push(char::from_digit((b & 0x0f) as u32, 16).unwrap_or('0'));
    }
    out
}

/// Parse a string of hex-digit pairs into bytes (DuckDB `unhex`). Returns `None`
/// for an odd number of digits or any non-hex character.
fn hex_decode(v: &str) -> Option<Vec<u8>> {
    let bytes = v.as_bytes();
    if bytes.len() % 2 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let hi = (pair[0] as char).to_digit(16)?;
        let lo = (pair[1] as char).to_digit(16)?;
        out.push(((hi << 4) | lo) as u8);
    }
    Some(out)
}

/// 1-based character position of the first occurrence of `pat` in `v`, or 0 if it
/// does not occur (SQL `POSITION` / DuckDB `strpos`).
fn char_position(v: &str, pat: &str) -> i64 {
    match v.find(pat) {
        Some(byte_idx) => v[..byte_idx].chars().count() as i64 + 1,
        None => 0,
    }
}

/// The largest byte length a single Arrow `Utf8` value can hold: its offsets are
/// `i32`, so a value (or the array's running total) beyond `i32::MAX` overflows the
/// offset buffer — an allocator abort or a builder panic. String functions whose
/// output length is user-controlled (`repeat`, `lpad`, `rpad`) guard against it and
/// return a clean error instead.
const MAX_STR_BYTES: usize = i32::MAX as usize;

/// Pad `v` to `width` characters with `fill` cycled (left or right). If `v` is
/// already at least `width` chars it is truncated to the first `width` (DuckDB
/// `lpad`/`rpad` semantics). An empty `fill` cannot pad, so `v` is returned as-is.
/// Returns `None` if the padded result would exceed [`MAX_STR_BYTES`].
fn pad_checked(v: &str, width: usize, fill: &str, left: bool) -> Option<String> {
    let chars: Vec<char> = v.chars().collect();
    if chars.len() >= width {
        return Some(chars[..width].iter().collect());
    }
    let fill_chars: Vec<char> = fill.chars().collect();
    if fill_chars.is_empty() {
        return Some(v.to_string());
    }
    let pad_len = width - chars.len();
    let max_fill_bytes = fill_chars.iter().map(|c| c.len_utf8()).max().unwrap_or(1);
    // Upper bound on the output before allocating the padding — reject overflow early.
    if pad_len.checked_mul(max_fill_bytes)?.checked_add(v.len())? > MAX_STR_BYTES {
        return None;
    }
    let padding: String = (0..pad_len)
        .map(|i| fill_chars[i % fill_chars.len()])
        .collect();
    Some(if left {
        format!("{padding}{v}")
    } else {
        format!("{v}{padding}")
    })
}

fn map_str(s: &StringArray, f: impl Fn(&str) -> String) -> StringArray {
    s.iter().map(|o| o.map(&f)).collect()
}

/// `map_str` whose closure may fail: `None` becomes a clean [`ExprError::InvalidArgument`]
/// (a result too large for Arrow's 32-bit offsets), never an allocator abort or panic.
fn map_str_checked(
    s: &StringArray,
    func: StrFunc,
    f: impl Fn(&str) -> Option<String>,
) -> Result<StringArray, ExprError> {
    use arrow::array::{Array, StringBuilder};
    let mut b = StringBuilder::with_capacity(s.len(), s.value_data().len());
    for o in s.iter() {
        match o {
            Some(v) => match f(v) {
                Some(r) => b.append_value(r),
                None => {
                    return Err(ExprError::InvalidArgument {
                        func: format!("{func:?}"),
                        reason: format!(
                            "result exceeds the maximum string length of {MAX_STR_BYTES} bytes"
                        ),
                    });
                }
            },
            None => b.append_null(),
        }
    }
    Ok(b.finish())
}

/// `map_str` for functions whose result is a *slice* of the input row (e.g.
/// `substr`): no per-row `String` allocation — the builder copies each `&str` into
/// the output's one contiguous buffer.
fn map_str_borrow<'a>(s: &'a StringArray, f: impl Fn(&'a str) -> &'a str) -> StringArray {
    s.iter().map(|o| o.map(&f)).collect()
}

fn map_bool(s: &StringArray, f: impl Fn(&str) -> bool) -> BooleanArray {
    s.iter().map(|o| o.map(&f)).collect()
}

/// Character-oriented substring matching DuckDB `substring`/`substr`.
///
/// Rules (verified against DuckDB): the string is 1-based; a negative `start`
/// counts from the end (`start = n + start + 1`); a positive `length` spans the
/// inclusive window `[start, start + length - 1]`, a negative `length` flips it to
/// `[start + length, start - 1]`, and no `length` runs to the end. The window is
/// then clipped to `[1, n]` (out-of-range positions are dropped, not shifted), so
/// e.g. `substring('abcdef', 0, 3)` = `'ab'` and `substring('abcdef', -2, 4)` =
/// `'ef'`. An empty intersection yields `""`.
/// `substr`/`substring` (DuckDB semantics) returning a borrowed slice of `v` — the
/// allocation-free form. Computes the char-window byte boundaries via `char_indices`,
/// so it allocates nothing per row (no `Vec<char>`, no `String`); correct for
/// multi-byte UTF-8.
fn substr_slice(v: &str, start: i64, length: Option<i64>) -> &str {
    let n = v.chars().count() as i64;
    // Saturating arithmetic: `start`/`length` are user i64s, so `n + start + 1`,
    // `s + len - 1`, etc. overflowed at the i64 extremes (panic in debug, wrap in
    // release — a wrapped window could even yield a wrong slice). Clamped to `[1, n]`
    // the saturated bounds are identical for in-range inputs.
    let s = if start < 0 {
        n.saturating_add(start).saturating_add(1)
    } else {
        start
    }; // 1-based, may be <= 0
    let (lo, hi) = match length {
        None => (s, n), // to the end, inclusive
        Some(len) if len >= 0 => (s, s.saturating_add(len).saturating_sub(1)),
        Some(len) => (s.saturating_add(len), s.saturating_sub(1)), // negative length flips
    };
    let (lo, hi) = (lo.max(1), hi.min(n)); // clip to [1, n] inclusive
    if hi < lo {
        return "";
    }
    // Byte offset of char index `k` (or the string's end when `k == n`).
    let byte_at = |k: i64| v.char_indices().nth(k as usize).map_or(v.len(), |(b, _)| b);
    &v[byte_at(lo - 1)..byte_at(hi)]
}

/// FNV-1a 64-bit hash of `bytes` — a tiny, deterministic, dependency-free hash whose
/// digest is stable across partitions, runs, and machines (unlike `ahash`). Used by
/// `StrFunc::Hash64` for surrogate keys and SCD change detection.
fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// 64-bit xxHash of `bytes` with seed 0 — fast, deterministic, and stable across
/// machines (the standard bucketing/sharding hash). Uses the portable `Hasher` API.
fn xxhash64(bytes: &[u8]) -> u64 {
    use std::hash::Hasher;
    let mut h = twox_hash::XxHash64::with_seed(0);
    h.write(bytes);
    h.finish()
}

/// Translate a SQL `LIKE`/`ILIKE` pattern into an anchored `regex::Regex`.
///
/// `%` → `.*` (any run, incl. empty), `_` → `.` (exactly one char); every other
/// character is literal (regex metacharacters in literal runs are escaped via
/// `regex::escape`). The whole pattern is anchored with `^…$`, and `(?s)` makes
/// `.` match newlines too — matching SQL's "any character" semantics. `ilike`
/// additionally prepends `(?i)` for case-insensitivity.
fn like_regex(pattern: &str, case_insensitive: bool) -> Result<Arc<regex::Regex>, ExprError> {
    let mut re = String::with_capacity(pattern.len() + 8);
    if case_insensitive {
        re.push_str("(?i)");
    }
    re.push_str("(?s)^");
    // Accumulate literal runs and escape them in one shot so regex metacharacters
    // (`.`, `*`, `(`, …) in the pattern match only themselves.
    let mut literal = String::new();
    for c in pattern.chars() {
        match c {
            '%' | '_' => {
                if !literal.is_empty() {
                    re.push_str(&regex::escape(&literal));
                    literal.clear();
                }
                re.push_str(if c == '%' { ".*" } else { "." });
            }
            other => literal.push(other),
        }
    }
    if !literal.is_empty() {
        re.push_str(&regex::escape(&literal));
    }
    re.push('$');
    // Memoized: `eval_str` runs once per morsel, so without the cache a `LIKE` over a large
    // column rebuilt this automaton for every 16,384 rows. Desugaring the pattern is O(len) and
    // stays here; the compile is the expensive half and is shared.
    regex_cache::compile_cached(&re).ok_or(ExprError::InvalidRegex { pattern: re })
}

/// Compile the (required) regex `pattern` of a regexp string function.
///
/// Memoized on the pattern source — see [`regex_cache`]. The pattern is a plan literal, so the
/// cache is keyed by a handful of strings per query, not one per row.
fn compile_regex(pattern: Option<&str>, func: StrFunc) -> Result<Arc<regex::Regex>, ExprError> {
    let pat = require_pattern(pattern, func)?;
    regex_cache::compile_cached(pat).ok_or_else(|| ExprError::InvalidRegex {
        pattern: pat.to_string(),
    })
}

/// A DuckDB/RE2-style `regexp_replace` rewrite template, as opposed to the `regex`
/// crate's own `$1`/`$name` syntax.
///
/// In RE2 (what DuckDB uses) the replacement escapes are `\0`..`\9` — `\0` is the whole
/// match, `\1`..`\9` are capture groups — `\\` is a literal backslash, and **every other
/// character, including `$`, is literal**. Passing the template straight to
/// `regex::Regex::replace` instead interprets `$1` as a group and leaves `\1` untouched,
/// so `regexp_replace('ab', '(a)(b)', '\2\1')` returned the literal `\2\1` rather than
/// `ba`. This wraps the template so the substitution matches DuckDB.
struct Re2Rewrite<'a>(&'a str);

impl regex::Replacer for Re2Rewrite<'_> {
    fn replace_append(&mut self, caps: &regex::Captures, dst: &mut String) {
        let mut chars = self.0.chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some(d) if d.is_ascii_digit() => {
                        if let Some(m) = caps.get(d as usize - '0' as usize) {
                            dst.push_str(m.as_str());
                        }
                    }
                    Some('\\') => dst.push('\\'),
                    // Unreachable for a template that passed `re2_rewrite_valid`; kept
                    // total so the function never panics on a hand-written IR document.
                    Some(other) => dst.push(other),
                    None => {}
                }
            } else {
                dst.push(c);
            }
        }
    }
}

/// Whether a rewrite `template` is valid for a regex with `max_group` capture groups.
///
/// RE2 rejects a template whose only escapes are not `\0`..`\9` or `\\`, or that
/// references a group past the pattern's count; DuckDB then returns the input row
/// **unchanged**. Mirrors that so an invalid template is a no-op, not a wrong answer.
fn re2_rewrite_valid(template: &str, max_group: usize) -> bool {
    let mut chars = template.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some(d) if d.is_ascii_digit() => {
                    if (d as usize - '0' as usize) > max_group {
                        return false;
                    }
                }
                Some('\\') => {}
                _ => return false, // `\<non-digit>` or a trailing backslash
            }
        }
    }
    true
}

/// Apply a `regexp_replace`/`regexp_replace_all` with DuckDB rewrite semantics.
/// `global` picks first-match vs all-matches. An invalid rewrite template leaves each
/// row unchanged (DuckDB behaviour).
fn regexp_replace_with(s: &StringArray, re: &regex::Regex, rep: &str, global: bool) -> StringArray {
    if !re2_rewrite_valid(rep, re.captures_len().saturating_sub(1)) {
        return map_str_borrow(s, |v| v);
    }
    map_str(s, |v| {
        if global {
            re.replace_all(v, Re2Rewrite(rep)).into_owned()
        } else {
            re.replace(v, Re2Rewrite(rep)).into_owned()
        }
    })
}

fn require_pattern(pattern: Option<&str>, func: StrFunc) -> Result<&str, ExprError> {
    pattern.ok_or_else(|| ExprError::MissingArgument {
        func: format!("{func:?}"),
        arg: "pattern",
    })
}

#[cfg(test)]
mod tests {
    use super::{eval_str, fnv1a64, hex_lower, xxhash64};

    /// `chunk` collects the per-row `List<Utf8>` back into `Vec<Vec<String>>`; a null
    /// row becomes `None`.
    #[cfg(test)]
    fn chunks_of(input: Vec<Option<&str>>, size: i64, overlap: i64) -> Vec<Option<Vec<String>>> {
        use crate::StrFunc;
        use arrow::array::{Array, ArrayRef, ListArray, StringArray};
        use std::sync::Arc;
        let arr: ArrayRef = Arc::new(StringArray::from(input));
        let out = eval_str(StrFunc::Chunk, &arr, None, None, Some(overlap), Some(size)).unwrap();
        let list = out.as_any().downcast_ref::<ListArray>().unwrap();
        (0..list.len())
            .map(|i| {
                if list.is_null(i) {
                    return None;
                }
                let vals = list.value(i);
                let vals = vals.as_any().downcast_ref::<StringArray>().unwrap();
                Some((0..vals.len()).map(|j| vals.value(j).to_string()).collect())
            })
            .collect()
    }

    #[test]
    fn chunk_splits_without_overlap() {
        let got = chunks_of(vec![Some("abcdefg")], 3, 0);
        assert_eq!(got[0].as_deref().unwrap(), ["abc", "def", "g"]);
    }

    #[test]
    fn chunk_overlaps_by_the_requested_characters() {
        // size 4, overlap 2 → stride 2: starts at 0, 2; the chunk from 2 reaches the end.
        let got = chunks_of(vec![Some("abcdef")], 4, 2);
        assert_eq!(got[0].as_deref().unwrap(), ["abcd", "cdef"]);
    }

    /// A trailing start whose chunk would be wholly inside the previous one is not
    /// emitted: "abcdefg"/4/2 stops at "efg" rather than adding a redundant "g".
    #[test]
    fn chunk_does_not_emit_a_redundant_tail() {
        let got = chunks_of(vec![Some("abcdefg")], 4, 2);
        assert_eq!(got[0].as_deref().unwrap(), ["abcd", "cdef", "efg"]);
    }

    /// Every character appears in some chunk, and consecutive chunks share exactly
    /// `overlap` characters — the property a retrieval pipeline depends on.
    #[test]
    fn chunk_overlap_preserves_coverage() {
        for size in 2..8usize {
            for overlap in 0..size {
                let text = "abcdefghijklmno";
                let got = chunks_of(vec![Some(text)], size as i64, overlap as i64);
                let got = got[0].as_deref().unwrap();
                // Rebuild the text: the first chunk in full, then each later chunk
                // minus the `overlap` characters it repeats from its predecessor.
                let mut covered = got[0].clone();
                for c in &got[1..] {
                    covered.extend(c.chars().skip(overlap));
                }
                assert_eq!(covered, text, "size={size} overlap={overlap}");
            }
        }
    }

    #[test]
    fn chunk_covers_every_character_when_not_overlapping() {
        let text = "the quick brown fox";
        let got = chunks_of(vec![Some(text)], 5, 0);
        assert_eq!(got[0].as_ref().unwrap().concat(), text);
    }

    #[test]
    fn chunk_handles_empty_and_null_and_short_inputs() {
        let got = chunks_of(vec![Some(""), None, Some("ab")], 5, 0);
        assert_eq!(got[0].as_deref().unwrap(), [] as [String; 0]);
        assert!(got[1].is_none());
        assert_eq!(got[2].as_deref().unwrap(), ["ab"]);
    }

    /// Chunking is by Unicode character, so a multi-byte codepoint is never split
    /// (a byte-wise slice would panic here).
    #[test]
    fn chunk_splits_on_character_boundaries() {
        let got = chunks_of(vec![Some("héllo→wörld")], 3, 0);
        let got = got[0].as_deref().unwrap();
        assert_eq!(got, ["hél", "lo→", "wör", "ld"]);
        assert_eq!(got.concat(), "héllo→wörld");
    }

    #[test]
    fn chunk_rejects_a_degenerate_frame() {
        use crate::StrFunc;
        use arrow::array::{ArrayRef, StringArray};
        use std::sync::Arc;
        let arr: ArrayRef = Arc::new(StringArray::from(vec![Some("abc")]));
        // overlap == size would never advance; size 0 has no defined chunk.
        assert!(eval_str(StrFunc::Chunk, &arr, None, None, Some(3), Some(3)).is_err());
        assert!(eval_str(StrFunc::Chunk, &arr, None, None, Some(0), Some(0)).is_err());
        assert!(eval_str(StrFunc::Chunk, &arr, None, None, Some(-1), Some(3)).is_err());
        // missing `length` (the chunk size)
        assert!(eval_str(StrFunc::Chunk, &arr, None, None, Some(0), None).is_err());
    }

    /// A string function over a `Binary`-typed column (ClickBench's `hits` shape) coerces
    /// the column to `Utf8` and applies, instead of erroring on the non-string type.
    #[test]
    fn string_function_over_binary_column() {
        use crate::StrFunc;
        use arrow::array::{Array, ArrayRef, BinaryArray, Int64Array};
        use std::sync::Arc;
        let bin: ArrayRef = Arc::new(BinaryArray::from_opt_vec(vec![
            Some(b"hello".as_ref()),
            Some(b""),
            None,
        ]));
        let out = eval_str(StrFunc::Len, &bin, None, None, None, None).unwrap();
        let got = out.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(got.value(0), 5);
        assert_eq!(got.value(1), 0);
        assert!(got.is_null(2));
    }

    #[test]
    fn crypto_hash_known_vectors() {
        use md5::{Digest, Md5};
        use sha2::Sha256;
        // Published digests for "abc".
        assert_eq!(
            hex_lower(Md5::digest(b"abc").as_slice()),
            "900150983cd24fb0d6963f7d28e17f72"
        );
        assert_eq!(
            hex_lower(Sha256::digest(b"abc").as_slice()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // crc32(IEEE) "abc" = 0x352441C2; empty = 0.
        assert_eq!(crc32fast::hash(b"abc"), 0x3524_41c2);
        assert_eq!(crc32fast::hash(b""), 0);
    }

    #[test]
    fn xxhash64_known_vector_and_determinism() {
        // xxHash64(seed 0) of the empty input is the canonical 0xEF46DB3751D8E999.
        assert_eq!(xxhash64(b""), 0xEF46_DB37_51D8_E999);
        assert_eq!(xxhash64(b"customer-42"), xxhash64(b"customer-42"));
        assert_ne!(xxhash64(b"a"), xxhash64(b"b"));
    }

    #[test]
    fn fnv1a64_known_vectors() {
        // Standard FNV-1a 64-bit test vectors.
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn fnv1a64_is_deterministic() {
        // Same input → same digest (partition/run independence).
        assert_eq!(
            fnv1a64(b"customer-42|2024-06-23"),
            fnv1a64(b"customer-42|2024-06-23")
        );
        assert_ne!(fnv1a64(b"a"), fnv1a64(b"b"));
    }

    /// Argument-less `trim`/`ltrim`/`rtrim` strip the Unicode `Zs` space-separator
    /// category (matching DuckDB), NOT the C0 control whitespace (tab/newline/CR/…)
    /// that Rust's `str::trim` also removes.
    #[test]
    fn trim_strips_only_space_separators_like_duckdb() {
        use crate::StrFunc;
        use arrow::array::{Array, ArrayRef, StringArray};
        use std::sync::Arc;
        // "\t\n" is pure control whitespace → DuckDB (and now Batcher) keep it whole.
        // " \u{a0}x\u{3000} " mixes ASCII space, NBSP, ideographic space around 'x'.
        let arr: ArrayRef = Arc::new(StringArray::from(vec![
            Some("\t\n"),
            Some(" \u{a0}x\u{3000} "),
            Some("\tx\t"),
        ]));
        let val = |f, row: usize| {
            let out = eval_str(f, &arr, None, None, None, None).unwrap();
            out.as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(row)
                .to_string()
        };
        assert_eq!(val(StrFunc::Trim, 0), "\t\n"); // controls kept (was "")
        assert_eq!(val(StrFunc::Trim, 1), "x"); // Zs stripped both sides
        assert_eq!(val(StrFunc::Trim, 2), "\tx\t"); // tabs kept (was "x")
        assert_eq!(val(StrFunc::LTrim, 1), "x\u{3000} ");
        assert_eq!(val(StrFunc::RTrim, 1), " \u{a0}x");
        assert_eq!(val(StrFunc::LTrim, 0), "\t\n");
        assert_eq!(val(StrFunc::RTrim, 2), "\tx\t");
    }

    use super::{overlay, split_part, substring_index};

    /// `right(s, n)` with a negative `n` drops the first `|n|` characters (DuckDB),
    /// where the old `.max(0)` collapsed it to the empty string.
    #[test]
    fn right_negative_drops_the_leading_chars() {
        use crate::StrFunc;
        use arrow::array::{ArrayRef, StringArray};
        use std::sync::Arc;
        let arr: ArrayRef = Arc::new(StringArray::from(vec![Some("abcdef"), Some("中文字")]));
        let out = eval_str(StrFunc::Right, &arr, None, None, Some(-2), None).unwrap();
        let got = out.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(got.value(0), "cdef"); // was "" before the fix
        assert_eq!(got.value(1), "字"); // char-based, not byte-based
                                        // |n| beyond the length yields the empty string, not a panic.
        let out = eval_str(StrFunc::Right, &arr, None, None, Some(i64::MIN), None).unwrap();
        let got = out.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(got.value(0), "");
    }

    /// `split_part` counts from the right for a negative index and splits into
    /// characters for an empty delimiter (both DuckDB semantics).
    #[test]
    fn split_part_negative_and_empty_delimiter() {
        assert_eq!(split_part("a-b-c", "-", -1), "c"); // was ""
        assert_eq!(split_part("a-b-c", "-", -3), "a");
        assert_eq!(split_part("a-b-c", "-", -4), "");
        assert_eq!(split_part("a-b-c", "-", 0), "");
        assert_eq!(split_part("abc", "", 2), "b"); // was "a" (Rust split("") artifact)
        assert_eq!(split_part("中a", "", 1), "中");
        assert_eq!(split_part("中a", "", 2), "a");
        // i64::MIN must not overflow.
        assert_eq!(split_part("a-b-c", "-", i64::MIN), "");
    }

    /// An empty delimiter splits into individual characters (DuckDB), not the
    /// `["", …, ""]` that Rust's `str::split("")` produces.
    #[test]
    fn split_empty_delimiter_yields_characters() {
        use crate::StrFunc;
        use arrow::array::{Array, ArrayRef, ListArray, StringArray};
        use std::sync::Arc;
        let arr: ArrayRef = Arc::new(StringArray::from(vec![Some("abc"), Some("中a"), Some("")]));
        let out = eval_str(StrFunc::Split, &arr, Some(""), None, None, None).unwrap();
        let list = out.as_any().downcast_ref::<ListArray>().unwrap();
        let row = |i: usize| {
            let v = list.value(i);
            let v = v.as_any().downcast_ref::<StringArray>().unwrap();
            (0..v.len())
                .map(|j| v.value(j).to_string())
                .collect::<Vec<_>>()
        };
        assert_eq!(row(0), ["a", "b", "c"]);
        assert_eq!(row(1), ["中", "a"]);
        assert_eq!(row(2), [""]);
    }

    /// `replace(s, '', r)` returns `s` unchanged (DuckDB); Rust's `str::replace("")`
    /// would splice `r` between every character.
    #[test]
    fn replace_empty_pattern_is_a_noop() {
        use crate::StrFunc;
        use arrow::array::{ArrayRef, StringArray};
        use std::sync::Arc;
        let arr: ArrayRef = Arc::new(StringArray::from(vec![Some("abc"), Some("")]));
        let out = eval_str(StrFunc::Replace, &arr, Some(""), Some("X"), None, None).unwrap();
        let got = out.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(got.value(0), "abc"); // was "XaXbXcX"
        assert_eq!(got.value(1), "");
    }

    /// `substring_index` with `i64::MIN` must not overflow (`-count` panicked before).
    #[test]
    fn substring_index_extremes_do_not_overflow() {
        assert_eq!(substring_index("a-b-c", "-", 2), "a-b");
        assert_eq!(substring_index("a-b-c", "-", -2), "b-c");
        assert_eq!(substring_index("a-b-c", "-", i64::MIN), "a-b-c");
        assert_eq!(substring_index("a-b-c", "-", i64::MAX), "a-b-c");
    }

    /// `overlay` at the i64 extremes clamps instead of overflowing (`pos - 1` and
    /// `start + len` panicked before).
    #[test]
    fn overlay_extremes_do_not_overflow() {
        assert_eq!(overlay("hello", "XY", 2, None), "hXYlo");
        assert_eq!(overlay("hello", "XY", i64::MAX, None), "helloXY");
        // pos clamps to the front; the default length (2) still overwrites two chars.
        assert_eq!(overlay("hello", "XY", i64::MIN, None), "XYllo");
        assert_eq!(overlay("hello", "XY", 2, Some(i64::MAX)), "hXY");
        assert_eq!(overlay("hello", "XY", 2, Some(i64::MIN)), "hXYello");
    }

    /// `repeat`/`lpad`/`rpad` with an absurd count return a clean error rather than
    /// aborting the process on a failed multi-gigabyte allocation.
    #[test]
    fn oversized_output_errors_cleanly() {
        use crate::StrFunc;
        use arrow::array::{ArrayRef, StringArray};
        use std::sync::Arc;
        let arr: ArrayRef = Arc::new(StringArray::from(vec![Some("abcdef")]));
        assert!(eval_str(StrFunc::Repeat, &arr, None, None, Some(1_000_000_000), None).is_err());
        assert!(eval_str(StrFunc::Lpad, &arr, Some("*"), None, Some(i64::MAX), None).is_err());
        assert!(eval_str(StrFunc::Rpad, &arr, Some("*"), None, Some(i64::MAX), None).is_err());
        // A legitimate, bounded size still succeeds.
        assert!(eval_str(StrFunc::Repeat, &arr, None, None, Some(3), None).is_ok());
    }

    /// `substr` with i64-extreme start/length clips to the string instead of
    /// overflowing (panic in debug, wrong slice in release).
    #[test]
    fn substr_extremes_do_not_overflow() {
        use crate::StrFunc;
        use arrow::array::{ArrayRef, StringArray};
        use std::sync::Arc;
        let arr: ArrayRef = Arc::new(StringArray::from(vec![Some("abcdef")]));
        let one = |start: i64, len: Option<i64>| {
            let out = eval_str(StrFunc::Substr, &arr, None, None, Some(start), len).unwrap();
            out.as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(0)
                .to_string()
        };
        assert_eq!(one(1, Some(i64::MAX)), "abcdef");
        assert_eq!(one(2, Some(i64::MAX)), "bcdef");
        assert_eq!(one(i64::MIN, Some(3)), "");
        assert_eq!(one(i64::MAX, Some(3)), "");
        assert_eq!(one(3, Some(i64::MIN)), "ab");
    }

    /// `levenshtein` counts UTF-8 **bytes**, matching DuckDB (a codepoint-wise distance
    /// would return 5 for `'héllo'` vs `'abc'`; DuckDB and PostgreSQL return 6).
    #[test]
    fn levenshtein_is_byte_based_like_duckdb() {
        use super::levenshtein;
        assert_eq!(levenshtein("héllo", "abc"), 6); // 'é' is two bytes → 6, not 5
        assert_eq!(levenshtein("é", "e"), 2); // 0xC3 0xA9 vs 0x65
        assert_eq!(levenshtein("café", "cafe"), 2);
        assert_eq!(levenshtein("日本", "日"), 3); // 6 bytes vs 3 bytes
                                                  // ASCII and empties are unaffected.
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("abc", "abc"), 0);
    }

    /// `regexp_replace`/`regexp_replace_all` use DuckDB's RE2 rewrite template: `\1`..`\9`
    /// are capture-group backreferences, `\0` the whole match, `\\` a literal backslash,
    /// and `$` is literal. The old code passed the template to the `regex` crate verbatim,
    /// so `\2\1` came out literal and `$1` was (wrongly) interpreted as a group.
    #[test]
    fn regexp_replace_uses_re2_backreferences() {
        use crate::StrFunc;
        use arrow::array::{Array, ArrayRef, StringArray};
        use std::sync::Arc;
        let arr: ArrayRef = Arc::new(StringArray::from(vec![
            Some("ab"),
            Some("xaby"),
            Some("abab"),
        ]));
        let vals = |f, pat: &str, rep: &str| {
            let out = eval_str(f, &arr, Some(pat), Some(rep), None, None).unwrap();
            let a = out.as_any().downcast_ref::<StringArray>().unwrap();
            (0..a.len())
                .map(|i| a.value(i).to_string())
                .collect::<Vec<_>>()
        };
        // Swap the two captured letters, first match only vs global.
        assert_eq!(
            vals(StrFunc::RegexpReplace, "(a)(b)", r"\2\1"),
            ["ba", "xbay", "baab"]
        );
        assert_eq!(
            vals(StrFunc::RegexpReplaceAll, "(a)(b)", r"\2\1"),
            ["ba", "xbay", "baba"]
        );
        // `\0` is the whole match; `$` is a literal character.
        let one: ArrayRef = Arc::new(StringArray::from(vec![Some("a")]));
        let out = eval_str(
            StrFunc::RegexpReplace,
            &one,
            Some("(a)"),
            Some(r"[\0]"),
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            out.as_any().downcast_ref::<StringArray>().unwrap().value(0),
            "[a]"
        );
        let out = eval_str(
            StrFunc::RegexpReplace,
            &one,
            Some("a"),
            Some("$1"),
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            out.as_any().downcast_ref::<StringArray>().unwrap().value(0),
            "$1"
        );
        // An invalid rewrite (`\q`, or a group past the pattern's count) leaves the row
        // unchanged, like DuckDB.
        let out = eval_str(
            StrFunc::RegexpReplace,
            &one,
            Some("a"),
            Some(r"\q"),
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            out.as_any().downcast_ref::<StringArray>().unwrap().value(0),
            "a"
        );
        let out = eval_str(
            StrFunc::RegexpReplace,
            &one,
            Some("(a)"),
            Some(r"\9"),
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            out.as_any().downcast_ref::<StringArray>().unwrap().value(0),
            "a"
        );
    }
}
