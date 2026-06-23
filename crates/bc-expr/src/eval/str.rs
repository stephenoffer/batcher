//! String-function evaluation for `Expr::Str` (split out of `lib.rs`).

use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanArray, Int64Array, StringArray};

use crate::{ExprError, StrFunc};

/// Evaluate a string function over a Utf8 array (preserving nulls).
pub(crate) fn eval_str(
    func: StrFunc,
    arr: &ArrayRef,
    pattern: Option<&str>,
    replacement: Option<&str>,
    start: Option<i64>,
    length: Option<i64>,
) -> Result<ArrayRef, ExprError> {
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
            let start = start.unwrap_or(1);
            Arc::new(map_str(s, |v| substr(v, start, length)))
        }
        StrFunc::Replace => {
            let pat = require_pattern(pattern, func)?;
            let rep = replacement.ok_or_else(|| ExprError::MissingArgument {
                func: format!("{func:?}"),
                arg: "replacement",
            })?;
            Arc::new(map_str(s, |v| v.replace(pat, rep)))
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
            None => Arc::new(map_str(s, |v| v.trim().to_string())),
        },
        StrFunc::LTrim => match pattern {
            Some(chars) => {
                let set: Vec<char> = chars.chars().collect();
                Arc::new(map_str(s, |v| {
                    v.trim_start_matches(|c| set.contains(&c)).to_string()
                }))
            }
            None => Arc::new(map_str(s, |v| v.trim_start().to_string())),
        },
        StrFunc::RTrim => match pattern {
            Some(chars) => {
                let set: Vec<char> = chars.chars().collect();
                Arc::new(map_str(s, |v| {
                    v.trim_end_matches(|c| set.contains(&c)).to_string()
                }))
            }
            None => Arc::new(map_str(s, |v| v.trim_end().to_string())),
        },
        StrFunc::Reverse => Arc::new(map_str(s, |v| v.chars().rev().collect())),
        StrFunc::Repeat => {
            let n = start.unwrap_or(0).max(0) as usize;
            Arc::new(map_str(s, |v| v.repeat(n)))
        }
        StrFunc::Lpad => {
            let width = start.unwrap_or(0).max(0) as usize;
            let fill = pattern.unwrap_or(" ");
            Arc::new(map_str(s, |v| pad(v, width, fill, true)))
        }
        StrFunc::Rpad => {
            let width = start.unwrap_or(0).max(0) as usize;
            let fill = pattern.unwrap_or(" ");
            Arc::new(map_str(s, |v| pad(v, width, fill, false)))
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
            let n = start.unwrap_or(0).max(0) as usize;
            Arc::new(map_str(s, |v| {
                let chars: Vec<char> = v.chars().collect();
                let begin = chars.len().saturating_sub(n);
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
            Arc::new(map_str(s, |v| re.replace(v, rep).into_owned()))
        }
        StrFunc::RegexpReplaceAll => {
            let re = compile_regex(pattern, func)?;
            let rep = replacement.ok_or_else(|| ExprError::MissingArgument {
                func: format!("{func:?}"),
                arg: "replacement",
            })?;
            Arc::new(map_str(s, |v| re.replace_all(v, rep).into_owned()))
        }
        StrFunc::SplitPart => {
            let delim = require_pattern(pattern, func)?;
            let n = start.unwrap_or(1);
            Arc::new(map_str(s, |v| {
                if n < 1 {
                    return String::new();
                }
                v.split(delim)
                    .nth((n - 1) as usize)
                    .unwrap_or("")
                    .to_string()
            }))
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
            let path = require_pattern(pattern, func)?;
            let keys = json_path_keys(path);
            // Nullable result: null where input is not valid JSON or path is absent.
            Arc::new(
                s.iter()
                    .map(|o| o.and_then(|v| json_extract_string(v, &keys)))
                    .collect::<StringArray>(),
            )
        }
        StrFunc::JsonExtractInt => {
            let path = require_pattern(pattern, func)?;
            let keys = json_path_keys(path);
            Arc::new(
                s.iter()
                    .map(|o| {
                        o.and_then(|v| json_navigate(v, &keys))
                            .and_then(|j| j.as_i64())
                    })
                    .collect::<Int64Array>(),
            )
        }
        StrFunc::JsonExtractFloat => {
            use arrow::array::Float64Array;
            let path = require_pattern(pattern, func)?;
            let keys = json_path_keys(path);
            Arc::new(
                s.iter()
                    .map(|o| {
                        o.and_then(|v| json_navigate(v, &keys))
                            .and_then(|j| j.as_f64())
                    })
                    .collect::<Float64Array>(),
            )
        }
        StrFunc::JsonExtractBool => {
            let path = require_pattern(pattern, func)?;
            let keys = json_path_keys(path);
            Arc::new(
                s.iter()
                    .map(|o| {
                        o.and_then(|v| json_navigate(v, &keys))
                            .and_then(|j| j.as_bool())
                    })
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
        StrFunc::Hex => Arc::new(map_str(s, hex_encode)),
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
            use arrow::array::{ListBuilder, StringBuilder};
            let delim = require_pattern(pattern, func)?;
            let mut builder = ListBuilder::new(StringBuilder::new());
            for o in s.iter() {
                match o {
                    Some(v) => {
                        for part in v.split(delim) {
                            builder.values().append_value(part);
                        }
                        builder.append(true);
                    }
                    None => builder.append(false),
                }
            }
            Arc::new(builder.finish())
        }
    };
    Ok(out)
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

/// Pad `v` to `width` characters with `fill` cycled (left or right). If `v` is
/// already at least `width` chars it is truncated to the first `width` (DuckDB
/// `lpad`/`rpad` semantics). An empty `fill` cannot pad, so `v` is returned as-is.
fn pad(v: &str, width: usize, fill: &str, left: bool) -> String {
    let chars: Vec<char> = v.chars().collect();
    if chars.len() >= width {
        return chars[..width].iter().collect();
    }
    let fill_chars: Vec<char> = fill.chars().collect();
    if fill_chars.is_empty() {
        return v.to_string();
    }
    let pad_len = width - chars.len();
    let padding: String = (0..pad_len)
        .map(|i| fill_chars[i % fill_chars.len()])
        .collect();
    if left {
        format!("{padding}{v}")
    } else {
        format!("{v}{padding}")
    }
}

fn map_str(s: &StringArray, f: impl Fn(&str) -> String) -> StringArray {
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
fn substr(v: &str, start: i64, length: Option<i64>) -> String {
    let chars: Vec<char> = v.chars().collect();
    let n = chars.len() as i64;
    let s = if start < 0 { n + start + 1 } else { start }; // 1-based, may be <= 0
    let (lo, hi) = match length {
        None => (s, n), // to the end, inclusive
        Some(len) if len >= 0 => (s, s + len - 1),
        Some(len) => (s + len, s - 1), // negative length flips the window
    };
    let (lo, hi) = (lo.max(1), hi.min(n)); // clip to [1, n] inclusive
    if hi < lo {
        return String::new();
    }
    chars[(lo - 1) as usize..hi as usize].iter().collect()
}

/// Split a JSON path like `$.a.b` or `a.b` into its keys.
fn json_path_keys(path: &str) -> Vec<String> {
    path.trim_start_matches('$')
        .split('.')
        .filter(|k| !k.is_empty())
        .map(|k| k.to_string())
        .collect()
}

/// Navigate `keys` into the JSON document `text`, returning the leaf as a string
/// (string leaves verbatim; numbers/bools as their text). `None` if invalid JSON
/// or the path is missing.
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

/// Navigate `text` (parsed as JSON) down the `$.a.b` path `keys`, returning the
/// `Value` at that location (or `None` if the text isn't valid JSON or the path is
/// absent). Shared by the typed `json_extract_*` extractors.
fn json_navigate(text: &str, keys: &[String]) -> Option<serde_json::Value> {
    let mut cur: serde_json::Value = serde_json::from_str(text).ok()?;
    for k in keys {
        cur = cur.get(k)?.clone();
    }
    Some(cur)
}

fn json_extract_string(text: &str, keys: &[String]) -> Option<String> {
    match json_navigate(text, keys)? {
        serde_json::Value::String(s) => Some(s),
        serde_json::Value::Null => None,
        other => Some(other.to_string()),
    }
}

/// Translate a SQL `LIKE`/`ILIKE` pattern into an anchored `regex::Regex`.
///
/// `%` → `.*` (any run, incl. empty), `_` → `.` (exactly one char); every other
/// character is literal (regex metacharacters in literal runs are escaped via
/// `regex::escape`). The whole pattern is anchored with `^…$`, and `(?s)` makes
/// `.` match newlines too — matching SQL's "any character" semantics. `ilike`
/// additionally prepends `(?i)` for case-insensitivity.
fn like_regex(pattern: &str, case_insensitive: bool) -> Result<regex::Regex, ExprError> {
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
    regex::Regex::new(&re).map_err(|_| ExprError::InvalidRegex { pattern: re })
}

/// Compile the (required) regex `pattern` of a regexp string function.
fn compile_regex(pattern: Option<&str>, func: StrFunc) -> Result<regex::Regex, ExprError> {
    let pat = require_pattern(pattern, func)?;
    regex::Regex::new(pat).map_err(|_| ExprError::InvalidRegex {
        pattern: pat.to_string(),
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
    use super::fnv1a64;

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
}
