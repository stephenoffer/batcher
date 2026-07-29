//! String-function evaluation for `Expr::Str` (split out of `lib.rs`).

use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanArray, Int64Array, StringArray};
use arrow::compute::cast;
use arrow::datatypes::DataType;

use crate::{ExprError, StrFunc};

mod case;
mod chunk;
mod compress;
mod html;
mod jaro;
mod json;
mod like;
mod minhash;
mod numfmt;
mod regex_cache;
mod uri_path;

/// Evaluate a string function over a Utf8 array (preserving nulls).
/// Apply a string function to a **dictionary** column's distinct values and gather the
/// result through its keys, instead of decoding the column and applying it per row.
///
/// The identity is the one `try_dict_compare` already relies on, generalized from
/// comparison to any elementwise function: **an elementwise function commutes with a
/// gather**, so `f(take(values, keys)) == take(f(values), keys)`. Every `StrFunc` is
/// elementwise — each output row depends only on the same input row and the *constant*
/// `pattern`/`replacement`/`start`/`length` arguments — so the rewrite is exact rather
/// than approximate, which is what lets it live inside the correctness oracle.
///
/// The win is the ratio of rows to distinct values, and low-cardinality string columns
/// are the normal shape of analytic data: a country, a status, a user agent. On 6M rows
/// over 25 distinct values, `upper()` runs 25 times instead of 6,000,000.
///
/// Nulls need no special handling and are worth stating because it looks like they
/// might: a null *key* gathers to null whatever `f` did, and a null *value* makes
/// `f(value)` null which then gathers to null. Both match the decoded path exactly.
///
/// Returns `None` for any non-dictionary input, so the caller falls back to the array
/// path unchanged.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_dict_str(
    func: StrFunc,
    input: &crate::Expr,
    batch: &arrow::array::RecordBatch,
    pattern: Option<&str>,
    replacement: Option<&str>,
    start: Option<i64>,
    length: Option<i64>,
) -> Result<Option<ArrayRef>, ExprError> {
    use arrow::array::AsArray;

    // Only a bare column: any other expression has already been evaluated (and so already
    // decoded) by the time it gets here, leaving no dictionary to exploit.
    let crate::Expr::Col { name } = input else {
        return Ok(None);
    };
    // Read the column straight from the batch rather than through `eval`, which decodes
    // dictionaries at the leaf.
    let Some(arr) = batch.column_by_name(name) else {
        return Ok(None);
    };
    if !matches!(arr.data_type(), DataType::Dictionary(_, _)) {
        return Ok(None);
    }
    let dict = arr.as_any_dictionary();
    // One call per *distinct* value, not per row.
    let over_values = eval_str(func, dict.values(), pattern, replacement, start, length)?;
    Ok(Some(arrow::compute::take(&over_values, dict.keys(), None)?))
}

pub(crate) fn eval_str(
    func: StrFunc,
    arr: &ArrayRef,
    pattern: Option<&str>,
    replacement: Option<&str>,
    start: Option<i64>,
    length: Option<i64>,
) -> Result<ArrayRef, ExprError> {
    // Byte-oriented functions (`octet_length`, `bit_length`, `hex`, `md5`, `sha*`,
    // `base64`, `crc32`, `xxhash64`, `hash64`) are defined on the *raw bytes* of a BLOB
    // and MUST NOT route through the Utf8 cast below — that cast nulls any row whose
    // bytes are not valid UTF-8, so e.g. `hex(BLOB '\xDE\xAD\xBE\xEF')` silently became
    // NULL instead of `'DEADBEEF'`, and `md5`/`sha256`/`base64`/`octet_length` likewise
    // dropped every non-UTF-8 row. DuckDB's `hex(BLOB)`/`md5(BLOB)`/… operate on the
    // bytes regardless of textual validity; do the same here.
    if matches!(arr.data_type(), DataType::Binary | DataType::LargeBinary) {
        if let Some(out) = eval_bytes(func, arr, pattern)? {
            return Ok(out);
        }
    }
    // Int → Utf8 functions (`chr`, `to_base`/`bin`, `format_bytes`, and `hex` of an
    // integer) are dispatched before the Utf8 downcast below, which would reject their
    // argument outright. Every other function declines here and takes the string path.
    if let Some(out) = numfmt::eval_numeric_input(func, arr, start)? {
        return Ok(out);
    }
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
            Arc::new(like::LikeMatcher::contains(pat).eval(s))
        }
        StrFunc::StartsWith => {
            let pat = require_pattern(pattern, func)?;
            Arc::new(like::LikeMatcher::starts_with(pat).eval(s))
        }
        StrFunc::EndsWith => {
            let pat = require_pattern(pattern, func)?;
            Arc::new(like::LikeMatcher::ends_with(pat).eval(s))
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
            // Case-sensitive `LIKE` with no `_` desugars to an ordered substring scan (the fast
            // path); `_` (single-char wildcard) and `ILIKE` (Unicode case-folding) keep the
            // cached anchored regex, which the matcher wraps so both go through one `eval`.
            let ci = matches!(func, StrFunc::Ilike);
            let matcher = if ci || pat.contains('_') {
                like::LikeMatcher::Regex(like_regex(pat, ci)?)
            } else {
                like::LikeMatcher::classify(pat)
            };
            Arc::new(matcher.eval(s))
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
        StrFunc::JsonArrayLength => {
            let path = json::parse_path(require_pattern(pattern, func)?);
            Arc::new(
                s.iter()
                    .map(|o| o.and_then(|v| json::array_length(v, &path)))
                    .collect::<Int64Array>(),
            )
        }
        StrFunc::JsonType => {
            let path = json::parse_path(require_pattern(pattern, func)?);
            Arc::new(
                s.iter()
                    .map(|o| o.and_then(|v| json::value_type(v, &path)))
                    .collect::<StringArray>(),
            )
        }
        StrFunc::JsonValue => {
            let path = json::parse_path(require_pattern(pattern, func)?);
            Arc::new(
                s.iter()
                    .map(|o| o.and_then(|v| json::json_value(v, &path)))
                    .collect::<StringArray>(),
            )
        }
        StrFunc::JsonContains => {
            let needle = require_pattern(pattern, func)?;
            Arc::new(
                s.iter()
                    .map(|o| o.map(|v| json::contains(v, needle)))
                    .collect::<BooleanArray>(),
            )
        }
        StrFunc::JsonPretty => Arc::new(
            s.iter()
                .map(|o| o.and_then(json::pretty))
                .collect::<StringArray>(),
        ),
        StrFunc::JsonStructure => Arc::new(
            s.iter()
                .map(|o| o.and_then(json::structure))
                .collect::<StringArray>(),
        ),
        // Int → Utf8, handled before the Utf8 downcast above. Reaching here means the
        // argument was text, which these have no meaning for.
        StrFunc::Chr | StrFunc::ToBase | StrFunc::FormatBytes | StrFunc::FormatBytesSi => {
            return Err(ExprError::ExpectedString {
                func: format!("{func:?}"),
                got: "Utf8 (this function takes an integer)".into(),
            })
        }
        StrFunc::JsonExists => {
            let path = json::parse_path(require_pattern(pattern, func)?);
            Arc::new(
                s.iter()
                    .map(|o| o.map(|v| json::path_exists(v, &path)))
                    .collect::<BooleanArray>(),
            )
        }
        StrFunc::JsonObjectKeys | StrFunc::JsonArrayValues => {
            use arrow::array::{Array, ListBuilder, StringBuilder};
            let path = json::parse_path(require_pattern(pattern, func)?);
            let keys = matches!(func, StrFunc::JsonObjectKeys);
            // Both shapes emit one `List<Utf8>` per row; the extracted text is a subset
            // of the document, so the input's value bytes bound the output's.
            let mut builder = ListBuilder::with_capacity(
                StringBuilder::with_capacity(s.len(), s.value_data().len()),
                s.len(),
            );
            for o in s.iter() {
                // A null input, an absent path, or a value of the wrong shape all yield a
                // null list rather than an empty one, keeping "no answer" distinct from
                // "an empty object/array", which is a real and different fact.
                let produced = o.is_some_and(|v| {
                    if keys {
                        match json::object_keys(v, &path) {
                            Some(ks) => {
                                for k in ks {
                                    builder.values().append_value(k);
                                }
                                true
                            }
                            None => false,
                        }
                    } else {
                        match json::array_values(v, &path) {
                            Some(vs) => {
                                for e in vs {
                                    builder.values().append_option(e);
                                }
                                true
                            }
                            None => false,
                        }
                    }
                });
                builder.append(produced);
            }
            Arc::new(builder.finish())
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
        StrFunc::Chunk => chunk::eval_chunk(s, start, length, pattern)?,
        StrFunc::TokenNgrams => {
            use arrow::array::{Array, ListBuilder, StringBuilder};
            // `length` carries `n`; clamp to at least 1 so a bad plan never panics.
            let n = length.unwrap_or(1).max(1) as usize;
            // Each n-gram re-emits its tokens plus one joining space, so the values are
            // ~n× the input bytes; size the value buffer generously to skip regrowth.
            let mut builder = ListBuilder::with_capacity(
                StringBuilder::with_capacity(s.len(), s.value_data().len() * n),
                s.len(),
            );
            for o in s.iter() {
                match o {
                    Some(v) => {
                        let tokens: Vec<&str> = v.split_whitespace().collect();
                        if tokens.is_empty() {
                            // Empty or whitespace-only input → an empty list (matches
                            // `chunk`'s empty-string behaviour), not a `[""]`.
                        } else if tokens.len() < n {
                            // Fewer than `n` tokens: emit the single n-gram of all of
                            // them so a short document still contributes one gram.
                            builder.values().append_value(tokens.join(" "));
                        } else {
                            for window in tokens.windows(n) {
                                builder.values().append_value(window.join(" "));
                            }
                        }
                        builder.append(true);
                    }
                    None => builder.append(false),
                }
            }
            Arc::new(builder.finish())
        }
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
            // The capture-group index rides `start`, as it does for the scalar
            // `RegexpExtract`. Without it every call collected the *whole* match, so
            // `regexp_extract_all('100-200', '(\d+)-(\d+)', 1)` answered
            // `['100-200']` where DuckDB says `['100']` — a wrong answer, not a refusal,
            // because the group argument was simply dropped on the way down.
            let group = start.unwrap_or(0).max(0) as usize;
            // DuckDB rejects a group the pattern does not have rather than returning
            // empty lists; `captures_len` counts group 0, so the last valid index is
            // one less.
            if group >= re.captures_len() {
                return Err(ExprError::InvalidArgument {
                    func: format!("{func:?}"),
                    reason: format!(
                        "pattern has {} group(s); cannot access group {group}",
                        re.captures_len() - 1
                    ),
                });
            }
            // One list per row; match volume per row is unknown, so pre-size only the
            // outer offset buffer and let the inner value builder grow as matches land.
            let mut builder = ListBuilder::with_capacity(StringBuilder::new(), s.len());
            for o in s.iter() {
                match o {
                    Some(v) => {
                        if group == 0 {
                            for m in re.find_iter(v) {
                                builder.values().append_value(m.as_str());
                            }
                        } else {
                            for c in re.captures_iter(v) {
                                // A group that did not participate in this match is a
                                // NULL element in DuckDB (the scalar `regexp_extract`
                                // yields `''` instead — the two genuinely differ).
                                match c.get(group) {
                                    Some(m) => builder.values().append_value(m.as_str()),
                                    None => builder.values().append_null(),
                                }
                            }
                        }
                        builder.append(true);
                    }
                    None => builder.append(false),
                }
            }
            Arc::new(builder.finish())
        }
        StrFunc::RegexpSplit => {
            use arrow::array::{Array, ListBuilder, StringBuilder};
            let re = compile_regex(pattern, func)?;
            // The pieces together are at most the input bytes, so both buffers pre-size
            // from the input the way the literal `Split` does.
            let mut builder = ListBuilder::with_capacity(
                StringBuilder::with_capacity(s.len(), s.value_data().len()),
                s.len(),
            );
            for o in s.iter() {
                match o {
                    Some(v) => {
                        for part in re.split(v) {
                            builder.values().append_value(part);
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
        StrFunc::UrlEncode => Arc::new(map_str(s, uri_path::url_encode)),
        StrFunc::UrlDecode => Arc::new(map_str(s, uri_path::url_decode)),
        StrFunc::RegexpEscape => Arc::new(map_str(s, uri_path::regexp_escape)),
        StrFunc::ParseFilename => Arc::new(map_str_borrow(s, uri_path::parse_filename)),
        StrFunc::ParseDirname => Arc::new(map_str_borrow(s, uri_path::parse_dirname)),
        StrFunc::ParseDirpath => Arc::new(map_str_borrow(s, uri_path::parse_dirpath)),
        StrFunc::ParsePath => {
            use arrow::array::{Array, ListBuilder, StringBuilder};
            // Components are borrowed slices of the input, so the value buffer needs at
            // most the input's bytes — the same pre-sizing `Split` uses.
            let mut builder = ListBuilder::with_capacity(
                StringBuilder::with_capacity(s.len(), s.value_data().len()),
                s.len(),
            );
            for o in s.iter() {
                match o {
                    Some(v) => {
                        for part in uri_path::parse_path(v) {
                            builder.values().append_value(part);
                        }
                        builder.append(true);
                    }
                    None => builder.append(false),
                }
            }
            Arc::new(builder.finish())
        }
        StrFunc::ToBinary => Arc::new(map_str(s, uri_path::to_binary)),
        StrFunc::FromBinary => Arc::new(
            s.iter()
                .map(|o| o.and_then(uri_path::from_binary))
                .collect::<StringArray>(),
        ),
        StrFunc::Hamming => {
            use arrow::array::Array;
            let target = require_pattern(pattern, func)?;
            // Unequal lengths have no Hamming distance; DuckDB raises rather than
            // comparing a prefix, and a silent prefix comparison would answer a
            // caller's bug with a plausible number.
            let mut out = Vec::with_capacity(s.len());
            for o in s.iter() {
                match o {
                    None => out.push(None),
                    Some(v) => match uri_path::hamming(v, target) {
                        Some(d) => out.push(Some(d)),
                        None => {
                            return Err(ExprError::InvalidArgument {
                                func: format!("{func:?}"),
                                reason: "strings must be of equal length".into(),
                            })
                        }
                    },
                }
            }
            Arc::new(Int64Array::from(out))
        }
        StrFunc::JaccardSimilarity => {
            use arrow::array::Float64Array;
            let target = require_pattern(pattern, func)?;
            Arc::new(
                s.iter()
                    .map(|o| o.map(|v| uri_path::jaccard(v, target)))
                    .collect::<Float64Array>(),
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
        StrFunc::DamerauLevenshtein => {
            let target = require_pattern(pattern, func)?;
            Arc::new(
                s.iter()
                    .map(|o| o.map(|v| damerau_levenshtein(v, target) as i64))
                    .collect::<Int64Array>(),
            )
        }
        StrFunc::JaroSimilarity => {
            let target = require_pattern(pattern, func)?;
            Arc::new(
                s.iter()
                    .map(|o| o.map(|v| jaro::jaro(v, target)))
                    .collect::<arrow::array::Float64Array>(),
            )
        }
        StrFunc::JaroWinklerSimilarity => {
            let target = require_pattern(pattern, func)?;
            Arc::new(
                s.iter()
                    .map(|o| o.map(|v| jaro::jaro_winkler(v, target)))
                    .collect::<arrow::array::Float64Array>(),
            )
        }
        StrFunc::Soundex => Arc::new(map_str(s, soundex)),
        // A text column compresses its UTF-8 bytes. Decompressing text is accepted for
        // the same reason `unhex` accepts it: the bytes are what matter, and refusing
        // would only force a `cast("binary")` that changes nothing.
        StrFunc::Compress | StrFunc::Decompress => {
            use arrow::array::Array;
            let rows: Vec<Option<&[u8]>> = (0..s.len())
                .map(|i| (!s.is_null(i)).then(|| s.value(i).as_bytes()))
                .collect();
            compress_rows(func, &rows, pattern)?
        }
        StrFunc::ToCase => {
            let style = pattern.ok_or_else(|| ExprError::MissingArgument {
                func: "ToCase".to_string(),
                arg: "style",
            })?;
            // Reject the style once, before touching a row: an unknown style is a plan
            // error, and raising it per row would emit the same message n times.
            if !case::STYLES.contains(&style) {
                return Err(case::unknown_style(style));
            }
            Arc::new(map_str(s, |v| case::to_case(v, style).unwrap_or_default()))
        }
    };
    Ok(out)
}

/// Apply `Compress`/`Decompress` to a column already reduced to `Option<&[u8]>` rows.
///
/// Both the Utf8 and the Binary path funnel here, so the two cannot drift: a text column
/// and the same bytes as a BLOB compress identically. The codec is validated once, before
/// any row is touched — an unknown codec is a plan error, and raising it per row would
/// emit the same message n times.
fn compress_rows(
    func: StrFunc,
    rows: &[Option<&[u8]>],
    pattern: Option<&str>,
) -> Result<ArrayRef, ExprError> {
    use arrow::array::BinaryBuilder;

    let codec = pattern.ok_or_else(|| ExprError::MissingArgument {
        func: format!("{func:?}"),
        arg: "codec",
    })?;
    if !compress::CODECS.contains(&codec) {
        return Err(compress::unknown_codec(&format!("{func:?}"), codec));
    }
    let mut b = BinaryBuilder::with_capacity(rows.len(), rows.len() * 16);
    for row in rows {
        match row {
            None => b.append_null(),
            Some(v) => match func {
                StrFunc::Compress => {
                    b.append_value(compress::compress(v, codec).expect("codec validated above")?)
                }
                // A frame that will not decode is a null row, not a failed batch.
                _ => match compress::decompress(v, codec).expect("codec validated above") {
                    Some(out) => b.append_value(out),
                    None => b.append_null(),
                },
            },
        }
    }
    Ok(Arc::new(b.finish()))
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

/// True (unrestricted) Damerau-Levenshtein distance between `a` and `b`, over UTF-8
/// **bytes** to match DuckDB's `damerau_levenshtein` (same octet rationale as
/// [`levenshtein`]). It adds a fourth edit — transposing two adjacent characters at cost 1 —
/// so a swapped-letter typo (`teh`↔`the`) is one edit.
///
/// This is the Lowrance-Wagner algorithm, **not** the simpler Optimal String Alignment
/// variant: OSA forbids editing a substring more than once and so scores `ca`→`abc` as 3,
/// whereas true DL (and DuckDB) score it 2. The distinction is why this keeps the full
/// `(n+2)×(m+2)` matrix and a per-symbol "last seen row" table (`da`) rather than rolling
/// rows. `da` is indexed by byte value, so its alphabet is a fixed 256 entries.
fn damerau_levenshtein(a: &str, b: &str) -> usize {
    let a = a.as_bytes();
    let b = b.as_bytes();
    let (n, m) = (a.len(), b.len());
    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }
    let big = n + m; // a cost no real alignment can reach, used as the border sentinel
    let w = m + 2;
    let mut d = vec![0usize; (n + 2) * w];
    let at = |i: usize, j: usize| i * w + j;
    d[at(0, 0)] = big;
    for i in 0..=n {
        d[at(i + 1, 0)] = big;
        d[at(i + 1, 1)] = i;
    }
    for j in 0..=m {
        d[at(0, j + 1)] = big;
        d[at(1, j + 1)] = j;
    }
    let mut da = [0usize; 256];
    for i in 1..=n {
        let mut db = 0; // last column j where a[i-1] matched b[j-1]
        for j in 1..=m {
            let k = da[b[j - 1] as usize]; // last row where this symbol was seen in `a`
            let l = db;
            let cost = if a[i - 1] == b[j - 1] {
                db = j;
                0
            } else {
                1
            };
            d[at(i + 1, j + 1)] = (d[at(i, j)] + cost) // substitution / match
                .min(d[at(i + 1, j)] + 1) // insertion
                .min(d[at(i, j + 1)] + 1) // deletion
                .min(d[at(k, l)] + (i - k - 1) + 1 + (j - l - 1)); // transposition
        }
        da[a[i - 1] as usize] = i;
    }
    d[at(n + 1, m + 1)]
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

/// Byte-oriented string functions applied directly to a `Binary`/`LargeBinary` column,
/// bypassing the Utf8 cast that nulls non-UTF-8 rows. Returns `Some(array)` for the
/// functions defined over raw bytes (matching DuckDB's `hex`/`md5`/`sha*`/`base64`/
/// `octet_length` over a BLOB), and `None` for a text-oriented function so the caller
/// falls back to the Utf8 path.
fn eval_bytes(
    func: StrFunc,
    arr: &ArrayRef,
    pattern: Option<&str>,
) -> Result<Option<ArrayRef>, ExprError> {
    use arrow::array::{BinaryArray, LargeBinaryArray};
    // Iterate the rows as `Option<&[u8]>` for either binary offset width.
    let bytes: Vec<Option<&[u8]>> = match arr.data_type() {
        DataType::Binary => arr
            .as_any()
            .downcast_ref::<BinaryArray>()
            .expect("binary")
            .iter()
            .collect(),
        DataType::LargeBinary => arr
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .expect("large binary")
            .iter()
            .collect(),
        _ => return Ok(None),
    };
    let out: ArrayRef = match func {
        StrFunc::Compress | StrFunc::Decompress => {
            return compress_rows(func, &bytes, pattern).map(Some)
        }
        StrFunc::OctetLength => Arc::new(
            bytes
                .iter()
                .map(|o| o.map(|v| v.len() as i64))
                .collect::<Int64Array>(),
        ),
        StrFunc::BitLength => Arc::new(
            bytes
                .iter()
                .map(|o| o.map(|v| (v.len() as i64) * 8))
                .collect::<Int64Array>(),
        ),
        StrFunc::Hex => Arc::new(
            bytes
                .iter()
                .map(|o| o.map(hex_upper))
                .collect::<StringArray>(),
        ),
        StrFunc::Md5 => {
            use md5::{Digest, Md5};
            Arc::new(
                bytes
                    .iter()
                    .map(|o| o.map(|v| hex_lower(Md5::digest(v).as_slice())))
                    .collect::<StringArray>(),
            )
        }
        StrFunc::Sha1 => {
            use sha1::{Digest, Sha1};
            Arc::new(
                bytes
                    .iter()
                    .map(|o| o.map(|v| hex_lower(Sha1::digest(v).as_slice())))
                    .collect::<StringArray>(),
            )
        }
        StrFunc::Sha256 => {
            use sha2::{Digest, Sha256};
            Arc::new(
                bytes
                    .iter()
                    .map(|o| o.map(|v| hex_lower(Sha256::digest(v).as_slice())))
                    .collect::<StringArray>(),
            )
        }
        StrFunc::Base64 => {
            use base64::Engine as _;
            Arc::new(
                bytes
                    .iter()
                    .map(|o| o.map(|v| base64::engine::general_purpose::STANDARD.encode(v)))
                    .collect::<StringArray>(),
            )
        }
        StrFunc::Crc32 => Arc::new(
            bytes
                .iter()
                .map(|o| o.map(|v| crc32fast::hash(v) as i64))
                .collect::<Int64Array>(),
        ),
        StrFunc::XxHash64 => Arc::new(
            bytes
                .iter()
                .map(|o| o.map(|v| xxhash64(v) as i64))
                .collect::<Int64Array>(),
        ),
        StrFunc::Hash64 => Arc::new(
            bytes
                .iter()
                .map(|o| o.map(|v| fnv1a64(v) as i64))
                .collect::<Int64Array>(),
        ),
        // Text-oriented function — defer to the Utf8 path.
        _ => return Ok(None),
    };
    Ok(Some(out))
}

/// Uppercase hexadecimal of the UTF-8 bytes (DuckDB `hex`).
fn hex_encode(v: &str) -> String {
    hex_upper(v.as_bytes())
}

/// Uppercase hexadecimal of arbitrary bytes — DuckDB `hex(BLOB)`, which operates on
/// the raw bytes regardless of UTF-8 validity (the byte-oriented sibling of
/// [`hex_encode`]).
fn hex_upper(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
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

/// Expand one match's rewrite `template` into `dst`, reading groups out of `locs`.
///
/// DuckDB/RE2 rewrite semantics: `\N` expands to capture group `N`'s text, an unmatched
/// group expands to nothing, and `\\` is a literal backslash. Groups are read from a
/// reusable [`regex::CaptureLocations`] plus the subject string rather than an allocated
/// `Captures`, which is what keeps the replace loop free of per-row allocation.
fn expand_rewrite(template: &str, locs: &regex::CaptureLocations, hay: &str, dst: &mut String) {
    let mut chars = template.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some(d) if d.is_ascii_digit() => {
                    if let Some((a, b)) = locs.get(d as usize - '0' as usize) {
                        dst.push_str(&hay[a..b]);
                    }
                }
                Some('\\') => dst.push('\\'),
                // Unreachable for a template that passed `re2_rewrite_valid`; kept total
                // so the function never panics on a hand-written IR document.
                Some(other) => dst.push(other),
                None => {}
            }
        } else {
            dst.push(c);
        }
    }
}

/// Apply a `regexp_replace`/`regexp_replace_all` with DuckDB rewrite semantics.
/// `global` picks first-match vs all-matches. An invalid rewrite template leaves each
/// row unchanged (DuckDB behaviour).
///
/// Written against a **reused** capture buffer rather than `Regex::replace`, which is a
/// per-row allocator: each call builds a fresh `Captures` (an owned slot vector) and a
/// `CaptureMatches` iterator, then hands back a `Cow` that `into_owned` copies again. On
/// ClickBench q28 (`regexp_replace` over 10M `Referer` values) that dominated the query --
/// profiled at 29.9% in `Regex::create_captures`, 15.9% in `CaptureMatches::next` and 10.8%
/// dropping the iterator, against only ~10% in the actual regex search.
///
/// `capture_locations` is allocated once per column and `captures_read_at` writes into it,
/// so the per-row cost is the search plus the output bytes. Match semantics are the
/// regex crate's own leftmost-first scan either way, and the rewrite expansion is shared
/// with [`expand_rewrite`], so the result is bit-identical to the `Replacer` path.
fn regexp_replace_with(s: &StringArray, re: &regex::Regex, rep: &str, global: bool) -> StringArray {
    use arrow::array::{Array, StringBuilder};

    if !re2_rewrite_valid(rep, re.captures_len().saturating_sub(1)) {
        return map_str_borrow(s, |v| v);
    }
    let mut locs = re.capture_locations();
    let mut out = String::new();
    let mut b = StringBuilder::with_capacity(s.len(), s.value_data().len());
    for row in s.iter() {
        let Some(hay) = row else {
            b.append_null();
            continue;
        };
        // `at` is the byte offset the next search starts from; `copied` is how much of
        // `hay` has already been written out. They differ only for an empty match, where
        // the scan must advance a character to terminate but nothing has been consumed.
        let mut at = 0usize;
        let mut copied = 0usize;
        let mut matched = false;
        let mut last_end: Option<usize> = None;
        while at <= hay.len() {
            let Some(m) = re.captures_read_at(&mut locs, hay, at) else {
                break;
            };
            let (ms, me) = (m.start(), m.end());
            // The regex crate's match iterator drops an **empty** match that ends where the
            // previous match ended, so `a*` over "ab" yields "a" then the empty match at 2 --
            // not a second empty match at 1. Skipping it here is what makes this loop agree
            // with `replace_all` (pinned by `regexp_replace_matches_the_regex_crates_own_replacer`).
            if ms == me && Some(me) == last_end {
                match hay[me..].chars().next() {
                    Some(c) => {
                        at = me + c.len_utf8();
                        continue;
                    }
                    None => break,
                }
            }
            if !matched {
                matched = true;
                out.clear();
                out.reserve(hay.len());
            }
            out.push_str(&hay[copied..ms]);
            expand_rewrite(rep, &locs, hay, &mut out);
            copied = me;
            last_end = Some(me);
            if !global {
                break;
            }
            // An empty match would otherwise spin on the same offset forever; step one
            // character past it, exactly as the regex crate's own `replace_all` does.
            at = if me == ms {
                match hay[me..].chars().next() {
                    Some(c) => me + c.len_utf8(),
                    None => break,
                }
            } else {
                me
            };
        }
        if matched {
            out.push_str(&hay[copied..]);
            b.append_value(&out);
        } else {
            b.append_value(hay);
        }
    }
    b.finish()
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
    /// The dictionary fast path must be **bit-identical** to decoding the column and
    /// applying the function per row — it lives inside the correctness oracle, so
    /// "close enough" is not available to it.
    ///
    /// Swept across the string-function family rather than one representative, because
    /// the identity being relied on (an elementwise function commutes with a gather) is
    /// claimed for *all* of them, and a function that quietly depended on row position
    /// would only show up here.
    #[test]
    fn dict_string_functions_match_the_decoded_column() {
        use super::{eval_str, try_dict_str};
        use crate::{Expr, StrFunc};
        use arrow::array::{ArrayRef, DictionaryArray, Int32Array, RecordBatch, StringArray};
        use arrow::datatypes::{Field, Int32Type, Schema};
        use std::sync::Arc;

        // Repeats and nulls in both the keys and the values — the two ways a gather can
        // differ from a per-row map.
        let values = StringArray::from(vec![
            Some("  Hello World  "),
            Some("aBc"),
            None,
            Some(""),
            Some("ünïcødé"),
        ]);
        let keys = Int32Array::from(vec![
            Some(0),
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(0),
            Some(1),
            None,
            Some(4),
        ]);
        let dict: ArrayRef =
            Arc::new(DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values)).unwrap());
        let decoded = crate::eval::dispatch::decode_dict(dict.clone()).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "s",
            dict.data_type().clone(),
            true,
        )]));
        let batch = RecordBatch::try_new(schema, vec![dict]).unwrap();
        let col = Expr::Col {
            name: "s".to_string(),
        };

        /// One dispatch case: the function plus whichever of its optional arguments it
        /// takes — pattern, replacement, start, length.
        type Case = (
            StrFunc,
            Option<&'static str>,
            Option<&'static str>,
            Option<i64>,
            Option<i64>,
        );

        let cases: Vec<Case> = vec![
            (StrFunc::Upper, None, None, None, None),
            (StrFunc::Lower, None, None, None, None),
            (StrFunc::Len, None, None, None, None),
            (StrFunc::Trim, None, None, None, None),
            (StrFunc::LTrim, None, None, None, None),
            (StrFunc::RTrim, None, None, None, None),
            (StrFunc::Reverse, None, None, None, None),
            (StrFunc::Ascii, None, None, None, None),
            (StrFunc::Contains, Some("l"), None, None, None),
            (StrFunc::StartsWith, Some("a"), None, None, None),
            (StrFunc::EndsWith, Some("c"), None, None, None),
            (StrFunc::Position, Some("l"), None, None, None),
            (StrFunc::Substr, None, None, Some(2), Some(3)),
            (StrFunc::Right, None, None, Some(2), None),
            (StrFunc::Repeat, None, None, Some(2), None),
            (StrFunc::Lpad, Some("*"), None, Some(8), None),
            (StrFunc::Rpad, Some("*"), None, Some(8), None),
            (StrFunc::Replace, Some("l"), Some("L"), None, None),
            (StrFunc::RegexpMatches, Some("[A-Z]"), None, None, None),
        ];
        for (func, pattern, replacement, start, length) in cases {
            let fast = try_dict_str(func, &col, &batch, pattern, replacement, start, length)
                .unwrap()
                .unwrap_or_else(|| panic!("{func:?} did not take the dictionary path"));
            let slow = eval_str(func, &decoded, pattern, replacement, start, length).unwrap();
            assert_eq!(
                fast.as_ref(),
                slow.as_ref(),
                "{func:?} disagreed with the decoded column"
            );
        }
    }

    /// Manual throughput check (ignored). Run:
    ///   cargo test -p bc-expr dict_str_bench -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dict_str_bench() {
        use super::{eval_str, try_dict_str};
        use crate::{Expr, StrFunc};
        use arrow::array::{ArrayRef, DictionaryArray, Int32Array, RecordBatch, StringArray};
        use arrow::datatypes::{Field, Int32Type, Schema};
        use std::sync::Arc;
        use std::time::Instant;

        // ClickBench's shape: millions of rows over a few dozen distinct strings.
        const ROWS: usize = 6_000_000;
        const DISTINCT: usize = 25;
        let values = StringArray::from(
            (0..DISTINCT)
                .map(|i| Some(format!("user-agent-string-number-{i}")))
                .collect::<Vec<_>>(),
        );
        let keys = Int32Array::from((0..ROWS).map(|i| (i % DISTINCT) as i32).collect::<Vec<_>>());
        let dict: ArrayRef =
            Arc::new(DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values)).unwrap());
        let decoded = crate::eval::dispatch::decode_dict(dict.clone()).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "s",
            dict.data_type().clone(),
            true,
        )]));
        let batch = RecordBatch::try_new(schema, vec![dict]).unwrap();
        let col = Expr::Col {
            name: "s".to_string(),
        };

        for func in [StrFunc::Upper, StrFunc::Len, StrFunc::Reverse] {
            let t = Instant::now();
            let slow = eval_str(func, &decoded, None, None, None, None).unwrap();
            let slow_ms = t.elapsed().as_secs_f64() * 1000.0;
            let t = Instant::now();
            let fast = try_dict_str(func, &col, &batch, None, None, None, None)
                .unwrap()
                .unwrap();
            let fast_ms = t.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(fast.as_ref(), slow.as_ref());
            println!(
                "{func:?}: decoded {slow_ms:.1} ms -> dict {fast_ms:.1} ms ({:.1}x)",
                slow_ms / fast_ms
            );
        }
    }

    /// A non-dictionary column must decline the fast path rather than mis-handle it.
    #[test]
    fn a_plain_column_does_not_take_the_dictionary_path() {
        use super::try_dict_str;
        use crate::{Expr, StrFunc};
        use arrow::array::{ArrayRef, RecordBatch, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let arr: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), Some("b")]));
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();
        let col = Expr::Col {
            name: "s".to_string(),
        };

        assert!(
            try_dict_str(StrFunc::Upper, &col, &batch, None, None, None, None)
                .unwrap()
                .is_none()
        );
    }

    fn chunks_bounded(
        input: Vec<Option<&str>>,
        size: i64,
        overlap: i64,
        boundary: &str,
    ) -> Vec<Option<Vec<String>>> {
        use crate::StrFunc;
        use arrow::array::{Array, ArrayRef, ListArray, StringArray};
        use std::sync::Arc;
        let arr: ArrayRef = Arc::new(StringArray::from(input));
        let out = eval_str(
            StrFunc::Chunk,
            &arr,
            Some(boundary),
            None,
            Some(overlap),
            Some(size),
        )
        .unwrap();
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

    /// A word boundary must never cut a word in half — the failure that silently costs
    /// retrieval recall, because the truncated fragment embeds as something else.
    #[test]
    fn chunk_word_boundary_never_splits_a_word() {
        let text = "alpha beta gamma delta epsilon";
        let got = chunks_bounded(vec![Some(text)], 12, 0, "word")
            .remove(0)
            .unwrap();
        for c in &got {
            let trimmed = c.trim();
            assert!(
                trimmed.is_empty() || text.split_whitespace().any(|w| trimmed.ends_with(w)),
                "chunk {c:?} ends mid-word"
            );
        }
    }

    /// With no overlap, chunking is lossless whatever the boundary mode: a separator ends
    /// the chunk it belongs to instead of being skipped.
    #[test]
    fn chunk_with_no_overlap_is_lossless_in_every_mode() {
        let text = "One sentence. Another one here! A third? And a tail with no stop";
        for mode in ["char", "word", "sentence", "line"] {
            let got = chunks_bounded(vec![Some(text)], 16, 0, mode)
                .remove(0)
                .unwrap();
            assert_eq!(got.concat(), text, "mode {mode} dropped or duplicated text");
        }
    }

    /// A sentence boundary ends chunks on terminal punctuation where one is available.
    #[test]
    fn chunk_sentence_boundary_ends_on_punctuation() {
        let text = "First one. Second one. Third one.";
        let got = chunks_bounded(vec![Some(text)], 16, 0, "sentence")
            .remove(0)
            .unwrap();
        // Every chunk but the last ends at a sentence terminator.
        for c in &got[..got.len() - 1] {
            assert!(c.ends_with('.'), "chunk {c:?} does not end a sentence");
        }
    }

    /// When a window holds no sentence terminator, `sentence` must fall back to a *word*
    /// boundary, not to an arbitrary character — asking for readable chunks and getting a
    /// mid-word cut is worse than not asking.
    #[test]
    fn chunk_sentence_falls_back_to_a_word_boundary_not_a_hard_cut() {
        // The first full stop is past the 40-character window.
        let text = "The patient was diagnosed with hypertension. Treatment began.";
        let got = chunks_bounded(vec![Some(text)], 40, 0, "sentence")
            .remove(0)
            .unwrap();
        assert!(
            got[0].ends_with(' ') || got[0].ends_with('.'),
            "first chunk {:?} was cut mid-word",
            got[0]
        );
        assert_eq!(got.concat(), text);
    }

    /// A token longer than the whole chunk has no boundary to back off to, and must
    /// still be emitted rather than stalling the loop.
    #[test]
    fn chunk_emits_a_token_longer_than_the_chunk_size() {
        let text = "aaaaaaaaaaaaaaaaaaaaaaaa bb";
        let got = chunks_bounded(vec![Some(text)], 8, 0, "word")
            .remove(0)
            .unwrap();
        assert!(!got.is_empty());
        assert_eq!(got.concat(), text);
    }

    /// Boundary mode must not disturb the null/empty contract.
    #[test]
    fn chunk_boundary_keeps_the_null_and_empty_contract() {
        let got = chunks_bounded(vec![None, Some("")], 8, 0, "word");
        assert_eq!(got[0], None);
        assert_eq!(got[1], Some(vec![]));
    }

    /// An unknown mode is rejected rather than silently treated as `char`.
    #[test]
    fn chunk_rejects_an_unknown_boundary() {
        use crate::StrFunc;
        use arrow::array::{ArrayRef, StringArray};
        use std::sync::Arc;
        let arr: ArrayRef = Arc::new(StringArray::from(vec![Some("abc")]));
        let err = eval_str(
            StrFunc::Chunk,
            &arr,
            Some("paragraph"),
            None,
            Some(0),
            Some(2),
        )
        .unwrap_err();
        assert!(format!("{err}").contains("unknown chunk boundary"), "{err}");
    }

    /// Non-ASCII text takes the offset-table path; boundaries must work there too.
    #[test]
    fn chunk_word_boundary_handles_non_ascii() {
        let text = "café naïve résumé";
        let got = chunks_bounded(vec![Some(text)], 8, 0, "word")
            .remove(0)
            .unwrap();
        assert_eq!(got.concat(), text);
    }

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

    /// Byte-oriented functions over a `Binary` column with **non-UTF-8** bytes operate on
    /// the raw bytes (DuckDB `hex`/`md5`/`sha256`/`base64`/`octet_length` over a BLOB),
    /// instead of routing through the Utf8 cast that nulled every non-textual row.
    /// Regression: `hex(BLOB '\xDE\xAD\xBE\xEF')` used to return NULL, not `'DEADBEEF'`.
    #[test]
    fn byte_functions_over_non_utf8_binary_use_the_raw_bytes() {
        use crate::StrFunc;
        use arrow::array::{Array, ArrayRef, BinaryArray, Int64Array, StringArray};
        use md5::{Digest, Md5};
        use sha2::Sha256;
        use std::sync::Arc;

        // `\xDE\xAD\xBE\xEF` is not valid UTF-8 (a stray continuation byte at index 2).
        let raw = &[0xDEu8, 0xAD, 0xBE, 0xEF];
        let bin: ArrayRef = Arc::new(BinaryArray::from_opt_vec(vec![
            Some(raw.as_ref()),
            Some(b""),
            None,
        ]));

        let str_of = |f: StrFunc| {
            let out = eval_str(f, &bin, None, None, None, None).unwrap();
            let a = out.as_any().downcast_ref::<StringArray>().unwrap();
            (a.value(0).to_string(), a.value(1).to_string(), a.is_null(2))
        };
        let int_of = |f: StrFunc| {
            let out = eval_str(f, &bin, None, None, None, None).unwrap();
            let a = out.as_any().downcast_ref::<Int64Array>().unwrap();
            (a.value(0), a.value(1), a.is_null(2))
        };

        assert_eq!(int_of(StrFunc::OctetLength), (4, 0, true));
        assert_eq!(int_of(StrFunc::BitLength), (32, 0, true));
        assert_eq!(
            str_of(StrFunc::Hex),
            ("DEADBEEF".to_string(), String::new(), true)
        );
        assert_eq!(
            str_of(StrFunc::Md5).0,
            hex_lower(Md5::digest(raw).as_slice())
        );
        assert_eq!(
            str_of(StrFunc::Sha256).0,
            hex_lower(Sha256::digest(raw).as_slice())
        );
        // base64 of the four raw bytes.
        assert_eq!(str_of(StrFunc::Base64).0, "3q2+7w==");
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

    /// `token_ngrams` splits on whitespace and joins each window of `n` adjacent tokens:
    /// overlapping n-grams, a single all-tokens gram when there are fewer than `n`, and an
    /// empty list for empty/whitespace-only input.
    #[test]
    fn token_ngrams_windows_of_n_tokens() {
        use crate::StrFunc;
        use arrow::array::{Array, ArrayRef, ListArray, StringArray};
        use std::sync::Arc;
        let arr: ArrayRef = Arc::new(StringArray::from(vec![
            Some("the cat sat"),
            Some("solo"),
            Some("  "),
            Some(""),
            None,
        ]));
        let out = eval_str(StrFunc::TokenNgrams, &arr, None, None, None, Some(2)).unwrap();
        let list = out.as_any().downcast_ref::<ListArray>().unwrap();
        let row = |i: usize| {
            let v = list.value(i);
            let v = v.as_any().downcast_ref::<StringArray>().unwrap();
            (0..v.len())
                .map(|j| v.value(j).to_string())
                .collect::<Vec<_>>()
        };
        assert_eq!(row(0), ["the cat", "cat sat"]); // overlapping bigrams
        assert_eq!(row(1), ["solo"]); // fewer than n tokens → one all-tokens gram
        assert!(row(2).is_empty()); // whitespace only → empty list
        assert!(row(3).is_empty()); // empty string → empty list
        assert!(list.is_null(4)); // null → null list
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

    #[test]
    fn damerau_levenshtein_scores_transpositions_as_one() {
        use super::{damerau_levenshtein, levenshtein};
        // An adjacent swap is one edit here, two under plain Levenshtein.
        assert_eq!(damerau_levenshtein("ca", "ac"), 1);
        assert_eq!(levenshtein("ca", "ac"), 2);
        assert_eq!(damerau_levenshtein("teh", "the"), 1);
        // Non-transposition cases agree with Levenshtein.
        assert_eq!(damerau_levenshtein("kitten", "sitting"), 3);
        assert_eq!(damerau_levenshtein("abc", "abc"), 0);
        assert_eq!(damerau_levenshtein("", "abc"), 3);
        assert_eq!(damerau_levenshtein("abc", ""), 3);
        // True (unrestricted) DL, matching DuckDB: transpose ca→ac then insert b = 2.
        assert_eq!(damerau_levenshtein("ca", "abc"), 2);
    }

    /// `regexp_extract_all` collects a *capture group* of every match when one is asked
    /// for, not the whole match. The group index rides `start`, as it does for the scalar
    /// `RegexpExtract`; before it was honoured the kernel called `find_iter`
    /// unconditionally, so group 1 of `(\d+)-(\d+)` came back as the whole `100-200`.
    #[test]
    fn regexp_extract_all_collects_the_requested_capture_group() {
        use crate::StrFunc;
        use arrow::array::{Array, ArrayRef, ListArray, StringArray};
        use std::sync::Arc;

        let arr: ArrayRef = Arc::new(StringArray::from(vec![Some("100-200, 300-400")]));
        let row = |group: i64| -> Vec<Option<String>> {
            let out = eval_str(
                StrFunc::RegexpExtractAll,
                &arr,
                Some(r"(\d+)-(\d+)"),
                None,
                Some(group),
                None,
            )
            .unwrap();
            let list = out.as_any().downcast_ref::<ListArray>().unwrap();
            let values = list.value(0);
            let strings = values.as_any().downcast_ref::<StringArray>().unwrap();
            (0..strings.len())
                .map(|i| (!strings.is_null(i)).then(|| strings.value(i).to_string()))
                .collect()
        };
        let owned = |v: &[&str]| -> Vec<Option<String>> {
            v.iter().map(|s| Some((*s).to_string())).collect()
        };
        // Group 0 (and the default) is the whole match; 1 and 2 are the two halves.
        assert_eq!(row(0), owned(&["100-200", "300-400"]));
        assert_eq!(row(1), owned(&["100", "300"]));
        assert_eq!(row(2), owned(&["200", "400"]));

        // A branch that did not participate is a NULL element, matching DuckDB — the
        // scalar `regexp_extract` yields `""` for the same case, so the two differ.
        let alt: ArrayRef = Arc::new(StringArray::from(vec![Some("a1 b")]));
        let out = eval_str(
            StrFunc::RegexpExtractAll,
            &alt,
            Some(r"(\d)|(x)"),
            None,
            Some(2),
            None,
        )
        .unwrap();
        let list = out.as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(list.value(0).len(), 1);
        assert!(list.value(0).is_null(0));

        // A group the pattern does not have is rejected, not silently empty.
        assert!(eval_str(
            StrFunc::RegexpExtractAll,
            &arr,
            Some(r"(\d+)"),
            None,
            Some(5),
            None,
        )
        .is_err());
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

    /// The allocation-free replace loop must be **bit-identical** to driving the `regex`
    /// crate's own `replace`/`replace_all` with the same rewrite semantics.
    ///
    /// `regexp_replace_with` was rewritten to reuse one `CaptureLocations` instead of
    /// allocating a `Captures` per row (29.9% of ClickBench q28 was `create_captures`).
    /// That hand-rolls the scan loop, so the two places it could drift from the library
    /// are pinned here: **empty matches**, where the scan must step a character to
    /// terminate without consuming input, and **multibyte UTF-8**, where stepping by one
    /// byte would slice a char boundary and panic. Patterns that can match empty (`a*`,
    /// `x?`, `(?:)`) are included deliberately, as are non-ASCII subjects.
    #[test]
    fn regexp_replace_matches_the_regex_crates_own_replacer() {
        use crate::StrFunc;
        use arrow::array::{Array, ArrayRef, StringArray};
        use std::sync::Arc;

        // The reference: expand the RE2 template through the regex crate's `Captures`.
        fn reference(hay: &str, re: &regex::Regex, rep: &str, global: bool) -> String {
            let expand = |caps: &regex::Captures, dst: &mut String| {
                let mut chars = rep.chars();
                while let Some(c) = chars.next() {
                    if c == '\\' {
                        match chars.next() {
                            Some(d) if d.is_ascii_digit() => {
                                if let Some(m) = caps.get(d as usize - '0' as usize) {
                                    dst.push_str(m.as_str());
                                }
                            }
                            Some('\\') => dst.push('\\'),
                            Some(other) => dst.push(other),
                            None => {}
                        }
                    } else {
                        dst.push(c);
                    }
                }
            };
            if global {
                re.replace_all(hay, |c: &regex::Captures| {
                    let mut s = String::new();
                    expand(c, &mut s);
                    s
                })
                .into_owned()
            } else {
                re.replace(hay, |c: &regex::Captures| {
                    let mut s = String::new();
                    expand(c, &mut s);
                    s
                })
                .into_owned()
            }
        }

        let subjects = [
            "",
            "ab",
            "xaby",
            "abab",
            "aaa",
            "no-match-here",
            "https://www.example.com/a/b",
            "héllo wörld",
            "日本語テキスト",
            "a日b日c",
            "trailing",
            "//",
            "a",
            "\\escaped",
        ];
        let patterns = [
            "(a)(b)",
            "a*",
            "x?",
            "(?:)",
            "[^/]+",
            "^https?://(?:www\\.)?([^/]+)/.*$",
            "(\\w+)",
            "日",
            "(é)",
            "$",
            "^",
        ];
        let templates = [r"\1", r"\0", r"\2\1", "-", "", r"\\", r"[\1]"];

        let arr: ArrayRef = Arc::new(StringArray::from(
            subjects.iter().map(|s| Some(*s)).collect::<Vec<_>>(),
        ));
        for pat in patterns {
            let re = regex::Regex::new(pat).unwrap();
            for rep in templates {
                if !super::re2_rewrite_valid(rep, re.captures_len().saturating_sub(1)) {
                    continue; // invalid template is a documented no-op, covered elsewhere
                }
                for (func, global) in [
                    (StrFunc::RegexpReplace, false),
                    (StrFunc::RegexpReplaceAll, true),
                ] {
                    let out = eval_str(func, &arr, Some(pat), Some(rep), None, None).unwrap();
                    let got = out.as_any().downcast_ref::<StringArray>().unwrap();
                    for (i, hay) in subjects.iter().enumerate() {
                        assert_eq!(
                            got.value(i),
                            reference(hay, &re, rep, global),
                            "pattern={pat:?} template={rep:?} global={global} subject={hay:?}"
                        );
                    }
                }
            }
        }
    }
}
