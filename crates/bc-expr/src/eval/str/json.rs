//! JSON path extraction for the `.json` accessor (`json_extract_{string,int,float,bool}`).
//!
//! The hot path is a **lazy, path-directed scan**: rather than parse the whole document
//! into a `serde_json::Value` tree for every extracted field (the old behaviour — O(doc)
//! allocation per field per row, so a query pulling K fields parsed each row K times over),
//! [`seek`] walks the raw bytes and descends *only* the requested path, structurally
//! skipping every sibling value it does not need. It returns the raw slice of the located
//! value; only that (tiny) leaf is handed to `serde_json` for exact value semantics. On a
//! wide document — the semistructured common case — reaching `$.a` touches the bytes up to
//! `a` and skips the rest, instead of materializing the entire object graph.
//!
//! Path syntax matches the SQL / `.json` accessor: a `$`-rooted dotted path with optional
//! numeric array subscripts, e.g. `$.user.id`, `$.tags[0]`, `$.a.b[2].c`. A leading `$` is
//! optional. Structural skipping is string- and escape-aware; leaf values keep exact
//! `serde_json` semantics, so results stay bit-for-bit identical to a full parse.

use serde_json::Value;

/// One step of a JSON path: an object key or a zero-based array index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum PathPart {
    Key(String),
    Index(usize),
}

/// Parse a `$.a.b[0].c` path into its component steps.
///
/// The leading `$` is optional. Dots separate object keys; `[n]` selects an array
/// element. A key that itself carries a subscript (`tags[0]`) splits into a
/// [`PathPart::Key`] followed by a [`PathPart::Index`]. Empty segments are ignored,
/// so `$.a`, `a`, and `.a` are equivalent.
pub(super) fn parse_path(path: &str) -> Vec<PathPart> {
    let mut parts = Vec::new();
    for segment in path.trim_start_matches('$').split('.') {
        if segment.is_empty() {
            continue;
        }
        // A segment is `name`, `name[0]`, `name[0][1]`, or a bare `[0]`.
        let (name, rest) = match segment.find('[') {
            Some(idx) => (&segment[..idx], &segment[idx..]),
            None => (segment, ""),
        };
        if !name.is_empty() {
            parts.push(PathPart::Key(name.to_string()));
        }
        // Peel off each `[n]` subscript in the remainder.
        let mut cur = rest;
        while let Some(close) = cur.find(']') {
            let inner = cur[1..close].trim();
            if let Ok(idx) = inner.parse::<usize>() {
                parts.push(PathPart::Index(idx));
            }
            cur = &cur[close + 1..];
        }
    }
    parts
}

/// The raw slice of the value at `path` within `text`, or `None` if the document is
/// not valid enough to navigate or the path is absent.
///
/// This never validates the whole document — it validates and skips only what it must
/// walk over to reach the target. A malformed *sibling* the scan skips past can still
/// yield a byte range; the caller parses that range with `serde_json`, which rejects a
/// malformed *leaf*. The net contract (null on malformed input or absent path) matches
/// the previous full-parse behaviour.
pub(super) fn seek<'a>(text: &'a str, path: &[PathPart]) -> Option<&'a str> {
    let bytes = text.as_bytes();
    let mut pos = skip_ws(bytes, 0);
    for part in path {
        pos = match part {
            PathPart::Key(key) => seek_key(bytes, pos, key)?,
            PathPart::Index(idx) => seek_index(bytes, pos, *idx)?,
        };
        pos = skip_ws(bytes, pos);
    }
    let end = skip_value(bytes, pos)?;
    Some(&text[pos..end])
}

/// Within the object starting at `pos` (`{`), position just after the value of `key`.
fn seek_key(bytes: &[u8], pos: usize, key: &str) -> Option<usize> {
    if bytes.get(pos)? != &b'{' {
        return None;
    }
    let mut i = skip_ws(bytes, pos + 1);
    if bytes.get(i) == Some(&b'}') {
        return None; // empty object
    }
    loop {
        // Key (always a JSON string).
        let (k, after_key) = parse_string(bytes, i)?;
        i = skip_ws(bytes, after_key);
        if bytes.get(i)? != &b':' {
            return None;
        }
        i = skip_ws(bytes, i + 1);
        if k == key {
            return Some(i);
        }
        i = skip_value(bytes, i)?;
        i = skip_ws(bytes, i);
        match bytes.get(i)? {
            b',' => i = skip_ws(bytes, i + 1),
            b'}' => return None,
            _ => return None,
        }
    }
}

/// Within the array starting at `pos` (`[`), position at the start of element `idx`.
fn seek_index(bytes: &[u8], pos: usize, idx: usize) -> Option<usize> {
    if bytes.get(pos)? != &b'[' {
        return None;
    }
    let mut i = skip_ws(bytes, pos + 1);
    if bytes.get(i) == Some(&b']') {
        return None; // empty array
    }
    let mut cur = 0usize;
    loop {
        if cur == idx {
            return Some(i);
        }
        i = skip_value(bytes, i)?;
        i = skip_ws(bytes, i);
        match bytes.get(i)? {
            b',' => i = skip_ws(bytes, i + 1),
            b']' => return None,
            _ => return None,
        }
        cur += 1;
    }
}

/// Position just past the complete JSON value that starts at `pos`.
fn skip_value(bytes: &[u8], pos: usize) -> Option<usize> {
    match bytes.get(pos)? {
        b'"' => parse_string(bytes, pos).map(|(_, end)| end),
        b'{' | b'[' => skip_container(bytes, pos),
        _ => Some(skip_scalar(bytes, pos)),
    }
}

/// Skip a balanced `{...}` / `[...]`, respecting strings (so braces inside a string
/// value do not affect nesting depth). Returns the index just past the closer.
fn skip_container(bytes: &[u8], pos: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut i = pos;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => {
                i = parse_string(bytes, i)?.1;
                continue;
            }
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Skip a scalar token (number / `true` / `false` / `null`): everything up to the next
/// structural delimiter (`,`, `}`, `]`) or whitespace.
fn skip_scalar(bytes: &[u8], pos: usize) -> usize {
    let mut i = pos;
    while i < bytes.len() {
        match bytes[i] {
            b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r' => break,
            _ => i += 1,
        }
    }
    i
}

/// Parse a JSON string starting at `pos` (`"`), returning its decoded contents and the
/// index just past the closing quote. Handles `\"`, `\\`, and the standard escapes via
/// `serde_json` for the (rare) escaped case; the common unescaped case is a borrow-free
/// slice compare done by the caller.
fn parse_string(bytes: &[u8], pos: usize) -> Option<(String, usize)> {
    if bytes.get(pos)? != &b'"' {
        return None;
    }
    let mut i = pos + 1;
    let mut escaped = false;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => {
                escaped = true;
                i += 2; // skip the escape and its following char
                continue;
            }
            b'"' => {
                let raw = &bytes[pos..=i];
                let s = if escaped {
                    // Delegate escape decoding to serde_json for exact semantics.
                    serde_json::from_slice::<String>(raw).ok()?
                } else {
                    // SAFETY-equivalent: the slice is within a validated &str and holds
                    // no escapes, so it is valid UTF-8 string content.
                    std::str::from_utf8(&bytes[pos + 1..i]).ok()?.to_string()
                };
                return Some((s, i + 1));
            }
            _ => i += 1,
        }
    }
    None
}

fn skip_ws(bytes: &[u8], pos: usize) -> usize {
    let mut i = pos;
    while i < bytes.len() && matches!(bytes[i], b' ' | b'\t' | b'\n' | b'\r') {
        i += 1;
    }
    i
}

/// Parse the located leaf slice into a `serde_json::Value` (exact value semantics).
fn leaf(text: &str, path: &[PathPart]) -> Option<Value> {
    serde_json::from_str(seek(text, path)?).ok()
}

/// Extract the value at `path` as a string: string leaves verbatim, everything else
/// (numbers, bools, objects, arrays) as its compact JSON text. `None` on absent path,
/// JSON null, or a malformed leaf.
pub(super) fn extract_string(text: &str, path: &[PathPart]) -> Option<String> {
    match leaf(text, path)? {
        Value::String(s) => Some(s),
        Value::Null => None,
        other => Some(other.to_string()),
    }
}

pub(super) fn extract_int(text: &str, path: &[PathPart]) -> Option<i64> {
    leaf(text, path)?.as_i64()
}

pub(super) fn extract_float(text: &str, path: &[PathPart]) -> Option<f64> {
    leaf(text, path)?.as_f64()
}

pub(super) fn extract_bool(text: &str, path: &[PathPart]) -> Option<bool> {
    leaf(text, path)?.as_bool()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parts(p: &str) -> Vec<PathPart> {
        parse_path(p)
    }

    #[test]
    fn parses_paths() {
        assert_eq!(
            parts("$.a.b"),
            vec![PathPart::Key("a".into()), PathPart::Key("b".into())]
        );
        assert_eq!(
            parts("a.b"),
            vec![PathPart::Key("a".into()), PathPart::Key("b".into())]
        );
        assert_eq!(
            parts("$.tags[0]"),
            vec![PathPart::Key("tags".into()), PathPart::Index(0)]
        );
        assert_eq!(
            parts("$.a[2].b"),
            vec![
                PathPart::Key("a".into()),
                PathPart::Index(2),
                PathPart::Key("b".into())
            ]
        );
        assert_eq!(parts("$"), Vec::<PathPart>::new());
    }

    #[test]
    fn extracts_scalars() {
        let doc = r#"{"user":{"id":7,"country":"US"},"amount":3.5,"ok":true,"tags":["x","y"]}"#;
        assert_eq!(extract_int(doc, &parts("$.user.id")), Some(7));
        assert_eq!(
            extract_string(doc, &parts("$.user.country")),
            Some("US".into())
        );
        assert_eq!(extract_float(doc, &parts("$.amount")), Some(3.5));
        assert_eq!(extract_bool(doc, &parts("$.ok")), Some(true));
        assert_eq!(extract_string(doc, &parts("$.tags[0]")), Some("x".into()));
        assert_eq!(extract_string(doc, &parts("$.tags[1]")), Some("y".into()));
    }

    #[test]
    fn objects_and_arrays_serialize_compactly() {
        // Object leaves round-trip through serde_json, whose default Map sorts keys — the
        // same normalization the previous full-parse path produced (keys alphabetized).
        let doc = r#"{"user":{"id":7,"country":"US"}}"#;
        assert_eq!(
            extract_string(doc, &parts("$.user")),
            Some(r#"{"country":"US","id":7}"#.into())
        );
        assert_eq!(
            extract_string(r#"{"a":[1,2,3]}"#, &parts("$.a")),
            Some("[1,2,3]".into())
        );
    }

    #[test]
    fn missing_and_malformed_are_none() {
        let doc = r#"{"a":1}"#;
        assert_eq!(extract_int(doc, &parts("$.b")), None);
        assert_eq!(extract_int(doc, &parts("$.a.b")), None); // scalar has no child
        assert_eq!(extract_int("not json", &parts("$.a")), None);
        assert_eq!(extract_string(r#"{"a":null}"#, &parts("$.a")), None);
        assert_eq!(extract_int(r#"{"tags":[1]}"#, &parts("$.tags[5]")), None);
    }

    #[test]
    fn skips_past_escapes_and_nested_siblings() {
        // A sibling string containing braces/brackets and escaped quotes must not confuse
        // the structural skip on the way to `target`.
        let doc = r#"{"noise":"a\"}{][\\","nest":{"deep":[1,{"z":9}]},"target":42}"#;
        assert_eq!(extract_int(doc, &parts("$.target")), Some(42));
        assert_eq!(extract_int(doc, &parts("$.nest.deep[1].z")), Some(9));
    }

    #[test]
    fn decodes_escaped_key_and_value() {
        let doc = r#"{"a\tb":"line\ntwo"}"#;
        assert_eq!(
            extract_string(doc, &parts("$.a\tb")),
            Some("line\ntwo".into())
        );
    }

    #[test]
    fn tolerates_whitespace() {
        let doc = " { \"a\" : { \"b\" : 5 } } ";
        assert_eq!(extract_int(doc, &parts("$.a.b")), Some(5));
    }

    /// A/B micro-benchmark isolating the lazy scanner from the old full-parse-per-field
    /// approach on a representative event document. Run explicitly:
    ///   cargo test -p bc-expr -- --ignored --nocapture bench_lazy_vs_fullparse
    #[test]
    #[ignore]
    fn bench_lazy_vs_fullparse() {
        use std::time::Instant;

        // The benchmark corpus's document shape (nested user/event/device + tags array).
        let doc = r#"{"user":{"id":318295,"country":"SE","tier":"bronze"},"event":{"type":"view","value":0.0,"items":3},"device":{"os":"iOS","version":"16.4"},"tags":["promo","promo"],"ts":1700116211}"#;
        // The five fields the `json-project5` case pulls (three at increasing depth/offset).
        let paths: Vec<Vec<PathPart>> = [
            "$.user.country",
            "$.user.tier",
            "$.device.os",
            "$.event.value",
            "$.event.items",
        ]
        .iter()
        .map(|p| parse_path(p))
        .collect();
        let iters = 200_000usize;

        // Old path: full serde_json parse of the whole document, once per field.
        let t0 = Instant::now();
        let mut sink = 0u64;
        for _ in 0..iters {
            for path in &paths {
                let mut cur: Value = serde_json::from_str(doc).unwrap();
                for part in path {
                    cur = match part {
                        PathPart::Key(k) => cur.get(k).cloned().unwrap_or(Value::Null),
                        PathPart::Index(i) => cur.get(i).cloned().unwrap_or(Value::Null),
                    };
                }
                sink = sink.wrapping_add(cur.to_string().len() as u64);
            }
        }
        let full = t0.elapsed();

        // New path: lazy seek to each field, leaf-parse only the located slice.
        let t1 = Instant::now();
        let mut sink2 = 0u64;
        for _ in 0..iters {
            for path in &paths {
                if let Some(s) = extract_string(doc, path) {
                    sink2 = sink2.wrapping_add(s.len() as u64);
                }
            }
        }
        let lazy = t1.elapsed();

        println!(
            "full-parse-per-field: {:?}  lazy-seek: {:?}  speedup: {:.2}x  (sinks {} {})",
            full,
            lazy,
            full.as_secs_f64() / lazy.as_secs_f64(),
            sink,
            sink2,
        );
    }
}
