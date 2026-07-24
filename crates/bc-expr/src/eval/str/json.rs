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

/// One step of a JSON path: an object key or an array index.
///
/// The index is signed: a non-negative index counts from the front (`[0]` is the
/// first element), a negative index counts from the back (`[-1]` is the last),
/// matching DuckDB's JSON path semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum PathPart {
    Key(String),
    Index(i64),
}

/// Parse a `$.a.b[0].c` path into its component steps.
///
/// The leading `$` is optional. Dots separate object keys; `[n]` selects an array
/// element (`[-1]` the last, counting from the back — DuckDB semantics). A key that
/// itself carries a subscript (`tags[0]`) splits into a [`PathPart::Key`] followed by
/// a [`PathPart::Index`]. Empty segments are ignored, so `$.a`, `a`, and `.a` are
/// equivalent.
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
            if let Ok(idx) = inner.parse::<i64>() {
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
///
/// A non-negative `idx` counts from the front and terminates as soon as it is reached
/// (the fast path — no need to see the rest of the array). A negative `idx` counts from
/// the back (`-1` is the last element, DuckDB semantics); since the element count is not
/// known up front, the array is scanned once, recording each element's start, then
/// indexed from the end.
fn seek_index(bytes: &[u8], pos: usize, idx: i64) -> Option<usize> {
    if bytes.get(pos)? != &b'[' {
        return None;
    }
    let mut i = skip_ws(bytes, pos + 1);
    if bytes.get(i) == Some(&b']') {
        return None; // empty array
    }
    if idx >= 0 {
        let target = idx as usize;
        let mut cur = 0usize;
        loop {
            if cur == target {
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
    // Negative index: fold from the end, so scan the whole array once recording starts.
    let mut starts: Vec<usize> = Vec::new();
    loop {
        starts.push(i);
        i = skip_value(bytes, i)?;
        i = skip_ws(bytes, i);
        match bytes.get(i)? {
            b',' => i = skip_ws(bytes, i + 1),
            b']' => break,
            _ => return None,
        }
    }
    let eff = starts.len() as i64 + idx;
    if eff < 0 {
        return None;
    }
    starts.get(eff as usize).copied()
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

/// Minify a *validated* JSON slice: drop whitespace that sits outside string tokens,
/// copying every string verbatim. Unlike round-tripping through `serde_json::Value`,
/// this **preserves object key order** — `serde_json`'s default `Map` re-sorts keys,
/// which silently reordered an extracted sub-object relative to its source (and to
/// DuckDB's `json_extract_string`, which keeps insertion order). The input is already
/// structurally valid (its caller parsed it), so stripping inter-token whitespace
/// yields the same compact text DuckDB does.
fn compact(raw: &str) -> String {
    let bytes = raw.as_bytes();
    let mut out = String::with_capacity(raw.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => {
                // A string may hold whitespace and escaped quotes — copy it whole.
                let end = parse_string(bytes, i).map_or(bytes.len(), |(_, e)| e);
                out.push_str(&raw[i..end]);
                i = end;
            }
            b' ' | b'\t' | b'\n' | b'\r' => i += 1,
            // Everything else outside a string in valid JSON is ASCII structure.
            c => {
                out.push(c as char);
                i += 1;
            }
        }
    }
    out
}

/// Extract the value at `path` as a string: string leaves verbatim, everything else
/// (numbers, bools, objects, arrays) as its compact JSON text. `None` on absent path,
/// JSON null, or a malformed leaf.
///
/// Object/array leaves keep their **source key/element order** (see [`compact`]) rather
/// than being re-serialized through `serde_json::Value`, which alphabetizes object keys.
pub(super) fn extract_string(text: &str, path: &[PathPart]) -> Option<String> {
    render_leaf(seek(text, path)?)
}

/// Whether `s` is a bare JSON integer literal (`-?[0-9]+`, no fraction or exponent).
fn is_integer_literal(s: &str) -> bool {
    let digits = s.strip_prefix('-').unwrap_or(s);
    !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
}

/// Extract the value at `path` as an `i64`, matching DuckDB's `json_extract(...)::BIGINT`
/// **numeric** cast: an integer leaf verbatim, a JSON float rounded to nearest (ties to
/// even, as DuckDB does), and a JSON bool as `1`/`0`. A value outside `i64`'s range
/// (DuckDB raises there) yields `None`, as do a container/string/JSON-null leaf, a
/// missing path, or malformed JSON. Previously a JSON *float* (`3.5`, or an integrally
/// valued `42.0`/`1e2`) returned `None`, silently dropping data DuckDB extracts.
pub(super) fn extract_int(text: &str, path: &[PathPart]) -> Option<i64> {
    match leaf(text, path)? {
        Value::Number(n) => number_to_i64(&n),
        Value::Bool(b) => Some(b as i64),
        _ => None,
    }
}

/// A JSON number as `i64`: exact for an integer that fits, else the float rounded to the
/// nearest integer (ties to even — DuckDB's cast rounding). Out-of-`i64`-range → `None`
/// (DuckDB errors; we stay lenient). `2^63` is the exclusive upper bound so the `as i64`
/// never saturates on a value the check let through.
fn number_to_i64(n: &serde_json::Number) -> Option<i64> {
    if let Some(i) = n.as_i64() {
        return Some(i);
    }
    let r = n.as_f64()?.round_ties_even();
    // -2^63 ..= just-below 2^63 (2^63 is exactly representable and out of i64 range).
    if (-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&r) {
        Some(r as i64)
    } else {
        None
    }
}

/// Extract the value at `path` as an `f64`, matching DuckDB's `json_extract(...)::DOUBLE`:
/// any JSON number (widened, `inf` on overflow like DuckDB), and a JSON bool as `1.0`/`0.0`.
/// A container/string/JSON-null leaf, a missing path, or malformed JSON yields `None`.
pub(super) fn extract_float(text: &str, path: &[PathPart]) -> Option<f64> {
    match leaf(text, path)? {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// Extract the value at `path` as a `bool`, matching DuckDB's `json_extract(...)::BOOLEAN`:
/// a JSON bool verbatim, and a JSON *number* as `n != 0` (so `1`/`2`/`1.0` → `true`,
/// `0`/`-0.0` → `false`). A container/string/JSON-null leaf, a missing path, or malformed
/// JSON yields `None`. Previously a numeric leaf (a `0`/`1` flag) returned `None`.
pub(super) fn extract_bool(text: &str, path: &[PathPart]) -> Option<bool> {
    match leaf(text, path)? {
        Value::Bool(b) => Some(b),
        Value::Number(n) => n.as_f64().map(|f| f != 0.0),
        _ => None,
    }
}

/// Whether a value exists at `path` (a JSON `null` counts as present — the path is
/// there, the value is null, which is the distinction `extract_*` cannot express).
pub(super) fn path_exists(text: &str, path: &[PathPart]) -> bool {
    seek(text, path).is_some_and(|raw| serde_json::from_str::<Value>(raw).is_ok())
}

/// The JSON type name at `path`: `object`, `array`, `string`, `number`, `boolean`, or
/// `null`. `None` on an absent path or a malformed leaf.
///
/// Reads only the first non-whitespace byte of the located slice for the container and
/// scalar-shape cases, so typing a huge sub-object costs no parse; only the `t`/`f`/`n`
/// literals need confirming, which [`serde_json`] does on a token.
pub(super) fn value_type(text: &str, path: &[PathPart]) -> Option<&'static str> {
    let raw = seek(text, path)?.trim();
    Some(match raw.as_bytes().first()? {
        b'{' => "object",
        b'[' => "array",
        b'"' => "string",
        b't' | b'f' => {
            if raw == "true" || raw == "false" {
                "boolean"
            } else {
                return None;
            }
        }
        b'n' => {
            if raw == "null" {
                "null"
            } else {
                return None;
            }
        }
        _ => {
            // Confirm it really is a number rather than a malformed token.
            serde_json::from_str::<serde_json::Number>(raw).ok()?;
            "number"
        }
    })
}

/// The number of elements in the array at `path`. `None` if the path is absent or the
/// value there is not an array.
///
/// Counts by structural skipping rather than materializing the elements, so the cost is
/// one pass over the array's bytes with no allocation per element.
pub(super) fn array_length(text: &str, path: &[PathPart]) -> Option<i64> {
    let raw = seek(text, path)?;
    let bytes = raw.as_bytes();
    if bytes.first()? != &b'[' {
        return None;
    }
    let mut i = skip_ws(bytes, 1);
    if bytes.get(i)? == &b']' {
        return Some(0);
    }
    let mut n = 0i64;
    loop {
        i = skip_value(bytes, i)?;
        i = skip_ws(bytes, i);
        n += 1;
        match bytes.get(i)? {
            b',' => i = skip_ws(bytes, i + 1),
            b']' => return Some(n),
            _ => return None,
        }
    }
}

/// The keys of the object at `path`, **in source order**. `None` if the path is absent
/// or the value there is not an object.
///
/// Source order, not sorted order, because `serde_json`'s default map re-sorts keys and
/// a caller zipping `keys` against [`array_values`]-style extraction would then get a
/// different pairing than the document states. The scan skips each value structurally,
/// so only the keys are decoded.
pub(super) fn object_keys(text: &str, path: &[PathPart]) -> Option<Vec<String>> {
    let raw = seek(text, path)?;
    let bytes = raw.as_bytes();
    if bytes.first()? != &b'{' {
        return None;
    }
    let mut i = skip_ws(bytes, 1);
    if bytes.get(i)? == &b'}' {
        return Some(Vec::new());
    }
    let mut keys = Vec::new();
    loop {
        let (k, after) = parse_string(bytes, i)?;
        keys.push(k);
        i = skip_ws(bytes, after);
        if bytes.get(i)? != &b':' {
            return None;
        }
        i = skip_value(bytes, skip_ws(bytes, i + 1))?;
        i = skip_ws(bytes, i);
        match bytes.get(i)? {
            b',' => i = skip_ws(bytes, i + 1),
            b'}' => return Some(keys),
            _ => return None,
        }
    }
}

/// The elements of the array at `path`, each rendered the way [`extract_string`] renders
/// a leaf: a string element verbatim, a container compacted, a JSON `null` as `None`.
/// `None` (the outer option) if the path is absent or the value is not an array.
///
/// This is what turns a JSON array column into a Batcher list column, so `explode` and
/// the whole `.list` namespace apply to it.
pub(super) fn array_values(text: &str, path: &[PathPart]) -> Option<Vec<Option<String>>> {
    let raw = seek(text, path)?;
    let bytes = raw.as_bytes();
    if bytes.first()? != &b'[' {
        return None;
    }
    let mut i = skip_ws(bytes, 1);
    if bytes.get(i)? == &b']' {
        return Some(Vec::new());
    }
    let mut out = Vec::new();
    loop {
        let end = skip_value(bytes, i)?;
        out.push(render_leaf(&raw[i..end]));
        i = skip_ws(bytes, end);
        match bytes.get(i)? {
            b',' => i = skip_ws(bytes, i + 1),
            b']' => return Some(out),
            _ => return None,
        }
    }
}

/// Render one already-located JSON value as text: a string leaf verbatim, a container as
/// its compact JSON, a JSON `null` as `None`, everything else as its canonical form.
///
/// The single renderer behind `extract_string` and [`array_values`], so an element pulled
/// out by `[i]` and the same element seen through `array_values` cannot read differently.
fn render_leaf(raw: &str) -> Option<String> {
    match serde_json::from_str::<Value>(raw).ok()? {
        Value::String(s) => Some(s),
        Value::Null => None,
        // Compact the original slice so object keys keep their source order.
        Value::Object(_) | Value::Array(_) => Some(compact(raw)),
        Value::Number(n) => {
            // An integer literal larger than u64 (e.g. a 20+ digit id) is parsed by
            // serde_json as f64, whose Display renders it in lossy scientific form
            // (`1e+20`). DuckDB keeps the exact digits, so return the source token for
            // that case; everything representable (i64/u64) or fractional uses serde's
            // canonical form (which also matches DuckDB: `1.50` -> `1.5`).
            if n.as_i64().is_none() && n.as_u64().is_none() && is_integer_literal(raw) {
                // `-0` is an integer literal serde parses as f64 (as_i64/as_u64 both
                // None); numerically it is zero, and DuckDB canonicalizes it to "0"
                // rather than keeping the raw "-0". A genuine out-of-i64/u64-range
                // integer (e.g. `1e20` worth of digits) never has an all-zero magnitude,
                // so it still keeps its exact digits.
                let magnitude = raw.strip_prefix('-').unwrap_or(raw);
                if magnitude.bytes().all(|b| b == b'0') {
                    Some("0".to_string())
                } else {
                    Some(raw.to_string())
                }
            } else {
                Some(n.to_string())
            }
        }
        // Bool has no ordering or precision to preserve.
        other => Some(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parts(p: &str) -> Vec<PathPart> {
        parse_path(p)
    }

    #[test]
    fn array_length_counts_without_parsing_elements() {
        assert_eq!(array_length(r#"{"a": [1, 2, 3]}"#, &parts("$.a")), Some(3));
        assert_eq!(array_length(r#"{"a": []}"#, &parts("$.a")), Some(0));
        // Nested containers and strings holding brackets do not confuse the scan.
        assert_eq!(
            array_length(r#"[{"x": [1,2]}, "a],b", [3]]"#, &parts("$")),
            Some(3)
        );
        assert_eq!(array_length(r#"{"a": 1}"#, &parts("$.a")), None);
        assert_eq!(array_length(r#"{"a": [1]}"#, &parts("$.missing")), None);
    }

    #[test]
    fn object_keys_are_in_source_order() {
        assert_eq!(
            object_keys(r#"{"z": 1, "a": 2, "m": 3}"#, &parts("$")),
            Some(vec!["z".to_string(), "a".to_string(), "m".to_string()])
        );
        assert_eq!(object_keys("{}", &parts("$")), Some(vec![]));
        assert_eq!(object_keys("[1]", &parts("$")), None);
    }

    #[test]
    fn array_values_render_like_extract_string() {
        let doc = r#"{"a": ["x", 1, {"b": 2}, null, true]}"#;
        assert_eq!(
            array_values(doc, &parts("$.a")),
            Some(vec![
                Some("x".to_string()),
                Some("1".to_string()),
                Some(r#"{"b":2}"#.to_string()),
                None,
                Some("true".to_string()),
            ])
        );
        // Element i seen through the array must equal element i seen through `[i]`.
        for i in 0..5 {
            let via_index = extract_string(doc, &parts(&format!("$.a[{i}]")));
            let via_values = array_values(doc, &parts("$.a")).unwrap()[i].clone();
            assert_eq!(via_index, via_values, "element {i} disagreed");
        }
    }

    #[test]
    fn value_type_names_every_json_shape() {
        let doc = r#"{"o": {}, "a": [], "s": "x", "n": 1.5, "b": false, "z": null}"#;
        for (path, want) in [
            ("$.o", "object"),
            ("$.a", "array"),
            ("$.s", "string"),
            ("$.n", "number"),
            ("$.b", "boolean"),
            ("$.z", "null"),
        ] {
            assert_eq!(value_type(doc, &parts(path)), Some(want), "{path}");
        }
        assert_eq!(value_type(doc, &parts("$.missing")), None);
    }

    #[test]
    fn path_exists_distinguishes_absent_from_json_null() {
        let doc = r#"{"z": null}"#;
        assert!(path_exists(doc, &parts("$.z")));
        assert!(!path_exists(doc, &parts("$.other")));
        // `extract_string` cannot tell these apart — that is why `path_exists` exists.
        assert_eq!(extract_string(doc, &parts("$.z")), None);
        assert_eq!(extract_string(doc, &parts("$.other")), None);
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
        // Object leaves keep their SOURCE key order (matching DuckDB's
        // json_extract_string), not serde_json's alphabetized Map order.
        let doc = r#"{"user":{"id":7,"country":"US"}}"#;
        assert_eq!(
            extract_string(doc, &parts("$.user")),
            Some(r#"{"id":7,"country":"US"}"#.into())
        );
        assert_eq!(
            extract_string(r#"{"a":[1,2,3]}"#, &parts("$.a")),
            Some("[1,2,3]".into())
        );
    }

    #[test]
    fn a_huge_integer_keeps_its_digits_rather_than_scientific_notation() {
        // An integer beyond u64 is parsed as f64 by serde_json; its Display is lossy
        // scientific (`1e20`). DuckDB preserves the exact digits — so must we.
        let doc = r#"{"id":100000000000000000000}"#;
        assert_eq!(
            extract_string(doc, &parts("$.id")),
            Some("100000000000000000000".into())
        );
        // Values that fit and true floats keep their canonical serde form.
        assert_eq!(
            extract_string(r#"{"x":30}"#, &parts("$.x")),
            Some("30".into())
        );
        assert_eq!(
            extract_string(r#"{"x":1.50}"#, &parts("$.x")),
            Some("1.5".into())
        );
        // `-0` (an integer literal serde parses as f64) is numerically zero; DuckDB
        // renders it "0", not the raw "-0". Only genuine huge integers keep raw digits.
        assert_eq!(
            extract_string(r#"{"x":-0}"#, &parts("$.x")),
            Some("0".into())
        );
        // A negative-zero *float* keeps its sign (DuckDB: "-0.0").
        assert_eq!(
            extract_string(r#"{"x":-0.0}"#, &parts("$.x")),
            Some("-0.0".into())
        );
    }

    #[test]
    fn object_leaves_preserve_source_key_order_and_compact_whitespace() {
        // Regression: serde_json's default Map sorts keys, so extracting a sub-object
        // reordered its keys (b,a,c -> a,b,c) and disagreed with DuckDB, which keeps
        // insertion order. Whitespace between tokens is still stripped (compact).
        let doc = r#"{"obj":{ "b" : 1,  "a":2, "c":3 }}"#;
        assert_eq!(
            extract_string(doc, &parts("$.obj")),
            Some(r#"{"b":1,"a":2,"c":3}"#.into())
        );
        // A string value containing significant whitespace/braces is copied verbatim.
        let doc2 = r#"{"o":{"msg":"a  b { x }","z":9,"a":1}}"#;
        assert_eq!(
            extract_string(doc2, &parts("$.o")),
            Some(r#"{"msg":"a  b { x }","z":9,"a":1}"#.into())
        );
    }

    #[test]
    fn extract_int_coerces_floats_and_bools_like_duckdb() {
        // A JSON float extracted as int rounds to nearest, ties to even — exactly what
        // DuckDB's `json_extract(...)::BIGINT` does (verified: 0.5->0, 2.5->2, 3.5->4,
        // -2.5->-2). Previously every one of these returned None (silent data loss).
        assert_eq!(extract_int(r#"{"x":42.0}"#, &parts("$.x")), Some(42));
        assert_eq!(extract_int(r#"{"x":1e2}"#, &parts("$.x")), Some(100));
        assert_eq!(extract_int(r#"{"x":3.5}"#, &parts("$.x")), Some(4));
        assert_eq!(extract_int(r#"{"x":2.5}"#, &parts("$.x")), Some(2));
        assert_eq!(extract_int(r#"{"x":0.5}"#, &parts("$.x")), Some(0));
        assert_eq!(extract_int(r#"{"x":-2.5}"#, &parts("$.x")), Some(-2));
        assert_eq!(extract_int(r#"{"x":2.6}"#, &parts("$.x")), Some(3));
        // A JSON bool -> 1/0 (DuckDB coerces it too).
        assert_eq!(extract_int(r#"{"x":true}"#, &parts("$.x")), Some(1));
        assert_eq!(extract_int(r#"{"x":false}"#, &parts("$.x")), Some(0));
        // Out of i64 range stays None (DuckDB errors; we are lenient), not a wrapped value.
        assert_eq!(extract_int(r#"{"x":1e30}"#, &parts("$.x")), None);
        // A plain integer that fits is still exact (no float round-trip).
        assert_eq!(
            extract_int(r#"{"x":9223372036854775807}"#, &parts("$.x")),
            Some(i64::MAX)
        );
    }

    #[test]
    fn extract_bool_coerces_numbers_like_duckdb() {
        // A numeric flag extracted as bool: nonzero -> true, zero (incl -0.0) -> false,
        // matching DuckDB's `json_extract(...)::BOOLEAN`. Previously all returned None.
        assert_eq!(extract_bool(r#"{"x":1}"#, &parts("$.x")), Some(true));
        assert_eq!(extract_bool(r#"{"x":2}"#, &parts("$.x")), Some(true));
        assert_eq!(extract_bool(r#"{"x":1.0}"#, &parts("$.x")), Some(true));
        assert_eq!(extract_bool(r#"{"x":0}"#, &parts("$.x")), Some(false));
        assert_eq!(extract_bool(r#"{"x":-0.0}"#, &parts("$.x")), Some(false));
        // A genuine JSON bool is unchanged.
        assert_eq!(extract_bool(r#"{"x":true}"#, &parts("$.x")), Some(true));
        // A container/null leaf is still None.
        assert_eq!(extract_bool(r#"{"x":[1]}"#, &parts("$.x")), None);
        assert_eq!(extract_bool(r#"{"x":null}"#, &parts("$.x")), None);
    }

    #[test]
    fn extract_float_coerces_bools_like_duckdb() {
        assert_eq!(extract_float(r#"{"x":true}"#, &parts("$.x")), Some(1.0));
        assert_eq!(extract_float(r#"{"x":false}"#, &parts("$.x")), Some(0.0));
        assert_eq!(extract_float(r#"{"x":42}"#, &parts("$.x")), Some(42.0));
        assert_eq!(extract_float(r#"{"x":3.5}"#, &parts("$.x")), Some(3.5));
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
    fn negative_array_index_counts_from_the_end() {
        // DuckDB's JSON path folds a negative subscript from the back: `[-1]` is the
        // last element, `[-2]` the second-to-last; out of range -> None. Previously the
        // `-1` failed to parse as a `usize` and the subscript was silently DROPPED, so
        // `$.arr[-1]` returned the whole parent array instead of its last element.
        let doc = r#"{"arr":[10,20,30]}"#;
        assert_eq!(extract_int(doc, &parts("$.arr[-1]")), Some(30));
        assert_eq!(extract_int(doc, &parts("$.arr[-2]")), Some(20));
        assert_eq!(extract_int(doc, &parts("$.arr[-3]")), Some(10));
        assert_eq!(extract_int(doc, &parts("$.arr[-4]")), None); // past the front
                                                                 // A negative index on a root array works too, and the fast forward path is intact.
        assert_eq!(extract_int("[1,2,3]", &parts("$[-1]")), Some(3));
        assert_eq!(extract_int("[1,2,3]", &parts("$[0]")), Some(1));
        // Negative into a nested object element.
        assert_eq!(
            extract_int(r#"{"a":[{"z":1},{"z":2}]}"#, &parts("$.a[-1].z")),
            Some(2)
        );
        // Chained negative subscripts on an array-of-arrays, each folded independently.
        let m = r#"{"a":[[1,2],[3,4]]}"#;
        assert_eq!(extract_int(m, &parts("$.a[-1][-1]")), Some(4));
        assert_eq!(extract_int(m, &parts("$.a[1][-2]")), Some(3));
        assert_eq!(extract_string(m, &parts("$.a[-1]")), Some("[3,4]".into()));
        // A negative index into an empty array is out of range -> None (not the array).
        assert_eq!(extract_string(r#"{"a":[]}"#, &parts("$.a[-1]")), None);
        // The parser now keeps the negative subscript as a real step.
        assert_eq!(
            parts("$.arr[-1]"),
            vec![PathPart::Key("arr".into()), PathPart::Index(-1)]
        );
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
                        PathPart::Index(i) => cur.get(*i as usize).cloned().unwrap_or(Value::Null),
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
