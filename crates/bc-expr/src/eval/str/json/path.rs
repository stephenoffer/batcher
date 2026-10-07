//! The JSONPath subset the `.json` kernels navigate, and the parser that refuses the rest.
//!
//! The subset is RFC 9535's *singular query*: a `$`-rooted chain of member names and array
//! indices, each selecting at most one value. Concretely:
//!
//! | Form | Meaning |
//! |---|---|
//! | `$` | the whole document (the `$` itself is optional) |
//! | `.name` | the member `name` (any characters but `.`, `[` and `]`) |
//! | `."x.y"`, `['x.y']`, `["x.y"]` | a quoted member name, the only way to reach a key holding a `.` |
//! | `[n]`, `[-n]` | the `n`th element from the front, or from the back (`[-1]` is the last) |
//!
//! Everything else RFC 9535 defines selects *several* values — wildcards (`[*]`, `.*`),
//! recursive descent (`..`), slices (`[0:2]`), unions (`[0,1]`) and filters (`[?...]`) — and
//! is refused with a reason. It used to be skipped instead: an unrecognised subscript was
//! dropped, so `$.a[0:1]` read the *whole* array and `$.a[*]` returned the array as one
//! string, which is a plausible wrong answer rather than an error. The control plane runs
//! the same grammar at plan time (`plan/expr_ir/namespaces/_json_path.py`); this parser is
//! the defence for a path that only exists per row (`StrFuncDyn`), and for a plan built
//! without the Python front-end.

use crate::error::ExprError;

/// One step of a JSON path: an object key or an array index.
///
/// The index is signed: a non-negative index counts from the front (`[0]` is the
/// first element), a negative index counts from the back (`[-1]` is the last),
/// matching DuckDB's JSON path semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::eval::str) enum PathPart {
    Key(String),
    Index(i64),
}

/// Parse a `$.a."b.c"[0]` path into its steps, or refuse it with the reason.
///
/// A path with no leading `.` after the optional `$` starts with a bare key, so `a.b`,
/// `$.a.b` and `$a.b` all name the same value; that leniency predates this parser and
/// the SQL front-end relies on it.
pub(in crate::eval::str) fn parse_path(path: &str) -> Result<Vec<PathPart>, ExprError> {
    let err = |reason: &str| ExprError::InvalidArgument {
        func: "json path".to_string(),
        reason: format!("unsupported JSONPath `{path}`: {reason}"),
    };
    let body = path.strip_prefix('$').unwrap_or(path);
    let bytes = body.as_bytes();
    let mut parts = Vec::new();
    let mut i = 0;
    if !bytes.is_empty() && bytes[0] != b'.' && bytes[0] != b'[' {
        let (name, next) = read_name(body, 0);
        parts.push(member(name).map_err(err)?);
        i = next;
    }
    while i < bytes.len() {
        match bytes[i] {
            b'.' => {
                i += 1;
                match bytes.get(i) {
                    None => return Err(err("a path cannot end with `.`")),
                    Some(b'.') => return Err(err(RECURSIVE)),
                    Some(b'"') | Some(b'\'') => {
                        let (key, next) = read_quoted(body, i).map_err(err)?;
                        parts.push(PathPart::Key(key));
                        i = next;
                    }
                    Some(b'[') => return Err(err("`.` must be followed by a member name")),
                    Some(_) => {
                        let (name, next) = read_name(body, i);
                        parts.push(member(name).map_err(err)?);
                        i = next;
                    }
                }
            }
            b'[' => {
                let (part, next) = read_bracket(body, i).map_err(err)?;
                parts.push(part);
                i = next;
            }
            _ => return Err(err("expected `.` or `[` between two steps")),
        }
    }
    Ok(parts)
}

const RECURSIVE: &str = "recursive descent (`..`) is not supported";
const WILDCARD: &str = "a wildcard (`*`) is not supported; read an array's elements with \
                        `.json.values(path)` instead";

/// An unquoted member name, refusing the one spelling that means something else.
fn member(name: &str) -> Result<PathPart, &'static str> {
    match name {
        "*" => Err(WILDCARD),
        _ if name.contains(']') => Err("a `]` with no matching `[`"),
        _ => Ok(PathPart::Key(name.to_string())),
    }
}

/// The unquoted name starting at `start`: everything up to the next `.` or `[`.
fn read_name(body: &str, start: usize) -> (&str, usize) {
    let end = body[start..]
        .find(['.', '['])
        .map_or(body.len(), |off| start + off);
    (&body[start..end], end)
}

/// A quoted name starting at `start` (on the quote), and the index just past its close.
///
/// A backslash takes the next character literally, which is enough to write either quote
/// inside a name of the same quote; there are no other escapes.
fn read_quoted(body: &str, start: usize) -> Result<(String, usize), &'static str> {
    let quote = body.as_bytes()[start] as char;
    let mut out = String::new();
    let mut chars = body[start + 1..].char_indices();
    while let Some((off, c)) = chars.next() {
        match c {
            '\\' => match chars.next() {
                Some((_, escaped)) => out.push(escaped),
                None => break,
            },
            c if c == quote => return Ok((out, start + 1 + off + 1)),
            c => out.push(c),
        }
    }
    Err("an unterminated quoted member name")
}

/// A `[...]` selector starting at `start` (on the `[`), and the index just past its `]`.
fn read_bracket(body: &str, start: usize) -> Result<(PathPart, usize), &'static str> {
    let bytes = body.as_bytes();
    let mut i = start + 1;
    while bytes.get(i).is_some_and(|b| *b == b' ') {
        i += 1;
    }
    if matches!(bytes.get(i), Some(b'"') | Some(b'\'')) {
        let (key, mut next) = read_quoted(body, i)?;
        while bytes.get(next).is_some_and(|b| *b == b' ') {
            next += 1;
        }
        return match bytes.get(next) {
            Some(b']') => Ok((PathPart::Key(key), next + 1)),
            _ => Err("a quoted name in `[...]` must be followed by `]`"),
        };
    }
    let close = body[i..].find(']').ok_or("a `[` with no closing `]`")? + i;
    let inner = body[i..close].trim();
    Ok((PathPart::Index(index(inner)?), close + 1))
}

/// Classify the inside of an unquoted `[...]`: an integer index, or the reason it is not one.
fn index(inner: &str) -> Result<i64, &'static str> {
    let digits = inner.strip_prefix(['-', '+']).unwrap_or(inner);
    if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
        return inner
            .parse::<i64>()
            .map_err(|_| "an array index out of range");
    }
    Err(match inner {
        "*" => WILDCARD,
        _ if inner.starts_with('?') => "a filter selector (`[?...]`) is not supported",
        _ if inner.starts_with('#') => {
            "`[#-n]` is DuckDB's last-element syntax; write `[-n]` (`[-1]` is the last element)"
        }
        _ if inner.contains(':') => "an array slice (`[start:end]`) is not supported",
        _ if inner.contains(',') => "a union selector (`[a,b]`) is not supported",
        _ => "an array subscript must be an integer or a quoted member name",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(k: &str) -> PathPart {
        PathPart::Key(k.into())
    }

    #[test]
    fn the_supported_subset_parses() {
        let ok: &[(&str, Vec<PathPart>)] = &[
            ("$", vec![]),
            ("", vec![]),
            ("$.a.b", vec![key("a"), key("b")]),
            ("a.b", vec![key("a"), key("b")]),
            ("$.tags[0]", vec![key("tags"), PathPart::Index(0)]),
            ("$.a[2].b", vec![key("a"), PathPart::Index(2), key("b")]),
            ("$.a[-1]", vec![key("a"), PathPart::Index(-1)]),
            ("$[0][1]", vec![PathPart::Index(0), PathPart::Index(1)]),
            ("$.a[ 1 ]", vec![key("a"), PathPart::Index(1)]),
            ("$.\"x.y\"", vec![key("x.y")]),
            ("$[\"x.y\"].z", vec![key("x.y"), key("z")]),
            ("$['x.y']", vec![key("x.y")]),
            ("$['it\\'s']", vec![key("it's")]),
            ("$.my-key", vec![key("my-key")]),
        ];
        for (path, want) in ok {
            assert_eq!(&parse_path(path).unwrap(), want, "{path}");
        }
    }

    #[test]
    fn every_multi_value_selector_is_refused_with_its_reason() {
        let refused: &[(&str, &str)] = &[
            ("$.a[*]", "wildcard"),
            ("$.a.*", "wildcard"),
            ("$..b", "recursive descent"),
            ("$.a[0:1]", "slice"),
            ("$.a[0,1]", "union"),
            ("$.a[?(@>1)]", "filter"),
            ("$.a[#-1]", "[-n]"),
            ("$.a[x]", "integer or a quoted"),
            ("$.a[0", "closing"),
            ("$.\"x", "unterminated"),
            ("$.a.", "end with"),
            ("$.\"x\"y", "expected `.` or `[`"),
            ("$.a[0]x", "expected `.` or `[`"),
        ];
        for (path, reason) in refused {
            let msg = parse_path(path).unwrap_err().to_string();
            assert!(msg.contains(reason), "{path}: {msg}");
            assert!(msg.contains(path), "the message names the path: {msg}");
        }
    }
}
