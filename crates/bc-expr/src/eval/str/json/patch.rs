//! `json.merge_patch(patch)`: RFC 7386 JSON Merge Patch, DuckDB's `json_merge_patch`.
//!
//! A patch that is not an object replaces the document outright. An object patch is applied
//! member by member: a `null` member deletes that key, an object member merges into the
//! key's value recursively (a non-object value there is first replaced by `{}`), and
//! anything else replaces the key's value. Arrays are values, never merged.
//!
//! Key order follows DuckDB, which matters because the result is text: a key the patch
//! touches is removed and re-inserted at the end, so `{"b":1,"a":2}` patched with
//! `{"c":3,"a":4}` is `{"b":1,"c":3,"a":4}`. A SQL-null document is patched as if it were
//! JSON `null`; a SQL-null patch is null. Text that does not parse, on either side, is null
//! where DuckDB raises, which is the standing divergence of every reader in this module.

use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray, StringArray};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use serde_json::{Map, Value};

use crate::ExprError;

/// Apply each row's `patch` to the same row's `doc`.
pub(in crate::eval) fn merge_patch(
    doc: &ArrayRef,
    patch: &ArrayRef,
) -> Result<ArrayRef, ExprError> {
    let (doc, patch) = (cast(doc, &DataType::Utf8)?, cast(patch, &DataType::Utf8)?);
    let (doc, patch) = (doc.as_string::<i32>(), patch.as_string::<i32>());
    let out: StringArray = doc
        .iter()
        .zip(patch.iter())
        .map(|(d, p)| {
            let patch: Value = serde_json::from_str(p?).ok()?;
            let target = match d {
                None => Value::Null,
                Some(t) => serde_json::from_str(t).ok()?,
            };
            Some(apply(target, patch).to_string())
        })
        .collect();
    Ok(Arc::new(out))
}

/// RFC 7386 `MergePatch(Target, Patch)`.
fn apply(target: Value, patch: Value) -> Value {
    let Value::Object(members) = patch else {
        return patch;
    };
    let mut target = match target {
        Value::Object(m) => m,
        _ => Map::new(),
    };
    for (key, value) in members {
        let previous = target.shift_remove(&key);
        if !value.is_null() {
            let merged = apply(previous.unwrap_or(Value::Null), value);
            target.insert(key, merged);
        }
    }
    Value::Object(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(doc: Option<&str>, patch: Option<&str>) -> Option<String> {
        let d: ArrayRef = Arc::new(StringArray::from(vec![doc]));
        let p: ArrayRef = Arc::new(StringArray::from(vec![patch]));
        let out = merge_patch(&d, &p).unwrap();
        let out = out.as_string::<i32>();
        (!arrow::array::Array::is_null(out, 0)).then(|| out.value(0).to_string())
    }

    #[test]
    fn the_rfc_rules_and_duckdbs_key_order() {
        let s = |v: &str| Some(v.to_string());
        let cases = [
            (
                Some(r#"{"a":1,"b":{"c":1,"d":2}}"#),
                Some(r#"{"b":{"c":null,"e":3},"a":[1]}"#),
                s(r#"{"b":{"d":2,"e":3},"a":[1]}"#),
            ),
            (Some(r#"{"a":1}"#), Some("5"), s("5")),
            (Some("[1,2]"), Some(r#"{"a":1}"#), s(r#"{"a":1}"#)),
            (None, Some(r#"{"a":1}"#), s(r#"{"a":1}"#)),
            (Some(r#"{"a":1}"#), None, None),
            (Some(r#"{"a":1}"#), Some("null"), s("null")),
            (
                Some(r#"{"a":{"x":1}}"#),
                Some(r#"{"a":{"x":{"y":null}}}"#),
                s(r#"{"a":{"x":{}}}"#),
            ),
            (
                Some(r#"{"b":1,"a":2}"#),
                Some(r#"{"c":3,"a":4}"#),
                s(r#"{"b":1,"c":3,"a":4}"#),
            ),
            (Some("{}"), Some(r#"{"a":null}"#), s("{}")),
            (Some("nope"), Some("{}"), None),
        ];
        for (doc, patch, want) in cases {
            assert_eq!(run(doc, patch), want, "{doc:?} + {patch:?}");
        }
    }
}
