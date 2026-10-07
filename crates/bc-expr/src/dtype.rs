//! The wire form of a **nested** Arrow type, for expressions that name a target type.
//!
//! `Expr::Cast` names its target with a flat string (`"int64"`, `"decimal(12,4)"`), parsed by
//! [`bc_arrow::dtype_from_name`], and a flat string cannot say "a struct of a list of
//! strings". This is the extension that can, as plain JSON the control plane emits from
//! `plan/types/registry.py::dtype_to_wire`:
//!
//! | Wire | Arrow type |
//! |---|---|
//! | `"int64"` (any flat cast name) | that type, via `dtype_from_name` |
//! | `["list", T]` | `List<T>` |
//! | `["struct", [["a", T], ["b", U]]]` | `Struct<a: T, b: U>`, fields in that order |
//! | `["map", K, V]` | `Map<K, V>` |
//!
//! Arrays rather than objects keep the Python node fully declarative: a nested tuple is a
//! legal scalar field there, where a dict is not. Every child field is nullable, which is
//! what reading semi-structured data needs: any key can be absent from any document.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Fields};
use serde_json::Value;

use crate::ExprError;

/// Resolve a wire type, or report the part of it that does not parse.
pub(crate) fn dtype_from_wire(wire: &Value) -> Result<DataType, ExprError> {
    let bad = |what: &str| ExprError::UnknownType(format!("{what} in nested type {wire}"));
    match wire {
        Value::String(name) => {
            bc_arrow::dtype_from_name(name).ok_or_else(|| bad(&format!("`{name}`")))
        }
        Value::Array(parts) => match (parts.first().and_then(Value::as_str), parts.len()) {
            (Some("list"), 2) => Ok(DataType::List(Arc::new(Field::new_list_field(
                dtype_from_wire(&parts[1])?,
                true,
            )))),
            (Some("struct"), 2) => {
                let fields = parts[1].as_array().ok_or_else(|| bad("a struct body"))?;
                let fields = fields
                    .iter()
                    .map(|f| match f.as_array().map(Vec::as_slice) {
                        Some([Value::String(name), ty]) => {
                            Ok(Field::new(name, dtype_from_wire(ty)?, true))
                        }
                        _ => Err(bad("a struct field")),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(DataType::Struct(Fields::from(fields)))
            }
            (Some("map"), 3) => {
                let entries = Field::new(
                    "entries",
                    DataType::Struct(Fields::from(vec![
                        Field::new("key", dtype_from_wire(&parts[1])?, false),
                        Field::new("value", dtype_from_wire(&parts[2])?, true),
                    ])),
                    false,
                );
                Ok(DataType::Map(Arc::new(entries), false))
            }
            _ => Err(bad("an unknown composite")),
        },
        _ => Err(bad("a value that is neither a name nor a composite")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_composite_resolves_and_nests() {
        let wire: Value = serde_json::from_str(
            r#"["struct", [["a", "int64"], ["b", ["list", "utf8"]], ["m", ["map", "utf8", "float64"]]]]"#,
        )
        .unwrap();
        let DataType::Struct(fields) = dtype_from_wire(&wire).unwrap() else {
            panic!("not a struct");
        };
        assert_eq!(fields[0].data_type(), &DataType::Int64);
        assert!(
            matches!(fields[1].data_type(), DataType::List(f) if f.data_type() == &DataType::Utf8)
        );
        assert!(matches!(fields[2].data_type(), DataType::Map(..)));
        assert!(fields.iter().all(|f| f.is_nullable()));
    }

    #[test]
    fn a_malformed_wire_type_names_itself() {
        for bad in [r#""nope""#, r#"["list"]"#, r#"["struct", [["a"]]]"#, "5"] {
            let wire: Value = serde_json::from_str(bad).unwrap();
            let err = dtype_from_wire(&wire).unwrap_err().to_string();
            assert!(err.contains("nested type"), "{bad}: {err}");
        }
    }
}
