//! `Expr::StructUpdate`: add, replace, rename and drop a struct's fields without rebuilding it.
//!
//! The obvious implementation — read every field out and pack a new struct with
//! `MakeStruct` — builds a struct that is never null, so a null input row would come back
//! as a struct of nulls. Spark's `withField` and Polars' `with_fields` keep a null struct
//! null, and so does this; DuckDB's `struct_update(NULL, x := 1)` is `{x: 1, y: NULL}`, a
//! divergence the differential tests pin. Editing the field list and keeping the input's own
//! null mask is also what preserves an untouched field's nullability and metadata exactly,
//! since that `Field` is carried over as it was.
//!
//! A new or replaced field's values are taken as computed. Under a null struct row they are
//! still present in the child array but invisible, which is how every Arrow struct child
//! behaves.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, StructArray};
use arrow::datatypes::{DataType, Field, FieldRef, Fields};

use crate::ExprError;

/// Apply `drop`, then `rename`, then `set` to the struct column `input`.
pub(crate) fn eval_struct_update(
    input: &ArrayRef,
    set: &[(&str, ArrayRef)],
    drop: &[String],
    rename: &[(String, String)],
) -> Result<ArrayRef, ExprError> {
    let DataType::Struct(_) = input.data_type() else {
        return Err(ExprError::ExpectedType {
            func: "struct update".into(),
            want: "a Struct argument",
            got: crate::error::type_name(input.data_type()),
        });
    };
    let s = input.as_struct();
    let (fields, columns, nulls) = s.clone().into_parts();
    let mut kept: Vec<(FieldRef, ArrayRef)> = fields.iter().cloned().zip(columns).collect();
    let unknown = |name: &str, kept: &[(FieldRef, ArrayRef)]| ExprError::UnknownField {
        field: name.to_string(),
        available: kept
            .iter()
            .map(|(f, _)| f.name().as_str())
            .collect::<Vec<_>>()
            .join(", "),
    };
    for name in drop {
        let at = kept
            .iter()
            .position(|(f, _)| f.name() == name)
            .ok_or_else(|| unknown(name, &kept))?;
        kept.remove(at);
    }
    for (old, new) in rename {
        let at = kept
            .iter()
            .position(|(f, _)| f.name() == old)
            .ok_or_else(|| unknown(old, &kept))?;
        let renamed = kept[at].0.as_ref().clone().with_name(new);
        kept[at].0 = Arc::new(renamed);
    }
    for (name, values) in set {
        let values = if values.len() == s.len() {
            Arc::clone(values)
        } else {
            return Err(ExprError::InvalidArgument {
                func: "struct.with_fields".into(),
                reason: format!(
                    "field `{name}` has {} rows, expected {}",
                    values.len(),
                    s.len()
                ),
            });
        };
        let field = Arc::new(Field::new(*name, values.data_type().clone(), true));
        match kept.iter().position(|(f, _)| f.name() == name) {
            Some(at) => kept[at] = (field, values),
            None => kept.push((field, values)),
        }
    }
    if kept.is_empty() {
        return Err(ExprError::InvalidArgument {
            func: "struct update".into(),
            reason: "a struct must keep at least one field".into(),
        });
    }
    let (fields, columns): (Vec<FieldRef>, Vec<ArrayRef>) = kept.into_iter().unzip();
    Ok(Arc::new(StructArray::try_new(
        Fields::from(fields),
        columns,
        nulls,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, StringArray};
    use arrow::buffer::NullBuffer;
    use std::collections::HashMap;

    fn input() -> ArrayRef {
        let tagged = Field::new("a", DataType::Int64, false)
            .with_metadata(HashMap::from([("unit".to_string(), "m".to_string())]));
        let s = StructArray::try_new(
            Fields::from(vec![tagged, Field::new("b", DataType::Utf8, true)]),
            vec![
                Arc::new(Int64Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec![Some("x"), None])),
            ],
            Some(NullBuffer::from(vec![true, false])),
        )
        .unwrap();
        Arc::new(s)
    }

    #[test]
    fn a_null_struct_stays_null_and_untouched_fields_keep_their_metadata() {
        let c: ArrayRef = Arc::new(StringArray::from(vec!["new", "new"]));
        let out = eval_struct_update(&input(), &[("c", c)], &[], &[]).unwrap();
        let s = out.as_struct();
        assert!(s.is_valid(0) && s.is_null(1));
        let DataType::Struct(fields) = out.data_type() else {
            unreachable!()
        };
        assert_eq!(
            fields[0].metadata().get("unit").map(String::as_str),
            Some("m")
        );
        assert!(!fields[0].is_nullable());
        assert_eq!(
            fields.iter().map(|f| f.name().as_str()).collect::<Vec<_>>(),
            ["a", "b", "c"]
        );
    }

    #[test]
    fn drop_rename_and_replace_compose_in_order() {
        let a: ArrayRef = Arc::new(StringArray::from(vec!["p", "q"]));
        let rename = [("b".to_string(), "bee".to_string())];
        let out = eval_struct_update(&input(), &[("a", a)], &[], &rename).unwrap();
        let DataType::Struct(fields) = out.data_type() else {
            unreachable!()
        };
        assert_eq!(fields[0].data_type(), &DataType::Utf8, "replaced in place");
        assert_eq!(fields[1].name(), "bee");

        let out = eval_struct_update(&input(), &[], &["a".to_string()], &[]).unwrap();
        assert_eq!(out.as_struct().num_columns(), 1);
        let err = eval_struct_update(&input(), &[], &["zz".to_string()], &[]).unwrap_err();
        assert!(err.to_string().contains("zz"));
        let all = ["a".to_string(), "b".to_string()];
        assert!(eval_struct_update(&input(), &[], &all, &[]).is_err());
    }
}
