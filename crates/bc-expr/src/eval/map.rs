//! Map-column evaluation for `Expr::Map` (`map_keys`/`map_values`/`element_at`).
//!
//! An Arrow `Map` is a `List<Struct<key, value>>`. `map_keys`/`map_values` re-wrap
//! the entries' key/value child under the map's own offsets to yield a `List`;
//! `element_at` scans each row's entries for a literal key and `take`s the matching
//! value (null if absent). The JIT does not compile `Map`, so this is the only path.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, ListArray, MapArray, StructArray, UInt32Array};
use arrow::compute::take;
use arrow::datatypes::Field;

use crate::eval::list::eq_against_literal;
use crate::{ExprError, Literal, MapFunc};

/// Evaluate a map function over a `Map` array.
pub(crate) fn eval_map(
    func: MapFunc,
    arr: &ArrayRef,
    key: Option<&Literal>,
) -> Result<ArrayRef, ExprError> {
    // A `Struct` is a keyed container too, and SQL spells the lookup the same way:
    // `s['a']` is a field access, and every dialect here writes it as a subscript. The
    // translator cannot tell the two apart — it has no schema — so the disambiguation
    // has to happen where the array's type is actually known, which is here. Without
    // this, `s['a']` reached the `MapArray` downcast and failed with "expected a Map
    // argument, got Struct" on a query that is valid in DuckDB, Spark and Polars alike.
    if let Some(fields) = arr.as_any().downcast_ref::<StructArray>() {
        return struct_field(func, fields, key);
    }
    let map = arr
        .as_any()
        .downcast_ref::<MapArray>()
        .ok_or_else(|| ExprError::ExpectedType {
            func: format!("{func:?}"),
            want: "a Map or Struct argument",
            got: arr.data_type().to_string(),
        })?;
    match func {
        MapFunc::MapKeys => Ok(list_of(
            map.offsets().clone(),
            map.keys().clone(),
            map.nulls(),
        )),
        MapFunc::MapValues => Ok(list_of(
            map.offsets().clone(),
            map.values().clone(),
            map.nulls(),
        )),
        MapFunc::ElementAt => element_at(map, key),
    }
}

/// `s['a']` over a `Struct`: the named child column, or the field names as a list.
///
/// A struct's keys are fixed by its *type*, not carried per row, which is the whole
/// difference from a map: the lookup is a name resolution done once rather than a scan
/// per row, and a missing name is a plan error rather than a null — asking a struct for
/// a field it does not have is a mistake, where asking a map for an absent key is an
/// ordinary result.
///
/// A null struct row still has to answer null. Arrow keeps a struct's null mask on the
/// parent and its children are free to hold arbitrary values under it, so the child is
/// returned with the parent's nulls merged in rather than as-is.
fn struct_field(
    func: MapFunc,
    fields: &StructArray,
    key: Option<&Literal>,
) -> Result<ArrayRef, ExprError> {
    match func {
        MapFunc::MapKeys => {
            let names: Vec<&str> = struct_fields(fields)
                .iter()
                .map(|f| f.name().as_str())
                .collect();
            Ok(constant_list(&names, fields.len(), fields.nulls()))
        }
        MapFunc::MapValues => Err(ExprError::ExpectedType {
            func: "map_values".into(),
            want: "a Map argument (a struct's values need not share one type)",
            got: fields.data_type().to_string(),
        }),
        MapFunc::ElementAt => {
            let name = match key {
                Some(Literal::Str(s)) => s.as_str(),
                Some(_) => {
                    return Err(ExprError::ExpectedType {
                        func: "element_at".into(),
                        want: "a string field name for a Struct argument",
                        got: "a non-string key".to_string(),
                    })
                }
                None => {
                    return Err(ExprError::MissingArgument {
                        func: "element_at".into(),
                        arg: "key",
                    })
                }
            };
            let child = fields
                .column_by_name(name)
                .ok_or_else(|| ExprError::ExpectedType {
                    func: "element_at".into(),
                    want: "a field this struct has",
                    got: format!("no field {name:?} in {}", fields.data_type()),
                })?;
            Ok(merge_parent_nulls(child, fields.nulls()))
        }
    }
}

/// The struct's fields, whatever concrete `DataType` spelling it carries.
fn struct_fields(fields: &StructArray) -> &arrow::datatypes::Fields {
    match fields.data_type() {
        arrow::datatypes::DataType::Struct(f) => f,
        _ => unreachable!("a StructArray's data type is always Struct"),
    }
}

/// The same list of names on every row — a struct's keys come from its type.
fn constant_list(
    names: &[&str],
    rows: usize,
    nulls: Option<&arrow::buffer::NullBuffer>,
) -> ArrayRef {
    let flat = arrow::array::StringArray::from(
        names
            .iter()
            .cycle()
            .take(names.len() * rows)
            .copied()
            .collect::<Vec<_>>(),
    );
    let offsets = arrow::buffer::OffsetBuffer::from_lengths(std::iter::repeat_n(names.len(), rows));
    list_of(offsets, Arc::new(flat), nulls)
}

/// A struct's child under the parent's null mask: a null struct row is null throughout,
/// even where the child buffer holds a value.
fn merge_parent_nulls(child: &ArrayRef, parent: Option<&arrow::buffer::NullBuffer>) -> ArrayRef {
    let Some(parent) = parent else {
        return Arc::clone(child);
    };
    let merged = arrow::buffer::NullBuffer::union(Some(parent), child.nulls());
    arrow::array::make_array(
        child
            .to_data()
            .into_builder()
            .nulls(merged)
            .build()
            .expect("re-nulling a child cannot change its layout"),
    )
}

/// Wrap a flat child array under the map's offsets/nulls as a `List` column.
fn list_of(
    offsets: arrow::buffer::OffsetBuffer<i32>,
    child: ArrayRef,
    nulls: Option<&arrow::buffer::NullBuffer>,
) -> ArrayRef {
    let field = Arc::new(Field::new("item", child.data_type().clone(), true));
    Arc::new(ListArray::new(field, offsets, child, nulls.cloned()))
}

/// `element_at(m, key)`: for each row, the value whose key equals the literal `key`
/// (the first match), or null if the row is null or the key is absent.
fn element_at(map: &MapArray, key: Option<&Literal>) -> Result<ArrayRef, ExprError> {
    let key = key.ok_or_else(|| ExprError::MissingArgument {
        func: "element_at".into(),
        arg: "key",
    })?;
    // Compare every key element against the literal, both promoted to a common type, so
    // the lookup works regardless of the map's key width/encoding (Int32, LargeUtf8, …).
    // Previously only exact Int64/Utf8 keys matched, so a get on an Int32-keyed map (the
    // narrow types are not normalized inside a nested Map) always returned null.
    let eq = eq_against_literal(map.keys(), key)?;
    let offsets = map.value_offsets();
    let mut idx: Vec<Option<u32>> = Vec::with_capacity(map.len());
    for row in 0..map.len() {
        if map.is_null(row) {
            idx.push(None);
            continue;
        }
        let (s, e) = (offsets[row] as usize, offsets[row + 1] as usize);
        idx.push(
            (s..e)
                .find(|&j| eq.is_valid(j) && eq.value(j))
                .map(|j| j as u32),
        );
    }
    Ok(take(map.values().as_ref(), &UInt32Array::from(idx), None)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        AsArray, Int32Builder, Int64Array, Int64Builder, MapBuilder, StringArray, StringBuilder,
    };
    use arrow::buffer::NullBuffer;
    use arrow::datatypes::Int64Type;
    use arrow::datatypes::{DataType, Fields};

    fn sample_map() -> ArrayRef {
        // Rows: {a:1, b:2}, {c:3}, null.
        let mut b = MapBuilder::new(None, StringBuilder::new(), Int64Builder::new());
        b.keys().append_value("a");
        b.values().append_value(1);
        b.keys().append_value("b");
        b.values().append_value(2);
        b.append(true).unwrap();
        b.keys().append_value("c");
        b.values().append_value(3);
        b.append(true).unwrap();
        b.append(false).unwrap(); // null map row
        Arc::new(b.finish())
    }

    #[test]
    fn element_at_finds_value_or_null() {
        let m = sample_map();
        let out = eval_map(MapFunc::ElementAt, &m, Some(&Literal::Str("a".into()))).unwrap();
        let a = out.as_primitive::<Int64Type>();
        assert_eq!(a.value(0), 1); // {a:1,b:2} → 1
        assert!(a.is_null(1)); // {c:3} has no 'a'
        assert!(a.is_null(2)); // null map → null
    }

    #[test]
    fn element_at_matches_a_narrow_int_key() {
        // A Map with Int32 keys (narrow types are not normalized inside a nested Map): a
        // lookup with an Int literal must still match, not silently return null.
        let mut b = MapBuilder::new(None, Int32Builder::new(), Int64Builder::new());
        b.keys().append_value(1);
        b.values().append_value(10);
        b.keys().append_value(2);
        b.values().append_value(20);
        b.append(true).unwrap();
        let m: ArrayRef = Arc::new(b.finish());
        let out = eval_map(MapFunc::ElementAt, &m, Some(&Literal::Int(2))).unwrap();
        let a = out.as_primitive::<Int64Type>();
        assert_eq!(a.value(0), 20);
        // A key that is not present is still null.
        let miss = eval_map(MapFunc::ElementAt, &m, Some(&Literal::Int(9))).unwrap();
        assert!(miss.as_primitive::<Int64Type>().is_null(0));
    }

    #[test]
    fn map_keys_wraps_under_offsets() {
        let m = sample_map();
        let out = eval_map(MapFunc::MapKeys, &m, None).unwrap();
        let list = out.as_list::<i32>();
        assert_eq!(list.value_length(0), 2); // 2 keys in row 0
        assert_eq!(list.value_length(1), 1); // 1 key in row 1
        assert!(list.is_null(2)); // null map → null list
    }

    /// Rows: {a:1, b:"x"}, null, {a:3, b:"z"} — heterogeneous children, and a null
    /// parent whose children still hold values underneath it.
    fn sample_struct() -> ArrayRef {
        let a = Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef;
        let b = Arc::new(StringArray::from(vec!["x", "y", "z"])) as ArrayRef;
        let fields = Fields::from(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Utf8, true),
        ]);
        Arc::new(StructArray::new(
            fields,
            vec![a, b],
            Some(NullBuffer::from(vec![true, false, true])),
        ))
    }

    #[test]
    fn a_struct_subscript_is_a_field_access() {
        let s = sample_struct();
        let out = eval_map(MapFunc::ElementAt, &s, Some(&Literal::Str("a".into()))).unwrap();
        assert_eq!(out.as_primitive::<Int64Type>().value(0), 1);
        assert_eq!(out.as_primitive::<Int64Type>().value(2), 3);
    }

    #[test]
    fn a_null_struct_row_is_null_even_though_its_child_holds_a_value() {
        // The case a bare `column_by_name` gets wrong: Arrow keeps the struct's null mask
        // on the parent, and the child buffer under it is unconstrained — here it holds 2.
        let s = sample_struct();
        let out = eval_map(MapFunc::ElementAt, &s, Some(&Literal::Str("a".into()))).unwrap();
        assert!(
            out.is_null(1),
            "a null struct row must answer null, not the child's 2"
        );
    }

    #[test]
    fn a_struct_field_that_does_not_exist_is_an_error_not_a_null() {
        // Unlike a map, whose absent key is an ordinary null result: a struct's fields are
        // fixed by its type, so naming one it lacks is a mistake.
        let s = sample_struct();
        let err = eval_map(MapFunc::ElementAt, &s, Some(&Literal::Str("nope".into())));
        assert!(err.is_err());
    }

    #[test]
    fn struct_keys_are_the_field_names_on_every_row() {
        let s = sample_struct();
        let out = eval_map(MapFunc::MapKeys, &s, None).unwrap();
        let list = out.as_list::<i32>();
        assert_eq!(list.value_length(0), 2);
        assert!(list.is_null(1)); // null struct → null list
        let row0 = list.value(0);
        let names = row0.as_string::<i32>();
        assert_eq!((names.value(0), names.value(1)), ("a", "b"));
    }

    #[test]
    fn a_non_string_key_on_a_struct_is_refused() {
        let s = sample_struct();
        assert!(eval_map(MapFunc::ElementAt, &s, Some(&Literal::Int(0))).is_err());
    }
}
