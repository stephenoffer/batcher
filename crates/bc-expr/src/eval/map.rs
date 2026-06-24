//! Map-column evaluation for `Expr::Map` (`map_keys`/`map_values`/`element_at`).
//!
//! An Arrow `Map` is a `List<Struct<key, value>>`. `map_keys`/`map_values` re-wrap
//! the entries' key/value child under the map's own offsets to yield a `List`;
//! `element_at` scans each row's entries for a literal key and `take`s the matching
//! value (null if absent). The JIT does not compile `Map`, so this is the only path.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int64Array, ListArray, MapArray, StringArray, UInt32Array};
use arrow::compute::take;
use arrow::datatypes::Field;

use crate::{ExprError, Literal, MapFunc};

/// Evaluate a map function over a `Map` array.
pub(crate) fn eval_map(
    func: MapFunc,
    arr: &ArrayRef,
    key: Option<&Literal>,
) -> Result<ArrayRef, ExprError> {
    let map = arr
        .as_any()
        .downcast_ref::<MapArray>()
        .ok_or_else(|| ExprError::ExpectedString {
            func: format!("{func:?}"),
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
    let keys = map.keys();
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
                .find(|&j| key_matches(keys, j, key))
                .map(|j| j as u32),
        );
    }
    Ok(take(map.values().as_ref(), &UInt32Array::from(idx), None)?)
}

/// Does the key at flat index `j` equal the literal `key`? Supports the common key
/// types (Utf8, Int64); other key types never match (→ null lookup).
fn key_matches(keys: &ArrayRef, j: usize, key: &Literal) -> bool {
    if keys.is_null(j) {
        return false;
    }
    match key {
        Literal::Str(s) => keys
            .as_any()
            .downcast_ref::<StringArray>()
            .is_some_and(|a| a.value(j) == s),
        Literal::Int(n) => keys
            .as_any()
            .downcast_ref::<Int64Array>()
            .is_some_and(|a| a.value(j) == *n),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Builder, MapBuilder, StringBuilder};

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
        let a = out.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(a.value(0), 1); // {a:1,b:2} → 1
        assert!(a.is_null(1)); // {c:3} has no 'a'
        assert!(a.is_null(2)); // null map → null
    }

    #[test]
    fn map_keys_wraps_under_offsets() {
        let m = sample_map();
        let out = eval_map(MapFunc::MapKeys, &m, None).unwrap();
        let list = out.as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(list.value_length(0), 2); // 2 keys in row 0
        assert_eq!(list.value_length(1), 1); // 1 key in row 1
        assert!(list.is_null(2)); // null map → null list
    }
}
