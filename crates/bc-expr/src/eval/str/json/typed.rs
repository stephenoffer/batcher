//! `json.decode(dtype)` and `json.encode()`: JSON text to a typed Arrow value and back.
//!
//! **Decode** follows DuckDB's `json_transform`, which is the oracle the differential tests
//! hold it to. A key the document does not have, a JSON `null`, and a value of the wrong
//! shape all decode to null; a key the type does not name is ignored. Leaves convert the way
//! DuckDB's casts do: a JSON float rounds into an integer (ties to even), a bool reads as
//! `1`/`0`, a numeric string parses, and a text target takes a string unquoted and anything
//! else as its compact JSON. A document that does not parse is null rather than an error,
//! which is this module's standing divergence from DuckDB. `strict` is DuckDB's
//! `json_transform_strict`: any of those nulls, and an unparseable document, raises instead.
//!
//! The decode is **columnar**: each document is parsed once, and the target type is then
//! walked one level at a time over every row together, so a struct field or list element
//! becomes one child column built in a single pass rather than a per-row tree of builders.
//!
//! **Encode** is DuckDB's `to_json`: compact text, a null field as `null`, a null row as SQL
//! null. A non-finite float has no JSON spelling and is written `null` (DuckDB writes `NaN`,
//! which no JSON parser reads back). A temporal or other non-JSON leaf is written as the
//! string Arrow's cast to text gives it.

use std::sync::Arc;

use arrow::array::{
    new_null_array, Array, ArrayRef, AsArray, BooleanArray, Float64Array, Int64Array, ListArray,
    MapArray, StringArray, StructArray, UInt64Array,
};
use arrow::buffer::{NullBuffer, OffsetBuffer};
use arrow::compute::{cast, cast_with_options, CastOptions};
use arrow::datatypes::DataType;
use serde_json::Value;

use crate::ExprError;

/// Decode each row of the Utf8 column `arr` into `target`.
pub(in crate::eval) fn decode(
    arr: &ArrayRef,
    target: &DataType,
    strict: bool,
) -> Result<ArrayRef, ExprError> {
    let text = cast(arr, &DataType::Utf8)?;
    let text = text.as_string::<i32>();
    let docs: Vec<Option<Value>> = text
        .iter()
        .map(|row| match row {
            None => Ok(None),
            Some(t) => match serde_json::from_str::<Value>(t) {
                Ok(v) => Ok(Some(v)),
                Err(_) if !strict => Ok(None),
                Err(e) => Err(failure(&format!("malformed JSON ({e})"), target)),
            },
        })
        .collect::<Result<_, _>>()?;
    let refs: Vec<Option<&Value>> = docs.iter().map(Option::as_ref).collect();
    column(&refs, target, strict)
}

fn failure(what: &str, target: &DataType) -> ExprError {
    ExprError::InvalidArgument {
        func: "json.decode(strict=True)".into(),
        reason: format!("cannot read {what} as {target}"),
    }
}

/// One column of `dtype` from one JSON value (or absence) per row.
fn column(
    values: &[Option<&Value>],
    dtype: &DataType,
    strict: bool,
) -> Result<ArrayRef, ExprError> {
    let mismatch = |v: &Value| -> Result<(), ExprError> {
        if strict {
            Err(failure(&v.to_string(), dtype))
        } else {
            Ok(())
        }
    };
    match dtype {
        DataType::List(field) => {
            let mut offsets = vec![0i32];
            let mut valid = Vec::with_capacity(values.len());
            let mut items: Vec<Option<&Value>> = Vec::new();
            for v in values {
                match present(*v) {
                    Some(Value::Array(xs)) => {
                        items.extend(xs.iter().map(Some));
                        valid.push(true);
                    }
                    Some(other) => {
                        mismatch(other)?;
                        valid.push(false);
                    }
                    None => valid.push(false),
                }
                offsets.push(items.len() as i32);
            }
            let child = column(&items, field.data_type(), strict)?;
            Ok(Arc::new(ListArray::try_new(
                Arc::clone(field),
                OffsetBuffer::new(offsets.into()),
                child,
                Some(NullBuffer::from(valid)),
            )?))
        }
        DataType::Struct(fields) => {
            let mut valid = Vec::with_capacity(values.len());
            let objects: Vec<Option<&serde_json::Map<String, Value>>> = values
                .iter()
                .map(|v| match present(*v) {
                    Some(Value::Object(m)) => Ok(Some(m)),
                    Some(other) => mismatch(other).map(|()| None),
                    None => Ok(None),
                })
                .collect::<Result<_, _>>()?;
            valid.extend(objects.iter().map(Option::is_some));
            let children = fields
                .iter()
                .map(|f| {
                    let sub: Vec<Option<&Value>> = objects
                        .iter()
                        .map(|o| o.and_then(|m| m.get(f.name())))
                        .collect();
                    column(&sub, f.data_type(), strict)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Arc::new(StructArray::try_new(
                fields.clone(),
                children,
                Some(NullBuffer::from(valid)),
            )?))
        }
        DataType::Map(entries, sorted) => {
            let DataType::Struct(kv) = entries.data_type() else {
                return Err(failure("a map", dtype));
            };
            let mut offsets = vec![0i32];
            let mut valid = Vec::with_capacity(values.len());
            let (mut keys, mut vals): (Vec<String>, Vec<Option<&Value>>) = (vec![], vec![]);
            for v in values {
                match present(*v) {
                    Some(Value::Object(m)) => {
                        for (k, x) in m {
                            keys.push(k.clone());
                            vals.push(Some(x));
                        }
                        valid.push(true);
                    }
                    Some(other) => {
                        mismatch(other)?;
                        valid.push(false);
                    }
                    None => valid.push(false),
                }
                offsets.push(keys.len() as i32);
            }
            let key_text: ArrayRef = Arc::new(StringArray::from(keys));
            let key_col = cast(&key_text, kv[0].data_type())?;
            if key_col.null_count() > 0 {
                return Err(failure("an object key", kv[0].data_type()));
            }
            let value_col = column(&vals, kv[1].data_type(), strict)?;
            let entries_arr = StructArray::try_new(kv.clone(), vec![key_col, value_col], None)?;
            Ok(Arc::new(MapArray::try_new(
                Arc::clone(entries),
                OffsetBuffer::new(offsets.into()),
                entries_arr,
                Some(NullBuffer::from(valid)),
                *sorted,
            )?))
        }
        DataType::Boolean => leaf(values, strict, dtype, |v| match v {
            Value::Bool(b) => Some(*b),
            Value::Number(n) => n.as_f64().map(|f| f != 0.0),
            Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
                "true" => Some(true),
                "false" => Some(false),
                _ => None,
            },
            _ => None,
        })
        .map(|a: BooleanArray| Arc::new(a) as ArrayRef),
        t if t.is_integer() => {
            let wide: Int64Array = leaf(values, strict, dtype, |v| match v {
                Value::Number(n) => super::number_to_i64(n),
                Value::Bool(b) => Some(i64::from(*b)),
                Value::String(s) => parse_i64(s.trim()),
                _ => None,
            })?;
            narrow(Arc::new(wide), dtype, strict)
        }
        t if t.is_floating() => {
            let wide: Float64Array = leaf(values, strict, dtype, |v| match v {
                Value::Number(n) => n.as_f64(),
                Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
                Value::String(s) => s.trim().parse::<f64>().ok(),
                _ => None,
            })?;
            narrow(Arc::new(wide), dtype, strict)
        }
        DataType::Null => Ok(new_null_array(&DataType::Null, values.len())),
        _ => {
            // Text, and every leaf JSON has no spelling of its own (a date, a timestamp, a
            // decimal): read as text, then let Arrow's cast parse it.
            let text: StringArray = leaf(values, strict, dtype, |v| match v {
                Value::String(s) => Some(s.clone()),
                other => Some(other.to_string()),
            })?;
            narrow(Arc::new(text), dtype, strict)
        }
    }
}

/// A JSON null is absence: every target reads it as null, and strict mode accepts it.
fn present(v: Option<&Value>) -> Option<&Value> {
    v.filter(|v| !v.is_null())
}

/// A leaf column, converting each present value with `read`; a value `read` rejects is
/// null, or an error under `strict`.
fn leaf<T, A>(
    values: &[Option<&Value>],
    strict: bool,
    dtype: &DataType,
    read: impl Fn(&Value) -> Option<T>,
) -> Result<A, ExprError>
where
    A: FromIterator<Option<T>>,
{
    values
        .iter()
        .map(|v| match present(*v) {
            None => Ok(None),
            Some(x) => match read(x) {
                Some(t) => Ok(Some(t)),
                None if strict => Err(failure(&x.to_string(), dtype)),
                None => Ok(None),
            },
        })
        .collect()
}

/// Cast a wide leaf to the exact target; a value that does not fit is null, or an error
/// under `strict`.
fn narrow(wide: ArrayRef, dtype: &DataType, strict: bool) -> Result<ArrayRef, ExprError> {
    if wide.data_type() == dtype {
        return Ok(wide);
    }
    let out = cast_with_options(&wide, dtype, &CastOptions::default())?;
    if strict && out.null_count() > wide.null_count() {
        return Err(failure("a value outside the type's range", dtype));
    }
    Ok(out)
}

/// An integer from text, as DuckDB's cast reads one: digits, or a float rounded to even.
fn parse_i64(s: &str) -> Option<i64> {
    s.parse::<i64>().ok().or_else(|| {
        let f = s.parse::<f64>().ok()?.round_ties_even();
        (-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0)
            .contains(&f)
            .then_some(f as i64)
    })
}

/// Encode every row of `arr` as compact JSON text; a null row is SQL null.
pub(in crate::eval) fn encode(arr: &ArrayRef) -> Result<ArrayRef, ExprError> {
    let enc = Enc::new(arr)?;
    let mut out = String::new();
    let rows = (0..arr.len()).map(|i| {
        if arr.is_null(i) {
            return None;
        }
        out.clear();
        enc.write(i, &mut out);
        Some(out.clone())
    });
    Ok(Arc::new(rows.collect::<StringArray>()))
}

/// A column prepared for encoding: leaves widened to one array type per JSON kind, so the
/// per-row walk does no casting and no downcasting.
enum Enc {
    Null,
    Bool(BooleanArray),
    Int(Int64Array),
    UInt(UInt64Array),
    Float(Float64Array),
    /// Written verbatim: a decimal, whose text is already a JSON number.
    Number(StringArray),
    /// Written as a JSON string.
    Text(StringArray),
    List {
        offsets: OffsetBuffer<i32>,
        nulls: Option<NullBuffer>,
        child: Box<Enc>,
    },
    Struct {
        names: Vec<String>,
        nulls: Option<NullBuffer>,
        children: Vec<Enc>,
    },
    Map {
        offsets: OffsetBuffer<i32>,
        nulls: Option<NullBuffer>,
        keys: Box<Enc>,
        values: Box<Enc>,
    },
}

impl Enc {
    fn new(arr: &ArrayRef) -> Result<Enc, ExprError> {
        let widen = |t: &DataType| cast(arr, t);
        Ok(match arr.data_type() {
            DataType::Null => Enc::Null,
            DataType::Boolean => Enc::Bool(arr.as_boolean().clone()),
            t if t.is_signed_integer() => Enc::Int(widen(&DataType::Int64)?.as_primitive().clone()),
            t if t.is_unsigned_integer() => {
                Enc::UInt(widen(&DataType::UInt64)?.as_primitive().clone())
            }
            t if t.is_floating() => Enc::Float(widen(&DataType::Float64)?.as_primitive().clone()),
            DataType::Decimal128(..) | DataType::Decimal256(..) => {
                Enc::Number(widen(&DataType::Utf8)?.as_string::<i32>().clone())
            }
            DataType::Dictionary(_, value) => Enc::new(&cast(arr, value)?)?,
            DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(..) => {
                let list = crate::eval::list_ops::as_var_list(arr, "json.encode")?;
                let list = list.as_list::<i32>();
                Enc::List {
                    offsets: list.offsets().clone(),
                    nulls: list.nulls().cloned(),
                    child: Box::new(Enc::new(list.values())?),
                }
            }
            DataType::Struct(fields) => {
                let s = arr.as_struct();
                Enc::Struct {
                    names: fields.iter().map(|f| f.name().clone()).collect(),
                    nulls: s.nulls().cloned(),
                    children: s.columns().iter().map(Enc::new).collect::<Result<_, _>>()?,
                }
            }
            DataType::Map(..) => {
                let m = arr.as_map();
                let keys: ArrayRef = cast(m.keys(), &DataType::Utf8)?;
                Enc::Map {
                    offsets: m.offsets().clone(),
                    nulls: m.nulls().cloned(),
                    keys: Box::new(Enc::Text(keys.as_string::<i32>().clone())),
                    values: Box::new(Enc::new(m.values())?),
                }
            }
            _ => Enc::Text(widen(&DataType::Utf8)?.as_string::<i32>().clone()),
        })
    }

    fn is_null(&self, i: usize) -> bool {
        match self {
            Enc::Null => true,
            Enc::Bool(a) => a.is_null(i),
            Enc::Int(a) => a.is_null(i),
            Enc::UInt(a) => a.is_null(i),
            Enc::Float(a) => a.is_null(i) || !a.value(i).is_finite(),
            Enc::Number(a) | Enc::Text(a) => a.is_null(i),
            Enc::List { nulls, .. } | Enc::Struct { nulls, .. } | Enc::Map { nulls, .. } => {
                nulls.as_ref().is_some_and(|n| n.is_null(i))
            }
        }
    }

    fn write(&self, i: usize, out: &mut String) {
        if self.is_null(i) {
            out.push_str("null");
            return;
        }
        match self {
            Enc::Null => out.push_str("null"),
            Enc::Bool(a) => out.push_str(if a.value(i) { "true" } else { "false" }),
            Enc::Int(a) => out.push_str(&a.value(i).to_string()),
            Enc::UInt(a) => out.push_str(&a.value(i).to_string()),
            // `Debug` is the shortest text that reads back to the same double, and keeps a
            // `.0` on an integral one, so a float column stays a float when parsed back.
            Enc::Float(a) => out.push_str(&format!("{:?}", a.value(i))),
            Enc::Number(a) => out.push_str(a.value(i)),
            Enc::Text(a) => quote(a.value(i), out),
            Enc::List { offsets, child, .. } => {
                out.push('[');
                for (n, k) in (offsets[i] as usize..offsets[i + 1] as usize).enumerate() {
                    if n > 0 {
                        out.push(',');
                    }
                    child.write(k, out);
                }
                out.push(']');
            }
            Enc::Struct {
                names, children, ..
            } => {
                out.push('{');
                for (n, (name, child)) in names.iter().zip(children).enumerate() {
                    if n > 0 {
                        out.push(',');
                    }
                    quote(name, out);
                    out.push(':');
                    child.write(i, out);
                }
                out.push('}');
            }
            Enc::Map {
                offsets,
                keys,
                values,
                ..
            } => {
                out.push('{');
                for (n, k) in (offsets[i] as usize..offsets[i + 1] as usize).enumerate() {
                    if n > 0 {
                        out.push(',');
                    }
                    keys.write(k, out);
                    out.push(':');
                    values.write(k, out);
                }
                out.push('}');
            }
        }
    }
}

/// Append `s` as a JSON string literal.
fn quote(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{Field, Fields};

    fn field(name: &str, t: DataType) -> Field {
        Field::new(name, t, true)
    }

    fn text(rows: &[Option<&str>]) -> ArrayRef {
        Arc::new(StringArray::from(rows.to_vec()))
    }

    fn shape() -> DataType {
        DataType::Struct(Fields::from(vec![
            field("a", DataType::Int64),
            field(
                "b",
                DataType::List(Arc::new(Field::new_list_field(DataType::Int64, true))),
            ),
            field("c", DataType::Utf8),
        ]))
    }

    #[test]
    fn decode_follows_json_transform_and_encode_reverses_it() {
        let rows = [
            Some(r#"{"a": 1, "b": [1, "x", 2.5], "c": "hi", "extra": 0}"#),
            Some(r#"{"a": 1.5, "c": {"d": 5}}"#),
            Some(r#"{"a": "7", "b": "notalist"}"#),
            Some("[1]"),
            Some("nope"),
            None,
        ];
        let out = decode(&text(&rows), &shape(), false).unwrap();
        let back = encode(&out).unwrap();
        let back: Vec<Option<&str>> = back.as_string::<i32>().iter().collect();
        assert_eq!(
            back,
            vec![
                Some(r#"{"a":1,"b":[1,null,2],"c":"hi"}"#),
                Some(r#"{"a":2,"b":null,"c":"{\"d\":5}"}"#),
                Some(r#"{"a":7,"b":null,"c":null}"#),
                None,
                None,
                None,
            ]
        );
    }

    #[test]
    fn strict_decode_raises_where_lenient_decode_nulls() {
        for bad in [r#"{"a": "x"}"#, r#"{"b": 3}"#, "nope", "[1]"] {
            let err = decode(&text(&[Some(bad)]), &shape(), true);
            assert!(err.is_err(), "{bad} should raise under strict");
        }
        let ok = decode(&text(&[Some(r#"{"a": null}"#), None]), &shape(), true).unwrap();
        assert_eq!(ok.null_count(), 1);
    }

    #[test]
    fn a_map_target_keeps_every_key_in_document_order() {
        let map = DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(Fields::from(vec![
                    Field::new("key", DataType::Utf8, false),
                    field("value", DataType::Float64),
                ])),
                false,
            )),
            false,
        );
        let out = decode(&text(&[Some(r#"{"z": 1, "a": 2.5}"#)]), &map, false).unwrap();
        let back = encode(&out).unwrap();
        assert_eq!(back.as_string::<i32>().value(0), r#"{"z":1.0,"a":2.5}"#);
    }

    #[test]
    fn encode_writes_valid_json_for_awkward_values() {
        let floats: ArrayRef = Arc::new(Float64Array::from(vec![Some(2.0), Some(f64::NAN), None]));
        let s: ArrayRef = Arc::new(StringArray::from(vec![
            Some("q\"\\\n\u{1}"),
            None,
            Some(""),
        ]));
        let st: ArrayRef = Arc::new(StructArray::from(vec![
            (Arc::new(field("f", DataType::Float64)), floats),
            (Arc::new(field("s", DataType::Utf8)), s),
        ]));
        let out = encode(&st).unwrap();
        let rows: Vec<&str> = out.as_string::<i32>().iter().flatten().collect();
        assert_eq!(rows[0], r#"{"f":2.0,"s":"q\"\\\n\u0001"}"#);
        assert_eq!(rows[1], r#"{"f":null,"s":null}"#);
        assert_eq!(rows[2], r#"{"f":null,"s":""}"#);
        for r in rows {
            serde_json::from_str::<Value>(r).unwrap();
        }
    }
}
