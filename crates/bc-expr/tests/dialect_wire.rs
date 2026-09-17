//! The IR documents the Python `.str` parameters emit for another engine's semantics,
//! deserialized and evaluated here.
//!
//! Each parameter lowers to a `StrFunc` tag or a value in an existing slot, and the golden
//! shapes are pinned on the Python side by `tests/unit/data/ir_snapshot_golden.json`. This
//! file is the other half of that contract: the same JSON, byte for byte, must deserialize
//! into the `Expr` the kernel expects and answer what the parameter promises. A tag that
//! drifted on either side fails here as a deserialization error rather than reaching a user
//! as a wrong result.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, BinaryArray, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use bc_expr::Expr;

fn batch(values: Vec<Option<&str>>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));
    let column: ArrayRef = Arc::new(StringArray::from(values));
    RecordBatch::try_new(schema, vec![column]).unwrap()
}

fn eval(json: &str, values: Vec<Option<&str>>) -> Result<ArrayRef, bc_expr::ExprError> {
    let expr: Expr = serde_json::from_str(json).expect("the Python IR deserializes");
    expr.eval(&batch(values))
}

fn strings(out: &ArrayRef) -> Vec<Option<String>> {
    out.as_string::<i32>()
        .iter()
        .map(|v| v.map(str::to_string))
        .collect()
}

fn owned(values: &[Option<&str>]) -> Vec<Option<String>> {
    values.iter().map(|v| v.map(str::to_string)).collect()
}

const COL: &str = r#"{"e":"col","name":"s"}"#;

fn str_ir(func: &str, extra: &str) -> String {
    format!(r#"{{"e":"str","fn":"{func}","input":{COL}{extra}}}"#)
}

#[test]
fn every_new_tag_deserializes_and_keeps_nulls() {
    for (func, extra) in [
        ("initcap_space", ""),
        ("regexp_extract_or_null", r#","pattern":"(a)","start":1"#),
        (
            "regexp_extract_all_or_empty",
            r#","pattern":"(a)?b","start":1"#,
        ),
        (
            "regexp_replace_dollar",
            r#","pattern":"(a)","replacement":"$1""#,
        ),
        (
            "regexp_replace_all_dollar",
            r#","pattern":"(a)","replacement":"$1""#,
        ),
        ("unhex_binary", ""),
        ("from_base64_binary", ""),
        ("url_encode_form", ""),
        ("url_decode_form", ""),
        ("damerau_levenshtein_osa", r#","pattern":"abc""#),
        ("mask_by_class", r#","pattern":"Xxn\u0000""#),
    ] {
        let out = eval(&str_ir(func, extra), vec![None, Some("")]).unwrap();
        assert_eq!(out.len(), 2, "{func}");
        assert!(out.is_null(0), "{func}: a null input must stay null");
    }
}

#[test]
fn the_parameters_answer_what_they_promise() {
    let initcap = eval(&str_ir("initcap_space", ""), vec![Some("hello-world foo")]).unwrap();
    assert_eq!(strings(&initcap), owned(&[Some("Hello-world Foo")]));

    let extract = str_ir(
        "regexp_extract_or_null",
        r#","pattern":"b([0-9])?","start":1"#,
    );
    let out = eval(&extract, vec![Some("b1"), Some("b"), Some("x")]).unwrap();
    assert_eq!(strings(&out), owned(&[Some("1"), None, None]));

    let dollar = str_ir(
        "regexp_replace_all_dollar",
        r#","pattern":"(l)","replacement":"[$1]""#,
    );
    let out = eval(&dollar, vec![Some("hello")]).unwrap();
    assert_eq!(strings(&out), owned(&[Some("he[l][l]o")]));

    let unhex = eval(&str_ir("unhex_binary", ""), vec![Some("ff00"), Some("zz")]).unwrap();
    let unhex = unhex.as_any().downcast_ref::<BinaryArray>().unwrap();
    assert_eq!(unhex.value(0), [0xff, 0x00]);
    assert!(unhex.is_null(1));

    let osa = str_ir("damerau_levenshtein_osa", r#","pattern":"abc""#);
    let out = eval(&osa, vec![Some("ca")]).unwrap();
    assert_eq!(
        out.as_any().downcast_ref::<Int64Array>().unwrap().value(0),
        3
    );

    let masked = str_ir("mask_by_class", r#","pattern":"Xxn\u0000""#);
    let out = eval(&masked, vec![Some("AbCD123-@$#")]).unwrap();
    assert_eq!(strings(&out), owned(&[Some("XxXXnnn-@$#")]));
}

#[test]
fn the_slot_parameters_ride_the_existing_fields() {
    let seeded = eval(&str_ir("xxhash64", r#","start":42"#), vec![Some("ABC")]).unwrap();
    let seeded = seeded.as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(seeded.value(0), 4_105_715_581_806_190_027);

    let split = str_ir("regexp_split", r#","pattern":"-","length":2"#);
    let out = eval(&split, vec![Some("a-b-c")]).unwrap();
    let list = out.as_list::<i32>();
    assert_eq!(list.value(0).len(), 2);
    assert_eq!(list.value(0).as_string::<i32>().value(1), "b-c");
}

#[test]
fn a_strict_strptime_raises_where_the_lenient_one_nulls() {
    let lenient = format!(r#"{{"e":"strptime","input":{COL},"format":"%Y-%m-%d"}}"#);
    let out = eval(&lenient, vec![Some("bad"), None]).unwrap();
    assert!(out.is_null(0) && out.is_null(1));

    let strict = format!(r#"{{"e":"strptime","input":{COL},"format":"%Y-%m-%d","strict":true}}"#);
    assert!(eval(&strict, vec![Some("2024-02-15"), None]).is_ok());
    let err = eval(&strict, vec![Some("bad")]).unwrap_err().to_string();
    assert!(err.contains("does not match the format"), "{err}");
}
