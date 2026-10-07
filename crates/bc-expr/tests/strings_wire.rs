//! The IR the Python `.str` namespace emits for the Unicode, group-extract, offset-chunk,
//! path-separator and bounded-decompress parameters, deserialized and evaluated here.
//!
//! This is the Rust half of the wire contract: the JSON `to_ir()` produces for each new tag
//! or slot must deserialize into the `Expr` the kernel expects. A tag that drifted on
//! either side fails here as a deserialization error rather than reaching a user as a
//! wrong answer. The Python half is `tests/unit/test_str_wire_shapes.py`.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Int64Type, Schema};
use bc_expr::Expr;

fn eval(json: &str, values: Vec<Option<&str>>) -> ArrayRef {
    let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));
    let column: ArrayRef = Arc::new(StringArray::from(values));
    let batch = RecordBatch::try_new(schema, vec![column]).unwrap();
    let expr: Expr = serde_json::from_str(json).expect("the Python IR deserializes");
    expr.eval(&batch).unwrap()
}

fn strings(out: &ArrayRef) -> Vec<Option<String>> {
    out.as_string::<i32>()
        .iter()
        .map(|v| v.map(str::to_string))
        .collect()
}

const COL: &str = r#"{"e":"col","name":"s"}"#;

fn str_ir(func: &str, extra: &str) -> String {
    format!(r#"{{"e":"str","fn":"{func}","input":{COL}{extra}}}"#)
}

#[test]
fn normalize_and_casefold_deserialize() {
    let out = eval(
        &str_ir("normalize", r#","pattern":"NFC""#),
        vec![Some("e\u{301}"), None],
    );
    assert_eq!(strings(&out), [Some("\u{e9}".to_string()), None]);
    let out = eval(&str_ir("casefold", ""), vec![Some("Stra\u{df}e")]);
    assert_eq!(strings(&out), [Some("strasse".to_string())]);
}

#[test]
fn grapheme_tags_deserialize() {
    let family = "\u{1F468}\u{200D}\u{1F469}x";
    let out = eval(&str_ir("length_grapheme", ""), vec![Some(family)]);
    assert_eq!(out.as_primitive::<Int64Type>().value(0), 2);
    let out = eval(
        &str_ir("substring_grapheme", r#","start":2,"length":1"#),
        vec![Some(family)],
    );
    assert_eq!(strings(&out), [Some("x".to_string())]);
}

#[test]
fn extract_groups_is_a_struct_named_by_the_groups() {
    let out = eval(
        &str_ir("regexp_extract_groups", r#","pattern":"(?P<k>\\w)=(\\d)""#),
        vec![Some("a=1"), Some("zz"), None],
    );
    let st = out.as_struct();
    let names: Vec<&str> = st.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, ["k", "2"]);
    assert_eq!(st.column(0).as_string::<i32>().value(1), "");
    assert!(out.is_null(2));
    let out = eval(
        &str_ir("regexp_extract_groups_or_null", r#","pattern":"(a)""#),
        vec![Some("b")],
    );
    assert!(out.as_struct().column(0).is_null(0));
}

#[test]
fn chunk_offsets_is_a_list_of_text_and_start() {
    let out = eval(
        &str_ir("chunk_offsets", r#","pattern":"char","start":1,"length":3"#),
        vec![Some("abcde")],
    );
    let list = out.as_list::<i32>();
    let pieces = list.value(0);
    let pieces = pieces.as_struct();
    let starts: Vec<i64> = pieces
        .column_by_name("start")
        .unwrap()
        .as_primitive::<Int64Type>()
        .values()
        .to_vec();
    assert_eq!(starts, [0, 2]);
}

#[test]
fn path_separator_rides_the_pattern_slot() {
    let out = eval(
        &str_ir("parse_filename", r#","pattern":"forward""#),
        vec![Some("/a/b\\c.txt")],
    );
    assert_eq!(strings(&out), [Some("b\\c.txt".to_string())]);
    let out = eval(&str_ir("parse_filename", ""), vec![Some("/a/b\\c.txt")]);
    assert_eq!(strings(&out), [Some("c.txt".to_string())]);
}

#[test]
fn decompress_bound_rides_the_length_slot() {
    // `compress` then `decompress` under a bound below and at the payload size.
    let payload = "x".repeat(100);
    let packed = r#"{"e":"str","fn":"compress","input":{"e":"col","name":"s"},"pattern":"gzip"}"#;
    for (cap, expect_null) in [(99, true), (100, false)] {
        let json = format!(
            r#"{{"e":"str","fn":"decompress","input":{packed},"pattern":"gzip","length":{cap}}}"#
        );
        let out = eval(&json, vec![Some(&payload)]);
        assert_eq!(out.is_null(0), expect_null, "cap {cap}");
    }
}
