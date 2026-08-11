//! `Expr::Seq` end to end: the JSON the Python control plane emits, deserialized and
//! evaluated over a real Arrow batch.
//!
//! This is the wire-contract test for the `.seq` surface. Every case is written as the JSON
//! document `to_ir()` produces rather than as a Rust `Expr` literal, because a Rust literal
//! would still pass if the serde tags drifted from what Python sends — which is precisely the
//! failure this file exists to catch. The three shapes that matter are all covered: a bare op
//! carrying no optional keys at all, one carrying `k`/`window`, and one carrying
//! `frame`/`to_stop`.
//!
//! The kernels themselves are unit-tested next to their implementations in `eval/seq/`; what
//! is under test here is that the wire shape reaches them intact.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, ListArray, RecordBatch, StringArray,
    StructArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use bc_expr::Expr;

/// A batch shaped like a real FASTQ read table: sequence, quality string, and a protein.
fn batch() -> RecordBatch {
    let dna = StringArray::from(vec![
        Some("ATGGCCTAA"),
        Some("atggcctaa"),
        Some("ACGTN"),
        Some(""),
        None,
    ]);
    let qual = StringArray::from(vec![
        Some("IIIIIIIII"),
        Some("!5I"),
        Some("IIIII"),
        Some(""),
        None,
    ]);
    let protein = StringArray::from(vec![
        Some("MAKV"),
        Some("KKKK"),
        Some("IIIVVVLLL"),
        Some(""),
        None,
    ]);
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("dna", DataType::Utf8, true),
            Field::new("qual", DataType::Utf8, true),
            Field::new("protein", DataType::Utf8, true),
        ])),
        vec![Arc::new(dna), Arc::new(qual), Arc::new(protein)],
    )
    .expect("batch")
}

fn eval(json: &str) -> ArrayRef {
    let e: Expr = serde_json::from_str(json).unwrap_or_else(|err| panic!("{json}: {err}"));
    e.eval(&batch())
        .unwrap_or_else(|err| panic!("{json}: {err}"))
}

fn strings(json: &str) -> Vec<Option<String>> {
    let a = eval(json);
    let s = a.as_any().downcast_ref::<StringArray>().expect("Utf8");
    (0..s.len())
        .map(|i| (!s.is_null(i)).then(|| s.value(i).to_string()))
        .collect()
}

fn floats(json: &str) -> Vec<Option<f64>> {
    let a = eval(json);
    let f = a.as_any().downcast_ref::<Float64Array>().expect("Float64");
    (0..f.len())
        .map(|i| (!f.is_null(i)).then(|| f.value(i)))
        .collect()
}

fn ints(json: &str) -> Vec<Option<i64>> {
    let a = eval(json);
    let v = a.as_any().downcast_ref::<Int64Array>().expect("Int64");
    (0..v.len())
        .map(|i| (!v.is_null(i)).then(|| v.value(i)))
        .collect()
}

fn str_lists(json: &str) -> Vec<Option<Vec<String>>> {
    let a = eval(json);
    let l = a.as_any().downcast_ref::<ListArray>().expect("List");
    (0..l.len())
        .map(|i| {
            (!l.is_null(i)).then(|| {
                let v = l.value(i);
                let s = v
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("List<Utf8>");
                (0..s.len()).map(|j| s.value(j).to_string()).collect()
            })
        })
        .collect()
}

/// The bare shape: `{"e": "seq", "fn": ..., "input": ...}` with no optional keys.
///
/// This is the case a stray `#[serde(default)]` omission would break, and it is the shape most
/// of the family emits — so it is checked first and on more than one function.
#[test]
fn a_bare_op_deserializes_with_no_optional_keys() {
    assert_eq!(
        strings(r#"{"e":"seq","fn":"reverse_complement","input":{"e":"col","name":"dna"}}"#),
        vec![
            Some("TTAGGCCAT".into()),
            Some("ttaggccat".into()),
            Some("NACGT".into()),
            Some("".into()),
            None,
        ]
    );
    assert_eq!(
        strings(r#"{"e":"seq","fn":"complement","input":{"e":"col","name":"dna"}}"#)[0],
        Some("TACCGGATT".into())
    );
    assert_eq!(
        strings(r#"{"e":"seq","fn":"transcribe","input":{"e":"col","name":"dna"}}"#)[0],
        Some("AUGGCCUAA".into())
    );
}

#[test]
fn composition_measures_reach_their_kernels() {
    assert_eq!(
        floats(r#"{"e":"seq","fn":"gc_content","input":{"e":"col","name":"dna"}}"#),
        vec![Some(4.0 / 9.0), Some(4.0 / 9.0), Some(0.5), None, None,]
    );
    assert_eq!(
        ints(r#"{"e":"seq","fn":"max_homopolymer","input":{"e":"col","name":"dna"}}"#),
        vec![Some(2), Some(2), Some(1), Some(0), None]
    );
}

#[test]
fn base_counts_arrives_as_a_seven_field_struct() {
    let a = eval(r#"{"e":"seq","fn":"base_counts","input":{"e":"col","name":"dna"}}"#);
    let st = a.as_any().downcast_ref::<StructArray>().expect("Struct");
    let read = |name: &str, row: usize| {
        st.column_by_name(name)
            .unwrap_or_else(|| panic!("no field {name}"))
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(row)
    };
    // "ATGGCCTAA": 3 A, 2 C, 2 G, 2 T.
    assert_eq!(
        [
            read("a", 0),
            read("c", 0),
            read("g", 0),
            read("t", 0),
            read("u", 0),
            read("n", 0),
            read("other", 0)
        ],
        [3, 2, 2, 2, 0, 0, 0]
    );
    assert!(st.is_null(4), "a null sequence has no counts");
}

/// The `k`/`window` shape, which the sketching functions carry.
#[test]
fn the_kmer_arguments_survive_the_wire() {
    assert_eq!(
        str_lists(r#"{"e":"seq","fn":"kmers","input":{"e":"col","name":"dna"},"k":3}"#)[0],
        Some(vec![
            "ATG".into(),
            "TGG".into(),
            "GGC".into(),
            "GCC".into(),
            "CCT".into(),
            "CTA".into(),
            "TAA".into(),
        ])
    );
    // Lower case folds, which is what makes a soft-masked repeat count with its copies.
    assert_eq!(
        str_lists(r#"{"e":"seq","fn":"kmers","input":{"e":"col","name":"dna"},"k":3}"#)[1],
        str_lists(r#"{"e":"seq","fn":"kmers","input":{"e":"col","name":"dna"},"k":3}"#)[0]
    );
    let mins = str_lists(
        r#"{"e":"seq","fn":"minimizers","input":{"e":"col","name":"dna"},"k":3,"window":4}"#,
    );
    let all =
        str_lists(r#"{"e":"seq","fn":"canonical_kmers","input":{"e":"col","name":"dna"},"k":3}"#);
    let (mins, all) = (mins[0].clone().unwrap(), all[0].clone().unwrap());
    assert!(!mins.is_empty());
    assert!(
        mins.len() < all.len(),
        "a sketch should be smaller than the whole"
    );
    for m in &mins {
        assert!(all.contains(m), "{m} is not one of the canonical k-mers");
    }
}

/// The `frame`/`to_stop` shape, which only `translate` carries.
#[test]
fn the_translate_arguments_survive_the_wire() {
    assert_eq!(
        strings(r#"{"e":"seq","fn":"translate","input":{"e":"col","name":"dna"}}"#)[0],
        Some("MA*".into())
    );
    assert_eq!(
        strings(r#"{"e":"seq","fn":"translate","input":{"e":"col","name":"dna"},"to_stop":true}"#)
            [0],
        Some("MA".into())
    );
    // Frame 1 of "ATGGCCTAA" reads TGG CCT -> WP.
    assert_eq!(
        strings(r#"{"e":"seq","fn":"translate","input":{"e":"col","name":"dna"},"frame":1}"#)[0],
        Some("WP".into())
    );
    // "ACGTN" is one complete codon (ACG -> T) plus a two-base remainder. The remainder is
    // dropped rather than padded, which is why the `N` never reaches the table — padding it
    // would have fabricated a second residue. (The `X`-for-an-ambiguous-codon rule is pinned
    // beside the kernel, in `eval/seq/translate.rs`.)
    assert_eq!(
        strings(r#"{"e":"seq","fn":"translate","input":{"e":"col","name":"dna"}}"#)[2],
        Some("T".into())
    );
}

/// The `offset` shape, which the FASTQ decoders carry.
#[test]
fn the_quality_offset_survives_the_wire() {
    assert_eq!(
        floats(r#"{"e":"seq","fn":"mean_quality","input":{"e":"col","name":"qual"}}"#)[0],
        Some(40.0)
    );
    assert_eq!(
        floats(r#"{"e":"seq","fn":"mean_quality","input":{"e":"col","name":"qual"},"offset":64}"#)
            [0],
        Some(9.0)
    );
    // Q0, Q20, Q40 -> 1 + 0.01 + 0.0001 expected errors.
    let ee = floats(r#"{"e":"seq","fn":"expected_errors","input":{"e":"col","name":"qual"}}"#)[1]
        .unwrap();
    assert!((ee - 1.0101).abs() < 1e-9, "{ee}");
}

/// The `alphabet` shape.
#[test]
fn the_alphabet_argument_survives_the_wire() {
    let a =
        eval(r#"{"e":"seq","fn":"is_valid","input":{"e":"col","name":"dna"},"alphabet":"dna"}"#);
    let b = a.as_any().downcast_ref::<BooleanArray>().expect("Boolean");
    assert_eq!(
        (0..5)
            .map(|i| (!b.is_null(i)).then(|| b.value(i)))
            .collect::<Vec<_>>(),
        vec![Some(true), Some(true), Some(false), Some(true), None]
    );
    let mw = floats(
        r#"{"e":"seq","fn":"molecular_weight","input":{"e":"col","name":"protein"},"alphabet":"protein"}"#,
    );
    assert!(mw[0].unwrap() > 400.0, "{mw:?}");
    assert_eq!(mw[4], None);
}

/// The `pattern` shape, and the ambiguity rule that distinguishes a motif search from a
/// substring search.
#[test]
fn the_motif_argument_survives_the_wire() {
    assert_eq!(
        ints(r#"{"e":"seq","fn":"count_motif","input":{"e":"col","name":"dna"},"pattern":"GG"}"#),
        vec![Some(1), Some(1), Some(0), Some(0), None]
    );
    // A degenerate motif matches every base it stands for, on both sides.
    assert_eq!(
        ints(r#"{"e":"seq","fn":"count_motif","input":{"e":"col","name":"dna"},"pattern":"NNN"}"#)
            [2],
        Some(3)
    );
    let pos =
        eval(r#"{"e":"seq","fn":"find_motif","input":{"e":"col","name":"dna"},"pattern":"GG"}"#);
    let l = pos.as_any().downcast_ref::<ListArray>().expect("List");
    let v = l.value(0);
    assert_eq!(
        v.as_any().downcast_ref::<Int64Array>().unwrap().value(0),
        3,
        "GG starts at 1-based position 3 of ATGGCCTAA"
    );
}

/// Physical properties, and the deliberate nulls that keep them from reporting a number the
/// data does not support.
#[test]
fn physical_properties_reach_their_kernels() {
    let tm = floats(r#"{"e":"seq","fn":"melting_temp","input":{"e":"col","name":"dna"}}"#);
    assert!(tm[0].is_some(), "pure ACGT has a melting temperature");
    assert_eq!(tm[2], None, "an N has no defined stacking energy");
    let g = floats(r#"{"e":"seq","fn":"gravy","input":{"e":"col","name":"protein"}}"#);
    assert!(
        g[2].unwrap() > 3.0,
        "IIIVVVLLL is strongly hydrophobic: {g:?}"
    );
    let pi = floats(r#"{"e":"seq","fn":"isoelectric_point","input":{"e":"col","name":"protein"}}"#);
    assert!(pi[1].unwrap() > 9.0, "poly-lysine is basic: {pi:?}");
}

/// A caller error is an error, not a column of plausible answers.
#[test]
fn a_bad_argument_fails_the_batch_rather_than_nulling_it() {
    for (json, expect) in [
        (
            r#"{"e":"seq","fn":"kmers","input":{"e":"col","name":"dna"},"k":0}"#,
            "k must be in",
        ),
        (
            r#"{"e":"seq","fn":"translate","input":{"e":"col","name":"dna"},"frame":3}"#,
            "frame must be",
        ),
        (
            r#"{"e":"seq","fn":"count_motif","input":{"e":"col","name":"dna"},"pattern":"AC-GT"}"#,
            "IUPAC",
        ),
        (
            r#"{"e":"seq","fn":"is_valid","input":{"e":"col","name":"dna"},"alphabet":"peptide"}"#,
            "alphabet must be",
        ),
    ] {
        let e: Expr = serde_json::from_str(json).unwrap_or_else(|err| panic!("{json}: {err}"));
        let err = e.eval(&batch()).expect_err(json).to_string();
        assert!(err.contains(expect), "{json}: {err}");
    }
}

/// An unknown `fn` is rejected at deserialization, so a control-plane typo cannot reach a
/// kernel and be silently ignored.
#[test]
fn an_unknown_function_name_does_not_deserialize() {
    let bad = r#"{"e":"seq","fn":"reverse_compliment","input":{"e":"col","name":"dna"}}"#;
    assert!(serde_json::from_str::<Expr>(bad).is_err());
}
