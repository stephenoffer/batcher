//! The streaming executor's metrics must agree with the oracle's — they feed the learned loop.
//!
//! These are not telemetry. Kyber calibrates its cost coefficients from `rows_in` and learns
//! cardinalities from `rows_out`; Carbonite fits its per-family memory model on `peak_bytes` and
//! sizes admission, spill routing and buffer reservation from it. A streaming executor that
//! reported *plausible-looking rubbish* would not fail loudly — it would quietly teach the
//! optimizer wrong things, and every subsequent plan would be a little worse for reasons nobody
//! could trace back here.
//!
//! So the row counts are pinned against `execute_metered`, per `op_id`, per operator kind.
//!
//! `peak_bytes` deliberately does **not** have to match, and that is the point of the streaming
//! tier: a streamed join's peak excludes the probe side it no longer materializes, and a folded
//! aggregate's excludes the input it no longer holds. Those are smaller and *truer* numbers.
//! Asserting they match would be asserting the executor did not work.

use arrow::array::{ArrayRef, Int64Array};
use arrow::record_batch::RecordBatch;
use bc_interp::{execute_metered, execute_streaming_metered, execute_streaming_parallel_metered};
use bc_ir::RelOp;
use std::collections::HashMap;
use std::sync::Arc;

const N: i64 = 200_000;

fn plan(json: &str) -> RelOp {
    serde_json::from_str(json).unwrap()
}

fn facts() -> Vec<RecordBatch> {
    let k: ArrayRef = Arc::new(Int64Array::from(
        (0..N).map(|i| i % 100).collect::<Vec<_>>(),
    ));
    let v: ArrayRef = Arc::new(Int64Array::from((0..N).collect::<Vec<_>>()));
    vec![RecordBatch::try_from_iter(vec![("k", k), ("v", v)]).unwrap()]
}

fn dim() -> Vec<RecordBatch> {
    let k: ArrayRef = Arc::new(Int64Array::from((0..100i64).collect::<Vec<_>>()));
    let d: ArrayRef = Arc::new(Int64Array::from(
        (0..100i64).map(|i| i * 10).collect::<Vec<_>>(),
    ));
    vec![RecordBatch::try_from_iter(vec![("k", k), ("d", d)]).unwrap()]
}

/// `op_id -> (kind, rows_in, rows_build, rows_out)`.
type Shape = HashMap<u32, (String, u64, u64, u64)>;

fn shape(m: &bc_interp::ExecMetrics) -> Shape {
    m.ops
        .iter()
        .map(|o| {
            (
                o.op_id,
                (o.kind.to_string(), o.rows_in, o.rows_build, o.rows_out),
            )
        })
        .collect()
}

/// Every operator the streaming executor reports must carry the oracle's numbers, under the
/// oracle's `op_id` and the oracle's `kind`.
///
/// Only operators the streaming tier *ran itself* are compared: a subtree it defers to the oracle
/// (`Distinct`, `Window`, …) reports nothing here by design, and reporting nothing is honest —
/// reporting a fabricated zero-row operator would not be.
fn assert_metrics_agree(json: &str, sources: &[Vec<RecordBatch>]) {
    let p = plan(json);
    let (_, want) = execute_metered(&p, sources).expect("oracle");
    let want = shape(&want);

    for (label, got) in [
        (
            "streaming",
            execute_streaming_metered(&p, sources, 0)
                .expect("streaming")
                .1,
        ),
        (
            "streaming-parallel",
            execute_streaming_parallel_metered(&p, sources, 4, 0)
                .expect("streaming-parallel")
                .1,
        ),
    ] {
        let got = shape(&got);
        assert!(!got.is_empty(), "{label} reported no metrics at all");
        for (op_id, (kind, rows_in, rows_build, rows_out)) in &got {
            let Some((w_kind, w_in, w_build, w_out)) = want.get(op_id) else {
                panic!("{label}: op {op_id} ({kind}) is not an operator the oracle reported");
            };
            assert_eq!(kind, w_kind, "{label}: op {op_id} kind");
            assert_eq!(
                rows_out, w_out,
                "{label}: op {op_id} ({kind}) rows_out — this is the learned cardinality"
            );
            assert_eq!(
                rows_in, w_in,
                "{label}: op {op_id} ({kind}) rows_in — Kyber's selectivity denominator"
            );
            assert_eq!(
                rows_build, w_build,
                "{label}: op {op_id} ({kind}) rows_build — the join's build-side cardinality"
            );
        }
    }
}

const SCAN: &str = r#"{"op":"scan","source_id":0}"#;

#[test]
fn a_linear_pipeline_reports_the_oracles_rows() {
    let json = format!(
        r#"{{"op":"project","input":{{"op":"filter","input":{SCAN},
            "predicate":{{"e":"binary","op":"lt","left":{{"e":"col","name":"k"}},
                          "right":{{"e":"lit","value":{{"int":40}}}}}}}},
            "exprs":[{{"expr":{{"e":"col","name":"v"}},"alias":"v"}}]}}"#
    );
    assert_metrics_agree(&json, &[facts()]);
}

#[test]
fn a_folded_aggregate_reports_the_oracles_rows() {
    // The aggregate never holds its input now, but it must still report how many rows went in —
    // a zero there would not read as "missing", it would read as "perfectly selective".
    let json = format!(
        r#"{{"op":"aggregate","input":{SCAN},
            "group_keys":[{{"expr":{{"e":"col","name":"k"}},"alias":"k"}}],
            "aggregates":[{{"func":"sum","input":{{"e":"col","name":"v"}},"alias":"sv"}}]}}"#
    );
    assert_metrics_agree(&json, &[facts()]);
}

#[test]
fn a_streamed_join_splits_probe_and_build_rows_the_way_the_contract_requires() {
    // `rows_in` is the probe side, `rows_build` the build side — never their sum. Kyber
    // calibrates the join's asymmetric build and probe coefficients from exactly these two, and
    // `rows_out / rows_in` is its fan-out. The build table is hashed once no matter how many
    // morsels probe it, so `rows_build` must be recorded once too — counting it per morsel would
    // multiply the build cardinality by the morsel count.
    let json = format!(
        r#"{{"op":"hash_join","left":{SCAN},"right":{{"op":"scan","source_id":1}},
            "left_keys":["k"],"right_keys":["k"],"join_type":"inner",
            "output":[{{"side":"left","name":"v","alias":"v"}},
                      {{"side":"right","name":"d","alias":"d"}}],
            "strategy":"hash"}}"#
    );
    assert_metrics_agree(&json, &[facts(), dim()]);
}

#[test]
fn a_join_feeding_an_aggregate_reports_the_oracles_rows() {
    let j = format!(
        r#"{{"op":"hash_join","left":{SCAN},"right":{{"op":"scan","source_id":1}},
            "left_keys":["k"],"right_keys":["k"],"join_type":"inner",
            "output":[{{"side":"left","name":"k","alias":"k"}},
                      {{"side":"left","name":"v","alias":"v"}},
                      {{"side":"right","name":"d","alias":"d"}}],
            "strategy":"hash"}}"#
    );
    let json = format!(
        r#"{{"op":"aggregate","input":{j},
            "group_keys":[{{"expr":{{"e":"col","name":"k"}},"alias":"k"}}],
            "aggregates":[{{"func":"sum","input":{{"e":"col","name":"v"}},"alias":"sv"}}]}}"#
    );
    assert_metrics_agree(&json, &[facts(), dim()]);
}

#[test]
fn the_streaming_aggregates_peak_is_smaller_than_the_materializing_ones() {
    // The metric that legitimately *changes*, and the reason it must not be copied from the
    // oracle: a folded aggregate never holds its input, so its peak is its state plus its result.
    // Carbonite reserves memory from this number — reporting the materializing path's larger one
    // would make it provision for a materialization that no longer happens.
    let json = format!(
        r#"{{"op":"aggregate","input":{SCAN},
            "group_keys":[{{"expr":{{"e":"col","name":"k"}},"alias":"k"}}],
            "aggregates":[{{"func":"sum","input":{{"e":"col","name":"v"}},"alias":"sv"}}]}}"#
    );
    let p = plan(&json);
    let (_, oracle) = execute_metered(&p, &[facts()]).unwrap();
    let (_, streamed) = execute_streaming_metered(&p, &[facts()], 0).unwrap();

    let peak = |m: &bc_interp::ExecMetrics| {
        m.ops
            .iter()
            .find(|o| o.kind == "aggregate")
            .expect("an aggregate metric")
            .peak_bytes
    };
    assert!(
        peak(&streamed) < peak(&oracle),
        "the streamed aggregate should hold less than the materializing one: {} vs {}",
        peak(&streamed),
        peak(&oracle)
    );
}

/// The `backend` tag must say whether Tier-1 actually ran, on the tier that is the default.
///
/// This tier reported a hardcoded `"interp"` for every operator. That was right for filter and
/// project — they stay on the interpreter here on purpose — and wrong for the aggregate, which
/// compiles its computed group keys and inputs. TPC-H q1's
/// `sum(l_extendedprice * (1 - l_discount))` is compiled and was reported as `interp`, so the
/// one column a user profiling a query reads to see whether the JIT fired could never say yes.
#[test]
fn a_computed_aggregate_reports_the_jit_and_a_bare_one_does_not() {
    // `sum(v * 2)` — a computed input, so a JIT candidate.
    let computed = plan(
        r#"{"op":"aggregate","input":{"op":"scan","source_id":0},
            "group_keys":[{"expr":{"e":"col","name":"k"},"alias":"k"}],
            "aggregates":[{"func":"sum","alias":"s","input":
              {"e":"binary","op":"mul","left":{"e":"col","name":"v"},
               "right":{"e":"lit","value":{"int":2}}}}]}"#,
    );
    for (path, m) in [
        (
            "serial",
            execute_streaming_metered(&computed, &[facts()], 0)
                .unwrap()
                .1,
        ),
        (
            // The sharded fold is the arm the engine default reaches, and it has its own
            // aggregate implementation — a tag recorded only on the serial breaker would
            // still read `interp` for every real query.
            "sharded",
            execute_streaming_parallel_metered(&computed, &[facts()], 4, 0)
                .unwrap()
                .1,
        ),
    ] {
        let agg = m.ops.iter().find(|o| o.kind == "aggregate").unwrap();
        assert_eq!(
            agg.backend, "jit",
            "{path}: a compiled aggregate input must not report the interpreter"
        );
    }

    // `sum(v)` — a bare column, which the JIT declines on purpose (the interpreter evaluates it
    // as a zero-copy Arc clone). That is not a fallback, and must still read as `interp`.
    let bare = plan(
        r#"{"op":"aggregate","input":{"op":"scan","source_id":0},
            "group_keys":[{"expr":{"e":"col","name":"k"},"alias":"k"}],
            "aggregates":[{"func":"sum","alias":"s","input":{"e":"col","name":"v"}}]}"#,
    );
    let (_, m) = execute_streaming_metered(&bare, &[facts()], 0).unwrap();
    let agg = m.ops.iter().find(|o| o.kind == "aggregate").unwrap();
    assert_eq!(agg.backend, "interp");
}
