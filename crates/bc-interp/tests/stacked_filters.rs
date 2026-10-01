//! A filter stacked on a filter gathers its rows once — `par::run_chain` and the streaming
//! `pipeline::filter_stream` both carry the inner mask to the outer filter — and must still
//! compute exactly what the sequential oracle computes —
//! including the one thing that could go wrong: evaluating the outer predicate at a row the
//! inner one removed. A predicate that can *fail* on such a row must never see it.
//!
//! Plans are JSON IR, so the wire shape is what is tested. The fixture is larger than
//! `MIN_ROWS_TO_SHARD` (65,536 rows) so the parallel executor genuinely shards.

use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, StringArray};
use arrow::record_batch::RecordBatch;
use bc_interp::{
    execute, execute_parallel, execute_parallel_with, execute_parallel_with_metrics,
    execute_streaming, execute_streaming_metered, execute_streaming_parallel, ExecOptions,
};
use bc_ir::RelOp;

const N: i64 = 200_000;

/// `k` cycles 0..100 with every 13th row null, `v` counts up, `s` is a number spelled as text
/// except on every 9th row, where it is `"bad"` — which `CAST(s AS BIGINT)` rejects.
fn facts() -> Vec<RecordBatch> {
    let k: ArrayRef = Arc::new(Int64Array::from(
        (0..N)
            .map(|i| (i % 13 != 0).then_some(i % 100))
            .collect::<Vec<_>>(),
    ));
    let v: ArrayRef = Arc::new(Int64Array::from((0..N).collect::<Vec<_>>()));
    let s: ArrayRef = Arc::new(StringArray::from(
        (0..N)
            .map(|i| {
                if i % 9 == 0 {
                    "bad".to_string()
                } else {
                    (i % 50).to_string()
                }
            })
            .collect::<Vec<_>>(),
    ));
    vec![RecordBatch::try_from_iter(vec![("k", k), ("v", v), ("s", s)]).unwrap()]
}

fn plan(json: &str) -> RelOp {
    serde_json::from_str(json).unwrap_or_else(|e| panic!("bad plan JSON: {e}\n{json}"))
}

fn rows(batches: &[RecordBatch]) -> Vec<String> {
    let mut out = Vec::new();
    for b in batches {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = (0..b.num_columns())
                .map(|c| arrow::util::display::array_value_to_string(b.column(c), r).unwrap())
                .collect();
            out.push(cells.join("|"));
        }
    }
    out
}

fn col(name: &str) -> String {
    format!(r#"{{"e":"col","name":"{name}"}}"#)
}

fn cmp(op: &str, name: &str, v: i64) -> String {
    format!(
        r#"{{"e":"binary","op":"{op}","left":{},"right":{{"e":"lit","value":{{"int":{v}}}}}}}"#,
        col(name)
    )
}

fn filter(input: &str, predicate: &str) -> String {
    format!(r#"{{"op":"filter","input":{input},"predicate":{predicate}}}"#)
}

const SCAN: &str = r#"{"op":"scan","source_id":0}"#;

/// The parallel executor as the control plane runs it: with linear-chain fusion on
/// (`ExecutionConfig.fuse_linear` defaults to true), which is the path `run_chain` serves.
/// `ExecOptions::default()` leaves it off, and a test using that never reaches the code.
fn fused() -> ExecOptions {
    ExecOptions {
        fuse_linear: true,
        ..ExecOptions::default()
    }
}

/// Same rows in the same order as the oracle, from the parallel executor and both streaming
/// executors (the sequential one, and the one that shards the scan across four workers).
fn assert_matches_oracle(json: &str) {
    let p = plan(json);
    let src = [facts()];
    let want = rows(&execute(&p, &src).expect("oracle"));
    let runs = [
        ("parallel", execute_parallel(&p, &src).expect("parallel")),
        (
            "parallel-fused",
            execute_parallel_with(&p, &src, &fused()).expect("parallel-fused"),
        ),
        (
            "streaming",
            execute_streaming(&p, &src, 0).expect("streaming"),
        ),
        (
            "streaming-parallel",
            execute_streaming_parallel(&p, &src, 4, 0, None).expect("streaming-parallel"),
        ),
    ];
    for (label, got) in runs {
        assert_eq!(rows(&got), want, "{label} diverged for:\n{json}");
    }
}

#[test]
fn stacked_filters_match_the_oracle_at_every_selectivity() {
    // Inner keeps most / about half / few rows; outer selective or not; nulls in `k`.
    for (inner, outer) in [
        (cmp("lt", "k", 95), cmp("ge", "v", 150_000)),
        (cmp("lt", "k", 50), cmp("ne", "k", 7)),
        (cmp("lt", "k", 2), cmp("gt", "v", 10)),
        (cmp("ge", "k", 0), cmp("lt", "k", 0)),
    ] {
        assert_matches_oracle(&filter(&filter(SCAN, &inner), &outer));
    }
    // Three deep, and with a range pair in the outer predicate.
    let range = format!(
        r#"{{"e":"binary","op":"and","left":{},"right":{}}}"#,
        cmp("ge", "v", 1_000),
        cmp("lt", "v", 190_000)
    );
    let three = filter(
        &filter(&filter(SCAN, &cmp("lt", "k", 90)), &range),
        &cmp("ne", "k", 3),
    );
    assert_matches_oracle(&three);
}

/// `CAST(s AS BIGINT) > 10` fails on `"bad"`, and `s <> 'bad'` beneath it removes exactly those
/// rows. The oracle filters first and never sees one; deferring the inner filter's gather must
/// not hand the cast a `"bad"` row either.
#[test]
fn an_outer_predicate_that_can_fail_never_sees_a_removed_row() {
    let not_bad = format!(
        r#"{{"e":"binary","op":"ne","left":{},"right":{{"e":"lit","value":{{"str":"bad"}}}}}}"#,
        col("s")
    );
    let cast_gt = format!(
        r#"{{"e":"binary","op":"gt","left":{{"e":"cast","input":{},"dtype":"int64","try_cast":false}},
            "right":{{"e":"lit","value":{{"int":10}}}}}}"#,
        col("s")
    );
    let json = filter(&filter(SCAN, &not_bad), &cast_gt);
    assert_matches_oracle(&json);
    // And the cast really does fail on the rows the inner filter removes, so the test above is
    // not passing for want of a failure to provoke.
    let unguarded = filter(SCAN, &cast_gt);
    assert!(execute(&plan(&unguarded), &[facts()]).is_err());
}

/// Each stacked filter still reports its own selectivity — the inner one's rows out are the
/// count its mask keeps, not the count after the outer one.
#[test]
fn each_stacked_filter_reports_its_own_row_count() {
    let json = filter(&filter(SCAN, &cmp("lt", "k", 50)), &cmp("ge", "v", 100_000));
    let (_, par) =
        execute_parallel_with_metrics(&plan(&json), &[facts()], &fused()).expect("parallel");
    let (_, streamed) = execute_streaming_metered(&plan(&json), &[facts()], 0).expect("stream");
    let count = |json: &str| -> u64 {
        execute(&plan(json), &[facts()])
            .unwrap()
            .iter()
            .map(|b| b.num_rows() as u64)
            .sum()
    };
    let inner_out = count(&filter(SCAN, &cmp("lt", "k", 50)));
    let outer_out = count(&json);
    for metrics in [par, streamed] {
        let filters: Vec<(u64, u64)> = metrics
            .ops
            .iter()
            .filter(|m| m.kind == "filter")
            .map(|m| (m.rows_in, m.rows_out))
            .collect();
        assert!(
            filters.contains(&(N as u64, inner_out)),
            "inner filter metrics {filters:?}, want ({N}, {inner_out})"
        );
        assert!(
            filters.contains(&(inner_out, outer_out)),
            "outer filter metrics {filters:?}, want ({inner_out}, {outer_out})"
        );
    }
}

/// Stacked filters feeding an aggregate run through the fused filter→aggregate path.
#[test]
fn stacked_filters_under_an_aggregate_match_the_oracle() {
    let json = format!(
        r#"{{"op":"aggregate","input":{},"group_keys":[{{"expr":{},"alias":"k"}}],
            "aggregates":[{{"func":"sum","input":{},"alias":"sv"}},{{"func":"count_star","alias":"n"}}]}}"#,
        filter(&filter(SCAN, &cmp("lt", "k", 80)), &cmp("gt", "v", 5_000)),
        col("k"),
        col("v")
    );
    let p = plan(&json);
    let src = [facts()];
    let mut want = rows(&execute(&p, &src).expect("oracle"));
    want.sort();
    assert!(!want.is_empty());
    for got in [
        execute_parallel(&p, &src).expect("parallel"),
        execute_parallel_with(&p, &src, &fused()).expect("parallel-fused"),
        execute_streaming_parallel(&p, &src, 4, 0, None).expect("streaming-parallel"),
    ] {
        let mut got = rows(&got);
        got.sort();
        assert_eq!(got, want);
    }
}
