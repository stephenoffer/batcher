//! The key-disjoint-run aggregate == `execute`, the sequential oracle.
//!
//! `agg_par::key_disjoint_runs` lets the materializing executor skip the partition gather when
//! the group key arrives ordered: runs of morsels whose key ranges do not overlap are already
//! key-disjoint, so each one's partial is final and the runs finalize independently. That is a
//! *scheduling* claim about a `GROUP BY`, and the thing that licenses it is producing the same
//! relation the oracle does — including the case where the licence is wrong to grant.
//!
//! **Why this needs its own file.** The path is only taken when at least one run per worker can
//! be cut, so on a 96-core box it needs tens of thousands of ordered rows and never fires in the
//! differential suite's small in-memory tables. Pinning `parallelism` is what brings it into
//! reach of a test, and asserting the recorded backend is what keeps the test honest: without
//! that assertion this file would keep passing if the path stopped being taken at all.

use arrow::array::{ArrayRef, Int64Array};
use arrow::record_batch::RecordBatch;
use bc_interp::{execute, execute_parallel_with_metrics, ExecOptions};
use bc_ir::RelOp;
use std::sync::Arc;

/// Four rows per key, so most morsel edges fall *inside* a group and the cuts have to be found.
const PER_KEY: i64 = 4;
const ROWS: i64 = 200_000;

fn plan() -> RelOp {
    serde_json::from_str(
        r#"{"op":"aggregate",
            "input":{"op":"scan","source_id":0},
            "group_keys":[{"expr":{"e":"col","name":"k"},"alias":"k"}],
            "aggregates":[
              {"func":"sum","input":{"e":"col","name":"v"},"alias":"s"},
              {"func":"min","input":{"e":"col","name":"v"},"alias":"lo"},
              {"func":"count_star","alias":"n"}]}"#,
    )
    .expect("plan JSON")
}

/// `ordered`: keys ascend across the whole relation, so runs can be cut.
/// Otherwise the key cycles, every morsel spans the whole range, and no cut is legal.
fn source(ordered: bool) -> Vec<Vec<RecordBatch>> {
    let groups = ROWS / PER_KEY;
    let k: ArrayRef = Arc::new(Int64Array::from(
        (0..ROWS)
            .map(|i| if ordered { i / PER_KEY } else { i % groups })
            .collect::<Vec<_>>(),
    ));
    let v: ArrayRef = Arc::new(Int64Array::from((0..ROWS).collect::<Vec<_>>()));
    vec![vec![
        RecordBatch::try_from_iter(vec![("k", k), ("v", v)]).expect("batch")
    ]]
}

/// Small morsels and a pinned width, so a few hundred thousand rows produce far more runs than
/// workers — the condition `key_disjoint_runs` is gated on.
fn opts() -> ExecOptions {
    ExecOptions {
        morsel_rows: 512,
        parallelism: 4,
        ..ExecOptions::default()
    }
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

fn run(ordered: bool) -> (Vec<String>, Vec<String>, String) {
    let p = plan();
    let sources = source(ordered);
    let want = execute(&p, &sources).expect("oracle");
    let (got, m) = execute_parallel_with_metrics(&p, &sources, &opts()).expect("parallel");
    let backend = m
        .ops
        .iter()
        .find(|o| o.kind == "aggregate")
        .map(|o| o.backend.to_string())
        .expect("an aggregate metric");
    let (mut a, mut b) = (rows(&got), rows(&want));
    a.sort();
    b.sort();
    (a, b, backend)
}

/// The whole point: an ordered key takes the run path *and* answers what the oracle answers.
#[test]
fn the_disjoint_run_aggregate_matches_the_sequential_oracle() {
    let (got, want, backend) = run(true);
    assert_eq!(
        backend, "par-agg-disjoint-runs",
        "an ordered key must reach the run path, or this test proves nothing"
    );
    assert_eq!(got.len(), (ROWS / PER_KEY) as usize, "one row per key");
    assert_eq!(
        got, want,
        "the run path diverged from the sequential oracle"
    );
}

/// And a key that is *not* ordered must not reach it — the runs would split a group in two and
/// emit it twice. This is the assertion that makes the one above mean something.
#[test]
fn an_unordered_key_does_not_reach_the_run_path() {
    let (got, want, backend) = run(false);
    assert_ne!(
        backend, "par-agg-disjoint-runs",
        "a cycling key has no legal cut and must not be run-partitioned"
    );
    assert_eq!(got, want, "the fallback path diverged from the oracle");
}
