//! The streaming executor's sideways prepass returns what the sequential oracle returns.
//!
//! With `ExecOptions::prefer_sideways`, a join whose build side is `Aggregate(scan, by k)` has its
//! probe side evaluated first and the aggregate's source restricted to the probe keys
//! (`stream::builds::build_sideways`). That is a wrong-answer-shaped change in two ways — the
//! restriction could drop a group a probe row needs, and the stored probe side could be read once
//! per worker if a sharded spine ran past it — so every join type it serves is held to the oracle
//! with NULL keys, keys absent from either side, and a probe spine large enough to shard.
//!
//! The control that makes the comparison mean something: the restricted build side is not
//! metered (its counts describe this query's restriction, not the subplan), so the aggregate's
//! metric is present with the verdict off and absent with it on. Without that, a prepass that
//! never engaged would pass every equality below.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int64Array};
use arrow::record_batch::RecordBatch;
use bc_interp::{execute, execute_streaming_parallel, execute_streaming_parallel_metered};
use bc_ir::RelOp;

/// The build side's source: past `join_par::sideways`' 262,144-row floor, and at least 4x the
/// probe side, so the restriction is admitted.
const BIG: i64 = 400_000;
/// The probe side: past `MIN_ROWS_TO_SHARD` (65,536), so the spine above the join could shard.
const SMALL: i64 = 80_000;

fn col(name: &str) -> String {
    format!(r#"{{"e":"col","name":"{name}"}}"#)
}

fn batch(cols: Vec<(&str, Vec<Option<i64>>)>) -> RecordBatch {
    let arrays: Vec<(&str, ArrayRef)> = cols
        .into_iter()
        .map(|(n, v)| (n, Arc::new(Int64Array::from(v)) as ArrayRef))
        .collect();
    RecordBatch::try_from_iter(arrays).unwrap()
}

fn sources() -> Vec<Vec<RecordBatch>> {
    // Probe keys are every 7th integer up to 3x the build side's key range, with some NULLs, so
    // part of the probe side matches nothing and part of the build side is never probed.
    let probe = batch(vec![
        (
            "ok",
            (0..SMALL)
                .map(|i| (i % 97 != 0).then_some(i * 7 % (BIG * 3)))
                .collect(),
        ),
        ("p", (0..SMALL).map(Some).collect()),
    ]);
    let build = batch(vec![
        (
            "k",
            (0..BIG)
                .map(|i| (i % 101 != 0).then_some(i % (BIG / 2)))
                .collect(),
        ),
        ("v", (0..BIG).map(|i| Some(i % 1_000)).collect()),
    ]);
    vec![vec![probe], vec![build]]
}

fn plan(join_type: &str) -> RelOp {
    let right_out = if matches!(join_type, "semi" | "anti") {
        String::new()
    } else {
        r#",{"side":"right","name":"s","alias":"s"},{"side":"right","name":"n","alias":"n"}"#
            .to_string()
    };
    let json = format!(
        r#"{{"op":"hash_join",
            "left":{{"op":"scan","source_id":0}},
            "right":{{"op":"aggregate","input":{{"op":"scan","source_id":1}},
                      "group_keys":[{{"expr":{},"alias":"gk"}}],
                      "aggregates":[{{"func":"sum","input":{},"alias":"s"}},
                                    {{"func":"count_star","alias":"n"}}]}},
            "left_keys":["ok"],"right_keys":["gk"],"join_type":"{join_type}",
            "output":[{{"side":"left","name":"ok","alias":"ok"}},
                      {{"side":"left","name":"p","alias":"p"}}{right_out}],
            "strategy":"hash"}}"#,
        col("k"),
        col("v")
    );
    serde_json::from_str(&json).unwrap_or_else(|e| panic!("bad plan JSON: {e}\n{json}"))
}

/// The relation as sorted rows of optional values, for an order-independent comparison.
fn rows(batches: &[RecordBatch]) -> Vec<Vec<Option<i64>>> {
    let mut out = Vec::new();
    for b in batches {
        for r in 0..b.num_rows() {
            out.push(
                b.columns()
                    .iter()
                    .map(|c| {
                        let a = c.as_any().downcast_ref::<Int64Array>().unwrap();
                        a.is_valid(r).then(|| a.value(r))
                    })
                    .collect(),
            );
        }
    }
    out.sort();
    out
}

fn sideways(on: bool) -> bc_interp::ExecOptions {
    bc_interp::ExecOptions {
        prefer_sideways: on,
        ..bc_interp::ExecOptions::default()
    }
}

#[test]
fn every_join_type_matches_the_oracle_with_the_prepass_on() {
    let src = sources();
    for jt in ["inner", "left", "semi", "anti"] {
        let p = plan(jt);
        let want = rows(&execute(&p, &src).unwrap());
        assert!(!want.is_empty(), "{jt}: the fixture must produce rows");
        for workers in [1, 8] {
            let got =
                execute_streaming_parallel(&p, &src, workers, 0, Some(&sideways(true))).unwrap();
            assert_eq!(rows(&got), want, "{jt} with {workers} workers");
        }
    }
}

#[test]
fn the_prepass_engages_only_on_the_verdict() {
    let src = sources();
    let p = plan("left");
    let aggregate_metered = |on: bool| {
        let (_, metrics) =
            execute_streaming_parallel_metered(&p, &src, 8, 0, Some(&sideways(on))).unwrap();
        metrics.ops.iter().any(|m| m.kind == "aggregate")
    };
    assert!(
        aggregate_metered(false),
        "positive control: the aggregate is metered normally"
    );
    assert!(
        !aggregate_metered(true),
        "with the verdict the build side ran restricted, which is not metered"
    );
}
