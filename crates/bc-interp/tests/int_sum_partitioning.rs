//! An integer `SUM`'s success must not depend on how its input is partitioned (F212).
//!
//! Every executor here splits the same relation differently -- the sequential oracle folds
//! one partial per input batch, the parallel executor one per 512-row morsel, the spilling
//! aggregate grace-partitions those partials to disk, and the streaming executors fold shards.
//! The relation is built so that one early morsel holds `i64::MAX` and `1` for group 0 and a
//! late one holds `-2`: the true total, `i64::MAX - 1`, fits, but that early partial's own
//! total does not. Before the integer-`SUM` partial state became an exact 128-bit total
//! (`bc_runtime::agg::int_sum`), the partitioned paths raised `SumOverflow` there while the
//! sequential oracle, which sees both rows in one partial, answered.
//!
//! 200,000 rows is above `MIN_ROWS_TO_SHARD` (4 morsels), so `execute_streaming_parallel`
//! really shards rather than falling back to the sequential path; `spill_is_taken` checks
//! that the spilling run actually spilled.

use arrow::array::{Array, ArrayRef, AsArray, Int64Array};
use arrow::datatypes::{DataType, Int64Type};
use arrow::record_batch::RecordBatch;
use bc_interp::par::SpillOptions;
use bc_interp::{
    execute, execute_parallel_with, execute_parallel_with_metrics, execute_streaming,
    execute_streaming_parallel, ExecOptions, InterpError, SpillCodec,
};
use bc_ir::RelOp;
use std::sync::Arc;

const N: usize = 200_000;
const GROUPS: usize = 100;

fn plan(grouped: bool) -> RelOp {
    let keys = if grouped {
        r#"[{"expr":{"e":"col","name":"k"},"alias":"k"}]"#
    } else {
        "[]"
    };
    serde_json::from_str(&format!(
        r#"{{"op":"aggregate","input":{{"op":"scan","source_id":0}},"group_keys":{keys},
            "aggregates":[{{"func":"sum","input":{{"e":"col","name":"v"}},"alias":"s"}}]}}"#
    ))
    .expect("plan JSON")
}

/// `k = i % 100`. Group 0 is zeros except `i64::MAX` at row 0, `1` at row 100 (same first
/// morsel) and `last` at row `N - 100` (the last morsel). Group 1 is all NULL. Every other
/// group holds `-(i % 7)`, so the global total is group 0's minus a small amount -- it fits
/// too, and its partials overflow the same way.
fn source(last: i64) -> Vec<Vec<RecordBatch>> {
    let k: ArrayRef = Arc::new(Int64Array::from(
        (0..N).map(|i| (i % GROUPS) as i64).collect::<Vec<_>>(),
    ));
    let v: ArrayRef = Arc::new(Int64Array::from(
        (0..N)
            .map(|i| match (i, i % GROUPS) {
                (0, _) => Some(i64::MAX),
                (100, _) => Some(1),
                (i, _) if i == N - 100 => Some(last),
                (_, 0) => Some(0),
                (_, 1) => None,
                _ => Some(-((i % 7) as i64)),
            })
            .collect::<Vec<_>>(),
    ));
    // Two input batches, so the sequential oracle folds more than one partial too -- but both
    // overflow-relevant rows of the early morsel stay together in the first.
    let b = RecordBatch::try_from_iter(vec![("k", k), ("v", v)]).expect("batch");
    vec![vec![b.slice(0, N / 2), b.slice(N / 2, N - N / 2)]]
}

fn spill_opts() -> ExecOptions {
    ExecOptions {
        morsel_rows: 512,
        parallelism: 4,
        agg_spill: Some(SpillOptions {
            memory_budget_bytes: 1024,
            dir: std::env::temp_dir(),
            codec: SpillCodec::None,
        }),
        ..ExecOptions::default()
    }
}

/// Every executor's answer, labelled.
fn every_path(
    p: &RelOp,
    s: &[Vec<RecordBatch>],
) -> Vec<(&'static str, Result<Vec<RecordBatch>, InterpError>)> {
    let par = ExecOptions {
        morsel_rows: 512,
        parallelism: 4,
        ..ExecOptions::default()
    };
    vec![
        ("sequential", execute(p, s)),
        ("parallel", execute_parallel_with(p, s, &par)),
        (
            "parallel-spilled",
            execute_parallel_with(p, s, &spill_opts()),
        ),
        ("streaming", execute_streaming(p, s, 0)),
        (
            "streaming-parallel",
            execute_streaming_parallel(p, s, 4, 0, None),
        ),
    ]
}

/// `(key, sum)` rows, sorted; `key` is None for the global aggregate.
fn rows(batches: &[RecordBatch]) -> Vec<(Option<i64>, Option<i64>)> {
    let mut out = Vec::new();
    for b in batches {
        let s = b.column(b.num_columns() - 1);
        assert_eq!(
            s.data_type(),
            &DataType::Int64,
            "SUM(BIGINT) returns BIGINT"
        );
        let s = s.as_primitive::<Int64Type>();
        for r in 0..b.num_rows() {
            let k =
                (b.num_columns() == 2).then(|| b.column(0).as_primitive::<Int64Type>().value(r));
            out.push((k, s.is_valid(r).then(|| s.value(r))));
        }
    }
    out.sort();
    out
}

/// The exact answer, computed in `i128` over the same rows.
fn expected(grouped: bool, last: i64) -> Vec<(Option<i64>, Option<i64>)> {
    let s = source(last);
    let mut sums = vec![(0i128, false); GROUPS];
    for b in &s[0] {
        let (k, v) = (
            b.column(0).as_primitive::<Int64Type>(),
            b.column(1).as_primitive::<Int64Type>(),
        );
        for r in 0..b.num_rows() {
            if v.is_valid(r) {
                let g = &mut sums[k.value(r) as usize];
                g.0 += i128::from(v.value(r));
                g.1 = true;
            }
        }
    }
    let narrow = |t: i128| i64::try_from(t).expect("the fixture's totals fit");
    if grouped {
        (0..GROUPS)
            .map(|g| (Some(g as i64), sums[g].1.then(|| narrow(sums[g].0))))
            .collect()
    } else {
        vec![(None, Some(narrow(sums.iter().map(|g| g.0).sum())))]
    }
}

#[test]
fn a_total_that_fits_succeeds_on_every_executor() {
    for grouped in [true, false] {
        let p = plan(grouped);
        let want = expected(grouped, -2);
        if grouped {
            assert_eq!(want[0], (Some(0), Some(i64::MAX - 1)));
            assert_eq!(want[1], (Some(1), None), "the all-null group is NULL");
        }
        for (label, got) in every_path(&p, &source(-2)) {
            let got = got.unwrap_or_else(|e| panic!("{label} (grouped={grouped}) raised {e:?}"));
            assert_eq!(rows(&got), want, "{label} (grouped={grouped})");
        }
    }
}

#[test]
fn a_true_overflow_raises_on_every_executor() {
    // `last = 10^9` puts group 0 at `i64::MAX + 1 + 10^9`, and the global total past `i64` by
    // more than the ~600k the negative filler takes back.
    for grouped in [true, false] {
        let p = plan(grouped);
        for (label, got) in every_path(&p, &source(1_000_000_000)) {
            assert!(
                matches!(
                    got,
                    Err(InterpError::Runtime(bc_runtime::RuntimeError::SumOverflow))
                ),
                "{label} (grouped={grouped}) must raise SumOverflow, got {:?}",
                got.map(|b| rows(&b))
            );
        }
    }
}

/// The spilled run must actually spill, or `parallel-spilled` above is a second in-memory run.
#[test]
fn spill_is_taken() {
    let (_, m) = execute_parallel_with_metrics(&plan(true), &source(-2), &spill_opts()).unwrap();
    let agg = m
        .ops
        .iter()
        .find(|o| o.kind == "aggregate")
        .expect("aggregate metric");
    assert!(
        agg.spilled,
        "a 1 KiB budget over 100 groups x 390 morsels must spill"
    );
}
