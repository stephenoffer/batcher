//! The `ExecMetrics` wire contract, as the control plane depends on it.
//!
//! Carbonite fits its per-family memory model on `peak_bytes` and drives admission, spill
//! routing, buffer reservation and per-worker sizing from it. Kyber calibrates its cost
//! coefficients from `rows_in`. Both were wrong:
//!
//! * `peak_bytes` recorded the operator's *output* array size, so a 60M-row aggregate over
//!   4 groups reported ~0 peak — a catastrophic under-count for exactly the
//!   cardinality-reducing breakers that spill.
//! * a join's `rows_in` was the *sum* of both sides, so `rows_out / rows_in` was neither
//!   the probe rate nor the fan-out, and one calibrated coefficient conflated the
//!   asymmetric build and probe costs.
//! * `batch_bytes` used `get_array_memory_size()`, which reports the whole parent buffer
//!   for a *sliced* array — so morselizing one 32 MB table into 122 morsels measured 3.9 GB.
//!
//! These assertions are the guard. They are deliberately about *relationships* (peak vs
//! result, probe vs build, bytes vs a known row width) rather than exact byte totals, so
//! they survive an Arrow allocator change.

use arrow::array::{ArrayRef, Int64Array};
use arrow::record_batch::RecordBatch;
use bc_interp::execute_metered;
use bc_ir::RelOp;
use std::sync::Arc;

/// `rows` rows of two Int64 columns = 16 bytes of values per row.
const BYTES_PER_ROW: u64 = 16;

fn batch(rows: i64) -> RecordBatch {
    let k: ArrayRef = Arc::new(Int64Array::from(
        (0..rows).map(|i| i % 4).collect::<Vec<_>>(),
    ));
    let v: ArrayRef = Arc::new(Int64Array::from((0..rows).collect::<Vec<_>>()));
    RecordBatch::try_from_iter(vec![("k", k), ("v", v)]).unwrap()
}

fn plan(json: &str) -> RelOp {
    serde_json::from_str(json).expect("valid IR")
}

fn metric<'a>(m: &'a bc_interp::ExecMetrics, kind: &str) -> &'a bc_interp::OpMetric {
    m.ops
        .iter()
        .find(|o| o.kind == kind)
        .unwrap_or_else(|| panic!("no `{kind}` metric in {:?}", m.ops))
}

#[test]
fn scan_bytes_are_physical_not_the_whole_parent_buffer() {
    // The source is one batch, so there is no slicing to over-count — this pins the
    // absolute scale that the sliced case below is compared against.
    let rows = 10_000i64;
    let (_out, m) = execute_metered(
        &plan(r#"{"op":"scan","source_id":0}"#),
        &[vec![batch(rows)]],
    )
    .expect("scan runs");
    let scan = metric(&m, "scan");
    assert_eq!(scan.rows_out, rows as u64);
    assert_eq!(
        scan.peak_bytes, scan.result_bytes,
        "a scan holds only its result"
    );
    assert_eq!(scan.peak_bytes, rows as u64 * BYTES_PER_ROW);
}

#[test]
fn a_sliced_source_is_not_counted_once_per_slice() {
    // Feeding the same rows as many small batches must measure the same bytes as one big
    // one. `get_array_memory_size()` reported the whole parent buffer per slice, so this
    // used to scale with the *number of morsels*.
    let rows = 10_000i64;
    let one = vec![batch(rows)];
    let whole = batch(rows);
    let sliced: Vec<RecordBatch> = (0..10).map(|i| whole.slice(i * 1000, 1000)).collect();

    let scan_ir = plan(r#"{"op":"scan","source_id":0}"#);
    let (_o, m_one) = execute_metered(&scan_ir, &[one]).unwrap();
    let (_o, m_sliced) = execute_metered(&scan_ir, &[sliced]).unwrap();

    assert_eq!(
        metric(&m_one, "scan").peak_bytes,
        metric(&m_sliced, "scan").peak_bytes,
        "slicing a source must not multiply its measured bytes"
    );
}

#[test]
fn an_aggregate_reports_the_input_it_materialized_not_its_tiny_output() {
    // 10k rows collapsing to 4 groups. The output is a handful of bytes; the operator
    // holds the whole input. Reporting the output as `peak_bytes` under-counted by ~1000x.
    let rows = 10_000i64;
    let ir = plan(
        r#"{"op":"aggregate",
            "input":{"op":"scan","source_id":0},
            "group_keys":[{"expr":{"e":"col","name":"k"},"alias":"k"}],
            "aggregates":[{"func":"sum","input":{"e":"col","name":"v"},"alias":"s"}]}"#,
    );
    let (out, m) = execute_metered(&ir, &[vec![batch(rows)]]).expect("aggregate runs");
    assert_eq!(out.iter().map(|b| b.num_rows()).sum::<usize>(), 4);

    let agg = metric(&m, "aggregate");
    assert_eq!(agg.rows_in, rows as u64);
    assert_eq!(agg.rows_out, 4);
    assert!(
        agg.peak_bytes >= rows as u64 * BYTES_PER_ROW,
        "peak {} must account for the materialized {rows}-row input",
        agg.peak_bytes
    );
    assert!(
        agg.result_bytes * 100 < agg.peak_bytes,
        "result {} should be a tiny fraction of peak {}",
        agg.result_bytes,
        agg.peak_bytes
    );
    assert_eq!(agg.rows_build, 0, "only a join has a build side");
}

#[test]
fn a_join_reports_probe_and_build_rows_separately() {
    let probe = 1_000i64;
    let build = 10i64;
    let ir = plan(
        r#"{"op":"hash_join",
            "left":{"op":"scan","source_id":0},
            "right":{"op":"scan","source_id":1},
            "left_keys":["k"],"right_keys":["k"],
            "join_type":"inner",
            "output":[{"side":"left","name":"v","alias":"lv"},
                      {"side":"right","name":"v","alias":"rv"}]}"#,
    );
    let (_out, m) = execute_metered(&ir, &[vec![batch(probe)], vec![batch(build)]]).expect("join");

    let join = metric(&m, "hash_join");
    assert_eq!(
        join.rows_in, probe as u64,
        "rows_in is the probe side alone"
    );
    assert_eq!(
        join.rows_build, build as u64,
        "the build side is reported separately"
    );
    assert_ne!(
        join.rows_in,
        (probe + build) as u64,
        "summing both sides made `selectivity` meaningless"
    );
    assert!(
        join.peak_bytes >= (probe + build) as u64 * BYTES_PER_ROW,
        "a batch join materializes both inputs, so its peak accounts for both"
    );
}

#[test]
fn a_fixed_count_sample_is_a_breaker_not_a_streaming_op() {
    // `n`-smallest-hash sampling scans the whole input to pick the k winners, so its peak
    // is the materialized input — not the tiny sampled output a streaming metric would log.
    let rows = 10_000i64;
    let ir = plan(
        r#"{"op":"sample","input":{"op":"scan","source_id":0},
            "fraction":1.0,"seed":0,"n":10}"#,
    );
    let (out, m) = execute_metered(&ir, &[vec![batch(rows)]]).expect("sample runs");
    assert_eq!(out.iter().map(|b| b.num_rows()).sum::<usize>(), 10);
    let s = metric(&m, "sample");
    assert!(
        s.peak_bytes >= rows as u64 * BYTES_PER_ROW,
        "a fixed-count sample holds its {rows}-row input; peak {} under-counts it",
        s.peak_bytes
    );
    assert!(
        s.result_bytes * 100 < s.peak_bytes,
        "the sampled output is tiny vs the input"
    );
}

#[test]
fn a_deduplicating_union_is_a_breaker() {
    // UNION (DISTINCT) materializes + hashes its inputs; its peak is that input, not the
    // deduped result. UNION ALL, by contrast, streams (covered by the streaming test).
    let rows = 10_000i64;
    let ir = plan(
        r#"{"op":"union","distinct":true,
            "inputs":[{"op":"scan","source_id":0},{"op":"scan","source_id":1}]}"#,
    );
    let (_out, m) = execute_metered(&ir, &[vec![batch(rows)], vec![batch(rows)]]).expect("union");
    let u = metric(&m, "union");
    assert!(
        u.peak_bytes > u.result_bytes,
        "a dedup union holds its input above the deduped result: peak {} result {}",
        u.peak_bytes,
        u.result_bytes
    );
}

#[test]
fn a_streaming_operator_holds_only_its_result() {
    let ir = plan(
        r#"{"op":"filter",
            "input":{"op":"scan","source_id":0},
            "predicate":{"e":"binary","op":"gt",
                         "left":{"e":"col","name":"v"},
                         "right":{"e":"lit","value":{"int":9000}}}}"#,
    );
    let (_out, m) = execute_metered(&ir, &[vec![batch(10_000)]]).expect("filter runs");
    let filter = metric(&m, "filter");
    assert_eq!(
        filter.peak_bytes, filter.result_bytes,
        "a filter streams: its peak is the result it emits"
    );
    assert_eq!(filter.rows_build, 0);
}
