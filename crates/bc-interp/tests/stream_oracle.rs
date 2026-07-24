//! `execute_streaming` == `execute`, the sequential oracle.
//!
//! The streaming executor is a new *scheduling* of the same operator semantics — exactly what
//! `par` is to `execute` — so the only thing that licenses it is producing the same relation.
//! It calls the same `ops::`/`bc-runtime` kernels; what changes is that morsels are pulled
//! through the linear runs instead of every operator's full output being collected.
//!
//! Two of its claims are stronger than "same rows" and are pinned separately:
//!
//! * **The hash-join probe streams**, morsel by morsel, against a build table hashed once.
//!   Morsels are contiguous in-order row ranges, so the emitted rows must come out in the
//!   *same order* as joining the concatenated probe relation. Order is asserted, not just the
//!   multiset — an out-of-order join would sail past a set comparison.
//! * **The aggregate folds incrementally** (`partial` → `combine` per morsel) rather than
//!   reading its input into RAM. That leans on `combine` being associative; if it were not, a
//!   different morsel split would give a different answer. The plans here are deliberately
//!   larger than one morsel (16,384 rows) so the fold actually runs more than once.
//!
//! Plans are built from the **JSON IR**, so these also exercise the real wire contract rather
//! than a hand-assembled enum a `to_ir()` change could drift away from.

use arrow::array::{ArrayRef, Int64Array, StringArray};
use arrow::record_batch::RecordBatch;
use bc_interp::{execute, execute_streaming, execute_streaming_parallel};
use bc_ir::RelOp;
use std::sync::Arc;

/// Big enough that every streaming path is genuinely exercised rather than skipped:
///
/// * **> 1 morsel** (16,384 rows), so the join probes several morsels and the aggregate folds
///   several partials instead of trivially handling one.
/// * **> `MIN_ROWS_TO_SHARD`** (4 morsels = 65,536 rows), so `execute_streaming_parallel`
///   actually *shards* across workers. Below that it declines and falls back to the sequential
///   path — which would make every "parallel" assertion in this file a second run of the
///   sequential one, passing while testing nothing. That is the exact trap this constant exists
///   to avoid, so do not lower it without lowering the threshold too.
const N: i64 = 200_000;

fn plan(json: &str) -> RelOp {
    serde_json::from_str(json).unwrap_or_else(|e| panic!("bad plan JSON: {e}\n{json}"))
}

/// `k` cycles over 100 values (a realistic group/join key), `v` counts up, `s` is a string
/// column so the non-integer paths are exercised too.
fn facts() -> Vec<RecordBatch> {
    let k: ArrayRef = Arc::new(Int64Array::from(
        (0..N).map(|i| i % 100).collect::<Vec<_>>(),
    ));
    let v: ArrayRef = Arc::new(Int64Array::from((0..N).collect::<Vec<_>>()));
    let s: ArrayRef = Arc::new(StringArray::from(
        (0..N).map(|i| format!("s{}", i % 7)).collect::<Vec<_>>(),
    ));
    vec![RecordBatch::try_from_iter(vec![("k", k), ("v", v), ("s", s)]).unwrap()]
}

/// A small dimension table to join against — 100 keys, one row each.
fn dim() -> Vec<RecordBatch> {
    let k: ArrayRef = Arc::new(Int64Array::from((0..100i64).collect::<Vec<_>>()));
    let d: ArrayRef = Arc::new(Int64Array::from(
        (0..100i64).map(|i| i * 10).collect::<Vec<_>>(),
    ));
    vec![RecordBatch::try_from_iter(vec![("k", k), ("d", d)]).unwrap()]
}

/// Render a relation as one string per row, so two executors' outputs compare regardless of how
/// they happened to batch it (the batching is a scheduling detail; the rows are the contract).
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

/// Same rows, same order.
fn assert_ordered(json: &str, sources: &[Vec<RecordBatch>]) {
    let p = plan(json);
    let want = execute(&p, sources).expect("oracle");
    for (label, got) in executors(&p, sources) {
        assert_eq!(
            rows(&got),
            rows(&want),
            "{label} diverged from the oracle (order-sensitive) for:\n{json}"
        );
    }
}

/// Both streaming executors, checked against the oracle by every assertion.
///
/// The parallel one shards the driving scan across workers, so it is the one that can get row
/// *order* wrong (shards must concatenate in order) and the aggregate wrong (shard partials must
/// be combined, never finalized per shard). Running the whole suite through both is what pins
/// that — the sequential path passing proves nothing about the parallel one.
fn executors(p: &RelOp, sources: &[Vec<RecordBatch>]) -> Vec<(&'static str, Vec<RecordBatch>)> {
    vec![
        (
            "streaming",
            execute_streaming(p, sources, 0).expect("streaming"),
        ),
        (
            "streaming-parallel",
            execute_streaming_parallel(p, sources, 4, 0).expect("streaming-parallel"),
        ),
    ]
}

/// Same rows, any order — for the operators whose output order SQL does not define (a hash
/// aggregate's group order is its hash table's, and folding partials may visit groups in a
/// different order than one big partial would).
fn assert_multiset(json: &str, sources: &[Vec<RecordBatch>]) {
    let p = plan(json);
    let want = execute(&p, sources).expect("oracle");
    for (label, got) in executors(&p, sources) {
        let (mut a, mut b) = (rows(&got), rows(&want));
        a.sort();
        b.sort();
        assert_eq!(
            a, b,
            "{label} diverged from the oracle (multiset) for:\n{json}"
        );
    }
}

const SCAN: &str = r#"{"op":"scan","source_id":0}"#;

fn col(name: &str) -> String {
    format!(r#"{{"e":"col","name":"{name}"}}"#)
}

// ---- linear pipeline operators ---------------------------------------------------------

#[test]
fn scan_matches_the_oracle() {
    assert_ordered(SCAN, &[facts()]);
}

#[test]
fn filter_matches_the_oracle() {
    // A predicate that keeps most rows (the high-selectivity shape) and one that keeps few.
    for lit in [10, 90] {
        let json = format!(
            r#"{{"op":"filter","input":{SCAN},"predicate":{{"e":"binary","op":"lt",
                "left":{},"right":{{"e":"lit","value":{{"int":{lit}}}}}}}}}"#,
            col("k")
        );
        assert_ordered(&json, &[facts()]);
    }
}

#[test]
fn project_matches_the_oracle() {
    let json = format!(
        r#"{{"op":"project","input":{SCAN},"exprs":[
            {{"expr":{},"alias":"k"}},
            {{"expr":{{"e":"binary","op":"mul","left":{},"right":{{"e":"lit","value":{{"int":2}}}}}},"alias":"v2"}}
        ]}}"#,
        col("k"),
        col("v")
    );
    assert_ordered(&json, &[facts()]);
}

#[test]
fn a_chain_of_linear_operators_matches_the_oracle() {
    // The shape the whole executor exists for: scan → filter → project, never materialized.
    let filter = format!(
        r#"{{"op":"filter","input":{SCAN},"predicate":{{"e":"binary","op":"ge",
            "left":{},"right":{{"e":"lit","value":{{"int":50}}}}}}}}"#,
        col("k")
    );
    let json = format!(
        r#"{{"op":"project","input":{filter},"exprs":[
            {{"expr":{},"alias":"k"}},{{"expr":{},"alias":"v"}}]}}"#,
        col("k"),
        col("v")
    );
    assert_ordered(&json, &[facts()]);
}

// ---- limit: the operator whose streaming form changes complexity, not just memory --------

#[test]
fn limit_matches_the_oracle_including_its_edges() {
    for (n, offset) in [
        (10, 0),
        (0, 0),
        (5, 3),
        (N as usize + 10, 0),
        (10, N as usize),
    ] {
        let json = format!(r#"{{"op":"limit","input":{SCAN},"n":{n},"offset":{offset}}}"#);
        assert_ordered(&json, &[facts()]);
    }
}

#[test]
fn limit_stops_pulling_once_it_is_satisfied() {
    // The early exit, observed rather than asserted about: the streaming limit must not read
    // the whole relation. `execute` produces the same 3 rows, but only after scanning all
    // 50,000 — this test pins the *result*; the memory/complexity win is what motivates it.
    let json = format!(r#"{{"op":"limit","input":{SCAN},"n":3,"offset":0}}"#);
    let p = plan(&json);
    let got = execute_streaming(&p, &[facts()], 0).expect("streaming");
    assert_eq!(rows(&got).len(), 3);
    assert_eq!(rows(&got), rows(&execute(&p, &[facts()]).expect("oracle")));
}

// ---- the aggregate: folded incrementally, state bounded by the group count ---------------

#[test]
fn aggregate_matches_the_oracle() {
    let json = format!(
        r#"{{"op":"aggregate","input":{SCAN},
            "group_keys":[{{"expr":{},"alias":"k"}}],
            "aggregates":[
                {{"func":"sum","input":{},"alias":"sv"}},
                {{"func":"count_star","alias":"n"}},
                {{"func":"min","input":{},"alias":"mn"}},
                {{"func":"max","input":{},"alias":"mx"}}
            ]}}"#,
        col("k"),
        col("v"),
        col("v"),
        col("v")
    );
    assert_multiset(&json, &[facts()]);
}

#[test]
fn global_aggregate_matches_the_oracle() {
    let json = format!(
        r#"{{"op":"aggregate","input":{SCAN},"group_keys":[],
            "aggregates":[{{"func":"sum","input":{},"alias":"sv"}},
                          {{"func":"count_star","alias":"n"}}]}}"#,
        col("v")
    );
    assert_multiset(&json, &[facts()]);
}

#[test]
fn aggregate_over_an_empty_input_matches_the_oracle() {
    // A global aggregate over nothing still yields one row (`COUNT` 0, `SUM` NULL) — the case
    // where the incremental fold has no partial to finalize and must defer to the oracle.
    let empty = format!(
        r#"{{"op":"filter","input":{SCAN},"predicate":{{"e":"binary","op":"lt",
            "left":{},"right":{{"e":"lit","value":{{"int":-1}}}}}}}}"#,
        col("k")
    );
    let json = format!(
        r#"{{"op":"aggregate","input":{empty},"group_keys":[],
            "aggregates":[{{"func":"count_star","alias":"n"}},
                          {{"func":"sum","input":{},"alias":"sv"}}]}}"#,
        col("v")
    );
    assert_multiset(&json, &[facts()]);
}

#[test]
fn string_grouped_aggregate_matches_the_oracle() {
    let json = format!(
        r#"{{"op":"aggregate","input":{SCAN},
            "group_keys":[{{"expr":{},"alias":"s"}}],
            "aggregates":[{{"func":"sum","input":{},"alias":"sv"}}]}}"#,
        col("s"),
        col("v")
    );
    assert_multiset(&json, &[facts()]);
}

// ---- the hash join: build once, stream the probe -----------------------------------------

fn join(join_type: &str) -> String {
    format!(
        r#"{{"op":"hash_join","left":{SCAN},"right":{{"op":"scan","source_id":1}},
            "left_keys":["k"],"right_keys":["k"],"join_type":"{join_type}",
            "output":[{{"side":"left","name":"k","alias":"k"}},
                      {{"side":"left","name":"v","alias":"v"}},
                      {{"side":"right","name":"d","alias":"d"}}],
            "strategy":"hash"}}"#
    )
}

#[test]
fn a_streamed_inner_join_matches_the_oracle_in_order() {
    // The probe streams; the order claim is the load-bearing one.
    assert_ordered(&join("inner"), &[facts(), dim()]);
}

#[test]
fn a_streamed_left_join_matches_the_oracle_in_order() {
    assert_ordered(&join("left"), &[facts(), dim()]);
}

#[test]
fn right_and_full_joins_fall_back_and_still_match_the_oracle() {
    // `Right`/`Full` must reconcile build-side rows nothing matched, which no single morsel can
    // know — `BroadcastProbe` declines them and the materialized path takes over. The point of
    // the test is that declining is *silent and correct*, not that it streams.
    for jt in ["right", "full"] {
        let json = format!(
            r#"{{"op":"hash_join","left":{SCAN},"right":{{"op":"scan","source_id":1}},
                "left_keys":["k"],"right_keys":["k"],"join_type":"{jt}",
                "output":[{{"side":"left","name":"v","alias":"v"}},
                          {{"side":"right","name":"d","alias":"d"}}],
                "strategy":"hash"}}"#
        );
        assert_multiset(&json, &[facts(), dim()]);
    }
}

#[test]
fn semi_and_anti_joins_match_the_oracle() {
    for jt in ["semi", "anti"] {
        let json = format!(
            r#"{{"op":"hash_join","left":{SCAN},"right":{{"op":"scan","source_id":1}},
                "left_keys":["k"],"right_keys":["k"],"join_type":"{jt}",
                "output":[{{"side":"left","name":"k","alias":"k"}},
                          {{"side":"left","name":"v","alias":"v"}}],
                "strategy":"hash"}}"#
        );
        assert_ordered(&json, &[facts(), dim()]);
    }
}

#[test]
fn an_anti_join_over_a_limit_zero_build_keeps_every_probe_row() {
    // The shape that escaped the first pass. Kyber rewrites a provably-empty build side to
    // `LIMIT 0`, and a streaming `LIMIT 0` that never pulls its child never learns the schema,
    // so it yielded *nothing* — not even the schema-only batch `ops::limit` returns. An
    // anti/left join over that empty build then wrongly dropped every probe row. Nulls in the
    // probe key make it the exact differential case (`test_diff_kyber2_optimizer_matrix`).
    let l_k: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), Some(3), None, Some(0)]));
    let l_v: ArrayRef = Arc::new(Int64Array::from(vec![Some(2), None, Some(3), Some(3)]));
    let left = vec![RecordBatch::try_from_iter(vec![("k", l_k), ("v", l_v)]).unwrap()];

    for jt in ["anti", "left"] {
        let json = format!(
            r#"{{"op":"hash_join","left":{SCAN},
                "right":{{"op":"limit","input":{{"op":"scan","source_id":1}},"n":0,"offset":0}},
                "left_keys":["k"],"right_keys":["k"],"join_type":"{jt}",
                "output":[{{"side":"left","name":"k","alias":"k"}},
                          {{"side":"left","name":"v","alias":"v"}}],
                "strategy":"hash"}}"#
        );
        assert_multiset(&json, &[left.clone(), dim()]);
    }
}

#[test]
fn a_join_against_an_empty_build_side_matches_the_oracle() {
    let empty_dim = {
        let k: ArrayRef = Arc::new(Int64Array::from(Vec::<i64>::new()));
        let d: ArrayRef = Arc::new(Int64Array::from(Vec::<i64>::new()));
        vec![RecordBatch::try_from_iter(vec![("k", k), ("d", d)]).unwrap()]
    };
    assert_multiset(&join("inner"), &[facts(), empty_dim]);
}

// ---- the shape that OOMs today: a deep left-deep join tree feeding an aggregate ----------

#[test]
fn a_deep_join_tree_into_an_aggregate_matches_the_oracle() {
    // The q3/q4/q5 shape. Materializing every operator's output is what peaks at 133 GB at
    // sf100; here the probe threads one morsel through both joins and the aggregate folds it,
    // so nothing between the operators is ever held. Correctness first — the memory claim is
    // only worth making once this passes.
    let j1 = join("inner");
    let j2 = format!(
        r#"{{"op":"hash_join","left":{j1},"right":{{"op":"scan","source_id":2}},
            "left_keys":["k"],"right_keys":["k"],"join_type":"inner",
            "output":[{{"side":"left","name":"k","alias":"k"}},
                      {{"side":"left","name":"v","alias":"v"}},
                      {{"side":"left","name":"d","alias":"d"}},
                      {{"side":"right","name":"e","alias":"e"}}],
            "strategy":"hash"}}"#
    );
    let json = format!(
        r#"{{"op":"aggregate","input":{j2},
            "group_keys":[{{"expr":{},"alias":"k"}}],
            "aggregates":[{{"func":"sum","input":{},"alias":"sv"}},
                          {{"func":"count_star","alias":"n"}}]}}"#,
        col("k"),
        col("v")
    );

    let dim2 = {
        let k: ArrayRef = Arc::new(Int64Array::from((0..100i64).collect::<Vec<_>>()));
        let e: ArrayRef = Arc::new(Int64Array::from(
            (0..100i64).map(|i| i + 7).collect::<Vec<_>>(),
        ));
        vec![RecordBatch::try_from_iter(vec![("k", k), ("e", e)]).unwrap()]
    };
    assert_multiset(&json, &[facts(), dim(), dim2]);
}

// ---- breakers deferred to the oracle: they must still be reachable and correct ------------

#[test]
fn sort_matches_the_oracle() {
    let json = format!(
        r#"{{"op":"sort","input":{SCAN},"keys":[{{"expr":{},"descending":true}}],"limit":null}}"#,
        col("v")
    );
    assert_ordered(&json, &[facts()]);
}

#[test]
fn top_n_matches_the_oracle() {
    let json = format!(
        r#"{{"op":"sort","input":{SCAN},"keys":[{{"expr":{},"descending":true}}],"limit":10}}"#,
        col("v")
    );
    assert_ordered(&json, &[facts()]);
}

#[test]
fn distinct_matches_the_oracle() {
    let proj = format!(
        r#"{{"op":"project","input":{SCAN},"exprs":[{{"expr":{},"alias":"k"}}]}}"#,
        col("k")
    );
    let json = format!(r#"{{"op":"distinct","input":{proj}}}"#);
    assert_multiset(&json, &[facts()]);
}

/// A `DISTINCT` is sharded across workers like a root aggregate: each shard dedups its own rows
/// and the driver merges. What makes that sound is that dedup is a mergeable group-by, so the
/// merge of per-shard survivors is the relation's distinct set — and what would make it *unsound*
/// is a shard answering for itself, which shows up as duplicate rows the oracle does not have.
///
/// Two columns of different kinds (an integer and a string), because the shard partitioner and
/// the dedup take different paths per key type and a single-integer key exercises neither the
/// mixed-key hash nor the byte comparison.
#[test]
fn a_multi_column_distinct_shards_and_matches_the_oracle() {
    let proj = format!(
        r#"{{"op":"project","input":{SCAN},"exprs":[
            {{"expr":{},"alias":"k"}},{{"expr":{},"alias":"s"}}]}}"#,
        col("k"),
        col("s")
    );
    let json = format!(r#"{{"op":"distinct","input":{proj}}}"#);
    assert_multiset(&json, &[facts()]);
}

/// The shape Kyber lowers a grouped `COUNT(DISTINCT x)` to: `count(*)` over `DISTINCT (k, x)`,
/// with a filter underneath. This is what serialized on the streaming executor — the dedup was
/// refused a shard, so the scan and filter feeding it ran on one core — and it is the plan the
/// ClickBench q10/q11/q13 timings come from, so it is pinned as a plan and not just as an
/// operator.
#[test]
fn a_grouped_count_distinct_over_a_filter_matches_the_oracle() {
    let filter = format!(
        r#"{{"op":"filter","input":{SCAN},"predicate":{{"e":"binary","op":"ge",
            "left":{},"right":{{"e":"lit","value":{{"int":10}}}}}}}}"#,
        col("k")
    );
    let proj = format!(
        r#"{{"op":"project","input":{filter},"exprs":[
            {{"expr":{},"alias":"k"}},{{"expr":{},"alias":"s"}}]}}"#,
        col("k"),
        col("s")
    );
    let distinct = format!(r#"{{"op":"distinct","input":{proj}}}"#);
    let json = format!(
        r#"{{"op":"aggregate","input":{distinct},
            "group_keys":[{{"expr":{},"alias":"k"}}],
            "aggregates":[{{"func":"count_star","alias":"n"}}]}}"#,
        col("k")
    );
    assert_multiset(&json, &[facts()]);
}

#[test]
fn union_matches_the_oracle() {
    let json = format!(r#"{{"op":"union","inputs":[{SCAN},{SCAN}],"distinct":false}}"#);
    assert_multiset(&json, &[facts()]);
}

// ---- a breaker feeding a pipeline, and vice versa -----------------------------------------

#[test]
fn a_pipeline_above_a_breaker_matches_the_oracle() {
    // sort → filter → project: the breaker materializes (as it must), and the linear run above
    // it streams its output. Both directions of the boundary have to work.
    let sort = format!(
        r#"{{"op":"sort","input":{SCAN},"keys":[{{"expr":{},"descending":false}}],"limit":null}}"#,
        col("v")
    );
    let filter = format!(
        r#"{{"op":"filter","input":{sort},"predicate":{{"e":"binary","op":"lt",
            "left":{},"right":{{"e":"lit","value":{{"int":25}}}}}}}}"#,
        col("k")
    );
    let json = format!(
        r#"{{"op":"project","input":{filter},"exprs":[{{"expr":{},"alias":"v"}}]}}"#,
        col("v")
    );
    assert_ordered(&json, &[facts()]);
}

// ---- the streaming paths must actually be TAKEN, or every test above is vacuous ----------

#[test]
fn the_join_in_these_tests_really_does_stream() {
    // Everything above would pass just as happily if `BroadcastProbe` declined every shape and
    // the materialized fallback quietly did all the work. It does not: this is the precondition
    // the executor consults, asserted for exactly the shape the join tests use (inner join, one
    // Int64 key, a 100-row build side). If this ever goes false, the join tests stop testing
    // streaming and start testing the oracle against itself.
    use bc_runtime::join::{streaming_supported, JoinType};
    let int64 = arrow::datatypes::DataType::Int64;

    assert!(
        streaming_supported(JoinType::Inner, &[&int64], 100),
        "the inner-join tests must exercise the streaming probe, not the fallback"
    );
    assert!(streaming_supported(JoinType::Left, &[&int64], 100));
    assert!(streaming_supported(JoinType::Semi, &[&int64], 100));
    assert!(streaming_supported(JoinType::Anti, &[&int64], 100));

    // ...and the ones that must NOT stream, because a morsel cannot know what the build side
    // failed to match until every other morsel has been probed.
    assert!(!streaming_supported(JoinType::Right, &[&int64], 100));
    assert!(!streaming_supported(JoinType::Full, &[&int64], 100));
}

#[test]
fn limit_does_not_evaluate_rows_it_will_never_return() {
    // The early exit, proven rather than asserted about — and the one place streaming is not
    // merely a rescheduling of the oracle.
    //
    // The source's first batch casts cleanly; the second holds "abc", which a non-`try` cast to
    // Int64 rejects. `execute` projects the whole relation before the limit ever runs, so it
    // *errors*. The streaming limit stops pulling after three rows and never evaluates the
    // second batch at all, so it succeeds. Nothing but genuine laziness can produce that.
    //
    // This is also the one place streaming is not merely a rescheduling of the oracle, and it
    // is deliberate. `SELECT CAST(s AS BIGINT) FROM t LIMIT 3` should not be broken by rows the
    // query was never going to return — DuckDB and Postgres both agree. The difference is
    // strictly a *reduction* in spurious failures: streaming never errors where the oracle
    // succeeds, it only succeeds where the oracle errored doing work it should not have done.
    // A difference nonetheless, so it lives in a test that says so rather than a footnote.
    let ok: ArrayRef = Arc::new(StringArray::from(vec!["1", "2", "3", "4"]));
    let bad: ArrayRef = Arc::new(StringArray::from(vec!["abc", "def"]));
    let src = vec![
        RecordBatch::try_from_iter(vec![("s", ok)]).unwrap(),
        RecordBatch::try_from_iter(vec![("s", bad)]).unwrap(),
    ];

    let proj = format!(
        r#"{{"op":"project","input":{SCAN},"exprs":[
            {{"expr":{{"e":"cast","input":{},"dtype":"int64","try_cast":false}},"alias":"q"}}]}}"#,
        col("s")
    );
    let json = format!(r#"{{"op":"limit","input":{proj},"n":3,"offset":0}}"#);
    let p = plan(&json);

    assert!(
        execute(&p, std::slice::from_ref(&src)).is_err(),
        "the oracle projects the whole relation and must hit the bad cast"
    );
    let got = execute_streaming(&p, &[src], 0).expect("streaming stops before the bad batch");
    assert_eq!(rows(&got), vec!["1", "2", "3"]);
}

#[test]
fn the_aggregate_folds_more_than_once() {
    // `AGG_FOLD_EVERY` is 32, so the fold-and-clear branch only runs past 32 morsels — i.e.
    // above 32 x 16,384 rows. Under that, the test above only ever exercises the single-combine
    // tail. Push past it, so the incremental fold (the thing that keeps the aggregate's memory
    // bounded by group count rather than input size) is actually executed.
    let n: i64 = 600_000; // ≈37 morsels
    let k: ArrayRef = Arc::new(Int64Array::from((0..n).map(|i| i % 50).collect::<Vec<_>>()));
    let v: ArrayRef = Arc::new(Int64Array::from((0..n).collect::<Vec<_>>()));
    let big = vec![RecordBatch::try_from_iter(vec![("k", k), ("v", v)]).unwrap()];

    let json = format!(
        r#"{{"op":"aggregate","input":{SCAN},
            "group_keys":[{{"expr":{},"alias":"k"}}],
            "aggregates":[{{"func":"sum","input":{},"alias":"sv"}},
                          {{"func":"count_star","alias":"n"}}]}}"#,
        col("k"),
        col("v")
    );
    assert_multiset(&json, &[big]);
}

#[test]
fn an_aggregate_over_a_filtered_join_matches_the_oracle() {
    // scan → join(streamed probe) → filter → aggregate(folded): every kind of edge at once.
    let j = join("inner");
    let filter = format!(
        r#"{{"op":"filter","input":{j},"predicate":{{"e":"binary","op":"ge",
            "left":{},"right":{{"e":"lit","value":{{"int":500}}}}}}}}"#,
        col("d")
    );
    let json = format!(
        r#"{{"op":"aggregate","input":{filter},
            "group_keys":[{{"expr":{},"alias":"k"}}],
            "aggregates":[{{"func":"sum","input":{},"alias":"sv"}}]}}"#,
        col("k"),
        col("v")
    );
    assert_multiset(&json, &[facts(), dim()]);
}

// ---- fractional sample is a pipeline operator, not a breaker ------------------------------

/// A **fractional** sample keeps a row iff a seeded hash of its values falls under the
/// fraction — a per-row predicate. So it streams, and because it only ever drops rows it must
/// also preserve order: `assert_ordered`, not `assert_multiset`. A multiset comparison would
/// pass even if the streaming arm reordered, which is exactly the blind spot the join test at
/// the top of this file warns about.
///
/// The oracle is the authority that this is per-row: it maps `sample_batch` over each batch
/// independently, and records only the fixed-`n` arm as a breaker.
#[test]
fn fractional_sample_streams_and_matches_the_oracle_in_order() {
    for (fraction, seed) in [(0.0, 7u64), (0.25, 7), (0.5, 1), (1.0, 99)] {
        let json = format!(
            r#"{{"op":"sample","input":{SCAN},"fraction":{fraction},"seed":{seed},"n":null}}"#
        );
        assert_ordered(&json, &[facts()]);
    }
}

/// The sample must not depend on where morsel boundaries fall. Same plan, same seed, over an
/// input pre-split into many batches vs one — a per-row hash predicate cannot notice, but a
/// per-morsel *draw* would, so this is what separates the two.
#[test]
fn fractional_sample_is_independent_of_batch_boundaries() {
    let json = format!(r#"{{"op":"sample","input":{SCAN},"fraction":0.3,"seed":42,"n":null}}"#);
    let p = plan(&json);
    let one = execute_streaming(&p, &[facts()], 0).expect("single batch");

    // Re-slice the same rows into many small batches.
    let split: Vec<RecordBatch> = facts()
        .iter()
        .flat_map(|b| {
            (0..b.num_rows())
                .step_by(1000)
                .map(|off| b.slice(off, 1000.min(b.num_rows() - off)))
                .collect::<Vec<_>>()
        })
        .collect();
    let many = execute_streaming(&p, &[split], 0).expect("many batches");

    assert_eq!(
        rows(&one),
        rows(&many),
        "a fractional sample must select the same rows regardless of batch boundaries"
    );
}

// ---- deferred breakers respect the memory envelope: spill, never OOM ----------------------
//
// `Distinct`/`Window`/fixed-`n` `Sample`/`Union`/`AsofJoin` are handed to the sequential
// oracle (a *fractional* `Sample` streams — see above), which
// materializes their whole input. Under a memory budget that would OOM where the spilling
// parallel executor survives, so the streaming path now bounds the deferral: within budget it
// re-runs the oracle over the drained input via synthetic scans (must stay transparent); over
// budget it yields `MemoryBudgetExceeded` so the FFI re-runs on the executor that spills.

/// The deferred-breaker plans used by both budget tests (single-input and multi-input shapes).
fn deferred_breaker_plans() -> Vec<String> {
    let proj_k = format!(
        r#"{{"op":"project","input":{SCAN},"exprs":[{{"expr":{},"alias":"k"}}]}}"#,
        col("k")
    );
    vec![
        // DISTINCT — single input, grace-spillable in the parallel executor.
        format!(r#"{{"op":"distinct","input":{proj_k}}}"#),
        // UNION DISTINCT — multi input; the dedup is grace-spillable.
        format!(r#"{{"op":"union","inputs":[{SCAN},{SCAN}],"distinct":true}}"#),
        // SAMPLE fixed-count — single input, a breaker over the whole relation.
        format!(r#"{{"op":"sample","input":{SCAN},"fraction":1.0,"seed":7,"n":50}}"#),
    ]
}

/// Within a generous budget, every deferred breaker still equals the oracle — the synthetic-scan
/// re-execution the bounded path introduces must be invisible.
#[test]
fn deferred_breakers_match_the_oracle_under_a_budget() {
    let big = 1usize << 40; // 1 TiB — fits everything here; exercises the budget>0 rewrite path.
    for json in deferred_breaker_plans() {
        let p = plan(&json);
        let want = execute(&p, &[facts()]).expect("oracle");
        let got =
            execute_streaming(&p, &[facts()], big).expect("streaming under a generous budget");
        let (mut a, mut b) = (rows(&got), rows(&want));
        a.sort();
        b.sort();
        assert_eq!(
            a, b,
            "deferred breaker diverged from the oracle under a budget:\n{json}"
        );
    }
}

/// Under a 1-byte budget, a deferred breaker must not OOM: it gives way with
/// `MemoryBudgetExceeded` (the signal the FFI catches to re-run on the spilling executor),
/// instead of materializing its whole input in the oracle.
#[test]
fn deferred_breakers_yield_over_budget_instead_of_oom() {
    for json in deferred_breaker_plans() {
        let p = plan(&json);
        let err =
            execute_streaming(&p, &[facts()], 1).expect_err("must yield over a 1-byte budget");
        assert!(
            matches!(err, bc_interp::InterpError::MemoryBudgetExceeded { .. }),
            "expected MemoryBudgetExceeded for:\n{json}\ngot: {err:?}"
        );
    }
}

/// A semi/anti join whose **build side is the big one** — the shape the streaming executor flips,
/// building the small left and streaming the right past it while marking (`semi_by_marking_left`).
///
/// It only engages once the build is past `RADIX_MIN_BUILD_ROWS_BROADCAST` (~2.1M rows), which is
/// why no other test here reaches it: the rest are deliberately small. TPC-H q4
/// (`orders SEMI lineitem`) is this shape. The flip is only sound because a semi join emits no
/// build column and only asks whether a key exists — so the answer, *and the row order*, must be
/// exactly what building the right produces. Nulls on both sides and duplicate keys on both sides
/// are the cases that separate SEMI from ANTI (`NOT EXISTS` is true for a null-keyed left row).
#[test]
fn a_semi_join_with_a_huge_build_matches_the_oracle() {
    // Left: small, with null keys, duplicate keys, and keys absent from the right.
    let n_left = 5_000usize;
    let l_k: ArrayRef = Arc::new(Int64Array::from(
        (0..n_left)
            .map(|i| match i % 7 {
                0 => None,                   // null key: SEMI drops it, ANTI keeps it
                1 => Some(-(i as i64) - 1),  // never on the right
                _ => Some((i % 600) as i64), // duplicated, and present on the right
            })
            .collect::<Vec<_>>(),
    ));
    let l_v: ArrayRef = Arc::new(Int64Array::from(
        (0..n_left).map(|i| Some(i as i64)).collect::<Vec<_>>(),
    ));
    let left = vec![RecordBatch::try_from_iter(vec![("k", l_k), ("v", l_v)]).unwrap()];

    // Right: > 2.1M rows so the per-morsel probe declines and the flip engages; keys 0..500 only,
    // so left keys in 500..600 are genuinely unmatched. Nulls match nothing.
    let n_right = 2_200_000usize;
    let r_k: ArrayRef = Arc::new(Int64Array::from(
        (0..n_right)
            .map(|i| {
                if i % 13 == 0 {
                    None
                } else {
                    Some((i % 500) as i64)
                }
            })
            .collect::<Vec<_>>(),
    ));
    let right = vec![RecordBatch::try_from_iter(vec![("k", r_k)]).unwrap()];

    for jt in ["semi", "anti"] {
        let json = format!(
            r#"{{"op":"hash_join","left":{SCAN},"right":{{"op":"scan","source_id":1}},
                "left_keys":["k"],"right_keys":["k"],"join_type":"{jt}",
                "output":[{{"side":"left","name":"k","alias":"k"}},
                          {{"side":"left","name":"v","alias":"v"}}],
                "strategy":"hash"}}"#
        );
        assert_ordered(&json, &[left.clone(), right.clone()]);
    }
}

// ---- a breaker ON THE PROBE SPINE: materialized in parallel, then treated as a leaf ---------
//
// A breaker between the plan root and the driving scan cannot be sharded *through* — one shard
// answers for one shard — so the parallel streaming executor evaluates that subtree first, over
// the unsharded sources, and caches the finished relation as a materialized leaf. Everything in
// this section is the wrong-answer shape that change can produce: a worker that sees a shard where
// it should see the whole relation duplicates rows (`workers x` too many) or drops them, and both
// mistakes sail past a test that only checks *which* rows came back if the counts are not compared
// too. `rows()` is a full multiset of every row, so a duplicated relation fails — but the
// order-sensitive assertions below are the sharper instrument and are used wherever SQL defines
// an order.

/// `k` cycles over 100 values, so this joins back to `dim()` on the aggregate's group key.
fn agg_by_k(input: &str) -> String {
    format!(
        r#"{{"op":"aggregate","input":{input},
            "group_keys":[{{"expr":{},"alias":"k"}}],
            "aggregates":[{{"func":"sum","input":{},"alias":"sv"}},
                          {{"func":"count_star","alias":"n"}}]}}"#,
        col("k"),
        col("v")
    )
}

/// `left JOIN dim ON k`, keeping the left's aggregate columns — the q17 shape in miniature: a
/// breaker sitting on the probe spine of a join.
fn join_to_dim(left: &str, left_cols: &[&str]) -> String {
    let mut output: Vec<String> = left_cols
        .iter()
        .map(|c| format!(r#"{{"side":"left","name":"{c}","alias":"{c}"}}"#))
        .collect();
    output.push(r#"{"side":"right","name":"d","alias":"d"}"#.to_string());
    format!(
        r#"{{"op":"hash_join","left":{left},"right":{{"op":"scan","source_id":1}},
            "left_keys":["k"],"right_keys":["k"],"join_type":"inner",
            "output":[{}],"strategy":"hash"}}"#,
        output.join(",")
    )
}

#[test]
fn a_mid_spine_aggregate_matches_the_oracle() {
    // The TPC-H q17 shape: `join(project(aggregate(...)), dim)`. Before spine materialization the
    // aggregate made `spine_is_shardable` refuse the *whole* plan, so its 200k-row input ran on
    // one core; now it is evaluated across every core and handed to the join as a leaf. What must
    // not change is the relation.
    let agg = agg_by_k(SCAN);
    let proj = format!(
        r#"{{"op":"project","input":{agg},"exprs":[
            {{"expr":{},"alias":"k"}},
            {{"expr":{{"e":"binary","op":"mul","left":{},"right":{{"e":"lit","value":{{"int":2}}}}}},"alias":"sv2"}}
        ]}}"#,
        col("k"),
        col("sv")
    );
    assert_multiset(&join_to_dim(&proj, &["k", "sv2"]), &[facts(), dim()]);
}

#[test]
fn a_mid_spine_sort_matches_the_oracle_in_order() {
    // The sharpest of these: a `Sort` is where sharding through a breaker is *visibly* wrong (sort
    // each shard, concatenate, and the result is not sorted), and a join's probe preserves its
    // input order — so the joined output must come back in the sort's order, not merely contain
    // the same rows. `assert_ordered`, deliberately: a multiset comparison here would pass on a
    // result that had been shuffled by exactly the bug this guards.
    let sort = format!(
        r#"{{"op":"sort","input":{SCAN},"keys":[{{"expr":{},"descending":true}}],"limit":null}}"#,
        col("v")
    );
    assert_ordered(&join_to_dim(&sort, &["k", "v"]), &[facts(), dim()]);
}

#[test]
fn a_mid_spine_top_n_matches_the_oracle_in_order() {
    // A `LIMIT`ed sort mid-spine: sharding through it yields `k x workers` rows, which is the
    // failure a row-count-blind comparison misses.
    let top = format!(
        r#"{{"op":"sort","input":{SCAN},"keys":[{{"expr":{},"descending":true}}],"limit":25}}"#,
        col("v")
    );
    assert_ordered(&join_to_dim(&top, &["k", "v"]), &[facts(), dim()]);
}

#[test]
fn a_mid_spine_distinct_matches_the_oracle() {
    // Dedup each shard independently and duplicates survive *across* shards — the classic reason
    // a breaker may not see a shard.
    let proj = format!(
        r#"{{"op":"project","input":{SCAN},"exprs":[
            {{"expr":{},"alias":"k"}},{{"expr":{},"alias":"s"}}]}}"#,
        col("k"),
        col("s")
    );
    let distinct = format!(r#"{{"op":"distinct","input":{proj}}}"#);
    assert_multiset(&join_to_dim(&distinct, &["k", "s"]), &[facts(), dim()]);
}

#[test]
fn a_mid_spine_limit_matches_the_oracle_in_order() {
    // `Limit` is not a breaker but is positional in the same way: a per-shard limit returns
    // `n x workers` rows. It is refused by `spine_is_shardable` for that reason and materialized
    // here like any other spine stopper.
    let limited = format!(r#"{{"op":"limit","input":{SCAN},"n":137,"offset":11}}"#);
    assert_ordered(&join_to_dim(&limited, &["k", "v"]), &[facts(), dim()]);
}

#[test]
fn nested_breakers_on_the_spine_match_the_oracle() {
    // A breaker under a breaker: `join(aggregate(sort(scan)), dim)`. Materializing the outer one
    // re-enters the parallel executor on its subtree, which runs its own pass and finds the inner
    // one — so each is evaluated exactly once, by exactly one call, and neither ever sees a shard.
    // Getting the recursion wrong here is an infinite loop or a doubled evaluation, not a wrong
    // row, so this test is as much about termination as about the relation.
    let sort = format!(
        r#"{{"op":"sort","input":{SCAN},"keys":[{{"expr":{},"descending":false}}],"limit":null}}"#,
        col("v")
    );
    let agg = agg_by_k(&sort);
    assert_multiset(&join_to_dim(&agg, &["k", "sv", "n"]), &[facts(), dim()]);
}

#[test]
fn a_breaker_over_a_distinct_over_a_limit_matches_the_oracle() {
    // Three deep, mixing the breaker kinds, so the recursion is exercised past the two-level case.
    let limited = format!(r#"{{"op":"limit","input":{SCAN},"n":50000,"offset":3}}"#);
    let proj = format!(
        r#"{{"op":"project","input":{limited},"exprs":[
            {{"expr":{},"alias":"k"}},{{"expr":{},"alias":"v"}}]}}"#,
        col("k"),
        col("v")
    );
    let distinct = format!(r#"{{"op":"distinct","input":{proj}}}"#);
    let agg = agg_by_k(&distinct);
    assert_multiset(&join_to_dim(&agg, &["k", "sv", "n"]), &[facts(), dim()]);
}

#[test]
fn a_spine_breaker_reading_the_same_source_as_the_other_side_matches_the_oracle() {
    // The self-join-through-a-breaker shape, in both directions. Sharding rewrites what one `Scan`
    // sees, so a source read in two places must not be sharded — and a materialized breaker adds a
    // second way for that to happen (its subtree's scans are not executed during the sharded
    // phase at all). Both arms must decline to shard and still produce the oracle's relation.
    let agg = agg_by_k(SCAN);

    // (a) the breaker is on the probe spine, and the build side re-reads source 0.
    let probe_side = format!(
        r#"{{"op":"hash_join","left":{agg},"right":{SCAN},
            "left_keys":["k"],"right_keys":["k"],"join_type":"inner",
            "output":[{{"side":"left","name":"k","alias":"k"}},
                      {{"side":"left","name":"sv","alias":"sv"}},
                      {{"side":"right","name":"v","alias":"v"}}],
            "strategy":"hash"}}"#
    );
    assert_multiset(&probe_side, &[facts()]);

    // (b) the driving scan is source 0 and the *build* side aggregates source 0 as well.
    let build_side = format!(
        r#"{{"op":"hash_join","left":{SCAN},"right":{agg},
            "left_keys":["k"],"right_keys":["k"],"join_type":"inner",
            "output":[{{"side":"left","name":"k","alias":"k"}},
                      {{"side":"left","name":"v","alias":"v"}},
                      {{"side":"right","name":"sv","alias":"sv"}}],
            "strategy":"hash"}}"#
    );
    assert_multiset(&build_side, &[facts()]);
}

#[test]
fn spine_breakers_match_the_oracle_over_empty_and_single_row_inputs() {
    // The degenerate inputs, over every spine-breaker shape above. An empty relation still owes
    // its schema upward, and a global aggregate over nothing is one row, not none — the case where
    // the fold has no partial and must defer to the oracle. A single row is the other end: below
    // every sharding threshold, so the plan declines to shard but the materialization pass still
    // runs, which is exactly the interaction worth pinning.
    let one = facts()
        .iter()
        .map(|b| b.slice(0, 1))
        .collect::<Vec<RecordBatch>>();
    let none = facts()
        .iter()
        .map(|b| b.slice(0, 0))
        .collect::<Vec<RecordBatch>>();

    let sort = format!(
        r#"{{"op":"sort","input":{SCAN},"keys":[{{"expr":{},"descending":true}}],"limit":null}}"#,
        col("v")
    );
    let proj_k = format!(
        r#"{{"op":"project","input":{SCAN},"exprs":[{{"expr":{},"alias":"k"}}]}}"#,
        col("k")
    );
    let distinct = format!(r#"{{"op":"distinct","input":{proj_k}}}"#);
    let agg = agg_by_k(SCAN);

    for src in [one, none] {
        assert_multiset(&join_to_dim(&agg, &["k", "sv", "n"]), &[src.clone(), dim()]);
        assert_ordered(&join_to_dim(&sort, &["k", "v"]), &[src.clone(), dim()]);
        assert_multiset(&join_to_dim(&distinct, &["k"]), &[src.clone(), dim()]);
        // A *global* aggregate over an empty spine: one row out of no rows in.
        let global = format!(
            r#"{{"op":"aggregate","input":{SCAN},"group_keys":[],
                "aggregates":[{{"func":"count_star","alias":"n"}},
                              {{"func":"sum","input":{},"alias":"sv"}}]}}"#,
            col("v")
        );
        let over = format!(
            r#"{{"op":"project","input":{global},"exprs":[{{"expr":{},"alias":"n"}}]}}"#,
            col("n")
        );
        assert_multiset(&over, &[src, dim()]);
    }
}

// ---- a root `LIMIT n OFFSET m` over a breaker: peeled, so the breaker runs on every core -----
//
// `ORDER BY … LIMIT n OFFSET m` is the tail of most reporting queries, and it used to force the
// whole plan onto the serial fallback: `spine_is_shardable` refuses `Limit` (a per-shard limit
// returns `n x workers` rows), and only a root `Sort`/`Project`/`Filter` was peeled off. ClickBench
// q38–q42 — all `Limit(Sort(Aggregate(…)))` — measured 0.97–1.00x CPU parallelism, i.e. entirely
// serial, while the same shape *without* the `OFFSET` parallelized fine.
//
// The peel runs the child across every core and then takes the window from its result, which is
// only correct because that result is in the oracle's order. An offset into a differently-ordered
// relation returns the *wrong rows*, silently, so every assertion here is order-sensitive.

/// `sort(scan) → limit`, the ClickBench tail shape (minus the aggregate, added separately below).
fn limit_over_sort(n: usize, offset: usize) -> String {
    let sort = format!(
        r#"{{"op":"sort","input":{SCAN},"keys":[{{"expr":{},"descending":true}}],"limit":null}}"#,
        col("v")
    );
    format!(r#"{{"op":"limit","input":{sort},"n":{n},"offset":{offset}}}"#)
}

#[test]
fn a_root_limit_with_an_offset_matches_the_oracle_in_order() {
    // The offsets are chosen against the shard geometry: `execute_streaming_parallel` is called
    // with 4 workers over 200,000 rows, so shards are 50,000 rows each. 49,990 and 50,001
    // straddle the first boundary in both directions — the window a per-shard limit would get
    // wrong, and the one an off-by-one in the peel would get wrong.
    for (n, offset) in [
        (10, 0),
        (10, 5),
        (10, 49_990),  // spans the first shard boundary
        (10, 50_001),  // starts just past it
        (10, 150_000), // in the last shard
        (1, 199_999),  // the final row
        (0, 10),       // LIMIT 0 — no rows, but still a schema
        (10, N as usize),
        (10, N as usize + 1_000), // offset past the end: empty, not an error
        (N as usize + 10, 0),     // limit past the end: the whole relation
    ] {
        assert_ordered(&limit_over_sort(n, offset), &[facts()]);
    }
}

#[test]
fn a_root_limit_over_an_aggregate_and_sort_matches_the_oracle_in_order() {
    // The exact ClickBench q38–q42 shape: `Limit(Sort(Aggregate(...)))`. Grouping on `s` gives 7
    // groups, so the offsets here walk off the end of a genuinely small relation as well.
    let agg = format!(
        r#"{{"op":"aggregate","input":{SCAN},
            "group_keys":[{{"expr":{},"alias":"s"}}],
            "aggregates":[{{"func":"count_star","alias":"n"}}]}}"#,
        col("s")
    );
    let sort = format!(
        r#"{{"op":"sort","input":{agg},"keys":[{{"expr":{},"descending":false}}],"limit":null}}"#,
        col("s")
    );
    for (n, offset) in [(10, 0), (3, 2), (10, 6), (10, 7), (10, 99), (0, 0)] {
        let json = format!(r#"{{"op":"limit","input":{sort},"n":{n},"offset":{offset}}}"#);
        assert_ordered(&json, &[facts()]);
    }
}

#[test]
fn a_root_limit_over_a_distinct_matches_the_oracle() {
    // A non-`Sort` breaker child, so the peel is not accidentally specific to the sort shape.
    let proj = format!(
        r#"{{"op":"project","input":{SCAN},"exprs":[{{"expr":{},"alias":"k"}}]}}"#,
        col("k")
    );
    let distinct = format!(r#"{{"op":"distinct","input":{proj}}}"#);
    for (n, offset) in [(10, 0), (10, 95), (10, 100), (5, 3)] {
        let json = format!(r#"{{"op":"limit","input":{distinct},"n":{n},"offset":{offset}}}"#);
        assert_multiset(&json, &[facts()]);
    }
}

#[test]
fn a_root_limit_over_a_pipeline_still_stops_pulling_early() {
    // The other half of the peel, and the reason it is restricted to a *breaker* child.
    //
    // Over a pipeline child the streaming `Limit` stops pulling once it is satisfied — the one
    // place this executor changes complexity rather than memory. Peeling there would run the whole
    // child first and throw the early exit away, so the peel must NOT apply. Proven the same way
    // the sequential test proves it: the second batch holds a value a strict cast rejects, so any
    // executor that evaluates it *errors*. Succeeding is only possible by never pulling it.
    //
    // Without the breaker-child restriction this test fails with a cast error, which is exactly
    // the regression it exists to catch.
    let ok: ArrayRef = Arc::new(StringArray::from(vec!["1", "2", "3", "4"]));
    let bad: ArrayRef = Arc::new(StringArray::from(vec!["abc", "def"]));
    let src = vec![
        RecordBatch::try_from_iter(vec![("s", ok)]).unwrap(),
        RecordBatch::try_from_iter(vec![("s", bad)]).unwrap(),
    ];
    let proj = format!(
        r#"{{"op":"project","input":{SCAN},"exprs":[
            {{"expr":{{"e":"cast","input":{},"dtype":"int64","try_cast":false}},"alias":"q"}}]}}"#,
        col("s")
    );
    let json = format!(r#"{{"op":"limit","input":{proj},"n":3,"offset":0}}"#);
    let p = plan(&json);

    let got = execute_streaming_parallel(&p, &[src], 4, 0).expect(
        "the parallel limit must stop before the bad batch, exactly as the sequential one does",
    );
    assert_eq!(rows(&got), vec!["1", "2", "3"]);
}
