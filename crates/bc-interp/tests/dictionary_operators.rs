//! Every operator must produce the same relation over a dictionary-encoded column as over
//! that column decoded.
//!
//! # Why this exists
//!
//! `bc_py::normalize_to` decodes every `Dictionary` column to its value type **at the FFI
//! boundary**, so today no dictionary reaches the engine from Python at all. That single
//! decode sits above every dictionary-native fast path the engine has —
//! `try_dict_compare`, the `InList` dictionary path, and `assign_groups`' dictionary
//! grouping — which makes each of them reachable only from a Rust caller that builds a
//! dictionary directly, i.e. only from its own unit test. Preserving the encoding across
//! the boundary is `rfc-streaming-executor.md` Proposal 3 and the top remaining item on
//! `competitor_technique_review.md`'s backlog.
//!
//! The blocker recorded in `normalize.rs`'s own `NOTE` is not the kernels: it is that a
//! preserved `Dictionary` propagates into intermediate schemas, so an operator that decodes
//! one while reusing its input's schema fails Arrow's `RecordBatch::try_new` validation on
//! the type mismatch. That is an operator-by-operator question, and this file is the audit
//! of it. `bc_interp::execute` is the engine's correctness oracle, so proving the answer
//! here is the strongest statement that can be made without the Python differential suite.
//!
//! # What "the same relation" means here
//!
//! The decoded run is the oracle. For each operator the identical plan is run twice — once
//! with the column dictionary-encoded, once with it decoded — and the two results are
//! compared **after** decoding any dictionary column in the output. So a dictionary that
//! survives to the output is allowed and is the point; a dictionary that changes a *value*,
//! a row count, or an order is not.
//!
//! Comparison is order-sensitive on purpose for `Sort`, because an order-independent
//! comparison cannot see a sort bug. Everything else is compared as a multiset, because
//! aggregation and join output order is not part of their contract.
//!
//! # Why there is no memory-budget dimension here
//!
//! The obvious next axis is spill, and this repo has good reason to reach for it — a
//! `sort(descending=True)` once returned unsorted data under spill while every gate stayed
//! green. It was tried here and **removed deliberately**, because it is not a sound
//! equivalence axis for this particular property: a dictionary-encoded column genuinely
//! occupies less memory than the same column decoded, which is the entire reason the encoding
//! exists. So under a budget the two runs are *supposed* to be able to diverge — one fitting
//! where the other spills, or errors — and an assertion that they agree would be asserting
//! that dictionary encoding does not save memory.
//!
//! (Passing a budget of 1 also does not reach a spill at all: `execute_streaming` raises
//! "operator state exceeds the memory budget and cannot spill: the streaming sort does not
//! spill" rather than handing off, so the variant was testing an error path, not the spilling
//! executor.) Spill behaviour over dictionaries is worth testing; it needs its own harness
//! that compares against a spilled *oracle*, not against the other encoding.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, DictionaryArray, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Int32Type};
use bc_expr::Expr;
use bc_ir::{
    AggFunc, AggregateItem, JoinOutputCol, JoinSide, JoinType, ProjectionItem, RangeCondition,
    RangeOp, RelOp, SortKey, WindowFn, WindowFunc,
};

/// Rows per morsel, deliberately more than one morsel's worth of data so an operator that
/// mishandles a dictionary only on a later batch is still caught.
const ROWS: usize = 300;

/// Distinct dictionary values. Small, so grouping and joining actually collapse rows, which
/// is what makes the dictionary path worth taking in the first place.
const CARDINALITY: usize = 7;

/// Every seventh row is NULL.
///
/// A dictionary has **two** places a null can live — a null *key* (this row has no value) and a
/// null entry in the *values* array (this row points at a null) — and they are different
/// physical encodings of the same logical row. Both are built below, because a key path that
/// reads one and not the other is wrong in a way no non-null test can see, and because
/// `null_mask`, `NULL_HASH` and the join's null exclusion all read nullness off the key column
/// whose encoding just changed.
fn label(i: usize) -> Option<String> {
    if i % 7 == 3 {
        None
    } else {
        Some(format!("cat-{:02}", i % CARDINALITY))
    }
}

/// One batch with a dictionary-encoded `k` and a plain `v`, and the same batch with `k`
/// decoded. Identical values, so the two plans below differ *only* in the encoding.
///
/// `null_in_values` selects which of the two null encodings the dictionary uses: a null key
/// pointing at nothing, or a valid key pointing at a null dictionary entry. The decoded oracle
/// is the same array either way, which is exactly the point.
fn pair(offset: usize, rows: usize, null_in_values: bool) -> (RecordBatch, RecordBatch) {
    let labels: Vec<Option<String>> = (offset..offset + rows).map(label).collect();
    let opt: Vec<Option<&str>> = labels.iter().map(|s| s.as_deref()).collect();
    let values: ArrayRef = Arc::new(StringArray::from(opt.clone()));

    let dict: ArrayRef = if null_in_values {
        // Every row has a valid key; the *dictionary* carries the null. Built by hand because
        // the `collect` below folds nulls into null keys instead.
        let mut distinct: Vec<Option<&str>> = vec![None];
        for v in opt.iter().flatten() {
            if !distinct.contains(&Some(*v)) {
                distinct.push(Some(*v));
            }
        }
        let keys: Vec<i32> = opt
            .iter()
            .map(|v| distinct.iter().position(|d| d == v).unwrap() as i32)
            .collect();
        Arc::new(
            DictionaryArray::<Int32Type>::try_new(
                arrow::array::Int32Array::from(keys),
                Arc::new(StringArray::from(distinct)),
            )
            .expect("dict with a null value"),
        )
    } else {
        Arc::new(opt.iter().copied().collect::<DictionaryArray<Int32Type>>())
    };

    let v: ArrayRef = Arc::new(Int64Array::from(
        (offset..offset + rows)
            .map(|i| i as i64)
            .collect::<Vec<_>>(),
    ));

    // Nullability is stated rather than inferred. `try_from_iter` derives it from
    // `null_count() > 0`, which makes the *empty* batch below declare `k` non-nullable while
    // its siblings declare it nullable — four sources with three schemas, which the engine
    // rightly rejects. That is a property of this harness, not of the engine, and pinning it
    // here is what keeps a real disagreement from hiding behind a schema error.
    let dict_batch =
        RecordBatch::try_from_iter_with_nullable(vec![("k", dict, true), ("v", v.clone(), false)])
            .expect("dict");
    let plain_batch =
        RecordBatch::try_from_iter_with_nullable(vec![("k", values, true), ("v", v, false)])
            .expect("plain");
    (dict_batch, plain_batch)
}

/// The input shapes every operator is checked over.
///
/// More than one morsel, so an operator that mishandles a dictionary only on a later batch is
/// caught; an **empty** morsel, because a zero-row batch is where a schema is carried without
/// any values to infer it from; and both null encodings.
fn shapes() -> Vec<(&'static str, Vec<RecordBatch>, Vec<RecordBatch>)> {
    let mut out = Vec::new();
    for (tag, null_in_values) in [("null keys", false), ("null dictionary values", true)] {
        let (d0, p0) = pair(0, ROWS, null_in_values);
        let (d1, p1) = pair(ROWS, ROWS, null_in_values);
        // A single row exercises the paths that special-case "fewer than a morsel".
        let (d2, p2) = pair(2 * ROWS, 1, null_in_values);
        let (de, pe) = pair(0, 0, null_in_values);
        out.push((tag, vec![d0, de.clone(), d1, d2], vec![p0, pe, p1, p2]));
    }
    out
}

fn col(name: &str) -> Expr {
    Expr::Col { name: name.into() }
}

fn item(expr: Expr, alias: &str) -> ProjectionItem {
    ProjectionItem {
        expr,
        alias: alias.into(),
    }
}

/// Decode any dictionary column so the comparison is over values, not encodings.
fn decoded(batches: &[RecordBatch]) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    for b in batches {
        let cols: Vec<ArrayRef> = b
            .columns()
            .iter()
            .map(|c| match c.data_type() {
                DataType::Dictionary(_, value) => {
                    arrow::compute::cast(c, value).expect("decode dictionary")
                }
                _ => c.clone(),
            })
            .collect();
        for r in 0..b.num_rows() {
            let mut row = Vec::with_capacity(cols.len());
            for c in &cols {
                // Rendering through the formatter keeps this type-agnostic, which matters
                // because the whole question is whether a type changed.
                let s = arrow::util::display::array_value_to_string(c.as_ref(), r)
                    .expect("render value");
                row.push(s);
            }
            rows.push(row);
        }
    }
    rows
}

/// The three executors a query can actually take.
///
/// Checking only `execute` would be checking the one path a real query does *not* use:
/// `EngineConfig.streaming` defaults true, so `execute_streaming`/`execute_streaming_parallel`
/// are the default. They share the `ops` module with the sequential path but not their
/// batch-assembly or scheduling, and a schema mismatch is raised by whichever code builds the
/// output batch — so agreeing on one executor says nothing about the others.
#[derive(Clone, Copy)]
enum Executor {
    Sequential,
    Streaming,
    StreamingParallel,
}

impl Executor {
    fn name(self) -> &'static str {
        match self {
            Executor::Sequential => "execute",
            Executor::Streaming => "execute_streaming",
            Executor::StreamingParallel => "execute_streaming_parallel",
        }
    }

    fn run(self, plan: &RelOp, sources: &[Vec<RecordBatch>]) -> Vec<RecordBatch> {
        let r = match self {
            Executor::Sequential => bc_interp::execute(plan, sources),
            Executor::Streaming => bc_interp::execute_streaming(plan, sources, 0),
            Executor::StreamingParallel => {
                bc_interp::execute_streaming_parallel(plan, sources, 4, 0)
            }
        };
        r.unwrap_or_else(|e| panic!("{}: plan failed to run: {e}", self.name()))
    }
}

/// Run `plan` over both encodings, on every executor, and assert the relations agree.
///
/// `ordered` selects an order-sensitive comparison, which `Sort` needs and which an
/// order-independent helper would silently hide.
fn assert_agrees(what: &str, plan_for: impl Fn() -> RelOp, ordered: bool) {
    for ex in [
        Executor::Sequential,
        Executor::Streaming,
        Executor::StreamingParallel,
    ] {
        for (shape, dict_src, plain_src) in shapes() {
            let case = format!("{what} on {} [{shape}]", ex.name());
            let dict_out = ex.run(&plan_for(), &[dict_src.clone(), dict_src]);
            let plain_out = ex.run(&plan_for(), &[plain_src.clone(), plain_src]);

            let mut got = decoded(&dict_out);
            let mut want = decoded(&plain_out);
            assert_eq!(
                got.len(),
                want.len(),
                "{case}: row count differs — dictionary {} vs decoded {}",
                got.len(),
                want.len()
            );
            if !ordered {
                got.sort();
                want.sort();
            }
            assert_eq!(
                got, want,
                "{case}: the dictionary-encoded run disagrees with the decoded oracle"
            );
        }
    }
}

#[test]
fn filter_over_a_dictionary_agrees_with_the_decoded_oracle() {
    assert_agrees(
        "filter",
        || RelOp::Filter {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            predicate: Expr::Binary {
                op: bc_expr::BinaryOp::Eq,
                left: Box::new(col("k")),
                right: Box::new(Expr::Lit {
                    value: bc_expr::Literal::Str("cat-03".into()),
                }),
            },
        },
        false,
    );
}

#[test]
fn project_over_a_dictionary_agrees_with_the_decoded_oracle() {
    // The bare pass-through is the interesting one: it is where a preserved dictionary
    // would reach an output schema built from the input's.
    assert_agrees(
        "project",
        || RelOp::Project {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            exprs: vec![item(col("k"), "k"), item(col("v"), "v")],
        },
        false,
    );
}

#[test]
fn group_by_a_dictionary_agrees_with_the_decoded_oracle() {
    assert_agrees(
        "aggregate",
        || RelOp::Aggregate {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            group_keys: vec![item(col("k"), "k")],
            aggregates: vec![
                AggregateItem {
                    func: AggFunc::Sum,
                    input: Some(col("v")),
                    input2: None,
                    param: None,
                    alias: "s".into(),
                },
                AggregateItem {
                    func: AggFunc::CountStar,
                    input: None,
                    input2: None,
                    param: None,
                    alias: "n".into(),
                },
            ],
        },
        false,
    );
}

#[test]
fn aggregating_a_dictionary_value_agrees_with_the_decoded_oracle() {
    // The dictionary as an aggregate *input* rather than a key.
    assert_agrees(
        "aggregate over a dictionary value",
        || RelOp::Aggregate {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            group_keys: vec![],
            aggregates: vec![AggregateItem {
                func: AggFunc::Min,
                input: Some(col("k")),
                input2: None,
                param: None,
                alias: "lo".into(),
            }],
        },
        false,
    );
}

#[test]
fn sorting_by_a_dictionary_agrees_with_the_decoded_oracle() {
    // Order-sensitive: an order-independent comparison cannot see a sort bug.
    assert_agrees(
        "sort",
        || RelOp::Sort {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            keys: vec![
                SortKey {
                    expr: col("k"),
                    descending: false,
                    nulls_first: false,
                },
                SortKey {
                    expr: col("v"),
                    descending: false,
                    nulls_first: false,
                },
            ],
            limit: None,
        },
        true,
    );
}

#[test]
fn top_n_by_a_dictionary_agrees_with_the_decoded_oracle() {
    assert_agrees(
        "top-n",
        || RelOp::Sort {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            keys: vec![
                SortKey {
                    expr: col("k"),
                    descending: true,
                    nulls_first: false,
                },
                SortKey {
                    expr: col("v"),
                    descending: false,
                    nulls_first: false,
                },
            ],
            limit: Some(20),
        },
        true,
    );
}

#[test]
fn distinct_over_a_dictionary_agrees_with_the_decoded_oracle() {
    // Named in `normalize.rs`'s NOTE as the operator that exposed the schema mismatch.
    assert_agrees(
        "distinct",
        || RelOp::Distinct {
            input: Box::new(RelOp::Project {
                input: Box::new(RelOp::Scan { source_id: 0 }),
                exprs: vec![item(col("k"), "k")],
            }),
        },
        false,
    );
}

#[test]
fn joining_on_a_dictionary_key_agrees_with_the_decoded_oracle() {
    assert_agrees(
        "hash join",
        || RelOp::HashJoin {
            left: Box::new(RelOp::Scan { source_id: 0 }),
            right: Box::new(RelOp::Distinct {
                input: Box::new(RelOp::Project {
                    input: Box::new(RelOp::Scan { source_id: 1 }),
                    exprs: vec![item(col("k"), "rk")],
                }),
            }),
            left_keys: vec!["k".into()],
            right_keys: vec!["rk".into()],
            join_type: JoinType::Inner,
            output: vec![
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "k".into(),
                    alias: "k".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "v".into(),
                    alias: "v".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Right,
                    name: "rk".into(),
                    alias: "rk".into(),
                },
            ],
            strategy: Default::default(),
        },
        false,
    );
}

#[test]
fn union_over_a_dictionary_agrees_with_the_decoded_oracle() {
    // Two inputs whose *encodings* must agree for the concatenation to be well-typed. If a
    // dictionary ever reaches only one side of a union, this is where it shows.
    assert_agrees(
        "union",
        || RelOp::Union {
            inputs: vec![RelOp::Scan { source_id: 0 }, RelOp::Scan { source_id: 1 }],
            distinct: false,
        },
        false,
    );
}

#[test]
fn windowing_over_a_dictionary_partition_agrees_with_the_decoded_oracle() {
    // Window straddles two crates (kernels in bc-runtime, orchestration in bc-interp) and
    // partitions by key, so it is the other operator that derives key identity from a
    // possibly-dictionary column.
    assert_agrees(
        "window",
        || RelOp::Window {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            partition_keys: vec![col("k")],
            order_keys: vec![SortKey {
                expr: col("v"),
                descending: false,
                nulls_first: false,
            }],
            functions: vec![
                WindowFunc {
                    func: WindowFn::RowNumber,
                    input: None,
                    offset: 1,
                    frame: None,
                    alias: "rn".into(),
                },
                WindowFunc {
                    func: WindowFn::Sum,
                    input: Some(col("v")),
                    offset: 1,
                    frame: None,
                    alias: "rsum".into(),
                },
            ],
            rank_limit: None,
        },
        false,
    );
}

#[test]
fn a_range_join_over_a_dictionary_agrees_with_the_decoded_oracle() {
    // `supported_key_type` accepts `Dictionary(_, v)` recursively, but `u64_order_keys`
    // declines a dictionary, so this lands on the row-encoder axis — where the two sides'
    // encodings have to agree for the same reason the hash join's do.
    assert_agrees(
        "range join",
        || RelOp::RangeJoin {
            left: Box::new(RelOp::Scan { source_id: 0 }),
            right: Box::new(RelOp::Project {
                input: Box::new(RelOp::Scan { source_id: 1 }),
                exprs: vec![item(col("k"), "rk")],
            }),
            conditions: vec![RangeCondition {
                left_key: "k".into(),
                right_key: "rk".into(),
                op: RangeOp::Le,
            }],
            join_type: JoinType::Inner,
            output: vec![
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "k".into(),
                    alias: "k".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Right,
                    name: "rk".into(),
                    alias: "rk".into(),
                },
            ],
        },
        false,
    );
}

#[test]
fn limit_over_a_dictionary_agrees_with_the_decoded_oracle() {
    assert_agrees(
        "limit",
        || RelOp::Limit {
            input: Box::new(RelOp::Sort {
                input: Box::new(RelOp::Scan { source_id: 0 }),
                keys: vec![SortKey {
                    expr: col("v"),
                    descending: false,
                    nulls_first: false,
                }],
                limit: None,
            }),
            n: 25,
            offset: 5,
        },
        true,
    );
}
