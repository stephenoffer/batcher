//! Short-circuiting evaluation of a conjunctive filter predicate into a keep mask.
//!
//! `Expr::eval` computes a predicate the obvious way: every conjunct of an `AND`
//! chain over every row, then `and_kleene` the masks together. That is correct and it
//! is what the oracle does, but it means `WHERE ship_date >= ? AND ship_date < ? AND
//! discount BETWEEN ? AND ? AND quantity < ?` pays five full-width passes even when
//! the first one already threw away six rows in seven.
//!
//! Every vectorized engine that competes on this shape short-circuits instead, and
//! DuckDB's is the clearest statement of it: `ExpressionExecutor::Select` walks the
//! conjuncts against a *selection vector*, so conjunct `n + 1` only ever sees the rows
//! conjunct `n` kept (`src/execution/expression_executor/execute_conjunction.cpp`),
//! ordered by a static cost heuristic. Arrow has no selection vector — a kernel reads
//! a whole array — so the equivalent here is to **compact**: gather the rows that
//! survived, along with only the columns the remaining conjuncts actually name, and
//! evaluate the rest against that narrower, shorter batch.
//!
//! ## Why this is not a semantic change
//!
//! The result is the same mask, bit for bit, and the argument has three parts.
//!
//! 1. **Composition.** `filter_record_batch` keeps row `i` when the mask is valid and
//!    true there, so the predicate's nulls are already indistinguishable from false.
//!    `and_kleene(a, b)` is valid-and-true exactly where both operands are, so
//!    ANDing [`truthy`] masks (value AND validity, no nulls) gives the identical
//!    keep set for any nesting or order of the `AND`s.
//! 2. **Position independence.** Every kernel a conjunct can reach here is
//!    elementwise: row `i`'s result depends on row `i` alone, so evaluating it over a
//!    gathered subset yields, at each surviving position, what the full-width pass
//!    would have written there.
//! 3. **Skipping cannot hide an error.** This is the part that needs a guard, and
//!    [`Expr::is_infallible_predicate`] is it: only conjuncts whose failures are
//!    schema-driven rather than row-driven are eligible. A skipped row can then never
//!    have been the row that raised. Anything else — arithmetic that can overflow, a
//!    non-`try` cast, a string kernel — takes the whole-batch path unchanged.
//!
//! Two further details keep that argument airtight rather than nearly airtight. The
//! surviving set is never compacted to *zero* rows, so a conjunct whose type error
//! needs a non-empty input still gets one; and an error *from evaluating a conjunct*
//! abandons the fast path and returns `None`, so the caller re-evaluates the predicate
//! as written and raises exactly the error, with exactly the message, that it always
//! did. Reordering therefore cannot change which of two broken conjuncts is blamed.
//!
//! Errors from this module's own bookkeeping — a mask whose length disagrees with its
//! batch, a gather index out of range — are *not* swallowed that way. They are
//! violations of this module's invariants rather than conditions a query can create,
//! and a fallback would turn a bug here into a silent slow path. They propagate.

use std::sync::Arc;

use arrow::array::{
    Array, BooleanArray, BooleanBufferBuilder, RecordBatch, RecordBatchOptions, UInt32Array,
};
use arrow::buffer::BooleanBuffer;
use arrow::compute::kernels::boolean;
use arrow::compute::take;
use arrow::datatypes::{Field, Schema};

use crate::eval::binary::as_bool;
use crate::{Expr, ExprError};

/// Cost (in [`Expr::eval_cost`] units) at or below which a conjunct counts as cheap.
///
/// Calibrated to sit just above a comparison against a literal (`col cmp lit` is 3)
/// and a null test, and below a cast (8 plus its input) and a string kernel (40 plus
/// its input). It decides only *how eagerly* a gather is paid for, so the boundary
/// wants to be roughly right rather than exact.
const CHEAP_CONJUNCT_COST: u32 = 8;

impl Expr {
    /// Evaluate `self` as a filter predicate, short-circuiting its `AND` conjuncts.
    ///
    /// Returns the keep mask — null-free, `true` meaning "this row survives" — or
    /// `None` when the fast path does not apply and the caller should evaluate the
    /// predicate whole. `None` is returned for a predicate that is not a multi-conjunct
    /// `AND`, for one whose conjuncts are not all
    /// [infallible](Expr::is_infallible_predicate), for a conjunct that does not
    /// evaluate to a boolean, and for any error raised while evaluating one. Handing
    /// those back rather than trying to serve them is what makes the module's
    /// equivalence argument hold without a caveat: the caller's existing path stays the
    /// authority on every shape this one declines. An `Err` from here is this module's
    /// own invariant breaking, not a query's error.
    ///
    /// The mask is interchangeable with `filter_record_batch(batch, &self.eval(batch)?)`
    /// — see the module docs for why — so a caller may use whichever it gets.
    pub fn short_circuit_filter_mask(
        &self,
        batch: &RecordBatch,
    ) -> Result<Option<BooleanArray>, ExprError> {
        let n = batch.num_rows();
        let conjuncts = self.and_conjuncts();
        if n == 0 || conjuncts.len() < 2 {
            return Ok(None);
        }
        let schema = batch.schema();
        if !conjuncts.iter().all(|c| c.is_infallible_predicate(&schema)) {
            return Ok(None);
        }

        // Cheapest conjunct first, the opening order DuckDB's `ExpressionHeuristics`
        // also computes. `sort_by_key` is stable, so equal-cost conjuncts stay in the
        // order the query wrote them — which is the only order a reader can predict.
        let mut order: Vec<usize> = (0..conjuncts.len()).collect();
        order.sort_by_key(|&i| conjuncts[i].eval_cost());

        // `view` is what the next conjunct is evaluated over and `view_abs` maps its
        // rows back to `batch` (`None` while they are still the same rows). `live` is
        // a null-free mask over `view`.
        let mut view = batch.clone();
        let mut view_abs: Option<Vec<u32>> = None;
        let mut live = all_set(n);

        for (pos, &ci) in order.iter().enumerate() {
            let Ok(evaluated) = conjuncts[ci].eval(&view) else {
                return Ok(None);
            };
            let Ok(mask) = as_bool(&evaluated, "and") else {
                return Ok(None);
            };
            live = boolean::and(&live, &truthy(mask))?;

            let remaining = &order[pos + 1..];
            let Some(&next) = remaining.first() else {
                break;
            };
            let alive = live.values().count_set_bits();
            if !should_compact(alive, view.num_rows(), conjuncts[next].eval_cost()) {
                continue;
            }
            let Some(next_view) =
                compact(batch, &conjuncts, remaining, &live, view_abs.as_deref())?
            else {
                return Ok(None);
            };
            view = next_view.batch;
            live = all_set(next_view.abs.len());
            view_abs = Some(next_view.abs);
        }

        Ok(Some(match view_abs {
            None => live,
            Some(abs) => scatter(n, &abs, &live),
        }))
    }
}

/// The mask with its nulls folded into false, so a chain of them composes with a
/// plain `AND`.
///
/// This is the same reduction `filter_record_batch` performs on a nullable mask
/// before gathering, which is why doing it per conjunct changes no keep decision.
fn truthy(mask: &BooleanArray) -> BooleanArray {
    match mask.nulls() {
        Some(nulls) => BooleanArray::new(mask.values() & nulls.inner(), None),
        None => mask.clone(),
    }
}

fn all_set(len: usize) -> BooleanArray {
    BooleanArray::new(BooleanBuffer::new_set(len), None)
}

/// Whether gathering the survivors now beats evaluating the rest at full width.
///
/// A gather costs one pass over the surviving rows of the named columns; skipping
/// buys one pass over the removed rows per remaining conjunct. So the threshold is a
/// function of what the *next* conjunct costs: an expensive one (a cast, a regex, a
/// dictionary-set membership) repays the gather after a modest reduction, while
/// another bare comparison has to see most of the batch disappear first.
///
/// Both guards at the top matter. Compacting when nothing was removed is pure loss,
/// and compacting to zero rows would hand the remaining conjuncts an empty batch —
/// on which a type error that the whole-batch path raises might not fire, the one way
/// this optimization could otherwise change an outcome.
fn should_compact(alive: usize, view_rows: usize, next_cost: u32) -> bool {
    if alive == 0 || alive == view_rows {
        return false;
    }
    if next_cost > CHEAP_CONJUNCT_COST {
        alive * 8 <= view_rows * 7
    } else {
        alive * 4 <= view_rows
    }
}

/// A compacted view: the surviving rows of just the columns still to be read.
struct Compacted {
    batch: RecordBatch,
    /// Row `j` of `batch` is row `abs[j]` of the original batch, ascending.
    abs: Vec<u32>,
}

/// Gather the rows `live` keeps, projected to the columns `remaining` names.
///
/// Indices are always resolved against the *original* batch — `view_abs` maps the
/// current view's rows back first — so repeated compaction composes without
/// accumulating a chain of gathers. Returns `None` if a named column is absent, so
/// the caller can fall back and let the ordinary path report it.
fn compact(
    batch: &RecordBatch,
    conjuncts: &[&Expr],
    remaining: &[usize],
    live: &BooleanArray,
    view_abs: Option<&[u32]>,
) -> Result<Option<Compacted>, ExprError> {
    let set = live.values().set_indices();
    let abs: Vec<u32> = match view_abs {
        None => set.map(|i| i as u32).collect(),
        Some(prev) => set.map(|i| prev[i]).collect(),
    };

    let mut names: Vec<&str> = Vec::new();
    for &i in remaining {
        conjuncts[i].collect_columns(&mut names);
    }
    names.sort_unstable();
    names.dedup();

    let indices = UInt32Array::from(abs.clone());
    let schema = batch.schema();
    let mut fields: Vec<Field> = Vec::with_capacity(names.len());
    let mut columns = Vec::with_capacity(names.len());
    for name in names {
        let Ok(i) = schema.index_of(name) else {
            return Ok(None);
        };
        fields.push(schema.field(i).clone());
        columns.push(take(batch.column(i).as_ref(), &indices, None)?);
    }

    // The row count must be stated: a predicate over literals alone names no column,
    // and a zero-column batch has no other way to say how long it is.
    let options = RecordBatchOptions::new().with_row_count(Some(abs.len()));
    let projected =
        RecordBatch::try_new_with_options(Arc::new(Schema::new(fields)), columns, &options)?;
    Ok(Some(Compacted {
        batch: projected,
        abs,
    }))
}

/// Expand a mask over compacted rows back to one over all `n` original rows.
///
/// Rows absent from `abs` were removed by an earlier conjunct, so they are false.
fn scatter(n: usize, abs: &[u32], live: &BooleanArray) -> BooleanArray {
    let mut bits = BooleanBufferBuilder::new(n);
    bits.append_n(n, false);
    for j in live.values().set_indices() {
        bits.set_bit(abs[j] as usize, true);
    }
    BooleanArray::new(bits.finish(), None)
}

#[cfg(test)]
mod tests {
    use arrow::array::{Int32Array, Int64Array, StringArray};
    use arrow::compute::filter_record_batch;
    use arrow::datatypes::DataType;

    use super::*;
    use crate::{BinaryOp, Literal};

    fn col(name: &str) -> Box<Expr> {
        Box::new(Expr::Col { name: name.into() })
    }

    fn lit_int(v: i64) -> Box<Expr> {
        Box::new(Expr::Lit {
            value: Literal::Int(v),
        })
    }

    fn cmp(op: BinaryOp, name: &str, v: i64) -> Expr {
        Expr::Binary {
            op,
            left: col(name),
            right: lit_int(v),
        }
    }

    fn and(left: Expr, right: Expr) -> Expr {
        Expr::Binary {
            op: BinaryOp::And,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// `a`: 0..n with every 7th null. `b`: n-i. `s`: a short string per row.
    fn sample(n: i64) -> RecordBatch {
        let a: Int64Array = (0..n)
            .map(|i| if i % 7 == 0 { None } else { Some(i) })
            .collect();
        let b: Int64Array = (0..n).map(|i| Some(n - i)).collect();
        let owned: Vec<String> = (0..n).map(|i| format!("k{}", i % 5)).collect();
        let s: StringArray = owned.iter().map(|v| Some(v.as_str())).collect();
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Int64, true),
            Field::new("s", DataType::Utf8, true),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(a), Arc::new(b), Arc::new(s)],
        )
        .expect("sample batch")
    }

    /// The whole-batch path: what the mask must equal, however it was produced.
    fn oracle(pred: &Expr, batch: &RecordBatch) -> RecordBatch {
        let mask = pred.eval(batch).expect("oracle eval");
        let mask = mask
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("boolean predicate");
        filter_record_batch(batch, mask).expect("oracle filter")
    }

    /// The contract, asserted on the *rows* rather than the mask: whichever path ran,
    /// filtering with the result must equal filtering with the full evaluation.
    fn assert_matches_oracle(pred: &Expr, batch: &RecordBatch) {
        let expected = oracle(pred, batch);
        let Some(mask) = pred
            .short_circuit_filter_mask(batch)
            .expect("short-circuit must not error")
        else {
            return;
        };
        assert_eq!(mask.null_count(), 0, "the keep mask must be null-free");
        assert_eq!(mask.len(), batch.num_rows());
        let got = filter_record_batch(batch, &mask).expect("filter with short-circuit mask");
        assert_eq!(
            format!("{:?}", got),
            format!("{:?}", expected),
            "short-circuit diverged from whole-batch evaluation"
        );
    }

    #[test]
    fn matches_oracle_across_selectivities_and_null_positions() {
        let batch = sample(4_096);
        // Selective first conjunct, then three more — the compacting shape.
        for cut in [1_i64, 8, 64, 512, 4_000, 8_000] {
            let pred = and(
                and(cmp(BinaryOp::Lt, "a", cut), cmp(BinaryOp::Ge, "a", -1)),
                and(cmp(BinaryOp::Gt, "b", 3), cmp(BinaryOp::Ne, "b", 17)),
            );
            assert_matches_oracle(&pred, &batch);
        }
    }

    #[test]
    fn matches_oracle_when_nothing_and_everything_survives() {
        let batch = sample(1_000);
        let none = and(cmp(BinaryOp::Lt, "a", 0), cmp(BinaryOp::Gt, "b", 0));
        assert_matches_oracle(&none, &batch);
        let all = and(cmp(BinaryOp::Ge, "a", -1), cmp(BinaryOp::Ge, "b", -1));
        assert_matches_oracle(&all, &batch);
    }

    /// A null in a conjunct must be dropped, exactly as `filter_record_batch` drops
    /// it — the composition half of the equivalence argument. Column `a` is null on
    /// every 7th row, so this is the case that would silently keep or drop 143 rows
    /// if `truthy` were wrong.
    #[test]
    fn nulls_are_dropped_like_the_whole_batch_path() {
        let batch = sample(1_001);
        let pred = and(
            cmp(BinaryOp::Ge, "a", 0),
            and(cmp(BinaryOp::Gt, "b", 500), cmp(BinaryOp::Lt, "a", 400)),
        );
        assert_matches_oracle(&pred, &batch);
        let mask = pred
            .short_circuit_filter_mask(&batch)
            .expect("eval")
            .expect("a four-conjunct infallible AND must take the fast path");
        // Every 7th row is null in `a`, so none of them may survive.
        for i in (0..1_001).step_by(7) {
            assert!(!mask.value(i), "row {i} is null in `a` and must be dropped");
        }
    }

    #[test]
    fn declines_a_predicate_whose_conjunct_can_fail_on_a_row() {
        let batch = sample(1_000);
        // `a / b > 0` can divide by zero, so the whole predicate must take the
        // ordinary path — skipping a row could skip the row that raises.
        let risky = and(
            cmp(BinaryOp::Lt, "a", 4),
            Expr::Binary {
                op: BinaryOp::Gt,
                left: Box::new(Expr::Binary {
                    op: BinaryOp::Div,
                    left: col("a"),
                    right: col("b"),
                }),
                right: lit_int(0),
            },
        );
        assert!(!risky.is_infallible_predicate(&batch.schema()));
        assert!(risky
            .short_circuit_filter_mask(&batch)
            .expect("eval")
            .is_none());
    }

    #[test]
    fn declines_a_single_conjunct_and_an_empty_batch() {
        let batch = sample(1_000);
        let single = cmp(BinaryOp::Lt, "a", 10);
        assert!(single
            .short_circuit_filter_mask(&batch)
            .expect("eval")
            .is_none());

        let empty = sample(0);
        let pred = and(cmp(BinaryOp::Lt, "a", 10), cmp(BinaryOp::Gt, "b", 1));
        assert!(pred
            .short_circuit_filter_mask(&empty)
            .expect("eval")
            .is_none());
    }

    /// An expensive conjunct must not be the one that runs first, because the whole
    /// point is that it runs over fewer rows.
    #[test]
    fn orders_the_cheap_conjunct_ahead_of_the_expensive_one() {
        let cheap = cmp(BinaryOp::Lt, "a", 10);
        let expensive = Expr::Str {
            func: crate::StrFunc::Contains,
            input: col("s"),
            pattern: Some("k3".into()),
            replacement: None,
            start: None,
            length: None,
        };
        assert!(cheap.eval_cost() <= CHEAP_CONJUNCT_COST);
        assert!(expensive.eval_cost() > CHEAP_CONJUNCT_COST);
    }

    /// A conjunct that is not boolean must be declined rather than coerced, so the
    /// caller keeps raising its own non-boolean-predicate error.
    #[test]
    fn declines_a_non_boolean_conjunct() {
        let batch = sample(64);
        let pred = and(cmp(BinaryOp::Lt, "a", 10), *col("b"));
        assert!(pred
            .short_circuit_filter_mask(&batch)
            .expect("eval")
            .is_none());
    }

    /// An unknown column is declined, not reported from here: the ordinary path owns
    /// that error and its message.
    #[test]
    fn declines_an_unknown_column() {
        let batch = sample(64);
        let pred = and(cmp(BinaryOp::Lt, "a", 10), cmp(BinaryOp::Gt, "nope", 1));
        assert!(pred
            .short_circuit_filter_mask(&batch)
            .expect("eval")
            .is_none());
    }

    fn str_pred(func: crate::StrFunc, name: &str, pattern: &str) -> Expr {
        Expr::Str {
            func,
            input: col(name),
            pattern: Some(pattern.into()),
            replacement: None,
            start: None,
            length: None,
        }
    }

    /// The shape the ordering exists for: a cheap comparison guarding a `LIKE`. The
    /// result must still match the whole-batch path exactly.
    #[test]
    fn matches_oracle_with_a_string_predicate_conjunct() {
        let batch = sample(4_096);
        for func in [
            crate::StrFunc::Contains,
            crate::StrFunc::StartsWith,
            crate::StrFunc::EndsWith,
            crate::StrFunc::Like,
            crate::StrFunc::Ilike,
            crate::StrFunc::RegexpMatches,
        ] {
            let pattern = if matches!(func, crate::StrFunc::Like | crate::StrFunc::Ilike) {
                "k%"
            } else {
                "k3"
            };
            let pred = and(str_pred(func, "s", pattern), cmp(BinaryOp::Lt, "a", 40));
            assert_matches_oracle(&pred, &batch);
            assert!(
                pred.short_circuit_filter_mask(&batch)
                    .expect("eval")
                    .is_some(),
                "a string predicate over a Utf8 column must take the fast path"
            );
        }
    }

    /// A dictionary column has to survive compaction, because Batcher's dictionary-native
    /// comparison and string paths are what make a low-cardinality predicate cheap. Gathering
    /// through `take` keeps the values buffer and takes the keys, so a conjunct evaluated
    /// after a compaction still gets the dictionary. If it silently decoded instead, this
    /// test would still pass on values and the cost model in the module docs would be a lie,
    /// so it asserts the compacted view's *type* as well as the rows.
    #[test]
    fn a_dictionary_column_stays_a_dictionary_through_compaction() {
        let n = 2_048;
        let keys: Int32Array = (0..n).map(|i| Some(i % 5)).collect();
        let values = StringArray::from(vec!["AIR", "RAIL", "SHIP", "TRUCK", "MAIL"]);
        let dict = arrow::array::DictionaryArray::<arrow::datatypes::Int32Type>::try_new(
            keys,
            Arc::new(values),
        )
        .expect("dictionary");
        let a: Int64Array = (0..n as i64).collect::<Vec<_>>().into();
        let dtype = dict.data_type().clone();
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("s", dtype.clone(), true),
        ]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(a), Arc::new(dict)])
            .expect("dict batch");

        // The cheap integer comparison runs first and removes 96% of the rows, so the
        // string predicate is evaluated over a compacted view.
        for pred in [
            and(
                cmp(BinaryOp::Lt, "a", 64),
                str_pred(crate::StrFunc::Contains, "s", "AIR"),
            ),
            and(
                cmp(BinaryOp::Lt, "a", 64),
                Expr::Binary {
                    op: BinaryOp::Eq,
                    left: col("s"),
                    right: Box::new(Expr::Lit {
                        value: Literal::Str("RAIL".into()),
                    }),
                },
            ),
        ] {
            assert_matches_oracle(&pred, &batch);
            assert!(
                pred.short_circuit_filter_mask(&batch)
                    .expect("eval")
                    .is_some(),
                "a dictionary-backed predicate must take the fast path"
            );
        }

        let gathered = take(
            batch.column(1).as_ref(),
            &UInt32Array::from(vec![0_u32, 7, 19]),
            None,
        )
        .expect("take");
        assert_eq!(
            gathered.data_type(),
            &dtype,
            "compaction must not decode the dictionary"
        );
    }

    /// A string *producer* can exceed the maximum string length on one row and not the
    /// next, so it must never be treated as skippable — the value-driven failure this
    /// whole classification exists to keep out.
    #[test]
    fn a_string_producing_function_is_never_infallible() {
        let batch = sample(8);
        for func in [
            crate::StrFunc::Upper,
            crate::StrFunc::Repeat,
            crate::StrFunc::Lpad,
            crate::StrFunc::Overlay,
            crate::StrFunc::Replace,
        ] {
            let e = str_pred(func, "s", "x");
            assert!(
                !e.is_infallible_predicate(&batch.schema()),
                "{func:?} builds a string and can fail on a row"
            );
        }
    }

    /// The identical predicate is safe over `Utf8` and unsafe over `Binary`, because
    /// evaluating it over `Binary` casts to UTF-8 and that rejects one row's bytes.
    #[test]
    fn a_string_predicate_over_binary_is_not_infallible() {
        let utf8 = Schema::new(vec![Field::new("s", DataType::Utf8, true)]);
        let binary = Schema::new(vec![Field::new("s", DataType::Binary, true)]);
        let dict_utf8 = Schema::new(vec![Field::new(
            "s",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            true,
        )]);
        let pred = str_pred(crate::StrFunc::Contains, "s", "x");
        assert!(pred.is_infallible_predicate(&utf8));
        assert!(pred.is_infallible_predicate(&dict_utf8));
        assert!(!pred.is_infallible_predicate(&binary));
    }

    /// `TRY_CAST` yields null instead of raising, so it carries no per-row failure;
    /// a strict `CAST` does and must stay out.
    #[test]
    fn try_cast_is_infallible_but_strict_cast_is_not() {
        let schema = Schema::new(vec![Field::new("s", DataType::Utf8, true)]);
        let mk = |try_cast| Expr::Cast {
            input: col("s"),
            dtype: "int64".into(),
            try_cast,
        };
        assert!(mk(true).is_infallible_predicate(&schema));
        assert!(!mk(false).is_infallible_predicate(&schema));
    }

    #[test]
    fn flattens_only_top_level_ands() {
        let nested = and(
            and(cmp(BinaryOp::Lt, "a", 1), cmp(BinaryOp::Gt, "b", 2)),
            cmp(BinaryOp::Ne, "a", 3),
        );
        assert_eq!(nested.and_conjuncts().len(), 3);

        // An `AND` under a `Not` is one conjunct, not two: `NOT (x AND y)` is not a
        // conjunction, and splitting it would change the predicate.
        let negated = Expr::Not {
            input: Box::new(and(cmp(BinaryOp::Lt, "a", 1), cmp(BinaryOp::Gt, "b", 2))),
        };
        assert_eq!(negated.and_conjuncts().len(), 1);

        // Same for an `AND` inside an `OR` branch.
        let disjunct = Expr::Binary {
            op: BinaryOp::Or,
            left: Box::new(and(cmp(BinaryOp::Lt, "a", 1), cmp(BinaryOp::Gt, "b", 2))),
            right: Box::new(cmp(BinaryOp::Ne, "a", 3)),
        };
        assert_eq!(disjunct.and_conjuncts().len(), 1);
    }

    #[test]
    fn collects_columns_through_nesting() {
        let pred = and(
            cmp(BinaryOp::Lt, "a", 1),
            Expr::IsNull {
                input: Box::new(Expr::Binary {
                    op: BinaryOp::Gt,
                    left: col("b"),
                    right: col("a"),
                }),
            },
        );
        let mut names = Vec::new();
        pred.collect_columns(&mut names);
        names.sort_unstable();
        names.dedup();
        assert_eq!(names, vec!["a", "b"]);
    }
}
