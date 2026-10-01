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

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use arrow::array::{Array, BooleanArray, RecordBatch};
use arrow::compute::kernels::boolean;

use crate::eval::coerce::as_bool;
use crate::subset::{all_set, compact_rows, scatter_mask, should_compact, truthy};
use crate::{Expr, ExprError};

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
        self.short_circuit_filter_mask_with(batch, None)
    }

    /// [`Expr::short_circuit_filter_mask`], with the conjunct order taken from what
    /// earlier morsels *measured* rather than from the static cost model alone.
    ///
    /// Pass a [`ConjunctOrder`] built once per Filter operator and shared across its
    /// morsels and its workers. `None` reproduces the static-order behaviour exactly,
    /// which is what the sequential oracle uses.
    pub fn short_circuit_filter_mask_with(
        &self,
        batch: &RecordBatch,
        learned: Option<&ConjunctOrder>,
    ) -> Result<Option<BooleanArray>, ExprError> {
        self.short_circuit_from(batch, learned, None)
    }

    /// `live AND self` as a null-free keep mask over `batch`, for a filter stacked on another
    /// whose rows have not been gathered yet — or `None` when that is not sound and the caller
    /// should gather `live` and apply `self` as it always has.
    ///
    /// A filter over a filter gathers every column twice: once for the rows the inner one
    /// keeps, once for the rows the outer one keeps. When the inner keeps most rows that first
    /// gather is nearly a copy of the batch, and on string columns it is the dominant cost
    /// (`memmove` and `FilterBytes` were ~49% of `l_suppkey <= 8192` under an `IN` over TPC-H).
    /// Treating `live` as one more conjunct, already evaluated, leaves one gather: of the final
    /// rows, by the caller.
    ///
    /// That is the short-circuit path with its first mask given rather than computed, so it
    /// inherits that path's whole argument. `self` must be
    /// [infallible](Expr::is_infallible_predicate), because it may be evaluated at rows `live`
    /// removed; a `live` with no row set is declined, so a conjunct is never handed zero rows;
    /// and an error evaluating `self` declines rather than surfaces, so the caller's path raises
    /// it as written. Whether to gather the live rows before the first conjunct is decided by
    /// that conjunct's cost, exactly as between any two conjuncts ([`should_compact`]).
    pub fn filter_mask_within(
        &self,
        batch: &RecordBatch,
        live: &BooleanArray,
        learned: Option<&ConjunctOrder>,
    ) -> Result<Option<BooleanArray>, ExprError> {
        let n = batch.num_rows();
        if live.len() != n || live.null_count() != 0 {
            return Err(ExprError::Arrow(
                arrow::error::ArrowError::InvalidArgumentError(format!(
                    "filter_mask_within: a {n}-row null-free mask is required"
                )),
            ));
        }
        self.short_circuit_from(batch, learned, Some(live))
    }

    /// The short-circuit loop, optionally starting from a mask an earlier filter computed.
    fn short_circuit_from(
        &self,
        batch: &RecordBatch,
        learned: Option<&ConjunctOrder>,
        initial: Option<&BooleanArray>,
    ) -> Result<Option<BooleanArray>, ExprError> {
        let n = batch.num_rows();
        let raw = self.and_conjuncts();
        // With an initial mask, a single conjunct is already the second stage.
        let fewest = if initial.is_some() { 1 } else { 2 };
        if n == 0 || raw.len() < fewest {
            return Ok(None);
        }
        let schema = batch.schema();
        if !raw.iter().all(|c| c.is_infallible_predicate(&schema)) {
            return Ok(None);
        }
        let units = filter_units(&raw);
        if units.len() < fewest {
            // One fused range is the whole predicate: one pass, nothing to short-circuit.
            return Ok(None);
        }
        let conjuncts: Vec<&Expr> = units.iter().map(|u| u.as_ref()).collect();

        // A `ConjunctOrder` built for a different predicate would index the wrong slots,
        // so a mismatched width is ignored rather than trusted.
        let learned = learned.filter(|l| l.len() == conjuncts.len());
        let order = match learned {
            Some(l) => l.order(&conjuncts),
            None => static_order(&conjuncts),
        };

        // `view` is what the next conjunct is evaluated over and `view_abs` maps its
        // rows back to `batch` (`None` while they are still the same rows). `live` is
        // a null-free mask over `view`.
        let mut view = batch.clone();
        let mut view_abs: Option<Vec<u32>> = None;
        let mut live = match initial {
            Some(given) => given.clone(),
            None => all_set(n),
        };
        if let Some(given) = initial {
            if given.true_count() == 0
                || !compact_before(
                    &mut view,
                    &mut view_abs,
                    &mut live,
                    batch,
                    &conjuncts,
                    &order,
                )?
            {
                return Ok(None);
            }
        }

        for (pos, &ci) in order.iter().enumerate() {
            let rows_in = view.num_rows();
            let started = learned.map(|_| Instant::now());
            let Ok(evaluated) = conjuncts[ci].eval(&view) else {
                return Ok(None);
            };
            let Ok(mask) = as_bool(&evaluated, "and") else {
                return Ok(None);
            };
            let kept = truthy(mask);
            if let (Some(l), Some(started)) = (learned, started) {
                l.record(
                    ci,
                    rows_in,
                    kept.values().count_set_bits(),
                    started.elapsed(),
                );
            }
            live = boolean::and(&live, &kept)?;
            if pos + 1 == order.len() {
                break;
            }
            if !compact_before(
                &mut view,
                &mut view_abs,
                &mut live,
                batch,
                &conjuncts,
                &order[pos + 1..],
            )? {
                return Ok(None);
            }
        }

        Ok(Some(match view_abs {
            None => live,
            Some(abs) => scatter_mask(n, &abs, &live),
        }))
    }
}

/// Before evaluating `remaining[0]`, gather the rows `live` keeps when that pays, updating
/// `view`, `view_abs` and `live` in place. `false` when a named column is missing, which the
/// caller answers by declining.
fn compact_before(
    view: &mut RecordBatch,
    view_abs: &mut Option<Vec<u32>>,
    live: &mut BooleanArray,
    batch: &RecordBatch,
    conjuncts: &[&Expr],
    remaining: &[usize],
) -> Result<bool, ExprError> {
    let Some(&next) = remaining.first() else {
        return Ok(true);
    };
    let alive = live.values().count_set_bits();
    // `should_compact` deliberately does not decide `alive == 0`, because its two
    // callers want opposite answers. Here the answer is no: handing a conjunct an
    // empty batch is the one way this rewrite could change an outcome, since a type
    // error the whole-batch path raises might not fire on no rows.
    if alive == 0 || !should_compact(alive, view.num_rows(), conjuncts[next].eval_cost()) {
        return Ok(true);
    }
    let still_to_read: Vec<&Expr> = remaining.iter().map(|&i| conjuncts[i]).collect();
    let Some(next_view) = compact_rows(batch, &still_to_read, live, view_abs.as_deref())? else {
        return Ok(false);
    };
    *view = next_view.batch;
    *live = all_set(next_view.abs.len());
    *view_abs = Some(next_view.abs);
    Ok(true)
}

/// The units a filter evaluates one at a time: each conjunct, except that a lower and an upper
/// bound of one column against literals are paired into a single `AND`.
///
/// The pair is the shape `cmp::try_prim_range` and `cmp::try_string_range` answer in one walk
/// of the column. Evaluated as two units it cost a full-width pass, a gather of the surviving
/// rows, a second pass over them and a scatter of the mask back — `l_shipdate BETWEEN …` over
/// TPC-H spent more in that bookkeeping than in either comparison. Pairing changes no answer:
/// the conjuncts of an `AND` commute, and the paired `AND` is evaluated by the same `eval`
/// that would have evaluated the predicate whole. Each bound joins at most one pair, the first
/// partner found in written order, so the grouping is a pure function of the predicate and
/// [`ConjunctOrder`] sizes itself from the same call.
fn filter_units<'a>(conjuncts: &[&'a Expr]) -> Vec<std::borrow::Cow<'a, Expr>> {
    use std::borrow::Cow;
    let mut taken = vec![false; conjuncts.len()];
    let mut units = Vec::with_capacity(conjuncts.len());
    for i in 0..conjuncts.len() {
        if taken[i] {
            continue;
        }
        taken[i] = true;
        let partner = (i + 1..conjuncts.len())
            .find(|&j| !taken[j] && crate::eval::cmp::is_range_pair(conjuncts[i], conjuncts[j]));
        units.push(match partner {
            Some(j) => {
                taken[j] = true;
                Cow::Owned(Expr::Binary {
                    op: crate::BinaryOp::And,
                    left: Box::new(conjuncts[i].clone()),
                    right: Box::new(conjuncts[j].clone()),
                })
            }
            None => Cow::Borrowed(conjuncts[i]),
        });
    }
    units
}

/// Cheapest conjunct first, the opening order DuckDB's `ExpressionHeuristics` also
/// computes. `sort_by_key` is stable, so equal-cost conjuncts stay in the order the query
/// wrote them, which is the only order a reader can predict.
fn static_order(conjuncts: &[&Expr]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..conjuncts.len()).collect();
    order.sort_by_key(|&i| conjuncts[i].eval_cost());
    order
}

/// What one Filter operator has measured about each of its conjuncts, so the next morsel
/// orders them by observed work rather than by a static guess.
///
/// ## Why measurement beats the cost model here
///
/// Cost is not selectivity, and the static order only knows cost. `WHERE is_active AND
/// url LIKE '%checkout%'` opens with the boolean because it is cheap, and if nearly every
/// row is active that ordering buys nothing: the expensive conjunct still runs at full
/// width. The quantity that actually matters is *work removed per unit of work spent*,
/// and both halves of it are observable after one morsel.
///
/// DuckDB reaches the same place differently: `AdaptiveFilter` swaps a random adjacent
/// pair, keeps the swap if the *total* runtime improved over the next ten batches, and
/// halves that position's swap likeliness when it did not
/// (`src/execution/adaptive_filter.cpp`). That is a hill-climb on an aggregate signal, and
/// it needs tens of batches to walk a permutation. Because
/// [`Expr::short_circuit_filter_mask_with`] evaluates the conjuncts one at a time anyway,
/// it can attribute rows and time to each conjunct *individually* and jump straight to the
/// implied order after a single morsel. The trade is that the measurement is conditional:
/// a conjunct that runs third only ever sees rows the first two kept, so its keep-rate is
/// conditional on them. That biases the estimate but does not destabilize it, because the
/// ordering it produces feeds back into the next measurement.
///
/// ## Why it needs no lock
///
/// Every counter is a plain atomic and each morsel derives its own permutation from
/// whatever it reads. Two workers may briefly disagree about the best order, which costs
/// nothing: the conjuncts of an `AND` commute, so **every** order yields the identical
/// mask. That is the same property [`Expr::is_infallible_predicate`] guarantees for
/// skipping, and it is why this can be adaptive without being a correctness surface.
#[derive(Debug)]
pub struct ConjunctOrder {
    slots: Vec<Slot>,
}

#[derive(Debug, Default)]
struct Slot {
    rows_in: AtomicU64,
    rows_out: AtomicU64,
    nanos: AtomicU64,
}

impl ConjunctOrder {
    /// Build state for `predicate`, or `None` when it has fewer than two conjuncts and
    /// there is therefore no order to choose.
    #[must_use]
    pub fn new(predicate: &Expr) -> Option<Self> {
        let width = filter_units(&predicate.and_conjuncts()).len();
        (width >= 2).then(|| Self {
            slots: (0..width).map(|_| Slot::default()).collect(),
        })
    }

    /// How many conjuncts this was built for.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// True when there are no conjuncts. Present because clippy asks for it beside
    /// [`Self::len`]; a `ConjunctOrder` is never actually built empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    fn record(&self, index: usize, rows_in: usize, rows_out: usize, elapsed: Duration) {
        let slot = &self.slots[index];
        slot.rows_in.fetch_add(rows_in as u64, Ordering::Relaxed);
        slot.rows_out.fetch_add(rows_out as u64, Ordering::Relaxed);
        slot.nanos
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    /// The order to evaluate `conjuncts` in, cheapest expected work first.
    ///
    /// The rank is `time per row / rows removed per row`: a conjunct that removes most of
    /// what it sees earns its cost, one that removes nothing never does however cheap it
    /// is. A conjunct with no measurement yet falls back to its static cost, scaled into
    /// the same units so the two can be compared before every slot has been filled.
    fn order(&self, conjuncts: &[&Expr]) -> Vec<usize> {
        let mut ranked: Vec<(usize, f64)> = (0..conjuncts.len())
            .map(|i| (i, self.rank(i, conjuncts[i])))
            .collect();
        // A total order over finite, non-NaN ranks; ties keep index order, so an
        // unmeasured predicate stays in the order it was written. Being a total order over
        // distinct indices is also what makes the *unstable* sort deterministic here, so it
        // needs no stable merge sort and no scratch allocation.
        ranked.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        ranked.into_iter().map(|(i, _)| i).collect()
    }

    fn rank(&self, index: usize, conjunct: &Expr) -> f64 {
        let slot = &self.slots[index];
        let rows_in = slot.rows_in.load(Ordering::Relaxed);
        if rows_in < MIN_MEASURED_ROWS {
            // Not enough evidence yet. Keep the static cost, offset above the measured
            // range so a measured conjunct is preferred once one exists — an unmeasured
            // conjunct is the one whose selectivity we most want to learn, and it learns
            // it by being evaluated, not by going first over the whole batch.
            return UNMEASURED_RANK_BASE + f64::from(conjunct.eval_cost());
        }
        let rows_out = slot.rows_out.load(Ordering::Relaxed);
        let nanos = slot.nanos.load(Ordering::Relaxed);
        let per_row = nanos as f64 / rows_in as f64;
        // Rows removed per row seen. Floored so a conjunct that removes nothing ranks
        // last by a wide margin instead of dividing by zero.
        let removed = ((rows_in - rows_out) as f64 / rows_in as f64).max(MIN_REMOVED_FRACTION);
        per_row / removed
    }
}

/// Rows a conjunct must have been evaluated over before its measurement is preferred to
/// the static cost. One morsel is enough evidence; a handful of rows is not.
const MIN_MEASURED_ROWS: u64 = 4_096;

/// Floor on the "fraction of rows removed" denominator, so a conjunct that has never
/// removed a row ranks last rather than producing an infinite rank.
const MIN_REMOVED_FRACTION: f64 = 1.0 / 65_536.0;

/// Rank offset for a conjunct with no measurement yet. Above any plausible measured rank
/// (nanoseconds per row divided by a fraction), so measured conjuncts sort first.
const UNMEASURED_RANK_BASE: f64 = 1.0e9;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Array, Int32Array, Int64Array, StringArray, UInt32Array};
    use arrow::compute::{filter_record_batch, take};
    use arrow::datatypes::{DataType, Field, Schema};

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
            format!("{got:?}"),
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

    /// A lower and an upper bound of one column become one unit, wherever they sit in the
    /// chain; a bound with no partner, a second bound of the same side, and a bound of another
    /// column stay single. The masks still match the whole-batch oracle.
    #[test]
    fn range_bounds_pair_into_one_unit_and_keep_the_mask() {
        let lo = cmp(BinaryOp::Ge, "a", 10);
        let hi = cmp(BinaryOp::Lt, "a", 900);
        let other = cmp(BinaryOp::Gt, "b", 3);
        let lo2 = cmp(BinaryOp::Gt, "a", 20);
        let pred = and(and(lo.clone(), other.clone()), and(lo2.clone(), hi.clone()));
        let conjuncts = pred.and_conjuncts();
        let units = filter_units(&conjuncts);
        assert_eq!(units.len(), 3, "{units:?}");
        let dbg = |e: &Expr| format!("{e:?}");
        assert_eq!(dbg(&units[0]), dbg(&and(lo.clone(), hi.clone())));
        assert_eq!(dbg(&units[1]), dbg(&other));
        assert_eq!(dbg(&units[2]), dbg(&lo2));
        assert_eq!(ConjunctOrder::new(&pred).map(|o| o.len()), Some(3));
        let batch = sample(4_096);
        assert_matches_oracle(&pred, &batch);
        // A predicate that *is* one range has nothing to short-circuit.
        let range = and(lo, hi);
        assert!(range
            .short_circuit_filter_mask(&batch)
            .expect("eval")
            .is_none());
        assert!(ConjunctOrder::new(&range).is_none());
    }

    /// `filter_mask_within` is `live AND self` — held to that, computed the slow way, for a
    /// `live` keeping almost every row (no gather), few rows (gather first), a single conjunct
    /// and a chain; and it declines what it must: an empty `live`, a fallible predicate.
    #[test]
    fn a_mask_within_another_is_their_conjunction() {
        let batch = sample(4_096);
        let preds = [
            cmp(BinaryOp::Gt, "b", 100),
            and(cmp(BinaryOp::Lt, "a", 3_000), cmp(BinaryOp::Ne, "b", 17)),
            and(
                and(cmp(BinaryOp::Ge, "a", 5), cmp(BinaryOp::Lt, "a", 4_000)),
                cmp(BinaryOp::Gt, "b", 2),
            ),
        ];
        for keep_every in [1_usize, 2, 9, 600] {
            let live =
                BooleanArray::from((0..4_096).map(|i| i % keep_every == 0).collect::<Vec<_>>());
            for pred in &preds {
                let got = pred
                    .filter_mask_within(&batch, &live, None)
                    .expect("eval")
                    .expect("served");
                let whole = truthy(as_bool(&pred.eval(&batch).unwrap(), "t").unwrap());
                let want = boolean::and(&live, &whole).unwrap();
                assert_eq!(got, want, "keep 1 in {keep_every}: {pred:?}");
            }
        }
        let none = BooleanArray::from(vec![false; 4_096]);
        assert!(preds[0]
            .filter_mask_within(&batch, &none, None)
            .unwrap()
            .is_none());
        let fallible = Expr::Binary {
            op: BinaryOp::Gt,
            left: Box::new(Expr::Binary {
                op: BinaryOp::Div,
                left: col("a"),
                right: col("b"),
            }),
            right: lit_int(1),
        };
        let all = BooleanArray::from(vec![true; 4_096]);
        assert!(!fallible.is_infallible_predicate(&batch.schema()));
        assert!(fallible
            .filter_mask_within(&batch, &all, None)
            .unwrap()
            .is_none());
        assert!(preds[0]
            .filter_mask_within(&batch, &BooleanArray::from(vec![true; 3]), None)
            .is_err());
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
        assert!(cheap.eval_cost() <= crate::subset::CHEAP_EXPR_COST);
        assert!(expensive.eval_cost() > crate::subset::CHEAP_EXPR_COST);
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
        let a: Int64Array = (0..i64::from(n)).collect::<Vec<_>>().into();
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

    /// The case the static cost model gets wrong, and the reason to measure at all: two
    /// conjuncts of *identical* cost where only one is selective. Cost cannot separate
    /// them, so the static order keeps the written one, which here is the useless
    /// predicate first. After one morsel the measured order must put the selective one
    /// first.
    #[test]
    fn a_measured_order_promotes_the_selective_conjunct() {
        let n = 8_192;
        // `keep_all` is true for every row; `keep_few` for one row in 512.
        // Two columns holding the same values: a lower and an upper bound of *one* column would
        // be fused into a single range unit (`filter_units`), leaving no order to learn.
        let a: Int64Array = (0..i64::from(n)).map(Some).collect();
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Int64, true),
        ]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(a.clone()), Arc::new(a)])
            .expect("batch");

        let keep_all = cmp(BinaryOp::Ge, "a", 0);
        let keep_few = cmp(BinaryOp::Lt, "b", 16);
        let pred = and(keep_all.clone(), keep_few.clone());
        assert_eq!(
            keep_all.eval_cost(),
            keep_few.eval_cost(),
            "the premise is that cost cannot tell these apart"
        );

        let conjuncts = pred.and_conjuncts();
        let learned = ConjunctOrder::new(&pred).expect("two conjuncts");
        // Before any measurement the order is the static one, which is written order here.
        assert_eq!(learned.order(&conjuncts), vec![0, 1]);

        // One morsel is enough evidence.
        pred.short_circuit_filter_mask_with(&batch, Some(&learned))
            .expect("eval")
            .expect("fast path");
        assert_eq!(
            learned.order(&conjuncts),
            vec![1, 0],
            "the conjunct that removes 511 rows in 512 must be evaluated first"
        );
    }

    /// Whatever the order converges to, the mask may not move. This is the property that
    /// lets the ordering be adaptive without being a correctness surface, so it is asserted
    /// against the whole-batch oracle over many morsels rather than argued.
    #[test]
    fn a_measured_order_never_changes_the_mask() {
        let pred = and(
            and(cmp(BinaryOp::Ge, "a", 0), cmp(BinaryOp::Lt, "a", 3_000)),
            and(cmp(BinaryOp::Ne, "b", 17), cmp(BinaryOp::Gt, "b", 2)),
        );
        let learned = ConjunctOrder::new(&pred).expect("four conjuncts");
        for round in 0..8 {
            let batch = sample(4_096 + round * 37);
            let expected = oracle(&pred, &batch);
            let mask = pred
                .short_circuit_filter_mask_with(&batch, Some(&learned))
                .expect("eval")
                .expect("fast path");
            let got = filter_record_batch(&batch, &mask).expect("filter");
            assert_eq!(
                format!("{got:?}"),
                format!("{expected:?}"),
                "round {round}: a measured order changed the result"
            );
        }
    }

    /// A `ConjunctOrder` built for a different predicate must be ignored, not indexed
    /// into. Nothing in the type system stops a caller pairing the two, and a mismatched
    /// width would otherwise panic on a slot that does not exist.
    #[test]
    fn a_mismatched_order_width_is_ignored() {
        let batch = sample(4_096);
        let two = and(cmp(BinaryOp::Lt, "a", 100), cmp(BinaryOp::Gt, "b", 2));
        let three = and(
            and(cmp(BinaryOp::Lt, "a", 100), cmp(BinaryOp::Gt, "b", 2)),
            cmp(BinaryOp::Ne, "a", 7),
        );
        let wrong = ConjunctOrder::new(&three).expect("three conjuncts");
        assert_eq!(wrong.len(), 3);
        let expected = oracle(&two, &batch);
        let mask = two
            .short_circuit_filter_mask_with(&batch, Some(&wrong))
            .expect("must not panic on a mismatched order")
            .expect("fast path");
        let got = filter_record_batch(&batch, &mask).expect("filter");
        assert_eq!(format!("{got:?}"), format!("{expected:?}"));
    }

    /// A single-conjunct predicate has no order to choose, so it gets no state.
    #[test]
    fn no_order_state_for_a_single_conjunct() {
        assert!(ConjunctOrder::new(&cmp(BinaryOp::Lt, "a", 1)).is_none());
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
