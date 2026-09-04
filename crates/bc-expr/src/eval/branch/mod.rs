//! Short-circuiting evaluation of the branch-selecting forms: `CASE` and `COALESCE`.
//!
//! Both are written in SQL as if only the branch a row selects is evaluated, and both
//! were evaluated here at full width: a four-branch `CASE` ran four `THEN` expressions
//! over every row and then picked one per row with `zip`, and `COALESCE` evaluated every
//! argument whether or not the first one already had a value. That is correct whenever
//! the branches are total, and it is wrong in two separate ways when they are not.
//!
//! **It is slow in proportion to the branch count.** A four-branch `CASE` whose arms each
//! run a regex costs four regex passes per row where one row needs one — measured here at
//! 310 ns/row against 149 ns/row for the same regex alone, so more than three-quarters of
//! the work produced values that `zip` immediately discarded. The cost of a branchy
//! expression grew with the number of branches rather than with the number of rows that
//! reach them, which is a scaling defect and not a constant factor.
//!
//! **And it raises errors SQL says cannot happen.** `CASE WHEN false THEN s::BIGINT ELSE
//! 1 END` over a non-numeric `s` failed the whole query here while DuckDB returned `1`,
//! because the `THEN` was evaluated over rows that never selected it. That is a
//! divergence from the oracle, and laziness is what fixes it — the speed is the same
//! change seen from the other side.
//!
//! ## What this module does
//!
//! Masks first, values second. The `WHEN`s are evaluated in order and folded into a set
//! of **disjoint** per-branch selections, and only then is each branch's value computed.
//! Both halves are evaluated over a gather of just the rows that need them whenever that
//! is worth its cost — a condition over the rows no earlier arm has claimed, a body over
//! the rows its own arm selected — and an arm no row selects is not evaluated at all.
//!
//! `crate::subset` owns the gather/scatter machinery and states the equivalence argument
//! shared with [`crate::select`]. The one part specific to here: because the selections
//! are disjoint and together cover every row, the `zip` fold that combines them is
//! order-independent, and a position left null by a scatter is always a position some
//! other branch fills.
//!
//! ## Errors
//!
//! A branch that a row selects must still raise. So the fallback runs the other way
//! round from the fast path: when full-width evaluation *fails*, it is retried over the
//! branch's own rows, and only an error there propagates. That makes the error behaviour
//! a property of the query rather than of whichever heuristic happened to fire — a branch
//! raises exactly when it raises on a row that selects it.

use arrow::array::{Array, ArrayRef, BooleanArray, RecordBatch};

use crate::subset::{compact_rows, scatter_values, CHEAP_EXPR_COST};
use crate::{Expr, ExprError};

mod case;
mod coalesce;

pub(crate) use case::eval_case;
pub(crate) use coalesce::eval_coalesce;

/// Evaluate `expr` over the rows `selection` marks, returning a full-length array that is
/// null everywhere else.
///
/// Three outcomes, in the order they are tried:
///
/// 1. **No row selected** — return a null column without evaluating anything. The type
///    still has to be right, so the expression is evaluated against an *empty* gather,
///    which costs nothing per row and reports the same schema-driven errors (an unknown
///    column, a type mismatch) that a full-width pass would.
/// 2. **Few enough rows selected** — gather them, evaluate, scatter back.
/// 3. **Otherwise** — evaluate at full width, and if that *fails*, retry over the
///    selected rows so an error on a row this branch never claims cannot escape.
fn eval_over(
    expr: &Expr,
    batch: &RecordBatch,
    selection: &BooleanArray,
    n: usize,
) -> Result<ArrayRef, ExprError> {
    let alive = selection.values().count_set_bits();
    if alive == 0 || worth_gathering(alive, n, expr.eval_cost()) {
        if let Some(out) = try_eval_over(expr, batch, selection, n)? {
            return Ok(out);
        }
    }
    match expr.eval(batch) {
        Ok(out) => Ok(out),
        // The full-width pass saw rows this branch does not select, so its error may
        // belong to one of them. Ask again over the rows that do select it, and report
        // the original error if the answer is still an error — never the narrower one,
        // which would blame a different row than the query's own semantics do.
        Err(full) if alive < n => match try_eval_over(expr, batch, selection, n) {
            Ok(Some(out)) => Ok(out),
            _ => Err(full),
        },
        Err(full) => Err(full),
    }
}

/// Whether gathering this arm's rows beats evaluating it over the whole batch.
///
/// Deliberately **not** [`crate::subset::should_compact`], which `select` uses. There a
/// single gather is shared by every remaining conjunct and the compacted batch persists, so
/// paying it for a cheap predicate can still repay. Here each arm gathers for itself and
/// uses the result once: a gather is a `take` over the arm's columns *plus* a scatter that
/// builds an `n`-long index and gathers through it, which is more passes over the batch
/// than simply evaluating a cheap expression across all of it. So a cheap arm never
/// gathers, however few rows select it.
///
/// Reusing `should_compact` here cost a measured 1.55x on the JIT's own shape. A four-arm
/// numeric `CASE` selects ~25% of rows per arm and its bodies are `col * lit`
/// (`eval_cost` 5, under the cheap bound), and `should_compact`'s cheap arm gathers at
/// `alive * 4 <= rows` — true at exactly 25%. Every arm therefore bought a gather and a
/// 16M-entry scatter to avoid one multiply: 1.68-1.77 ns/row became 2.63-2.72 over three
/// alternations with no overlap. The expensive arms this exists for are unaffected, a
/// string kernel costing 40 and a regex more.
fn worth_gathering(alive: usize, rows: usize, cost: u32) -> bool {
    cost > CHEAP_EXPR_COST && alive != rows && alive * 8 <= rows * 7
}

/// The gather / evaluate / scatter itself. `Ok(None)` means the gather could not be built
/// (an expression naming a column the batch does not have), so the caller falls back and
/// lets the ordinary path report it.
fn try_eval_over(
    expr: &Expr,
    batch: &RecordBatch,
    selection: &BooleanArray,
    n: usize,
) -> Result<Option<ArrayRef>, ExprError> {
    let Some(view) = compact_rows(batch, &[expr], selection, None)? else {
        return Ok(None);
    };
    let Ok(values) = expr.eval(&view.batch) else {
        return Ok(None);
    };
    if values.len() != view.abs.len() {
        return Ok(None);
    }
    if view.abs.len() == n {
        return Ok(Some(values));
    }
    // No row selected: the evaluation existed only to learn the type, and the answer is a
    // null column of it. Going through `scatter_values` would build an `n`-long index of
    // nulls and gather a zero-length array through it — two passes over the batch to
    // produce something `new_null_array` states directly. `COALESCE` reaches this on every
    // argument after the one that filled the column, so it is the common case, not a
    // corner.
    if view.abs.is_empty() {
        return Ok(Some(arrow::array::new_null_array(values.data_type(), n)));
    }
    Ok(Some(scatter_values(n, &view.abs, &values)?))
}
