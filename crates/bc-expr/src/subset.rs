//! Evaluating an expression over a *subset* of a batch's rows, and putting the answer
//! back where it came from.
//!
//! Arrow has no selection vector: a kernel reads a whole array, so the only way to make
//! an expression cost less than one full-width pass is to **gather** the rows that need
//! it — along with just the columns the expression names — evaluate against that
//! narrower, shorter batch, and scatter the result back. That is the shared machinery
//! for every short-circuiting form in this crate:
//!
//! - [`crate::select`] short-circuits the conjuncts of an `AND`, so conjunct `n + 1`
//!   only sees the rows conjunct `n` kept;
//! - [`crate::eval::branch`] short-circuits `CASE` and `COALESCE`, so a branch is
//!   evaluated only over the rows that actually select it.
//!
//! The equivalence argument is the same in both places and rests on one property:
//! **every kernel reachable from `Expr::eval` is elementwise**, so row `i`'s result
//! depends on row `i` alone and evaluating over a gathered subset yields, at each
//! gathered position, exactly what the full-width pass would have written there.
//!
//! What differs between the two callers is the treatment of *errors*, and it differs
//! because the semantics differ. A conjunct of an `AND` is defined over every row, so
//! skipping a row that would have raised would hide an error the query should see, and
//! `select` guards that with [`crate::Expr::is_infallible_predicate`]. A `CASE` branch
//! is *not* defined over every row — SQL says an unselected branch is not evaluated, and
//! DuckDB agrees — so there skipping a row is the specified behaviour rather than a
//! licence that has to be earned.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBufferBuilder, RecordBatch, RecordBatchOptions,
    UInt32Array, UInt32Builder,
};
use arrow::buffer::BooleanBuffer;
use arrow::compute::take;
use arrow::datatypes::{Field, Schema};

use crate::{Expr, ExprError};

/// Cost (in [`crate::Expr::eval_cost`] units) at or below which an expression counts as
/// cheap enough that a gather has to remove most of the batch before it pays.
///
/// Calibrated to sit just above a comparison against a literal (`col cmp lit` is 3) and
/// a null test, and below a cast (8 plus its input) and a string kernel (40 plus its
/// input). It decides only *how eagerly* a gather is paid for, so the boundary wants to
/// be roughly right rather than exact.
pub(crate) const CHEAP_EXPR_COST: u32 = 8;

/// The mask with its nulls folded into false, so a chain of them composes with a plain
/// `AND`.
///
/// This is the same reduction `filter_record_batch` performs on a nullable mask before
/// gathering, which is why doing it per conjunct changes no keep decision. `CASE` needs
/// it for a different reason with the same shape: a `WHEN` that evaluates to NULL is not
/// taken, so a null there is indistinguishable from false.
pub(crate) fn truthy(mask: &BooleanArray) -> BooleanArray {
    match mask.nulls() {
        Some(nulls) => BooleanArray::new(mask.values() & nulls.inner(), None),
        None => mask.clone(),
    }
}

/// An all-true, null-free mask of `len` rows.
pub(crate) fn all_set(len: usize) -> BooleanArray {
    BooleanArray::new(BooleanBuffer::new_set(len), None)
}

/// Whether gathering the live rows now beats evaluating what remains at full width.
///
/// A gather costs one pass over the live rows of the named columns; skipping buys one
/// pass over the dead rows per expression still to run. So the threshold is a function
/// of what that expression *costs*: an expensive one (a cast, a regex, a dictionary-set
/// membership) repays the gather after a modest reduction, while another bare comparison
/// has to see most of the batch disappear first.
///
/// `alive == view_rows` is pure loss and is refused. `alive == 0` is deliberately **not**
/// decided here, because the two callers want opposite answers: `select` must never hand
/// a conjunct an empty batch (a type error that the whole-batch path raises might not
/// fire on no rows, which is the one way its rewrite could change an outcome), while for
/// a `CASE` branch that no row selects, skipping it entirely *is* the SQL semantics.
pub(crate) fn should_compact(alive: usize, view_rows: usize, next_cost: u32) -> bool {
    if alive == view_rows {
        return false;
    }
    if next_cost > CHEAP_EXPR_COST {
        alive * 8 <= view_rows * 7
    } else {
        alive * 4 <= view_rows
    }
}

/// A compacted view: the live rows of just the columns still to be read.
pub(crate) struct Compacted {
    pub(crate) batch: RecordBatch,
    /// Row `j` of `batch` is row `abs[j]` of the original batch, ascending.
    pub(crate) abs: Vec<u32>,
}

/// Gather the rows `live` keeps, projected to the columns `exprs` name.
///
/// Indices are always resolved against the *original* batch — `view_abs` maps the
/// current view's rows back first — so repeated compaction composes without
/// accumulating a chain of gathers. Returns `None` if a named column is absent, so the
/// caller can fall back and let the ordinary path report it.
pub(crate) fn compact_rows(
    batch: &RecordBatch,
    exprs: &[&Expr],
    live: &BooleanArray,
    view_abs: Option<&[u32]>,
) -> Result<Option<Compacted>, ExprError> {
    let set = live.values().set_indices();
    let abs: Vec<u32> = match view_abs {
        None => set.map(|i| i as u32).collect(),
        Some(prev) => set.map(|i| prev[i]).collect(),
    };

    let mut names: Vec<&str> = Vec::new();
    for expr in exprs {
        expr.collect_columns(&mut names);
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

    // The row count must be stated: an expression over literals alone names no column,
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
pub(crate) fn scatter_mask(n: usize, abs: &[u32], live: &BooleanArray) -> BooleanArray {
    let mut bits = BooleanBufferBuilder::new(n);
    bits.append_n(n, false);
    for j in live.values().set_indices() {
        bits.set_bit(abs[j] as usize, true);
    }
    BooleanArray::new(bits.finish(), None)
}

/// Expand a *value* array over compacted rows back to one over all `n` original rows,
/// null at every row that was not gathered.
///
/// Built as a gather rather than a per-type scatter so it works for every Arrow type the
/// engine can produce, nested ones included: the index array is null wherever the row was
/// not selected, and `take` turns a null index into a null output element.
///
/// The nulls it writes are never read. Both callers combine the result with `zip` under
/// the very mask that selected the rows, so a position this leaves null is a position the
/// combination takes from somewhere else.
pub(crate) fn scatter_values(
    n: usize,
    abs: &[u32],
    values: &ArrayRef,
) -> Result<ArrayRef, ExprError> {
    debug_assert_eq!(values.len(), abs.len());
    let mut idx = UInt32Builder::with_capacity(n);
    let mut next = 0usize;
    for row in 0..n {
        if next < abs.len() && abs[next] as usize == row {
            idx.append_value(next as u32);
            next += 1;
        } else {
            idx.append_null();
        }
    }
    Ok(take(values.as_ref(), &idx.finish(), None)?)
}
