//! Row-level predicate pushdown *into* the Parquet decode (`RowFilter`).
//!
//! The pruning steps in [`crate::predicate`], [`crate::page_index`] and [`crate::bloom`] all
//! answer "can this **block** be skipped whole?" — row group, page, then bloom. None of them
//! help when the matching rows are *scattered*: a 2 %-selective predicate on an unclustered
//! column leaves every row group and every page alive, so all of them are fetched and every
//! column is fully decoded before the engine's `Filter` throws 98 % of it away.
//!
//! A `RowFilter` closes exactly that gap. Parquet decodes only the **predicate columns**
//! first, evaluates the predicate, and then decodes the remaining columns *for surviving rows
//! only*. The saving is the decode of every non-predicate column for every rejected row, which
//! is why the win grows with table width and shrinks to nothing on a narrow projection —
//! measured on TPC-H `lineitem` in the module tests' shape below.
//!
//! # Why this may drop rows at all
//!
//! Every other pruning step here is *superset-safe*: it may only skip blocks that provably
//! hold no match, and the engine keeps its own `Filter` regardless. This one removes
//! individual rows, so it needs a stronger guarantee — the pushed predicate must be
//! **equivalent** to the `Filter` above the scan, not merely implied by it.
//!
//! It is. `batcher.io.predicate.to_native_predicate` is all-or-nothing: any term it cannot
//! translate makes the *whole* expression unpushable and it emits nothing. So a predicate that
//! arrives here is a complete translation of that `Filter`, and dropping a row that fails it
//! removes a row the `Filter` would have removed anyway.
//!
//! # The subset hazard, and how this avoids it
//!
//! Returning *more* rows than the predicate selects is always safe (the `Filter` still runs).
//! Returning *fewer* is a silent wrong answer. Two ways that could happen, both closed here:
//!
//! * **A lossy literal cast.** `col_i32 < 5000000000` casts the literal to `Int32` under
//!   arrow's safe cast and yields `null` — every comparison then goes false and the read
//!   returns *no rows* where the truth is *all rows*. [`lit_array`] therefore casts and then
//!   casts **back**, and refuses the pushdown unless the value round-trips exactly.
//! * **A type this module reasons about differently from the engine.** Floats need the
//!   engine's `-0.0`/NaN canonicalization (`bc_arrow::canon_float_array`) to compare
//!   identically, and decimals need `eval_binary`'s precision/scale alignment. Rather than
//!   restate either here — a second semantics, in a crate that cannot see the first —
//!   [`pushable`] simply refuses both. They keep the block-level pruning they already had.
//!
//! Pushability is decided **once, up front, against the file schema** ([`build`]), never per
//! batch. A per-batch evaluation that somehow still fails returns an all-true mask rather than
//! an error, so the worst outcome is the read this module was trying to speed up.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Datum, Float64Array, Int64Array, RecordBatch, Scalar,
    StringArray,
};
use arrow::compute::kernels::{boolean, cmp};
use arrow::datatypes::{DataType, Schema};
use parquet::arrow::arrow_reader::{ArrowPredicateFn, RowFilter};
use parquet::file::metadata::RowGroupMetaData;
use parquet::file::statistics::Statistics;
use parquet::schema::types::SchemaDescriptor;

use crate::predicate::{CmpOp, Lit, Pred};

/// Every column the predicate reads, in first-seen order and without duplicates.
fn columns_of(pred: &Pred, out: &mut Vec<String>) {
    match pred {
        Pred::Cmp { col, .. } | Pred::IsNull { col, .. } => {
            if !out.iter().any(|c| c == col) {
                out.push(col.clone());
            }
        }
        Pred::And { left, right } | Pred::Or { left, right } => {
            columns_of(left, out);
            columns_of(right, out);
        }
    }
}

/// A length-1 array holding `lit` in column type `dt`, or `None` if it cannot be represented
/// there **exactly**.
///
/// The round-trip is the whole point: arrow's safe cast turns an out-of-range value into
/// `null` rather than erroring, and a `null` literal makes every comparison false — which
/// would drop every row of a predicate that in truth matches all of them. Casting back and
/// comparing catches that, and also catches a float literal that cannot be held exactly by an
/// integer column.
fn lit_array(lit: &Lit, dt: &DataType) -> Option<ArrayRef> {
    let base: ArrayRef = match lit {
        Lit::Bool(b) => Arc::new(BooleanArray::from(vec![*b])),
        Lit::Int(i) => Arc::new(Int64Array::from(vec![*i])),
        Lit::Float(f) => Arc::new(Float64Array::from(vec![*f])),
        Lit::Str(s) => Arc::new(StringArray::from(vec![s.as_str()])),
    };
    if base.data_type() == dt {
        return Some(base);
    }
    let cast = arrow::compute::cast(&base, dt).ok()?;
    if cast.is_null(0) {
        return None;
    }
    // Back to the literal's own type; equal only if nothing was lost on the way out.
    let back = arrow::compute::cast(&cast, base.data_type()).ok()?;
    (back.as_ref() == base.as_ref()).then_some(cast)
}

/// Whether this column type is one whose comparison semantics here are identical to the
/// engine's.
///
/// Floats and decimals are excluded on purpose — see the module docs. Everything admitted
/// compares by plain arrow kernels in both places.
fn comparable(dt: &DataType) -> bool {
    use DataType::*;
    matches!(
        dt,
        Boolean
            | Int8
            | Int16
            | Int32
            | Int64
            | UInt8
            | UInt16
            | UInt32
            | UInt64
            | Utf8
            | LargeUtf8
            | Date32
            | Date64
    )
}

/// Whether the whole predicate can be evaluated here with engine-identical semantics.
fn pushable(pred: &Pred, schema: &Schema) -> bool {
    match pred {
        Pred::Cmp { col, lit, .. } => match schema.field_with_name(col) {
            Ok(f) => comparable(f.data_type()) && lit_array(lit, f.data_type()).is_some(),
            Err(_) => false,
        },
        // `IS NULL` reads only the validity bitmap, so it is type-agnostic — but the column
        // must exist, or the per-batch lookup would have to invent an answer.
        Pred::IsNull { col, .. } => schema.field_with_name(col).is_ok(),
        Pred::And { left, right } | Pred::Or { left, right } => {
            pushable(left, schema) && pushable(right, schema)
        }
    }
}

/// Evaluate the predicate over a batch of just the predicate columns.
///
/// `None` means "could not evaluate" — the caller substitutes an all-true mask, which keeps
/// every row and leaves the engine's `Filter` to do the work. [`pushable`] has already proved
/// this cannot happen for a predicate that was installed.
fn eval(pred: &Pred, batch: &RecordBatch) -> Option<BooleanArray> {
    match pred {
        Pred::Cmp { col, op, lit } => {
            let arr = batch.column_by_name(col)?;
            let lit_arr = lit_array(lit, arr.data_type())?;
            let scalar = Scalar::new(lit_arr);
            let arr_dyn: &dyn Array = arr.as_ref();
            let lhs: &dyn Datum = &arr_dyn;
            let rhs: &dyn Datum = &scalar;
            match op {
                CmpOp::Eq => cmp::eq(lhs, rhs),
                CmpOp::Ne => cmp::neq(lhs, rhs),
                CmpOp::Lt => cmp::lt(lhs, rhs),
                CmpOp::Le => cmp::lt_eq(lhs, rhs),
                CmpOp::Gt => cmp::gt(lhs, rhs),
                CmpOp::Ge => cmp::gt_eq(lhs, rhs),
            }
            .ok()
        }
        Pred::IsNull { col, negated } => {
            let arr = batch.column_by_name(col)?;
            let m = arrow::compute::is_null(arr.as_ref()).ok()?;
            if *negated {
                boolean::not(&m).ok()
            } else {
                Some(m)
            }
        }
        // Kleene, matching the engine's `AND`/`OR` (`bc_expr::eval::binary`). A null result
        // becomes `false` at the mask boundary below, which is SQL `WHERE` semantics and what
        // the engine's `Filter` does with the same null.
        Pred::And { left, right } => {
            boolean::and_kleene(&eval(left, batch)?, &eval(right, batch)?).ok()
        }
        Pred::Or { left, right } => {
            boolean::or_kleene(&eval(left, batch)?, &eval(right, batch)?).ok()
        }
    }
}

/// The predicate's columns if it can be pushed into the decode at all, else `None`.
///
/// Separated from [`build`] so the caller can decode just these columns for the selectivity
/// probe before deciding whether the filter is worth installing.
pub(crate) fn plan(pred: &Pred, arrow_schema: &Schema) -> Option<Vec<String>> {
    if !pushable(pred, arrow_schema) {
        return None;
    }
    let mut cols = Vec::new();
    columns_of(pred, &mut cols);
    (!cols.is_empty()).then_some(cols)
}

/// The selection mask for one batch: null-free, all-true if evaluation somehow fails.
pub(crate) fn mask_of(pred: &Pred, batch: &RecordBatch) -> BooleanArray {
    let rows = batch.num_rows();
    let mask = eval(pred, batch).unwrap_or_else(|| BooleanArray::from(vec![true; rows]));
    // `RowFilter` selects on `true` only; a null would be ambiguous. Folding null to false
    // here is exactly `WHERE`'s three-valued semantics.
    if mask.null_count() > 0 {
        arrow::compute::prep_null_mask_filter(&mask)
    } else {
        mask
    }
}

/// Below this selected-fraction a `RowFilter` pays for itself; above it, it costs.
///
/// Not a tuning knob so much as a cliff. A `RowFilter` trades "decode every column for every
/// row" for "decode the predicate columns, then decode the rest for survivors" — plus the
/// cost of *applying* a row selection, which is proportional to how **fragmented** it is. A
/// permissive predicate produces a selection of thousands of tiny alternating select/skip
/// runs, and decoding through that is markedly slower than decoding straight through while
/// saving almost nothing, because almost nothing is skipped.
///
/// Measured on TPC-H sf1 `lineitem` (6M rows, 16 columns, 49 row groups), scattered `l_suppkey`
/// predicate, full projection: at ~2 % selected the filter runs **127.6 ms → 88.0 ms**; at
/// ~95 % selected it runs **179.3 ms → 282.1 ms**. The crossover is broad and flat, so this
/// sits well clear of it on the safe side.
const MAX_SELECTIVITY: f64 = 0.5;

/// Whether a measured selected-fraction is low enough for the filter to pay.
pub(crate) fn worth_it(selected: usize, total: usize) -> bool {
    total > 0 && worth_it_frac(selected as f64 / total as f64)
}

/// [`worth_it`] on an already-computed fraction.
pub(crate) fn worth_it_frac(frac: f64) -> bool {
    frac < MAX_SELECTIVITY
}

/// The min/max of a numeric column's footer statistics, as `f64`.
///
/// Unsigned columns are reinterpreted the way [`crate::predicate`] does — Parquet stores them
/// in a *signed* physical type, so a `UInt32` above `i32::MAX` reads back negative and a naive
/// span would be nonsense. A NaN bound (writers have emitted them, see
/// `predicate::float_range_survives`) yields `None` rather than a garbage span.
fn bounds_f64(stats: &Statistics, unsigned: bool) -> Option<(f64, f64)> {
    let (lo, hi) = match stats {
        Statistics::Int32(s) if unsigned => {
            (*s.min_opt()? as u32 as f64, *s.max_opt()? as u32 as f64)
        }
        Statistics::Int32(s) => (*s.min_opt()? as f64, *s.max_opt()? as f64),
        Statistics::Int64(s) if unsigned => {
            (*s.min_opt()? as u64 as f64, *s.max_opt()? as u64 as f64)
        }
        Statistics::Int64(s) => (*s.min_opt()? as f64, *s.max_opt()? as f64),
        Statistics::Float(s) => (*s.min_opt()? as f64, *s.max_opt()? as f64),
        Statistics::Double(s) => (*s.min_opt()?, *s.max_opt()?),
        _ => return None,
    };
    (lo.is_finite() && hi.is_finite() && hi >= lo).then_some((lo, hi))
}

/// The literal as an `f64`, or `None` for the types this estimator does not model.
fn lit_f64(lit: &Lit) -> Option<f64> {
    match lit {
        Lit::Int(v) => Some(*v as f64),
        Lit::Float(v) => Some(*v),
        // A string or boolean span has no meaningful linear interpolation. Answering `None`
        // sends the caller to the measured probe instead of inventing a number.
        Lit::Bool(_) | Lit::Str(_) => None,
    }
}

/// The estimated fraction of `rg`'s rows the predicate selects, or `None` when the footer
/// cannot support an estimate.
///
/// This is a *zone-map interpolation*: it assumes values are spread uniformly across each
/// column's `[min, max]`, and that `AND`/`OR` operands are independent. Both assumptions are
/// wrong on skewed or correlated data — which is why the caller only ever uses this estimate
/// to **decline** the filter, never to install it. Declining wrongly costs a speed-up;
/// installing wrongly costs a slowdown, and only the measured probe is trusted for that.
pub(crate) fn estimate(
    pred: &Pred,
    rg: &RowGroupMetaData,
    index: &crate::predicate::ColumnIndex,
) -> Option<f64> {
    match pred {
        Pred::IsNull { col, negated } => {
            let rows = rg.num_rows();
            if rows <= 0 {
                return None;
            }
            let (stats, _) = index.stats(rg, col)?;
            // Exact, not interpolated: the null count is recorded, not inferred.
            let f = stats.null_count_opt()? as f64 / rows as f64;
            Some(if *negated { 1.0 - f } else { f })
        }
        Pred::And { left, right } => Some(estimate(left, rg, index)? * estimate(right, rg, index)?),
        Pred::Or { left, right } => {
            let (a, b) = (estimate(left, rg, index)?, estimate(right, rg, index)?);
            Some(a + b - a * b)
        }
        Pred::Cmp { col, op, lit } => {
            let (stats, unsigned) = index.stats(rg, col)?;
            let (lo, hi) = bounds_f64(stats, unsigned)?;
            let v = lit_f64(lit)?;
            let span = hi - lo;
            // A constant column has no span to interpolate over; the predicate is then simply
            // true or false for all of its rows.
            let below = if span <= 0.0 {
                if v > lo {
                    1.0
                } else {
                    0.0
                }
            } else {
                ((v - lo) / span).clamp(0.0, 1.0)
            };
            // Equality over a span is modelled as one value's share of it. On a float column
            // that share is large, so `Eq` tends to look *unselective* and the filter is
            // declined — the conservative direction, and bloom pruning already serves
            // high-cardinality equality.
            let eq = if v >= lo && v <= hi {
                (1.0 / (span + 1.0)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            Some(match op {
                CmpOp::Lt | CmpOp::Le => below,
                CmpOp::Gt | CmpOp::Ge => 1.0 - below,
                CmpOp::Eq => eq,
                CmpOp::Ne => 1.0 - eq,
            })
        }
    }
}

/// A `RowFilter` for `pred` over the already-validated `cols`.
///
/// Call only with a `cols` from [`plan`] and after [`worth_it`] has approved the measured
/// selectivity — this function does no gating of its own.
pub(crate) fn build(pred: &Pred, cols: &[String], descr: &SchemaDescriptor) -> RowFilter {
    let mask = crate::projection::exact_columns(descr, cols.iter().map(|s| s.as_str()));
    let owned = pred.clone();
    let f = ArrowPredicateFn::new(mask, move |batch: RecordBatch| Ok(mask_of(&owned, &batch)));
    RowFilter::new(vec![Box::new(f)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::Field;

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("i32", DataType::Int32, true),
            Field::new("i64", DataType::Int64, true),
            Field::new("s", DataType::Utf8, true),
            Field::new("f", DataType::Float64, true),
            Field::new("d", DataType::Decimal128(15, 2), true),
        ])
    }

    fn cmp_pred(col: &str, op: CmpOp, lit: Lit) -> Pred {
        Pred::Cmp {
            col: col.to_string(),
            op,
            lit,
        }
    }

    #[test]
    fn float_and_decimal_columns_are_refused() {
        // Both need semantics this crate deliberately does not restate (canonical floats,
        // decimal scale alignment), so they must never install a row filter.
        assert!(!pushable(
            &cmp_pred("f", CmpOp::Lt, Lit::Float(1.0)),
            &schema()
        ));
        assert!(!pushable(
            &cmp_pred("d", CmpOp::Lt, Lit::Float(1.0)),
            &schema()
        ));
    }

    #[test]
    fn out_of_range_literal_is_refused() {
        // The subset hazard: 5e9 does not fit Int32. Under a safe cast it becomes null and
        // every comparison goes false, which would return zero rows where the truth is all
        // of them. The round-trip check must catch it.
        assert!(!pushable(
            &cmp_pred("i32", CmpOp::Lt, Lit::Int(5_000_000_000)),
            &schema()
        ));
        // The same literal against an Int64 column is exactly representable, so it pushes.
        assert!(pushable(
            &cmp_pred("i64", CmpOp::Lt, Lit::Int(5_000_000_000)),
            &schema()
        ));
    }

    #[test]
    fn in_range_narrow_literal_pushes() {
        assert!(pushable(
            &cmp_pred("i32", CmpOp::Lt, Lit::Int(200)),
            &schema()
        ));
    }

    #[test]
    fn unknown_column_is_refused() {
        assert!(!pushable(
            &cmp_pred("nope", CmpOp::Eq, Lit::Int(1)),
            &schema()
        ));
    }

    #[test]
    fn and_of_pushable_and_unpushable_is_refused() {
        // One unpushable arm must sink the whole predicate: evaluating only the pushable
        // half of an AND would still be a superset (safe), but of an OR it would be a
        // subset (wrong), so `pushable` is all-or-nothing for both.
        let p = Pred::And {
            left: Box::new(cmp_pred("i64", CmpOp::Lt, Lit::Int(5))),
            right: Box::new(cmp_pred("f", CmpOp::Lt, Lit::Float(1.0))),
        };
        assert!(!pushable(&p, &schema()));
    }

    #[test]
    fn is_null_pushes_on_any_type() {
        // It reads the validity bitmap only, so even the refused value types are fine.
        for c in ["i64", "f", "d"] {
            assert!(pushable(
                &Pred::IsNull {
                    col: c.to_string(),
                    negated: false
                },
                &schema()
            ));
        }
    }

    #[test]
    fn eval_matches_arrow_semantics_including_nulls() {
        use arrow::array::Int64Array;
        let s = Arc::new(Schema::new(vec![Field::new("i64", DataType::Int64, true)]));
        let col = Arc::new(Int64Array::from(vec![Some(1), None, Some(5), Some(9)])) as ArrayRef;
        let batch = RecordBatch::try_new(s, vec![col]).unwrap();

        let m = eval(&cmp_pred("i64", CmpOp::Lt, Lit::Int(5)), &batch).unwrap();
        // Null compares to null, not false — the caller folds it to false at the boundary.
        assert!(m.value(0));
        assert!(m.is_null(1));
        assert!(!m.value(2));
        assert!(!m.value(3));

        let folded = arrow::compute::prep_null_mask_filter(&m);
        assert!(!folded.value(1));
        assert_eq!(folded.null_count(), 0);
    }

    #[test]
    fn eval_or_keeps_kleene_semantics() {
        use arrow::array::Int64Array;
        let s = Arc::new(Schema::new(vec![Field::new("i64", DataType::Int64, true)]));
        let col = Arc::new(Int64Array::from(vec![None, Some(1)])) as ArrayRef;
        let batch = RecordBatch::try_new(s, vec![col]).unwrap();
        // `null OR true` is true under Kleene, not null — losing that would drop a row the
        // engine's Filter keeps.
        let p = Pred::Or {
            left: Box::new(cmp_pred("i64", CmpOp::Lt, Lit::Int(5))),
            right: Box::new(Pred::IsNull {
                col: "i64".to_string(),
                negated: false,
            }),
        };
        let m = eval(&p, &batch).unwrap();
        assert!(m.value(0));
        assert!(m.value(1));
    }
}
