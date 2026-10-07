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
//! Every other pruning step here skips whole blocks that provably hold no match, and the engine
//! keeps its own `Filter` regardless. This one removes individual rows, so it needs the
//! row-level form of the same guarantee: **every row the `Filter` above the scan keeps, the
//! pushed predicate keeps too.** Keeping *more* is fine, because that `Filter` still runs.
//!
//! `batcher.io.predicate.to_native_predicate` only ever widens: a conjunct it cannot translate
//! is dropped from its `AND` (once every negation has been carried to the leaves), and a
//! disjunction with an untranslatable side is not pushed at all. So a predicate that arrives
//! here keeps every row that `Filter` keeps, and the remaining question is only whether each
//! comparison is evaluated here at least as permissively as the engine evaluates it. [`Pred`]
//! has no negation, so a superset at every comparison is a superset of the whole.
//!
//! # The subset hazard, and how this avoids it
//!
//! Returning *more* rows than the predicate selects is always safe (the `Filter` still runs).
//! Returning *fewer* is a silent wrong answer. Three ways that could happen, all closed here:
//!
//! * **A lossy literal cast.** `col_i32 < 5000000000` casts the literal to `Int32` under
//!   arrow's safe cast and yields `null` — every comparison then goes false and the read
//!   returns *no rows* where the truth is *all rows*. [`lit_array`] therefore casts and then
//!   casts **back**, and refuses the pushdown unless the value round-trips exactly.
//! * **Floats, whose order here differs from the engine's.** Arrow's kernels use IEEE
//!   `totalOrder`; the engine canonicalizes first (`bc_arrow::canon_float_array`), so its
//!   zeros are one value and every NaN is the greatest. Rather than restate that
//!   canonicalization, [`float_superset`] keeps every row the two orders could disagree on.
//! * **Decimals**, which need `eval_binary`'s precision/scale alignment. [`pushable`] refuses
//!   them outright; they keep the block-level pruning they already had.
//!
//! Pushability is decided **once, up front, against the file schema** ([`build`]), never per
//! batch. A per-batch evaluation that somehow still fails returns an all-true mask rather than
//! an error, so the worst outcome is the read this module was trying to speed up.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Datum, DictionaryArray, Float64Array, Int64Array, RecordBatch,
    Scalar, StringArray,
};
use arrow::compute::kernels::{boolean, cmp};
use arrow::datatypes::{DataType, Int32Type, Schema};
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
    use DataType::{
        Boolean, Date32, Date64, Float32, Float64, Int16, Int32, Int64, Int8, LargeUtf8, UInt16,
        UInt32, UInt64, UInt8, Utf8,
    };
    matches!(
        dt,
        Boolean
            | Float32
            | Float64
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
            Ok(f) => {
                comparable(f.data_type())
                    && !matches!(lit, Lit::Float(v) if v.is_nan())
                    && lit_array(lit, f.data_type()).is_some()
            }
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
            // A string column the read decoded as a `Dictionary` (`late::dictionary_read`):
            // compare each distinct value once and give every row its key's answer. A null key
            // takes a null answer, which is what comparing the decoded null would give.
            if let Some(dict) = arr.as_any().downcast_ref::<DictionaryArray<Int32Type>>() {
                let per_value = cmp_mask(*op, lit, dict.values().as_ref())?;
                let mask = arrow::compute::take(&per_value, dict.keys(), None).ok()?;
                return mask.as_any().downcast_ref::<BooleanArray>().cloned();
            }
            cmp_mask(*op, lit, arr.as_ref())
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

/// `arr <op> lit`, element-wise, with the float widening of [`float_superset`].
fn cmp_mask(op: CmpOp, lit: &Lit, arr: &dyn Array) -> Option<BooleanArray> {
    let lit_arr = lit_array(lit, arr.data_type())?;
    let scalar = Scalar::new(lit_arr);
    let lhs: &dyn Datum = &arr;
    let rhs: &dyn Datum = &scalar;
    let mask = match op {
        CmpOp::Eq => cmp::eq(lhs, rhs),
        CmpOp::Ne => cmp::neq(lhs, rhs),
        CmpOp::Lt => cmp::lt(lhs, rhs),
        CmpOp::Le => cmp::lt_eq(lhs, rhs),
        CmpOp::Gt => cmp::gt(lhs, rhs),
        CmpOp::Ge => cmp::gt_eq(lhs, rhs),
    }
    .ok()?;
    float_superset(mask, arr, lit)
}

/// Widen a float comparison's mask to every row whose answer the engine could decide otherwise.
///
/// Arrow's float kernels order by IEEE `totalOrder`, where `-0.0 < 0.0` and a sign-bit NaN sorts
/// below `-inf`. The engine canonicalizes first (`bc_arrow::canon_float_array`): both zeros are
/// one value and every NaN is the greatest. The two orders disagree only on rows that are NaN,
/// and on rows that are a zero compared against a zero literal, so those rows are kept whatever
/// the kernel said.
///
/// That makes the float mask a **superset** of the engine's rather than equal to it, which is
/// all removing rows here needs: every row dropped fails the kernel comparison on a value where
/// the two orders agree, so the engine's `Filter` -- which still runs above the scan -- would
/// drop it too. Restating the canonicalization here instead would be the second float semantics
/// the module docs warn against; keeping the disputed rows needs none.
fn float_superset(mask: BooleanArray, arr: &dyn Array, lit: &Lit) -> Option<BooleanArray> {
    use arrow::array::AsArray;
    use arrow::datatypes::{Float32Type, Float64Type};

    let zero_lit = matches!(lit, Lit::Float(v) if *v == 0.0) || matches!(lit, Lit::Int(0));
    let disputed = match arr.data_type() {
        DataType::Float64 => {
            let v = arr.as_primitive::<Float64Type>().values();
            BooleanArray::from_iter(
                v.iter()
                    .map(|x| Some(x.is_nan() || (zero_lit && *x == 0.0))),
            )
        }
        DataType::Float32 => {
            let v = arr.as_primitive::<Float32Type>().values();
            BooleanArray::from_iter(
                v.iter()
                    .map(|x| Some(x.is_nan() || (zero_lit && *x == 0.0))),
            )
        }
        _ => return Some(mask),
    };
    boolean::or(&mask, &disputed).ok()
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
/// **The cliff is below a tenth, not at half.** This was 0.5, set from two points on sf1
/// `lineitem` (2 % faster, 95 % slower) on the reading that the crossover between them was
/// "broad and flat". Measured across it on sf10 `lineitem` (60M rows, 490 row groups, one file,
/// a scattered `l_partkey < k` predicate), it is neither. Two measurements, because they
/// disagree and the second is the one that decides:
///
/// * **The read alone**, filter on against off on one binary: the filter is already 1.1x slower
///   at 3 % on a narrow payload and 1.4-1.9x slower at 10 %. Taken alone that puts the cliff
///   near 2 %.
/// * **The whole query**, two builds alternating over two rounds, a filter plus an aggregate.
///   Declining the filter hands the caller an unfiltered read to re-filter, which the read-only
///   figure does not charge, and that moves the crossover up. With 0.02 the query was 1.10-1.34x
///   slower at 3-5 %, so the gate belongs above that; with this value, against the old 0.5:
///
/// | selected | narrow payload | wide payload |
/// |---|---|---|
/// | 1-5 % | 0.99-1.01x | 1.00-1.02x |
/// | 10 % | 0.87x | 0.95x |
/// | 20 % | 0.69x | 0.68x |
/// | 35 % | 0.68x | 0.72x |
///
/// Unchanged where the filter pays, and up to 1.47x faster where it did not. The old value's cost
/// reached TPC-H: q3 at sf10 installed the filter on a ~20 % predicate over `customer`.
const MAX_SELECTIVITY: f64 = 0.08;

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
        Statistics::Int32(s) if unsigned => (
            f64::from(*s.min_opt()? as u32),
            f64::from(*s.max_opt()? as u32),
        ),
        Statistics::Int32(s) => (f64::from(*s.min_opt()?), f64::from(*s.max_opt()?)),
        Statistics::Int64(s) if unsigned => {
            (*s.min_opt()? as u64 as f64, *s.max_opt()? as u64 as f64)
        }
        Statistics::Int64(s) => (*s.min_opt()? as f64, *s.max_opt()? as f64),
        Statistics::Float(s) => (f64::from(*s.min_opt()?), f64::from(*s.max_opt()?)),
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

/// What deciding whether to install a [`RowFilter`] on a read needs to know.
pub(crate) struct Probe<'a> {
    pub(crate) pred: &'a Pred,
    pub(crate) resolved: &'a crate::store::Resolved,
    pub(crate) size: u64,
    pub(crate) meta: &'a parquet::arrow::arrow_reader::ArrowReaderMetadata,
    pub(crate) targets: &'a [usize],
    pub(crate) batch_size: usize,
}

/// The columns a read's row filter should decode first, or `None` for no row filter.
///
/// A measured decision: the zone maps may decline it for free, and otherwise the predicate
/// columns of the first target row group are probed for their selected fraction.
///
/// # Errors
/// [`crate::IoError`] when the probe read fails.
pub(crate) async fn verdict(p: Probe<'_>) -> Result<Option<Vec<String>>, crate::IoError> {
    use futures::TryStreamExt;
    use parquet::arrow::ParquetRecordBatchStreamBuilder;
    let pred = p.pred;
    let row_groups = p.meta.metadata().row_groups();
    let probe_rows: usize = p
        .targets
        .iter()
        .filter_map(|&rg| row_groups.get(rg))
        .map(|rg| rg.num_rows() as usize)
        .sum();
    let mut row_filter_cols: Option<Vec<String>> = None;
    {
        // Under this size the whole read is already short and the probe would be a larger
        // share of it than anything the filter could save. A caller reading one row group at a
        // time (`read_parquet_row_group`) therefore never installs one: probing each file once
        // for all its row groups was measured and was a net loss (TPC-H sf10 q19 +20-50 ms).
        if crate::row_filter_enabled()
            && probe_rows >= crate::ROW_FILTER_MIN_ROWS
            && row_groups.get(p.targets[0]).is_some()
        {
            // Free pre-check before the probe: if the zone maps already say the predicate keeps
            // most rows, there is nothing for the filter to save and the probe itself would be
            // the only cost anyone measured. The estimate is only ever allowed to *decline*
            // (see `row_filter::estimate`) — installing stays a measured decision, because the
            // interpolation behind it is wrong on skewed data and a wrong install is a
            // slowdown while a wrong decline is only a missed speed-up.
            let permissive = {
                let (mut weighted, mut rows) = (0.0f64, 0.0f64);
                let mut usable = true;
                // Column positions resolved once for the file; the loop below asks for the
                // same columns in every row group.
                let col_index = crate::predicate::ColumnIndex::build(p.meta.metadata());
                for &rg in p.targets {
                    let Some(meta) = row_groups.get(rg) else {
                        continue;
                    };
                    if let Some(f) = estimate(pred, meta, &col_index) {
                        weighted += f * meta.num_rows() as f64;
                        rows += meta.num_rows() as f64;
                    } else {
                        usable = false;
                        break;
                    }
                }
                usable && rows > 0.0 && !worth_it_frac(weighted / rows)
            };
            if !permissive {
                if let Some(cols) = plan(pred, p.meta.schema()) {
                    let reader = crate::split_read::object_reader(
                        &p.resolved.store,
                        &p.resolved.path,
                        p.size,
                    );
                    let mask = crate::projection::exact_columns(
                        p.meta.parquet_schema(),
                        cols.iter().map(String::as_str),
                    );
                    let mut probe =
                        ParquetRecordBatchStreamBuilder::new_with_metadata(reader, p.meta.clone())
                            .with_batch_size(p.batch_size.max(1))
                            .with_row_groups(vec![p.targets[0]])
                            .with_projection(mask)
                            .build()?;
                    // Stop as soon as the estimate is good enough. Decoding the *whole* first row
                    // group to measure it cost more than the filter saved on a permissive
                    // predicate (~18 ms, turning a 179 ms read into 198 ms); a few thousand rows
                    // answer "is this selective?" just as well and cost ~1 ms. The estimate is a
                    // sample, so a clustered column can mislead it — which changes only speed.
                    let (mut selected, mut total) = (0usize, 0usize);
                    while total < crate::ROW_FILTER_PROBE_ROWS {
                        let Some(batch) = probe.try_next().await? else {
                            break;
                        };
                        let m = mask_of(pred, &batch);
                        total += m.len();
                        selected += m.true_count();
                    }
                    if worth_it(selected, total) {
                        row_filter_cols = Some(cols);
                    }
                }
            }
        }
    }
    Ok(row_filter_cols)
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
    fn decimal_columns_and_nan_literals_are_refused() {
        // Decimals need scale alignment this crate deliberately does not restate, so they
        // must never install a row filter. A NaN literal has no order to be a superset of.
        assert!(!pushable(
            &cmp_pred("d", CmpOp::Lt, Lit::Float(1.0)),
            &schema()
        ));
        assert!(!pushable(
            &cmp_pred("f", CmpOp::Lt, Lit::Float(f64::NAN)),
            &schema()
        ));
        // A finite float literal pushes, as a superset (see `float_superset`).
        assert!(pushable(
            &cmp_pred("f", CmpOp::Lt, Lit::Float(1.0)),
            &schema()
        ));
    }

    /// The engine's answer for `x <op> lit`: canonicalize both sides, then compare in the
    /// order the engine uses. A null compares to null, which `WHERE` treats as false.
    fn engine_keeps(values: &Float64Array, op: CmpOp, lit: f64) -> Vec<bool> {
        let canon = bc_arrow::canon_float_array(&(Arc::new(values.clone()) as ArrayRef));
        let lit =
            bc_arrow::canon_float_array(&(Arc::new(Float64Array::from(vec![lit])) as ArrayRef));
        let scalar = Scalar::new(lit);
        let lhs: &dyn Datum = &canon.as_ref();
        let m = match op {
            CmpOp::Eq => cmp::eq(lhs, &scalar),
            CmpOp::Ne => cmp::neq(lhs, &scalar),
            CmpOp::Lt => cmp::lt(lhs, &scalar),
            CmpOp::Le => cmp::lt_eq(lhs, &scalar),
            CmpOp::Gt => cmp::gt(lhs, &scalar),
            CmpOp::Ge => cmp::gt_eq(lhs, &scalar),
        }
        .unwrap();
        (0..m.len()).map(|i| m.is_valid(i) && m.value(i)).collect()
    }

    #[test]
    fn a_float_mask_keeps_every_row_the_engine_keeps() {
        // The values the two float orders disagree on, and their neighbours: both zeros, a
        // quiet and a sign-bit NaN, both infinities, and a null.
        let negative_nan = f64::from_bits(f64::NAN.to_bits() | (1 << 63));
        let values = Float64Array::from(vec![
            Some(0.0),
            Some(-0.0),
            Some(f64::NAN),
            Some(negative_nan),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
            Some(-1.5),
            Some(0.05),
            Some(0.07),
            Some(3.0),
            None,
        ]);
        let s = Arc::new(Schema::new(vec![Field::new("f", DataType::Float64, true)]));
        let batch = RecordBatch::try_new(s, vec![Arc::new(values.clone()) as ArrayRef]).unwrap();
        let ops = [
            ("=", CmpOp::Eq),
            ("!=", CmpOp::Ne),
            ("<", CmpOp::Lt),
            ("<=", CmpOp::Le),
            (">", CmpOp::Gt),
            (">=", CmpOp::Ge),
        ];
        for lit in [0.0, -0.0, 0.05, -1.5, 3.0, f64::INFINITY] {
            for (name, op) in ops {
                let kept = mask_of(&cmp_pred("f", op, Lit::Float(lit)), &batch);
                for (i, engine) in engine_keeps(&values, op, lit).into_iter().enumerate() {
                    let value = values.is_valid(i).then(|| values.value(i));
                    assert!(
                        !engine || kept.value(i),
                        "x = {value:?}, x {name} {lit}: the engine keeps the row and the row \
                         filter dropped it",
                    );
                }
            }
        }
    }

    #[test]
    fn a_float_mask_still_removes_rows_the_orders_agree_on() {
        // The positive control: a superset that kept everything would pass the test above.
        let values = Float64Array::from(vec![0.01, 0.06, 0.09, f64::NAN]);
        let s = Arc::new(Schema::new(vec![Field::new("f", DataType::Float64, true)]));
        let batch = RecordBatch::try_new(s, vec![Arc::new(values) as ArrayRef]).unwrap();
        let p = Pred::And {
            left: Box::new(cmp_pred("f", CmpOp::Ge, Lit::Float(0.05))),
            right: Box::new(cmp_pred("f", CmpOp::Le, Lit::Float(0.07))),
        };
        let kept = mask_of(&p, &batch);
        assert_eq!(
            (0..4).map(|i| kept.value(i)).collect::<Vec<_>>(),
            vec![false, true, false, true]
        );
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
            right: Box::new(cmp_pred("d", CmpOp::Lt, Lit::Float(1.0))),
        };
        assert!(!pushable(&p, &schema()));
    }

    #[test]
    fn is_null_pushes_on_any_type() {
        // It reads the validity bitmap only, so even the refused value types are fine.
        for c in ["i64", "d"] {
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

    #[test]
    fn a_dictionary_column_evaluates_exactly_as_its_plain_strings() {
        // The read decodes a dictionary-encoded string column as a `Dictionary` under the native
        // filter, so every comparison must give each row the answer its plain value gets --
        // nulls included, and a dictionary value no row uses must not leak into the mask.
        use arrow::array::DictionaryArray;
        let plain: Vec<Option<&str>> = vec![
            Some("AIR"),
            None,
            Some("MAIL"),
            Some("AIR REG"),
            Some("AIR"),
            None,
            Some(""),
        ];
        let dict: DictionaryArray<Int32Type> = plain.iter().copied().collect();
        let field = |dt: DataType| Arc::new(Schema::new(vec![Field::new("s", dt, true)]));
        let plain_batch = RecordBatch::try_new(
            field(DataType::Utf8),
            vec![Arc::new(StringArray::from(plain.clone())) as ArrayRef],
        )
        .unwrap();
        let dict_batch = RecordBatch::try_new(
            field(dict.data_type().clone()),
            vec![Arc::new(dict) as ArrayRef],
        )
        .unwrap();
        let s = |v: &str| Lit::Str(v.to_string());
        let preds = [
            cmp_pred("s", CmpOp::Eq, s("AIR")),
            cmp_pred("s", CmpOp::Ne, s("AIR")),
            cmp_pred("s", CmpOp::Lt, s("AIR REG")),
            cmp_pred("s", CmpOp::Ge, s("B")),
            cmp_pred("s", CmpOp::Eq, s("TRUCK")),
            Pred::Or {
                left: Box::new(cmp_pred("s", CmpOp::Eq, s("AIR"))),
                right: Box::new(Pred::IsNull {
                    col: "s".to_string(),
                    negated: false,
                }),
            },
            Pred::IsNull {
                col: "s".to_string(),
                negated: true,
            },
        ];
        for p in &preds {
            let want = eval(p, &plain_batch).unwrap();
            let got = eval(p, &dict_batch).unwrap();
            assert_eq!(got, want);
            assert_eq!(mask_of(p, &dict_batch), mask_of(p, &plain_batch));
        }
        // The positive control: the comparison does select something and reject something.
        let m = mask_of(&preds[0], &dict_batch);
        assert_eq!(m.true_count(), 2);
    }
}
