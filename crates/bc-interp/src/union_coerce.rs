//! One common column type for the branches of a set operation, before they are combined.
//!
//! `UNION` / `INTERSECT` / `EXCEPT` all lower to `RelOp::Union`, and their branches may
//! carry promotable-but-different numeric types. Arrow's `concat` rejects a type mismatch
//! outright, so the branches are cast up to the supertype here first — the same supertype
//! `Dataset.schema` already advertises, taken from `bc_expr::common_supertype` rather than
//! restated, so the two cannot disagree about what `int64 ∪ float64` is.

use arrow::array::{ArrayRef, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use std::sync::Arc;

use crate::error::InterpError;

/// Coerce the branches of a set operation (UNION / INTERSECT / EXCEPT — all of which lower
/// to `RelOp::Union`) to one common column type before they are concatenated / deduped.
///
/// A set op's branches may carry promotable-but-different numeric types — `int64 ∪
/// float64` is the canonical case — and the union's advertised output schema is already
/// the promoted supertype (`promote(int64, float64) = float64`, per the Python type
/// lattice). Arrow's `concat`/`materialize`, however, reject a type mismatch outright, so
/// the branches must first be cast up to the supertype here. DuckDB likewise coerces both
/// sides to DOUBLE and returns a result; without this an ordinary `A UNION B` errored even
/// though `Dataset.schema` promised the promoted type.
///
/// A no-op that returns the input untouched when every branch already shares a column type
/// (the overwhelmingly common single-type union), so it pays only one scan of the schemas.
pub(crate) fn coerce_union_branches(
    batches: Vec<RecordBatch>,
) -> Result<Vec<RecordBatch>, InterpError> {
    let Some(schema) = union_target_schema(&batches)? else {
        return Ok(batches);
    };
    batches
        .iter()
        .map(|b| cast_to_union_schema(b, &schema))
        .collect()
}

/// The one schema every branch of a set operation must be cast to — or `None` when they
/// already agree and no cast is due (the overwhelmingly common single-type union).
///
/// Split out of [`coerce_union_branches`] so the *streaming* UNION ALL can settle the target
/// from one peeked morsel per branch and then cast each morsel as it flows, rather than
/// holding every branch to concatenate. Both callers therefore promote by the same rule and
/// build the same schema; a second copy of it is exactly what would let the streamed and
/// materialized unions disagree about what `int64 ∪ float64` is.
///
/// Nullability follows the same rule as the types: a column is nullable if *any* branch's is.
///
/// Args:
///   `branches`: one representative batch per branch (any batch will do — every batch of a
///   branch carries that branch's schema), or every batch when the caller already holds them.
pub(crate) fn union_target_schema(
    branches: &[RecordBatch],
) -> Result<Option<SchemaRef>, InterpError> {
    let Some(first) = branches.first() else {
        return Ok(None);
    };
    let batches = branches;
    let ncols = first.num_columns();
    // Fold the per-column supertype across every branch's schema.
    let mut target: Vec<DataType> = first
        .schema()
        .fields()
        .iter()
        .map(|f| f.data_type().clone())
        .collect();
    let mut mismatch = false;
    for b in batches.iter().skip(1) {
        // Same guard as `cast_to_union_schema`, and for the same reason: indexing past a
        // narrower branch's columns panics, and this crate does not panic on user data.
        if b.num_columns() != ncols {
            return Err(arrow::error::ArrowError::InvalidArgumentError(format!(
                "set-operation branches have different column counts: {} vs {ncols}",
                b.num_columns()
            ))
            .into());
        }
        for (c, t) in target.iter_mut().enumerate().take(ncols) {
            let bt = b.column(c).data_type();
            if bt != t {
                // Only coerce when the two branch types have a SAFE common supertype
                // (a numeric widening). For a genuinely incompatible pair — e.g. int64
                // vs string — there is none, so fail fast with a typed error rather than
                // arrow's *lenient* string→int64 cast silently nulling the non-numeric
                // values (data corruption), or a downstream downcast panic on the
                // still-mismatched schema.
                match promote_union_type(t, bt) {
                    Some(common) => {
                        *t = common;
                        mismatch = true;
                    }
                    None => {
                        return Err(InterpError::IncompatibleSetOpTypes {
                            col: c,
                            left: t.to_string(),
                            right: bt.to_string(),
                        })
                    }
                }
            }
        }
    }
    if !mismatch {
        return Ok(None);
    }
    // Rebuild a schema carrying the promoted types (a column is nullable if any branch's is).
    let base = first.schema();
    let fields: Vec<Field> = (0..ncols)
        .map(|c| {
            let nullable = batches.iter().any(|b| b.schema().field(c).is_nullable());
            base.field(c)
                .clone()
                .with_data_type(target[c].clone())
                .with_nullable(nullable)
        })
        .collect();
    Ok(Some(Arc::new(Schema::new_with_metadata(
        fields,
        base.metadata().clone(),
    ))))
}

/// Cast one branch batch to the union's target schema, sharing every column that already
/// carries the target type rather than copying it.
pub(crate) fn cast_to_union_schema(
    batch: &RecordBatch,
    schema: &SchemaRef,
) -> Result<RecordBatch, InterpError> {
    // A branch of a different *width* is a malformed plan rather than a coercible mismatch,
    // and indexing past a batch's columns panics. Return the arrow error instead: this crate
    // never panics on a path that can see user data.
    if batch.num_columns() != schema.fields().len() {
        return Err(arrow::error::ArrowError::InvalidArgumentError(format!(
            "set-operation branches have different column counts: {} vs {}",
            batch.num_columns(),
            schema.fields().len()
        ))
        .into());
    }
    let cols: Vec<ArrayRef> = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(c, f)| {
            let col = batch.column(c);
            if col.data_type() == f.data_type() {
                Ok(Arc::clone(col))
            } else {
                Ok(arrow::compute::cast(col, f.data_type())?)
            }
        })
        .collect::<Result<_, InterpError>>()?;
    Ok(RecordBatch::try_new(Arc::clone(schema), cols)?)
}

/// The common type two set-operation branch columns must both widen to, so neither side is
/// narrowed — or `None` when there is none.
///
/// This is `bc_expr::common_supertype`, not a rule of its own. A set operation and a binary
/// comparison are asking the identical question ("what type holds both of these?"), and when
/// this function answered it separately the two disagreed on every case the narrow numeric
/// rule did not cover: an all-null branch, two decimals of differing scale, a `timestamp[ms]`
/// against a `timestamp[us]`, a date against a timestamp. Each of those is what reading a
/// directory of files written over time actually produces, and each raised
/// `IncompatibleSetOpTypes` on a union the control plane had already advertised a type for.
///
/// `None` still means the caller declines to coerce and the mismatch surfaces as a clean
/// typed error — never a lossy string→int cast that silently nulls the incompatible branch.
fn promote_union_type(a: &DataType, b: &DataType) -> Option<DataType> {
    bc_expr::common_supertype(a, b)
}

#[cfg(test)]
mod union_coerce_tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};

    fn batch(name: &str, ty: DataType, col: ArrayRef) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(name, ty, true)]));
        RecordBatch::try_new(schema, vec![col]).unwrap()
    }

    /// An `int64 ∪ float64` union must coerce both branches to Float64 so `concat`/`distinct`
    /// accept them — matching DuckDB (which promotes to DOUBLE) and the union's own advertised
    /// schema, instead of erroring on the type mismatch.
    #[test]
    fn int64_and_float64_branches_coerce_to_double() {
        let left = batch(
            "x",
            DataType::Int64,
            Arc::new(Int64Array::from(vec![1i64, 2])),
        );
        let right = batch(
            "x",
            DataType::Float64,
            Arc::new(Float64Array::from(vec![3.5f64, 4.5])),
        );
        let out = coerce_union_branches(vec![left, right]).unwrap();
        assert_eq!(out.len(), 2);
        for b in &out {
            assert_eq!(b.column(0).data_type(), &DataType::Float64);
        }
        // The int branch's values survive the widening exactly.
        let l = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(l.value(0), 1.0);
        assert_eq!(l.value(1), 2.0);
        // Concatenation of the coerced branches now succeeds (it would error pre-coercion).
        let cols: Vec<&dyn arrow::array::Array> =
            out.iter().map(|b| b.column(0).as_ref()).collect();
        let cat = arrow::compute::concat(&cols).unwrap();
        assert_eq!(cat.len(), 4);
    }

    /// A same-type union is returned untouched (identity), paying only the schema scan.
    #[test]
    fn matching_types_are_untouched() {
        let a = batch("x", DataType::Int64, Arc::new(Int64Array::from(vec![1i64])));
        let b = batch("x", DataType::Int64, Arc::new(Int64Array::from(vec![2i64])));
        let out = coerce_union_branches(vec![a, b]).unwrap();
        assert_eq!(out[0].column(0).data_type(), &DataType::Int64);
        assert_eq!(out[1].column(0).data_type(), &DataType::Int64);
    }

    /// A branch whose column is all-null carries the `Null` type, which has no values to
    /// lose, so it adopts the typed branch. This is the shape `SELECT NULL AS x UNION ALL
    /// SELECT 1` produces and the one an all-null Parquet column produces; both used to
    /// fail with `IncompatibleSetOpTypes` while the control plane advertised `int64`.
    #[test]
    fn an_all_null_branch_adopts_the_typed_branch() {
        let nulls = batch(
            "x",
            DataType::Null,
            Arc::new(arrow::array::NullArray::new(2)),
        );
        let typed = batch(
            "x",
            DataType::Int64,
            Arc::new(Int64Array::from(vec![1i64, 2])),
        );
        let out = coerce_union_branches(vec![nulls, typed]).unwrap();
        for b in &out {
            assert_eq!(b.column(0).data_type(), &DataType::Int64);
        }
        // The formerly-null branch reads as two nulls of the adopted type, not as values.
        assert_eq!(out[0].column(0).null_count(), 2);
    }

    /// Two decimal branches of differing scale — a `decimal(10,2)` day partition beside a
    /// `decimal(12,4)` one, which a schema change on an upstream table produces — meet at
    /// the finer scale rather than raising.
    #[test]
    fn decimal_branches_of_differing_scale_unify() {
        use arrow::array::Decimal128Array;
        let a = batch(
            "x",
            DataType::Decimal128(10, 2),
            Arc::new(
                Decimal128Array::from(vec![150i128])
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ),
        );
        let b = batch(
            "x",
            DataType::Decimal128(12, 4),
            Arc::new(
                Decimal128Array::from(vec![22_500i128])
                    .with_precision_and_scale(12, 4)
                    .unwrap(),
            ),
        );
        let out = coerce_union_branches(vec![a, b]).unwrap();
        for b in &out {
            assert_eq!(b.column(0).data_type(), &DataType::Decimal128(12, 4));
        }
        // 1.50 rescaled to four places is 15000 — the value is preserved, not truncated.
        let l = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .unwrap();
        assert_eq!(l.value(0), 15_000);
    }

    /// Two timestamp branches of differing resolution meet at the finer unit. Reading a
    /// directory whose older files wrote `timestamp[ms]` and newer ones `timestamp[us]` is
    /// the ordinary case, and it used to be a hard error.
    #[test]
    fn timestamp_branches_of_differing_unit_unify() {
        use arrow::array::{TimestampMicrosecondArray, TimestampMillisecondArray};
        use arrow::datatypes::TimeUnit;
        let ms = batch(
            "x",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            Arc::new(TimestampMillisecondArray::from(vec![1_000i64])),
        );
        let us = batch(
            "x",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            Arc::new(TimestampMicrosecondArray::from(vec![2_000_000i64])),
        );
        let out = coerce_union_branches(vec![ms, us]).unwrap();
        for b in &out {
            assert_eq!(
                b.column(0).data_type(),
                &DataType::Timestamp(TimeUnit::Microsecond, None)
            );
        }
        // One second in millis is one million micros — the instant is unchanged.
        let l = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();
        assert_eq!(l.value(0), 1_000_000);
    }

    /// A pair with no lossless common type still declines, so the mismatch surfaces as a
    /// typed error rather than arrow's lenient string→int cast silently nulling a branch.
    #[test]
    fn incompatible_branches_still_error() {
        use arrow::array::StringArray;
        let a = batch("x", DataType::Int64, Arc::new(Int64Array::from(vec![1i64])));
        let b = batch("x", DataType::Utf8, Arc::new(StringArray::from(vec!["z"])));
        assert!(matches!(
            coerce_union_branches(vec![a, b]),
            Err(InterpError::IncompatibleSetOpTypes { .. })
        ));
    }
}
