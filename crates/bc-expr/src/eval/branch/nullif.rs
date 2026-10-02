//! `NULLIF(left, right)`: `left`, except null wherever `left = right`.
//!
//! This is also the control plane's *typed NULL*. The IR has no untyped null, so a `CASE`
//! with no `ELSE` (and a `then(None)`) is lowered to `nullif(v, v)` over one of its own
//! branch values: null on every row, typed like `v`. That puts `NULLIF` on the path of
//! every `CASE` whose branches are a list, a struct or a map, and the flat comparison
//! kernels refuse nested types (`Nested comparison: List(Int64) == List(Int64)`), so a
//! `when(c).then(list_col)` failed where the same `CASE` with an explicit `otherwise` ran.
//!
//! Nested operands compare through `eval_binary`'s nested path
//! ([`crate::eval::cmp::eval_nested_cmp`]), whose equality is DuckDB's: two nested nulls are
//! equal, two NaNs are equal, `-0.0` equals `0.0`, and `NULLIF` of two equal lists, structs
//! or maps is `NULL`. A top-level null on either side is not a match, so the row keeps
//! `left`, exactly as the flat path does.

use arrow::array::{Array, ArrayRef};

use arrow::record_batch::RecordBatch;

use crate::eval::binary::{eval_binary, scalar_binary_on};
use crate::eval::coerce::as_bool;
use crate::{BinaryOp, Expr, ExprError};

/// `NULLIF(left, right)` from the expressions: against a literal `right` (`NULLIF(x, 0)`, the
/// usual spelling), the comparison broadcasts the literal through the scalar path instead of
/// materializing it as a column and canonicalizing both sides — on TPC-H's
/// `COALESCE(NULLIF(l_discount, 0.00), 1.00)` that canonicalizing pass alone was a third of
/// the query. The scalar path is bit-identical to `eval_binary`, so this is the same mask.
pub(crate) fn eval_nullif_expr(
    left: &Expr,
    right: &Expr,
    batch: &RecordBatch,
) -> Result<ArrayRef, ExprError> {
    let l = left.eval(batch)?;
    if let Expr::Lit { value } = right {
        if let Some(eq) = scalar_binary_on(BinaryOp::Eq, l.clone(), value, true)? {
            return Ok(arrow::compute::nullif(&l, as_bool(&eq, "nullif")?)?);
        }
    }
    eval_nullif(&l, &right.eval(batch)?)
}

/// Evaluate `NULLIF` over two already-evaluated, equal-length operands.
fn eval_nullif(l: &ArrayRef, r: &ArrayRef) -> Result<ArrayRef, ExprError> {
    // Nothing can match an all-null right side, whatever its type: `NULLIF(x, NULL)` is
    // `x`. Answered first so a NULL typed differently from `left` is not a type error.
    if r.null_count() == r.len() {
        return Ok(l.clone());
    }
    let eq = eval_binary(BinaryOp::Eq, l, r)?;
    let mask = as_bool(&eq, "nullif")?;
    Ok(arrow::compute::nullif(l, mask)?)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, Int64Builder, ListArray, MapBuilder, StructArray};
    use arrow::datatypes::{DataType, Field, Float64Type, Int64Type};

    use super::*;

    fn list(rows: Vec<Option<Vec<Option<i64>>>>) -> ArrayRef {
        Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(rows))
    }

    fn nulls(arr: &ArrayRef) -> Vec<bool> {
        (0..arr.len()).map(|i| arr.is_null(i)).collect()
    }

    /// `nullif(v, v)` is the typed NULL: null on every row, typed like `v`.
    #[test]
    fn nullif_of_a_list_with_itself_is_a_typed_null() {
        let l = list(vec![Some(vec![Some(1), None]), None, Some(vec![])]);
        let out = eval_nullif(&l, &l).expect("nested nullif");
        assert_eq!(out.data_type(), l.data_type());
        assert_eq!(nulls(&out), vec![true; 3]);
    }

    /// Unequal rows keep `left`; a null `right` is never a match.
    #[test]
    fn nullif_of_two_lists_nulls_only_the_equal_rows() {
        let l = list(vec![
            Some(vec![Some(1), Some(2)]),
            Some(vec![Some(1), Some(2)]),
            Some(vec![Some(3)]),
        ]);
        let r = list(vec![
            Some(vec![Some(1), Some(2)]),
            Some(vec![Some(1), Some(3)]),
            None,
        ]);
        let out = eval_nullif(&l, &r).expect("nested nullif");
        assert_eq!(nulls(&out), vec![true, false, false]);
    }

    #[test]
    fn nullif_of_a_struct_with_itself_is_a_typed_null() {
        let x: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None]));
        let s: ArrayRef = Arc::new(StructArray::from(vec![(
            Arc::new(Field::new("x", DataType::Int64, true)),
            x,
        )]));
        let out = eval_nullif(&s, &s).expect("struct nullif");
        assert_eq!(out.data_type(), s.data_type());
        assert_eq!(nulls(&out), vec![true, true]);
    }

    #[test]
    fn nullif_of_a_map_with_itself_is_a_typed_null() {
        let mut b = MapBuilder::new(None, Int64Builder::new(), Int64Builder::new());
        b.keys().append_value(1);
        b.values().append_value(2);
        b.append(true).expect("map row");
        b.append(false).expect("null map row");
        let m: ArrayRef = Arc::new(b.finish());
        let out = eval_nullif(&m, &m).expect("map nullif");
        assert_eq!(out.data_type(), m.data_type());
        assert_eq!(nulls(&out), vec![true, true]);
    }

    /// `NULLIF(x, NULL)` is `x` even when the NULL is typed differently from `x`.
    #[test]
    fn an_all_null_right_side_returns_left() {
        let l = list(vec![Some(vec![Some(1)])]);
        let r: ArrayRef = Arc::new(Int64Array::from(vec![None::<i64>]));
        let out = eval_nullif(&l, &r).expect("nullif with null");
        assert_eq!(nulls(&out), vec![false]);
    }

    /// `-0.0` and `0.0` are one value inside a list too, as in DuckDB.
    #[test]
    fn nullif_of_lists_uses_the_engines_float_identity() {
        let l: ArrayRef = Arc::new(ListArray::from_iter_primitive::<Float64Type, _, _>(vec![
            Some(vec![Some(-0.0)]),
        ]));
        let r: ArrayRef = Arc::new(ListArray::from_iter_primitive::<Float64Type, _, _>(vec![
            Some(vec![Some(0.0)]),
        ]));
        assert_eq!(nulls(&eval_nullif(&l, &r).expect("nullif")), vec![true]);
    }

    /// Against a literal, `NULLIF` broadcasts it through the scalar path; the mask must be the
    /// one the materialized comparison gives — floats with NaN and both zeros, an integer
    /// column against a float literal, and a date against a string.
    #[test]
    fn a_literal_right_side_matches_the_materialized_comparison() {
        use arrow::array::{Date32Array, Float64Array};
        use arrow::datatypes::Schema;

        use crate::Literal;

        let f: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(0.0),
            Some(-0.0),
            Some(f64::NAN),
            None,
            Some(1.0),
        ]));
        let i: ArrayRef = Arc::new(Int64Array::from(vec![Some(0), Some(2), None]));
        let d: ArrayRef = Arc::new(Date32Array::from(vec![Some(11_016), Some(0), None]));
        let cases: Vec<(ArrayRef, Vec<Literal>)> = vec![
            (
                f,
                vec![
                    Literal::Float(0.0),
                    Literal::Float(-0.0),
                    Literal::Float(f64::NAN),
                    Literal::Int(1),
                ],
            ),
            (i, vec![Literal::Float(2.0), Literal::Int(0)]),
            (d, vec![Literal::Str("2000-02-29".into())]),
        ];
        for (arr, lits) in cases {
            let schema = Schema::new(vec![Field::new("x", arr.data_type().clone(), true)]);
            let batch = RecordBatch::try_new(Arc::new(schema), vec![arr.clone()]).unwrap();
            for lit in lits {
                let right = Expr::Lit { value: lit.clone() };
                let left = Expr::Col { name: "x".into() };
                let got = eval_nullif_expr(&left, &right, &batch).expect("nullif");
                let want = eval_nullif(&arr, &lit.to_array(arr.len())).expect("nullif");
                assert_eq!(nulls(&got), nulls(&want), "{lit:?} on {}", arr.data_type());
                assert_eq!(got.data_type(), want.data_type());
            }
        }
    }
}
