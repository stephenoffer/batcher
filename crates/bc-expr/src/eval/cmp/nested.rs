//! `=`, `<>`, `<`, `<=`, `>`, `>=` over lists, structs and maps.
//!
//! Arrow's flat comparison kernels refuse nested types outright ("Nested comparison:
//! List(Int64) == List(Int64)"), so these go through `make_comparator`, which walks a nested
//! value element by element. The flat kernels are untouched: this path is reached only when
//! an operand is nested.
//!
//! The semantics are DuckDB's, each checked against it rather than assumed:
//!
//! * a **top-level** null on either side makes the comparison null (`[1] = NULL` is NULL);
//! * **inside** a value, null is an ordinary element that equals null and sorts *after*
//!   every non-null (`[1, NULL] = [1, NULL]` is true, `[1, 2] < [1, NULL]` is true);
//! * floats compare under the engine's float identity at every depth (`[-0.0] = [0.0]`, and
//!   NaN equals NaN and is greater than every number), which is why the operands are
//!   canonicalized with [`bc_arrow::float_ident::canon_float_nested`] first — the comparator
//!   alone ranks raw bits and would split the zeros;
//! * lists order lexicographically with a proper prefix first (`[1, 2] < [1, 2, 0]`),
//!   structs field by field, maps entry by entry in stored order (`MAP {1: 3, 2: 4}` is not
//!   `MAP {2: 4, 1: 3}` in DuckDB either).
//!
//! Operands of different nested types (`[1] = [1.0]`) are not unified here and raise.

use std::cmp::Ordering;
use std::sync::Arc;

use arrow::array::{make_comparator, new_null_array, Array, ArrayRef, BooleanArray};
use arrow::compute::SortOptions;
use arrow::datatypes::DataType;
use bc_arrow::float_ident::canon_float_nested;

use crate::{BinaryOp, ExprError};

/// Whether the flat comparison kernels refuse `dt`, so a comparison must come here.
pub(crate) fn is_nested(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::List(_)
            | DataType::LargeList(_)
            | DataType::FixedSizeList(..)
            | DataType::ListView(_)
            | DataType::LargeListView(_)
            | DataType::Struct(_)
            | DataType::Map(..)
    )
}

/// Compare two equal-length operands, at least one nested, row by row.
///
/// `op` must be one of the six comparison operators.
pub(crate) fn eval_nested_cmp(
    op: BinaryOp,
    l: &ArrayRef,
    r: &ArrayRef,
) -> Result<ArrayRef, ExprError> {
    let n = l.len();
    // An operand that is null on every row answers null on every row whatever its type,
    // so an untyped NULL (`col(list) == None`) is not a type error.
    if l.null_count() == n || r.null_count() == n {
        return Ok(new_null_array(&DataType::Boolean, n));
    }
    let accept: fn(Ordering) -> bool = match op {
        BinaryOp::Eq => Ordering::is_eq,
        BinaryOp::Ne => Ordering::is_ne,
        BinaryOp::Lt => Ordering::is_lt,
        BinaryOp::Le => Ordering::is_le,
        BinaryOp::Gt => Ordering::is_gt,
        BinaryOp::Ge => Ordering::is_ge,
        other => {
            return Err(ExprError::InvalidArgument {
                func: format!("{other:?}"),
                reason: "not a comparison of nested values".into(),
            })
        }
    };
    let (lc, rc) = (canon_float_nested(l), canon_float_nested(r));
    // Nulls last is DuckDB's rule for a null *inside* a value; top-level nulls never reach
    // the comparator (they are masked below), so this option decides only nested ones.
    let opts = SortOptions {
        descending: false,
        nulls_first: false,
    };
    let cmp = make_comparator(lc.as_ref(), rc.as_ref(), opts).map_err(|e| {
        ExprError::InvalidArgument {
            func: format!("{op:?}"),
            reason: format!(
                "cannot compare {} with {}: {e}",
                l.data_type(),
                r.data_type()
            ),
        }
    })?;
    let out: BooleanArray = (0..n)
        .map(|i| (l.is_valid(i) && r.is_valid(i)).then(|| accept(cmp(i, i))))
        .collect();
    Ok(Arc::new(out))
}

#[cfg(test)]
mod tests {
    use arrow::array::StructArray;
    use arrow::array::{AsArray, Float64Array, Int64Array, Int64Builder, ListArray, MapBuilder};
    use arrow::datatypes::{Field, Float64Type, Int64Type};

    use super::*;

    fn ints(rows: Vec<Option<Vec<Option<i64>>>>) -> ArrayRef {
        Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(rows))
    }

    fn floats(rows: Vec<Option<Vec<Option<f64>>>>) -> ArrayRef {
        Arc::new(ListArray::from_iter_primitive::<Float64Type, _, _>(rows))
    }

    fn run(op: BinaryOp, l: &ArrayRef, r: &ArrayRef) -> Vec<Option<bool>> {
        let out = eval_nested_cmp(op, l, r).expect("nested comparison");
        out.as_boolean().iter().collect()
    }

    /// Every expectation below is DuckDB's answer to the same comparison.
    #[test]
    fn list_equality_matches_duckdb() {
        let l = ints(vec![
            Some(vec![Some(1), None]),
            Some(vec![Some(1), Some(2)]),
            Some(vec![Some(1)]),
            Some(vec![]),
            Some(vec![None]),
        ]);
        let r = ints(vec![
            Some(vec![Some(1), None]),
            Some(vec![Some(1), Some(3)]),
            None,
            Some(vec![]),
            Some(vec![]),
        ]);
        assert_eq!(
            run(BinaryOp::Eq, &l, &r),
            vec![Some(true), Some(false), None, Some(true), Some(false)]
        );
        assert_eq!(
            run(BinaryOp::Ne, &l, &r),
            vec![Some(false), Some(true), None, Some(false), Some(true)]
        );
    }

    #[test]
    fn float_leaves_use_the_engines_float_identity() {
        let neg_nan = f64::from_bits(0xfff8_0000_0000_0001);
        let l = floats(vec![Some(vec![Some(-0.0)]), Some(vec![Some(f64::NAN)])]);
        let r = floats(vec![Some(vec![Some(0.0)]), Some(vec![Some(neg_nan)])]);
        assert_eq!(run(BinaryOp::Eq, &l, &r), vec![Some(true), Some(true)]);
        // NaN is greater than every number, however its sign bit is set.
        let big = floats(vec![Some(vec![Some(f64::INFINITY)])]);
        let nan = floats(vec![Some(vec![Some(neg_nan)])]);
        assert_eq!(run(BinaryOp::Gt, &nan, &big), vec![Some(true)]);
    }

    #[test]
    fn list_ordering_matches_duckdb() {
        let l = ints(vec![
            Some(vec![Some(1), Some(2)]),
            Some(vec![Some(1), Some(2)]),
            Some(vec![Some(1), None]),
            Some(vec![Some(1), Some(2)]),
        ]);
        let r = ints(vec![
            Some(vec![Some(1), Some(3)]),
            Some(vec![Some(1), Some(2), Some(0)]),
            Some(vec![Some(1), Some(2)]),
            Some(vec![Some(1), None]),
        ]);
        assert_eq!(
            run(BinaryOp::Lt, &l, &r),
            vec![Some(true), Some(true), Some(false), Some(true)]
        );
        assert_eq!(
            run(BinaryOp::Ge, &l, &r),
            vec![Some(false), Some(false), Some(true), Some(false)]
        );
    }

    #[test]
    fn struct_and_map_comparisons_match_duckdb() {
        let s = |v: Vec<Option<i64>>| -> ArrayRef {
            Arc::new(StructArray::from(vec![(
                Arc::new(Field::new("x", DataType::Int64, true)),
                Arc::new(Int64Array::from(v)) as ArrayRef,
            )]))
        };
        let (a, b) = (s(vec![None, Some(1)]), s(vec![None, Some(2)]));
        assert_eq!(run(BinaryOp::Eq, &a, &b), vec![Some(true), Some(false)]);
        assert_eq!(run(BinaryOp::Lt, &a, &b), vec![Some(false), Some(true)]);

        let map = |pairs: &[(i64, i64)]| -> ArrayRef {
            let mut m = MapBuilder::new(None, Int64Builder::new(), Int64Builder::new());
            for (k, v) in pairs {
                m.keys().append_value(*k);
                m.values().append_value(*v);
            }
            m.append(true).expect("map row");
            Arc::new(m.finish())
        };
        let (m1, m2) = (map(&[(1, 3), (2, 4)]), map(&[(2, 4), (1, 3)]));
        assert_eq!(run(BinaryOp::Eq, &m1, &m1), vec![Some(true)]);
        assert_eq!(run(BinaryOp::Eq, &m1, &m2), vec![Some(false)]);
        assert_eq!(
            run(BinaryOp::Lt, &map(&[(1, 2)]), &map(&[(1, 3)])),
            vec![Some(true)]
        );
    }

    #[test]
    fn an_all_null_operand_of_another_type_is_null() {
        let l = ints(vec![Some(vec![Some(1)]), None]);
        let r: ArrayRef = Arc::new(Float64Array::from(vec![None, None]));
        assert_eq!(run(BinaryOp::Eq, &l, &r), vec![None, None]);
    }

    #[test]
    fn different_nested_types_raise() {
        let l = ints(vec![Some(vec![Some(1)])]);
        let r = floats(vec![Some(vec![Some(1.0)])]);
        assert!(eval_nested_cmp(BinaryOp::Eq, &l, &r).is_err());
    }
}
