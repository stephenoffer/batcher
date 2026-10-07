//! Higher-order list ops for `Expr::ListTransform` / `Expr::ListFilter` (the
//! `.list.transform` / `.list.filter` accessors).
//!
//! Each carries an *element sub-expression* (the lambda body) evaluated over a batch with
//! one row per list element. Three kinds of name resolve in it:
//!
//! * `element` — the element itself (the list's flattened child);
//! * `element_index` — its 0-based position within its own list, as Int64;
//! * a **capture** — an expression the control plane lifted out of the body because it
//!   reads the enclosing row (`list.filter(element() > col("threshold"))` captures
//!   `threshold`). It is evaluated once over the outer batch and repeated for each of that
//!   row's elements, which is what DuckDB's `list_filter(a, x -> x > t)` means. The body
//!   used to see only `element`, so an outer reference failed at execution with "unknown
//!   column".
//!
//! Captures are explicit on the node, rather than the body reaching into the outer batch
//! by name, so every rewrite the optimizer applies to the enclosing projection (a rename
//! through a join, an inlined alias) reaches the captured expression as an ordinary child.
//!
//! `transform` evaluates the body over the whole child at once (columnar, not per row) and
//! rebuilds the list with the same offsets and null mask; `filter` evaluates a boolean
//! predicate and recomputes the offsets. The body is evaluated by the one `Expr::eval`
//! (the interpreter oracle), so there is no second representation and the JIT falls back.
//! Both are stateless and row-local, so single-node and distributed results are identical.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, Int64Array, ListArray, RecordBatch, UInt32Array,
};
use arrow::buffer::OffsetBuffer;
use arrow::compute::take;
use arrow::datatypes::Field;

use crate::eval::list::require_list;
use crate::{Expr, ExprError};

/// The reserved column name the element sub-expression reads (Polars `element()`).
const ELEMENT: &str = "element";
/// The reserved column name for the element's 0-based position in its list.
const ELEMENT_INDEX: &str = "element_index";

/// A list column narrowed to the child range its offsets actually cover, offsets rebased
/// to start at 0.
///
/// A sliced `ListArray` keeps its whole parent child array, so evaluating the body over
/// `values()` would run it on elements no row owns — and an outer column has no row to
/// take a value from for those.
fn compact(l: &ListArray) -> (OffsetBuffer<i32>, ArrayRef) {
    let offsets = l.value_offsets();
    let (first, last) = (offsets[0], offsets[l.len()]);
    if first == 0 && last as usize == l.values().len() {
        return (l.offsets().clone(), Arc::clone(l.values()));
    }
    let rebased: Vec<i32> = offsets.iter().map(|o| o - first).collect();
    let child = l.values().slice(first as usize, (last - first) as usize);
    (OffsetBuffer::new(rebased.into()), child)
}

/// The captured values a lambda body reads, evaluated over the enclosing batch: one
/// `(name, column)` per capture, aligned with the outer rows.
pub(crate) fn lambda_scope(
    names: &[String],
    captures: &[Expr],
    outer: &RecordBatch,
) -> Result<Vec<(String, ArrayRef)>, ExprError> {
    if names.len() != captures.len() {
        return Err(ExprError::InvalidArgument {
            func: "list lambda".into(),
            reason: format!(
                "{} capture names for {} captured expressions",
                names.len(),
                captures.len()
            ),
        });
    }
    names
        .iter()
        .zip(captures)
        .map(|(n, e)| Ok((n.clone(), e.eval(outer)?)))
        .collect()
}

/// The batch the lambda body evaluates over: one row per element of `child`, carrying
/// `element`, `element_index` when the body reads it, and each capture repeated for its
/// row's elements.
fn lambda_batch(
    offsets: &OffsetBuffer<i32>,
    child: &ArrayRef,
    body: &Expr,
    scope: &[(String, ArrayRef)],
) -> Result<RecordBatch, ExprError> {
    let mut names: Vec<&str> = Vec::new();
    body.collect_columns(&mut names);
    let reads_index = names.contains(&ELEMENT_INDEX);
    let mut columns: Vec<(&str, ArrayRef)> = vec![(ELEMENT, Arc::clone(child))];
    if reads_index || !scope.is_empty() {
        // One pass over the offsets gives both the per-element parent row and position.
        let mut parent = Vec::with_capacity(child.len());
        let mut position = Vec::with_capacity(if reads_index { child.len() } else { 0 });
        for (row, w) in offsets.windows(2).enumerate() {
            for k in 0..(w[1] - w[0]) {
                parent.push(row as u32);
                if reads_index {
                    position.push(i64::from(k));
                }
            }
        }
        if reads_index {
            columns.push((ELEMENT_INDEX, Arc::new(Int64Array::from(position))));
        }
        let parent = UInt32Array::from(parent);
        for (name, col) in scope {
            // `element`/`element_index` are bound by the lambda itself and win.
            if name != ELEMENT && name != ELEMENT_INDEX {
                columns.push((name.as_str(), take(col.as_ref(), &parent, None)?));
            }
        }
    }
    Ok(RecordBatch::try_from_iter(columns)?)
}

/// `list.transform(func)` — apply `func` to every element, preserving each row's
/// length and null mask. → `List<func's output type>`.
pub(crate) fn eval_list_transform(
    list: &ArrayRef,
    func: &Expr,
    scope: &[(String, ArrayRef)],
) -> Result<ArrayRef, ExprError> {
    let l = require_list(list, "list.transform")?;
    let l = l.as_list::<i32>();
    let (offsets, child) = compact(l);
    let new_child = func.eval(&lambda_batch(&offsets, &child, func, scope)?)?;
    let field = Arc::new(Field::new_list_field(new_child.data_type().clone(), true));
    Ok(Arc::new(ListArray::new(
        field,
        offsets,
        new_child,
        l.nulls().cloned(),
    )))
}

/// `list.filter(pred)` — keep the elements where the boolean element predicate `pred`
/// is true, recomputing each row's offsets. Type-preserving. Null list rows stay null.
pub(crate) fn eval_list_filter(
    list: &ArrayRef,
    pred: &Expr,
    scope: &[(String, ArrayRef)],
) -> Result<ArrayRef, ExprError> {
    let l = require_list(list, "list.filter")?;
    let l = l.as_list::<i32>();
    let (offsets, child) = compact(l);
    let mask_arr = pred.eval(&lambda_batch(&offsets, &child, pred, scope)?)?;
    let mask = mask_arr
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| ExprError::ExpectedBoolean {
            op: "list.filter".into(),
            got: crate::error::type_name(mask_arr.data_type()),
        })?;

    let mut keep: Vec<u32> = Vec::new();
    let mut new_offsets: Vec<i32> = Vec::with_capacity(l.len() + 1);
    new_offsets.push(0);
    for (row, w) in offsets.windows(2).enumerate() {
        if !l.is_null(row) {
            for k in w[0] as usize..w[1] as usize {
                if mask.is_valid(k) && mask.value(k) {
                    keep.push(k as u32);
                }
            }
        }
        new_offsets.push(keep.len() as i32);
    }
    let new_child = take(child.as_ref(), &UInt32Array::from(keep), None)?;
    let field = Arc::new(Field::new_list_field(child.data_type().clone(), true));
    Ok(Arc::new(ListArray::new(
        field,
        OffsetBuffer::new(new_offsets.into()),
        new_child,
        l.nulls().cloned(),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BinaryOp, Literal};
    use arrow::datatypes::Int64Type;

    fn col(name: &str) -> Expr {
        Expr::Col { name: name.into() }
    }

    fn bin(op: BinaryOp, left: Expr, right: Expr) -> Expr {
        Expr::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    type Rows = Vec<Option<Vec<Option<i64>>>>;

    fn lists(rows: Rows) -> ArrayRef {
        Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(rows))
    }

    fn rows(arr: &ArrayRef) -> Rows {
        let l = arr.as_list::<i32>();
        (0..l.len())
            .map(|i| {
                (!l.is_null(i)).then(|| {
                    let v = l.value(i);
                    let v = v.as_primitive::<Int64Type>();
                    (0..v.len())
                        .map(|k| v.is_valid(k).then(|| v.value(k)))
                        .collect()
                })
            })
            .collect()
    }

    fn scope(th: Vec<Option<i64>>) -> Vec<(String, ArrayRef)> {
        vec![("th".to_string(), Arc::new(Int64Array::from(th)) as ArrayRef)]
    }

    #[test]
    fn a_capture_resolves_to_the_enclosing_row() {
        let a = lists(vec![
            Some(vec![Some(1), Some(5), Some(9)]),
            None,
            Some(vec![Some(4), Some(6)]),
        ]);
        let batch = scope(vec![Some(4), Some(0), Some(5)]);
        let pred = bin(BinaryOp::Gt, col(ELEMENT), col("th"));
        let got = eval_list_filter(&a, &pred, &batch).unwrap();
        assert_eq!(
            rows(&got),
            vec![Some(vec![Some(5), Some(9)]), None, Some(vec![Some(6)])]
        );
    }

    #[test]
    fn element_index_is_the_position_within_each_list() {
        let a = lists(vec![
            Some(vec![Some(10), Some(20)]),
            Some(vec![]),
            Some(vec![Some(7), None, Some(9)]),
        ]);
        let batch = scope(vec![None, None, None]);
        let body = bin(BinaryOp::Add, col(ELEMENT), col(ELEMENT_INDEX));
        let got = eval_list_transform(&a, &body, &batch).unwrap();
        assert_eq!(
            rows(&got),
            vec![
                Some(vec![Some(10), Some(21)]),
                Some(vec![]),
                Some(vec![Some(7), None, Some(11)])
            ]
        );
    }

    #[test]
    fn a_sliced_list_lines_its_elements_up_with_its_own_rows() {
        let a = lists(vec![
            Some(vec![Some(1)]),
            Some(vec![Some(2), Some(3)]),
            Some(vec![Some(4)]),
        ]);
        let sliced = a.slice(1, 2);
        let batch = scope(vec![Some(100), Some(200)]);
        let body = bin(
            BinaryOp::Add,
            bin(BinaryOp::Add, col(ELEMENT), col("th")),
            col(ELEMENT_INDEX),
        );
        let got = eval_list_transform(&sliced, &body, &batch).unwrap();
        assert_eq!(
            rows(&got),
            vec![Some(vec![Some(102), Some(104)]), Some(vec![Some(204)])]
        );
        let first_only = bin(
            BinaryOp::Eq,
            col(ELEMENT_INDEX),
            Expr::Lit {
                value: Literal::Int(0),
            },
        );
        let got = eval_list_filter(&sliced, &first_only, &batch).unwrap();
        assert_eq!(rows(&got), vec![Some(vec![Some(2)]), Some(vec![Some(4)])]);
    }

    #[test]
    fn a_column_neither_bound_nor_captured_is_unknown() {
        let a = lists(vec![Some(vec![Some(1)])]);
        let batch = scope(vec![Some(0)]);
        let err = eval_list_transform(&a, &col("nope"), &batch).unwrap_err();
        assert!(matches!(err, ExprError::UnknownColumn(n) if n == "nope"));
    }
}
