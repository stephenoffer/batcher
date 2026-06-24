//! Higher-order list ops for `Expr::ListTransform` / `Expr::ListFilter` (the
//! `.list.transform` / `.list.filter` accessors).
//!
//! Each carries an *element sub-expression* that reads the reserved `element` column
//! — the list's flattened child. `transform` evaluates it over the whole child array
//! at once (columnar, not per row) and rebuilds the list with the **same** offsets
//! and null mask; `filter` evaluates a boolean element predicate and recomputes the
//! offsets to drop the rejected elements. Because the child is evaluated by the one
//! `Expr::eval` (the interpreter oracle), there is no second representation and the
//! JIT simply falls back. Both are stateless and streaming, so single-node and
//! distributed results are identical.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanArray, ListArray, RecordBatch, UInt32Array};
use arrow::buffer::OffsetBuffer;
use arrow::compute::take;
use arrow::datatypes::Field;

use crate::eval::list::require_list;
use crate::{Expr, ExprError};

/// The reserved column name the element sub-expression reads (Polars `element()`).
const ELEMENT: &str = "element";

/// Build the single-column batch `{element: child}` the sub-expression evaluates over.
fn element_batch(child: &ArrayRef) -> Result<RecordBatch, ExprError> {
    Ok(RecordBatch::try_from_iter(vec![(ELEMENT, child.clone())])?)
}

/// `list.transform(func)` — apply `func` to every element, preserving each row's
/// length and null mask. → `List<func's output type>`.
pub(crate) fn eval_list_transform(list: &ArrayRef, func: &Expr) -> Result<ArrayRef, ExprError> {
    let l = require_list(list, "list.transform")?;
    let new_child = func.eval(&element_batch(l.values())?)?;
    let field = Arc::new(Field::new_list_field(new_child.data_type().clone(), true));
    Ok(Arc::new(ListArray::new(
        field,
        l.offsets().clone(),
        new_child,
        l.nulls().cloned(),
    )))
}

/// `list.filter(pred)` — keep the elements where the boolean element predicate `pred`
/// is true, recomputing each row's offsets. Type-preserving. Null list rows stay null.
pub(crate) fn eval_list_filter(list: &ArrayRef, pred: &Expr) -> Result<ArrayRef, ExprError> {
    let l = require_list(list, "list.filter")?;
    let child = l.values();
    let mask_arr = pred.eval(&element_batch(child)?)?;
    let mask = mask_arr
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| ExprError::ExpectedBoolean {
            op: "list.filter".into(),
            got: mask_arr.data_type().to_string(),
        })?;

    let offsets = l.value_offsets();
    let mut keep: Vec<u32> = Vec::new();
    let mut new_offsets: Vec<i32> = Vec::with_capacity(l.len() + 1);
    new_offsets.push(0);
    for row in 0..l.len() {
        if !l.is_null(row) {
            for k in offsets[row] as usize..offsets[row + 1] as usize {
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
