//! Element-wise arithmetic between two numeric `List` columns for `Expr::ListZip`
//! (`list_add`/`list_subtract`/`list_multiply`) — the embedding-math primitive.
//!
//! Pairs elements positionally and returns a `List<Float64>`. Both operands are normalized
//! through [`as_var_list`], so a `FixedSizeList` (the fixed-shape-tensor / embedding type)
//! is accepted as readily as a variable list. Lengths must match per row — a mismatch is a
//! clean error, not a silent truncation to a bogus vector (the same discipline the distance
//! kernels use). A null list row on either side yields a null output row; a null *element*
//! yields a null at that position.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Builder, ListBuilder};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Float64Type};

use super::as_var_list;
use crate::{ExprError, ListZipOp};

pub(crate) fn eval_list_zip(
    op: ListZipOp,
    left: &ArrayRef,
    right: &ArrayRef,
) -> Result<ArrayRef, ExprError> {
    use arrow::array::AsArray;

    let name = format!("list.{op:?}");
    let left = as_var_list(left, &format!("{name} (left)"))?;
    let right = as_var_list(right, &format!("{name} (right)"))?;
    let (la, ra) = (left.as_list::<i32>(), right.as_list::<i32>());

    // Both children to Float64 so mixed widths (Int64 list + Float32 tensor) combine.
    let lc = cast(la.values(), &DataType::Float64)?;
    let rc = cast(ra.values(), &DataType::Float64)?;
    let lf = lc.as_primitive::<Float64Type>();
    let rf = rc.as_primitive::<Float64Type>();
    let (lo, ro) = (la.value_offsets(), ra.value_offsets());

    let mut b = ListBuilder::new(Float64Builder::new());
    for i in 0..la.len() {
        if la.is_null(i) || ra.is_null(i) {
            b.append(false);
            continue;
        }
        let (ls, le) = (lo[i] as usize, lo[i + 1] as usize);
        let (rs, re) = (ro[i] as usize, ro[i + 1] as usize);
        let (llen, rlen) = (le - ls, re - rs);
        if llen != rlen {
            return Err(ExprError::InvalidArgument {
                func: name,
                reason: format!(
                    "list lengths must be equal for element-wise arithmetic, \
                     got left {llen} and right {rlen}"
                ),
            });
        }
        for k in 0..llen {
            let (lk, rk) = (ls + k, rs + k);
            if !lf.is_valid(lk) || !rf.is_valid(rk) {
                b.values().append_null();
                continue;
            }
            let (x, y) = (lf.value(lk), rf.value(rk));
            let v = match op {
                ListZipOp::Add => x + y,
                ListZipOp::Subtract => x - y,
                ListZipOp::Multiply => x * y,
            };
            b.values().append_value(v);
        }
        b.append(true);
    }
    Ok(Arc::new(b.finish()) as ArrayRef)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float32Array, ListArray};
    use arrow::datatypes::Field;

    fn list_f64(rows: Vec<Option<Vec<Option<f64>>>>) -> ArrayRef {
        Arc::new(ListArray::from_iter_primitive::<Float64Type, _, _>(rows)) as ArrayRef
    }

    fn vals(out: &ArrayRef) -> Vec<Option<Vec<Option<f64>>>> {
        use arrow::array::AsArray;
        let l = out.as_list::<i32>();
        (0..l.len())
            .map(|i| {
                if l.is_null(i) {
                    None
                } else {
                    let row = l.value(i);
                    let p = row.as_primitive::<Float64Type>();
                    Some(
                        (0..p.len())
                            .map(|k| p.is_valid(k).then(|| p.value(k)))
                            .collect(),
                    )
                }
            })
            .collect()
    }

    #[test]
    fn add_subtract_multiply() {
        let a = list_f64(vec![
            Some(vec![Some(1.0), Some(2.0)]),
            Some(vec![Some(3.0), Some(4.0)]),
        ]);
        let b = list_f64(vec![
            Some(vec![Some(10.0), Some(20.0)]),
            Some(vec![Some(1.0), Some(1.0)]),
        ]);
        assert_eq!(
            vals(&eval_list_zip(ListZipOp::Add, &a, &b).unwrap()),
            vec![
                Some(vec![Some(11.0), Some(22.0)]),
                Some(vec![Some(4.0), Some(5.0)])
            ]
        );
        assert_eq!(
            vals(&eval_list_zip(ListZipOp::Subtract, &a, &b).unwrap()),
            vec![
                Some(vec![Some(-9.0), Some(-18.0)]),
                Some(vec![Some(2.0), Some(3.0)])
            ]
        );
        assert_eq!(
            vals(&eval_list_zip(ListZipOp::Multiply, &a, &b).unwrap()),
            vec![
                Some(vec![Some(10.0), Some(40.0)]),
                Some(vec![Some(3.0), Some(4.0)])
            ]
        );
    }

    #[test]
    fn length_mismatch_is_an_error() {
        let a = list_f64(vec![Some(vec![Some(1.0), Some(2.0)])]);
        let b = list_f64(vec![Some(vec![Some(1.0)])]);
        assert!(eval_list_zip(ListZipOp::Add, &a, &b).is_err());
    }

    #[test]
    fn nulls_propagate_per_row_and_per_element() {
        let a = list_f64(vec![None, Some(vec![Some(1.0), None])]);
        let b = list_f64(vec![
            Some(vec![Some(1.0)]),
            Some(vec![Some(5.0), Some(6.0)]),
        ]);
        let out = vals(&eval_list_zip(ListZipOp::Add, &a, &b).unwrap());
        assert_eq!(out[0], None); // null list row
        assert_eq!(out[1], Some(vec![Some(6.0), None])); // null element → null
    }

    #[test]
    fn accepts_fixed_size_list_tensor_columns() {
        let child = Arc::new(Float32Array::from(vec![1.0f32, 2.0, 3.0, 4.0])) as ArrayRef;
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let a = Arc::new(arrow::array::FixedSizeListArray::new(
            field.clone(),
            2,
            child,
            None,
        )) as ArrayRef;
        let out = eval_list_zip(ListZipOp::Add, &a, &a).unwrap();
        assert_eq!(
            vals(&out),
            vec![
                Some(vec![Some(2.0), Some(4.0)]),
                Some(vec![Some(6.0), Some(8.0)])
            ]
        );
    }
}
