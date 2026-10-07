//! Element-wise arithmetic between two numeric `List` columns for `Expr::ListZip`
//! (`list_add`/`list_subtract`/`list_multiply`) — the embedding-math primitive.
//!
//! Pairs elements positionally and returns a `List<Float64>`. Both operands are normalized
//! through [`as_var_list`], so a `FixedSizeList` (the fixed-shape-tensor / embedding type)
//! is accepted as readily as a variable list. Lengths must match per row — a mismatch is a
//! clean error, not a silent truncation to a bogus vector (the same discipline the distance
//! kernels use). A null list row on either side yields a null output row; a null *element*
//! yields a null at that position.
//!
//! [`eval_list_zip_struct`] is the other zip: it pairs elements into a struct rather than
//! combining them, for `list.zip` (DuckDB `list_zip`).

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Builder, ListArray, ListBuilder, StructArray};
use arrow::buffer::{NullBuffer, OffsetBuffer};
use arrow::compute::{cast, take};
use arrow::datatypes::{DataType, Field, Fields, Float64Type, UInt32Type};

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

/// `list.zip(other, pad=)`: pair the two lists of each row element by element into
/// `List<Struct<left, right>>`.
///
/// Two lists of different lengths are an error naming both lengths, unless `pad`, which
/// extends the shorter with nulls as DuckDB's `list_zip` does. A null list on either side is
/// a null row. The elements are gathered with one `take` per side, so the struct's children
/// keep their element types exactly.
pub(crate) fn eval_list_zip_struct(
    left: &ArrayRef,
    right: &ArrayRef,
    pad: bool,
) -> Result<ArrayRef, ExprError> {
    use arrow::array::{AsArray, PrimitiveArray};

    let left = as_var_list(left, "list.zip (left)")?;
    let right = as_var_list(right, "list.zip (right)")?;
    let (la, ra) = (left.as_list::<i32>(), right.as_list::<i32>());
    let (lo, ro) = (la.value_offsets(), ra.value_offsets());
    let (mut li, mut ri): (Vec<Option<u32>>, Vec<Option<u32>>) = (vec![], vec![]);
    let mut offsets = vec![0i32];
    let mut valid = Vec::with_capacity(la.len());
    for i in 0..la.len() {
        let ok = la.is_valid(i) && ra.is_valid(i);
        valid.push(ok);
        if ok {
            let (ls, llen) = (lo[i] as u32, (lo[i + 1] - lo[i]) as u32);
            let (rs, rlen) = (ro[i] as u32, (ro[i + 1] - ro[i]) as u32);
            if llen != rlen && !pad {
                return Err(ExprError::InvalidArgument {
                    func: "list.zip".into(),
                    reason: format!(
                        "the two lists must have the same length, got {llen} and {rlen}; \
                         pass pad=True to extend the shorter with nulls"
                    ),
                });
            }
            for k in 0..llen.max(rlen) {
                li.push((k < llen).then_some(ls + k));
                ri.push((k < rlen).then_some(rs + k));
            }
        }
        offsets.push(li.len() as i32);
    }
    let gather = |values: &ArrayRef, idx: Vec<Option<u32>>| {
        take(
            values.as_ref(),
            &PrimitiveArray::<UInt32Type>::from(idx),
            None,
        )
    };
    let (lv, rv) = (gather(la.values(), li)?, gather(ra.values(), ri)?);
    let fields = Fields::from(vec![
        Field::new("left", lv.data_type().clone(), true),
        Field::new("right", rv.data_type().clone(), true),
    ]);
    let pairs = StructArray::try_new(fields.clone(), vec![lv, rv], None)?;
    Ok(Arc::new(ListArray::try_new(
        Arc::new(Field::new_list_field(DataType::Struct(fields), true)),
        OffsetBuffer::new(offsets.into()),
        Arc::new(pairs),
        Some(NullBuffer::from(valid)),
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float32Array;
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
        let a = Arc::new(arrow::array::FixedSizeListArray::new(field, 2, child, None)) as ArrayRef;
        let out = eval_list_zip(ListZipOp::Add, &a, &a).unwrap();
        assert_eq!(
            vals(&out),
            vec![
                Some(vec![Some(2.0), Some(4.0)]),
                Some(vec![Some(6.0), Some(8.0)])
            ]
        );
    }

    #[test]
    fn zip_pairs_elements_and_pads_only_when_asked() {
        use arrow::array::{AsArray, StringArray};
        use arrow::datatypes::Int64Type;
        let l: ArrayRef = Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
            Some(vec![Some(1), Some(2), Some(3)]),
            None,
            Some(vec![]),
        ]));
        let words = ListArray::try_new(
            Arc::new(Field::new_list_field(DataType::Utf8, true)),
            OffsetBuffer::new(vec![0, 2, 3, 3].into()),
            Arc::new(StringArray::from(vec![Some("a"), None, Some("z")])),
            None,
        )
        .unwrap();
        let r: ArrayRef = Arc::new(words);
        let err = eval_list_zip_struct(&l, &r, false).unwrap_err().to_string();
        assert!(
            err.contains("got 3 and 2") && err.contains("pad=True"),
            "{err}"
        );

        let out = eval_list_zip_struct(&l, &r, true).unwrap();
        let out = out.as_list::<i32>();
        assert!(out.is_valid(0) && out.is_null(1) && out.is_valid(2));
        assert_eq!(out.value_length(0), 3);
        assert_eq!(out.value_length(2), 0);
        let pairs = out.value(0);
        let pairs = pairs.as_struct();
        let left = pairs.column(0).as_primitive::<Int64Type>();
        let right = pairs.column(1).as_string::<i32>();
        assert_eq!(left.values().to_vec(), vec![1, 2, 3]);
        assert_eq!(
            right.iter().collect::<Vec<_>>(),
            vec![Some("a"), None, None]
        );
    }
}
