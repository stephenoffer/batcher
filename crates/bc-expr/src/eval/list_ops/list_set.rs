//! Set operations between two `List` columns for `Expr::ListSet`
//! (`array_intersect`/`array_except`/`array_union`).
//!
//! Each produces, per row, the distinct elements selected from the two lists —
//! intersect keeps left elements present in the right, except keeps left elements
//! absent from the right, union keeps the left elements followed by the right
//! elements not already taken. First-occurrence order is preserved and duplicates
//! removed (Spark `array_intersect`/`array_except`/`array_union`; DuckDB
//! `list_intersect`). Element comparison is type-general via Arrow's order-preserving
//! row encoding, so any element type works. A null list row yields a null result row.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, ListArray, UInt32Array};
use arrow::buffer::{NullBuffer, OffsetBuffer};
use arrow::compute::{cast, concat, take};
use arrow::datatypes::Field;
use arrow::row::{OwnedRow, RowConverter, SortField};

use crate::eval::list::require_list;
use crate::{ExprError, ListSetOp};

/// Evaluate a list set op (`intersect`/`except`/`union`) over two `List` columns.
pub(crate) fn eval_list_set(
    op: ListSetOp,
    left: &ArrayRef,
    right: &ArrayRef,
) -> Result<ArrayRef, ExprError> {
    let l = require_list(left, "list set op")?;
    let r = require_list(right, "list set op")?;
    // Promote both children to a common numeric type when they differ (e.g.
    // `List<Int64>` ∩ `List<Float64>`) so `concat` and the comparison see one type.
    // Without this, mismatched-width numeric lists errored in `concat` where DuckDB
    // coerces (`list_intersect([1,2],[2.0,3.0])` → `[2.0]`). Only numeric↔numeric is
    // promoted; other type mismatches still surface a clean error.
    let (lc, rc) = coerce_children(l.values(), r.values())?;

    // Concatenate the two children into one array so output elements can be drawn from
    // either side (union needs both); a left element keeps its index, a right element
    // `k` maps to `lc.len() + k`. One row converter over the combined child makes every
    // element comparable regardless of which list it came from.
    let combined = concat(&[lc.as_ref(), rc.as_ref()])?;
    let roffset = lc.len();
    // Compare elements by a float-canonical key so `-0.0`/`0.0` (which `RowConverter`
    // encodes to *different* bytes) and every NaN collapse the way `=`, `GROUP BY` and the
    // join keys do — otherwise `list_intersect([0.0], [-0.0])` wrongly yielded `[]`. Output
    // values are still `take`n from the original `combined`, so their exact bits survive.
    let key = crate::eval::list::float_canonical_key(&combined)?;
    let converter = RowConverter::new(vec![SortField::new(key.data_type().clone())])?;
    let crows = converter.convert_columns(std::slice::from_ref(&key))?;
    let (lo, ro) = (l.value_offsets(), r.value_offsets());

    let mut keep: Vec<u32> = Vec::new(); // indices into `combined`
    let mut offsets: Vec<i32> = Vec::with_capacity(l.len() + 1);
    offsets.push(0);
    let mut valid: Vec<bool> = Vec::with_capacity(l.len());

    for row in 0..l.len() {
        if l.is_null(row) {
            offsets.push(*offsets.last().unwrap());
            valid.push(false);
            continue;
        }
        // The right row's element set (null right row → empty set). `OwnedRow` owns its
        // bytes, so it can live in the set across the loop.
        let mut rset: HashSet<OwnedRow> = HashSet::new();
        if row < r.len() && r.is_valid(row) {
            for k in ro[row] as usize..ro[row + 1] as usize {
                rset.insert(crows.row(roffset + k).owned());
            }
        }
        let mut seen: HashSet<OwnedRow> = HashSet::new();
        // Left elements: kept by membership for intersect/except, always for union.
        for k in lo[row] as usize..lo[row + 1] as usize {
            let owned = crows.row(k).owned();
            let keep_it = match op {
                ListSetOp::Intersect => rset.contains(&owned),
                ListSetOp::Except => !rset.contains(&owned),
                ListSetOp::Union => true,
            };
            if keep_it && seen.insert(owned) {
                keep.push(k as u32);
            }
        }
        // Union also appends the right elements not already taken from the left.
        if matches!(op, ListSetOp::Union) && row < r.len() && r.is_valid(row) {
            for k in ro[row] as usize..ro[row + 1] as usize {
                let idx = roffset + k;
                if seen.insert(crows.row(idx).owned()) {
                    keep.push(idx as u32);
                }
            }
        }
        offsets.push(keep.len() as i32);
        valid.push(true);
    }

    let values = take(combined.as_ref(), &UInt32Array::from(keep), None)?;
    let field = Arc::new(Field::new_list_field(combined.data_type().clone(), true));
    Ok(Arc::new(ListArray::new(
        field,
        OffsetBuffer::new(offsets.into()),
        values,
        Some(NullBuffer::from(valid)),
    )))
}

/// Cast two list children to a common numeric type when they differ, so a set op
/// over lists of different numeric widths compares and combines instead of erroring
/// in `concat`. Same-typed children are returned unchanged; a non-numeric mismatch is
/// left as-is (the later `concat` surfaces the clean type error, matching prior
/// behavior for genuinely incompatible element types).
fn coerce_children(lc: &ArrayRef, rc: &ArrayRef) -> Result<(ArrayRef, ArrayRef), ExprError> {
    if lc.data_type() == rc.data_type()
        || !(lc.data_type().is_numeric() && rc.data_type().is_numeric())
    {
        return Ok((lc.clone(), rc.clone()));
    }
    let common = crate::eval::list::compare_type(lc.data_type(), rc.data_type());
    Ok((cast(lc, &common)?, cast(rc, &common)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{AsArray, Int64Builder, ListBuilder};
    use arrow::datatypes::{Float64Type, Int64Type};

    fn list(rows: &[Option<&[i64]>]) -> ArrayRef {
        let mut b = ListBuilder::new(Int64Builder::new());
        for row in rows {
            match row {
                Some(vs) => {
                    for v in *vs {
                        b.values().append_value(*v);
                    }
                    b.append(true);
                }
                None => b.append(false),
            }
        }
        Arc::new(b.finish())
    }

    fn row(out: &ArrayRef, i: usize) -> Vec<i64> {
        let l = out.as_list::<i32>();
        let v = l.value(i);
        v.as_primitive::<Int64Type>().values().to_vec()
    }

    #[test]
    fn intersect_and_except_dedupe_and_order() {
        let a = list(&[Some(&[1, 2, 2, 3]), Some(&[5, 6]), None]);
        let b = list(&[Some(&[2, 3, 4]), Some(&[7]), Some(&[1])]);
        let inter = eval_list_set(ListSetOp::Intersect, &a, &b).unwrap();
        let exc = eval_list_set(ListSetOp::Except, &a, &b).unwrap();
        let uni = eval_list_set(ListSetOp::Union, &a, &b).unwrap();
        assert_eq!(row(&inter, 0), vec![2, 3]); // deduped, left order
        assert_eq!(row(&exc, 0), vec![1]);
        assert_eq!(row(&exc, 1), vec![5, 6]);
        assert_eq!(row(&uni, 0), vec![1, 2, 3, 4]); // left distinct ++ right-only
        assert_eq!(row(&uni, 1), vec![5, 6, 7]);
        assert!(inter.as_list::<i32>().is_null(2)); // null left → null
    }

    #[test]
    fn intersect_folds_negative_zero_into_zero() {
        use arrow::array::Float64Builder;
        fn flist(rows: &[&[f64]]) -> ArrayRef {
            let mut b = ListBuilder::new(Float64Builder::new());
            for r in rows {
                for v in *r {
                    b.values().append_value(*v);
                }
                b.append(true);
            }
            Arc::new(b.finish())
        }
        // `-0.0` and `0.0` are equal under `=`/`GROUP BY`, so the intersection is non-empty
        // (DuckDB `list_intersect([0.0], [-0.0])` == `[0.0]`). Before canonicalization the
        // `RowConverter` encoded them differently and the result was wrongly `[]`.
        let a = flist(&[&[0.0]]);
        let b = flist(&[&[-0.0]]);
        let out = eval_list_set(ListSetOp::Intersect, &a, &b).unwrap();
        let l = out.as_list::<i32>();
        let v = l.value(0);
        let f = v.as_primitive::<Float64Type>();
        assert_eq!(f.len(), 1);
        assert_eq!(f.value(0), 0.0);
    }

    #[test]
    fn set_op_coerces_mismatched_numeric_children() {
        use arrow::array::Float64Builder;
        fn flist(rows: &[&[f64]]) -> ArrayRef {
            let mut b = ListBuilder::new(Float64Builder::new());
            for r in rows {
                for v in *r {
                    b.values().append_value(*v);
                }
                b.append(true);
            }
            Arc::new(b.finish())
        }
        // `List<Int64>` ∩ `List<Float64>` used to error inside `concat` ("cannot
        // concatenate arrays of different data types"). It must instead promote to the
        // wider type and compare — DuckDB `list_intersect([1,2,3],[2.0,3.0])` == `[2.0, 3.0]`.
        let ints = list(&[Some(&[1, 2, 3])]);
        let floats = flist(&[&[2.0, 3.0, 4.0]]);
        let out = eval_list_set(ListSetOp::Intersect, &ints, &floats).unwrap();
        let l = out.as_list::<i32>();
        let v = l.value(0);
        let f = v.as_primitive::<Float64Type>();
        assert_eq!(f.values().to_vec(), vec![2.0, 3.0]);

        // Union likewise: left (promoted) distinct ++ right-only.
        let uni = eval_list_set(ListSetOp::Union, &ints, &floats).unwrap();
        let ul = uni.as_list::<i32>();
        let uv = ul.value(0);
        let uf = uv.as_primitive::<Float64Type>();
        assert_eq!(uf.values().to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }
}
