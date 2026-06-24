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
use arrow::compute::{concat, take};
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
    let lc = l.values();
    let rc = r.values();

    // Concatenate the two children into one array so output elements can be drawn from
    // either side (union needs both); a left element keeps its index, a right element
    // `k` maps to `lc.len() + k`. One row converter over the combined child makes every
    // element comparable regardless of which list it came from.
    let combined = concat(&[lc.as_ref(), rc.as_ref()])?;
    let roffset = lc.len();
    let converter = RowConverter::new(vec![SortField::new(combined.data_type().clone())])?;
    let crows = converter.convert_columns(std::slice::from_ref(&combined))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, Int64Builder, ListBuilder};

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
        let l = out.as_any().downcast_ref::<ListArray>().unwrap();
        let v = l.value(i);
        v.as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .to_vec()
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
        assert!(inter
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .is_null(2)); // null left → null
    }
}
