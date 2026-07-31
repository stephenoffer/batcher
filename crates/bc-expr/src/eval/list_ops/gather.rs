//! `list.gather` — reorder or select from a list by a second list of indices.
//!
//! This is the operation that makes `arg_sort` useful. `arg_sort` hands back the positions
//! that put a row's scores in order, and without a way to apply them the caller has indices
//! and no way to spend them — the reranking has to leave the engine and happen in Python,
//! one row at a time, which is exactly the hot-path work the control plane must not do.
//!
//! With `gather`, ranking a candidate set is `scores.arg_sort().reverse().head(k)` to pick
//! the positions and `candidates.gather(those)` to take them. The same pair reorders an
//! embedding by a learned permutation, selects the top logits' token ids, or applies one
//! column's ordering to another.
//!
//! An index outside the row's bounds yields a null element rather than an error: a `head(k)`
//! on a row shorter than `k` is a normal thing to write, and erroring there would make the
//! common case the caller's problem. A negative index counts from the end, as `list.get` does.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, ListArray, UInt32Builder};
use arrow::buffer::{NullBuffer, OffsetBuffer};
use arrow::compute::{cast, take};
use arrow::datatypes::{DataType, Field, Int64Type};

use crate::ExprError;

/// Take each row's elements at the positions its `indices` row names.
pub(crate) fn eval_list_gather(
    values: &ListArray,
    indices: &ListArray,
) -> Result<ArrayRef, ExprError> {
    let child = values.values();
    // Indices arrive as whatever integer width the plan produced; one cast up front keeps the
    // inner loop monomorphic.
    let idx = cast(indices.values(), &DataType::Int64)?;
    let idx = idx.as_primitive::<Int64Type>();

    let (vo, io) = (values.value_offsets(), indices.value_offsets());
    // Positions into `child`, with a null where the index was out of range or itself null.
    let mut picks = UInt32Builder::with_capacity(idx.len());
    let mut offsets: Vec<i32> = Vec::with_capacity(values.len() + 1);
    let mut validity: Vec<bool> = Vec::with_capacity(values.len());
    offsets.push(0);
    let mut written = 0i32;
    for i in 0..values.len() {
        if values.is_null(i) || indices.is_null(i) {
            validity.push(false);
            offsets.push(written);
            continue;
        }
        let (vs, ve) = (vo[i] as usize, vo[i + 1] as usize);
        let row_len = (ve - vs) as i64;
        for k in io[i] as usize..io[i + 1] as usize {
            if idx.is_null(k) {
                picks.append_null();
            } else {
                let raw = idx.value(k);
                let position = if raw < 0 { row_len + raw } else { raw };
                if position < 0 || position >= row_len {
                    picks.append_null();
                } else {
                    picks.append_value(vs as u32 + position as u32);
                }
            }
            written += 1;
        }
        validity.push(true);
        offsets.push(written);
    }

    let taken = take(child.as_ref(), &picks.finish(), None)?;
    let field = Field::new_list_field(taken.data_type().clone(), true);
    Ok(Arc::new(ListArray::try_new(
        Arc::new(field),
        OffsetBuffer::new(offsets.into()),
        taken,
        Some(NullBuffer::from(validity)),
    )?))
}

#[cfg(test)]
mod tests {
    use arrow::array::{Int64Builder, ListBuilder, StringBuilder};

    use super::*;

    fn strings(rows: &[Option<Vec<&str>>]) -> ListArray {
        let mut b = ListBuilder::new(StringBuilder::new());
        for row in rows {
            match row {
                Some(vs) => {
                    for v in vs {
                        b.values().append_value(v);
                    }
                    b.append(true);
                }
                None => b.append(false),
            }
        }
        b.finish()
    }

    fn ints(rows: &[Option<Vec<Option<i64>>>]) -> ListArray {
        let mut b = ListBuilder::new(Int64Builder::new());
        for row in rows {
            match row {
                Some(vs) => {
                    for v in vs {
                        match v {
                            Some(x) => b.values().append_value(*x),
                            None => b.values().append_null(),
                        }
                    }
                    b.append(true);
                }
                None => b.append(false),
            }
        }
        b.finish()
    }

    fn row_strings(out: &ArrayRef, i: usize) -> Vec<Option<String>> {
        let list = out.as_list::<i32>();
        let row = list.value(i);
        let s = row.as_string::<i32>();
        (0..s.len())
            .map(|k| (!s.is_null(k)).then(|| s.value(k).to_string()))
            .collect()
    }

    #[test]
    fn gather_reorders_by_the_given_positions() {
        let values = strings(&[Some(vec!["a", "b", "c"])]);
        let idx = ints(&[Some(vec![Some(2), Some(0)])]);
        let out = eval_list_gather(&values, &idx).unwrap();
        assert_eq!(
            row_strings(&out, 0),
            vec![Some("c".into()), Some("a".into())]
        );
    }

    /// The pairing that motivates the op: positions from one column, values from another.
    #[test]
    fn gather_applies_one_rows_ordering_to_another() {
        let candidates = strings(&[Some(vec!["low", "high", "mid"])]);
        // `arg_sort` of [0.1, 0.9, 0.5] reversed is [1, 2, 0].
        let ranked = ints(&[Some(vec![Some(1), Some(2), Some(0)])]);
        let out = eval_list_gather(&candidates, &ranked).unwrap();
        assert_eq!(
            row_strings(&out, 0),
            vec![Some("high".into()), Some("mid".into()), Some("low".into())]
        );
    }

    #[test]
    fn a_negative_index_counts_from_the_end() {
        let values = strings(&[Some(vec!["a", "b", "c"])]);
        let idx = ints(&[Some(vec![Some(-1), Some(-3)])]);
        let out = eval_list_gather(&values, &idx).unwrap();
        assert_eq!(
            row_strings(&out, 0),
            vec![Some("c".into()), Some("a".into())]
        );
    }

    /// A `head(k)` wider than the row is ordinary, so it must not error.
    #[test]
    fn an_out_of_range_index_yields_null_rather_than_an_error() {
        let values = strings(&[Some(vec!["a", "b"])]);
        let idx = ints(&[Some(vec![Some(0), Some(5), Some(-9)])]);
        let out = eval_list_gather(&values, &idx).unwrap();
        assert_eq!(row_strings(&out, 0), vec![Some("a".into()), None, None]);
    }

    #[test]
    fn a_null_index_yields_a_null_element() {
        let values = strings(&[Some(vec!["a", "b"])]);
        let idx = ints(&[Some(vec![Some(1), None])]);
        let out = eval_list_gather(&values, &idx).unwrap();
        assert_eq!(row_strings(&out, 0), vec![Some("b".into()), None]);
    }

    #[test]
    fn a_null_row_on_either_side_yields_a_null_row() {
        let values = strings(&[None, Some(vec!["a"])]);
        let idx = ints(&[Some(vec![Some(0)]), None]);
        let out = eval_list_gather(&values, &idx).unwrap();
        let list = out.as_list::<i32>();
        assert!(list.is_null(0) && list.is_null(1));
    }

    #[test]
    fn an_empty_index_row_yields_an_empty_row() {
        let values = strings(&[Some(vec!["a", "b"])]);
        let idx = ints(&[Some(vec![])]);
        let out = eval_list_gather(&values, &idx).unwrap();
        let list = out.as_list::<i32>();
        assert!(!list.is_null(0));
        assert_eq!(list.value(0).len(), 0);
    }

    /// Rows are independent: row 1's indices must not read row 0's values.
    #[test]
    fn rows_index_only_into_their_own_values() {
        let values = strings(&[Some(vec!["a0", "a1"]), Some(vec!["b0", "b1"])]);
        let idx = ints(&[Some(vec![Some(0)]), Some(vec![Some(0)])]);
        let out = eval_list_gather(&values, &idx).unwrap();
        assert_eq!(row_strings(&out, 0), vec![Some("a0".into())]);
        assert_eq!(row_strings(&out, 1), vec![Some("b0".into())]);
    }

    #[test]
    fn an_index_may_repeat_a_position() {
        let values = strings(&[Some(vec!["a", "b"])]);
        let idx = ints(&[Some(vec![Some(1), Some(1), Some(1)])]);
        let out = eval_list_gather(&values, &idx).unwrap();
        assert_eq!(
            row_strings(&out, 0),
            vec![Some("b".into()), Some("b".into()), Some("b".into())]
        );
    }
}
