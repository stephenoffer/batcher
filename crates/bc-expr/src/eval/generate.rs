//! Series generation for `Expr::Sequence` (`sequence`/`range`).
//!
//! `sequence(start, stop, step)` builds a `List<Int64>` per row — the integer series
//! from `start` to `stop` **inclusive**, stepping by `step` (Spark `sequence`, DuckDB
//! `generate_series`). A null in any argument yields a null list; `step == 0` errors.
//! This is a leaf generator (its inputs are ordinary expressions), so the JIT falls
//! back to this interpreter path.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Int64Builder, ListBuilder};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Int64Type};

use crate::ExprError;

/// Build the per-row integer series `[start, start±…, stop]` (inclusive) as a
/// `List<Int64>` column. Inputs are cast to Int64; a null argument → null row.
pub(crate) fn eval_sequence(
    start: &ArrayRef,
    stop: &ArrayRef,
    step: &ArrayRef,
) -> Result<ArrayRef, ExprError> {
    let n = start.len();
    let start = cast(start, &DataType::Int64)?;
    let stop = cast(stop, &DataType::Int64)?;
    let step = cast(step, &DataType::Int64)?;
    let start = start.as_primitive::<Int64Type>();
    let stop = stop.as_primitive::<Int64Type>();
    let step = step.as_primitive::<Int64Type>();

    let mut b = ListBuilder::new(Int64Builder::new());
    for i in 0..n {
        if start.is_null(i) || stop.is_null(i) || step.is_null(i) {
            b.append(false);
            continue;
        }
        let (s, e, d) = (start.value(i), stop.value(i), step.value(i));
        if d == 0 {
            return Err(ExprError::DivideByZero);
        }
        // Walk from `s` toward `e` by `d`; the direction of `d` must match s→e or the
        // series is empty (matches DuckDB `generate_series`).
        let mut v = s;
        while (d > 0 && v <= e) || (d < 0 && v >= e) {
            b.values().append_value(v);
            match v.checked_add(d) {
                Some(next) => v = next,
                None => break, // overflow: stop the series rather than panic
            }
        }
        b.append(true);
    }
    Ok(Arc::new(b.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, ListArray};

    fn seq(start: &[Option<i64>], stop: &[i64], step: &[i64]) -> ArrayRef {
        let s: ArrayRef = Arc::new(Int64Array::from(start.to_vec()));
        let e: ArrayRef = Arc::new(Int64Array::from(stop.to_vec()));
        let d: ArrayRef = Arc::new(Int64Array::from(step.to_vec()));
        eval_sequence(&s, &e, &d).unwrap()
    }

    #[test]
    fn inclusive_forward_backward_and_empty() {
        let out = seq(
            &[Some(1), Some(10), Some(5), None],
            &[5, 2, 1, 9],
            &[2, -3, 2, 1],
        );
        let list = out.as_any().downcast_ref::<ListArray>().unwrap();
        let row = |i: usize| {
            let v = list.value(i);
            v.as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec()
        };
        assert_eq!(row(0), vec![1, 3, 5]); // forward inclusive
        assert_eq!(row(1), vec![10, 7, 4]); // backward
        assert!(list.value(2).is_empty()); // step wrong direction → empty
        assert!(list.is_null(3)); // null arg → null row
    }

    #[test]
    fn zero_step_errors() {
        let s: ArrayRef = Arc::new(Int64Array::from(vec![1]));
        let e: ArrayRef = Arc::new(Int64Array::from(vec![5]));
        let d: ArrayRef = Arc::new(Int64Array::from(vec![0]));
        assert!(eval_sequence(&s, &e, &d).is_err());
    }
}
