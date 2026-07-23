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
use arrow::error::ArrowError;

use crate::ExprError;

/// The most elements a whole `sequence` column may hold. The `List<Int64>` output is
/// built with 32-bit offsets, so the child array cannot exceed `i32::MAX` values — and
/// well before that, an unbounded range (`sequence(1, 10_000_000_000)`) would exhaust
/// memory. DuckDB likewise refuses a list past `2^32` rather than attempting it; we cap
/// at the offset limit and error, so the failure is a clean message rather than an OOM
/// kill or a silently overflowed offset buffer.
const MAX_SEQUENCE_ELEMENTS: i128 = i32::MAX as i128;

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
    let mut total: i128 = 0;
    for i in 0..n {
        if start.is_null(i) || stop.is_null(i) || step.is_null(i) {
            b.append(false);
            continue;
        }
        let (s, e, d) = (start.value(i), stop.value(i), step.value(i));
        if d == 0 {
            return Err(ExprError::DivideByZero);
        }
        // Bound the allocation up front (in `i128` to dodge `e - s` overflow). The series
        // is empty when `d` points away from `s → e`; otherwise it has
        // `(e - s) / d + 1` elements. Refusing an over-large series here turns an OOM /
        // silently-overflowed 32-bit offset buffer into a clean error (see the cap docs).
        let count: i128 = if (d > 0 && s > e) || (d < 0 && s < e) {
            0
        } else {
            (i128::from(e) - i128::from(s)) / i128::from(d) + 1
        };
        total += count;
        if total > MAX_SEQUENCE_ELEMENTS {
            return Err(ArrowError::ComputeError(format!(
                "sequence: result of {total} elements exceeds the supported maximum of \
                 {MAX_SEQUENCE_ELEMENTS}"
            ))
            .into());
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
    fn an_over_large_range_errors_instead_of_exhausting_memory() {
        // ~10^10 elements: without the cap this allocates ~80 GB and overflows the 32-bit
        // list-offset buffer. It must return an error, cheaply, having built nothing.
        let s: ArrayRef = Arc::new(Int64Array::from(vec![1]));
        let e: ArrayRef = Arc::new(Int64Array::from(vec![10_000_000_000]));
        let d: ArrayRef = Arc::new(Int64Array::from(vec![1]));
        assert!(eval_sequence(&s, &e, &d).is_err());
    }

    #[test]
    fn the_count_bound_dodges_span_overflow() {
        // `stop - start` overflows `i64` here (MAX - MIN); the `i128` count computation
        // must still classify it as over-large and error, not panic on the subtraction.
        let s: ArrayRef = Arc::new(Int64Array::from(vec![i64::MIN]));
        let e: ArrayRef = Arc::new(Int64Array::from(vec![i64::MAX]));
        let d: ArrayRef = Arc::new(Int64Array::from(vec![1]));
        assert!(eval_sequence(&s, &e, &d).is_err());
    }

    #[test]
    fn zero_step_errors() {
        let s: ArrayRef = Arc::new(Int64Array::from(vec![1]));
        let e: ArrayRef = Arc::new(Int64Array::from(vec![5]));
        let d: ArrayRef = Arc::new(Int64Array::from(vec![0]));
        assert!(eval_sequence(&s, &e, &d).is_err());
    }
}
