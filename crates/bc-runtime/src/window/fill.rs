//! `forward_fill` / `backward_fill` — carry the nearest non-null value along an
//! ordered partition.
//!
//! The time-series gap filler. SQL spells it `last_value(x IGNORE NULLS) OVER (…
//! ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)`; because a fill has exactly one
//! sensible frame, the frame is implied here and there is none to pass.
//!
//! Like the positional value functions in [`crate::window`], a fill *selects a row*
//! rather than computing a value — so it is type-generic: one pass per partition builds
//! a per-row source-index map and `take` does the rest. Rows with no source (before the
//! first non-null, or after the last for a backward fill) stay null.

use arrow::array::{Array, ArrayRef, UInt32Array};
use arrow::compute::take;

use crate::error::RuntimeError;
use crate::window::WindowFn;

/// Fill each row from the nearest non-null row at or before it (`ForwardFill`) or at or
/// after it (`BackwardFill`) within its ordered partition.
///
/// Backward fill is forward fill over the reversed order, which is why the two share
/// everything but the iteration direction.
pub(crate) fn fill_window(
    func: WindowFn,
    ordered: &[Vec<usize>],
    values: &ArrayRef,
    num_rows: usize,
) -> Result<ArrayRef, RuntimeError> {
    let mut src: Vec<Option<u32>> = vec![None; num_rows];
    for part in ordered {
        let mut carried: Option<u32> = None;
        if func == WindowFn::ForwardFill {
            for &row in part.iter() {
                carry(values, row, &mut carried, &mut src);
            }
        } else {
            for &row in part.iter().rev() {
                carry(values, row, &mut carried, &mut src);
            }
        }
    }
    Ok(take(values.as_ref(), &UInt32Array::from(src), None)?)
}

/// Update the carried source row at `row`, then record it. A non-null row carries
/// itself, so a fill is the identity wherever the column is already non-null.
fn carry(values: &ArrayRef, row: usize, carried: &mut Option<u32>, src: &mut [Option<u32>]) {
    if values.is_valid(row) {
        *carried = Some(row as u32);
    }
    src[row] = *carried;
}
