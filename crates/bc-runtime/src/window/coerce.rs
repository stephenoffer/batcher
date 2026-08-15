//! What a window value column has to be reshaped into before the typed kernels read it.
//!
//! The whole-partition (`partition_agg`) and running (`window::mod`) aggregate kernels are
//! written against `Int64`/`Float64`/`Utf8`/`Boolean` and rejected everything else outright:
//!
//! ```text
//! window function min is not supported for column type Date32
//! window function sum is not supported for column type Int32
//! ```
//!
//! The *same* aggregate under a `GROUP BY` answers all of those, because `agg/` widens its
//! inputs first. So `MIN(order_date) OVER (PARTITION BY customer)` — an ordinary analytics
//! query — failed while `SELECT customer, MIN(order_date) ... GROUP BY customer` succeeded,
//! and the difference was invisible from the plan. This module is that missing widening,
//! shared by both kernels so they cannot drift apart on which types they admit.
//!
//! Two mechanisms, picked by what the function does to its input's type:
//!
//! * **Widening** for the reducing functions (`SUM`/`AVG`/`STDDEV`/…): narrow integers go to
//!   `Int64` and narrow floats to `Float64`, exactly as `bc_py::normalize_to` does at the FFI
//!   boundary and `agg::widen_mean_inputs` does under a `GROUP BY`.
//! * **Selection by index** for `MIN`/`MAX`, which are type-*preserving*: rather than widen a
//!   `Date32`/`Timestamp`/`Decimal` into something the numeric kernels read and then try to
//!   put the type back, find the winning row per group and `take` it. That returns the input
//!   type exactly, for every type Arrow can order, and needs no per-type kernel.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, UInt32Array};
use arrow::datatypes::DataType;
use arrow::row::{RowConverter, SortField};

use crate::error::RuntimeError;
use crate::window::WindowFn;

/// The engine's canonical width for a column the typed window kernels cannot read, or `None`
/// when the column is already one they can (the common case, which allocates nothing).
///
/// `Decimal` and `UInt64` promote to `Float64` rather than staying exact. That is the same
/// trade `bc_expr::eval::eval_math` makes for the math family and for the same reason:
/// the alternative on offer is not an exact answer, it is refusing to answer at all. It costs
/// exactness above 2^53 and returns DOUBLE where DuckDB returns DECIMAL. `MIN`/`MAX` do not
/// take this path — they keep the input type exactly, via [`select_extreme`].
pub(crate) fn widen_target(dt: &DataType) -> Option<DataType> {
    use DataType::*;
    match dt {
        Int8 | Int16 | Int32 | UInt8 | UInt16 | UInt32 => Some(Int64),
        Float16 | Float32 | UInt64 | Decimal128(_, _) | Decimal256(_, _) => Some(Float64),
        // An all-null column carries Arrow's `Null` type, which every typed kernel below
        // rejects. `agg::coerce_null_call_inputs` already widens it to an all-null `Int64`
        // under a `GROUP BY`, for exactly this reason and with exactly this target; without
        // the same step here `SUM(x) OVER (…)` errored on a column `SUM(x) … GROUP BY`
        // answered with NULL. The result type is immaterial because every value is null.
        Null => Some(Int64),
        _ => None,
    }
}

/// Widen `values` to the canonical width for a reducing window aggregate, or hand back
/// `None` when no widening is needed.
pub(crate) fn widen_values(values: &ArrayRef) -> Result<Option<ArrayRef>, RuntimeError> {
    let Some(target) = widen_target(values.data_type()) else {
        return Ok(None);
    };
    Ok(Some(arrow::compute::cast(values, &target)?))
}

/// True for the two functions [`select_extreme`] answers.
pub(crate) fn is_extreme(func: WindowFn) -> bool {
    matches!(func, WindowFn::Min | WindowFn::Max)
}

/// Per-group `MIN`/`MAX` of any orderable column, as a gather index per group.
///
/// `best[g]` is the row index holding group `g`'s extreme, or `None` when the group is
/// empty or wholly null — SQL's `MIN`/`MAX` ignore nulls and are null only when every value
/// is. Comparison goes through Arrow's `RowConverter`, the same total order the window's own
/// ORDER BY encoding uses, so "smallest" here means what it means everywhere else in the
/// engine — including that `-0.0` and `0.0` are one value and every NaN is one value, which
/// `keys::canonicalize_float_order_keys` has already folded by the time a float column
/// reaches a window.
///
/// `group_ids[i]` names row `i`'s group; rows are visited once, in order.
fn extreme_indices(
    func: WindowFn,
    group_ids: impl Iterator<Item = (usize, u32)>,
    num_groups: usize,
    values: &ArrayRef,
) -> Result<Vec<Option<usize>>, RuntimeError> {
    let conv = RowConverter::new(vec![SortField::new(values.data_type().clone())])?;
    let rows = conv.convert_columns(std::slice::from_ref(values))?;
    let want_min = func == WindowFn::Min;
    let mut best: Vec<Option<usize>> = vec![None; num_groups];
    for (i, g) in group_ids {
        if values.is_null(i) {
            continue;
        }
        let slot = &mut best[g as usize];
        match *slot {
            None => *slot = Some(i),
            Some(b) => {
                let better = if want_min {
                    rows.row(i) < rows.row(b)
                } else {
                    rows.row(i) > rows.row(b)
                };
                if better {
                    *slot = Some(i);
                }
            }
        }
    }
    Ok(best)
}

/// Whole-partition `MIN`/`MAX` broadcast back to every row, preserving the input type.
pub(crate) fn select_extreme(
    func: WindowFn,
    group_ids: &[u32],
    num_groups: usize,
    values: &ArrayRef,
) -> Result<ArrayRef, RuntimeError> {
    let best = extreme_indices(
        func,
        group_ids.iter().enumerate().map(|(i, &g)| (i, g)),
        num_groups,
        values,
    )?;
    let idx: UInt32Array = group_ids
        .iter()
        .map(|&g| best[g as usize].map(|i| i as u32))
        .collect();
    Ok(arrow::compute::take(values, &idx, None)?)
}

/// Running (`ORDER BY`-framed) `MIN`/`MAX` over each ordered partition, preserving the input
/// type. `ordered[p]` is partition `p`'s row indices in order; `peer_end(part, pos)` reports
/// whether `pos` closes a peer group, so tied order keys share one value the way every other
/// running window function here treats them.
pub(crate) fn running_extreme(
    func: WindowFn,
    ordered: &[Vec<usize>],
    values: &ArrayRef,
    num_rows: usize,
    peer_end: impl Fn(&[usize], usize) -> bool,
) -> Result<ArrayRef, RuntimeError> {
    let conv = RowConverter::new(vec![SortField::new(values.data_type().clone())])?;
    let rows = conv.convert_columns(std::slice::from_ref(values))?;
    let want_min = func == WindowFn::Min;
    let mut pick: Vec<Option<u32>> = vec![None; num_rows];
    for part in ordered {
        let mut best: Option<usize> = None;
        let mut group_start = 0usize;
        for pos in 0..part.len() {
            let i = part[pos];
            if values.is_valid(i) {
                let better = match best {
                    None => true,
                    Some(b) => {
                        if want_min {
                            rows.row(i) < rows.row(b)
                        } else {
                            rows.row(i) > rows.row(b)
                        }
                    }
                };
                if better {
                    best = Some(i);
                }
            }
            // Peers share the prefix that ends at the last of them, so the value is only
            // published once the peer group closes.
            if peer_end(part, pos) {
                for j in group_start..=pos {
                    pick[part[j]] = best.map(|b| b as u32);
                }
                group_start = pos + 1;
            }
        }
    }
    let idx: UInt32Array = pick.into_iter().collect();
    Ok(arrow::compute::take(values, &idx, None)?)
}

/// A widened copy of `values` for a reducing aggregate, or `values` itself when it already
/// has a width the kernels read. Kept as an owned `ArrayRef` so callers can hold one binding.
pub(crate) fn widened_or_original(values: &ArrayRef) -> Result<ArrayRef, RuntimeError> {
    Ok(widen_values(values)?.unwrap_or_else(|| Arc::clone(values)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An all-null column is `Null`-typed, and the reducing kernels only read the canonical
    /// widths. `agg` widens it to `Int64` under a `GROUP BY`; without the same rule here
    /// `SUM(x) OVER (…)` raised on a column `SUM(x) … GROUP BY` answered with NULL.
    #[test]
    fn a_null_typed_column_widens_to_int64() {
        assert_eq!(widen_target(&DataType::Null), Some(DataType::Int64));
    }

    /// `MIN`/`MAX` never widen — they select the winning row — so the types they preserve
    /// must keep reporting "no widening needed".
    #[test]
    fn the_canonical_widths_need_no_widening() {
        for dt in [
            DataType::Int64,
            DataType::Float64,
            DataType::Utf8,
            DataType::Boolean,
        ] {
            assert_eq!(widen_target(&dt), None, "{dt} should need no widening");
        }
    }
    use arrow::array::{Date32Array, Decimal128Array, Int32Array, TimestampMicrosecondArray};

    fn ids(v: &[u32]) -> Vec<u32> {
        v.to_vec()
    }

    /// `MIN`/`MAX` over a `Date32` partition returns `Date32`, not an error and not an int.
    #[test]
    fn extreme_over_dates_keeps_the_date_type() {
        let values: ArrayRef = Arc::new(Date32Array::from(vec![
            Some(100),
            Some(50),
            None,
            Some(400),
            Some(300),
        ]));
        let g = ids(&[0, 0, 0, 1, 1]);
        let out = select_extreme(WindowFn::Min, &g, 2, &values).unwrap();
        assert_eq!(out.data_type(), &DataType::Date32);
        let out = out.as_any().downcast_ref::<Date32Array>().unwrap();
        assert_eq!(out.value(0), 50);
        assert_eq!(out.value(2), 50);
        assert_eq!(out.value(3), 300);
    }

    /// A group whose every value is null is null, not the group's first row.
    #[test]
    fn an_all_null_group_is_null() {
        let values: ArrayRef = Arc::new(Date32Array::from(vec![None, None, Some(7)]));
        let g = ids(&[0, 0, 1]);
        let out = select_extreme(WindowFn::Max, &g, 2, &values).unwrap();
        assert!(out.is_null(0) && out.is_null(1));
        assert!(out.is_valid(2));
    }

    /// `MAX` over a timestamp keeps the unit and the timezone.
    #[test]
    fn extreme_over_timestamps_keeps_unit_and_zone() {
        let arr = TimestampMicrosecondArray::from(vec![Some(5), Some(9)]).with_timezone("UTC");
        let values: ArrayRef = Arc::new(arr);
        let want = values.data_type().clone();
        let out = select_extreme(WindowFn::Max, &ids(&[0, 0]), 1, &values).unwrap();
        assert_eq!(out.data_type(), &want);
    }

    /// `MIN` over a decimal stays exact — it never routes through the `Float64` widening.
    #[test]
    fn extreme_over_decimals_stays_decimal() {
        let values: ArrayRef = Arc::new(
            Decimal128Array::from(vec![Some(150), Some(120)])
                .with_precision_and_scale(10, 2)
                .unwrap(),
        );
        let out = select_extreme(WindowFn::Min, &ids(&[0, 0]), 1, &values).unwrap();
        assert_eq!(out.data_type(), &DataType::Decimal128(10, 2));
        let out = out.as_any().downcast_ref::<Decimal128Array>().unwrap();
        assert_eq!(out.value(0), 120);
    }

    /// A narrow integer widens to the engine's canonical `Int64` for a reducing aggregate.
    #[test]
    fn narrow_ints_widen_for_reducing_aggregates() {
        let values: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let w = widen_values(&values).unwrap().expect("Int32 widens");
        assert_eq!(w.data_type(), &DataType::Int64);
        // A column already at canonical width is left alone, with no allocation.
        let wide: ArrayRef = Arc::new(arrow::array::Int64Array::from(vec![1, 2, 3]));
        assert!(widen_values(&wide).unwrap().is_none());
    }

    /// The running form publishes one value per peer group and keeps the input type.
    #[test]
    fn running_extreme_keeps_type_and_respects_peers() {
        let values: ArrayRef = Arc::new(Date32Array::from(vec![
            Some(30),
            Some(10),
            Some(20),
            Some(5),
        ]));
        let ordered = vec![vec![0usize, 1, 2, 3]];
        // Every row is its own peer group.
        let out = running_extreme(WindowFn::Min, &ordered, &values, 4, |_, _| true).unwrap();
        assert_eq!(out.data_type(), &DataType::Date32);
        let out = out.as_any().downcast_ref::<Date32Array>().unwrap();
        assert_eq!(
            (0..4).map(|i| out.value(i)).collect::<Vec<_>>(),
            vec![30, 10, 10, 5]
        );
    }
}
