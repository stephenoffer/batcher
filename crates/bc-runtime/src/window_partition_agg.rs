//! Whole-partition window aggregates (`SUM`/`AVG`/`MIN`/`MAX`/`COUNT` with no ORDER BY
//! and no frame): one value per partition, broadcast to every row of that partition.
//!
//! Computed via **dense group ids** — reduce each group in one linear pass over the
//! rows, then broadcast its value back by index. This is exactly a group-by aggregate
//! followed by a scatter, and it avoids the per-partition index lists and the scattered
//! gather they force. The order within a partition never affects a whole-partition
//! aggregate, so the result is identical to ordering first.

use std::cell::Cell;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, Int64Array, PrimitiveArray, StringArray,
};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{ArrowPrimitiveType, DataType, Float64Type, Int64Type};
use rayon::prelude::*;

use crate::error::RuntimeError;
use crate::window::{require, WindowFn};

/// Whole-partition aggregate from per-partition row-index lists (the slow window path,
/// reached only when a no-ORDER-BY aggregate is mixed with non-aggregate functions).
/// Flattens the lists to dense group ids, then defers to [`broadcast_partition_aggregate`].
pub(crate) fn partition_aggregate(
    func: WindowFn,
    partitions: &[Vec<usize>],
    values: Option<&ArrayRef>,
    num_rows: usize,
) -> Result<ArrayRef, RuntimeError> {
    let mut group_ids = vec![0u32; num_rows];
    for (g, part) in partitions.iter().enumerate() {
        for &i in part {
            group_ids[i] = g as u32;
        }
    }
    broadcast_partition_aggregate(func, &group_ids, partitions.len(), values)
}

/// Whole-partition aggregate via dense group ids: reduce each group in one linear pass
/// over the rows, then broadcast its value to every member row.
pub(crate) fn broadcast_partition_aggregate(
    func: WindowFn,
    group_ids: &[u32],
    num_groups: usize,
    values: Option<&ArrayRef>,
) -> Result<ArrayRef, RuntimeError> {
    if func == WindowFn::Count {
        let values = require(values, func)?;
        let mut counts = vec![0i64; num_groups];
        for (i, &g) in group_ids.iter().enumerate() {
            if values.is_valid(i) {
                counts[g as usize] += 1;
            }
        }
        let out: Vec<i64> = group_ids.iter().map(|&g| counts[g as usize]).collect();
        return Ok(Arc::new(Int64Array::from(out)));
    }
    let values = require(values, func)?;
    match values.data_type() {
        DataType::Int64 => grouped_i64(func, group_ids, num_groups, values),
        DataType::Float64 => grouped_f64(func, group_ids, num_groups, values),
        DataType::Utf8 if matches!(func, WindowFn::Min | WindowFn::Max) => {
            grouped_str_minmax(func, group_ids, num_groups, values)
        }
        // Boolean MIN/MAX orders `false < true` (min = AND, max = OR), matching the
        // aggregate MIN/MAX (B23) and DuckDB; without it `min(flag) OVER (…)` errored.
        DataType::Boolean if matches!(func, WindowFn::Min | WindowFn::Max) => {
            grouped_bool_minmax(func, group_ids, num_groups, values)
        }
        other => Err(RuntimeError::UnsupportedWindow {
            func: func.name().to_string(),
            dtype: other.to_string(),
        }),
    }
}

/// Rows at or above which the final broadcast gather is worth spreading across cores.
const BROADCAST_PARALLEL_MIN_ROWS: usize = 1 << 17;

/// Reduce `group_ids`/`values` into `acc` with `combine`, specialized ONCE here rather
/// than re-matched per row.
///
/// The reduce is the hot loop (one pass over every row); taking the combiner as a generic
/// closure lets it inline and keeps the per-row work to a load, a compare and a store. The
/// null-free case skips the validity check entirely, which is the common shape.
/// `cnt[g] == 0` marks "group not seen yet", so it doubles as the seen-flag and the
/// `AVG` divisor and no `Option` is needed in the accumulator.
#[inline]
fn reduce_groups<T, F>(
    arr: &arrow::array::PrimitiveArray<T>,
    group_ids: &[u32],
    acc: &mut [T::Native],
    cnt: &mut [i64],
    combine: F,
) where
    T: ArrowPrimitiveType,
    F: Fn(T::Native, T::Native) -> T::Native,
{
    let vals = arr.values();
    if arr.null_count() == 0 {
        for (i, &g) in group_ids.iter().enumerate() {
            let g = g as usize;
            let v = vals[i];
            acc[g] = if cnt[g] == 0 { v } else { combine(acc[g], v) };
            cnt[g] += 1;
        }
    } else {
        for (i, &g) in group_ids.iter().enumerate() {
            if arr.is_valid(i) {
                let g = g as usize;
                let v = vals[i];
                acc[g] = if cnt[g] == 0 { v } else { combine(acc[g], v) };
                cnt[g] += 1;
            }
        }
    }
}

/// Broadcast one value per group back to every row, as an Arrow array.
///
/// Writes the values buffer directly (8 bytes/row). Collecting a `Vec<Option<T>>` instead
/// — as this did — costs 16 bytes/row plus a second full pass to split it into a values
/// buffer and a null buffer; at 6 M rows that intermediate alone was ~96 MB of traffic and
/// dominated the whole operator. A group is null iff it saw no valid value (`cnt == 0`),
/// so the null buffer is built only when some group is empty. The gather is per-row
/// independent, so it fans across cores above [`BROADCAST_PARALLEL_MIN_ROWS`].
fn broadcast<T: ArrowPrimitiveType>(
    group_ids: &[u32],
    per_group: &[T::Native],
    cnt: &[i64],
) -> ArrayRef
where
    T::Native: Send + Sync,
{
    let gather = |g: &u32| per_group[*g as usize];
    let values: Vec<T::Native> =
        if group_ids.len() >= BROADCAST_PARALLEL_MIN_ROWS && rayon::current_num_threads() > 1 {
            group_ids.par_iter().map(gather).collect()
        } else {
            group_ids.iter().map(gather).collect()
        };
    let nulls = cnt
        .contains(&0)
        .then(|| NullBuffer::from_iter(group_ids.iter().map(|&g| cnt[g as usize] > 0)));
    Arc::new(PrimitiveArray::<T>::new(values.into(), nulls))
}

fn grouped_i64(
    func: WindowFn,
    group_ids: &[u32],
    num_groups: usize,
    values: &ArrayRef,
) -> Result<ArrayRef, RuntimeError> {
    let arr = values.as_primitive::<Int64Type>();
    let mut cnt = vec![0i64; num_groups];
    if func == WindowFn::Avg {
        // Accumulate the sum EXACTLY in i128, then convert once to f64 for the division —
        // routing each i64 through f64 as it is added loses precision above 2^53 (a running
        // `sum += v as f64`), so `avg` of large integers drifts by ±1 from DuckDB (which
        // sums in 128-bit before dividing). i128 holds any i64 sum over any row count.
        let mut sum = vec![0i128; num_groups];
        for (i, &g) in group_ids.iter().enumerate() {
            if arr.is_valid(i) {
                sum[g as usize] += arr.value(i) as i128;
                cnt[g as usize] += 1;
            }
        }
        let grp: Vec<f64> = (0..num_groups)
            .map(|g| {
                if cnt[g] > 0 {
                    sum[g] as f64 / cnt[g] as f64
                } else {
                    0.0
                }
            })
            .collect();
        return Ok(broadcast::<Float64Type>(group_ids, &grp, &cnt));
    }
    let mut acc = vec![0i64; num_groups];
    match func {
        // `checked_add` so an integer window SUM that overflows i64 errors (matching the
        // aggregate SUM) instead of panicking in debug / silently wrapping in release.
        // The closure cannot return early, so it records the overflow in a flag and the
        // error is raised after the pass — one branch per row that never mispredicts on
        // data that does not overflow.
        WindowFn::Sum => {
            let overflowed = Cell::new(false);
            reduce_groups(arr, group_ids, &mut acc, &mut cnt, |a, v| {
                match a.checked_add(v) {
                    Some(s) => s,
                    None => {
                        overflowed.set(true);
                        a
                    }
                }
            });
            if overflowed.get() {
                return Err(RuntimeError::SumOverflow);
            }
        }
        WindowFn::Min => reduce_groups(arr, group_ids, &mut acc, &mut cnt, |a, v| a.min(v)),
        WindowFn::Max => reduce_groups(arr, group_ids, &mut acc, &mut cnt, |a, v| a.max(v)),
        // Any other function keeps the first value it saw, as before.
        _ => reduce_groups(arr, group_ids, &mut acc, &mut cnt, |a, _| a),
    }
    Ok(broadcast::<Int64Type>(group_ids, &acc, &cnt))
}

fn grouped_f64(
    func: WindowFn,
    group_ids: &[u32],
    num_groups: usize,
    values: &ArrayRef,
) -> Result<ArrayRef, RuntimeError> {
    let arr = values.as_primitive::<Float64Type>();
    let mut acc = vec![0f64; num_groups];
    let mut cnt = vec![0i64; num_groups];
    match func {
        WindowFn::Sum | WindowFn::Avg => {
            reduce_groups(arr, group_ids, &mut acc, &mut cnt, |a, v| a + v)
        }
        // Total-order min/max so NaN is greatest (matching aggregate MIN/MAX, ORDER BY,
        // and DuckDB); `f64::min`/`f64::max` silently drop NaN so `MAX OVER ()` returned
        // a finite value where the aggregate MAX returns NaN.
        WindowFn::Min => reduce_groups(arr, group_ids, &mut acc, &mut cnt, |a, v| {
            if crate::keys::float_total_cmp(v, a).is_lt() {
                v
            } else {
                a
            }
        }),
        WindowFn::Max => reduce_groups(arr, group_ids, &mut acc, &mut cnt, |a, v| {
            if crate::keys::float_total_cmp(v, a).is_gt() {
                v
            } else {
                a
            }
        }),
        _ => reduce_groups(arr, group_ids, &mut acc, &mut cnt, |a, _| a),
    }
    if func == WindowFn::Avg {
        for g in 0..num_groups {
            if cnt[g] > 0 {
                acc[g] /= cnt[g] as f64;
            }
        }
    }
    Ok(broadcast::<Float64Type>(group_ids, &acc, &cnt))
}

fn grouped_str_minmax(
    func: WindowFn,
    group_ids: &[u32],
    num_groups: usize,
    values: &ArrayRef,
) -> Result<ArrayRef, RuntimeError> {
    let arr = values.as_any().downcast_ref::<StringArray>().expect("utf8");
    let is_min = func == WindowFn::Min;
    let mut acc: Vec<Option<&str>> = vec![None; num_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        if arr.is_valid(i) {
            let v = arr.value(i);
            let slot = &mut acc[g as usize];
            *slot = Some(match *slot {
                None => v,
                Some(a) if (is_min && v < a) || (!is_min && v > a) => v,
                Some(a) => a,
            });
        }
    }
    let grp: Vec<Option<String>> = acc.iter().map(|o| o.map(|s| s.to_string())).collect();
    let out: Vec<Option<String>> = group_ids.iter().map(|&g| grp[g as usize].clone()).collect();
    Ok(Arc::new(StringArray::from(out)))
}

/// Whole-partition boolean MIN (AND) / MAX (OR), ordering `false < true`. Nulls are
/// ignored; an all-null partition yields null.
fn grouped_bool_minmax(
    func: WindowFn,
    group_ids: &[u32],
    num_groups: usize,
    values: &ArrayRef,
) -> Result<ArrayRef, RuntimeError> {
    let arr = values.as_boolean();
    let is_min = func == WindowFn::Min;
    let mut acc: Vec<Option<bool>> = vec![None; num_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        if arr.is_valid(i) {
            let v = arr.value(i);
            let slot = &mut acc[g as usize];
            *slot = Some(match *slot {
                None => v,
                Some(a) if is_min => a && v,
                Some(a) => a || v,
            });
        }
    }
    let out: Vec<Option<bool>> = group_ids.iter().map(|&g| acc[g as usize]).collect();
    Ok(Arc::new(BooleanArray::from(out)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f64s(a: &ArrayRef) -> Vec<f64> {
        let x = a.as_primitive::<Float64Type>();
        (0..x.len()).map(|i| x.value(i)).collect()
    }

    /// Whole-partition `avg` over i64 must sum EXACTLY (128-bit) before dividing —
    /// accumulating each value through f64 loses precision above 2^53, drifting ±1 from
    /// DuckDB. avg([2^53+1, 1]) is exactly 2^52+1 (4503599627370497), not 2^52
    /// (4503599627370496) that the old `sum += v as f64` produced.
    #[test]
    fn i64_avg_is_exact_above_2_53() {
        let vals: ArrayRef = Arc::new(Int64Array::from(vec![(1i64 << 53) + 1, 1]));
        let group_ids = [0u32, 0];
        let out = broadcast_partition_aggregate(WindowFn::Avg, &group_ids, 1, Some(&vals)).unwrap();
        assert_eq!(f64s(&out), vec![4503599627370497.0, 4503599627370497.0]);
    }

    /// A sum that overflows i64 but fits i128 (so DuckDB, which sums in 128-bit, still
    /// returns a finite avg) must not overflow the accumulator. avg([2^62, 2^62]) = 2^62.
    #[test]
    fn i64_avg_sum_overflowing_i64_does_not_panic() {
        let vals: ArrayRef = Arc::new(Int64Array::from(vec![1i64 << 62, 1i64 << 62]));
        let group_ids = [0u32, 0];
        let out = broadcast_partition_aggregate(WindowFn::Avg, &group_ids, 1, Some(&vals)).unwrap();
        assert_eq!(
            f64s(&out),
            vec![4611686018427387904.0, 4611686018427387904.0]
        );
    }
}
