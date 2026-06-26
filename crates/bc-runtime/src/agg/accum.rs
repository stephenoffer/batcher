//! Per-type accumulator helpers for `sum`/`min`/`max` and the masked-array and
//! concat utilities they share.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, Decimal128Array, Float64Array, Int64Array, StringArray,
};
use arrow::compute::concat;
use arrow::datatypes::{DataType, Decimal128Type, Float64Type, Int64Type};

use super::AggFunc;
use crate::error::RuntimeError;

pub(crate) fn sum_acc(
    values: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
    func: AggFunc,
) -> Result<ArrayRef, RuntimeError> {
    match values.data_type() {
        DataType::Int64 => {
            let arr = values.as_primitive::<Int64Type>();
            let mut sums = vec![0i64; num_groups];
            let mut valid = vec![false; num_groups];
            for (i, &g) in group_ids.iter().enumerate() {
                if arr.is_valid(i) {
                    // Checked: a silent i64 wrap would be a wrong answer. (DuckDB
                    // promotes BIGINT sums to 128-bit; we error rather than corrupt
                    // until that wider-output promotion lands.)
                    let slot = &mut sums[g as usize];
                    *slot = slot
                        .checked_add(arr.value(i))
                        .ok_or(RuntimeError::SumOverflow)?;
                    valid[g as usize] = true;
                }
            }
            Ok(Arc::new(masked_i64(sums, valid)))
        }
        DataType::Float64 => {
            let arr = values.as_primitive::<Float64Type>();
            let mut sums = vec![0f64; num_groups];
            if arr.null_count() == 0 {
                // No-null fast path: gather straight from the values slice, skipping
                // both the per-row validity branch *and* the per-row `valid` write —
                // every group is non-empty (it exists because a row mapped to it) and
                // has only non-null values, so all groups are valid. Removing the 6M
                // redundant bool writes (per aggregate) is the dominant SUM/AVG path.
                for (&g, &v) in group_ids.iter().zip(arr.values()) {
                    sums[g as usize] += v;
                }
                return Ok(Arc::new(masked_f64(sums, vec![true; num_groups])));
            }
            let mut valid = vec![false; num_groups];
            for (i, &g) in group_ids.iter().enumerate() {
                if arr.is_valid(i) {
                    sums[g as usize] += arr.value(i);
                    valid[g as usize] = true;
                }
            }
            Ok(Arc::new(masked_f64(sums, valid)))
        }
        // Decimal sums are exact (i128 accumulation, scale preserved).
        DataType::Decimal128(p, s) => {
            let arr = values.as_primitive::<Decimal128Type>();
            let mut sums = vec![0i128; num_groups];
            let mut valid = vec![false; num_groups];
            for (i, &g) in group_ids.iter().enumerate() {
                if arr.is_valid(i) {
                    sums[g as usize] += arr.value(i);
                    valid[g as usize] = true;
                }
            }
            Ok(masked_decimal(sums, valid, *p, *s)?)
        }
        other => Err(RuntimeError::UnsupportedAggregate {
            func: func.name().to_string(),
            dtype: other.to_string(),
        }),
    }
}

/// Build a masked `Decimal128Array` with the given precision/scale.
pub(crate) fn masked_decimal(
    values: Vec<i128>,
    valid: Vec<bool>,
    precision: u8,
    scale: i8,
) -> Result<ArrayRef, RuntimeError> {
    let arr: Decimal128Array = values
        .into_iter()
        .zip(valid)
        .map(|(v, ok)| ok.then_some(v))
        .collect();
    let arr = arr
        .with_precision_and_scale(precision, scale)
        .map_err(|e| RuntimeError::UnsupportedAggregate {
            func: "decimal".to_string(),
            dtype: e.to_string(),
        })?;
    Ok(Arc::new(arr))
}

pub(crate) fn minmax_acc(
    values: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
    is_min: bool,
    func: AggFunc,
) -> Result<ArrayRef, RuntimeError> {
    match values.data_type() {
        DataType::Int64 => {
            let arr = values.as_primitive::<Int64Type>();
            let mut cur = vec![0i64; num_groups];
            let mut valid = vec![false; num_groups];
            for (i, &g) in group_ids.iter().enumerate() {
                if arr.is_valid(i) {
                    let (g, v) = (g as usize, arr.value(i));
                    if !valid[g] || (is_min && v < cur[g]) || (!is_min && v > cur[g]) {
                        cur[g] = v;
                        valid[g] = true;
                    }
                }
            }
            Ok(Arc::new(masked_i64(cur, valid)))
        }
        DataType::Float64 => {
            let arr = values.as_primitive::<Float64Type>();
            let mut cur = vec![0f64; num_groups];
            let mut valid = vec![false; num_groups];
            for (i, &g) in group_ids.iter().enumerate() {
                if arr.is_valid(i) {
                    let (g, v) = (g as usize, arr.value(i));
                    if !valid[g] || (is_min && v < cur[g]) || (!is_min && v > cur[g]) {
                        cur[g] = v;
                        valid[g] = true;
                    }
                }
            }
            Ok(Arc::new(masked_f64(cur, valid)))
        }
        DataType::Decimal128(p, s) => {
            let arr = values.as_primitive::<Decimal128Type>();
            let mut cur = vec![0i128; num_groups];
            let mut valid = vec![false; num_groups];
            for (i, &g) in group_ids.iter().enumerate() {
                if arr.is_valid(i) {
                    let (g, v) = (g as usize, arr.value(i));
                    if !valid[g] || (is_min && v < cur[g]) || (!is_min && v > cur[g]) {
                        cur[g] = v;
                        valid[g] = true;
                    }
                }
            }
            masked_decimal(cur, valid, *p, *s)
        }
        DataType::Utf8 => {
            let arr = values.as_any().downcast_ref::<StringArray>().expect("utf8");
            let mut cur: Vec<Option<String>> = vec![None; num_groups];
            for (i, &g) in group_ids.iter().enumerate() {
                if arr.is_valid(i) {
                    let (g, v) = (g as usize, arr.value(i));
                    let replace = match &cur[g] {
                        None => true,
                        Some(c) => (is_min && v < c.as_str()) || (!is_min && v > c.as_str()),
                    };
                    if replace {
                        cur[g] = Some(v.to_string());
                    }
                }
            }
            Ok(Arc::new(StringArray::from(cur)))
        }
        other => Err(RuntimeError::UnsupportedAggregate {
            func: func.name().to_string(),
            dtype: other.to_string(),
        }),
    }
}

/// Boolean reduction per group: `bool_and` (logical AND of non-null values) or
/// `bool_or` (logical OR). Nulls are ignored; a group with no non-null value
/// yields null. Associative and idempotent over a single partial, so the same
/// function merges already-partial boolean state — AND/OR commute and associate.
pub(crate) fn bool_acc(
    values: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
    is_and: bool,
    func: AggFunc,
) -> Result<ArrayRef, RuntimeError> {
    let arr = values
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| RuntimeError::UnsupportedAggregate {
            func: func.name().to_string(),
            dtype: values.data_type().to_string(),
        })?;
    let mut cur = vec![false; num_groups];
    let mut valid = vec![false; num_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        if arr.is_valid(i) {
            let (g, v) = (g as usize, arr.value(i));
            if !valid[g] {
                cur[g] = v;
                valid[g] = true;
            } else if is_and {
                cur[g] = cur[g] && v;
            } else {
                cur[g] = cur[g] || v;
            }
        }
    }
    let out: BooleanArray = cur
        .into_iter()
        .zip(valid)
        .map(|(v, ok)| ok.then_some(v))
        .collect();
    Ok(Arc::new(out))
}

/// Fold each group's non-null Int64 values with a bitwise op (`bit_and`/`bit_or`/
/// `bit_xor`). Null-skipping; an all-null group yields null. The op is associative
/// and commutative, so the same fold merges already-partial state across partitions.
pub(crate) fn bitfold_acc(
    values: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
    func: AggFunc,
) -> Result<ArrayRef, RuntimeError> {
    let arr = values
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| RuntimeError::UnsupportedAggregate {
            func: func.name().to_string(),
            dtype: values.data_type().to_string(),
        })?;
    let mut cur = vec![0i64; num_groups];
    let mut valid = vec![false; num_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        if arr.is_valid(i) {
            let (g, v) = (g as usize, arr.value(i));
            if !valid[g] {
                cur[g] = v;
                valid[g] = true;
            } else {
                cur[g] = match func {
                    AggFunc::BitAnd => cur[g] & v,
                    AggFunc::BitOr => cur[g] | v,
                    AggFunc::BitXor => cur[g] ^ v,
                    _ => unreachable!("bitfold_acc on non-bitwise func"),
                };
            }
        }
    }
    Ok(Arc::new(masked_i64(cur, valid)))
}

/// Product of each group's non-null values as Float64 (DuckDB `product` returns
/// DOUBLE; f64 avoids the silent integer overflow a wrapping i64 product would hit).
/// Null-skipping; an all-null group yields null. Associative — the same fold merges
/// the (already-Float64) partial state.
pub(crate) fn product_acc(
    values: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
) -> Result<ArrayRef, RuntimeError> {
    let f = arrow::compute::cast(values, &DataType::Float64)?;
    let arr = f.as_primitive::<Float64Type>();
    let mut cur = vec![1f64; num_groups];
    let mut valid = vec![false; num_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        if arr.is_valid(i) {
            let g = g as usize;
            cur[g] *= arr.value(i);
            valid[g] = true;
        }
    }
    Ok(Arc::new(masked_f64(cur, valid)))
}

pub(crate) fn masked_i64(vals: Vec<i64>, valid: Vec<bool>) -> Int64Array {
    Int64Array::from_iter(vals.into_iter().zip(valid).map(|(v, ok)| ok.then_some(v)))
}

pub(crate) fn masked_f64(vals: Vec<f64>, valid: Vec<bool>) -> Float64Array {
    Float64Array::from_iter(vals.into_iter().zip(valid).map(|(v, ok)| ok.then_some(v)))
}

pub(crate) fn concat_col<'a>(
    it: impl Iterator<Item = &'a ArrayRef>,
) -> Result<ArrayRef, RuntimeError> {
    let cols: Vec<&dyn Array> = it.map(|a| a.as_ref()).collect();
    Ok(concat(&cols)?)
}

pub(crate) fn require(values: Option<&ArrayRef>, func: AggFunc) -> Result<&ArrayRef, RuntimeError> {
    values.ok_or_else(|| RuntimeError::MissingAggregateInput {
        func: func.name().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int64_sum_overflow_errors_instead_of_wrapping() {
        // i64::MAX + 1 in one group must error, not silently wrap to i64::MIN.
        let values: ArrayRef = Arc::new(Int64Array::from(vec![i64::MAX, 1]));
        let group_ids = [0u32, 0];
        let r = sum_acc(&values, &group_ids, 1, AggFunc::Sum);
        assert!(matches!(r, Err(RuntimeError::SumOverflow)), "got {r:?}");
    }

    #[test]
    fn int64_sum_in_range_is_unaffected() {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![10, 20, 30]));
        let group_ids = [0u32, 0, 0];
        let out = sum_acc(&values, &group_ids, 1, AggFunc::Sum).unwrap();
        assert_eq!(out.as_primitive::<Int64Type>().value(0), 60);
    }

    #[test]
    fn product_skips_nulls_and_uses_f64() {
        // [2, 3, null, 4] in one group → 24.0; the null is skipped, no overflow.
        let values: ArrayRef = Arc::new(Int64Array::from(vec![Some(2), Some(3), None, Some(4)]));
        let group_ids = [0u32, 0, 0, 0];
        let out = product_acc(&values, &group_ids, 1).unwrap();
        assert_eq!(out.as_primitive::<Float64Type>().value(0), 24.0);
    }

    #[test]
    fn bitfold_and_or_xor() {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![6, 3, 5]));
        let g = [0u32, 0, 0];
        let and = bitfold_acc(&values, &g, 1, AggFunc::BitAnd).unwrap();
        let or = bitfold_acc(&values, &g, 1, AggFunc::BitOr).unwrap();
        let xor = bitfold_acc(&values, &g, 1, AggFunc::BitXor).unwrap();
        assert_eq!(and.as_primitive::<Int64Type>().value(0), 6 & 3 & 5);
        assert_eq!(or.as_primitive::<Int64Type>().value(0), 6 | 3 | 5);
        assert_eq!(xor.as_primitive::<Int64Type>().value(0), 6 ^ 3 ^ 5);
    }
}
