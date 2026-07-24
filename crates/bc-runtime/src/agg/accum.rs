//! Per-type accumulator helpers for `sum`/`min`/`max` and the masked-array and
//! concat utilities they share.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BinaryArray, BooleanArray, Decimal128Array, Float64Array, Int64Array,
    LargeBinaryArray, LargeStringArray, StringArray, UInt32Array,
};
use arrow::compute::take;
use arrow::datatypes::{DataType, Decimal128Type, Float64Type, Int64Type};

use super::{accumulate, arg_extreme_state, covar_state, AggCall, AggFunc, Partial};
use crate::error::RuntimeError;

/// Produce the partial-state columns for one aggregate call. The two-input functions
/// (`arg_min`/`arg_max` carry an ordering key; `covar`/`corr` carry a second value) go to
/// their dedicated builders; every single-input function goes to [`accumulate`].
pub(super) fn accumulate_call(
    call: &AggCall,
    group_ids: &[u32],
    num_groups: usize,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    match call.func {
        AggFunc::ArgMin | AggFunc::ArgMax => arg_extreme_state(
            require(call.values.as_ref(), call.func)?,
            require(call.key.as_ref(), call.func)?,
            group_ids,
            num_groups,
            matches!(call.func, AggFunc::ArgMax),
        ),
        AggFunc::CovarPop | AggFunc::CovarSamp | AggFunc::Corr => covar_state(
            require(call.values.as_ref(), call.func)?,
            require(call.key.as_ref(), call.func)?,
            group_ids,
            num_groups,
        ),
        _ => accumulate(call.func, call.values.as_ref(), group_ids, num_groups),
    }
}

/// Global (no-`GROUP BY`) partial aggregation: every row is the single group 0. The scalar
/// reductions (`sum`/`min`/`max`/`count`/`mean`) hit their `num_groups == 1` kernels; the
/// rest scatter into the one group through a shared zero-id buffer. Group columns are empty
/// (a global aggregate has no key), matching the grouped `partial`'s `Partial` shape.
pub(crate) fn global_partial(calls: &[AggCall], num_rows: usize) -> Result<Partial, RuntimeError> {
    let zeros = vec![0u32; num_rows];
    let mut states = Vec::with_capacity(calls.len());
    for call in calls {
        states.push(accumulate_call(call, &zeros, 1)?);
    }
    Ok(Partial {
        group_columns: Vec::new(),
        states,
    })
}

pub(crate) fn sum_acc(
    values: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
    func: AggFunc,
) -> Result<ArrayRef, RuntimeError> {
    // Global-sum fast path: when there is a single group every row maps to it, so the
    // group sum equals the whole-column sum — arrow's SIMD reduction kernels beat the
    // scalar scatter loop below (the dominant cost of a global `SUM`/`COUNT`-derived
    // aggregate, and of every distributed `combine` that folds a handful of partials).
    // Null-only / empty input yields a null sum (SQL semantics), matching `masked_*`.
    if num_groups == 1 {
        match values.data_type() {
            DataType::Int64 => {
                let s = arrow::compute::sum_checked(values.as_primitive::<Int64Type>())
                    .map_err(|_| RuntimeError::SumOverflow)?;
                return Ok(Arc::new(masked_i64(
                    vec![s.unwrap_or(0)],
                    vec![s.is_some()],
                )));
            }
            DataType::Float64 => {
                let s = arrow::compute::sum(values.as_primitive::<Float64Type>());
                return Ok(Arc::new(masked_f64(
                    vec![s.unwrap_or(0.0)],
                    vec![s.is_some()],
                )));
            }
            // Decimal keeps the scatter loop below (exact i128 accumulation, scale-aware).
            _ => {}
        }
    }
    match values.data_type() {
        DataType::Int64 => {
            let arr = values.as_primitive::<Int64Type>();
            let mut sums = vec![0i64; num_groups];
            // Checked add throughout: a silent i64 wrap would be a wrong answer. (DuckDB
            // promotes BIGINT sums to 128-bit; we error rather than corrupt until that
            // wider-output promotion lands.)
            if arr.null_count() == 0 {
                // No-null fast path: gather straight from the values slice, skipping the
                // per-row validity branch and the per-row `valid` write (every group is
                // non-empty and all-valid) — mirrors the Float64 path below.
                for (&g, &v) in group_ids.iter().zip(arr.values()) {
                    let slot = &mut sums[g as usize];
                    *slot = slot.checked_add(v).ok_or(RuntimeError::SumOverflow)?;
                }
                return Ok(Arc::new(masked_i64(sums, vec![true; num_groups])));
            }
            let mut valid = vec![false; num_groups];
            for (i, &g) in group_ids.iter().enumerate() {
                if arr.is_valid(i) {
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
        // Decimal sums accumulate in i128 (scale preserved). `checked_add` so a sum past
        // i128's range errors instead of silently wrapping to a negative value, mirroring
        // the i64 SUM path (DuckDB raises on decimal overflow).
        DataType::Decimal128(p, s) => {
            let arr = values.as_primitive::<Decimal128Type>();
            let mut sums = vec![0i128; num_groups];
            let mut valid = vec![false; num_groups];
            for (i, &g) in group_ids.iter().enumerate() {
                if arr.is_valid(i) {
                    let slot = &mut sums[g as usize];
                    *slot = slot
                        .checked_add(arr.value(i))
                        .ok_or(RuntimeError::SumOverflow)?;
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
                    // NOT `v < cur` / `v > cur`: raw IEEE comparison is false against NaN, so a
                    // NaN could never win and `max()` silently skipped it — disagreeing with our
                    // own ORDER BY (which sorts NaN last, i.e. greatest) and with DuckDB. The
                    // total order lives in `crate::keys` with the rest of the float semantics.
                    let ord = crate::keys::float_total_cmp(v, cur[g]);
                    let wins = if is_min {
                        ord == std::cmp::Ordering::Less
                    } else {
                        ord == std::cmp::Ordering::Greater
                    };
                    if !valid[g] || wins {
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
        // Byte-ordered min/max over string and binary columns. Rust's `<`/`>` on `&str` and
        // `&[u8]` is lexicographic by byte, which is exactly DuckDB's default (binary) collation,
        // so `min`/`max` agree on ordering, NULL-skipping, empty strings, and unicode. The four
        // arms cover the 32- and 64-bit offset variants of both `Utf8` and `Binary`; without the
        // `LargeUtf8`/`Binary`/`LargeBinary` arms these raised "not supported for column type",
        // while DuckDB happily computes them.
        DataType::Utf8 => byte_minmax(
            values.as_string::<i32>(),
            group_ids,
            num_groups,
            is_min,
            |cur| {
                Arc::new(
                    cur.into_iter()
                        .map(bytes_to_string)
                        .collect::<StringArray>(),
                )
            },
        ),
        DataType::LargeUtf8 => byte_minmax(
            values.as_string::<i64>(),
            group_ids,
            num_groups,
            is_min,
            |cur| {
                Arc::new(
                    cur.into_iter()
                        .map(bytes_to_string)
                        .collect::<LargeStringArray>(),
                )
            },
        ),
        DataType::Binary => byte_minmax(
            values.as_binary::<i32>(),
            group_ids,
            num_groups,
            is_min,
            |cur| Arc::new(BinaryArray::from_iter(cur)),
        ),
        DataType::LargeBinary => byte_minmax(
            values.as_binary::<i64>(),
            group_ids,
            num_groups,
            is_min,
            |cur| Arc::new(LargeBinaryArray::from_iter(cur)),
        ),
        // SQL orders booleans `false < true`, so a group's minimum is the AND of its values
        // and its maximum is the OR — the very folds `bool_and`/`bool_or` already perform
        // (which is how DuckDB defines them too). Delegating rather than restating the fold
        // keeps one implementation of the boolean reduction.
        //
        // This arm is not a new capability so much as the closing of a gap that had already
        // become a *wrong answer*: a boolean column's footer records an exact min/max, so
        // `min(flag)` over a Parquet scan was answered `false` from metadata while the same
        // query over the same rows in memory raised "aggregate min is not supported for
        // column type Boolean". A metadata shortcut that can answer what the engine cannot
        // is not a shortcut; it is a second, disagreeing implementation.
        DataType::Boolean => bool_acc(values, group_ids, num_groups, is_min, func),
        // Temporal min/max. Date/Time/Timestamp/Duration are stored as integers whose
        // natural (chronological) order IS their integer order — a later instant has a larger
        // underlying value in the same unit, and a tz-aware Timestamp's i64 is the UTC instant,
        // so comparison is correct regardless of the display zone. We compare on the cast-to-i64
        // representation but `take` the *winner from the original typed array*, so the result
        // keeps the exact unit and timezone (a cast of the reduced integer back could not).
        //
        // Without this arm `min`/`max` over a Date/Timestamp column raised "not supported for
        // column type Date32" while the Parquet footer answered the same query from metadata —
        // the very metadata-vs-engine disagreement the Boolean arm above was added to close.
        DataType::Date32
        | DataType::Date64
        | DataType::Time32(_)
        | DataType::Time64(_)
        | DataType::Timestamp(_, _)
        | DataType::Duration(_) => temporal_minmax(values, group_ids, num_groups, is_min),
        other => Err(RuntimeError::UnsupportedAggregate {
            func: func.name().to_string(),
            dtype: other.to_string(),
        }),
    }
}

/// Per-group min/max over a temporal column via its underlying `i64` order, returning the
/// winning rows `take`n from the original array (so unit/timezone are preserved exactly).
/// Null-skipping; an empty/all-null group yields null. Associative — the partial state is
/// itself a temporal column, so merging re-enters this same reducer.
fn temporal_minmax(
    values: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
    is_min: bool,
) -> Result<ArrayRef, RuntimeError> {
    let ints = arrow::compute::cast(values, &DataType::Int64)?;
    let arr = ints.as_primitive::<Int64Type>();
    let mut cur = vec![0i64; num_groups];
    let mut best = vec![0u32; num_groups];
    let mut valid = vec![false; num_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        if arr.is_valid(i) {
            let (g, v) = (g as usize, arr.value(i));
            if !valid[g] || (is_min && v < cur[g]) || (!is_min && v > cur[g]) {
                cur[g] = v;
                best[g] = i as u32;
                valid[g] = true;
            }
        }
    }
    // A null index makes `take` emit null, so empty/all-null groups become null.
    let idx: UInt32Array = best
        .into_iter()
        .zip(valid)
        .map(|(i, ok)| ok.then_some(i))
        .collect();
    Ok(take(values.as_ref(), &idx, None)?)
}

/// Per-group byte-lexicographic min/max over a string/binary column. Stores each group's
/// winning bytes and lets `build` re-wrap them as the concrete output array (string or
/// binary, 32- or 64-bit offsets). Null-skipping; an all-null group stays `None` (→ null).
/// The comparison is `&[u8]` order — identical to DuckDB's default binary collation, and to
/// `&str` order for valid UTF-8.
fn byte_minmax<T, F>(
    arr: &arrow::array::GenericByteArray<T>,
    group_ids: &[u32],
    num_groups: usize,
    is_min: bool,
    build: F,
) -> Result<ArrayRef, RuntimeError>
where
    T: arrow::array::types::ByteArrayType,
    T::Native: AsRef<[u8]>,
    F: FnOnce(Vec<Option<Vec<u8>>>) -> ArrayRef,
{
    let mut cur: Vec<Option<Vec<u8>>> = vec![None; num_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        if arr.is_valid(i) {
            let g = g as usize;
            let v: &[u8] = arr.value(i).as_ref();
            let replace = match &cur[g] {
                None => true,
                Some(c) => (is_min && v < c.as_slice()) || (!is_min && v > c.as_slice()),
            };
            if replace {
                cur[g] = Some(v.to_vec());
            }
        }
    }
    Ok(build(cur))
}

/// Reconstruct a `String` from bytes that came out of a `Utf8`/`LargeUtf8` array (so they are
/// valid UTF-8 by construction — the `expect` guards a genuine invariant, never user data).
fn bytes_to_string(b: Option<Vec<u8>>) -> Option<String> {
    b.map(|b| String::from_utf8(b).expect("bytes from a Utf8 array are valid UTF-8"))
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

/// Concatenate the partials' columns for a `combine`.
///
/// Through [`crate::gather::concat_columns`], not arrow's `concat` directly — on a
/// high-cardinality string key this is the copy that dominates the whole combine.
pub(crate) fn concat_col<'a>(
    it: impl Iterator<Item = &'a ArrayRef>,
) -> Result<ArrayRef, RuntimeError> {
    let cols: Vec<&dyn Array> = it.map(|a| a.as_ref()).collect();
    crate::gather::concat_columns(&cols)
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
    fn minmax_over_binary_and_large_strings() {
        use arrow::array::{BinaryArray, LargeBinaryArray, LargeStringArray};
        // Two groups over each byte type; nulls are skipped, ordering is bytewise.
        let g = [0u32, 0, 1, 0];

        // Binary: g0 = {b"foo", b"bar", null}; g1 = {b"zzz"}.
        let bin: ArrayRef = Arc::new(BinaryArray::from_opt_vec(vec![
            Some(b"foo"),
            Some(b"bar"),
            Some(b"zzz"),
            None,
        ]));
        let min = minmax_acc(&bin, &g, 2, true, AggFunc::Min).unwrap();
        let max = minmax_acc(&bin, &g, 2, false, AggFunc::Max).unwrap();
        let min = min.as_any().downcast_ref::<BinaryArray>().unwrap();
        let max = max.as_any().downcast_ref::<BinaryArray>().unwrap();
        assert_eq!(min.value(0), b"bar");
        assert_eq!(max.value(0), b"foo");
        assert_eq!(min.value(1), b"zzz");

        // LargeUtf8 min/max.
        let ls: ArrayRef = Arc::new(LargeStringArray::from(vec![
            Some("delta"),
            Some("alpha"),
            Some("omega"),
            None,
        ]));
        let lmin = minmax_acc(&ls, &g, 2, true, AggFunc::Min).unwrap();
        let lmin = lmin.as_any().downcast_ref::<LargeStringArray>().unwrap();
        assert_eq!(lmin.value(0), "alpha");

        // LargeBinary max.
        let lb: ArrayRef = Arc::new(LargeBinaryArray::from_opt_vec(vec![
            Some(b"aa"),
            Some(b"ab"),
            Some(b"zz"),
            None,
        ]));
        let lbmax = minmax_acc(&lb, &g, 2, false, AggFunc::Max).unwrap();
        let lbmax = lbmax.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
        assert_eq!(lbmax.value(0), b"ab");
    }

    #[test]
    fn temporal_minmax_over_date_and_timestamp() {
        use arrow::array::{Date32Array, TimestampMicrosecondArray};
        // Date32 (days since epoch), two groups, a null skipped.
        let g = [0u32, 0, 1, 0];
        let dates: ArrayRef = Arc::new(Date32Array::from(vec![
            Some(19_723), // 2024-01-03
            Some(19_721), // 2024-01-01
            Some(19_725), // 2024-01-05 (group 1)
            None,
        ]));
        let min = minmax_acc(&dates, &g, 2, true, AggFunc::Min).unwrap();
        let max = minmax_acc(&dates, &g, 2, false, AggFunc::Max).unwrap();
        // Type preserved (not cast to Int64).
        assert_eq!(min.data_type(), &DataType::Date32);
        let min = min.as_any().downcast_ref::<Date32Array>().unwrap();
        let max = max.as_any().downcast_ref::<Date32Array>().unwrap();
        assert_eq!(min.value(0), 19_721);
        assert_eq!(max.value(0), 19_723);
        assert_eq!(min.value(1), 19_725);

        // Timestamp with a timezone: the underlying i64 (UTC instant) drives the order, and the
        // timezone is carried through by `take`ing the original array.
        let tz_ty = DataType::Timestamp(
            arrow::datatypes::TimeUnit::Microsecond,
            Some("+05:00".into()),
        );
        let ts: ArrayRef = Arc::new(
            TimestampMicrosecondArray::from(vec![30i64, 10, 99, 20])
                .with_timezone_opt(Some("+05:00")),
        );
        let tmin = minmax_acc(&ts, &[0u32, 0, 0, 0], 1, true, AggFunc::Min).unwrap();
        assert_eq!(tmin.data_type(), &tz_ty);
        let tmin = tmin
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();
        assert_eq!(tmin.value(0), 10);
    }

    #[test]
    fn temporal_minmax_is_mergeable_across_partitions() {
        use crate::agg::{combine, finalize, group_aggregate, AggCall};
        use arrow::array::Date32Array;
        // The mergeable-algebra invariant: combine_finalize(partition(partial(x))) == single-node.
        let keys = |k: Vec<&str>| -> ArrayRef { Arc::new(StringArray::from(k)) };
        let dates = |d: Vec<Option<i32>>| -> ArrayRef { Arc::new(Date32Array::from(d)) };
        let funcs = [AggFunc::Min, AggFunc::Max];
        let call = |v: &ArrayRef| {
            vec![
                AggCall::new(AggFunc::Min, Some(v.clone())),
                AggCall::new(AggFunc::Max, Some(v.clone())),
            ]
        };

        // Whole input, single node.
        let k_all = keys(vec!["a", "a", "b", "a", "b"]);
        let v_all = dates(vec![Some(5), None, Some(9), Some(2), Some(7)]);
        let whole = group_aggregate(std::slice::from_ref(&k_all), &call(&v_all), 5).unwrap();

        // Split into two partitions, partial each, combine, finalize.
        let (k1, v1) = (
            keys(vec!["a", "a", "b"]),
            dates(vec![Some(5), None, Some(9)]),
        );
        let (k2, v2) = (keys(vec!["a", "b"]), dates(vec![Some(2), Some(7)]));
        let p1 = crate::agg::partial(std::slice::from_ref(&k1), &call(&v1), 3).unwrap();
        let p2 = crate::agg::partial(std::slice::from_ref(&k2), &call(&v2), 2).unwrap();
        let merged = combine(&[p1, p2], &funcs).unwrap();
        let dist_cols = finalize(&funcs, &merged).unwrap();

        // Align both by group key and compare min/max per group.
        let by_key =
            |gk: &ArrayRef, aggs: &[ArrayRef]| -> std::collections::HashMap<String, (i32, i32)> {
                let k = gk.as_string::<i32>();
                let mn = aggs[0].as_any().downcast_ref::<Date32Array>().unwrap();
                let mx = aggs[1].as_any().downcast_ref::<Date32Array>().unwrap();
                (0..k.len())
                    .map(|i| (k.value(i).to_string(), (mn.value(i), mx.value(i))))
                    .collect()
            };
        let single = by_key(&whole.group_columns[0], &whole.agg_columns);
        let dist = by_key(&merged.group_columns[0], &dist_cols);
        assert_eq!(
            single, dist,
            "temporal min/max must be partition-independent"
        );
        assert_eq!(single["a"], (2, 5));
        assert_eq!(single["b"], (7, 9));
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
