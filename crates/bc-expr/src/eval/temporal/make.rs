//! Temporal construction for `Expr::MakeTemporal` — calendar parts and epoch counts in.
//!
//! The inverse direction of `eval::temporal::date`, which pulls fields *out* of a Date/Timestamp.
//! Two shapes share this module because they share every hard part: null propagation
//! across several inputs, range validation, and building the Arrow array.
//!
//! The epoch conversions exist because an `Int64` column of epoch values carries no
//! record of its unit. `CAST(x AS TIMESTAMP)` has to assume one — Arrow assumes
//! microseconds — so a column of epoch *seconds* casts to a 1970 timestamp, silently and
//! with no error. Naming the unit is the only way to be right.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Date32Array, TimestampMicrosecondArray};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Int64Type};
use chrono::NaiveDate;

use crate::{ExprError, MakeTemporalFunc};

/// Days from the Unix epoch to `y-m-d`, or `None` if that date does not exist.
///
/// `NaiveDate::from_ymd_opt` is the range check: it rejects month 0 or 13, February 30,
/// and a year outside chrono's range, so an impossible date becomes a null value rather
/// than a query-aborting error. That is the same leniency `try_cast` and the JSON
/// extractors already take, and it matters most in exactly the place this is used —
/// assembling a date from three columns of dirty upstream integers.
fn ymd_to_days(y: i64, m: i64, d: i64) -> Option<i32> {
    let date = NaiveDate::from_ymd_opt(
        i32::try_from(y).ok()?,
        u32::try_from(m).ok()?,
        u32::try_from(d).ok()?,
    )?;
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1)?;
    Some((date - epoch).num_days() as i32)
}

/// Microseconds from the Unix epoch to `y-m-d h:mi:s`, or `None` if it does not exist.
fn ymdhms_to_micros(y: i64, m: i64, d: i64, h: i64, mi: i64, s: i64) -> Option<i64> {
    let days = i64::from(ymd_to_days(y, m, d)?);
    if !(0..24).contains(&h) || !(0..60).contains(&mi) || !(0..60).contains(&s) {
        return None;
    }
    // Leap seconds are not representable in Arrow's timestamp, so second 60 is rejected
    // above rather than folded into the next minute.
    days.checked_mul(86_400)?
        .checked_add(h * 3_600 + mi * 60 + s)?
        .checked_mul(1_000_000)
}

/// The number of arguments `func` takes. Stated here rather than at each call site so the
/// error message and the evaluation cannot disagree about the arity.
pub(crate) const fn arity(func: MakeTemporalFunc) -> usize {
    match func {
        MakeTemporalFunc::MakeDate => 3,
        MakeTemporalFunc::MakeTimestamp => 6,
        _ => 1,
    }
}

/// Evaluate a temporal constructor over already-evaluated integer argument arrays.
///
/// Every argument is cast to `Int64` first, so an `Int32` year column or a `Float64` that
/// happens to hold whole numbers works without the caller pre-casting. A null in *any*
/// argument makes the row null, which is SQL's rule for a multi-argument function.
pub(crate) fn eval_make_temporal(
    func: MakeTemporalFunc,
    args: &[ArrayRef],
) -> Result<ArrayRef, ExprError> {
    if args.len() != arity(func) {
        return Err(ExprError::InvalidArgument {
            func: format!("{func:?}"),
            reason: format!("expected {} argument(s), got {}", arity(func), args.len()),
        });
    }
    let ints: Vec<ArrayRef> = args
        .iter()
        .map(|a| cast(a, &DataType::Int64).map_err(ExprError::from))
        .collect::<Result<_, _>>()?;
    let cols: Vec<&arrow::array::PrimitiveArray<Int64Type>> =
        ints.iter().map(|a| a.as_primitive::<Int64Type>()).collect();
    let n = cols[0].len();
    // `value_at` reads argument `j` of row `i`, returning `None` for a null so the
    // per-row closures below stay a plain chain of `?`.
    let value_at = |j: usize, i: usize| -> Option<i64> {
        let c = cols[j];
        (!c.is_null(i)).then(|| c.value(i))
    };

    Ok(match func {
        MakeTemporalFunc::MakeDate => Arc::new(
            (0..n)
                .map(|i| ymd_to_days(value_at(0, i)?, value_at(1, i)?, value_at(2, i)?))
                .collect::<Date32Array>(),
        ),
        MakeTemporalFunc::MakeTimestamp => Arc::new(
            (0..n)
                .map(|i| {
                    ymdhms_to_micros(
                        value_at(0, i)?,
                        value_at(1, i)?,
                        value_at(2, i)?,
                        value_at(3, i)?,
                        value_at(4, i)?,
                        value_at(5, i)?,
                    )
                })
                .collect::<TimestampMicrosecondArray>(),
        ),
        MakeTemporalFunc::FromUnixDate => Arc::new(
            (0..n)
                .map(|i| i32::try_from(value_at(0, i)?).ok())
                .collect::<Date32Array>(),
        ),
        // The epoch conversions scale into microseconds. Seconds and millis multiply
        // (checked, so a far-future value is null rather than a wrapped instant); nanos
        // divide with a *floor*, matching `DateFunc::Epoch`'s floor so a pre-1970
        // sub-microsecond instant lands in the microsecond that contains it rather than
        // one microsecond late.
        MakeTemporalFunc::FromUnixSeconds => scaled(n, &value_at, |v| v.checked_mul(1_000_000)),
        MakeTemporalFunc::FromUnixMillis => scaled(n, &value_at, |v| v.checked_mul(1_000)),
        MakeTemporalFunc::FromUnixMicros => scaled(n, &value_at, Some),
        MakeTemporalFunc::FromUnixNanos => scaled(n, &value_at, |v| Some(v.div_euclid(1_000))),
    })
}

/// Build a `Timestamp(µs)` array by applying `scale` to each non-null epoch value.
fn scaled(
    n: usize,
    value_at: &dyn Fn(usize, usize) -> Option<i64>,
    scale: impl Fn(i64) -> Option<i64>,
) -> ArrayRef {
    Arc::new(
        (0..n)
            .map(|i| scale(value_at(0, i)?))
            .collect::<TimestampMicrosecondArray>(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;

    fn ints(vs: Vec<Option<i64>>) -> ArrayRef {
        Arc::new(vs.into_iter().collect::<Int64Array>())
    }

    fn days(a: &ArrayRef) -> Vec<Option<i32>> {
        let d = a.as_primitive::<arrow::datatypes::Date32Type>();
        (0..d.len())
            .map(|i| (!d.is_null(i)).then(|| d.value(i)))
            .collect()
    }

    fn micros(a: &ArrayRef) -> Vec<Option<i64>> {
        let t = a.as_primitive::<arrow::datatypes::TimestampMicrosecondType>();
        (0..t.len())
            .map(|i| (!t.is_null(i)).then(|| t.value(i)))
            .collect()
    }

    #[test]
    fn make_date_builds_the_epoch_and_a_known_date() {
        let out = eval_make_temporal(
            MakeTemporalFunc::MakeDate,
            &[
                ints(vec![Some(1970), Some(2024)]),
                ints(vec![Some(1), Some(2)]),
                ints(vec![Some(1), Some(29)]),
            ],
        )
        .unwrap();
        assert_eq!(days(&out), vec![Some(0), Some(19782)]);
    }

    #[test]
    fn an_impossible_date_is_null_not_an_error() {
        let out = eval_make_temporal(
            MakeTemporalFunc::MakeDate,
            &[
                ints(vec![Some(2023), Some(2024), Some(2024)]),
                ints(vec![Some(2), Some(13), Some(1)]),
                // 2023 is not a leap year, so Feb 29 does not exist; month 13 does not
                // exist; day 0 does not exist.
                ints(vec![Some(29), Some(1), Some(0)]),
            ],
        )
        .unwrap();
        assert_eq!(days(&out), vec![None, None, None]);
    }

    #[test]
    fn a_null_in_any_argument_nulls_the_row() {
        let out = eval_make_temporal(
            MakeTemporalFunc::MakeDate,
            &[
                ints(vec![None, Some(2024), Some(2024)]),
                ints(vec![Some(1), None, Some(1)]),
                ints(vec![Some(1), Some(1), None]),
            ],
        )
        .unwrap();
        assert_eq!(days(&out), vec![None, None, None]);
    }

    #[test]
    fn make_timestamp_rejects_out_of_range_clock_fields() {
        let out = eval_make_temporal(
            MakeTemporalFunc::MakeTimestamp,
            &[
                ints(vec![Some(1970), Some(1970), Some(1970), Some(1970)]),
                ints(vec![Some(1), Some(1), Some(1), Some(1)]),
                ints(vec![Some(1), Some(1), Some(1), Some(1)]),
                ints(vec![Some(0), Some(24), Some(0), Some(0)]),
                ints(vec![Some(0), Some(0), Some(60), Some(0)]),
                // Second 60 is a leap second, which Arrow's timestamp cannot represent.
                ints(vec![Some(1), Some(0), Some(0), Some(60)]),
            ],
        )
        .unwrap();
        assert_eq!(micros(&out), vec![Some(1_000_000), None, None, None]);
    }

    #[test]
    fn epoch_units_scale_to_the_same_instant() {
        // 2023-11-14T22:13:20Z expressed four ways must land on one microsecond value.
        let want = Some(1_700_000_000_000_000);
        for (func, v) in [
            (MakeTemporalFunc::FromUnixSeconds, 1_700_000_000),
            (MakeTemporalFunc::FromUnixMillis, 1_700_000_000_000),
            (MakeTemporalFunc::FromUnixMicros, 1_700_000_000_000_000),
            (MakeTemporalFunc::FromUnixNanos, 1_700_000_000_000_000_000),
        ] {
            let out = eval_make_temporal(func, &[ints(vec![Some(v)])]).unwrap();
            assert_eq!(micros(&out), vec![want], "{func:?}");
        }
    }

    #[test]
    fn nanos_floor_rather_than_truncate_toward_zero() {
        // -1500 ns is 1.5 µs before the epoch: it belongs to microsecond -2, not -1.
        // Truncation toward zero would put it one microsecond late, the bug
        // `DateFunc::Epoch` already documents for the opposite direction.
        let out = eval_make_temporal(
            MakeTemporalFunc::FromUnixNanos,
            &[ints(vec![Some(-1500), Some(1500)])],
        )
        .unwrap();
        assert_eq!(micros(&out), vec![Some(-2), Some(1)]);
    }

    #[test]
    fn an_overflowing_epoch_value_is_null_not_a_wrapped_instant() {
        let out = eval_make_temporal(
            MakeTemporalFunc::FromUnixSeconds,
            &[ints(vec![Some(i64::MAX), Some(0)])],
        )
        .unwrap();
        assert_eq!(micros(&out), vec![None, Some(0)]);
    }

    #[test]
    fn from_unix_date_counts_days() {
        let out = eval_make_temporal(
            MakeTemporalFunc::FromUnixDate,
            &[ints(vec![Some(0), Some(19782)])],
        )
        .unwrap();
        assert_eq!(days(&out), vec![Some(0), Some(19782)]);
    }

    #[test]
    fn wrong_arity_is_a_clear_error() {
        let err =
            eval_make_temporal(MakeTemporalFunc::MakeDate, &[ints(vec![Some(1)])]).unwrap_err();
        assert!(
            format!("{err}").contains("expected 3 argument(s), got 1"),
            "{err}"
        );
    }
}
