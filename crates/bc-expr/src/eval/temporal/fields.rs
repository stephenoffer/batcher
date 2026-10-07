//! The field extractions that depend on what a value *is* rather than on a calendar:
//! a duration's components, the precision-aware epoch and nanosecond, and reading a
//! tz-aware instant as its own zone's wall clock before any calendar field is taken.
//!
//! They sit in front of `date::eval_date`, which owns the calendar fields proper. Each one
//! exists because the calendar path got the question wrong for a type it was not written
//! for: it read a `Duration` as an instant (so `hour` of 49h05m was 49), read a
//! `Timestamp(ns)` through a microsecond cast (so its last three digits vanished), and read
//! the UTC micros of a zoned column for half of its functions (so `dayname` disagreed with
//! `hour` about which day it was).

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Int64Array};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Int64Type, TimeUnit};

use crate::{DateFunc, ExprError};

/// Nanoseconds per one count of `unit`.
fn nanos_per(unit: &TimeUnit) -> i64 {
    match unit {
        TimeUnit::Second => 1_000_000_000,
        TimeUnit::Millisecond => 1_000_000,
        TimeUnit::Microsecond => 1_000,
        TimeUnit::Nanosecond => 1,
    }
}

/// Map each non-null Int64 value through `f`.
fn map_i64(raw: &ArrayRef, f: impl Fn(i64) -> Option<i64>) -> ArrayRef {
    let r = raw.as_primitive::<Int64Type>();
    let out: Int64Array = (0..r.len())
        .map(|i| if r.is_null(i) { None } else { f(r.value(i)) })
        .collect();
    Arc::new(out)
}

/// The stored count of a timestamp or date column with its resolution in nanoseconds. A
/// `Date32` is days; anything else is parsed as a microsecond timestamp first, the way the
/// rest of the family treats text.
fn raw_count(arr: &ArrayRef) -> Result<(ArrayRef, i64), ExprError> {
    match arr.data_type() {
        DataType::Timestamp(unit, _) => Ok((cast(arr, &DataType::Int64)?, nanos_per(unit))),
        DataType::Date32 => Ok((
            cast(&cast(arr, &DataType::Int32)?, &DataType::Int64)?,
            86_400 * 1_000_000_000,
        )),
        _ => {
            let ts = cast(arr, &DataType::Timestamp(TimeUnit::Microsecond, None))?;
            Ok((cast(&ts, &DataType::Int64)?, 1_000))
        }
    }
}

/// `epoch_ns` and `nanosecond`, read at the input's own resolution.
pub(crate) fn eval_precise(func: DateFunc, arr: &ArrayRef) -> Result<ArrayRef, ExprError> {
    let (raw, per) = raw_count(arr)?;
    Ok(match func {
        // An overflowing scale is null rather than a wrapped instant, as `from_epoch` is.
        DateFunc::EpochNs => map_i64(&raw, |v| v.checked_mul(per)),
        // A date has no time of day.
        _ if per > 1_000_000_000 => map_i64(&raw, |_| Some(0)),
        // The floored remainder, so a pre-1970 instant reports the nanosecond it reads.
        _ => {
            let per_second = 1_000_000_000 / per;
            map_i64(&raw, |v| Some(v.rem_euclid(per_second) * per))
        }
    })
}

/// A component of a `Duration`, as DuckDB reads an interval: `day` is the whole days, and
/// `hour`/`minute`/`second` the clock parts within them (`hour` of 49h05m is 1, not 49).
/// Each truncates toward zero, so a negative duration has negative components, as DuckDB's
/// do. A total is `dt.total(unit)`; any other field has no meaning for a duration.
pub(crate) fn eval_duration_part(
    func: DateFunc,
    arr: &ArrayRef,
    unit: &TimeUnit,
) -> Result<ArrayRef, ExprError> {
    const NS_PER_SEC: i64 = 1_000_000_000;
    let (span, modulus) = match func {
        DateFunc::Day => (86_400 * NS_PER_SEC, None),
        DateFunc::Hour => (3_600 * NS_PER_SEC, Some(24)),
        DateFunc::Minute => (60 * NS_PER_SEC, Some(60)),
        DateFunc::Second => (NS_PER_SEC, Some(60)),
        other => {
            return Err(ExprError::InvalidArgument {
                func: format!("{other:?}").to_lowercase(),
                reason: "a duration has day/hour/minute/second components only; use \
                         dt.total(unit) for a total"
                    .into(),
            })
        }
    };
    let per = nanos_per(unit);
    let raw = cast(arr, &DataType::Int64)?;
    // Whole `span`s in the value, computed in the value's own unit so nothing overflows.
    Ok(map_i64(&raw, |v| {
        let whole = if span >= per {
            v / (span / per)
        } else {
            v * (per / span)
        };
        Some(modulus.map_or(whole, |m| whole % m))
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{DurationMicrosecondArray, TimestampNanosecondArray};

    fn ints(a: &ArrayRef) -> Vec<Option<i64>> {
        a.as_primitive::<Int64Type>().iter().collect()
    }

    #[test]
    fn duration_components_match_duckdb_intervals() {
        // 2 days 01:05:07 and its negation.
        let us: i64 = (2 * 86_400 + 3_600 + 5 * 60 + 7) * 1_000_000;
        let arr: ArrayRef = Arc::new(DurationMicrosecondArray::from(vec![
            Some(us),
            Some(-us),
            None,
        ]));
        let part = |f| ints(&eval_duration_part(f, &arr, &TimeUnit::Microsecond).unwrap());
        assert_eq!(part(DateFunc::Day), vec![Some(2), Some(-2), None]);
        assert_eq!(part(DateFunc::Hour), vec![Some(1), Some(-1), None]);
        assert_eq!(part(DateFunc::Minute), vec![Some(5), Some(-5), None]);
        assert_eq!(part(DateFunc::Second), vec![Some(7), Some(-7), None]);
        assert!(eval_duration_part(DateFunc::Year, &arr, &TimeUnit::Microsecond).is_err());
    }

    #[test]
    fn nanosecond_precision_survives() {
        let arr: ArrayRef = Arc::new(TimestampNanosecondArray::from(vec![1_000_000_001, -1]));
        assert_eq!(
            ints(&eval_precise(DateFunc::EpochNs, &arr).unwrap()),
            vec![Some(1_000_000_001), Some(-1)]
        );
        assert_eq!(
            ints(&eval_precise(DateFunc::Nanosecond, &arr).unwrap()),
            vec![Some(1), Some(999_999_999)]
        );
    }
}
