//! Per-row calendar shifts: `BinaryOp::AddMonths` and `BinaryOp::AddDays`.
//!
//! `offset_by` moves every row by one plan-time constant; these move each row by its own
//! count, which is how a validity period stored beside the date it starts from is applied
//! without a Python callback. They are *calendar* shifts, not elapsed time: a month added to
//! January 31 lands on the last day of February (chrono `checked_add_months` clamping, the
//! same rule `offset_by` follows through `bc_arrow::offset`), and on a zoned timestamp a day
//! is a local day, so the wall clock reads the same across a DST change. An elapsed shift is
//! `ts + duration`, which already works per row.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Date32Array, TimestampMicrosecondArray};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Date32Type, Int64Type, TimeUnit};
use chrono::{DateTime, Months, NaiveDate};

use super::timezone::on_wall_clock;
use crate::ExprError;

const MICROS_PER_DAY: i64 = 86_400_000_000;

/// Which calendar unit a per-row shift counts.
#[derive(Clone, Copy)]
enum Unit {
    Months,
    Days,
}

/// Add `months[i]` calendar months to each Date32/Timestamp `dates[i]` (negative to
/// subtract), preserving the input type. Null on either side → null.
pub(crate) fn add_months(dates: &ArrayRef, months: &ArrayRef) -> Result<ArrayRef, ExprError> {
    shift(dates, months, Unit::Months, "add_months")
}

/// Add `days[i]` calendar days to each Date32/Timestamp `dates[i]` (negative to subtract),
/// preserving the input type. Null on either side → null.
pub(crate) fn add_days(dates: &ArrayRef, days: &ArrayRef) -> Result<ArrayRef, ExprError> {
    shift(dates, days, Unit::Days, "add_days")
}

fn shift(
    dates: &ArrayRef,
    counts: &ArrayRef,
    unit: Unit,
    func: &str,
) -> Result<ArrayRef, ExprError> {
    if let Some(out) = on_wall_clock(dates, |wall| shift(wall, counts, unit, func))? {
        return Ok(out);
    }
    let n = cast(counts, &DataType::Int64)?;
    let n = n.as_primitive::<Int64Type>();
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).expect("the Unix epoch is a valid date");
    // One day count moved by one count of `unit`; `None` past chrono's range.
    let move_day = |day: i64, k: i64| -> Option<i64> {
        match unit {
            Unit::Days => day.checked_add(k),
            Unit::Months => {
                let d = epoch.checked_add_signed(chrono::Duration::try_days(day)?)?;
                let moved = if k >= 0 {
                    d.checked_add_months(Months::new(u32::try_from(k).ok()?))
                } else {
                    d.checked_sub_months(Months::new(u32::try_from(k.checked_neg()?).ok()?))
                }?;
                Some((moved - epoch).num_days())
            }
        }
    };
    match dates.data_type() {
        DataType::Date32 => {
            let a = dates.as_primitive::<Date32Type>();
            let out: Date32Array = (0..a.len())
                .map(|i| {
                    if a.is_null(i) || n.is_null(i) {
                        return None;
                    }
                    let day = move_day(i64::from(a.value(i)), n.value(i))?;
                    i32::try_from(day).ok()
                })
                .collect();
            Ok(Arc::new(out))
        }
        DataType::Timestamp(_, tz) => {
            let micros = cast(
                dates,
                &DataType::Timestamp(TimeUnit::Microsecond, tz.clone()),
            )?;
            let micros = cast(&micros, &DataType::Int64)?;
            let a = micros.as_primitive::<Int64Type>();
            let out: TimestampMicrosecondArray = (0..a.len())
                .map(|i| {
                    if a.is_null(i) || n.is_null(i) {
                        return None;
                    }
                    let us = a.value(i);
                    // Validate the instant is representable, then split it into a day and a
                    // time of day (floored, so a pre-1970 instant keeps its clock).
                    DateTime::from_timestamp_micros(us)?;
                    let (day, clock) =
                        (us.div_euclid(MICROS_PER_DAY), us.rem_euclid(MICROS_PER_DAY));
                    move_day(day, n.value(i))?
                        .checked_mul(MICROS_PER_DAY)?
                        .checked_add(clock)
                })
                .collect();
            Ok(Arc::new(out.with_timezone_opt(tz.clone())))
        }
        DataType::Null => Ok(cast(dates, &DataType::Date32)?),
        other => Err(ExprError::UnknownType(format!("{func} on {other}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;

    fn date(y: i32, m: u32, d: u32) -> i32 {
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        (NaiveDate::from_ymd_opt(y, m, d).unwrap() - epoch).num_days() as i32
    }

    #[test]
    fn per_row_months_clamp_and_days_are_exact() {
        let dates: ArrayRef = Arc::new(Date32Array::from(vec![
            Some(date(2024, 1, 31)),
            Some(date(2024, 1, 31)),
            None,
        ]));
        let n: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), Some(-2), Some(1)]));
        let m = add_months(&dates, &n).unwrap();
        let m = m.as_primitive::<Date32Type>();
        assert_eq!(m.value(0), date(2024, 2, 29));
        assert_eq!(m.value(1), date(2023, 11, 30));
        assert!(m.is_null(2));
        let d = add_days(&dates, &n).unwrap();
        assert_eq!(d.as_primitive::<Date32Type>().value(1), date(2024, 1, 29));
    }

    #[test]
    fn a_zoned_day_is_a_local_day_across_dst() {
        // 2024-03-09 12:00 EST (17:00Z) plus one local day is 2024-03-10 12:00 EDT (16:00Z).
        let noon = NaiveDate::from_ymd_opt(2024, 3, 9)
            .unwrap()
            .and_hms_opt(17, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_micros();
        let ts: ArrayRef =
            Arc::new(TimestampMicrosecondArray::from(vec![noon]).with_timezone("America/New_York"));
        let one: ArrayRef = Arc::new(Int64Array::from(vec![1]));
        let out = add_days(&ts, &one).unwrap();
        let v = out
            .as_primitive::<arrow::datatypes::TimestampMicrosecondType>()
            .value(0);
        assert_eq!(v - noon, 23 * 3_600_000_000);
        assert_eq!(
            out.data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, Some("America/New_York".into()))
        );
    }
}
