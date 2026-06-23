//! Date/time evaluation for `Expr::Date`/`DateTrunc`, dtype parsing, and the
//! month-shift used by `BinaryOp::AddMonths` (split out of `lib.rs`).

use std::sync::Arc;

use arrow::array::{ArrayRef, Date32Array, Int64Array, TimestampMicrosecondArray};
use arrow::compute::cast;
use arrow::datatypes::DataType;

use crate::{DateFunc, ExprError};

/// Evaluate a date/time field extraction (→ Int64, preserving nulls).
pub(crate) fn eval_date(func: DateFunc, arr: &ArrayRef) -> Result<ArrayRef, ExprError> {
    use arrow::compute::kernels::temporal::DatePart;
    use arrow::datatypes::TimeUnit;

    // `epoch` isn't a date-part: truncate to whole seconds then read as Int64
    // (seconds since the Unix epoch), matching DuckDB. Works for Date and Timestamp.
    if let DateFunc::Epoch = func {
        let secs = cast(arr, &DataType::Timestamp(TimeUnit::Second, None))?;
        return Ok(cast(&secs, &DataType::Int64)?);
    }

    // `dayname`/`monthname` return strings (chrono %A / %B), not date-parts. Cast to
    // Timestamp(Microsecond) then format each non-null instant; null → null.
    if matches!(func, DateFunc::Dayname | DateFunc::Monthname) {
        use arrow::array::{Array, AsArray, StringBuilder};
        use chrono::DateTime;
        let micros = cast(arr, &DataType::Timestamp(TimeUnit::Microsecond, None))?;
        let ts = micros.as_primitive::<arrow::datatypes::TimestampMicrosecondType>();
        let mut b = StringBuilder::with_capacity(ts.len(), ts.len() * 8);
        for i in 0..ts.len() {
            if ts.is_null(i) {
                b.append_null();
                continue;
            }
            match DateTime::from_timestamp_micros(ts.value(i)) {
                Some(dt) => {
                    let naive = dt.naive_utc();
                    let fmt = match func {
                        DateFunc::Dayname => naive.format("%A"),
                        DateFunc::Monthname => naive.format("%B"),
                        _ => unreachable!("matched dayname/monthname above"),
                    };
                    b.append_value(fmt.to_string());
                }
                None => b.append_null(),
            }
        }
        return Ok(Arc::new(b.finish()));
    }

    // `isodow` (Monday = 1 … Sunday = 7): Arrow's DayOfWeekMonday0 gives Monday = 0
    // … Sunday = 6, so add 1. Widen to Int64; nulls propagate.
    if let DateFunc::Isodow = func {
        use arrow::array::{Array, Int32Array};
        let part = arrow::compute::kernels::temporal::date_part(arr, DatePart::DayOfWeekMonday0)?;
        let dow = part.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
            ExprError::ExpectedString {
                func: "isodow".into(),
                got: part.data_type().to_string(),
            }
        })?;
        let out: Int64Array = (0..dow.len())
            .map(|i| (!dow.is_null(i)).then(|| dow.value(i) as i64 + 1))
            .collect();
        return Ok(Arc::new(out));
    }

    // `century`/`decade`/`millennium` are derived from the extracted year (DuckDB):
    //   century    = (Y - 1).div_euclid(100) + 1   (e.g. 2021 → 21, 1999/2000 → 20)
    //   decade     = Y.div_euclid(10)              (e.g. 2021 → 202)
    //   millennium = (Y - 1).div_euclid(1000) + 1  (e.g. 2021 → 3, 2000 → 2)
    if matches!(
        func,
        DateFunc::Century | DateFunc::Decade | DateFunc::Millennium
    ) {
        use arrow::array::{Array, Int32Array};
        let years = arrow::compute::kernels::temporal::date_part(arr, DatePart::Year)?;
        let y = years.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
            ExprError::ExpectedString {
                func: "century/decade/millennium".into(),
                got: years.data_type().to_string(),
            }
        })?;
        let out: Int64Array = (0..y.len())
            .map(|i| {
                (!y.is_null(i)).then(|| {
                    let yr = y.value(i) as i64;
                    match func {
                        DateFunc::Century => (yr - 1).div_euclid(100) + 1,
                        DateFunc::Decade => yr.div_euclid(10),
                        DateFunc::Millennium => (yr - 1).div_euclid(1000) + 1,
                        _ => unreachable!("matched century/decade/millennium above"),
                    }
                })
            })
            .collect();
        return Ok(Arc::new(out));
    }

    // `last_day` returns the last day of the instant's month at 00:00:00, as a
    // Timestamp(Microsecond) (mirrors how `date_trunc` builds its result). Null →
    // null. Computed via chrono: first day of the next month minus one day.
    if let DateFunc::LastDay = func {
        use arrow::array::{Array, AsArray};
        use arrow::datatypes::{Int64Type, TimeUnit};
        use chrono::{DateTime, Datelike, NaiveDate};

        let ts = cast(arr, &DataType::Timestamp(TimeUnit::Microsecond, None))?;
        let micros = cast(&ts, &DataType::Int64)?;
        let m = micros.as_primitive::<Int64Type>();

        let last_day = |us: i64| -> Option<i64> {
            let d = DateTime::from_timestamp_micros(us)?.naive_utc().date();
            let (y, mo) = (d.year(), d.month());
            // First day of the following month, then step back one day.
            let (ny, nmo) = if mo == 12 { (y + 1, 1) } else { (y, mo + 1) };
            let first_next = NaiveDate::from_ymd_opt(ny, nmo, 1)?;
            let last = first_next.pred_opt()?;
            last.and_hms_opt(0, 0, 0)?
                .and_utc()
                .timestamp_micros()
                .into()
        };

        let out: Int64Array = (0..m.len())
            .map(|i| {
                if m.is_null(i) {
                    None
                } else {
                    last_day(m.value(i))
                }
            })
            .collect();
        return Ok(cast(
            &(Arc::new(out) as ArrayRef),
            &DataType::Timestamp(TimeUnit::Microsecond, None),
        )?);
    }

    // `is_leap_year` (→ Bool), `days_in_month` (→ Int64), `iso_year` (→ Int64):
    // calendar-derived via chrono.
    if matches!(
        func,
        DateFunc::IsLeapYear | DateFunc::DaysInMonth | DateFunc::IsoYear
    ) {
        use arrow::array::{Array, AsArray, BooleanArray};
        use arrow::datatypes::{Int64Type, TimeUnit};
        use chrono::{DateTime, Datelike, NaiveDate};

        let ts = cast(arr, &DataType::Timestamp(TimeUnit::Microsecond, None))?;
        let micros = cast(&ts, &DataType::Int64)?;
        let m = micros.as_primitive::<Int64Type>();

        if let DateFunc::IsLeapYear = func {
            let out: BooleanArray = (0..m.len())
                .map(|i| {
                    (!m.is_null(i)).then(|| {
                        DateTime::from_timestamp_micros(m.value(i)).is_some_and(|dt| {
                            NaiveDate::from_ymd_opt(dt.naive_utc().year(), 2, 29).is_some()
                        })
                    })
                })
                .collect();
            return Ok(Arc::new(out));
        }

        let out: Int64Array = (0..m.len())
            .map(|i| {
                if m.is_null(i) {
                    return None;
                }
                let d = DateTime::from_timestamp_micros(m.value(i))?
                    .naive_utc()
                    .date();
                match func {
                    DateFunc::DaysInMonth => {
                        let (y, mo) = (d.year(), d.month());
                        let (ny, nmo) = if mo == 12 { (y + 1, 1) } else { (y, mo + 1) };
                        let first_next = NaiveDate::from_ymd_opt(ny, nmo, 1)?;
                        let first_this = NaiveDate::from_ymd_opt(y, mo, 1)?;
                        Some((first_next - first_this).num_days())
                    }
                    DateFunc::IsoYear => Some(d.iso_week().year() as i64),
                    _ => unreachable!("matched is_leap_year/days_in_month/iso_year above"),
                }
            })
            .collect();
        return Ok(Arc::new(out));
    }

    let part = match func {
        DateFunc::Year => DatePart::Year,
        DateFunc::Month => DatePart::Month,
        DateFunc::Day => DatePart::Day,
        DateFunc::Hour => DatePart::Hour,
        DateFunc::Minute => DatePart::Minute,
        DateFunc::Second => DatePart::Second,
        DateFunc::Quarter => DatePart::Quarter,
        DateFunc::Week => DatePart::Week,
        DateFunc::DayOfWeek => DatePart::DayOfWeekSunday0,
        DateFunc::DayOfYear => DatePart::DayOfYear,
        DateFunc::Epoch => unreachable!("handled above"),
        DateFunc::Dayname | DateFunc::Monthname => unreachable!("handled above"),
        DateFunc::Isodow | DateFunc::Century | DateFunc::Decade => {
            unreachable!("handled above")
        }
        DateFunc::Millennium | DateFunc::LastDay => unreachable!("handled above"),
        DateFunc::IsLeapYear | DateFunc::DaysInMonth | DateFunc::IsoYear => {
            unreachable!("handled above")
        }
    };
    // `date_part` yields Int32; widen to Int64 for a uniform numeric type.
    let i32s = arrow::compute::kernels::temporal::date_part(arr, part)?;
    Ok(cast(&i32s, &DataType::Int64)?)
}

/// `date_trunc(unit, ts)` — truncate each timestamp to the start of `unit`,
/// returning Timestamp(microsecond). Calendar-correct via chrono.
pub(crate) fn eval_date_trunc(arr: &ArrayRef, unit: &str) -> Result<ArrayRef, ExprError> {
    use arrow::array::{Array, AsArray};
    use arrow::datatypes::{Int64Type, TimeUnit};
    use chrono::{DateTime, Datelike, NaiveDate, Timelike};

    let ts = cast(arr, &DataType::Timestamp(TimeUnit::Microsecond, None))?;
    let micros = cast(&ts, &DataType::Int64)?;
    let m = micros.as_primitive::<Int64Type>();

    let truncate = |us: i64| -> Option<i64> {
        let dt = DateTime::from_timestamp_micros(us)?.naive_utc();
        let d = dt.date();
        let out = match unit {
            "year" => NaiveDate::from_ymd_opt(d.year(), 1, 1)?.and_hms_opt(0, 0, 0)?,
            "month" => NaiveDate::from_ymd_opt(d.year(), d.month(), 1)?.and_hms_opt(0, 0, 0)?,
            "day" => d.and_hms_opt(0, 0, 0)?,
            "hour" => d.and_hms_opt(dt.hour(), 0, 0)?,
            "minute" => d.and_hms_opt(dt.hour(), dt.minute(), 0)?,
            "second" => d.and_hms_opt(dt.hour(), dt.minute(), dt.second())?,
            _ => return None,
        };
        Some(out.and_utc().timestamp_micros())
    };

    // Reject an unknown unit up front (so a typo errors rather than nulls out).
    if !matches!(
        unit,
        "year" | "month" | "day" | "hour" | "minute" | "second"
    ) {
        return Err(ExprError::MissingArgument {
            func: "date_trunc".into(),
            arg: "unit (year|month|day|hour|minute|second)",
        });
    }
    let out: Int64Array = (0..m.len())
        .map(|i| {
            if m.is_null(i) {
                None
            } else {
                truncate(m.value(i))
            }
        })
        .collect();
    Ok(cast(
        &(Arc::new(out) as ArrayRef),
        &DataType::Timestamp(TimeUnit::Microsecond, None),
    )?)
}

/// `strftime(ts, format)` — format each instant with a chrono/strftime `format`
/// string (→ Utf8). Works for Date and Timestamp; null → null. An invalid format
/// produces the same per-row behavior as chrono (the format is applied per value).
pub(crate) fn eval_strftime(arr: &ArrayRef, format: &str) -> Result<ArrayRef, ExprError> {
    use arrow::array::{Array, AsArray, StringBuilder};
    use arrow::datatypes::{TimeUnit, TimestampMicrosecondType};
    use chrono::DateTime;

    let micros = cast(arr, &DataType::Timestamp(TimeUnit::Microsecond, None))?;
    let ts = micros.as_primitive::<TimestampMicrosecondType>();
    let mut b = StringBuilder::with_capacity(ts.len(), ts.len() * format.len().max(8));
    for i in 0..ts.len() {
        if ts.is_null(i) {
            b.append_null();
            continue;
        }
        match DateTime::from_timestamp_micros(ts.value(i)) {
            Some(dt) => b.append_value(dt.naive_utc().format(format).to_string()),
            None => b.append_null(),
        }
    }
    Ok(Arc::new(b.finish()))
}

/// `strptime(s, format)` — parse each string with a chrono/strftime `format` into a
/// Timestamp(microsecond). A value that does not match the format (or a null) yields
/// NULL rather than erroring — DuckDB `try_strptime` semantics, the safe-ingest
/// behavior for dirty source columns. A date-only format (no time fields) parses at
/// midnight, matching DuckDB (`strptime` always returns a TIMESTAMP).
pub(crate) fn eval_strptime(arr: &ArrayRef, format: &str) -> Result<ArrayRef, ExprError> {
    use arrow::array::{Array, AsArray};
    use chrono::{NaiveDate, NaiveDateTime};

    let strings = cast(arr, &DataType::Utf8)?;
    let s = strings.as_string::<i32>();
    let out: TimestampMicrosecondArray = (0..s.len())
        .map(|i| {
            if s.is_null(i) {
                return None;
            }
            let v = s.value(i);
            // Try a full datetime first; fall back to a date-only format at midnight.
            let dt = NaiveDateTime::parse_from_str(v, format).ok().or_else(|| {
                NaiveDate::parse_from_str(v, format)
                    .ok()
                    .and_then(|d| d.and_hms_opt(0, 0, 0))
            });
            dt.map(|d| d.and_utc().timestamp_micros())
        })
        .collect();
    Ok(Arc::new(out))
}

/// `offset_by` — shift a Date32/Timestamp by `months` calendar months (end-of-month
/// clamping), `days` exact days, and `micros` exact microseconds. Type-preserving;
/// `micros != 0` on a Date32 errors. Null → null.
pub(crate) fn eval_date_offset(
    arr: &ArrayRef,
    months: i64,
    days: i64,
    micros: i64,
) -> Result<ArrayRef, ExprError> {
    use arrow::array::{Array, AsArray};
    use arrow::datatypes::{Date32Type, Int64Type, TimeUnit};
    use chrono::{DateTime, Duration, Months, NaiveDate};

    let shift_months = |d: NaiveDate| -> Option<NaiveDate> {
        if months >= 0 {
            d.checked_add_months(Months::new(months as u32))
        } else {
            d.checked_sub_months(Months::new((-months) as u32))
        }
    };

    match arr.data_type() {
        DataType::Date32 => {
            if micros != 0 {
                return Err(ExprError::MissingArgument {
                    func: "offset_by".into(),
                    arg: "a sub-day offset (h/m/s) on a Date — cast to timestamp first",
                });
            }
            let a = arr.as_primitive::<Date32Type>();
            let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
            let out: Date32Array = (0..a.len())
                .map(|i| {
                    if a.is_null(i) {
                        return None;
                    }
                    let d = epoch + Duration::days(a.value(i) as i64);
                    shift_months(d)
                        .and_then(|d| d.checked_add_signed(Duration::days(days)))
                        .map(|nd| (nd - epoch).num_days() as i32)
                })
                .collect();
            Ok(Arc::new(out))
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            let m = cast(arr, &DataType::Int64)?;
            let a = m.as_primitive::<Int64Type>();
            let out: TimestampMicrosecondArray = (0..a.len())
                .map(|i| {
                    if a.is_null(i) {
                        return None;
                    }
                    let dt = DateTime::from_timestamp_micros(a.value(i))?.naive_utc();
                    let shifted = shift_months(dt.date())?.and_time(dt.time());
                    shifted
                        .checked_add_signed(Duration::days(days))?
                        .checked_add_signed(Duration::microseconds(micros))
                        .map(|x| x.and_utc().timestamp_micros())
                })
                .collect();
            Ok(Arc::new(out))
        }
        other => Err(ExprError::UnknownType(format!("offset_by on {other}"))),
    }
}

/// Map a type name (the wire contract) to an Arrow `DataType`.
pub(crate) fn parse_dtype(name: &str) -> Result<DataType, ExprError> {
    Ok(match name {
        "int64" | "long" => DataType::Int64,
        "int32" | "int" => DataType::Int32,
        "float64" | "double" => DataType::Float64,
        "float32" | "float" => DataType::Float32,
        "bool" | "boolean" => DataType::Boolean,
        "string" | "utf8" => DataType::Utf8,
        "date" | "date32" => DataType::Date32,
        "timestamp" | "datetime" => {
            DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None)
        }
        other => return Err(ExprError::UnknownType(other.to_string())),
    })
}

/// Add `months[i]` calendar months to each Date32/Timestamp `dates[i]` (negative
/// to subtract), preserving the input type. Null on either side → null. Month
/// overflow clamps to the last valid day (chrono `checked_add_months` semantics).
pub(crate) fn add_months(dates: &ArrayRef, months: &ArrayRef) -> Result<ArrayRef, ExprError> {
    use arrow::array::{Array, AsArray};
    use arrow::datatypes::{Date32Type, Int64Type, TimeUnit};
    use chrono::{DateTime, Months, NaiveDate};

    let m = cast(months, &DataType::Int64)?;
    let m = m.as_primitive::<Int64Type>();
    let shift = |d: NaiveDate, n: i64| -> Option<NaiveDate> {
        if n >= 0 {
            d.checked_add_months(Months::new(n as u32))
        } else {
            d.checked_sub_months(Months::new((-n) as u32))
        }
    };
    match dates.data_type() {
        DataType::Date32 => {
            let a = dates.as_primitive::<Date32Type>();
            let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
            let out: Date32Array = (0..a.len())
                .map(|i| {
                    if a.is_null(i) || m.is_null(i) {
                        return None;
                    }
                    let d = epoch + chrono::Duration::days(a.value(i) as i64);
                    shift(d, m.value(i)).map(|nd| (nd - epoch).num_days() as i32)
                })
                .collect();
            Ok(Arc::new(out))
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            let micros = cast(dates, &DataType::Int64)?;
            let a = micros.as_primitive::<Int64Type>();
            let out: TimestampMicrosecondArray = (0..a.len())
                .map(|i| {
                    if a.is_null(i) || m.is_null(i) {
                        return None;
                    }
                    let dt = DateTime::from_timestamp_micros(a.value(i))?.naive_utc();
                    shift(dt.date(), m.value(i))
                        .map(|nd| nd.and_time(dt.time()).and_utc().timestamp_micros())
                })
                .collect();
            Ok(Arc::new(out))
        }
        other => Err(ExprError::UnknownType(format!("add_months on {other}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, AsArray, Date32Array};
    use arrow::datatypes::Date32Type;
    use chrono::NaiveDate;

    fn date(y: i32, m: u32, d: u32) -> i32 {
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        (NaiveDate::from_ymd_opt(y, m, d).unwrap() - epoch).num_days() as i32
    }

    #[test]
    fn offset_months_clamps_end_of_month() {
        // 2024-01-31 + 1 month → 2024-02-29 (clamp); + days is exact.
        let arr: ArrayRef = Arc::new(Date32Array::from(vec![Some(date(2024, 1, 31)), None]));
        let out = eval_date_offset(&arr, 1, 0, 0).unwrap();
        let o = out.as_primitive::<Date32Type>();
        assert_eq!(o.value(0), date(2024, 2, 29));
        assert!(o.is_null(1));
    }

    #[test]
    fn offset_negative_days() {
        let arr: ArrayRef = Arc::new(Date32Array::from(vec![date(2024, 3, 1)]));
        let out = eval_date_offset(&arr, 0, -1, 0).unwrap();
        assert_eq!(out.as_primitive::<Date32Type>().value(0), date(2024, 2, 29));
    }

    #[test]
    fn strptime_parses_datetime_and_nulls_bad() {
        use arrow::array::StringArray;
        use arrow::datatypes::TimestampMicrosecondType;
        use chrono::NaiveDate;

        let arr: ArrayRef = Arc::new(StringArray::from(vec![
            Some("2024-02-15 13:45:30"),
            Some("not a date"),
            None,
        ]));
        let out = eval_strptime(&arr, "%Y-%m-%d %H:%M:%S").unwrap();
        let ts = out.as_primitive::<TimestampMicrosecondType>();
        let expected = NaiveDate::from_ymd_opt(2024, 2, 15)
            .unwrap()
            .and_hms_opt(13, 45, 30)
            .unwrap()
            .and_utc()
            .timestamp_micros();
        assert_eq!(ts.value(0), expected);
        assert!(ts.is_null(1), "unparseable string → null");
        assert!(ts.is_null(2), "null input → null");
    }

    #[test]
    fn strptime_date_only_format_parses_at_midnight() {
        use arrow::array::StringArray;
        use arrow::datatypes::TimestampMicrosecondType;
        use chrono::NaiveDate;

        let arr: ArrayRef = Arc::new(StringArray::from(vec![Some("2024-02-15")]));
        let out = eval_strptime(&arr, "%Y-%m-%d").unwrap();
        let ts = out.as_primitive::<TimestampMicrosecondType>();
        let expected = NaiveDate::from_ymd_opt(2024, 2, 15)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_micros();
        assert_eq!(ts.value(0), expected);
    }

    #[test]
    fn offset_subday_on_date_errors() {
        let arr: ArrayRef = Arc::new(Date32Array::from(vec![date(2024, 1, 1)]));
        assert!(eval_date_offset(&arr, 0, 0, 3_600_000_000).is_err());
    }
}
