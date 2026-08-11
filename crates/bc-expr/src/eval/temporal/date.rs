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

    // A text column is parsed as a timestamp first, uniformly.
    //
    // Half of this function already did that as a side effect of how it computed:
    // `dayname`, `last_day`, `is_leap_year` and friends cast to Timestamp(µs) before
    // doing their own work, so they accepted a string column, while `year`, `month` and
    // `second` handed the array straight to Arrow's `date_part` kernel and failed with
    // "Year does not support: Utf8". Which of the twenty-one functions worked on text was
    // therefore an accident of implementation, not a decision — and the ones that failed
    // are the common ones.
    //
    // Hoisting the cast here makes the whole family agree. It also makes a Spark or
    // pandas port work unchanged (`year('2016-07-30')` is legal Spark); DuckDB rejects the
    // string form outright, so this accepts *more* than the oracle rather than answering
    // differently from it, which is the only direction a compatibility convenience may go.
    if matches!(arr.data_type(), DataType::Utf8 | DataType::LargeUtf8) {
        let parsed = cast(arr, &DataType::Timestamp(TimeUnit::Microsecond, None))?;
        return eval_date(func, &parsed);
    }

    // An all-null column types as `Null`, which is a real type and not an error: it is what
    // `SELECT NULL AS x`, an all-`None` column, and a left join that matched nothing all
    // produce. Arithmetic, `cast` and `coalesce` already answer null for it, and the
    // aggregate, string, list and math families were each taught to; the temporal one still
    // handed the array to Arrow's kernel, which reported "Year does not support: Null" and
    // failed the query. Hoisting the cast here covers all twenty-one functions at once, the
    // same way the text coercion above does, and DuckDB returns NULL for every one.
    if matches!(arr.data_type(), DataType::Null) {
        let nulls = cast(arr, &DataType::Timestamp(TimeUnit::Microsecond, None))?;
        return eval_date(func, &nulls);
    }

    // `epoch` isn't a date-part: whole seconds since the Unix epoch, as Int64.
    //
    // The division MUST floor, not truncate toward zero. Arrow's
    // Timestamp(µs)→Timestamp(s) cast truncates, which put every sub-second instant
    // *before* 1970 in the wrong second: 1969-12-31T23:59:59.5 (−500_000 µs) came
    // back as second 0 — i.e. 1970-01-01, one second late, and indistinguishable
    // from 1970-01-01T00:00:00.5. Flooring agrees with `date_trunc('second', …)`
    // and with DuckDB (whose `epoch` is a DOUBLE: −0.5 → floor → −1).
    if let DateFunc::Epoch = func {
        use arrow::array::{Array, AsArray};
        use arrow::datatypes::Int64Type;

        let per_sec: i64 = match arr.data_type() {
            DataType::Timestamp(TimeUnit::Second, _) => 1,
            DataType::Timestamp(TimeUnit::Millisecond, _) => 1_000,
            DataType::Timestamp(TimeUnit::Microsecond, _) => 1_000_000,
            DataType::Timestamp(TimeUnit::Nanosecond, _) => 1_000_000_000,
            // Date32/Date64 carry no sub-second part, so seconds are exact — and the
            // second cast cannot overflow i64 the way a µs cast of a far-future date can.
            _ => {
                let secs = cast(arr, &DataType::Timestamp(TimeUnit::Second, None))?;
                return Ok(cast(&secs, &DataType::Int64)?);
            }
        };
        let raw = cast(arr, &DataType::Int64)?;
        let r = raw.as_primitive::<Int64Type>();
        let out: Int64Array = (0..r.len())
            .map(|i| (!r.is_null(i)).then(|| floor_div(r.value(i), per_sec)))
            .collect();
        return Ok(Arc::new(out));
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
        let dow =
            part.as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| ExprError::ExpectedType {
                    func: "isodow".into(),
                    want: "an Int32 day-of-week kernel result",
                    got: part.data_type().to_string(),
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
        let y =
            years
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| ExprError::ExpectedType {
                    func: "century/decade/millennium".into(),
                    want: "an Int32 year kernel result",
                    got: years.data_type().to_string(),
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
        let stamps = cast(
            &(Arc::new(out) as ArrayRef),
            &DataType::Timestamp(TimeUnit::Microsecond, None),
        )?;
        // A **date**, not a timestamp — `last_day` names a day, and DuckDB, Spark and
        // Polars all return a DATE for either input type. Returning midnight-of-that-day
        // as a timestamp was visible to a user as `2024-03-31 00:00:00` beside their
        // other date columns, changed the column's type in a `with_columns`, and forced
        // every differential test of it to cast DuckDB's answer to compare at all.
        return Ok(cast(&stamps, &DataType::Date32)?);
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
        // Sub-second units floor the raw micros directly (correct for pre-1970,
        // where flooring toward −∞ differs from truncating toward zero).
        match unit {
            "microsecond" | "microseconds" => return Some(us),
            "millisecond" | "milliseconds" => return Some(floor_div(us, 1_000) * 1_000),
            _ => {}
        }
        let dt = DateTime::from_timestamp_micros(us)?.naive_utc();
        let d = dt.date();
        let midnight = |nd: NaiveDate| nd.and_hms_opt(0, 0, 0);
        let out = match unit {
            "millennium" | "millenium" => midnight(NaiveDate::from_ymd_opt(
                (d.year() as i64).div_euclid(1000) as i32 * 1000,
                1,
                1,
            )?)?,
            "century" => midnight(NaiveDate::from_ymd_opt(
                (d.year() as i64).div_euclid(100) as i32 * 100,
                1,
                1,
            )?)?,
            "decade" => midnight(NaiveDate::from_ymd_opt(
                (d.year() as i64).div_euclid(10) as i32 * 10,
                1,
                1,
            )?)?,
            "year" => midnight(NaiveDate::from_ymd_opt(d.year(), 1, 1)?)?,
            // First month of the quarter: 1, 4, 7, 10.
            "quarter" => midnight(NaiveDate::from_ymd_opt(
                d.year(),
                (d.month0() / 3) * 3 + 1,
                1,
            )?)?,
            "month" => midnight(NaiveDate::from_ymd_opt(d.year(), d.month(), 1)?)?,
            // ISO week starts Monday; step back to the most recent Monday.
            "week" => {
                let back = d.weekday().num_days_from_monday() as i64;
                midnight(d.checked_sub_signed(chrono::Duration::try_days(back)?)?)?
            }
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
        "millennium"
            | "millenium"
            | "century"
            | "decade"
            | "year"
            | "quarter"
            | "month"
            | "week"
            | "day"
            | "hour"
            | "minute"
            | "second"
            | "millisecond"
            | "milliseconds"
            | "microsecond"
            | "microseconds"
    ) {
        return Err(ExprError::MissingArgument {
            func: "date_trunc".into(),
            arg: "unit (millennium|century|decade|year|quarter|month|week|day|\
                  hour|minute|second|millisecond|microsecond)",
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

/// Rewrite a DuckDB/C-style `strftime` format for chrono's formatter.
///
/// The only divergence that matters for a microsecond engine is `%f`: DuckDB (and
/// Python/C `strftime`) render it as microseconds zero-padded to **6** digits, but
/// chrono's `%f` renders **9** digits (nanoseconds), so `.123456` came back as
/// `.123456000`. chrono's `%6f` is the fixed-6-digit form, so map `%f → %6f`. `%%`
/// (an escaped percent) is left untouched.
fn strftime_format_for_chrono(format: &str) -> String {
    let mut out = String::with_capacity(format.len() + 2);
    let mut chars = format.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.peek() {
            Some('%') => {
                out.push_str("%%");
                chars.next();
            }
            Some('f') => {
                out.push_str("%6f");
                chars.next();
            }
            Some(&other) => {
                out.push('%');
                out.push(other);
                chars.next();
            }
            None => out.push('%'),
        }
    }
    out
}

/// Rewrite a DuckDB/C-style `strptime` format for chrono's parser.
///
/// DuckDB's `%f` parses the fractional-second digits as microseconds, right-padded
/// to 6 (`.5` → 500000, `.123` → 123000, `.123456` → 123456). chrono's `%f` instead
/// reads the digit run as a raw **nanosecond** integer (`.123456` → 123456 ns → 123
/// µs; `.5` → 5 ns → 0 µs) — a silently-scaled, ~1000× wrong result. chrono's `%.f`
/// has the correct decimal-fraction semantics but consumes its own leading dot, so
/// fold an immediately-preceding literal `.` into it (`.%f` → `%.f`). `%%` untouched.
fn strptime_format_for_chrono(format: &str) -> String {
    let mut out = String::with_capacity(format.len() + 2);
    let mut chars = format.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.peek() {
            Some('%') => {
                out.push_str("%%");
                chars.next();
            }
            Some('f') => {
                // chrono's `%.f` matches DuckDB's fraction scaling but eats the dot,
                // so drop a literal `.` we just emitted before it.
                if out.ends_with('.') {
                    out.pop();
                }
                out.push_str("%.f");
                chars.next();
            }
            Some(&other) => {
                out.push('%');
                out.push(other);
                chars.next();
            }
            None => out.push('%'),
        }
    }
    out
}

/// `strftime(ts, format)` — format each instant with a chrono/strftime `format`
/// string (→ Utf8). Works for Date and Timestamp; null → null. An invalid format
/// produces the same per-row behavior as chrono (the format is applied per value).
pub(crate) fn eval_strftime(arr: &ArrayRef, format: &str) -> Result<ArrayRef, ExprError> {
    use std::fmt::Write;

    use arrow::array::{Array, AsArray, StringBuilder};
    use arrow::datatypes::{TimeUnit, TimestampMicrosecondType};
    use chrono::DateTime;

    let format = strftime_format_for_chrono(format);
    let format = format.as_str();
    let micros = cast(arr, &DataType::Timestamp(TimeUnit::Microsecond, None))?;
    let ts = micros.as_primitive::<TimestampMicrosecondType>();
    let mut b = StringBuilder::with_capacity(ts.len(), ts.len() * format.len().max(8));
    let mut buf = String::new();
    for i in 0..ts.len() {
        if ts.is_null(i) {
            b.append_null();
            continue;
        }
        match DateTime::from_timestamp_micros(ts.value(i)) {
            Some(dt) => {
                // `DelayedFormat::Display` returns `Err` (not panics) for a specifier
                // that needs data a naive instant lacks — e.g. `%Z`/`%z` (no offset).
                // `to_string()` would turn that `Err` into a panic; format via `write!`
                // and map the error to null instead of crashing the batch.
                buf.clear();
                match write!(buf, "{}", dt.naive_utc().format(format)) {
                    Ok(()) => b.append_value(&buf),
                    Err(_) => b.append_null(),
                }
            }
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

    let format = strptime_format_for_chrono(format);
    let format = format.as_str();
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
    use bc_arrow::TypedOffset;

    // The scalar arithmetic — month-end clamping, exact days, and the checked-overflow-to-null
    // rule — lives in `bc_arrow::offset`, because the window frame and ASOF-tolerance paths in
    // `bc-runtime` ask the identical question and cannot reach this crate. What stays here is
    // the array plumbing and the type-specific errors.
    let offset = TypedOffset::new(months, days, micros);

    match arr.data_type() {
        DataType::Date32 => {
            if micros != 0 {
                return Err(ExprError::MissingArgument {
                    func: "offset_by".into(),
                    arg: "a sub-day offset (h/m/s) on a Date — cast to timestamp first",
                });
            }
            let a = arr.as_primitive::<Date32Type>();
            let out: Date32Array = (0..a.len())
                .map(|i| {
                    if a.is_null(i) {
                        return None;
                    }
                    offset
                        .shift_scalar(&DataType::Date32, a.value(i) as i64, 1)
                        .map(|d| d as i32)
                })
                .collect();
            Ok(Arc::new(out))
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            let ty = arr.data_type().clone();
            let m = cast(arr, &DataType::Int64)?;
            let a = m.as_primitive::<Int64Type>();
            let out: TimestampMicrosecondArray = (0..a.len())
                .map(|i| {
                    if a.is_null(i) {
                        return None;
                    }
                    offset.shift_scalar(&ty, a.value(i), 1)
                })
                .collect();
            Ok(Arc::new(out))
        }
        other => Err(ExprError::UnknownType(format!("offset_by on {other}"))),
    }
}

/// Map a type name (the wire contract) to an Arrow `DataType`.
///
/// Thin wrapper over the canonical [`bc_arrow::dtype_from_name`] table — the
/// single home for the name↔type vocabulary across every tier — surfacing the
/// `bc-expr` error on an unknown name.
pub(crate) fn parse_dtype(name: &str) -> Result<DataType, ExprError> {
    bc_arrow::dtype_from_name(name).ok_or_else(|| ExprError::UnknownType(name.to_string()))
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
                    // Checked: a far-out Date32 must not panic `NaiveDate + Duration`.
                    let d =
                        epoch.checked_add_signed(chrono::Duration::try_days(a.value(i) as i64)?)?;
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

/// Floor-divide `a` by positive `b` (rounds toward −∞, unlike Rust's `/`).
fn floor_div(a: i64, b: i64) -> i64 {
    let q = a / b;
    if (a % b != 0) && ((a < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

/// The input's timezone, when it has one. `window_start`/`window_buckets` label an
/// instant with the window containing it, so the label must carry the same timezone the
/// instant did: dropping it turns a `Timestamp(us, "UTC")` column into a naive one with
/// the right value and a type that no longer says what the value means, which anything
/// rendering a local time downstream reads as a wrong answer. The *unit* is deliberately
/// normalized to microseconds instead of preserved, matching `date_trunc` and the rest of
/// the temporal surface — the boundaries are microsecond quantities either way.
fn timestamp_zone(arr: &ArrayRef) -> Option<Arc<str>> {
    match arr.data_type() {
        DataType::Timestamp(_, tz) => tz.clone(),
        _ => None,
    }
}

/// `window_start(ts, width, origin)` — the start of the fixed-width tumbling window
/// containing each instant: `origin + ⌊(t−origin)/width⌋·width`. → Timestamp(us), keeping
/// the input's timezone.
pub(crate) fn eval_window_start(
    arr: &ArrayRef,
    width_micros: i64,
    origin_micros: i64,
) -> Result<ArrayRef, ExprError> {
    use arrow::array::{Array, AsArray};
    use arrow::datatypes::{Int64Type, TimeUnit};

    if width_micros <= 0 {
        return Err(ExprError::MissingArgument {
            func: "window_start".into(),
            arg: "width (must be > 0)",
        });
    }
    let zone = timestamp_zone(arr);
    let ts = cast(arr, &DataType::Timestamp(TimeUnit::Microsecond, None))?;
    let micros = cast(&ts, &DataType::Int64)?;
    let m = micros.as_primitive::<Int64Type>();
    let out: Int64Array = (0..m.len())
        .map(|i| {
            (!m.is_null(i)).then(|| {
                origin_micros + floor_div(m.value(i) - origin_micros, width_micros) * width_micros
            })
        })
        .collect();
    Ok(cast(
        &(Arc::new(out) as ArrayRef),
        &DataType::Timestamp(TimeUnit::Microsecond, zone),
    )?)
}

/// `window_buckets(ts, width, slide)` — the starts of every sliding window that
/// contains each instant, as a `List<Timestamp(us)>`. A window with start `s = k·slide`
/// contains `t` iff `s ≤ t < s+width`, so `k ∈ (⌊(t−width)/slide⌋, ⌊t/slide⌋]`.
pub(crate) fn eval_window_buckets(
    arr: &ArrayRef,
    width_micros: i64,
    slide_micros: i64,
) -> Result<ArrayRef, ExprError> {
    use arrow::array::{Array, AsArray, Int64Builder, ListArray};
    use arrow::buffer::OffsetBuffer;
    use arrow::datatypes::{Field, Int64Type, TimeUnit};

    if width_micros <= 0 || slide_micros <= 0 {
        return Err(ExprError::MissingArgument {
            func: "window_buckets".into(),
            arg: "width/slide (must be > 0)",
        });
    }
    let zone = timestamp_zone(arr);
    let ts = cast(arr, &DataType::Timestamp(TimeUnit::Microsecond, None))?;
    let micros = cast(&ts, &DataType::Int64)?;
    let m = micros.as_primitive::<Int64Type>();
    let mut values = Int64Builder::new();
    let mut lengths: Vec<usize> = Vec::with_capacity(m.len());
    for i in 0..m.len() {
        if m.is_null(i) {
            lengths.push(0);
            continue;
        }
        let t = m.value(i);
        let k_hi = floor_div(t, slide_micros);
        let k_lo = floor_div(t - width_micros, slide_micros) + 1;
        let mut n = 0usize;
        let mut k = k_lo;
        while k <= k_hi {
            values.append_value(k * slide_micros);
            n += 1;
            k += 1;
        }
        lengths.push(n);
    }
    let bucket_type = DataType::Timestamp(TimeUnit::Microsecond, zone);
    let child = cast(&(Arc::new(values.finish()) as ArrayRef), &bucket_type)?;
    let field = Arc::new(Field::new("item", bucket_type, true));
    let offsets = OffsetBuffer::from_lengths(lengths);
    Ok(Arc::new(ListArray::new(field, offsets, child, None)))
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
    fn epoch_floors_for_pre_1970_subsecond() {
        // Regression: a µs→s cast truncates toward zero, so a sub-second instant just
        // before 1970 wrongly landed in second 0. epoch must FLOOR (match date_trunc
        // and DuckDB's DOUBLE epoch): −500_000 µs → second −1, not 0.
        let arr: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![
            Some(-500_000), // 1969-12-31T23:59:59.5 → −1
            Some(-1),       // 1969-12-31T23:59:59.999999 → −1
            Some(-1_000_000),
            Some(500_000), //  1970-01-01T00:00:00.5 → 0
            Some(0),
            None,
        ]));
        let out = eval_date(DateFunc::Epoch, &arr).unwrap();
        let e = out.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(e.value(0), -1);
        assert_eq!(e.value(1), -1);
        assert_eq!(e.value(2), -1);
        assert_eq!(e.value(3), 0);
        assert_eq!(e.value(4), 0);
        assert!(e.is_null(5));
    }

    #[test]
    fn offset_huge_interval_nulls_instead_of_panicking() {
        use arrow::datatypes::TimestampMicrosecondType;
        // Regression: `Duration::days(days)` and `NaiveDate + Duration` panic on
        // overflow. A huge offset (or a far-out base date) must yield null, not crash.
        let arr: ArrayRef = Arc::new(Date32Array::from(vec![date(2024, 1, 15)]));
        let out = eval_date_offset(&arr, 0, i64::MAX, 0).unwrap();
        assert!(out.as_primitive::<Date32Type>().is_null(0));

        let ts: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![Some(0i64)]));
        let out = eval_date_offset(&ts, 0, i64::MAX, 0).unwrap();
        assert!(out.as_primitive::<TimestampMicrosecondType>().is_null(0));

        // A far-future Date32 value the epoch offset can't represent → null, no panic.
        let far: ArrayRef = Arc::new(Date32Array::from(vec![i32::MAX]));
        let out = eval_date_offset(&far, 0, 1, 0).unwrap();
        assert!(out.as_primitive::<Date32Type>().is_null(0));
    }

    #[test]
    fn add_months_far_date_nulls_instead_of_panicking() {
        use arrow::array::Int64Array as I64;
        let far: ArrayRef = Arc::new(Date32Array::from(vec![i32::MAX]));
        let months: ArrayRef = Arc::new(I64::from(vec![1i64]));
        let out = add_months(&far, &months).unwrap();
        assert!(out.as_primitive::<Date32Type>().is_null(0));
    }

    #[test]
    fn strftime_unsupported_specifier_nulls_instead_of_panicking() {
        // %Z/%z need a timezone offset a naive instant lacks; chrono's Display errors,
        // and to_string() would panic. We must produce null for that row, not crash.
        let arr: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![Some(0i64), None]));
        let out = eval_strftime(&arr, "%Z").unwrap();
        let s = out.as_string::<i32>();
        assert!(s.is_null(0));
        assert!(s.is_null(1));
        // A valid format still works on the same path.
        let ok = eval_strftime(&arr, "%Y-%m-%d").unwrap();
        assert_eq!(ok.as_string::<i32>().value(0), "1970-01-01");
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
    fn strftime_subsecond_f_is_microseconds_not_nanoseconds() {
        // Regression: chrono's `%f` renders 9 digits (nanoseconds), so a µs engine's
        // `.123456` came back as `.123456000`. DuckDB/Python render 6-digit micros.
        let arr: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![
            Some(1_708_004_730_123_456), // 2024-02-15 13:45:30.123456
            Some(-1_000_000),            // 1969-12-31 23:59:59.000000
            None,
        ]));
        let out = eval_strftime(&arr, "%Y-%m-%d %H:%M:%S.%f").unwrap();
        let s = out.as_string::<i32>();
        assert_eq!(s.value(0), "2024-02-15 13:45:30.123456");
        assert_eq!(s.value(1), "1969-12-31 23:59:59.000000");
        assert!(s.is_null(2));
        // An escaped `%%f` stays a literal `f`, not a fraction.
        let lit = eval_strftime(&arr, "%Y%%f").unwrap();
        assert_eq!(lit.as_string::<i32>().value(0), "2024%f");
    }

    #[test]
    fn strptime_subsecond_f_scales_as_microseconds_not_nanoseconds() {
        use arrow::array::StringArray;
        use arrow::datatypes::TimestampMicrosecondType;
        // Regression: chrono's `%f` reads the digit run as raw nanoseconds, so
        // `.123456` parsed to 123 µs, `.5` to 0 µs. DuckDB scales it as a decimal
        // fraction → microseconds (.5 → 500000, .123 → 123000, .123456 → 123456).
        let arr: ArrayRef = Arc::new(StringArray::from(vec![
            Some("2024-02-15 13:45:30.123456"),
            Some("2024-02-15 13:45:30.5"),
            Some("2024-02-15 13:45:30.123"),
        ]));
        let out = eval_strptime(&arr, "%Y-%m-%d %H:%M:%S.%f").unwrap();
        let ts = out.as_primitive::<TimestampMicrosecondType>();
        let frac = |i: usize| ts.value(i).rem_euclid(1_000_000);
        assert_eq!(frac(0), 123_456);
        assert_eq!(frac(1), 500_000);
        assert_eq!(frac(2), 123_000);
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

    #[test]
    fn date_trunc_supports_all_duckdb_units() {
        use arrow::array::AsArray;
        use arrow::datatypes::{TimeUnit, TimestampMicrosecondType};
        use chrono::NaiveDate;

        let ts = |y, mo, d, h, mi, s, us: u32| -> i64 {
            NaiveDate::from_ymd_opt(y, mo, d)
                .unwrap()
                .and_hms_micro_opt(h, mi, s, us)
                .unwrap()
                .and_utc()
                .timestamp_micros()
        };
        let midnight = |y, mo, d| ts(y, mo, d, 0, 0, 0, 0);

        // Two instants: a post-1970 sub-second time and a pre-1970 one (flooring
        // toward −∞ is what makes the sub-second units correct before the epoch).
        let vals = vec![
            Some(ts(2024, 2, 15, 13, 45, 30, 123_456)),
            Some(ts(1969, 6, 15, 13, 45, 30, 500_000)),
            None,
        ];
        let arr: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vals));

        // (unit, expected[0], expected[1]) — expectations match DuckDB date_trunc.
        let cases: &[(&str, i64, i64)] = &[
            ("quarter", midnight(2024, 1, 1), midnight(1969, 4, 1)),
            ("week", midnight(2024, 2, 12), midnight(1969, 6, 9)), // Mondays
            ("decade", midnight(2020, 1, 1), midnight(1960, 1, 1)),
            ("century", midnight(2000, 1, 1), midnight(1900, 1, 1)),
            ("millennium", midnight(2000, 1, 1), midnight(1000, 1, 1)),
            (
                "millisecond",
                ts(2024, 2, 15, 13, 45, 30, 123_000),
                ts(1969, 6, 15, 13, 45, 30, 500_000),
            ),
            (
                "microsecond",
                ts(2024, 2, 15, 13, 45, 30, 123_456),
                ts(1969, 6, 15, 13, 45, 30, 500_000),
            ),
        ];
        for (unit, e0, e1) in cases {
            let out = eval_date_trunc(&arr, unit).unwrap();
            assert_eq!(
                out.data_type(),
                &DataType::Timestamp(TimeUnit::Microsecond, None)
            );
            let t = out.as_primitive::<TimestampMicrosecondType>();
            assert_eq!(t.value(0), *e0, "unit={unit} row0");
            assert_eq!(t.value(1), *e1, "unit={unit} row1");
            assert!(t.is_null(2), "unit={unit} null row");
        }

        // An unknown unit still errors cleanly (no silent null-out).
        assert!(eval_date_trunc(&arr, "fortnight").is_err());
    }

    #[test]
    fn window_start_keeps_the_inputs_timezone() {
        // A window label is a statement about the same instant the input was, so it must
        // carry the same timezone. Dropping it left the value right and the type lying.
        use arrow::array::TimestampMicrosecondArray;
        use arrow::datatypes::TimeUnit;
        let arr: ArrayRef =
            Arc::new(TimestampMicrosecondArray::from(vec![1_000i64, 250]).with_timezone("UTC"));
        let out = eval_window_start(&arr, 100, 0).unwrap();
        assert_eq!(
            out.data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
        );
    }

    #[test]
    fn window_start_leaves_a_naive_column_naive() {
        use arrow::array::TimestampMicrosecondArray;
        use arrow::datatypes::TimeUnit;
        let arr: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![1_000i64]));
        let out = eval_window_start(&arr, 100, 0).unwrap();
        assert_eq!(
            out.data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, None)
        );
    }

    #[test]
    fn window_buckets_keep_the_inputs_timezone_on_the_list_item() {
        use arrow::array::TimestampMicrosecondArray;
        use arrow::datatypes::{Field, TimeUnit};
        let arr: ArrayRef =
            Arc::new(TimestampMicrosecondArray::from(vec![1_000i64]).with_timezone("UTC"));
        let out = eval_window_buckets(&arr, 200, 100).unwrap();
        let item = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
        assert_eq!(
            out.data_type(),
            &DataType::List(Arc::new(Field::new("item", item, true)))
        );
    }

    #[test]
    fn window_start_floors_to_width_and_handles_negative() {
        use arrow::datatypes::{TimeUnit, TimestampMicrosecondType};
        // width = 100us; 250→200, 100→100, 99→0, -1→-100 (floored), null→null.
        let arr: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![
            Some(250),
            Some(100),
            Some(99),
            Some(-1),
            None,
        ]));
        let out = eval_window_start(&arr, 100, 0).unwrap();
        let ts = out.as_primitive::<TimestampMicrosecondType>();
        assert_eq!(
            out.data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, None)
        );
        assert_eq!(ts.value(0), 200);
        assert_eq!(ts.value(1), 100);
        assert_eq!(ts.value(2), 0);
        assert_eq!(ts.value(3), -100);
        assert!(ts.is_null(4));
    }

    #[test]
    fn window_buckets_emits_overlapping_windows() {
        use arrow::array::AsArray;
        use arrow::datatypes::TimestampMicrosecondType;
        // width=100, slide=50 → 2 windows per row. t=120 ∈ windows [50,150) and
        // [100,200): starts {50, 100}. t=0 ∈ only [0,100) (start 0; window [-50,50)
        // also contains 0 → start -50). So t=0 → {-50, 0}.
        let arr: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![
            Some(120),
            Some(0),
            None,
        ]));
        let out = eval_window_buckets(&arr, 100, 50).unwrap();
        let list = out.as_list::<i32>();
        let row0 = list.value(0);
        let v0 = row0.as_primitive::<TimestampMicrosecondType>();
        assert_eq!(
            (0..v0.len()).map(|i| v0.value(i)).collect::<Vec<_>>(),
            vec![50, 100]
        );
        let row1 = list.value(1);
        let v1 = row1.as_primitive::<TimestampMicrosecondType>();
        assert_eq!(
            (0..v1.len()).map(|i| v1.value(i)).collect::<Vec<_>>(),
            vec![-50, 0]
        );
        assert_eq!(list.value(2).len(), 0); // null input → empty list
    }
}
