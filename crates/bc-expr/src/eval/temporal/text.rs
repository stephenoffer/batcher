//! Text ↔ instant: `strftime` renders one, `strptime` reads one back.
//!
//! Split out of `date` because it is the one direction pair in this package that is not
//! about a *field* of an instant but about its whole textual form, and because both
//! halves share a problem the rest of the module does not have: **chrono's format
//! vocabulary is not DuckDB's**. Two of the differences are silent rather than loud, and
//! each cost a wrong answer rather than an error:
//!
//! * `%f` renders 9 digits in chrono and 6 in DuckDB, so a microsecond timestamp came
//!   back with three spurious zeros; and on the parsing side chrono reads the digit run
//!   as a raw nanosecond integer, which scales the value by ~1000.
//! * chrono's whole-value parsers each demand a *complete* date or a complete instant, so
//!   a format naming only some fields — `%Y`, `%Y-%m`, an hour bucket — parsed as NULL
//!   where DuckDB fills the rest in. `parse_partial` is that fill-in.
//!
//! Both format rewriters therefore translate before chrono sees the string, and neither
//! is a general dialect translation: each names exactly the specifiers that diverge and
//! passes everything else through.

use std::sync::Arc;

use arrow::array::{ArrayRef, TimestampMicrosecondArray};
use arrow::compute::cast;
use arrow::datatypes::DataType;

use crate::ExprError;

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
    use chrono::NaiveDateTime;

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
            // Try a full datetime first, then fill in whatever the format left unnamed.
            //
            // The second step used to be `NaiveDate::parse_from_str` at midnight, which is
            // both narrower and *wrong for the formats it did match*: chrono's date parser
            // ignores time fields, so `'2024-03-05 13'` with `%Y-%m-%d %H` parsed as the
            // date and threw the hour away, answering midnight. `parse_partial` subsumes
            // it — a date-only format defaults to midnight through the same path — and
            // keeps the hour.
            let dt = NaiveDateTime::parse_from_str(v, format)
                .ok()
                .or_else(|| parse_partial(v, format));
            dt.map(|d| d.and_utc().timestamp_micros())
        })
        .collect();
    Ok(Arc::new(out))
}

/// Parse a format that names only *some* of a timestamp's fields, defaulting the rest.
///
/// `strptime('1900', '%Y')` is a DuckDB query that answers `1900-01-01 00:00:00`, and every
/// coarser rollup key is written this way — `'2024-03'` with `%Y-%m`, an hour bucket with
/// `%Y-%m-%d %H`. chrono's two whole-value parsers cannot express it: `NaiveDate` demands a
/// complete calendar date and `NaiveDateTime` a complete instant, so both refuse and the
/// engine returned NULL for the entire column. Silently, because `strptime` is documented to
/// null what it cannot parse — which is right for a malformed *value* and wrong for a format
/// it simply could not represent.
///
/// `chrono::format::Parsed` is the level below those two: it collects whatever fields the
/// format named and leaves the rest unset, so the defaults can be supplied here. Month and
/// day default to 1 and the time to midnight, which is what DuckDB, Python's `strptime` and
/// Spark all fill in. A format naming *no* date field at all is still refused: `'12:30'`
/// with `%H:%M` has no year to invent, and picking one would be a fabricated instant rather
/// than a default.
fn parse_partial(value: &str, format: &str) -> Option<chrono::NaiveDateTime> {
    use chrono::format::{parse, Parsed, StrftimeItems};
    use chrono::NaiveDate;

    let mut parsed = Parsed::new();
    parse(&mut parsed, value, StrftimeItems::new(format)).ok()?;
    // A year is the one field with no sensible default. `year()` covers `%Y` and `%y`
    // alike (chrono resolves the century for the latter); an ISO week year is a different
    // field and is deliberately not accepted, since it pins neither a month nor a day.
    let year = parsed.year()?;
    NaiveDate::from_ymd_opt(year, parsed.month().unwrap_or(1), parsed.day().unwrap_or(1))?
        .and_hms_nano_opt(
            // chrono stores the hour as a 12-hour half plus an offset (so `%p` can set the
            // half independently); an absent half means the format named no hour at all.
            parsed.hour_div_12().unwrap_or(0) * 12 + parsed.hour_mod_12().unwrap_or(0),
            parsed.minute().unwrap_or(0),
            parsed.second().unwrap_or(0),
            parsed.nanosecond().unwrap_or(0),
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, AsArray};
    use chrono::NaiveDate;

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

    /// `strptime` with a format that names only some of a timestamp's fields.
    ///
    /// Every one of these is a query DuckDB answers and this kernel returned NULL for,
    /// because chrono's two whole-value parsers each demand a complete date or a complete
    /// instant. The failure was silent — `strptime` nulls what it cannot parse, which is
    /// right for a malformed value and wrong for a format it could not represent.
    #[test]
    fn strptime_fills_the_fields_a_partial_format_does_not_name() {
        let micros = |y, m, d, hh, mm, ss| {
            NaiveDate::from_ymd_opt(y, m, d)
                .unwrap()
                .and_hms_opt(hh, mm, ss)
                .unwrap()
                .and_utc()
                .timestamp_micros()
        };
        let cases: [(&str, &str, i64); 4] = [
            ("1900", "%Y", micros(1900, 1, 1, 0, 0, 0)),
            ("2024-03", "%Y-%m", micros(2024, 3, 1, 0, 0, 0)),
            ("2024-03-05 13", "%Y-%m-%d %H", micros(2024, 3, 5, 13, 0, 0)),
            (
                "2024-03-05 13:45",
                "%Y-%m-%d %H:%M",
                micros(2024, 3, 5, 13, 45, 0),
            ),
        ];
        for (value, format, want) in cases {
            let arr: ArrayRef = Arc::new(arrow::array::StringArray::from(vec![Some(value), None]));
            let out = eval_strptime(&arr, format).unwrap();
            let o = out
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap();
            assert_eq!(o.value(0), want, "{value:?} with {format:?}");
            assert!(o.is_null(1), "a null input must stay null");
        }
    }

    /// The complete formats must be unaffected: the partial parser is only ever reached
    /// after both whole-value parsers have refused, so it can turn a NULL into a value and
    /// never a value into a different one.
    #[test]
    fn strptime_still_nulls_what_does_not_match_and_still_parses_what_does() {
        let arr: ArrayRef = Arc::new(arrow::array::StringArray::from(vec![
            Some("2024-03-05"),
            Some("not a date"),
            Some("2024-13-05"), // an impossible month, not a partial format
            Some("2024-03-05 extra"),
        ]));
        let out = eval_strptime(&arr, "%Y-%m-%d").unwrap();
        let o = out
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();
        assert_eq!(
            o.value(0),
            NaiveDate::from_ymd_opt(2024, 3, 5)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc()
                .timestamp_micros()
        );
        assert!(o.is_null(1));
        assert!(o.is_null(2));
        assert!(o.is_null(3), "trailing input must not be silently accepted");
    }

    /// A format naming no year has no instant to default to, so it stays NULL rather than
    /// inventing one.
    #[test]
    fn strptime_refuses_a_format_with_no_year() {
        let arr: ArrayRef = Arc::new(arrow::array::StringArray::from(vec![Some("12:30")]));
        let out = eval_strptime(&arr, "%H:%M").unwrap();
        assert!(out
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .is_null(0));
    }
}
