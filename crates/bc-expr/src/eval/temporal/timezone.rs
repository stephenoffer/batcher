//! Time zones: `convert_timezone`, `replace_timezone`, and the local-time helpers the
//! calendar kernels use on a tz-aware column.
//!
//! An Arrow timestamp is a count since the Unix epoch *in UTC*; its zone, when it has one,
//! is a label saying which wall clock to read it as. Two operations follow from that and
//! they are not the same thing. **Converting** keeps the instant and asks what a clock in
//! another zone reads (DuckDB `AT TIME ZONE`, `convert_timezone`). **Replacing** keeps the
//! clock and asks which instant it denotes in another zone (Polars `replace_time_zone`,
//! pandas `tz_localize`). The second is the only one that can meet a wall clock that does
//! not exist (a DST gap) or exists twice (an overlap), and `Ambiguous` / `Nonexistent` are
//! the caller's choice of answer. The JIT compiles none of this; the interpreter is the
//! only path.
//!
//! Zone rules come from the IANA database compiled in through chrono-tz
//! ([`crate::TZDB_VERSION`]), and a column's own zone is parsed by Arrow's `Tz`, which also
//! reads a fixed offset such as `+05:30`.

use std::str::FromStr;
use std::sync::Arc;

use arrow::array::timezone::Tz;
use arrow::array::{Array, ArrayRef, AsArray, TimestampMicrosecondArray};
use arrow::compute::cast;
use arrow::datatypes::{DataType, TimeUnit, TimestampMicrosecondType};
use chrono::{DateTime, LocalResult, NaiveDateTime, Offset, TimeZone};

use crate::{Ambiguous, ExprError, Nonexistent};

/// Parse a zone name or fixed offset, naming the function on failure.
pub(crate) fn parse_zone(name: &str, func: &str) -> Result<Tz, ExprError> {
    Tz::from_str(name).map_err(|_| ExprError::InvalidArgument {
        func: func.to_string(),
        reason: format!("unknown time zone {name:?}; use an IANA name such as \"Europe/Paris\""),
    })
}

/// The zone a timestamp column is labelled with, when it has one.
pub(crate) fn column_zone(arr: &ArrayRef, func: &str) -> Result<Option<Tz>, ExprError> {
    match arr.data_type() {
        DataType::Timestamp(_, Some(tz)) => parse_zone(tz, func).map(Some),
        _ => Ok(None),
    }
}

/// The wall clock `zone` reads at the UTC instant `utc_micros`, as naive microseconds.
pub(crate) fn local_micros(utc_micros: i64, zone: &Tz) -> Option<i64> {
    let utc = DateTime::from_timestamp_micros(utc_micros)?.naive_utc();
    let offset = zone.offset_from_utc_datetime(&utc).fix().local_minus_utc();
    utc_micros.checked_add(i64::from(offset) * 1_000_000)
}

/// The UTC instant a wall clock (naive microseconds) denotes in `zone`, resolving a DST
/// overlap or gap by the two policies. `Ok(None)` is a null row; a `raise` policy errors.
pub(crate) fn localize(
    local: i64,
    zone: &Tz,
    ambiguous: Ambiguous,
    nonexistent: Nonexistent,
    func: &str,
) -> Result<Option<i64>, ExprError> {
    let Some(naive) = DateTime::from_timestamp_micros(local).map(|d| d.naive_utc()) else {
        return Ok(None);
    };
    match zone.offset_from_local_datetime(&naive) {
        LocalResult::Single(off) => Ok(utc_of(local, off.fix().local_minus_utc())),
        LocalResult::Ambiguous(early, late) => match ambiguous {
            Ambiguous::Null => Ok(None),
            Ambiguous::Earliest => Ok(utc_of(local, early.fix().local_minus_utc())),
            Ambiguous::Latest => Ok(utc_of(local, late.fix().local_minus_utc())),
            Ambiguous::Raise => Err(dst_error(
                func,
                &naive,
                zone,
                "is ambiguous (a DST overlap)",
            )),
        },
        LocalResult::None => match nonexistent {
            Nonexistent::Null => Ok(None),
            Nonexistent::ShiftForward => Ok(gap_end(&naive, zone)),
            Nonexistent::Raise => Err(dst_error(func, &naive, zone, "does not exist (a DST gap)")),
        },
    }
}

/// `local − offset`, the UTC instant of a wall clock read at a known offset.
fn utc_of(local: i64, offset_secs: i32) -> Option<i64> {
    local.checked_sub(i64::from(offset_secs) * 1_000_000)
}

fn dst_error(func: &str, naive: &NaiveDateTime, zone: &Tz, what: &str) -> ExprError {
    ExprError::InvalidArgument {
        func: func.to_string(),
        reason: format!(
            "the wall clock {naive} {what} in {zone}; pass ambiguous=/nonexistent= to choose \
             an answer instead of raising"
        ),
    }
}

/// The first instant after the DST gap that swallows the wall clock `naive`.
///
/// Inside a gap the wall clock read at the *pre*-transition offset lands after the
/// transition and read at the *post*-transition offset lands before it, so the transition
/// is bracketed by those two instants; the offset is piecewise constant, so a bisection on
/// whole seconds finds the first instant on the new offset. Transitions in the IANA data
/// fall on whole seconds.
fn gap_end(naive: &NaiveDateTime, zone: &Tz) -> Option<i64> {
    let local_secs = naive.and_utc().timestamp();
    let offset_at = |secs: i64| -> Option<i32> {
        let utc = DateTime::from_timestamp(secs, 0)?.naive_utc();
        Some(zone.offset_from_utc_datetime(&utc).fix().local_minus_utc())
    };
    // Two days either side bracket any real gap (the largest, Samoa's 2011 skip, is a day).
    let before = offset_at(local_secs - 86_400 * 2)?;
    let after = offset_at(local_secs + 86_400 * 2)?;
    let (mut lo, mut hi) = (
        local_secs - i64::from(after.max(before)),
        local_secs - i64::from(after.min(before)),
    );
    if offset_at(hi)? == before {
        return None; // not bracketed: no transition here to shift past
    }
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if offset_at(mid)? == before {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi.checked_mul(1_000_000)
}

/// Whether two zones agree on every offset across the instants a query plausibly holds.
///
/// Used to accept `from_tz` for a tz-aware column spelled differently from the column's own
/// label (`"UTC"` for `"+00:00"`, `"US/Eastern"` for `"America/New_York"`) while still
/// refusing one that would read the instants differently. Sampled each January and July of
/// 1900–2100, which covers every DST regime boundary in that range.
fn same_rules(a: &Tz, b: &Tz) -> bool {
    (1900..=2100).all(|year| {
        [1u32, 7].iter().all(|&month| {
            let Some(at) = chrono::NaiveDate::from_ymd_opt(year, month, 1)
                .and_then(|d| d.and_hms_opt(12, 0, 0))
            else {
                return true;
            };
            a.offset_from_utc_datetime(&at).fix() == b.offset_from_utc_datetime(&at).fix()
        })
    })
}

/// The input as `Timestamp(us, tz)` micros, keeping its zone label.
fn as_micros(arr: &ArrayRef, func: &str) -> Result<ArrayRef, ExprError> {
    match arr.data_type() {
        DataType::Timestamp(_, tz) => Ok(cast(
            arr,
            &DataType::Timestamp(TimeUnit::Microsecond, tz.clone()),
        )?),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Null | DataType::Date32 => Ok(cast(
            arr,
            &DataType::Timestamp(TimeUnit::Microsecond, None),
        )?),
        other => Err(ExprError::ExpectedType {
            func: func.into(),
            want: "a Timestamp argument",
            got: crate::error::type_name(other),
        }),
    }
}

/// Map every non-null micros value through `f`, which may null a row or error.
fn map_micros(
    arr: &ArrayRef,
    tz: Option<Arc<str>>,
    mut f: impl FnMut(i64) -> Result<Option<i64>, ExprError>,
) -> Result<ArrayRef, ExprError> {
    let ts = arr.as_primitive::<TimestampMicrosecondType>();
    let mut out: Vec<Option<i64>> = Vec::with_capacity(ts.len());
    for i in 0..ts.len() {
        out.push(if ts.is_null(i) { None } else { f(ts.value(i))? });
    }
    Ok(Arc::new(
        TimestampMicrosecondArray::from(out).with_timezone_opt(tz),
    ))
}

/// `convert_timezone(from_tz, to_tz, ts)` — the naive wall clock in `to_tz` of each
/// instant. A naive input is a wall clock in `from_tz`; an aware one is already an instant,
/// and `from_tz` must agree with its zone.
pub(crate) fn eval_convert_timezone(
    arr: &ArrayRef,
    from_tz: &str,
    to_tz: &str,
    ambiguous: Ambiguous,
    nonexistent: Nonexistent,
) -> Result<ArrayRef, ExprError> {
    const FUNC: &str = "convert_timezone";
    let from = parse_zone(from_tz, FUNC)?;
    let to = parse_zone(to_tz, FUNC)?;
    let column = column_zone(arr, FUNC)?;
    if let Some(zone) = &column {
        if !same_rules(zone, &from) {
            return Err(ExprError::InvalidArgument {
                func: FUNC.into(),
                reason: format!(
                    "the column is already tz-aware ({zone}), so its values are instants; \
                     from_tz={from_tz:?} would re-read them in another zone. Pass \
                     from_tz={:?}, or use replace_timezone to relabel the wall clock",
                    zone.to_string()
                ),
            });
        }
    }
    let micros = as_micros(arr, FUNC)?;
    map_micros(&micros, None, |v| {
        let instant = match column {
            Some(_) => Some(v),
            None => localize(v, &from, ambiguous, nonexistent, FUNC)?,
        };
        Ok(instant.and_then(|u| local_micros(u, &to)))
    })
}

/// `replace_timezone(tz, ts)` — keep each wall clock and label it with `tz` (a new
/// instant), or with no zone when `tz` is `None`.
pub(crate) fn eval_replace_timezone(
    arr: &ArrayRef,
    tz: Option<&str>,
    ambiguous: Ambiguous,
    nonexistent: Nonexistent,
) -> Result<ArrayRef, ExprError> {
    const FUNC: &str = "replace_timezone";
    let target = tz.map(|name| parse_zone(name, FUNC)).transpose()?;
    let column = column_zone(arr, FUNC)?;
    let micros = as_micros(arr, FUNC)?;
    map_micros(&micros, tz.map(Arc::from), |v| {
        let wall = match &column {
            Some(zone) => local_micros(v, zone),
            None => Some(v),
        };
        match (wall, &target) {
            (None, _) => Ok(None),
            (Some(w), None) => Ok(Some(w)),
            (Some(w), Some(zone)) => localize(w, zone, ambiguous, nonexistent, FUNC),
        }
    })
}

/// A tz-aware timestamp column as the naive wall clock its zone reads (`None` for any other
/// input), so a calendar field or calendar arithmetic is taken in the column's zone.
pub(crate) fn as_wall_clock(arr: &ArrayRef) -> Result<Option<ArrayRef>, ExprError> {
    let Some(zone) = column_zone(arr, "calendar arithmetic")? else {
        return Ok(None);
    };
    let micros = as_micros(arr, "calendar arithmetic")?;
    map_micros(&micros, None, |v| Ok(local_micros(v, &zone))).map(Some)
}

/// Run a wall-clock kernel on a tz-aware timestamp column: read each instant as its zone's
/// local clock, apply `kernel` to that naive column, and localize the result back in the
/// same zone, keeping the label. `None` when the input carries no zone.
///
/// The calendar kernels (`date_trunc`, `offset_by`, month/day arithmetic) are defined on a
/// wall clock; run on the stored UTC micros they answer the UTC calendar instead, so a New
/// York 23:00 truncated to the day landed on the next day. A truncation or a calendar shift
/// can land in a DST gap or overlap: it keeps the input's own offset through an overlap when
/// it can (else the earlier instant) and takes the first instant after a gap, so these
/// kernels never raise or null on a zone rule.
pub(crate) fn on_wall_clock(
    arr: &ArrayRef,
    kernel: impl FnOnce(&ArrayRef) -> Result<ArrayRef, ExprError>,
) -> Result<Option<ArrayRef>, ExprError> {
    let (Some(zone), Some(local)) = (
        column_zone(arr, "calendar arithmetic")?,
        as_wall_clock(arr)?,
    ) else {
        return Ok(None);
    };
    let label = match arr.data_type() {
        DataType::Timestamp(_, tz) => tz.clone(),
        _ => None,
    };
    let shifted = as_micros(&kernel(&local)?, "calendar arithmetic")?;
    let shifted = shifted.as_primitive::<TimestampMicrosecondType>();
    let before = as_micros(arr, "calendar arithmetic")?;
    let before = before.as_primitive::<TimestampMicrosecondType>();
    let wall = local.as_primitive::<TimestampMicrosecondType>();
    let mut out: Vec<Option<i64>> = Vec::with_capacity(shifted.len());
    for i in 0..shifted.len() {
        if shifted.is_null(i) {
            out.push(None);
            continue;
        }
        let v = shifted.value(i);
        // An overlap is resolved toward the offset the input instant had, so truncating
        // 01:30 EST to the hour gives 01:00 EST and 01:30 EDT gives 01:00 EDT: neither moves
        // the instant forward. With no such candidate the earlier instant is taken.
        let input_offset =
            (!before.is_null(i) && !wall.is_null(i)).then(|| wall.value(i) - before.value(i));
        let same = input_offset.and_then(|off| {
            let naive = DateTime::from_timestamp_micros(v)?.naive_utc();
            match zone.offset_from_local_datetime(&naive) {
                LocalResult::Ambiguous(a, b) => [a, b]
                    .into_iter()
                    .map(|o| i64::from(o.fix().local_minus_utc()) * 1_000_000)
                    .find(|o| *o == off)
                    .map(|o| v - o),
                _ => None,
            }
        });
        out.push(match same {
            Some(u) => Some(u),
            None => localize(
                v,
                &zone,
                Ambiguous::Earliest,
                Nonexistent::ShiftForward,
                "calendar arithmetic",
            )?,
        });
    }
    Ok(Some(Arc::new(
        TimestampMicrosecondArray::from(out).with_timezone_opt(label),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::TimestampMicrosecondArray;
    use chrono::NaiveDate;

    fn micros(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> i64 {
        NaiveDate::from_ymd_opt(y, mo, d)
            .unwrap()
            .and_hms_opt(h, mi, 0)
            .unwrap()
            .and_utc()
            .timestamp_micros()
    }

    fn values(arr: &ArrayRef) -> Vec<Option<i64>> {
        arr.as_primitive::<TimestampMicrosecondType>()
            .iter()
            .collect()
    }

    #[test]
    fn an_aware_column_is_converted_as_an_instant_not_a_wall_clock() {
        // 07:30Z is 16:30 in Tokyo, whatever zone label the column carries.
        let arr: ArrayRef = Arc::new(
            TimestampMicrosecondArray::from(vec![micros(2024, 3, 10, 7, 30)]).with_timezone("UTC"),
        );
        let out = eval_convert_timezone(
            &arr,
            "UTC",
            "Asia/Tokyo",
            Ambiguous::Null,
            Nonexistent::Null,
        )
        .unwrap();
        assert_eq!(values(&out), vec![Some(micros(2024, 3, 10, 16, 30))]);
        assert_eq!(
            out.data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, None)
        );
        // A from_tz naming a different zone is refused rather than shifting the instant.
        let err = eval_convert_timezone(
            &arr,
            "Asia/Tokyo",
            "America/New_York",
            Ambiguous::Null,
            Nonexistent::Null,
        );
        assert!(err.is_err());
    }

    #[test]
    fn dst_policies_resolve_the_gap_and_the_overlap() {
        let ny = parse_zone("America/New_York", "t").unwrap();
        let gap = micros(2024, 3, 10, 2, 30);
        let overlap = micros(2024, 11, 3, 1, 30);
        let pol = |a, n, v| localize(v, &ny, a, n, "t");
        assert_eq!(pol(Ambiguous::Null, Nonexistent::Null, gap).unwrap(), None);
        assert!(pol(Ambiguous::Null, Nonexistent::Raise, gap).is_err());
        // The gap ends at 03:00 EDT = 07:00Z.
        assert_eq!(
            pol(Ambiguous::Null, Nonexistent::ShiftForward, gap).unwrap(),
            Some(micros(2024, 3, 10, 7, 0))
        );
        // 01:30 happens at 05:30Z (EDT) and again at 06:30Z (EST).
        assert_eq!(
            pol(Ambiguous::Earliest, Nonexistent::Null, overlap).unwrap(),
            Some(micros(2024, 11, 3, 5, 30))
        );
        assert_eq!(
            pol(Ambiguous::Latest, Nonexistent::Null, overlap).unwrap(),
            Some(micros(2024, 11, 3, 6, 30))
        );
        assert_eq!(
            pol(Ambiguous::Null, Nonexistent::Null, overlap).unwrap(),
            None
        );
        assert!(pol(Ambiguous::Raise, Nonexistent::Null, overlap).is_err());
    }

    #[test]
    fn replace_keeps_the_clock_and_strip_keeps_the_local_clock() {
        let naive: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![micros(
            2024, 1, 15, 9, 0,
        )]));
        let tokyo = eval_replace_timezone(
            &naive,
            Some("Asia/Tokyo"),
            Ambiguous::Raise,
            Nonexistent::Raise,
        )
        .unwrap();
        assert_eq!(values(&tokyo), vec![Some(micros(2024, 1, 15, 0, 0))]);
        let back =
            eval_replace_timezone(&tokyo, None, Ambiguous::Raise, Nonexistent::Raise).unwrap();
        assert_eq!(values(&back), vec![Some(micros(2024, 1, 15, 9, 0))]);
    }
}
