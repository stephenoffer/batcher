//! Calendar field extraction as integer arithmetic, for the date parts a query groups by.
//!
//! Arrow's `date_part` answers `year`/`month`/`day` by building a `chrono::NaiveDateTime` per
//! value and reading the field off it. That is correct, and it costs more than the grouping it
//! feeds: `SUM(EXTRACT(YEAR FROM l_shipdate))` over TPC-H `lineitem` spent 111 ms of CPU where
//! the same aggregate over `l_linenumber + 1` spent 25 ms.
//!
//! The civil-from-days conversion (Howard Hinnant's, the one `std::chrono` specifies) needs no
//! calendar object and no table: an era split, a year-of-era by subtraction, and a March-based
//! month from one multiply-divide. Every division is by a constant, so the row costs a few
//! multiplies and no branch on the data.
//!
//! **The arithmetic is 32-bit and unsigned, and both halves of that are load-bearing.** The
//! same decomposition in `i64` measured *slower than the kernel it replaces* — 12.8 ns a row
//! against 10.8 — because a 64-bit division by a constant is a widening multiply; in `u32` it
//! is 7.3. The values are biased by a whole number of eras first (`BIAS_DAYS`), which puts
//! every supported day in the non-negative range, so no division has to handle a sign either.
//! Day-of-week needs no decomposition at all and costs 2.8 ns against the kernel's 12.4.
//!
//! **Arrow's kernel is the oracle, and this path declines rather than extends it.** Only a
//! `Date32` or a timezone-naive `Timestamp` takes this route — a zoned timestamp extracts in
//! local time, which is `date_part`'s business — and a column holding any value outside
//! chrono's representable range falls back whole, so a far-future date keeps whatever answer
//! the kernel gives it instead of a second one invented here. The tests below hold the two
//! paths equal across that whole range.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Int64Array};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, Date32Type, Int64Type, TimeUnit};

use crate::DateFunc;

/// Days from 0000-03-01 to 1970-01-01: the shift that puts the era boundary on a March 1.
const EPOCH_SHIFT: i32 = 719_468;
/// Days in one 400-year Gregorian era.
const DAYS_PER_ERA: u32 = 146_097;
/// Eras added to every value so the arithmetic never meets a negative number. 656 eras clears
/// the earliest supported day; the years they add come back off at the end.
const BIAS_ERAS: i32 = 656;
const BIAS_DAYS: i32 = DAYS_PER_ERA as i32 * BIAS_ERAS;
const BIAS_YEARS: i32 = BIAS_ERAS * 400;
/// The inclusive day range chrono represents (years −262,144 … 262,143). Arrow's kernel
/// cannot convert a value outside it, so neither does this.
const MIN_DAY: i64 = -96_465_292;
const MAX_DAY: i64 = 95_026_236;
const SECS_PER_DAY: i64 = 86_400;

/// The fields one value decomposes into; each `DateFunc` reads one of them.
#[derive(Clone, Copy)]
struct Civil {
    year: i32,
    month: u32,
    day: u32,
}

/// Year, month and day of `days` since 1970-01-01, on the proleptic Gregorian calendar.
///
/// `days` must be within `[MIN_DAY, MAX_DAY]`; `extract` checks the column before calling.
#[inline(always)]
fn civil_from_days(days: i32) -> Civil {
    let z = (days + EPOCH_SHIFT + BIAS_DAYS) as u32;
    let era = z / DAYS_PER_ERA;
    let doe = z - era * DAYS_PER_ERA; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365], March 1 = 0
    let mp = (5 * doy + 2) / 153; // [0, 11], March = 0
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = (yoe + era * 400) as i32 - BIAS_YEARS + i32::from(month <= 2);
    Civil { year, month, day }
}

/// 1-based day of the year for a civil date.
#[inline(always)]
fn ordinal(c: Civil) -> i64 {
    const CUMULATIVE: [u32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let leap = (c.year % 4 == 0 && c.year % 100 != 0) || c.year % 400 == 0;
    i64::from(CUMULATIVE[(c.month - 1) as usize] + c.day + u32::from(leap && c.month > 2))
}

/// Whether every non-null value in `vals` lies in `[lo, hi]`.
///
/// A min/max fold first, because it vectorizes and is the whole answer for real data; the
/// null-aware walk runs only when that fold sees an out-of-range value, which may be sitting
/// under a null slot.
fn within<T: Copy + Into<i64>>(vals: &[T], nulls: Option<&NullBuffer>, lo: i64, hi: i64) -> bool {
    let (min, max) = vals.iter().fold((i64::MAX, i64::MIN), |(mn, mx), &v| {
        let v = v.into();
        (mn.min(v), mx.max(v))
    });
    if vals.is_empty() || (lo <= min && max <= hi) {
        return true;
    }
    let Some(nulls) = nulls else { return false };
    vals.iter()
        .enumerate()
        .all(|(i, &v)| nulls.is_null(i) || (lo..=hi).contains(&v.into()))
}

/// Apply one field to every value, chosen once outside the loop so the loop body is branchless.
///
/// `ticks_per_sec` is `None` for a day count (`Date32`) and the unit's ticks per second for a
/// timestamp, whose values are clamped to `[lo, hi]` first: that only ever moves a slot under a
/// null (`within` has already vouched for the rest), and it keeps that slot's arithmetic in range.
fn map_fields<T: Copy + Into<i64>>(
    vals: &[T],
    ticks_per_sec: Option<i64>,
    (lo, hi): (i64, i64),
    func: DateFunc,
) -> Option<Vec<i64>> {
    fn apply<T: Copy + Into<i64>, F: Fn(i32, u32) -> i64>(
        vals: &[T],
        ticks_per_sec: Option<i64>,
        (lo, hi): (i64, i64),
        f: F,
    ) -> Vec<i64> {
        match ticks_per_sec {
            None => vals.iter().map(|&d| f(d.into() as i32, 0)).collect(),
            Some(per_sec) => {
                let per_day = per_sec * SECS_PER_DAY;
                vals.iter()
                    .map(|&v| {
                        let v = v.into().clamp(lo, hi);
                        let secs = v.rem_euclid(per_day) / per_sec;
                        f(v.div_euclid(per_day) as i32, secs as u32)
                    })
                    .collect()
            }
        }
    }
    let b = (lo, hi);
    let t = ticks_per_sec;
    let timed = ticks_per_sec.is_some();
    Some(match func {
        DateFunc::Year => apply(vals, t, b, |d, _| i64::from(civil_from_days(d).year)),
        DateFunc::Month => apply(vals, t, b, |d, _| i64::from(civil_from_days(d).month)),
        DateFunc::Day => apply(vals, t, b, |d, _| i64::from(civil_from_days(d).day)),
        DateFunc::Quarter => apply(vals, t, b, |d, _| {
            i64::from((civil_from_days(d).month - 1) / 3 + 1)
        }),
        DateFunc::DayOfYear => apply(vals, t, b, |d, _| ordinal(civil_from_days(d))),
        // 1970-01-01 was a Thursday; Sunday is 0.
        DateFunc::DayOfWeek => apply(vals, t, b, |d, _| i64::from((d + 4).rem_euclid(7))),
        DateFunc::Hour if timed => apply(vals, t, b, |_, s| i64::from(s / 3600)),
        DateFunc::Minute if timed => apply(vals, t, b, |_, s| i64::from(s / 60 % 60)),
        DateFunc::Second if timed => apply(vals, t, b, |_, s| i64::from(s % 60)),
        _ => return None,
    })
}

/// Extract `func` from `arr` by integer arithmetic, or `None` to leave it to Arrow's kernel.
///
/// Nulls are carried from the input's validity buffer; the value computed under a null slot is
/// never read.
pub(crate) fn extract(func: DateFunc, arr: &ArrayRef) -> Option<ArrayRef> {
    let nulls = arr.logical_nulls();
    let values = match arr.data_type() {
        DataType::Date32 => {
            let vals = arr.as_primitive::<Date32Type>().values();
            if !within(vals, nulls.as_ref(), MIN_DAY, MAX_DAY) {
                return None;
            }
            map_fields(vals, None, (MIN_DAY, MAX_DAY), func)?
        }
        DataType::Timestamp(unit, None) => {
            let per_sec: i64 = match unit {
                TimeUnit::Second => 1,
                TimeUnit::Millisecond => 1_000,
                TimeUnit::Microsecond => 1_000_000,
                TimeUnit::Nanosecond => 1_000_000_000,
            };
            let per_day = per_sec * SECS_PER_DAY;
            // The last day runs to its final tick, not to its midnight.
            let (lo, hi) = (
                MIN_DAY.saturating_mul(per_day),
                (MAX_DAY + 1)
                    .checked_mul(per_day)
                    .map_or(i64::MAX, |end| end - 1),
            );
            let ints = arrow::compute::cast(arr, &DataType::Int64).ok()?;
            let vals = ints.as_primitive::<Int64Type>().values();
            if !within(vals, nulls.as_ref(), lo, hi) {
                return None;
            }
            map_fields(vals, Some(per_sec), (lo, hi), func)?
        }
        _ => return None,
    };
    Some(Arc::new(Int64Array::new(values.into(), nulls)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        Date32Array, TimestampMicrosecondArray, TimestampMillisecondArray,
        TimestampNanosecondArray, TimestampSecondArray,
    };
    use arrow::compute::kernels::temporal::{date_part, DatePart};

    const FUNCS: [(DateFunc, DatePart); 9] = [
        (DateFunc::Year, DatePart::Year),
        (DateFunc::Month, DatePart::Month),
        (DateFunc::Day, DatePart::Day),
        (DateFunc::Quarter, DatePart::Quarter),
        (DateFunc::DayOfYear, DatePart::DayOfYear),
        (DateFunc::DayOfWeek, DatePart::DayOfWeekSunday0),
        (DateFunc::Hour, DatePart::Hour),
        (DateFunc::Minute, DatePart::Minute),
        (DateFunc::Second, DatePart::Second),
    ];

    fn oracle(part: DatePart, arr: &ArrayRef) -> ArrayRef {
        let i32s = date_part(arr, part).unwrap();
        arrow::compute::cast(&i32s, &DataType::Int64).unwrap()
    }

    fn assert_agrees(arr: &ArrayRef) {
        let mut exercised = 0;
        for (func, part) in FUNCS {
            if let Some(fast) = extract(func, arr) {
                let want = oracle(part, arr);
                let first_diff = (0..arr.len()).find(|&i| {
                    fast.is_null(i) != want.is_null(i)
                        || (!want.is_null(i)
                            && fast.as_primitive::<Int64Type>().value(i)
                                != want.as_primitive::<Int64Type>().value(i))
                });
                if let Some(i) = first_diff {
                    let raw = arrow::compute::cast(arr, &DataType::Int64).unwrap();
                    panic!(
                        "{func:?} at raw {}: fast {:?} kernel {:?}",
                        raw.as_primitive::<Int64Type>().value(i),
                        (!fast.is_null(i)).then(|| fast.as_primitive::<Int64Type>().value(i)),
                        (!want.is_null(i)).then(|| want.as_primitive::<Int64Type>().value(i)),
                    );
                }
                exercised += 1;
            }
        }
        assert!(
            exercised >= 6,
            "the fast path declined a type it exists for"
        );
    }

    /// Every day across several eras on both sides of the epoch, including each century rule.
    fn sweep_days() -> Vec<i32> {
        let mut days: Vec<i32> = (-800_000..800_000).step_by(7).collect();
        // Every day around the boundaries a calendar gets wrong: 1600/1700/1900/2000/2100
        // leap rules, the epoch, and year zero.
        for anchor in [-135_140, -98_615, -25_567, 0, 10_957, 47_482, -719_528] {
            days.extend(anchor - 400..anchor + 400);
        }
        days.extend([MIN_DAY as i32, MAX_DAY as i32]);
        days
    }

    #[test]
    fn the_range_bounds_are_chronos_own() {
        use chrono::NaiveDate;
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        assert_eq!((NaiveDate::MIN - epoch).num_days(), MIN_DAY);
        assert_eq!((NaiveDate::MAX - epoch).num_days(), MAX_DAY);
    }

    #[test]
    fn date32_fields_match_the_arrow_kernel() {
        let arr: ArrayRef = Arc::new(Date32Array::from(sweep_days()));
        assert_agrees(&arr);
    }

    #[test]
    fn timestamp_fields_match_the_arrow_kernel_including_before_the_epoch() {
        let micros: Vec<i64> = sweep_days()
            .iter()
            .map(|&d| i64::from(d.clamp(-200_000, 200_000)) * 86_400_000_000 - 1_234_567)
            .chain((-5_000..5_000).map(|s| s * 997_531))
            .collect();
        let arr: ArrayRef = Arc::new(TimestampMicrosecondArray::from(micros));
        assert_agrees(&arr);
        let secs: ArrayRef = Arc::new(TimestampSecondArray::from(vec![-1, 0, 59, 3_599, -86_401]));
        assert_agrees(&secs);
        // The first and last instant chrono can hold, at the unit where they fit in an i64.
        let day = 86_400_000_000_i64;
        let edges = vec![MIN_DAY * day, MAX_DAY * day + day - 1, -1, 0];
        assert_agrees(&(Arc::new(TimestampMicrosecondArray::from(edges)) as ArrayRef));
        let nanos = vec![i64::MIN, i64::MAX, -1, 0, -86_400_000_000_001];
        assert_agrees(&(Arc::new(TimestampNanosecondArray::from(nanos)) as ArrayRef));
        let millis = vec![-1, 0, -86_400_001, 1_700_000_000_123];
        assert_agrees(&(Arc::new(TimestampMillisecondArray::from(millis)) as ArrayRef));
    }

    #[test]
    fn nulls_are_carried_and_their_slots_never_decide_the_fallback() {
        let arr: ArrayRef = Arc::new(Date32Array::from(vec![Some(19_000), None, Some(-1)]));
        let out = extract(DateFunc::Year, &arr).unwrap();
        assert_eq!(out.as_ref(), oracle(DatePart::Year, &arr).as_ref());
        assert!(out.is_null(1));
    }

    #[test]
    fn a_value_outside_the_kernels_range_declines_the_whole_column() {
        let arr: ArrayRef = Arc::new(Date32Array::from(vec![0, MAX_DAY as i32 + 1]));
        assert!(extract(DateFunc::Year, &arr).is_none());
        let ok: ArrayRef = Arc::new(Date32Array::from(vec![0, MAX_DAY as i32]));
        assert!(extract(DateFunc::Year, &ok).is_some());
    }

    #[test]
    fn a_zoned_timestamp_and_a_time_field_on_a_date_are_left_to_the_kernel() {
        let zoned: ArrayRef =
            Arc::new(TimestampMicrosecondArray::from(vec![0_i64]).with_timezone("+05:00"));
        assert!(extract(DateFunc::Year, &zoned).is_none());
        let date: ArrayRef = Arc::new(Date32Array::from(vec![0]));
        assert!(extract(DateFunc::Hour, &date).is_none());
    }
}
