//! The business-day calendar behind `Expr::BusinessDay`: `add_business_days`,
//! `business_day_count`, and `is_business_day` with holidays.
//!
//! All three are answered from one definition — a day is a business day when the weekmask
//! admits its weekday and it is not a holiday — so they cannot disagree about a date. The
//! semantics are numpy's `busday_offset` / `busday_count` / `is_busday`, which is the
//! oracle the tests hold them to: a count is over the half-open `[start, end)`, and minus
//! the count over `(end, start]` when `end < start`, a holiday on a weekend changes nothing, and a start that is not a
//! business day is an error unless the caller asks to roll it.
//!
//! Nothing here walks the calendar a day at a time. `B(d)`, the signed number of business
//! days in `[0, d)`, is closed-form — whole weeks times the weekmask's count, plus a partial
//! week, minus a binary search over the sorted holidays — so a count is two evaluations of
//! it and an add is a bisection for the day where it reaches the target.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, BooleanArray, Date32Array, Int64Array};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Date32Type, Int64Type, TimeUnit};

use super::timezone::{as_wall_clock, on_wall_clock};
use crate::{BusinessDayFunc, BusinessRoll, ExprError};

const MICROS_PER_DAY: i64 = 86_400_000_000;
const FUNC: &str = "business days";

/// A weekmask plus its holidays, the whole definition of a business day.
struct Calendar {
    /// Monday first.
    mask: [bool; 7],
    per_week: i64,
    /// Holidays that fall on a weekmask day, sorted and unique, as days since the epoch.
    holidays: Vec<i64>,
}

impl Calendar {
    fn new(holidays: &[i32], weekmask: Option<[bool; 7]>) -> Result<Self, ExprError> {
        let mask = weekmask.unwrap_or([true, true, true, true, true, false, false]);
        let per_week = mask.iter().filter(|b| **b).count() as i64;
        if per_week == 0 {
            return Err(ExprError::InvalidArgument {
                func: FUNC.into(),
                reason: "the weekmask admits no weekday, so no day is a business day".into(),
            });
        }
        let mut days: Vec<i64> = holidays
            .iter()
            .map(|d| i64::from(*d))
            .filter(|d| mask[weekday(*d)])
            .collect();
        days.sort_unstable();
        days.dedup();
        Ok(Self {
            mask,
            per_week,
            holidays: days,
        })
    }

    fn is_business(&self, day: i64) -> bool {
        self.mask[weekday(day)] && self.holidays.binary_search(&day).is_err()
    }

    /// The signed number of business days in `[0, day)` (negative for `day < 0`).
    fn before(&self, day: i64) -> i64 {
        let (weeks, rem) = (day.div_euclid(7), day.rem_euclid(7));
        let partial = (0..rem).filter(|k| self.mask[weekday(*k)]).count() as i64;
        let holidays = self.holidays.partition_point(|h| *h < day) as i64;
        let below_zero = self.holidays.partition_point(|h| *h < 0) as i64;
        weeks * self.per_week + partial - (holidays - below_zero)
    }

    /// Business days in `[start, end)`, or minus those in `(end, start]` when `end < start`
    /// (numpy `busday_count`, which mirrors the half-open range rather than negating it).
    fn count(&self, start: i64, end: i64) -> i64 {
        if end >= start {
            self.before(end) - self.before(start)
        } else {
            self.before(end + 1) - self.before(start + 1)
        }
    }

    /// `start` rolled to a business day by `roll`, or an error for `Raise`.
    fn rolled(&self, start: i64, roll: BusinessRoll) -> Result<Option<i64>, ExprError> {
        if self.is_business(start) {
            return Ok(Some(start));
        }
        let step = match roll {
            BusinessRoll::Raise => {
                return Err(ExprError::InvalidArgument {
                    func: "add_business_days".into(),
                    reason: format!(
                        "day {start} since the epoch is not a business day; pass roll=\"forward\" \
                         or roll=\"backward\" to move it to one first"
                    ),
                })
            }
            BusinessRoll::Forward => 1,
            BusinessRoll::Backward => -1,
        };
        let mut day = start;
        // Every run of non-business days is shorter than a week plus the holidays in it.
        for _ in 0..=(7 + self.holidays.len()) {
            day += step;
            if self.is_business(day) {
                return Ok(Some(day));
            }
        }
        Ok(None)
    }

    /// The business day `n` business days from the business day `start`.
    fn add(&self, start: i64, n: i64) -> Option<i64> {
        let target = self.before(start).checked_add(n)?;
        // Each week holds at least one business day, less the holidays; this brackets the
        // answer with room to spare and stays far inside i64 for any representable date.
        let reach = n
            .checked_abs()?
            .checked_add(self.holidays.len() as i64 + 2)?
            .checked_mul(7)?;
        let (mut lo, mut hi) = (start.checked_sub(reach)?, start.checked_add(reach)?);
        // The smallest `d` with `before(d + 1) > target` is the business day where the
        // count reaches `target`.
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.before(mid + 1) > target {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        Some(lo)
    }
}

/// Monday = 0 … Sunday = 6 of a day since 1970-01-01 (a Thursday).
fn weekday(day: i64) -> usize {
    (day + 3).rem_euclid(7) as usize
}

/// Day counts of a date-like column, read on the wall clock for a zoned timestamp.
fn day_numbers(arr: &ArrayRef) -> Result<Vec<Option<i64>>, ExprError> {
    let local = as_wall_clock(arr)?;
    let arr = local.as_ref().unwrap_or(arr);
    Ok(match arr.data_type() {
        DataType::Timestamp(..) => {
            let us = cast(arr, &DataType::Timestamp(TimeUnit::Microsecond, None))?;
            let us = cast(&us, &DataType::Int64)?;
            us.as_primitive::<Int64Type>()
                .iter()
                .map(|v| v.map(|u| u.div_euclid(MICROS_PER_DAY)))
                .collect()
        }
        _ => {
            let days = cast(arr, &DataType::Date32)?;
            days.as_primitive::<Date32Type>()
                .iter()
                .map(|v| v.map(i64::from))
                .collect()
        }
    })
}

/// Evaluate one business-day operation.
pub(crate) fn eval_business_day(
    func: BusinessDayFunc,
    arr: &ArrayRef,
    other: Option<&ArrayRef>,
    holidays: &[i32],
    weekmask: Option<[bool; 7]>,
    roll: BusinessRoll,
) -> Result<ArrayRef, ExprError> {
    let cal = Calendar::new(holidays, weekmask)?;
    let need_other = || {
        other.ok_or_else(|| ExprError::MissingArgument {
            func: FUNC.into(),
            arg: "a second operand",
        })
    };
    match func {
        BusinessDayFunc::Is => {
            let out: BooleanArray = day_numbers(arr)?
                .into_iter()
                .map(|d| d.map(|d| cal.is_business(d)))
                .collect();
            Ok(Arc::new(out))
        }
        BusinessDayFunc::Count => {
            let ends = day_numbers(need_other()?)?;
            let out: Int64Array = day_numbers(arr)?
                .into_iter()
                .zip(ends)
                .map(|(s, e)| Some(cal.count(s?, e?)))
                .collect();
            Ok(Arc::new(out))
        }
        BusinessDayFunc::Add => {
            let n = cast(need_other()?, &DataType::Int64)?;
            add(&cal, arr, n.as_primitive::<Int64Type>(), roll)
        }
    }
}

/// `add_business_days`, type-preserving: a date stays a date, a timestamp keeps its time of
/// day (and, zoned, its zone, moving by local days).
fn add(
    cal: &Calendar,
    arr: &ArrayRef,
    n: &arrow::array::Int64Array,
    roll: BusinessRoll,
) -> Result<ArrayRef, ExprError> {
    if let Some(out) = on_wall_clock(arr, |wall| add(cal, wall, n, roll))? {
        return Ok(out);
    }
    let step = |day: i64, i: usize| -> Result<Option<i64>, ExprError> {
        if n.is_null(i) {
            return Ok(None);
        }
        Ok(cal.rolled(day, roll)?.and_then(|d| cal.add(d, n.value(i))))
    };
    match arr.data_type() {
        DataType::Timestamp(_, tz) => {
            let us = cast(arr, &DataType::Timestamp(TimeUnit::Microsecond, None))?;
            let us = cast(&us, &DataType::Int64)?;
            let us = us.as_primitive::<Int64Type>();
            let mut out = Vec::with_capacity(us.len());
            for i in 0..us.len() {
                out.push(if us.is_null(i) {
                    None
                } else {
                    let v = us.value(i);
                    let (day, clock) = (v.div_euclid(MICROS_PER_DAY), v.rem_euclid(MICROS_PER_DAY));
                    step(day, i)?
                        .and_then(|d| d.checked_mul(MICROS_PER_DAY))
                        .and_then(|d| d.checked_add(clock))
                });
            }
            Ok(Arc::new(
                arrow::array::TimestampMicrosecondArray::from(out).with_timezone_opt(tz.clone()),
            ))
        }
        _ => {
            let days = day_numbers(arr)?;
            let mut out = Vec::with_capacity(days.len());
            for (i, d) in days.into_iter().enumerate() {
                out.push(match d {
                    None => None,
                    Some(d) => step(d, i)?.and_then(|v| i32::try_from(v).ok()),
                });
            }
            Ok(Arc::new(Date32Array::from(out)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 2024-01-01 is a Monday: day 19723.
    const MON: i64 = 19_723;

    #[test]
    fn counts_and_adds_agree_with_numpy() {
        let cal = Calendar::new(&[], None).unwrap();
        // numpy.busday_count('2024-01-01', '2024-01-08') == 5; reversed == -5.
        assert_eq!(cal.count(MON, MON + 7), 5);
        assert_eq!(cal.count(MON + 7, MON), -5);
        // Reversed from a Saturday: numpy counts (Mon, Sat], Tuesday to Friday, as -4.
        assert_eq!(cal.count(MON + 5, MON), -4);
        // numpy.busday_offset('2024-01-05', 1) == '2024-01-08' (Fri -> Mon).
        assert_eq!(cal.add(MON + 4, 1), Some(MON + 7));
        assert_eq!(cal.add(MON + 7, -1), Some(MON + 4));
        assert_eq!(cal.add(MON, 0), Some(MON));
        assert_eq!(cal.add(MON, 10), Some(MON + 14));
    }

    #[test]
    fn holidays_and_weekmask_are_one_definition() {
        // Tuesday 2024-01-02 is a holiday; a Saturday holiday is ignored.
        let cal = Calendar::new(&[(MON + 1) as i32, (MON + 5) as i32], None).unwrap();
        assert!(!cal.is_business(MON + 1));
        assert_eq!(cal.count(MON, MON + 7), 4);
        assert_eq!(cal.add(MON, 1), Some(MON + 2));
        // Sunday-to-Thursday week.
        let gulf = Calendar::new(&[], Some([true, true, true, true, false, false, true])).unwrap();
        assert!(gulf.is_business(MON + 6));
        assert!(!gulf.is_business(MON + 4));
        assert!(Calendar::new(&[], Some([false; 7])).is_err());
    }

    #[test]
    fn a_weekend_start_rolls_or_raises() {
        let cal = Calendar::new(&[], None).unwrap();
        let sat = MON + 5;
        assert!(cal.rolled(sat, BusinessRoll::Raise).is_err());
        assert_eq!(
            cal.rolled(sat, BusinessRoll::Forward).unwrap(),
            Some(MON + 7)
        );
        assert_eq!(
            cal.rolled(sat, BusinessRoll::Backward).unwrap(),
            Some(MON + 4)
        );
    }

    #[test]
    fn count_matches_a_day_by_day_walk() {
        let cal = Calendar::new(&[(MON + 1) as i32, (MON + 30) as i32, -3], None).unwrap();
        for a in (MON - 40)..(MON + 40) {
            for b in [a - 9, a, a + 1, a + 13, a + 50] {
                let walk = if b >= a {
                    (a..b).filter(|d| cal.is_business(*d)).count() as i64
                } else {
                    -(((b + 1)..=a).filter(|d| cal.is_business(*d)).count() as i64)
                };
                assert_eq!(cal.count(a, b), walk, "count({a}, {b})");
            }
        }
        // And around the epoch, where the floored week split changes sign.
        for a in -30..30 {
            let walk = (a..a + 20).filter(|d| cal.is_business(*d)).count() as i64;
            assert_eq!(cal.count(a, a + 20), walk);
        }
    }
}
