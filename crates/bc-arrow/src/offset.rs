//! Typed `key ± offset` arithmetic — the one place a temporal distance is applied.
//!
//! Three unrelated-looking features are the same question: what value sits a given distance
//! from this one. `offset_by` shifts a whole column (`bc_expr::eval::temporal`); a value-based
//! `RANGE` window frame needs the frame edge `t - INTERVAL '7' DAY` for each row
//! (`bc_runtime::window::range`); an ASOF join with a tolerance needs `|left - right| <= delta`
//! (`bc_runtime::join::asof`). Answering it three times would be three chances to disagree
//! about month-end clamping or about what overflow does.
//!
//! It lives in `bc-arrow` because that is the only crate strictly below all three consumers:
//! `bc-runtime` does not depend on `bc-expr` (its manifest lists `bc-arrow`, `bc-ir` and
//! `bc-sketches`), so a helper in `bc-expr` would be unreachable from the window and join
//! kernels that need it most.
//!
//! ## The interval representation
//!
//! `(months, days, micros)` — the same triple `bc_expr::Expr::DateOffset` already puts on the
//! wire, deliberately, so the engine has one spelling of an interval rather than a second one
//! here. Arrow's own `Interval` types are not used anywhere in the engine (`shuffle` excludes
//! them from its key types for the same reason this splits the triple: a month-day-nano is not
//! a single totally-ordered scalar).
//!
//! The three parts are kept apart rather than normalized into one number because they are not
//! interconvertible: a month is 28-31 days depending on where it lands, and a day is not
//! always 86,400 seconds once a caller works in local time. `days` is therefore an *exact*
//! 24-hour multiple here, and only `months` needs a calendar.
//!
//! ## Exact vs calendar
//!
//! [`TypedOffset::is_exact`] separates the two, and callers care: an exact offset is a fixed
//! distance that can be applied to a raw integer key, while a calendar offset has to be routed
//! through a civil date and back. Both are monotone in the input — `t1 < t2` implies
//! `shift(t1) <= shift(t2)`, including across a month-end clamp — which is what lets a window
//! frame slide over them with a two-pointer instead of re-searching per row.
//!
//! Every operation is checked and returns `None` rather than panicking or wrapping. These run
//! on user data, where a far-future timestamp plus a large offset is an input, not a bug.

use arrow::datatypes::{DataType, TimeUnit};

/// Microseconds in one 24-hour day.
const MICROS_PER_DAY: i64 = 86_400_000_000;

/// A signed temporal offset: calendar months, exact days, and exact microseconds.
///
/// Matches `bc_expr::Expr::DateOffset`'s payload. A negative field shifts backwards; the three
/// may disagree in sign, though a caller building one from an SQL `INTERVAL` will not.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TypedOffset {
    /// Calendar months, clamped to the last valid day when the target month is shorter.
    pub months: i64,
    /// Exact 24-hour days.
    pub days: i64,
    /// Exact microseconds.
    pub micros: i64,
}

impl TypedOffset {
    /// An offset of `months` calendar months, `days` exact days and `micros` exact microseconds.
    pub const fn new(months: i64, days: i64, micros: i64) -> Self {
        Self {
            months,
            days,
            micros,
        }
    }

    /// An offset of exactly `micros` microseconds.
    pub const fn micros(micros: i64) -> Self {
        Self::new(0, 0, micros)
    }

    /// True when this offset moves nothing.
    pub const fn is_zero(&self) -> bool {
        self.months == 0 && self.days == 0 && self.micros == 0
    }

    /// True when this is a fixed duration — no calendar component, so no civil date needed.
    ///
    /// The distinction is not cosmetic. An exact offset is the same distance wherever it is
    /// applied, so a caller may add it to a raw key; a calendar offset is not, so it must go
    /// through a date. A range-partitioner or a bloom filter can use the first and not the
    /// second.
    pub const fn is_exact(&self) -> bool {
        self.months == 0
    }

    /// Whether `shift_scalar` can shift a value of this type at all.
    ///
    /// Separate from the `None` that `shift_scalar` returns on overflow: an unsupported *type*
    /// is a plan-time decline the caller should report once, while an overflowed *value* is a
    /// per-row null. Collapsing the two would turn "this column cannot carry a frame" into
    /// "every row of this column overflowed".
    pub fn supports(&self, dt: &DataType) -> bool {
        match dt {
            DataType::Date32 | DataType::Date64 => true,
            DataType::Timestamp(_, _) => true,
            // A duration is a span, not a point on a calendar, so months are meaningless on it
            // — but an exact offset is ordinary addition.
            DataType::Duration(_) => self.is_exact(),
            _ => false,
        }
    }

    /// Apply this offset to one `value` of type `dt`, in that type's own units.
    ///
    /// `sign` is `1` to add and `-1` to subtract; a frame's two edges differ only by it, so
    /// passing it beats making the caller negate all three fields and get one wrong.
    ///
    /// Returns `None` if `dt` is unsupported or the result leaves the type's range.
    pub fn shift_scalar(&self, dt: &DataType, value: i64, sign: i8) -> Option<i64> {
        let signed = self.signed(sign)?;
        match dt {
            DataType::Date32 => signed.shift_date32(value),
            DataType::Date64 => {
                let millis = signed.shift_epoch_micros(value.checked_mul(1_000)?)?;
                // Round toward −∞ so a shift stays monotone across the epoch, which
                // truncation toward zero would break either side of it.
                Some(millis.div_euclid(1_000))
            }
            DataType::Timestamp(unit, _) => signed.shift_timestamp(value, *unit),
            DataType::Duration(unit) if signed.is_exact() => {
                let delta = signed.exact_micros()?;
                value.checked_add(scale_micros(delta, *unit)?)
            }
            _ => None,
        }
    }

    /// This offset with every field negated when `sign` is negative.
    fn signed(&self, sign: i8) -> Option<Self> {
        match sign {
            1 => Some(*self),
            -1 => Some(Self::new(
                self.months.checked_neg()?,
                self.days.checked_neg()?,
                self.micros.checked_neg()?,
            )),
            _ => None,
        }
    }

    /// The whole offset as microseconds — only meaningful when `is_exact`.
    fn exact_micros(&self) -> Option<i64> {
        self.days
            .checked_mul(MICROS_PER_DAY)?
            .checked_add(self.micros)
    }

    /// Shift a day count since the epoch, rejecting any sub-day component.
    ///
    /// `bc_expr`'s `offset_by` refuses a sub-day offset on a `Date` rather than silently
    /// truncating it, and this agrees: a caller asking for 12 hours from a date has a type
    /// error, and rounding it to zero or one day would answer a question they did not ask.
    /// The result is additionally required to land on a date `chrono` can represent, which is a
    /// narrower range than `i32` days allows (roughly ±262,000 years against ±5.8 million). That
    /// is deliberate rather than incidental: `bc_expr`'s `offset_by` has always routed through
    /// `NaiveDate` and so has always nulled beyond it, and a shared primitive that quietly
    /// widened the accepted range would change an existing operator's answers as a side effect
    /// of being extracted.
    fn shift_date32(&self, days_since_epoch: i64) -> Option<i64> {
        if self.micros != 0 {
            return None;
        }
        let shifted = if self.months == 0 {
            days_since_epoch
        } else {
            add_months_to_days(days_since_epoch, self.months)?
        };
        let out = shifted.checked_add(self.days)?;
        representable_day(out)?;
        // Date32 is an i32 day count; leaving that range is an overflow, not a wrap.
        i32::try_from(out).ok().map(i64::from)
    }

    /// Shift a timestamp held in `unit` since the epoch.
    fn shift_timestamp(&self, value: i64, unit: TimeUnit) -> Option<i64> {
        let micros = to_micros(value, unit)?;
        let shifted = self.shift_epoch_micros(micros)?;
        from_micros(shifted, unit)
    }

    /// Shift a microsecond count since the epoch, applying months on the civil calendar.
    fn shift_epoch_micros(&self, micros_since_epoch: i64) -> Option<i64> {
        let base = if self.months == 0 {
            micros_since_epoch
        } else {
            add_months_to_micros(micros_since_epoch, self.months)?
        };
        base.checked_add(self.days.checked_mul(MICROS_PER_DAY)?)?
            .checked_add(self.micros)
    }
}

/// A fixed number of microseconds expressed in `unit`, rounding toward −∞.
fn scale_micros(micros: i64, unit: TimeUnit) -> Option<i64> {
    match unit {
        TimeUnit::Second => Some(micros.div_euclid(1_000_000)),
        TimeUnit::Millisecond => Some(micros.div_euclid(1_000)),
        TimeUnit::Microsecond => Some(micros),
        TimeUnit::Nanosecond => micros.checked_mul(1_000),
    }
}

/// A value in `unit` as microseconds.
fn to_micros(value: i64, unit: TimeUnit) -> Option<i64> {
    match unit {
        TimeUnit::Second => value.checked_mul(1_000_000),
        TimeUnit::Millisecond => value.checked_mul(1_000),
        TimeUnit::Microsecond => Some(value),
        TimeUnit::Nanosecond => Some(value.div_euclid(1_000)),
    }
}

/// Microseconds back into `unit`, rounding toward −∞ so the mapping stays monotone.
fn from_micros(micros: i64, unit: TimeUnit) -> Option<i64> {
    match unit {
        TimeUnit::Second => Some(micros.div_euclid(1_000_000)),
        TimeUnit::Millisecond => Some(micros.div_euclid(1_000)),
        TimeUnit::Microsecond => Some(micros),
        TimeUnit::Nanosecond => micros.checked_mul(1_000),
    }
}

/// `Some` if `days_since_epoch` names a date the calendar can represent.
fn representable_day(days_since_epoch: i64) -> Option<chrono::NaiveDate> {
    let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1)?;
    epoch.checked_add_signed(chrono::Duration::try_days(days_since_epoch)?)
}

/// Add `months` calendar months to a day count since the epoch.
fn add_months_to_days(days_since_epoch: i64, months: i64) -> Option<i64> {
    let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1)?;
    let shifted = shift_months(representable_day(days_since_epoch)?, months)?;
    Some((shifted - epoch).num_days())
}

/// Add `months` calendar months to a microsecond count since the epoch, keeping the time of day.
fn add_months_to_micros(micros_since_epoch: i64, months: i64) -> Option<i64> {
    use chrono::DateTime;

    let dt = DateTime::from_timestamp_micros(micros_since_epoch)?.naive_utc();
    let shifted = shift_months(dt.date(), months)?;
    Some(shifted.and_time(dt.time()).and_utc().timestamp_micros())
}

/// `date` shifted by `months`, clamping to the last valid day of a shorter target month.
///
/// January 31 plus one month is February 28 (or 29), which is chrono's `checked_add_months`
/// rule and SQL's. It is also what makes the shift monotone: every day in a long month maps
/// into the shorter one without crossing over its neighbours' images.
fn shift_months(date: chrono::NaiveDate, months: i64) -> Option<chrono::NaiveDate> {
    use chrono::Months;

    let magnitude = u32::try_from(months.unsigned_abs()).ok()?;
    if months >= 0 {
        date.checked_add_months(Months::new(magnitude))
    } else {
        date.checked_sub_months(Months::new(magnitude))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY: i64 = MICROS_PER_DAY;

    fn ts() -> DataType {
        DataType::Timestamp(TimeUnit::Microsecond, None)
    }

    /// Days since the epoch for a civil date, as `Date32` holds it.
    fn day_of(y: i32, m: u32, d: u32) -> i64 {
        let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        (chrono::NaiveDate::from_ymd_opt(y, m, d).unwrap() - epoch).num_days()
    }

    #[test]
    fn exact_offsets_need_no_calendar() {
        let week = TypedOffset::new(0, 7, 0);
        assert!(week.is_exact());
        assert_eq!(week.shift_scalar(&DataType::Date32, 0, 1), Some(7));
        assert_eq!(week.shift_scalar(&DataType::Date32, 0, -1), Some(-7));
        assert_eq!(week.shift_scalar(&ts(), 0, 1), Some(7 * DAY));
    }

    #[test]
    fn a_sub_day_offset_on_a_date_is_declined_not_truncated() {
        // Agreeing with `bc_expr`'s `offset_by`, which errors rather than rounding.
        let half_day = TypedOffset::micros(DAY / 2);
        assert_eq!(half_day.shift_scalar(&DataType::Date32, 0, 1), None);
        assert_eq!(half_day.shift_scalar(&ts(), 0, 1), Some(DAY / 2));
    }

    #[test]
    fn month_end_clamps_to_the_last_valid_day() {
        let month = TypedOffset::new(1, 0, 0);
        // 2023-01-31 + 1 month = 2023-02-28 (2023 is not a leap year).
        let jan31 = day_of(2023, 1, 31);
        assert_eq!(
            month.shift_scalar(&DataType::Date32, jan31, 1),
            Some(day_of(2023, 2, 28))
        );
        // 2024 is a leap year, so the same shift lands on the 29th.
        let jan31_leap = day_of(2024, 1, 31);
        assert_eq!(
            month.shift_scalar(&DataType::Date32, jan31_leap, 1),
            Some(day_of(2024, 2, 29))
        );
    }

    #[test]
    fn a_leap_day_survives_a_four_year_shift() {
        let four_years = TypedOffset::new(48, 0, 0);
        let leap = day_of(2024, 2, 29);
        assert_eq!(
            four_years.shift_scalar(&DataType::Date32, leap, 1),
            Some(day_of(2028, 2, 29))
        );
        // ... but a one-year shift has nowhere to land, so it clamps.
        let one_year = TypedOffset::new(12, 0, 0);
        assert_eq!(
            one_year.shift_scalar(&DataType::Date32, leap, 1),
            Some(day_of(2025, 2, 28))
        );
    }

    #[test]
    fn shifting_is_monotone_including_across_a_month_end_clamp() {
        // The property the window frame's two-pointer depends on: a later row's frame edge is
        // never earlier than an earlier row's. The clamp is where that could break, so walk
        // every day of a long month into a short one.
        let month = TypedOffset::new(1, 0, 0);
        let mut previous = i64::MIN;
        for day in 1..=31 {
            let shifted = month
                .shift_scalar(&DataType::Date32, day_of(2023, 1, day), 1)
                .unwrap();
            assert!(shifted >= previous, "day {day} went backwards");
            previous = shifted;
        }
    }

    #[test]
    fn every_timestamp_unit_shifts_in_its_own_units() {
        let day = TypedOffset::new(0, 1, 0);
        for (unit, per_day) in [
            (TimeUnit::Second, 86_400_i64),
            (TimeUnit::Millisecond, 86_400_000),
            (TimeUnit::Microsecond, DAY),
        ] {
            let dt = DataType::Timestamp(unit, None);
            assert_eq!(day.shift_scalar(&dt, 0, 1), Some(per_day), "{unit:?}");
        }
        let nanos = DataType::Timestamp(TimeUnit::Nanosecond, None);
        assert_eq!(day.shift_scalar(&nanos, 0, 1), Some(DAY * 1_000));
    }

    #[test]
    fn a_duration_takes_an_exact_offset_and_refuses_a_calendar_one() {
        let day = TypedOffset::new(0, 1, 0);
        let unit = DataType::Duration(TimeUnit::Microsecond);
        assert!(day.supports(&unit));
        assert_eq!(day.shift_scalar(&unit, 5, 1), Some(5 + DAY));

        let month = TypedOffset::new(1, 0, 0);
        assert!(!month.supports(&unit));
        assert_eq!(month.shift_scalar(&unit, 5, 1), None);
    }

    #[test]
    fn overflow_is_none_rather_than_a_wrap_or_a_panic() {
        let huge = TypedOffset::new(0, i64::MAX / 2, 0);
        assert_eq!(huge.shift_scalar(&ts(), 0, 1), None);
        assert_eq!(huge.shift_scalar(&DataType::Date32, 0, 1), None);
        // A Date32 that leaves the i32 day range overflows even though the i64 math fits.
        let far = TypedOffset::new(0, i64::from(i32::MAX), 0);
        assert_eq!(far.shift_scalar(&DataType::Date32, 1, 1), None);
    }

    #[test]
    fn an_unsupported_type_is_declined_up_front() {
        let day = TypedOffset::new(0, 1, 0);
        assert!(!day.supports(&DataType::Utf8));
        assert!(!day.supports(&DataType::Int64));
        assert_eq!(day.shift_scalar(&DataType::Utf8, 0, 1), None);
    }

    #[test]
    fn adding_then_subtracting_returns_the_original_for_an_exact_offset() {
        let offset = TypedOffset::new(0, 3, 42);
        for value in [-10 * DAY, 0, 1, 999_999, 10 * DAY] {
            let there = offset.shift_scalar(&ts(), value, 1).unwrap();
            assert_eq!(offset.shift_scalar(&ts(), there, -1), Some(value));
        }
    }

    #[test]
    fn zero_and_sign_are_handled() {
        let zero = TypedOffset::default();
        assert!(zero.is_zero());
        assert_eq!(zero.shift_scalar(&ts(), 1_234, 1), Some(1_234));
        assert_eq!(zero.shift_scalar(&ts(), 1_234, -1), Some(1_234));
        // Only +1 and -1 are meaningful directions.
        assert_eq!(zero.shift_scalar(&ts(), 1_234, 0), None);
    }
}
