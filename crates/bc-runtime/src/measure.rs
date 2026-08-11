//! Reading an ordered key as a number, so two of them can be *subtracted*.
//!
//! Most of the engine only ever needs to know whether one key sorts before another, which
//! arrow's row encoding answers for any type. Two operators need more than that — they need
//! a distance:
//!
//! * an **ASOF join with a tolerance** (`join::asof`) asks "how stale is this match";
//! * a **value-based `RANGE` window frame** (`window::frame`) asks "which rows are within
//!   five minutes of this one".
//!
//! Both questions are the same question, so the answer lives here once rather than being
//! restated with two subtly different unit conventions. The convention is:
//!
//! * **every temporal type normalizes to microseconds** — a `Timestamp` of any resolution,
//!   a `Date32`, a `Date64` — so one tolerance or one interval works against any of them
//!   and a column stored in nanoseconds does not silently mean a thousand times more;
//! * an **integer** key keeps its own units, and stays integral, because the comparison a
//!   tolerance or a frame bound makes has to be exact at the top of the integer range: an
//!   `f64` cannot resolve a single unit past 2^53, which a nanosecond timestamp passed in
//!   1970 and a microsecond one reaches in 2255, and a boundary comparison would flip;
//! * a **float** key keeps its own units as `f64`.
//!
//! A type with no distance at all (a string, a struct) reads as `None`. That is not an
//! error on its own — an ordinary ASOF or a peer-bounded `RANGE` frame orders such a key
//! perfectly well — so the callers decide whether the absence is fatal for what they were
//! asked to do.

use arrow::array::{Array, ArrayRef, AsArray};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Float64Type, Int64Type, TimeUnit};

use crate::error::RuntimeError;

/// An ordered key column read as numbers, in the units documented on this module.
///
/// Null slots hold a placeholder rather than an `Option`, because every caller has already
/// decided what a null key means before it asks for a distance (an ASOF null matches
/// nothing; a null `RANGE` order key frames only its own peers), and an `Option` per row
/// would cost a branch in the inner loop for a case that never reaches it.
pub(crate) enum NumericKeys {
    Ints(Vec<i64>),
    Floats(Vec<f64>),
}

impl NumericKeys {
    /// Read `arr` as numbers, or `None` when the type has no distance.
    pub(crate) fn read(arr: &ArrayRef) -> Result<Option<Self>, RuntimeError> {
        let micros = DataType::Timestamp(TimeUnit::Microsecond, None);
        let ints = |a: &ArrayRef| -> Result<Vec<i64>, RuntimeError> {
            let c = cast(a, &DataType::Int64)?;
            let c = c.as_primitive::<Int64Type>();
            Ok((0..c.len())
                .map(|i| if c.is_valid(i) { c.value(i) } else { 0 })
                .collect())
        };
        Ok(match arr.data_type() {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Duration(_) => Some(NumericKeys::Ints(ints(arr)?)),
            DataType::Timestamp(..) | DataType::Date32 | DataType::Date64 => {
                Some(NumericKeys::Ints(ints(&cast(arr, &micros)?)?))
            }
            DataType::Float16 | DataType::Float32 | DataType::Float64 => {
                let f = cast(arr, &DataType::Float64)?;
                let f = f.as_primitive::<Float64Type>();
                Some(NumericKeys::Floats(
                    (0..f.len())
                        .map(|i| if f.is_valid(i) { f.value(i) } else { 0.0 })
                        .collect(),
                ))
            }
            _ => None,
        })
    }

    /// Compare the value at `j` against the value at `i` shifted by `delta` — the primitive
    /// a value-based `RANGE` frame's binary search is expressed in.
    ///
    /// Integer keys compare in `i128` and saturate, so neither the shift nor the comparison
    /// can overflow for an extreme timestamp and a large interval. Float keys use arrow's
    /// total order, matching how the order key was sorted and peer-grouped.
    pub(crate) fn shifted_cmp(&self, i: usize, delta: i128, j: usize) -> std::cmp::Ordering {
        match self {
            NumericKeys::Ints(v) => (v[j] as i128).cmp(&(v[i] as i128).saturating_add(delta)),
            NumericKeys::Floats(v) => v[j].total_cmp(&(v[i] + delta as f64)),
        }
    }

    /// The absolute distance between this key at `i` and `other` at `j`, in the key's units.
    ///
    /// Returns `None` when the two sides read as different kinds, which a well-typed plan
    /// cannot produce (the two columns of a join key share a supertype by construction).
    pub(crate) fn distance(&self, i: usize, other: &Self, j: usize) -> Option<f64> {
        match (self, other) {
            // `i128` so the subtraction cannot overflow for two extreme timestamps; what
            // survives is a duration, small enough to be exact in `f64`.
            (NumericKeys::Ints(a), NumericKeys::Ints(b)) => {
                Some((a[i] as i128 - b[j] as i128).unsigned_abs() as f64)
            }
            (NumericKeys::Floats(a), NumericKeys::Floats(b)) => Some((a[i] - b[j]).abs()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Date32Array, Float64Array, Int64Array, StringArray};
    use arrow::datatypes::TimestampNanosecondType;

    use super::*;

    /// A nanosecond timestamp must read in *microseconds*, or a `"5m"` tolerance would
    /// silently mean five milliseconds against it.
    #[test]
    fn every_temporal_resolution_reads_in_microseconds() {
        let nanos: ArrayRef = Arc::new(
            arrow::array::PrimitiveArray::<TimestampNanosecondType>::from(vec![
                1_000_000_000i64,
                2_000_000_000,
            ]),
        );
        let keys = NumericKeys::read(&nanos)
            .unwrap()
            .expect("timestamp is measurable");
        assert_eq!(keys.distance(0, &keys, 1), Some(1_000_000.0), "one second");

        // A date reads as microseconds since the epoch, so a day is 86_400_000_000.
        let dates: ArrayRef = Arc::new(Date32Array::from(vec![0, 3]));
        let keys = NumericKeys::read(&dates)
            .unwrap()
            .expect("date is measurable");
        assert_eq!(keys.distance(0, &keys, 1), Some(3.0 * 86_400_000_000.0));
    }

    /// The whole reason integer keys stay integral: past 2^53 an `f64` cannot resolve a
    /// single unit, so a boundary comparison against a tolerance would flip. A nanosecond
    /// timestamp is already there. The exact `i128` path resolves it.
    #[test]
    fn an_integer_distance_is_exact_past_the_f64_integer_range() {
        let big = 1_700_000_000_000_000_000i64; // ~2023 in nanoseconds, well past 2^53
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![big, big + 1]));
        let keys = NumericKeys::read(&arr).unwrap().unwrap();
        assert_eq!(keys.distance(0, &keys, 1), Some(1.0));
        // The naive f64 form loses it entirely: the two values are the same double.
        assert_eq!((big as f64) - ((big + 1) as f64), 0.0);
    }

    #[test]
    fn a_float_key_keeps_its_fraction_and_a_string_has_no_distance() {
        let arr: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 1.25]));
        let keys = NumericKeys::read(&arr).unwrap().unwrap();
        assert_eq!(keys.distance(0, &keys, 1), Some(0.25));

        let strs: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        assert!(NumericKeys::read(&strs).unwrap().is_none());
    }

    /// The shifted comparison must not overflow at the extremes of the integer domain,
    /// where a large `RANGE` interval added to a large timestamp would wrap.
    #[test]
    fn a_shifted_comparison_saturates_instead_of_wrapping() {
        use std::cmp::Ordering;
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![i64::MAX, i64::MIN, 0]));
        let keys = NumericKeys::read(&arr).unwrap().unwrap();
        // `i64::MAX + a huge delta` still sits above every other value.
        assert_eq!(keys.shifted_cmp(0, i64::MAX as i128, 2), Ordering::Less);
        // `i64::MIN - a huge delta` still sits below every other value.
        assert_eq!(
            keys.shifted_cmp(1, -(i64::MAX as i128), 2),
            Ordering::Greater
        );
        // A zero shift is a plain comparison against the row's own value.
        assert_eq!(keys.shifted_cmp(2, 0, 2), Ordering::Equal);
    }

    /// A `RANGE` bound reads the same shifted comparison whichever unit the key is in.
    #[test]
    fn a_shifted_comparison_finds_the_interval_edge() {
        use std::cmp::Ordering;
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![10, 12, 15, 20]));
        let keys = NumericKeys::read(&arr).unwrap().unwrap();
        // Rows within 5 *before* row 3 (value 20): the edge is 15.
        assert_eq!(keys.shifted_cmp(3, -5, 2), Ordering::Equal, "15 == 20 - 5");
        assert_eq!(keys.shifted_cmp(3, -5, 1), Ordering::Less, "12 < 20 - 5");
    }
}
