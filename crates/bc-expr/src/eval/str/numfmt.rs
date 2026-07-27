//! String functions whose input is a **number**, not a string.
//!
//! `chr`, `to_base`/`bin`, `format_bytes` and `hex` of an integer all map Int → Utf8, so
//! they cannot go through `eval_str`'s Utf8 downcast — it would reject the column before
//! the kernel is reached. They are dispatched here instead, from the same place the
//! `Binary` family is: before that downcast, returning `None` for every function that is
//! not one of theirs so the ordinary string path is untouched.
//!
//! Any integer width is accepted (the column is cast to Int64 first), which is what makes
//! `chr(65)` work on a literal and on an Int32 column alike.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int64Array, StringArray};
use arrow::compute::cast;
use arrow::datatypes::DataType;

use crate::{ExprError, StrFunc};

/// Binary (IEC) and decimal (SI) unit ladders for the two `format_bytes` spellings.
// The base unit is the word `bytes`, not `B` — DuckDB writes `512 bytes` and only uses
// the letter for the scaled units (`1.5 KiB`).
const IEC_UNITS: [&str; 7] = ["bytes", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];
const SI_UNITS: [&str; 7] = ["bytes", "kB", "MB", "GB", "TB", "PB", "EB"];

/// Evaluate a numeric-input string function, or `None` when `func` is not one of them.
///
/// `start` carries the radix for [`StrFunc::ToBase`] (DuckDB's second argument), matching
/// how the other `StrFunc`s reuse that field for an integer parameter.
pub(super) fn eval_numeric_input(
    func: StrFunc,
    arr: &ArrayRef,
    start: Option<i64>,
) -> Result<Option<ArrayRef>, ExprError> {
    if !matches!(
        func,
        StrFunc::Chr
            | StrFunc::ToBase
            | StrFunc::FormatBytes
            | StrFunc::FormatBytesSi
            | StrFunc::Hex
    ) {
        return Ok(None);
    }
    if !arr.data_type().is_integer() {
        // `hex` is defined on strings and blobs too, and those paths own it; only an
        // integer argument belongs here.
        return Ok(None);
    }
    let ints = cast(arr, &DataType::Int64)?;
    let a =
        ints.as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| ExprError::ExpectedString {
                func: format!("{func:?}"),
                got: arr.data_type().to_string(),
            })?;

    let radix = match func {
        StrFunc::ToBase => {
            let base = start.unwrap_or(10);
            if !(2..=36).contains(&base) {
                return Err(ExprError::MissingArgument {
                    func: "to_base".into(),
                    arg: "a radix between 2 and 36",
                });
            }
            base as u32
        }
        StrFunc::Hex => 16,
        _ => 10,
    };

    let out: StringArray = (0..a.len())
        .map(|i| {
            if a.is_null(i) {
                return None;
            }
            let v = a.value(i);
            Some(match func {
                StrFunc::Chr => char::from_u32(u32::try_from(v).ok()?)?.to_string(),
                StrFunc::ToBase => to_base(v, radix),
                StrFunc::Hex => to_base(v, 16),
                StrFunc::FormatBytes => format_bytes(v, 1024.0, &IEC_UNITS),
                _ => format_bytes(v, 1000.0, &SI_UNITS),
            })
        })
        .collect();
    Ok(Some(Arc::new(out) as ArrayRef))
}

/// `value` in `radix`, sign-prefixed, with **uppercase** letter digits — DuckDB's
/// `to_base(255, 16)` is `FF`, and its `hex` agrees.
///
/// The magnitude is taken as `u64` so `i64::MIN` — whose negation overflows — converts
/// like every other value instead of panicking.
fn to_base(value: i64, radix: u32) -> String {
    let negative = value < 0;
    let mut magnitude = value.unsigned_abs();
    if magnitude == 0 {
        return "0".to_string();
    }
    let digits: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let mut out = Vec::new();
    while magnitude > 0 {
        out.push(digits[(magnitude % radix as u64) as usize]);
        magnitude /= radix as u64;
    }
    if negative {
        out.push(b'-');
    }
    out.reverse();
    String::from_utf8(out).unwrap_or_default()
}

/// A byte count as `"<value> <unit>"` with one decimal place, climbing the unit ladder
/// while the magnitude is at least `step`. Matches DuckDB: exact bytes carry no decimal
/// (`512` → `512 B`), larger values carry one (`1024` → `1.0 KiB`).
fn format_bytes(value: i64, step: f64, units: &[&str; 7]) -> String {
    let negative = value < 0;
    let mut magnitude = value.unsigned_abs() as f64;
    let mut unit = 0;
    while magnitude >= step && unit + 1 < units.len() {
        magnitude /= step;
        unit += 1;
    }
    let sign = if negative { "-" } else { "" };
    if unit == 0 {
        return format!("{sign}{} {}", value.unsigned_abs(), units[0]);
    }
    // Truncated, not rounded: DuckDB writes 8,364 bytes as `8.1 KiB` (8.167…), and
    // rounding would say 8.2.
    let tenths = (magnitude * 10.0).floor() / 10.0;
    format!("{sign}{tenths:.1} {}", units[unit])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(func: StrFunc, values: Vec<i64>, start: Option<i64>) -> Vec<Option<String>> {
        let arr: ArrayRef = Arc::new(Int64Array::from(values));
        let out = eval_numeric_input(func, &arr, start).unwrap().unwrap();
        let s = out.as_any().downcast_ref::<StringArray>().unwrap();
        (0..s.len())
            .map(|i| (!s.is_null(i)).then(|| s.value(i).to_string()))
            .collect()
    }

    #[test]
    fn chr_maps_a_code_point_to_its_character() {
        assert_eq!(
            call(StrFunc::Chr, vec![65, 0x1F600, 233], None),
            vec![Some("A".into()), Some("😀".into()), Some("é".into()),]
        );
    }

    #[test]
    fn chr_of_a_value_that_is_not_a_code_point_is_null_not_a_panic() {
        // Surrogates and out-of-range values have no character; DuckDB errors, and null
        // is the engine's rule for an unrepresentable conversion on a data path.
        assert_eq!(
            call(StrFunc::Chr, vec![0xD800, -1, 0x110000], None),
            vec![None, None, None]
        );
    }

    #[test]
    fn to_base_covers_the_radix_range_and_negatives() {
        assert_eq!(
            call(StrFunc::ToBase, vec![15], Some(2)),
            vec![Some("1111".into())]
        );
        assert_eq!(
            call(StrFunc::ToBase, vec![255], Some(16)),
            vec![Some("FF".into())]
        );
        assert_eq!(
            call(StrFunc::ToBase, vec![-42], Some(10)),
            vec![Some("-42".into())]
        );
        assert_eq!(
            call(StrFunc::ToBase, vec![0], Some(36)),
            vec![Some("0".into())]
        );
    }

    #[test]
    fn to_base_of_the_smallest_integer_does_not_overflow() {
        // `-i64::MIN` overflows; the magnitude is taken as u64 for exactly this row.
        let got = call(StrFunc::ToBase, vec![i64::MIN], Some(10));
        assert_eq!(got, vec![Some(i64::MIN.to_string())]);
    }

    #[test]
    fn hex_of_an_integer_is_uppercase() {
        assert_eq!(
            call(StrFunc::Hex, vec![255, 16], None),
            vec![Some("FF".into()), Some("10".into())]
        );
    }

    #[test]
    fn format_bytes_truncates_rather_than_rounding() {
        // 8,364 / 1024 is 8.167…; DuckDB writes 8.1, not 8.2.
        assert_eq!(
            call(StrFunc::FormatBytes, vec![8364], None),
            vec![Some("8.1 KiB".into())]
        );
    }

    #[test]
    fn format_bytes_climbs_the_binary_and_decimal_ladders() {
        assert_eq!(
            call(StrFunc::FormatBytes, vec![512, 1024, 1536, 1_048_576], None),
            vec![
                Some("512 bytes".into()),
                Some("1.0 KiB".into()),
                Some("1.5 KiB".into()),
                Some("1.0 MiB".into()),
            ]
        );
        assert_eq!(
            call(StrFunc::FormatBytesSi, vec![1000, 1_000_000], None),
            vec![Some("1.0 kB".into()), Some("1.0 MB".into())]
        );
    }

    #[test]
    fn a_string_input_is_declined_so_the_ordinary_path_keeps_it() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["abc"]));
        assert!(eval_numeric_input(StrFunc::Hex, &arr, None)
            .unwrap()
            .is_none());
    }

    #[test]
    fn an_out_of_range_radix_is_an_error_not_a_wrong_answer() {
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![1]));
        assert!(eval_numeric_input(StrFunc::ToBase, &arr, Some(37)).is_err());
        assert!(eval_numeric_input(StrFunc::ToBase, &arr, Some(1)).is_err());
    }
}
