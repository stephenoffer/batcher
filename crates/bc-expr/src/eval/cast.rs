//! `cast` evaluation with DuckDB float→int rounding semantics.
//!
//! Split out of `lib.rs` for file size; the `Expr::Cast` variant and its wire tags
//! stay in `lib.rs`. Behavior is unchanged.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, BooleanArray, Float64Array};
use arrow::compute::{cast_with_options, CastOptions};
use arrow::datatypes::Float64Type;
use arrow::error::ArrowError;

use crate::ExprError;

/// Cast `arr` to `target` with DuckDB float→int semantics. Arrow's float→int cast
/// truncates toward zero; DuckDB's `DOUBLE → BIGINT` rounds half-to-**even** (`cast(2.5)`
/// = 2, `cast(3.5)` = 4, `cast(0.5)` = 0), so float inputs are rounded to an integral
/// value before the cast. (Note: DuckDB's `DECIMAL → BIGINT` instead rounds half-*away*,
/// but the engine's cast operates on Float64 columns, i.e. the DOUBLE path.) All other
/// casts defer to the arrow kernel unchanged. (The JIT never compiles float→int, so this
/// interpreter-only behavior keeps tier parity intact.)
///
/// `try_cast` selects arrow's *safe* cast (a value that cannot be converted
/// becomes NULL — DuckDB `TRY_CAST`); the strict default (`false`) errors on an
/// invalid value (DuckDB `CAST`).
pub(crate) fn cast_expr(
    arr: &ArrayRef,
    target: &arrow::datatypes::DataType,
    try_cast: bool,
) -> Result<ArrayRef, ExprError> {
    use arrow::datatypes::DataType;
    use arrow::datatypes::DataType::{
        Float16, Float32, Float64, Int16, Int32, Int64, Int8, UInt16, UInt32, UInt64, UInt8,
    };
    // Identity cast: hand the array straight back. `a / b` lowers to
    // `div(cast(a, float64), b)` so integer operands divide truly, and that cast is
    // a no-op whenever `a` is already Float64 — the common case in a float pipeline.
    // Arrow's kernel would still walk and rebuild the array; this makes it free.
    if arr.data_type() == target {
        return Ok(Arc::clone(arr));
    }
    let opts = CastOptions {
        safe: try_cast,
        ..Default::default()
    };
    // String→Boolean: arrow's `Utf8 → Boolean` kernel is far looser than DuckDB — it
    // trims whitespace, accepts `on`/`off`, and even matches a *prefix* (`'tru'` → true),
    // so `TRY_CAST('tru' AS BOOLEAN)` silently returns `true` where DuckDB returns NULL —
    // a wrong non-null value on the safe-ingest path. DuckDB accepts exactly (ASCII
    // case-insensitive, no trimming) `{true, false, t, f, 1, 0, yes, no, y, n}`; anything
    // else is NULL (`TRY_CAST`) or an error (`CAST`). Parse against that exact set. Not
    // JIT-compiled, so tier parity is unaffected.
    if matches!(arr.data_type(), DataType::Utf8 | DataType::LargeUtf8)
        && target == &DataType::Boolean
    {
        return parse_string_to_bool(arr, try_cast);
    }
    // Float→string: DuckDB renders a NaN as the lowercase `nan` and a *negative* zero as
    // `0.0`, where arrow's formatter emits `NaN` and `-0.0` — wrong non-null strings on any
    // `CAST(<float> AS VARCHAR)`. Normalize just those two format-independent cases and keep
    // arrow's shortest-round-trip string for every other value. (Arrow and DuckDB still differ
    // on *scientific* notation — its threshold and its `e+NN` exponent form, e.g. `1e-5` vs
    // `0.00001`, `1e+20` vs `1e20`; closing that needs a dedicated `%g`-style formatter and is
    // left as a known gap.) Not JIT-compiled, so tier parity is intact.
    if matches!(arr.data_type(), Float16 | Float32 | Float64)
        && matches!(target, DataType::Utf8 | DataType::LargeUtf8)
    {
        return float_to_string(arr, target, &opts);
    }
    // Timestamp→string: arrow writes the ISO `T` separator and a fixed-width sub-second
    // field (`2021-01-02T03:04:05.500`); DuckDB — like Postgres, Spark and Polars — writes
    // a space and trims the fraction's trailing zeros (`2021-01-02 03:04:05.5`), omitting
    // it entirely when it is zero. Every `CAST(<timestamp> AS VARCHAR)`, and so every
    // concatenation or string comparison built on one, differed in both places.
    if matches!(arr.data_type(), DataType::Timestamp(_, None))
        && matches!(target, DataType::Utf8 | DataType::LargeUtf8)
    {
        return timestamp_to_string(arr, target, &opts);
    }
    // DuckDB trims leading/trailing whitespace before parsing a string into a
    // numeric or temporal value: `CAST('  12  ' AS BIGINT)` = 12, `' 3.14 '::DOUBLE`
    // = 3.14, `' 2024-01-05 '::DATE` = 2024-01-05. Arrow's kernel does not trim, so
    // the strict cast errors and `TRY_CAST` silently NULLs the padded value — data
    // loss on the advertised safe-ingest path. Trim the surrounding C-`isspace`
    // whitespace (space, `\t`, `\n`, `\x0b`, `\x0c`, `\r`) for string→{integer, float,
    // decimal, date, timestamp}, then let arrow parse the trimmed value (so its
    // overflow / invalid handling still applies). Boolean is excluded — DuckDB does
    // NOT trim there (`' true '::BOOLEAN` is NULL). Trimming only the outer ASCII
    // whitespace bytes never splits a UTF-8 codepoint. String parsing is never
    // JIT-compiled, so this interpreter-only path keeps tier parity intact.
    let int_target = matches!(
        target,
        Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64
    );
    if wants_trimmed_string_parse(arr.data_type(), target) {
        if let Some(trimmed) = trim_string_array(arr) {
            // DuckDB's `VARCHAR → <integer>` accepts a *fractional* or scientific value and
            // rounds it half-away from zero (`'1.5'→2`, `'2.5'→3`, `'-2.5'→-3`, `'1e3'→1000`,
            // `'12345.678'→12346`) — the same rounding it uses for `DECIMAL → <integer>`.
            // Arrow's integer parser rejects any non-integer string, so before this strict
            // `CAST` errored and `TRY_CAST` silently NULLed such a value. Route string→integer
            // through [`parse_string_to_int`], which keeps a clean integer string exact (no
            // f64 precision loss above 2^53) and only falls back to the rounded-double path for
            // the strings the exact parser rejects. Not JIT-compiled, so tier parity is intact.
            if int_target {
                return parse_string_to_int(&trimmed, target, try_cast);
            }
            return Ok(cast_with_options(&trimmed, target, &opts)?);
        }
    }
    let float_src = matches!(arr.data_type(), Float16 | Float32 | Float64);
    // DuckDB's `DECIMAL → <integer>` rounds half-**away**-from-zero (`2.5 → 3`,
    // `-2.5 → -3`, `0.5 → 1`), where arrow's kernel truncates toward zero (`2.5 → 2`).
    // Round the decimal to an integral (scale-0) decimal first, then let arrow cast it
    // (so its overflow handling — strict error / safe NULL — still applies). Floats keep
    // the DOUBLE half-to-even path below; the JIT never compiles a decimal cast, so this
    // stays interpreter-only and tier parity is intact.
    if int_target {
        if let Some(rounded) = round_decimal_to_integral(arr) {
            return Ok(cast_with_options(&rounded, target, &opts)?);
        }
    }
    if int_target && float_src {
        // Round half-to-even first (DuckDB DOUBLE→BIGINT), then cast the now-integral
        // floats. `f64::round_ties_even` is banker's rounding.
        let f = cast_with_options(arr, &Float64, &opts)?;
        let f = f.as_primitive::<Float64Type>();
        // `arity::unary` maps the values buffer and reuses the null buffer, where the
        // `Option`-per-row collect it replaces rebuilt the validity bitmap a bit at a time.
        // Rounding a null slot's arbitrary payload is harmless — the reused buffer masks it,
        // and `round_ties_even` cannot trap.
        // Measured, 20M Float64 with 30 % nulls: **19.7 ms -> 16.8 ms**.
        let rounded: Float64Array = arrow::compute::kernels::arity::unary(f, f64::round_ties_even);
        let rounded: ArrayRef = Arc::new(rounded);
        return Ok(cast_with_options(&rounded, target, &opts)?);
    }
    // Narrowing a wider float to a smaller one (`f64 -> f32`/`f16`, `f32 -> f16`):
    // arrow's kernel rounds an out-of-range *finite* value silently to ±inf, so
    // `cast(1e300 as float)` yields `inf`. DuckDB instead treats that as an overflow —
    // strict `CAST` errors, `TRY_CAST` yields NULL — and never fabricates an infinity
    // from a finite input. Detect the finite→infinite transition and honor whichever
    // DuckDB semantics `try_cast` selects. (A genuine ±inf input stays ±inf, matching
    // DuckDB.) The JIT never compiles a narrowing-float cast, so this stays interp-only.
    let narrowing_float = matches!(
        (arr.data_type(), target),
        (&DataType::Float64, Float32 | Float16) | (&DataType::Float32, Float16)
    );
    if narrowing_float {
        let out = cast_with_options(arr, target, &opts)?;
        let src64 = cast_with_options(arr, &Float64, &opts)?;
        let src64 = src64.as_primitive::<Float64Type>();
        let out64 = cast_with_options(&out, &Float64, &opts)?;
        let out64 = out64.as_primitive::<Float64Type>();
        let overflow: BooleanArray = (0..src64.len())
            .map(|i| {
                Some(
                    !src64.is_null(i) && src64.value(i).is_finite() && out64.value(i).is_infinite(),
                )
            })
            .collect();
        if overflow.true_count() > 0 {
            if try_cast {
                return Ok(arrow::compute::nullif(&out, &overflow)?);
            }
            return Err(ExprError::Arrow(ArrowError::CastError(format!(
                "Can't cast value out of range to type {target}"
            ))));
        }
        return Ok(out);
    }
    Ok(cast_with_options(arr, target, &opts)?)
}

/// Parse one string to a bool the way DuckDB's `VARCHAR → BOOLEAN` cast does: ASCII
/// case-insensitive, no whitespace trimming, accepting exactly `true/t/1/yes/y` (true) and
/// `false/f/0/no/n` (false). Any other token — including `on`/`off`, a padded value, or a
/// prefix like `tru` — is `None` (the caller then NULLs or errors it).
fn parse_bool_token(s: &str) -> Option<bool> {
    if ["true", "t", "1", "yes", "y"]
        .iter()
        .any(|w| s.eq_ignore_ascii_case(w))
    {
        Some(true)
    } else if ["false", "f", "0", "no", "n"]
        .iter()
        .any(|w| s.eq_ignore_ascii_case(w))
    {
        Some(false)
    } else {
        None
    }
}

/// Cast a `Utf8`/`LargeUtf8` array to `Boolean` with DuckDB's exact accepted set (see
/// [`parse_bool_token`]). Nulls pass through; an unrecognized token becomes NULL when
/// `try_cast`, else errors the cast (DuckDB strict `CAST`).
fn parse_string_to_bool(arr: &ArrayRef, try_cast: bool) -> Result<ArrayRef, ExprError> {
    use arrow::array::BooleanBuilder;
    use arrow::datatypes::DataType;
    let mut builder = BooleanBuilder::with_capacity(arr.len());
    let mut push = |opt: Option<&str>| -> Result<(), ExprError> {
        match opt {
            None => builder.append_null(),
            Some(s) => match parse_bool_token(s) {
                Some(v) => builder.append_value(v),
                None if try_cast => builder.append_null(),
                None => {
                    return Err(ExprError::Arrow(ArrowError::CastError(format!(
                        "Cannot cast string '{s}' to value of Boolean type"
                    ))))
                }
            },
        }
        Ok(())
    };
    match arr.data_type() {
        DataType::Utf8 => {
            let a = arr.as_string::<i32>();
            for o in a.iter() {
                push(o)?;
            }
        }
        DataType::LargeUtf8 => {
            let a = arr.as_string::<i64>();
            for o in a.iter() {
                push(o)?;
            }
        }
        _ => unreachable!("parse_string_to_bool only called for Utf8/LargeUtf8 source"),
    }
    Ok(Arc::new(builder.finish()) as ArrayRef)
}

/// A boundary whitespace char DuckDB strips before parsing a string value: the C
/// `isspace` set (space, horizontal tab, line feed, vertical tab, form feed, carriage
/// return). All are single-byte ASCII, so trimming them can never split a UTF-8 codepoint.
fn is_parse_whitespace(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\x0b' | '\x0c' | '\r')
}

/// True when a `src → target` cast parses a string into a numeric or temporal *value*,
/// the casts for which DuckDB trims surrounding whitespace first. Boolean is deliberately
/// excluded (DuckDB does not trim there), as are non-string sources.
fn wants_trimmed_string_parse(
    src: &arrow::datatypes::DataType,
    target: &arrow::datatypes::DataType,
) -> bool {
    use arrow::datatypes::DataType;
    if !matches!(src, DataType::Utf8 | DataType::LargeUtf8) {
        return false;
    }
    matches!(
        target,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
            | DataType::Date32
            | DataType::Date64
            | DataType::Timestamp(_, _)
    )
}

/// Trim boundary whitespace from every value of a `Utf8`/`LargeUtf8` array, preserving the
/// null mask and array type. Returns `None` for a non-string array (the caller then leaves
/// `arr` untouched). Nulls stay null; a value that is all-whitespace becomes `""` (which the
/// downstream arrow parse rejects → NULL/error, exactly as DuckDB treats `'   '::BIGINT`).
fn trim_string_array(arr: &ArrayRef) -> Option<ArrayRef> {
    use arrow::array::{LargeStringArray, StringArray};
    use arrow::datatypes::DataType;
    match arr.data_type() {
        DataType::Utf8 => {
            let a = arr.as_any().downcast_ref::<StringArray>()?;
            let out: StringArray = a
                .iter()
                .map(|o| o.map(|s| s.trim_matches(is_parse_whitespace)))
                .collect();
            Some(Arc::new(out) as ArrayRef)
        }
        DataType::LargeUtf8 => {
            let a = arr.as_any().downcast_ref::<LargeStringArray>()?;
            let out: LargeStringArray = a
                .iter()
                .map(|o| o.map(|s| s.trim_matches(is_parse_whitespace)))
                .collect();
            Some(Arc::new(out) as ArrayRef)
        }
        _ => None,
    }
}

/// Round a `Decimal128`/`Decimal256` array to an integral (scale-0) decimal, half-away
/// from zero — the rounding DuckDB applies for `DECIMAL → <integer>`. Returns `None` for
/// a non-decimal source (the caller then uses its float / kernel paths) and for an
/// already-integral (`scale <= 0`) decimal, whose arrow int cast is exact. The result
/// keeps the source precision at scale 0, so the caller's arrow cast handles overflow.
fn round_decimal_to_integral(arr: &ArrayRef) -> Option<ArrayRef> {
    use arrow::array::{Decimal128Array, Decimal256Array};
    use arrow::datatypes::{i256, ArrowNativeTypeOp, DataType};
    match arr.data_type() {
        DataType::Decimal128(p, s) if *s > 0 => {
            let a = arr.as_any().downcast_ref::<Decimal128Array>()?;
            let factor = 10i128.pow(*s as u32);
            let rounded: Decimal128Array = a
                .iter()
                .map(|o| {
                    o.map(|u| {
                        let (q, r) = (u / factor, u % factor);
                        // r carries the sign of u; round away from zero on a half-or-more.
                        if r.unsigned_abs() * 2 >= factor.unsigned_abs() {
                            q + u.signum()
                        } else {
                            q
                        }
                    })
                })
                .collect::<Decimal128Array>()
                .with_precision_and_scale(*p, 0)
                .ok()?;
            Some(Arc::new(rounded) as ArrayRef)
        }
        DataType::Decimal256(p, s) if *s > 0 => {
            let a = arr.as_any().downcast_ref::<Decimal256Array>()?;
            let factor = i256::from_i128(10).pow_checked(*s as u32).ok()?;
            let two = i256::from_i128(2);
            let rounded: Decimal256Array = a
                .iter()
                .map(|o| {
                    o.map(|u| {
                        let q = u / factor;
                        let r = u % factor;
                        let sign = if u < i256::ZERO {
                            i256::from_i128(-1)
                        } else {
                            i256::from_i128(1)
                        };
                        // |r| * 2 >= |factor|  (factor > 0), avoiding a separate abs.
                        if (r * sign) * two >= factor {
                            q + sign
                        } else {
                            q
                        }
                    })
                })
                .collect::<Decimal256Array>()
                .with_precision_and_scale(*p, 0)
                .ok()?;
            Some(Arc::new(rounded) as ArrayRef)
        }
        _ => None,
    }
}

/// Cast a trimmed `Utf8`/`LargeUtf8` array to an integer `target` with DuckDB's parse
/// semantics: a clean integer string parses exactly, and a *fractional* or scientific string
/// (`'1.5'`, `'1e3'`, `'12345.678'`) is read as a double and rounded half-away from zero
/// (`f64::round`) — the same rule DuckDB uses for `DECIMAL → <integer>`. The exact integer
/// parse is tried first and preferred, so an integer string wider than 2^53 stays exact rather
/// than losing precision through the f64 fallback. A row that parses as neither (`'abc'`, `''`,
/// `'inf'`, or a value out of the target's range) becomes NULL under `try_cast`, else errors
/// the whole cast — matching DuckDB's `TRY_CAST` / `CAST`.
fn parse_string_to_int(
    trimmed: &ArrayRef,
    target: &arrow::datatypes::DataType,
    try_cast: bool,
) -> Result<ArrayRef, ExprError> {
    use arrow::compute::{is_not_null, kernels::zip::zip};
    // `safe` so a parse failure is a NULL to reason about, not an error to bail on.
    let safe = CastOptions {
        safe: true,
        ..Default::default()
    };
    // Exact integer parse: non-null only where the string is a clean, in-range integer.
    let exact = cast_with_options(trimmed, target, &safe)?;
    // Float fallback for the rest: parse as f64, round half-away (`f64::round`), then to the
    // integer type (which NULLs an out-of-range or non-finite value under `safe`).
    let f = cast_with_options(trimmed, &arrow::datatypes::DataType::Float64, &safe)?;
    let f = f.as_primitive::<Float64Type>();
    let rounded: Float64Array = f.iter().map(|o| o.map(f64::round)).collect();
    let fallback = cast_with_options(&(Arc::new(rounded) as ArrayRef), target, &safe)?;
    // Prefer the exact parse; use the rounded double only where the exact parse yielded NULL.
    let has_exact = is_not_null(&exact)?;
    let merged = zip(&has_exact, &exact, &fallback)?;
    if try_cast {
        return Ok(merged);
    }
    // Strict `CAST`: a row that was non-null on input but is NULL after both attempts is an
    // unparseable / out-of-range value — error, don't silently drop it.
    let src_present = is_not_null(trimmed)?;
    let failed = arrow::compute::and(&src_present, &arrow::compute::not(&is_not_null(&merged)?)?)?;
    if failed.true_count() > 0 {
        return Err(ExprError::Arrow(ArrowError::CastError(format!(
            "Could not cast string to {target}"
        ))));
    }
    Ok(merged)
}

/// Cast a `Float16`/`Float32`/`Float64` array to a string `target`, normalizing the one case
/// where arrow's formatter disagrees with DuckDB: a NaN renders as `nan`, not `NaN`. Every
/// other value keeps arrow's shortest-round-trip string. Nulls pass through.
///
/// A negative zero keeps its sign. This previously rendered `-0.0` as `0.0`, on the stated
/// grounds that DuckDB does — which is true only of a *constant-folded literal*
/// (`SELECT (-0.0)::VARCHAR` → `0.0`, because the parser folds the sign away). For an actual
/// `-0.0` in a column, DuckDB emits `-0.0`, as do Polars, Arrow's own formatter, and Python's
/// `str`. The engine folds `-0.0` to `0.0` for *key identity* (grouping, joins, ordering);
/// that is about which rows are equal, not about how a value is displayed, and `sign(x)` and
/// `1/x` still distinguish the two. Rendering it away lost information every oracle keeps.
/// Render a naive `Timestamp` array the way DuckDB's `CAST(... AS VARCHAR)` does.
///
/// Built from arrow's own string, not from a re-derived calendar date: arrow already
/// handles the epoch arithmetic, the pre-1970 direction and the year padding, and the two
/// engines disagree only about punctuation. Replacing the `T` and trimming the fraction
/// keeps this a formatting fix rather than a second date implementation to keep in step.
fn timestamp_to_string(
    arr: &ArrayRef,
    target: &arrow::datatypes::DataType,
    opts: &CastOptions,
) -> Result<ArrayRef, ExprError> {
    use arrow::array::{LargeStringArray, StringArray};
    use arrow::datatypes::DataType;
    let strs = cast_with_options(arr, target, opts)?;
    match target {
        DataType::Utf8 => {
            let a = strs.as_string::<i32>();
            let out: StringArray = (0..a.len())
                .map(|i| (!a.is_null(i)).then(|| duckdb_timestamp_text(a.value(i))))
                .collect();
            Ok(Arc::new(out) as ArrayRef)
        }
        DataType::LargeUtf8 => {
            let a = strs.as_string::<i64>();
            let out: LargeStringArray = (0..a.len())
                .map(|i| (!a.is_null(i)).then(|| duckdb_timestamp_text(a.value(i))))
                .collect();
            Ok(Arc::new(out) as ArrayRef)
        }
        _ => unreachable!("timestamp_to_string only called for Utf8/LargeUtf8 target"),
    }
}

/// One arrow timestamp string in DuckDB's spelling: `T` → space, fraction de-padded.
///
/// A fraction that trims away entirely takes its `.` with it, because DuckDB writes a
/// whole second as `03:04:05` rather than `03:04:05.`.
fn duckdb_timestamp_text(s: &str) -> String {
    let spaced = s.replacen('T', " ", 1);
    let Some(dot) = spaced.rfind('.') else {
        return spaced;
    };
    let (head, frac) = spaced.split_at(dot);
    let trimmed = frac[1..].trim_end_matches('0');
    if trimmed.is_empty() {
        head.to_string()
    } else {
        format!("{head}.{trimmed}")
    }
}

fn float_to_string(
    arr: &ArrayRef,
    target: &arrow::datatypes::DataType,
    opts: &CastOptions,
) -> Result<ArrayRef, ExprError> {
    use arrow::array::{LargeStringArray, StringArray};
    use arrow::datatypes::DataType;
    let strs = cast_with_options(arr, target, opts)?;
    let f = cast_with_options(arr, &DataType::Float64, opts)?;
    let f = f.as_primitive::<Float64Type>();
    // Map arrow's string for row `i` to DuckDB's, given the row's float value.
    let fix = |i: usize, s: &str| -> String {
        let v = f.value(i);
        if v.is_nan() {
            "nan".to_string()
        } else {
            s.to_string()
        }
    };
    match target {
        DataType::Utf8 => {
            let a = strs.as_string::<i32>();
            let out: StringArray = (0..a.len())
                .map(|i| (!a.is_null(i)).then(|| fix(i, a.value(i))))
                .collect();
            Ok(Arc::new(out) as ArrayRef)
        }
        DataType::LargeUtf8 => {
            let a = strs.as_string::<i64>();
            let out: LargeStringArray = (0..a.len())
                .map(|i| (!a.is_null(i)).then(|| fix(i, a.value(i))))
                .collect();
            Ok(Arc::new(out) as ArrayRef)
        }
        _ => unreachable!("float_to_string only called for Utf8/LargeUtf8 target"),
    }
}

#[cfg(test)]
mod narrowing_float_tests {
    use super::*;
    // Assertion-only type tags: unused in the lib target, so scoped to the tests that
    // need them rather than left at module scope as dead code under `-D warnings`.
    use arrow::datatypes::{DataType, Float32Type, Int64Type};

    fn f64arr(v: Vec<Option<f64>>) -> ArrayRef {
        Arc::new(Float64Array::from(v))
    }

    /// A *finite* `f64` too large for `f32` is an overflow, not an infinity:
    /// `TRY_CAST(1e300 AS REAL)` must yield NULL (DuckDB), where arrow's kernel
    /// silently rounds it to `+inf`. A genuine `inf` input passes through unchanged.
    #[test]
    fn try_cast_finite_overflow_to_f32_is_null_not_inf() {
        let src = f64arr(vec![
            Some(1.5),
            Some(1e300),
            Some(-1e300),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
            None,
        ]);
        let out = cast_expr(&src, &DataType::Float32, true).unwrap();
        let out = out.as_primitive::<Float32Type>();
        let got: Vec<Option<f32>> = (0..out.len())
            .map(|i| (!out.is_null(i)).then(|| out.value(i)))
            .collect();
        assert_eq!(
            got,
            vec![
                Some(1.5f32),
                None,                    // 1e300 overflows → NULL, not +inf
                None,                    // -1e300 overflows → NULL, not -inf
                Some(f32::INFINITY),     // a real inf stays inf
                Some(f32::NEG_INFINITY), // a real -inf stays -inf
                None,                    // NULL passes through
            ]
        );
    }

    /// Strict `CAST(1e300 AS REAL)` must error on the finite overflow rather than
    /// silently producing `+inf` (DuckDB "value out of range").
    #[test]
    fn strict_cast_finite_overflow_to_f32_errors() {
        let src = f64arr(vec![Some(1e300)]);
        assert!(cast_expr(&src, &DataType::Float32, false).is_err());
        // An in-range value still casts fine under strict mode.
        let ok = cast_expr(&f64arr(vec![Some(2.5)]), &DataType::Float32, false).unwrap();
        assert_eq!(ok.len(), 1);
    }

    /// `DECIMAL → <integer>` rounds half-**away** from zero (DuckDB), not toward zero
    /// (arrow's truncating kernel): `2.5 → 3`, `-2.5 → -3`, `0.5 → 1`, `-0.5 → -1`,
    /// while `2.4 → 2` and `2.6 → 3` are unambiguous. Nulls pass through.
    #[test]
    fn decimal_to_int_rounds_half_away_from_zero() {
        use arrow::array::Decimal128Array;
        let src: ArrayRef = Arc::new(
            Decimal128Array::from(vec![
                Some(25),  // 2.5
                Some(35),  // 3.5
                Some(-25), // -2.5
                Some(5),   // 0.5
                Some(-5),  // -0.5
                Some(24),  // 2.4
                Some(26),  // 2.6
                None,
            ])
            .with_precision_and_scale(10, 1)
            .unwrap(),
        );
        let out = cast_expr(&src, &DataType::Int64, false).unwrap();
        let out = out.as_primitive::<Int64Type>();
        let got: Vec<Option<i64>> = (0..out.len())
            .map(|i| (!out.is_null(i)).then(|| out.value(i)))
            .collect();
        assert_eq!(
            got,
            vec![
                Some(3),
                Some(4),
                Some(-3),
                Some(1),
                Some(-1),
                Some(2),
                Some(3),
                None
            ]
        );
    }

    /// DuckDB trims surrounding whitespace before parsing a string into a numeric value:
    /// `'  12  '::BIGINT` = 12, `' +7'::BIGINT` = 7, `' 3.14 '::DOUBLE` = 3.14. Arrow's
    /// kernel does not, so before this fix strict `CAST` errored and `TRY_CAST` NULLed the
    /// padded value (data loss). All six C-`isspace` chars are trimmed; a whitespace-only or
    /// empty string still fails (→ NULL under try_cast); an internal space still fails.
    #[test]
    fn string_to_number_trims_surrounding_whitespace() {
        use arrow::array::StringArray;
        use arrow::datatypes::DataType;
        let src: ArrayRef = Arc::new(StringArray::from(vec![
            Some("  12  "),
            Some("\t-7\n"),
            Some(" +5"),
            Some("\x0b\x0c\r 9 \r\x0c\x0b"),
            Some("   "), // whitespace only → empty → NULL
            Some(""),
            Some("1 2"), // internal space → NULL (not trimmed)
            None,
        ]));
        // try_cast → NULL where the trimmed value still won't parse.
        let out = cast_expr(&src, &DataType::Int64, true).unwrap();
        let out = out.as_primitive::<Int64Type>();
        let got: Vec<Option<i64>> = (0..out.len())
            .map(|i| (!out.is_null(i)).then(|| out.value(i)))
            .collect();
        assert_eq!(
            got,
            vec![Some(12), Some(-7), Some(5), Some(9), None, None, None, None]
        );
        // Float target trims too.
        let fsrc: ArrayRef = Arc::new(StringArray::from(vec![Some(" 2.75 "), Some("  -0.5")]));
        let fout = cast_expr(&fsrc, &DataType::Float64, false).unwrap();
        let fout = fout.as_primitive::<Float64Type>();
        assert_eq!(fout.value(0), 2.75);
        assert_eq!(fout.value(1), -0.5);
        // A clean padded value parses under strict cast (no error).
        let ok = cast_expr(
            &(Arc::new(StringArray::from(vec![Some(" 42 ")])) as ArrayRef),
            &DataType::Int64,
            false,
        )
        .unwrap();
        assert_eq!(ok.as_primitive::<Int64Type>().value(0), 42);
    }

    /// String→Boolean matches DuckDB's exact set. Arrow's kernel trims whitespace, accepts
    /// `on`/`off`, and matches prefixes (`'tru'`→true) — all silent wrong non-null values.
    /// DuckDB accepts (case-insensitive, no trim) only `{true,t,1,yes,y}` / `{false,f,0,no,n}`;
    /// everything else is NULL (try_cast).
    #[test]
    fn string_to_bool_matches_duckdb_exact_set() {
        use arrow::array::StringArray;
        use arrow::datatypes::DataType;
        let src: ArrayRef = Arc::new(StringArray::from(vec![
            Some("true"),
            Some("TRUE"),
            Some("t"),
            Some("1"),
            Some("yes"),
            Some("Y"),
            Some("false"),
            Some("F"),
            Some("0"),
            Some("no"),
            Some("n"),
            Some(" true "), // padded → NULL (DuckDB does not trim)
            Some("on"),     // → NULL (not in DuckDB's set)
            Some("off"),    // → NULL
            Some("tru"),    // prefix → NULL (arrow would say true)
            Some(""),
            None,
        ]));
        let out = cast_expr(&src, &DataType::Boolean, true).unwrap();
        let out = out.as_boolean();
        let got: Vec<Option<bool>> = (0..out.len())
            .map(|i| (!out.is_null(i)).then(|| out.value(i)))
            .collect();
        assert_eq!(
            got,
            vec![
                Some(true),
                Some(true),
                Some(true),
                Some(true),
                Some(true),
                Some(true),
                Some(false),
                Some(false),
                Some(false),
                Some(false),
                Some(false),
                None,
                None,
                None,
                None,
                None,
                None,
            ]
        );
        // Strict cast errors on an unrecognized token instead of fabricating a bool.
        let bad: ArrayRef = Arc::new(StringArray::from(vec![Some("on")]));
        assert!(cast_expr(&bad, &DataType::Boolean, false).is_err());
        // A clean token still parses under strict cast.
        let ok: ArrayRef = Arc::new(StringArray::from(vec![Some("t")]));
        let ok = cast_expr(&ok, &DataType::Boolean, false).unwrap();
        assert!(ok.as_boolean().value(0));
    }
}

#[cfg(test)]
mod string_to_int_tests {
    use super::*;
    use arrow::array::StringArray;
    // See `narrowing_float_tests` — assertion-only type tags.
    use arrow::datatypes::{DataType, Int64Type, Int8Type};

    fn strs(v: Vec<Option<&str>>) -> ArrayRef {
        Arc::new(StringArray::from(v))
    }
    fn as_i64_opt(a: &ArrayRef) -> Vec<Option<i64>> {
        let a = a.as_primitive::<Int64Type>();
        (0..a.len())
            .map(|i| (!a.is_null(i)).then(|| a.value(i)))
            .collect()
    }

    /// DuckDB's `VARCHAR → <integer>` parses a fractional or scientific string and rounds it
    /// half-away from zero (`'1.5'→2`, `'2.5'→3`, `'-2.5'→-3`, `'0.5'→1`, `'-0.5'→-1`,
    /// `'2.4'→2`, `'1e3'→1000`, `'12345.678'→12346`), the same rule as `DECIMAL → <integer>`.
    /// Arrow's integer parser rejects any non-integer string, so before the fix strict `CAST`
    /// errored and `TRY_CAST` NULLed all of these. Nulls pass through.
    #[test]
    fn string_to_int_parses_fractional_and_scientific_half_away() {
        let src = strs(vec![
            Some("1.5"),
            Some("2.5"),
            Some("-2.5"),
            Some("0.5"),
            Some("-0.5"),
            Some("2.4"),
            Some("1e3"),
            Some("12345.678"),
            Some("1.5e0"),
            None,
        ]);
        // try_cast and strict cast agree on the value where the string is parseable.
        for try_cast in [true, false] {
            let out = cast_expr(&src, &DataType::Int64, try_cast).unwrap();
            assert_eq!(
                as_i64_opt(&out),
                vec![
                    Some(2),
                    Some(3),
                    Some(-3),
                    Some(1),
                    Some(-1),
                    Some(2),
                    Some(1000),
                    Some(12346),
                    Some(2),
                    None,
                ],
                "try_cast={try_cast}"
            );
        }
    }

    /// A clean integer string wider than 2^53 stays *exact* — the exact integer parser is
    /// preferred over the f64 fallback, which would lose precision. (`9007199254740993` is the
    /// first integer f64 cannot represent.)
    #[test]
    fn string_to_int_keeps_large_integers_exact() {
        let big = (1i64 << 53) + 1; // 9_007_199_254_740_993
        let src = strs(vec![Some("9007199254740993"), Some("9223372036854775807")]);
        let out = cast_expr(&src, &DataType::Int64, false).unwrap();
        assert_eq!(as_i64_opt(&out), vec![Some(big), Some(i64::MAX)]);
    }

    /// An unparseable, empty, non-finite, or out-of-range string is NULL under `TRY_CAST` and
    /// an error under strict `CAST` — matching DuckDB (`'inf'::BIGINT` and `'abc'::BIGINT`
    /// both error; `'1e19'::BIGINT` overflows). A clean value in the same batch still parses.
    #[test]
    fn string_to_int_rejects_unparseable_and_overflow() {
        // try_cast: bad rows NULL, the good row parses.
        let src = strs(vec![
            Some("abc"),
            Some(""),
            Some("inf"),
            Some("nan"),
            Some("1e19"), // overflows i64
            Some("42"),
        ]);
        let out = cast_expr(&src, &DataType::Int64, true).unwrap();
        assert_eq!(
            as_i64_opt(&out),
            vec![None, None, None, None, None, Some(42)]
        );
        // strict cast errors on any unparseable/out-of-range value.
        for bad in ["abc", "", "inf", "1e19"] {
            assert!(
                cast_expr(&strs(vec![Some(bad)]), &DataType::Int64, false).is_err(),
                "strict CAST('{bad}' AS BIGINT) must error"
            );
        }
        // A clean fractional string still parses under strict cast (no error).
        let ok = cast_expr(&strs(vec![Some("7.5")]), &DataType::Int64, false).unwrap();
        assert_eq!(as_i64_opt(&ok), vec![Some(8)]);
    }

    /// A narrower/unsigned target keeps DuckDB's range checking: `'300'` overflows `Int8`
    /// (NULL/try, error/strict) while an in-range fractional value rounds and fits.
    #[test]
    fn string_to_narrow_int_range_checks() {
        let src = strs(vec![Some("300"), Some("12.5"), Some("-5.5")]);
        let out = cast_expr(&src, &DataType::Int8, true).unwrap();
        let a = out.as_primitive::<Int8Type>();
        let got: Vec<Option<i8>> = (0..a.len())
            .map(|i| (!a.is_null(i)).then(|| a.value(i)))
            .collect();
        assert_eq!(got, vec![None, Some(13), Some(-6)]);
    }
}

#[cfg(test)]
mod float_to_string_tests {
    use super::*;
    use arrow::array::Float64Array;
    use arrow::datatypes::DataType;

    /// DuckDB renders a float NaN as `nan`, where arrow's formatter emits `NaN`; that one
    /// case is normalized. Every ordinary value keeps arrow's (DuckDB-matching)
    /// shortest-round-trip string, and nulls pass through.
    ///
    /// A negative zero keeps its sign. This test previously pinned `-0.0` → `"0.0"`, on the
    /// stated grounds that DuckDB does that — but DuckDB only renders `0.0` for a
    /// *constant-folded literal*; given a real `-0.0` in a column it emits `-0.0`, and so do
    /// Polars, Arrow and Python. The `-0.0`/`0.0` fold is a *key identity* rule (which rows
    /// are equal), not a display rule.
    #[test]
    fn float_to_string_normalizes_nan_and_keeps_signed_zero() {
        let src: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(f64::NAN),
            Some(-0.0),
            Some(0.0),
            Some(0.1),
            Some(-2.5),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
            Some(100000000.0),
            None,
        ]));
        let out = cast_expr(&src, &DataType::Utf8, false).unwrap();
        let out = out.as_string::<i32>();
        let got: Vec<Option<&str>> = (0..out.len())
            .map(|i| (!out.is_null(i)).then(|| out.value(i)))
            .collect();
        assert_eq!(
            got,
            vec![
                Some("nan"),
                Some("-0.0"), // the sign survives; only NaN's spelling is normalized
                Some("0.0"),
                Some("0.1"),
                Some("-2.5"),
                Some("inf"),
                Some("-inf"),
                Some("100000000.0"),
                None,
            ]
        );
    }
}
