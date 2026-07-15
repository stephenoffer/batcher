//! Numeric evaluation for `Expr::Math`/`Math2`/`Coalesce`/`Greatest`/`Least`
//! (split out of `lib.rs`).

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch};
use arrow::compute::kernels::arity::try_binary;
use arrow::compute::kernels::cmp;
use arrow::compute::kernels::zip::zip;
use arrow::compute::{cast, is_not_null};
use arrow::datatypes::DataType;
use arrow::error::ArrowError;

use crate::eval::binary::coerce_numeric;
use crate::{Expr, ExprError, Math2Func, MathFunc};

/// `is_nan(x)`: true where a value is IEEE NaN (a float-only notion, distinct from
/// null). Nulls propagate (null → null). NaN is impossible for non-float numerics,
/// so casting them to Float64 yields all-false, which is correct. This is a
/// first-class op rather than the `x != x` lowering, because the engine's `!=`
/// operator uses *total ordering* (where `NaN == NaN`), so `x != x` would never
/// flag a NaN. The Tier-1 JIT does not compile `IsNan` and falls back here.
pub(crate) fn eval_is_nan(array: &ArrayRef) -> Result<ArrayRef, ExprError> {
    let f = cast(array, &DataType::Float64)?;
    let a = f
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("cast to f64");
    let out: BooleanArray = if a.null_count() == 0 {
        a.values().iter().map(|&x| Some(x.is_nan())).collect()
    } else {
        (0..a.len())
            .map(|i| (!a.is_null(i)).then(|| a.value(i).is_nan()))
            .collect()
    };
    Ok(Arc::new(out))
}

/// `is_inf` — true where a float value is `+inf`/`-inf`; null → null. Mirrors
/// `eval_is_nan` (cast to f64, per-element predicate, validity preserved).
pub(crate) fn eval_is_inf(array: &ArrayRef) -> Result<ArrayRef, ExprError> {
    let f = cast(array, &DataType::Float64)?;
    let a = f
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("cast to f64");
    let out: BooleanArray = if a.null_count() == 0 {
        a.values().iter().map(|&x| Some(x.is_infinite())).collect()
    } else {
        (0..a.len())
            .map(|i| (!a.is_null(i)).then(|| a.value(i).is_infinite()))
            .collect()
    };
    Ok(Arc::new(out))
}

/// Two-argument math: align both sides to Float64, apply element-wise (nulls
/// propagate). `round`'s second argument is the (per-row) decimal-place count.
pub(crate) fn eval_math2(
    func: Math2Func,
    l: &ArrayRef,
    r: &ArrayRef,
) -> Result<ArrayRef, ExprError> {
    // `gcd`/`lcm` are integer functions (DuckDB returns an integer, not a double).
    // Routing an i64 through f64 (a) mistypes the public schema as `double` and
    // (b) silently loses precision above 2^53 (`gcd(2^53+1, 3)` returned 1.0 instead
    // of 3). Compute them on the true i64 bits and return Int64.
    if matches!(func, Math2Func::Gcd | Math2Func::Lcm) {
        return eval_int_math2(func, l, r);
    }
    let lf = cast(l, &DataType::Float64)?;
    let rf = cast(r, &DataType::Float64)?;
    let a = lf.as_any().downcast_ref::<Float64Array>().expect("f64");
    let b = rf.as_any().downcast_ref::<Float64Array>().expect("f64");
    let out: Float64Array = if a.null_count() == 0 && b.null_count() == 0 {
        // No-null fast path: walk both raw slices, no per-element validity branch.
        a.values()
            .iter()
            .zip(b.values())
            .map(|(&x, &y)| apply_binary(func, x, y))
            .collect()
    } else {
        (0..a.len())
            .map(|i| {
                (!a.is_null(i) && !b.is_null(i)).then(|| apply_binary(func, a.value(i), b.value(i)))
            })
            .collect()
    };
    Ok(Arc::new(out))
}

/// One two-argument math op on scalar `f64`s (shared by both null paths).
#[inline]
fn apply_binary(func: Math2Func, x: f64, y: f64) -> f64 {
    match func {
        Math2Func::Pow => x.powf(y),
        Math2Func::Atan2 => x.atan2(y),
        Math2Func::Round => {
            let f = 10f64.powi(y as i32);
            (x * f).round() / f
        }
        Math2Func::Hypot => x.hypot(y),
        // Gcd/Lcm are handled by the integer path (`eval_int_math2`).
        Math2Func::Gcd | Math2Func::Lcm => unreachable!("integer path"),
    }
}

/// Integer two-argument math (`gcd`/`lcm`): cast both sides to Int64 and apply on the
/// true i64 values, returning Int64. `lcm` errors on i64 overflow (matching DuckDB's
/// "lcm value is out of range") rather than wrapping or losing precision through f64.
fn eval_int_math2(func: Math2Func, l: &ArrayRef, r: &ArrayRef) -> Result<ArrayRef, ExprError> {
    let li = cast(l, &DataType::Int64)?;
    let ri = cast(r, &DataType::Int64)?;
    let a = li.as_any().downcast_ref::<Int64Array>().expect("i64");
    let b = ri.as_any().downcast_ref::<Int64Array>().expect("i64");
    let out: Int64Array = match func {
        Math2Func::Gcd => {
            if a.null_count() == 0 && b.null_count() == 0 {
                a.values()
                    .iter()
                    .zip(b.values())
                    .map(|(&x, &y)| Some(gcd_i64(x, y)))
                    .collect()
            } else {
                (0..a.len())
                    .map(|i| {
                        (!a.is_null(i) && !b.is_null(i)).then(|| gcd_i64(a.value(i), b.value(i)))
                    })
                    .collect()
            }
        }
        Math2Func::Lcm => {
            // `try_binary` propagates nulls and short-circuits on the first overflow.
            let out: Int64Array = try_binary(a, b, lcm_i64)?;
            return Ok(Arc::new(out));
        }
        _ => unreachable!("only gcd/lcm reach here"),
    };
    Ok(Arc::new(out))
}

/// Least common multiple of two i64s, erroring on overflow (never wrapping). `lcm(0, n)`
/// is 0. Computes `|a / gcd * b|` and checks the multiply, so a coprime pair whose product
/// exceeds i64 raises rather than returning a wrapped (wrong) value.
fn lcm_i64(a: i64, b: i64) -> Result<i64, ArrowError> {
    let g = gcd_i64(a, b);
    if g == 0 {
        return Ok(0);
    }
    // `a / g` is exact (g divides a); the remaining multiply is the only overflow risk.
    (a / g)
        .checked_mul(b)
        .and_then(i64::checked_abs)
        .ok_or_else(|| ArrowError::ComputeError("lcm value is out of range".into()))
}

/// `GREATEST`/`LEAST`: fold the inputs element-wise, ignoring nulls (a null
/// argument never wins; the result is null only where every argument is null).
pub(crate) fn eval_extreme(
    inputs: &[Expr],
    batch: &RecordBatch,
    greatest: bool,
) -> Result<ArrayRef, ExprError> {
    if inputs.is_empty() {
        return Err(ExprError::MissingArgument {
            func: if greatest { "greatest" } else { "least" }.into(),
            arg: "inputs",
        });
    }
    let mut acc = inputs[0].eval(batch)?;
    for next in &inputs[1..] {
        let b = next.eval(batch)?;
        // Promote mixed numeric operands to a common type (e.g. greatest(int, float) →
        // float) before comparing, so a valid int×float call returns a value instead of
        // erroring `Int64 >= Float64`. Matches DuckDB and the sibling `coalesce`.
        let (acc_c, b) = coerce_numeric(&acc, &b)?;
        let cmp = if greatest {
            cmp::gt_eq(&acc_c, &b)?
        } else {
            cmp::lt_eq(&acc_c, &b)?
        };
        // Where both are non-null, pick the winner; null elsewhere. Then coalesce
        // with each side so a lone non-null still survives.
        let both = zip(&cmp, &acc_c.as_ref(), &b.as_ref())?;
        acc = coalesce_arrays(&[both, acc_c, b])?;
    }
    Ok(acc)
}

/// First non-null per row across the given arrays (array-level COALESCE).
fn coalesce_arrays(arrs: &[ArrayRef]) -> Result<ArrayRef, ExprError> {
    let mut acc = arrs[arrs.len() - 1].clone();
    for a in arrs[..arrs.len() - 1].iter().rev() {
        let mask = is_not_null(a)?;
        acc = zip(&mask, &a.as_ref(), &acc.as_ref())?;
    }
    Ok(acc)
}

/// COALESCE: first non-null among the inputs, per row. Folds from the last input
/// upward so earlier inputs win.
pub(crate) fn eval_coalesce(inputs: &[Expr], batch: &RecordBatch) -> Result<ArrayRef, ExprError> {
    if inputs.is_empty() {
        return Err(ExprError::MissingArgument {
            func: "coalesce".to_string(),
            arg: "inputs",
        });
    }
    let mut acc = inputs[inputs.len() - 1].eval(batch)?;
    for expr in inputs[..inputs.len() - 1].iter().rev() {
        let v = expr.eval(batch)?;
        let present = is_not_null(&v)?;
        // Promote mixed numeric inputs to a common type (e.g. coalesce(int,float)
        // → float) so `zip` sees matching types, matching SQL coercion.
        let (v, acc_c) = coerce_numeric(&v, &acc)?;
        acc = zip(&present, &v.as_ref(), &acc_c.as_ref())?;
    }
    Ok(acc)
}

/// Unary math. `abs` keeps the input numeric type; `round`/`floor`/`ceil`/`sqrt`
/// yield Float64 (integer inputs are promoted).
pub(crate) fn eval_math(func: MathFunc, arr: &ArrayRef) -> Result<ArrayRef, ExprError> {
    use MathFunc::*;
    // `bit_count`/`factorial` are integer functions: their result is defined by the
    // two's-complement i64 bits, not an f64 approximation. Routing them through f64
    // (a) mistyped the schema as `double` and (b) gave wrong answers above 2^53 —
    // `bit_count(2^53+1)` returned 1.0 instead of 2. Compute on the true i64 value.
    if matches!(func, BitCount | Factorial) {
        return eval_int_math(func, arr);
    }
    match (func, arr.data_type()) {
        (Abs, DataType::Int64) => {
            let a = arr.as_any().downcast_ref::<Int64Array>().unwrap();
            // `i64::MIN.abs()` overflows (no positive i64 exists for it): `v.abs()` panicked
            // in debug and returned i64::MIN — a *negative* "absolute value" — in release.
            // `saturating_abs` maps i64::MIN → i64::MAX: no panic, always non-negative, and
            // the JIT emits the same saturation so the two tiers stay bit-for-bit identical.
            let out: Int64Array = if a.null_count() == 0 {
                a.values().iter().map(|&v| v.saturating_abs()).collect()
            } else {
                a.iter().map(|o| o.map(|v| v.saturating_abs())).collect()
            };
            Ok(Arc::new(out))
        }
        (_, DataType::Int64) => {
            // Promote integers to Float64 and apply the float function.
            let f = cast(arr, &DataType::Float64)?;
            eval_math(func, &f)
        }
        (_, DataType::Float64) => {
            let a = arr.as_any().downcast_ref::<Float64Array>().unwrap();
            // No-null fast path: map the raw slice (no per-element validity branch,
            // so the simple ops auto-vectorize); otherwise propagate nulls.
            let out: Float64Array = if a.null_count() == 0 {
                a.values().iter().map(|&v| apply_unary(func, v)).collect()
            } else {
                a.iter().map(|o| o.map(|v| apply_unary(func, v))).collect()
            };
            Ok(Arc::new(out))
        }
        (_, other) => Err(ExprError::ExpectedString {
            func: format!("{func:?}"),
            got: other.to_string(),
        }),
    }
}

/// One unary math op on a scalar `f64` (shared by both null paths of `eval_math`).
#[inline]
fn apply_unary(func: MathFunc, v: f64) -> f64 {
    use MathFunc::*;
    match func {
        Abs => v.abs(),
        Round => v.round(),
        Floor => v.floor(),
        Ceil => v.ceil(),
        Sqrt => v.sqrt(),
        Ln => v.ln(),
        Log10 => v.log10(),
        Log2 => v.log2(),
        Exp => v.exp(),
        Sin => v.sin(),
        Cos => v.cos(),
        Tan => v.tan(),
        Sign => {
            if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        Trunc => v.trunc(),
        Cbrt => v.cbrt(),
        Asin => v.asin(),
        Acos => v.acos(),
        Atan => v.atan(),
        Sinh => v.sinh(),
        Cosh => v.cosh(),
        Tanh => v.tanh(),
        Degrees => v.to_degrees(),
        Radians => v.to_radians(),
        Cot => 1.0 / v.tan(),
        // Integer-only functions are handled by `eval_int_math`.
        Factorial | BitCount => unreachable!("integer path"),
    }
}

/// Integer unary math (`bit_count`/`factorial`): cast the input to Int64 and compute on
/// the true i64 value, returning Int64.
///
/// `bit_count` is the population count of the two's-complement bits. `factorial` is
/// computed with a *checked* product that errors on i64 overflow (`21!` and above): this
/// both fixes a hang — the previous f64 path looped `1..=n` for a huge `n` — and refuses
/// to silently return a wrong (wrapped or rounded-double) value. `factorial(n)` for
/// `n > 20` therefore raises rather than overflowing i64. (DuckDB widens to HUGEINT and so
/// spans `n ≤ 33`; the engine has no 128-bit output type, so the exact-or-error contract
/// stops one step earlier — a documented, safe difference.)
fn eval_int_math(func: MathFunc, arr: &ArrayRef) -> Result<ArrayRef, ExprError> {
    use MathFunc::*;
    let i = cast(arr, &DataType::Int64)?;
    let a = i.as_any().downcast_ref::<Int64Array>().expect("i64");
    let out: Int64Array = match func {
        BitCount => {
            if a.null_count() == 0 {
                a.values()
                    .iter()
                    .map(|&v| Some(i64::from(v.count_ones())))
                    .collect()
            } else {
                a.iter()
                    .map(|o| o.map(|v| i64::from(v.count_ones())))
                    .collect()
            }
        }
        Factorial => a
            .iter()
            .map(|o| o.map(factorial_i64).transpose())
            .collect::<Result<Int64Array, ExprError>>()?,
        _ => unreachable!("only bit_count/factorial reach here"),
    };
    Ok(Arc::new(out))
}

/// `n!` in i64. `0! = 1`; a negative `n` errors (matching DuckDB, which raises "factorial
/// of a negative number"); overflow past `20!` raises instead of wrapping or hanging.
fn factorial_i64(n: i64) -> Result<i64, ExprError> {
    if n < 0 {
        return Err(ExprError::Arrow(ArrowError::ComputeError(
            "factorial of a negative number".into(),
        )));
    }
    let mut acc: i64 = 1;
    let mut k: i64 = 2;
    while k <= n {
        acc = acc.checked_mul(k).ok_or_else(|| {
            ExprError::Arrow(ArrowError::ComputeError(format!(
                "factorial({n}) is out of range for a 64-bit integer"
            )))
        })?;
        k += 1;
    }
    Ok(acc)
}

/// Greatest common divisor of two integers (Euclid; non-negative result).
fn gcd_i64(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.unsigned_abs(), b.unsigned_abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a as i64
}

#[cfg(test)]
mod int_math_tests {
    use super::*;

    fn i64arr(v: Vec<Option<i64>>) -> ArrayRef {
        Arc::new(Int64Array::from(v))
    }

    fn as_i64(a: &ArrayRef) -> Vec<Option<i64>> {
        let a = a.as_any().downcast_ref::<Int64Array>().expect("i64 out");
        (0..a.len())
            .map(|i| (!a.is_null(i)).then(|| a.value(i)))
            .collect()
    }

    /// `gcd`/`bit_count` stay Int64 and are exact above 2^53 — the old f64 route both
    /// mistyped the schema and gave wrong answers (`gcd(2^53+1, 3)` → 1.0 not 3).
    #[test]
    fn gcd_bit_count_are_exact_int64_above_2_pow_53() {
        let two53p1 = (1i64 << 53) + 1; // 9_007_199_254_740_993, not f64-representable
        let g = eval_math2(
            Math2Func::Gcd,
            &i64arr(vec![Some(two53p1)]),
            &i64arr(vec![Some(3)]),
        )
        .unwrap();
        assert_eq!(g.data_type(), &DataType::Int64);
        assert_eq!(as_i64(&g), vec![Some(3)]);

        let bc = eval_math(MathFunc::BitCount, &i64arr(vec![Some(two53p1)])).unwrap();
        assert_eq!(bc.data_type(), &DataType::Int64);
        assert_eq!(as_i64(&bc), vec![Some(2)]); // bit 53 and bit 0 set
    }

    /// `lcm` is Int64 and errors on i64 overflow rather than wrapping or losing precision.
    #[test]
    fn lcm_int64_and_overflow_errors() {
        let ok = eval_math2(
            Math2Func::Lcm,
            &i64arr(vec![Some(4)]),
            &i64arr(vec![Some(6)]),
        )
        .unwrap();
        assert_eq!(ok.data_type(), &DataType::Int64);
        assert_eq!(as_i64(&ok), vec![Some(12)]);
        assert_eq!(
            as_i64(
                &eval_math2(
                    Math2Func::Lcm,
                    &i64arr(vec![Some(0)]),
                    &i64arr(vec![Some(5)])
                )
                .unwrap()
            ),
            vec![Some(0)]
        );
        // Two large coprimes whose lcm exceeds i64 → error, never a wrapped value.
        let big = eval_math2(
            Math2Func::Lcm,
            &i64arr(vec![Some(3_037_000_500)]),
            &i64arr(vec![Some(3_037_000_507)]),
        );
        assert!(big.is_err(), "lcm overflow must error");
    }

    /// `factorial` is Int64, terminates on a huge input (no hang), and errors on overflow
    /// and on negatives instead of looping or returning a wrong double.
    #[test]
    fn factorial_int64_terminates_and_errors() {
        let f = eval_math(
            MathFunc::Factorial,
            &i64arr(vec![Some(0), Some(5), Some(20)]),
        )
        .unwrap();
        assert_eq!(f.data_type(), &DataType::Int64);
        assert_eq!(
            as_i64(&f),
            vec![Some(1), Some(120), Some(2_432_902_008_176_640_000)]
        );
        // 21! overflows i64 → error (not a wrapped/rounded value).
        assert!(eval_math(MathFunc::Factorial, &i64arr(vec![Some(21)])).is_err());
        // A huge input must not hang; it errors quickly at the first overflow.
        assert!(eval_math(MathFunc::Factorial, &i64arr(vec![Some(i64::MAX)])).is_err());
        // Negative → error (matches DuckDB).
        assert!(eval_math(MathFunc::Factorial, &i64arr(vec![Some(-1)])).is_err());
    }

    /// Nulls propagate through the integer-math paths.
    #[test]
    fn int_math_nulls_propagate() {
        assert_eq!(
            as_i64(&eval_math(MathFunc::BitCount, &i64arr(vec![Some(7), None])).unwrap()),
            vec![Some(3), None]
        );
        assert_eq!(
            as_i64(
                &eval_math2(Math2Func::Gcd, &i64arr(vec![None]), &i64arr(vec![Some(3)])).unwrap()
            ),
            vec![None]
        );
    }
}
