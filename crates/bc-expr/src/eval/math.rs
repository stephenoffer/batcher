//! Numeric evaluation for `Expr::Math`/`Math2`/`Coalesce`/`Greatest`/`Least`
//! (split out of `lib.rs`).

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, BooleanArray, Float64Array, Int64Array, RecordBatch};
use arrow::buffer::BooleanBuffer;
use arrow::compute::kernels::arity::{binary, try_binary, unary};
use arrow::compute::kernels::cmp;
use arrow::compute::kernels::zip::zip;
use arrow::compute::{cast, is_not_null};
use arrow::datatypes::{DataType, Float64Type, Int64Type};
use arrow::error::ArrowError;

use crate::eval::coerce::{align_decimals_for_cmp, coerce_numeric};
use crate::{Expr, ExprError, Math2Func, MathFunc};

/// `is_nan(x)`: true where a value is IEEE NaN (a float-only notion, distinct from
/// null). Nulls propagate (null → null). NaN is impossible for non-float numerics,
/// so casting them to Float64 yields all-false, which is correct. This is a
/// first-class op rather than the `x != x` lowering, because the engine's `!=`
/// operator uses *total ordering* (where `NaN == NaN`), so `x != x` would never
/// flag a NaN. The Tier-1 JIT does not compile `IsNan` and falls back here.
pub(crate) fn eval_is_nan(array: &ArrayRef) -> Result<ArrayRef, ExprError> {
    let f = cast(array, &DataType::Float64)?;
    let a = f.as_primitive::<Float64Type>();
    // One pass over the values buffer, carrying the input's null buffer through unchanged.
    // A null slot's payload is arbitrary but its answer is masked by that same buffer, so the
    // loop stays branchless on validity — the trade `join::key_filter::mask` already makes.
    let values = BooleanBuffer::collect_bool(a.len(), |i| a.value(i).is_nan());
    Ok(Arc::new(BooleanArray::new(values, a.nulls().cloned())))
}

/// `is_inf` — true where a float value is `+inf`/`-inf`; null → null. Mirrors
/// `eval_is_nan` (cast to f64, per-element predicate, validity preserved).
pub(crate) fn eval_is_inf(array: &ArrayRef) -> Result<ArrayRef, ExprError> {
    let f = cast(array, &DataType::Float64)?;
    let a = f.as_primitive::<Float64Type>();
    // Same shape as `eval_is_nan`: values in one pass, validity carried through.
    let values = BooleanBuffer::collect_bool(a.len(), |i| a.value(i).is_infinite());
    Ok(Arc::new(BooleanArray::new(values, a.nulls().cloned())))
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
    // `round` on an integer is an integer, for the same reason: DuckDB returns BIGINT
    // for `round(bigint, n)`, and the f64 round-trip corrupts values above 2^53
    // (`round(2^53+1, 0)` came back as `2^53`). `floor`/`ceil` genuinely *do* yield
    // double in DuckDB, so only `round` needs this — the blanket promotion is right
    // for its neighbours and wrong here.
    // Every integer width, not just `Int64`: a mid-plan `CAST(x AS SMALLINT)` produced an
    // `Int16` that fell through to the float promotion below, so `round(CAST(i AS SMALLINT), 0)`
    // came back DOUBLE while `round(i, 0)` came back BIGINT — the same expression typed two
    // ways depending on an upstream cast.
    if matches!(func, Math2Func::Round) && l.data_type().is_integer() {
        return round_int(l, r);
    }
    let lf = cast(l, &DataType::Float64)?;
    let rf = cast(r, &DataType::Float64)?;
    let a = lf.as_primitive::<Float64Type>();
    let b = rf.as_primitive::<Float64Type>();
    // `arity::binary` walks both values buffers and unions the two null buffers once, which is
    // exactly the `!a.is_null(i) && !b.is_null(i)` this replaces — but as a buffer operation
    // rather than a per-element test, so it vectorizes with or without nulls. Same reasoning as
    // the unary path in `eval_math`, including that slots under a null are computed and masked.
    //
    // Measured, `pow` over 20M Float64 with 30 % nulls: **24.9 ms -> 20.7 ms**.
    let out: Float64Array = binary(a, b, |x, y| apply_binary(func, x, y))?;
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
        // One ULP toward `y`. `f64::next_after` is unstable in std; stepping the
        // two's-complement-ordered bit pattern is the standard equivalent, and it is
        // exact rather than an approximation of "a bit more".
        Math2Func::NextAfter => next_after(x, y),
        // Gcd/Lcm are handled by the integer path (`eval_int_math2`).
        Math2Func::Gcd | Math2Func::Lcm => unreachable!("integer path"),
    }
}

/// The next representable `f64` after `from` in the direction of `to`.
///
/// IEEE-754 doubles of one sign are ordered by their bit pattern, so "one ULP" is
/// literally `+1`/`-1` on those bits; the sign bit is what breaks the ordering, which is
/// why zero and a sign change are handled before the increment.
#[inline]
fn next_after(from: f64, to: f64) -> f64 {
    if from.is_nan() || to.is_nan() {
        return f64::NAN;
    }
    if from == to {
        return to; // already there (this also settles +0.0 vs -0.0)
    }
    if from == 0.0 {
        // Both zeros step to the smallest subnormal on `to`'s side.
        return f64::from_bits(1) * to.signum();
    }
    let bits = from.to_bits();
    // Stepping "up" in magnitude means away from zero, which is +1 on the bits for a
    // positive value and -1 for a negative one.
    let toward_larger = (to > from) == (from > 0.0);
    f64::from_bits(if toward_larger { bits + 1 } else { bits - 1 })
}

/// `round(x, digits)` over an Int64 column, computed on the true i64 value.
///
/// A non-negative `digits` is the identity on an integer. A negative `digits` rounds to
/// that power of ten, half away from zero (DuckDB's direction), in `i128` so the scaled
/// intermediate cannot overflow on the way.
fn round_int(l: &ArrayRef, r: &ArrayRef) -> Result<ArrayRef, ExprError> {
    let ri = cast(r, &DataType::Int64)?;
    // `l` is cast rather than downcast: the caller admits every integer width, and
    // `as_primitive` on a narrower one panics rather than returning an error.
    let li = cast(l, &DataType::Int64)?;
    let a = li.as_primitive::<Int64Type>();
    let b = ri.as_primitive::<Int64Type>();
    // `binary` unions the two null buffers once instead of testing validity per row.
    let out: Int64Array = binary(a, b, round_i64)?;
    Ok(Arc::new(out))
}

/// One `round(x, digits)` on i64, half away from zero.
#[inline]
fn round_i64(x: i64, digits: i64) -> i64 {
    if digits >= 0 {
        return x; // an integer is already rounded to any non-negative place
    }
    // 10^19 already exceeds i64::MAX, so every wider step rounds to 0; clamp the
    // exponent rather than overflow the i128 power itself.
    let f = 10i128.pow((-digits).min(19) as u32);
    let v = x as i128;
    let half = f / 2;
    // Rust's `/` truncates toward zero, so biasing by ±half gives half-away-from-zero.
    let q = if v >= 0 {
        (v + half) / f
    } else {
        (v - half) / f
    };
    (q * f) as i64
}

/// Integer two-argument math (`gcd`/`lcm`): cast both sides to Int64 and apply on the
/// true i64 values, returning Int64. `lcm` errors on i64 overflow (matching DuckDB's
/// "lcm value is out of range") rather than wrapping or losing precision through f64.
fn eval_int_math2(func: Math2Func, l: &ArrayRef, r: &ArrayRef) -> Result<ArrayRef, ExprError> {
    let li = cast(l, &DataType::Int64)?;
    let ri = cast(r, &DataType::Int64)?;
    let a = li.as_primitive::<Int64Type>();
    let b = ri.as_primitive::<Int64Type>();
    let out: Int64Array = match func {
        // `gcd` cannot fail, so it takes the infallible `binary` beside `lcm`'s `try_binary`.
        Math2Func::Gcd => binary(a, b, gcd_i64)?,
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
        // Then align two decimals, which `coerce_numeric` deliberately leaves alone: it
        // runs ahead of *arithmetic* too, where pre-widening the operands compounds with
        // the operator's own scale rule and moves the result type. `greatest`/`least` are
        // the comparison case, and the comparison kernels below demand identical decimal
        // precision AND scale — `greatest(decimal(10,2), decimal(12,4))` otherwise raised
        // `Invalid comparison operation`. `binary.rs` does the same thing for `=`/`<`/...
        let (acc_c, b) = align_decimals_for_cmp(&acc_c, &b)?;
        // Rank on the engine's float identity, not arrow's raw-bit total order: a
        // *negative* NaN must rank greatest (not below -inf) and `-0.0`/`0.0` must
        // compare equal, so `greatest`/`least` agree with `MIN`/`MAX`, `GROUP BY`,
        // and scalar `=`. Compare canonicalized keys but select the *original*
        // values, so a `-0.0`/`-NaN` in is the same value out.
        let acc_k = bc_arrow::canon_float_array(&acc_c);
        let b_k = bc_arrow::canon_float_array(&b);
        let cmp = if greatest {
            cmp::gt_eq(&acc_k, &b_k)?
        } else {
            cmp::lt_eq(&acc_k, &b_k)?
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

/// Unary math. `abs` and `round` keep the input numeric type (both are integer-valued
/// on an integer, and DuckDB returns BIGINT for both); `floor`/`ceil`/`sqrt` yield
/// Float64, promoting integer inputs, as DuckDB does.
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
            let a = arr.as_primitive::<Int64Type>();
            // `i64::MIN.abs()` overflows (no positive i64 exists for it): `v.abs()` panicked
            // in debug and returned i64::MIN — a *negative* "absolute value" — in release.
            // `saturating_abs` maps i64::MIN → i64::MAX: no panic, always non-negative, and
            // the JIT emits the same saturation so the two tiers stay bit-for-bit identical.
            let out: Int64Array = unary(a, |v: i64| v.saturating_abs());
            Ok(Arc::new(out))
        }
        // `round` of an integer is that integer. Promoting to f64 first both mistyped
        // the schema as `double` (DuckDB returns BIGINT) and silently corrupted values
        // above 2^53 — `round(2^53+1)` came back as `2^53`. `floor`/`ceil`/`sqrt` really
        // do yield double in DuckDB, so the promotion below stays right for them.
        (Round, DataType::Int64) => Ok(Arc::clone(arr)),
        (_, DataType::Int64) | (_, DataType::Decimal128(..)) | (_, DataType::Decimal256(..)) => {
            // Promote to Float64 and apply the float function.
            //
            // **Decimal is here deliberately, and it is a trade.** Arithmetic, comparison,
            // aggregation and negation all keep a `Decimal` exact — only this family
            // promotes. Before, every one of them *rejected* a decimal column outright
            // ("Abs expected a numeric argument, got Decimal128(10, 2)"), so a Parquet
            // money column could be summed and compared but not rounded, floored, or
            // passed to `abs`. Refusing was not preserving precision; it was refusing to
            // answer.
            //
            // The result is DOUBLE. For `sqrt`/`ln`/the trig family that is also what
            // DuckDB returns. For `abs`/`floor`/`ceil`/`round`/`sign` DuckDB keeps
            // DECIMAL, so those diverge in *result type* — the divergence the census
            // already pinned for `ceil`/`floor`, now covering the family — and lose
            // exactness above 2^53. A decimal-preserving path for that subset is the
            // follow-on; it needs a scale-aware kernel per op rather than this promotion.
            let f = cast(arr, &DataType::Float64)?;
            eval_math(func, &f)
        }
        (_, DataType::Float64) => {
            let a = arr.as_primitive::<Float64Type>();
            // `arity::unary` maps the raw values buffer and *reuses* the input's null buffer,
            // so the op auto-vectorizes whether or not the column has nulls. The path this
            // replaces branched: a null-free column mapped the slice, but a nullable one went
            // through `iter().map(|o| o.map(..))`, rebuilding the validity bitmap one bit at a
            // time — the per-element `Option` is what stopped it vectorizing.
            //
            // Slots under a null are computed too, and that is safe here: their payload is
            // arbitrary but the result is masked away by the very null buffer being reused, and
            // no op in `apply_unary` traps (`sqrt`/`ln` of a negative are NaN, not a fault).
            //
            // Measured, `sqrt` over 20M Float64: **15.6 ms -> 13.3 ms with 30 % nulls**, and
            // 13.7 ms -> 13.0 ms with none — so the nullable path gains and the dense path does
            // not lose, which is what retires the branch.
            let out: Float64Array = unary(a, |v| apply_unary(func, v));
            Ok(Arc::new(out))
        }
        // An all-null column types as `Null`, which is a real type rather than an error: it
        // is what `SELECT NULL AS x`, an all-`None` column, and a left join that matched
        // nothing all produce. Arithmetic, `cast` and `coalesce` already answer null for it,
        // and the aggregate, string and list families were each taught to; this one still
        // rejected the column and failed the query. Promoting to an all-null `Float64` is the
        // same move the `Int64`/`Decimal` arms above make, and every function's own null
        // handling takes it from there — which is what DuckDB returns for all of them.
        (_, DataType::Null) => {
            let f = cast(arr, &DataType::Float64)?;
            eval_math(func, &f)
        }
        // Every other integer width. The FFI boundary normalizes Int8/16/32 to Int64 on the
        // way *in*, which is why only `Int64` was handled — but a mid-plan `CAST(x AS
        // SMALLINT)` mints an `Int16` the boundary never sees, and the arm below then
        // rejected it. `floor(CAST(i AS SMALLINT))` failed the whole query with "Floor
        // expected a numeric argument, got Int16", and so did every other function in this
        // family after any narrowing cast. Promoting to the engine's canonical integer width
        // and recursing reuses the `Int64` arms above verbatim, `abs`'s saturation included.
        (
            _,
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32,
        ) => {
            let i = cast(arr, &DataType::Int64)?;
            eval_math(func, &i)
        }
        // `UInt64` holds values no `Int64` does, so it cannot take the promotion above.
        // `abs`/`round` of an unsigned integer is that integer, exactly — returning it
        // untouched keeps the values above `i64::MAX` that a float promotion would round.
        (Abs | Round, DataType::UInt64) => Ok(Arc::clone(arr)),
        // The narrow floats, and `UInt64` for the genuinely float-valued functions.
        (_, DataType::Float16 | DataType::Float32 | DataType::UInt64) => {
            let f = cast(arr, &DataType::Float64)?;
            eval_math(func, &f)
        }
        (_, other) => Err(ExprError::ExpectedType {
            func: format!("{func:?}"),
            want: "a numeric argument",
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
        // std's routines are the FDLIBM ones, so they hold at the range ends where the
        // `ln(x ± sqrt(x*x ∓ 1))` identities overflow or cancel.
        Asinh => v.asinh(),
        Acosh => v.acosh(),
        Atanh => v.atanh(),
        Degrees => v.to_degrees(),
        Radians => v.to_radians(),
        Cot => 1.0 / v.tan(),
        Sec => 1.0 / v.cos(),
        Csc => 1.0 / v.sin(),
        // IEEE-754 `roundTiesToEven`, the rounding mode the hardware already uses for
        // arithmetic. `v.round()` is ties-away-from-zero, which is what `round` means
        // here and in DuckDB — the two are deliberately different functions.
        Rint => round_ties_even(v),
        // Round *outward* to an even integer: halve, round the magnitude up, double.
        // Halving and doubling are exact in binary floating point, so this introduces
        // no rounding of its own.
        Even => (v / 2.0).abs().ceil() * 2.0 * if v < 0.0 { -1.0 } else { 1.0 },
        // `libm` is the pure-Rust port of the same FDLIBM routines DuckDB reaches
        // through its libc, so these agree with it to the last bit rather than to a
        // series expansion's tolerance. `f64::gamma` is still unstable in std.
        Gamma => libm::tgamma(v),
        Lgamma => libm::lgamma(v),
        // Integer-only functions are handled by `eval_int_math`.
        Factorial | BitCount => unreachable!("integer path"),
    }
}

/// IEEE-754 `roundTiesToEven` for `f64` (`f64::round_ties_even` is stable only from
/// Rust 1.77; the workspace pins 1.85, but spelling it out keeps the tie rule visible
/// next to `round`'s opposite one).
#[inline]
fn round_ties_even(v: f64) -> f64 {
    let r = v.round();
    // `round` broke the tie away from zero. A tie is exactly a half, so detect it and
    // step back one when that landed on an odd integer.
    if (v - v.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - v.signum()
    } else {
        r
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
    let a = i.as_primitive::<Int64Type>();
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
        let a = a.as_primitive::<Int64Type>();
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

    /// `round` stays Int64 on an integer input and is exact above 2^53.
    ///
    /// Regression: unary and two-arg `round` both promoted integers to Float64 with the
    /// `floor`/`ceil`/`sqrt` blanket rule. But DuckDB returns BIGINT for `round(bigint)`,
    /// and the f64 round-trip silently corrupts: `round(2^53+1)` came back as `2^53`, and
    /// `round(i64::MAX, 0)` as `9.223372036854776e18`.
    #[test]
    fn round_is_exact_int64_above_2_pow_53() {
        let two53p1 = (1i64 << 53) + 1; // not representable as f64
        let vals = i64arr(vec![Some(two53p1), Some(i64::MAX), Some(-7)]);

        // Unary `round(x)`: the identity on an integer.
        let r = eval_math(MathFunc::Round, &vals).unwrap();
        assert_eq!(r.data_type(), &DataType::Int64);
        assert_eq!(as_i64(&r), vec![Some(two53p1), Some(i64::MAX), Some(-7)]);

        // Two-arg with non-negative digits: also the identity.
        for digits in [0i64, 2] {
            let r = eval_math2(Math2Func::Round, &vals, &i64arr(vec![Some(digits); 3])).unwrap();
            assert_eq!(r.data_type(), &DataType::Int64);
            assert_eq!(as_i64(&r), vec![Some(two53p1), Some(i64::MAX), Some(-7)]);
        }
    }

    /// A negative `digits` rounds an integer to that power of ten, half away from zero,
    /// without ever forming an overflowing intermediate.
    #[test]
    fn round_int_negative_digits() {
        let vals = i64arr(vec![
            Some(12345),
            Some(-12345),
            Some(-7),
            Some((1i64 << 53) + 1),
            None,
        ]);
        let r = eval_math2(Math2Func::Round, &vals, &i64arr(vec![Some(-2); 5])).unwrap();
        assert_eq!(r.data_type(), &DataType::Int64);
        assert_eq!(
            as_i64(&r),
            vec![
                Some(12300),
                Some(-12300),
                Some(0),
                Some(9_007_199_254_741_000),
                None
            ]
        );
        // An exponent past i64's range collapses to 0 rather than overflowing the scale.
        let wide = eval_math2(
            Math2Func::Round,
            &i64arr(vec![Some(12345)]),
            &i64arr(vec![Some(-40)]),
        )
        .unwrap();
        assert_eq!(as_i64(&wide), vec![Some(0)]);
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
