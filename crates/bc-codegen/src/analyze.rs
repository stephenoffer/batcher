//! Validate a [`bc_expr::Expr`] against the JIT's supported subset and infer the
//! scalar type each sub-expression evaluates to, recording referenced columns.

use arrow::array::RecordBatch;
use arrow::datatypes::{DataType, TimeUnit};

use crate::{libm_binary_symbol, libm_unary_symbol, CodegenError, ColumnSet, ScalarTy};

/// Whether `expr` is an integer literal safe to use as an integer divisor in the
/// JIT: nonzero (no div-by-zero trap) and not -1 (no `i64::MIN / -1` overflow
/// trap). For such a constant, cranelift `sdiv`/`srem` — truncate toward zero,
/// remainder takes the dividend's sign — are bit-identical to the interpreter's
/// Arrow `div`/`rem`, so an integer `x / k` / `x % k` compiles with exact parity.
fn is_safe_int_divisor(expr: &bc_expr::Expr) -> bool {
    matches!(
        expr,
        bc_expr::Expr::Lit {
            value: bc_expr::Literal::Int(k),
        } if *k != 0 && *k != -1
    )
}

/// Validate the expression and infer its result scalar type, recording every
/// referenced column in `cols`.
pub(crate) fn analyze(
    expr: &bc_expr::Expr,
    batch: &RecordBatch,
    cols: &mut ColumnSet,
) -> Result<ScalarTy, CodegenError> {
    use bc_expr::{BinaryOp::*, Expr, Literal};
    match expr {
        Expr::Col { name } => {
            let arr = batch
                .column_by_name(name)
                .ok_or_else(|| CodegenError::UnknownColumn(name.clone()))?;
            // Null handling is decided per batch in `CompiledExpr::eval`: a nullable
            // column falls back to the interpreter unless the whole expression is the
            // null-propagating subset, in which case the values are computed over the
            // raw buffer and a combined validity mask is applied to the result.
            let ty = match arr.data_type() {
                DataType::Int64 => ScalarTy::I64,
                DataType::Float64 => ScalarTy::F64,
                // Date32 is an i32 day count; the JIT supports it as a comparison-only
                // operand (loaded + sign-extended to i64). Comparison against another
                // Date32 is bit-identical to Arrow's date comparison.
                DataType::Date32 => ScalarTy::Date32,
                // tz-naive Timestamp(µs) is an i64 instant — comparison-only, like a
                // date. Other units/timezones fall back (they'd need rescaling).
                DataType::Timestamp(TimeUnit::Microsecond, None) => ScalarTy::TsUs,
                other => {
                    return Err(CodegenError::Unsupported(format!(
                        "column `{name}` has unsupported type {other:?}"
                    )))
                }
            };
            if !cols.order.iter().any(|c| c == name) {
                // Columns are passed as a runtime pointer array (`*const *const u8`),
                // so there is no fixed-arity ceiling on the distinct-column count.
                cols.order.push(name.clone());
                cols.ty.insert(name.clone(), ty);
            }
            Ok(ty)
        }
        Expr::Lit { value } => match value {
            Literal::Int(_) => Ok(ScalarTy::I64),
            Literal::Float(_) => Ok(ScalarTy::F64),
            Literal::Bool(_) => Err(CodegenError::Unsupported("bool literal".into())),
            Literal::Str(_) => Err(CodegenError::Unsupported("string literal".into())),
            // A date literal is an i32 day count (`Date32`); a timestamp literal is an
            // i64 microsecond instant (tz-naive `Timestamp(µs)`, per the interpreter's
            // `Literal::to_array`). Each compares against a column of the same type.
            Literal::Date(_) => Ok(ScalarTy::Date32),
            Literal::Timestamp(_) => Ok(ScalarTy::TsUs),
        },
        Expr::Binary { op, left, right } => {
            let l = analyze(left, batch, cols)?;
            let r = analyze(right, batch, cols)?;
            // Boolean AND/OR combine two boolean sub-results (e.g. compound filter
            // predicates `a > 1 AND b < 2`); on the JIT's null-free fast path this
            // is the bitwise op, matching the interpreter's non-Kleene `and`/`or`.
            if matches!(op, And | Or) {
                return if l == ScalarTy::Bool && r == ScalarTy::Bool {
                    Ok(ScalarTy::Bool)
                } else {
                    Err(CodegenError::Unsupported(
                        "and/or require boolean operands".into(),
                    ))
                };
            }
            if l == ScalarTy::Bool || r == ScalarTy::Bool {
                return Err(CodegenError::Unsupported(
                    "boolean operand to arithmetic/comparison".into(),
                ));
            }
            // Temporal types (date / timestamp) are comparison-only, and only against
            // the *same* temporal type (Arrow compares them by their integer value;
            // mixing with a numeric, or a date with a timestamp, would need a coercion
            // the JIT doesn't model). Anything else involving a temporal falls back.
            let is_temporal = |t: ScalarTy| matches!(t, ScalarTy::Date32 | ScalarTy::TsUs);
            if is_temporal(l) || is_temporal(r) {
                return if matches!(op, Eq | Ne | Lt | Le | Gt | Ge) && l == r {
                    Ok(ScalarTy::Bool)
                } else {
                    Err(CodegenError::Unsupported(
                        "temporal type supports comparison against the same type only".into(),
                    ))
                };
            }
            match op {
                Add | Sub | Mul | Div | Mod => {
                    // Promote to f64 if either side is f64 (Arrow semantics).
                    let result = if l == ScalarTy::F64 || r == ScalarTy::F64 {
                        ScalarTy::F64
                    } else {
                        ScalarTy::I64
                    };
                    // Integer div/rem by zero TRAPS (cranelift `sdiv`/`srem`), and
                    // `i64::MIN / -1` overflow-traps; both abort the process. The
                    // general case (a non-literal, possibly-zero divisor) stays on
                    // the interpreter, which guards a zero divisor. But a *constant*
                    // divisor that is neither 0 nor -1 can never trap, and
                    // cranelift's truncate-toward-zero `sdiv`/`srem` then match the
                    // interpreter's Arrow `div`/`rem` bit-for-bit — so the common
                    // `x / k` / `x % k` (bucketing) cases DO compile. Float div is
                    // IEEE (inf/nan), so it stays compilable unconditionally.
                    if matches!(op, Div | Mod)
                        && result == ScalarTy::I64
                        && !is_safe_int_divisor(right)
                    {
                        return Err(CodegenError::Unsupported(
                            "integer division by a non-constant (possibly zero) divisor".into(),
                        ));
                    }
                    Ok(result)
                }
                Eq | Ne | Lt | Le | Gt | Ge => Ok(ScalarTy::Bool),
                // String concatenation is not a scalar-numeric op; the
                // interpreter handles it (Utf8 cast + element-wise concat).
                Concat => Err(CodegenError::Unsupported("string concat".into())),
                BitAnd | BitOr | BitXor | ShiftLeft | ShiftRight => {
                    Err(CodegenError::Unsupported("bitwise op".into()))
                }
                AddMonths => Err(CodegenError::Unsupported("date month arithmetic".into())),
                And | Or => unreachable!("handled above"),
            }
        }
        Expr::Not { input } => match analyze(input, batch, cols)? {
            ScalarTy::Bool => Ok(ScalarTy::Bool),
            _ => Err(CodegenError::Unsupported(
                "not requires a boolean operand".into(),
            )),
        },
        Expr::Cast {
            input,
            dtype,
            try_cast,
        } => {
            // Only the SAFE, EXACT numeric casts compile; everything else falls
            // back to the interpreter (Arrow `cast` kernel) so parity holds.
            // `try_cast` (null-on-failure) is interpreter-only — its NULL-producing
            // semantics aren't modeled here, so always fall back.
            if *try_cast {
                return Err(CodegenError::Unsupported("try_cast".into()));
            }
            let inner = analyze(input, batch, cols)?;
            // Classify the target dtype name to a JIT scalar type. The name→type
            // vocabulary is resolved by the canonical `bc_arrow::dtype_from_name`
            // so aliases (`long`/`double`) never drift from the interpreter;
            // anything outside Int64/Float64 (int32, bool, string, date, ...) is
            // unsupported here and falls back to the interpreter.
            let target = match bc_arrow::dtype_from_name(dtype) {
                Some(DataType::Int64) => ScalarTy::I64,
                Some(DataType::Float64) => ScalarTy::F64,
                _ => {
                    return Err(CodegenError::Unsupported(format!(
                        "cast to dtype `{dtype}`"
                    )))
                }
            };
            match (inner, target) {
                // int64 -> float64: exact (`fcvt_from_sint` == Arrow int->float).
                (ScalarTy::I64, ScalarTy::F64) => Ok(ScalarTy::F64),
                // Same-type no-ops.
                (ScalarTy::I64, ScalarTy::I64) => Ok(ScalarTy::I64),
                (ScalarTy::F64, ScalarTy::F64) => Ok(ScalarTy::F64),
                // float64 -> int64: Arrow's rounding/saturation is subtle and
                // could mismatch `fcvt`, so leave it to the interpreter.
                (ScalarTy::F64, ScalarTy::I64) => {
                    Err(CodegenError::Unsupported("cast float64 -> int64".into()))
                }
                // Bool input (or any other combination) is unsupported.
                _ => Err(CodegenError::Unsupported("cast of boolean".into())),
            }
        }
        Expr::IsNull { .. } => Err(CodegenError::Unsupported("is_null".into())),
        Expr::IsNotNull { .. } => Err(CodegenError::Unsupported("is_not_null".into())),
        Expr::IsNan { .. } => Err(CodegenError::Unsupported("is_nan".into())),
        Expr::IsInf { .. } => Err(CodegenError::Unsupported("is_inf".into())),
        Expr::Case {
            branches,
            otherwise,
        } => {
            // The result is the common numeric type of `otherwise` and every
            // `then`: F64 if any of them is F64, else I64. Each `when` must be a
            // boolean. Any unsupported sub-expression (e.g. integer division)
            // bubbles up via `?`, so we never emit a Case containing a trapping
            // or non-compilable op.
            let otherwise_ty = analyze(otherwise, batch, cols)?;
            let mut result = match otherwise_ty {
                ScalarTy::I64 => ScalarTy::I64,
                ScalarTy::F64 => ScalarTy::F64,
                ScalarTy::Bool => {
                    return Err(CodegenError::Unsupported("case result is boolean".into()))
                }
                ScalarTy::Date32 | ScalarTy::TsUs => {
                    return Err(CodegenError::Unsupported("case result is temporal".into()))
                }
            };
            for branch in branches {
                let when_ty = analyze(&branch.when, batch, cols)?;
                if when_ty != ScalarTy::Bool {
                    return Err(CodegenError::Unsupported(
                        "case WHEN must be a boolean predicate".into(),
                    ));
                }
                match analyze(&branch.then, batch, cols)? {
                    ScalarTy::I64 => {}
                    ScalarTy::F64 => result = ScalarTy::F64,
                    ScalarTy::Bool => {
                        return Err(CodegenError::Unsupported("case THEN is boolean".into()))
                    }
                    ScalarTy::Date32 | ScalarTy::TsUs => {
                        return Err(CodegenError::Unsupported("case THEN is temporal".into()))
                    }
                }
            }
            Ok(result)
        }
        Expr::Str { .. } => Err(CodegenError::Unsupported("string function".into())),
        Expr::Date { .. } => Err(CodegenError::Unsupported("date function".into())),
        Expr::Image { .. } => Err(CodegenError::Unsupported("image function".into())),
        Expr::Audio { .. } => Err(CodegenError::Unsupported("audio function".into())),
        Expr::Video { .. } => Err(CodegenError::Unsupported("video function".into())),
        Expr::Coalesce { .. } => Err(CodegenError::Unsupported("coalesce".into())),
        Expr::InList { .. } => Err(CodegenError::Unsupported("in_list".into())),
        Expr::Array { .. } => Err(CodegenError::Unsupported("array literal".into())),
        Expr::Sequence { .. } => Err(CodegenError::Unsupported("sequence".into())),
        Expr::ListSet { .. } => Err(CodegenError::Unsupported("list set op".into())),
        Expr::ListTransform { .. } => Err(CodegenError::Unsupported("list transform".into())),
        Expr::ListFilter { .. } => Err(CodegenError::Unsupported("list filter".into())),
        Expr::MakeStruct { .. } => Err(CodegenError::Unsupported("struct construction".into())),
        Expr::ListJoin { .. } => Err(CodegenError::Unsupported("list join".into())),
        Expr::Math { func, input } => {
            use bc_expr::MathFunc::*;
            let inner = analyze(input, batch, cols)?;
            if matches!(inner, ScalarTy::Bool | ScalarTy::Date32 | ScalarTy::TsUs) {
                return Err(CodegenError::Unsupported(
                    "math function on boolean/temporal operand".into(),
                ));
            }
            match func {
                // `abs` preserves the input type (int abs -> int, float abs ->
                // float), matching the interpreter's `eval_math`.
                Abs => Ok(inner),
                // floor/ceil/sqrt/trunc always produce f64 (ints are promoted to
                // f64 first, exactly as the interpreter does via `cast`).
                Floor | Ceil | Sqrt | Trunc => Ok(ScalarTy::F64),
                // The transcendentals lower to a libm libcall (see
                // `libm_unary_symbol`); the int input is promoted to f64 first,
                // exactly as the interpreter does, so the result is f64.
                _ if libm_unary_symbol(*func).is_some() => Ok(ScalarTy::F64),
                // Round (different rounding mode), Sign (select), Degrees/Radians
                // (constant multiply) and Cot (reciprocal) are out of scope and
                // stay on the interpreter to preserve bit-for-bit parity.
                _ => Err(CodegenError::Unsupported(format!("math function {func:?}"))),
            }
        }
        Expr::List { .. } => Err(CodegenError::Unsupported("list function".into())),
        Expr::NullIf { .. } => Err(CodegenError::Unsupported("nullif".into())),
        Expr::Greatest { .. } => Err(CodegenError::Unsupported("greatest".into())),
        Expr::Least { .. } => Err(CodegenError::Unsupported("least".into())),
        Expr::Math2 { func, left, right } => {
            // `pow`/`atan2` lower to a libm libcall (see `libm_binary_symbol`);
            // `round(x, digits)` is not a single libm call and stays on the
            // interpreter. Both operands are promoted to f64 first (matching the
            // interpreter, which casts to Float64 before the float op).
            if libm_binary_symbol(*func).is_none() {
                return Err(CodegenError::Unsupported(format!(
                    "binary math function {func:?}"
                )));
            }
            let lt = analyze(left, batch, cols)?;
            let rt = analyze(right, batch, cols)?;
            if matches!(lt, ScalarTy::Bool | ScalarTy::Date32 | ScalarTy::TsUs)
                || matches!(rt, ScalarTy::Bool | ScalarTy::Date32 | ScalarTy::TsUs)
            {
                return Err(CodegenError::Unsupported(
                    "binary math function on boolean/temporal operand".into(),
                ));
            }
            Ok(ScalarTy::F64)
        }
        Expr::ListGet { .. } => Err(CodegenError::Unsupported("list index".into())),
        Expr::StructField { .. } => Err(CodegenError::Unsupported("struct field".into())),
        Expr::ListContains { .. } => Err(CodegenError::Unsupported("list contains".into())),
        Expr::ListPosition { .. } => Err(CodegenError::Unsupported("list position".into())),
        Expr::Map { .. } => Err(CodegenError::Unsupported("map function".into())),
        Expr::ListSlice { .. } => Err(CodegenError::Unsupported("list slice".into())),
        Expr::DateTrunc { .. } => Err(CodegenError::Unsupported("date_trunc".into())),
        Expr::Strftime { .. } => Err(CodegenError::Unsupported("strftime".into())),
        Expr::ConvertTimezone { .. } => Err(CodegenError::Unsupported("convert_timezone".into())),
        Expr::Strptime { .. } => Err(CodegenError::Unsupported("strptime".into())),
        Expr::ListBinary { .. } => Err(CodegenError::Unsupported("list binary op".into())),
        Expr::DateOffset { .. } => Err(CodegenError::Unsupported("offset_by".into())),
        Expr::WindowStart { .. } => Err(CodegenError::Unsupported("window_start".into())),
        Expr::WindowBuckets { .. } => Err(CodegenError::Unsupported("window_buckets".into())),
    }
}
