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
/// Largest `IN` set the JIT lowers to a compare chain. Kept equal to the interpreter's
/// `LINEAR_SCAN_MAX`: that is exactly the size at which it stops scanning linearly and
/// starts hashing, and a chain past that point is the slower algorithm.
pub(crate) const IN_LIST_JIT_MAX: usize = 8;

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
                // Floored division always falls back to the interpreter. Cranelift's
                // `sdiv` truncates toward zero, so compiling it would need the explicit
                // remainder-sign correction the interpreter applies — and the integer
                // arm additionally traps on a zero divisor. The interpreter is exact
                // and NULL-safe there; parity is worth more than the compile.
                FloorDiv => Err(CodegenError::Unsupported("floored division".into())),
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
        // `is_null` / `is_not_null` are *total*: they answer for a null input rather than
        // propagating it. That places them outside `is_null_propagating` deliberately,
        // and the two compile paths handle them differently.
        //
        // Under the Kleene ABI the answer is the operand's validity bit (negated for
        // `is_null`), and the result itself is always valid.
        //
        // On the null-free path the answer is a constant. That is safe rather than
        // optimistic: `CompiledExpr::eval` refuses any batch whose referenced columns
        // carry nulls when the expression is not `null_safe`, so a constant can only be
        // emitted for a batch that genuinely has none.
        Expr::IsNull { input } | Expr::IsNotNull { input } => {
            analyze(input, batch, cols)?;
            Ok(ScalarTy::Bool)
        }
        // `is_nan` / `is_inf` compile for a float operand. Both are null-propagating
        // (the interpreter carries the input's validity buffer through unchanged), so
        // they need no special validity handling beyond the unary propagation the
        // Kleene path already does.
        //
        // A non-float operand falls back. NaN and the infinities are impossible for an
        // integer, so the answer is a constant `false` there -- but the interpreter
        // reaches it by casting to Float64 first, and reproducing that constant here
        // would add a compile path whose only job is to be trivially true. Kyber folds
        // the non-nullable integer case away before the JIT ever sees it.
        Expr::IsNan { input } | Expr::IsInf { input } => match analyze(input, batch, cols)? {
            ScalarTy::F64 => Ok(ScalarTy::Bool),
            _ => Err(CodegenError::Unsupported(
                "is_nan/is_inf on a non-float operand".into(),
            )),
        },
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
        Expr::StrDyn { .. } => Err(CodegenError::Unsupported("string function".into())),
        Expr::ListGetDyn { .. } => Err(CodegenError::Unsupported("list element".into())),
        Expr::Date { .. } => Err(CodegenError::Unsupported("date function".into())),
        Expr::Image { .. } => Err(CodegenError::Unsupported("image function".into())),
        Expr::ImageCrop { .. } => Err(CodegenError::Unsupported("image crop".into())),
        Expr::Audio { .. } => Err(CodegenError::Unsupported("audio function".into())),
        Expr::Video { .. } => Err(CodegenError::Unsupported("video function".into())),
        // The JIT compiles fixed-width scalar arithmetic; every `.seq` op is a variable-length
        // pass over a row's bytes producing text, a list, or a struct. There is nothing here
        // for Cranelift to lower, so it declines and the interpreter — the oracle — runs it.
        // Declining is the contract: a JIT path that disagreed would be wrong *fast*.
        Expr::Seq { .. } => Err(CodegenError::Unsupported("sequence function".into())),
        // Geospatial work decodes WKB and walks variable-length geometry per row, which
        // has no scalar-register form at all — there is nothing here for a JIT to
        // compile, and falling back is the correct and permanent answer rather than a
        // gap waiting to be filled.
        Expr::Geo { .. } => Err(CodegenError::Unsupported("geospatial function".into())),
        // Declined rather than compiled, even though the arithmetic is exactly the
        // numeric subset this tier handles. The kernels call transcendental functions
        // (`atan2`, `asin`, `acos`, `sin_cos`) that the JIT would have to reach through
        // libm calls, and bit-for-bit agreement with the interpreter across those is a
        // parity claim nothing here proves. The interpreter's loop is already tight.
        Expr::Spatial { .. } => Err(CodegenError::Unsupported("rigid-body function".into())),
        Expr::Coalesce { .. } => Err(CodegenError::Unsupported("coalesce".into())),
        // `x IN (lit, ...)`: compiled as an OR-chain of equality compares, so the whole
        // surrounding predicate stays in one JIT pass instead of falling back wholesale.
        //
        // The cap mirrors the interpreter's own `LINEAR_SCAN_MAX`: below it the
        // interpreter linear-scans and an unrolled compare chain is the same shape; above
        // it the interpreter switches to a hash probe that a chain would lose to badly.
        // So this is not "as far as the JIT can go", it is "as far as the chain wins".
        //
        // Every literal must convert to the operand's type. The interpreter *filters* a
        // non-convertible literal out of the set (`filter_map`), so a mixed set is a
        // different set than it looks; rather than reproduce that filtering, a mixed set
        // is declined outright.
        Expr::InList { input, set } => {
            if set.is_empty() || set.len() > IN_LIST_JIT_MAX {
                return Err(CodegenError::Unsupported("in_list arity".into()));
            }
            match analyze(input, batch, cols)? {
                ScalarTy::I64 if set.iter().all(|l| matches!(l, bc_expr::Literal::Int(_))) => {
                    Ok(ScalarTy::Bool)
                }
                ScalarTy::F64
                    if set.iter().all(|l| {
                        matches!(l, bc_expr::Literal::Int(_) | bc_expr::Literal::Float(_))
                    }) =>
                {
                    Ok(ScalarTy::Bool)
                }
                _ => Err(CodegenError::Unsupported(
                    "in_list operand/literal types".into(),
                )),
            }
        }
        Expr::Array { .. } => Err(CodegenError::Unsupported("array literal".into())),
        // The row hash reads whole typed values (strings, canonicalized floats); the
        // JIT's numeric subset cannot express it, so the interpreter runs it.
        Expr::Hash { .. } => Err(CodegenError::Unsupported("hash".into())),
        Expr::Sequence { .. } => Err(CodegenError::Unsupported("sequence".into())),
        Expr::ListSet { .. } => Err(CodegenError::Unsupported("list set op".into())),
        Expr::ListZip { .. } => Err(CodegenError::Unsupported("list arithmetic".into())),
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
                // `abs` preserves the input type (int abs -> int, float abs -> float),
                // matching the interpreter's `eval_math`. Integer `abs(i64::MIN)` saturates
                // to i64::MAX in both tiers (see `emit.rs` / `eval/math.rs`).
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
        // `greatest`/`least` fold to a select chain over the engine's float identity.
        //
        // They *skip* nulls rather than propagating them, so they are absent from both
        // `kleene_supported` and `is_null_propagating`: like `Case`, they compile only on
        // the null-free path, where every operand is present and the skip never happens.
        //
        // Every operand must already share one scalar type. The interpreter promotes a
        // mixed `greatest(int, float)` via `coerce_numeric`; reproducing that promotion
        // here would add a second coercion model to keep in step with it, so a mixed call
        // falls back instead.
        Expr::Greatest { inputs } | Expr::Least { inputs } => {
            let mut it = inputs.iter();
            let first = it.next().ok_or_else(|| {
                CodegenError::Unsupported("greatest/least with no arguments".into())
            })?;
            let ty = analyze(first, batch, cols)?;
            if !matches!(ty, ScalarTy::I64 | ScalarTy::F64) {
                return Err(CodegenError::Unsupported(
                    "greatest/least operand type".into(),
                ));
            }
            for arg in it {
                if analyze(arg, batch, cols)? != ty {
                    return Err(CodegenError::Unsupported(
                        "greatest/least with mixed operand types".into(),
                    ));
                }
            }
            Ok(ty)
        }
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
        Expr::ListSimhash { .. } => Err(CodegenError::Unsupported("list simhash".into())),
        Expr::StructField { .. } => Err(CodegenError::Unsupported("struct field".into())),
        Expr::ListContains { .. } => Err(CodegenError::Unsupported("list contains".into())),
        Expr::ListPosition { .. } => Err(CodegenError::Unsupported("list position".into())),
        Expr::Map { .. } => Err(CodegenError::Unsupported("map function".into())),
        Expr::ListSlice { .. } => Err(CodegenError::Unsupported("list slice".into())),
        Expr::DateTrunc { .. } => Err(CodegenError::Unsupported("date_trunc".into())),
        // Calendar construction needs a date library; the interpreter owns it.
        Expr::MakeTemporal { .. } => Err(CodegenError::Unsupported("temporal constructor".into())),
        Expr::Strftime { .. } => Err(CodegenError::Unsupported("strftime".into())),
        Expr::ConvertTimezone { .. } => Err(CodegenError::Unsupported("convert_timezone".into())),
        Expr::Strptime { .. } => Err(CodegenError::Unsupported("strptime".into())),
        Expr::ListBinary { .. } => Err(CodegenError::Unsupported("list binary op".into())),
        Expr::DateOffset { .. } => Err(CodegenError::Unsupported("offset_by".into())),
        Expr::WindowStart { .. } => Err(CodegenError::Unsupported("window_start".into())),
        Expr::WindowBuckets { .. } => Err(CodegenError::Unsupported("window_buckets".into())),
    }
}
