//! Binary-operator evaluation for `Expr::Binary` plus the shared numeric/boolean
//! coercion helpers (split out of `lib.rs`).

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanArray, Datum, Int64Array, RecordBatch, Scalar};
use arrow::compute::cast;
use arrow::compute::kernels::{boolean, cmp, numeric};
use arrow::datatypes::DataType;

use crate::eval::date::add_months;
use crate::{BinaryOp, Expr, ExprError, Literal};

/// Fast path for `<numeric column-expr> <arith|cmp> <numeric literal>` (in either
/// operand order): broadcast the literal as a length-1 `Scalar` (a `Datum`) instead
/// of materializing it as a full N-length array. Bit-identical to [`eval_binary`] —
/// same kernels, same Int/Float promotion — so the interpreter oracle is unchanged.
/// Returns `None` for any other shape, so the caller falls back to the array path.
pub(crate) fn try_scalar_binary(
    op: BinaryOp,
    left: &Expr,
    right: &Expr,
    batch: &RecordBatch,
) -> Result<Option<ArrayRef>, ExprError> {
    use BinaryOp::*;
    use DataType::{Float64, Int64};

    // Only arithmetic and comparison broadcast cleanly and share the array path's
    // kernels. And/Or/Concat/bitwise/Div/Mod/AddMonths keep the array path.
    if !matches!(op, Add | Sub | Mul | Eq | Ne | Lt | Le | Gt | Ge) {
        return Ok(None);
    }
    // Exactly one operand a numeric (Int/Float) literal; the other is the column.
    let is_num_lit = |e: &Expr| matches!(e, Expr::Lit { value } if matches!(value, Literal::Int(_) | Literal::Float(_)));
    let (arr_expr, lit_expr, lit_on_right) = match (left, right) {
        (a, l) if is_num_lit(l) => (a, l, true),
        (l, a) if is_num_lit(l) => (a, l, false),
        _ => return Ok(None),
    };
    let Expr::Lit { value: lit } = lit_expr else {
        return Ok(None);
    };

    let arr = arr_expr.eval(batch)?;
    // Only Int64/Float64 columns broadcast here; defer decimals/strings/dates to the
    // array path's coercion, which handles their wider promotion rules.
    if !matches!(arr.data_type(), Int64 | Float64) {
        return Ok(None);
    }
    let lit_arr = lit.to_array(1);
    // Mirror `coerce_numeric`: a mixed Int/Float pair promotes to Float64.
    let (arr, lit_arr) = match (arr.data_type(), lit_arr.data_type()) {
        (Int64, Float64) => (cast(&arr, &Float64)?, lit_arr),
        (Float64, Int64) => (arr, cast(&lit_arr, &Float64)?),
        _ => (arr, lit_arr),
    };

    let scalar = Scalar::new(lit_arr);
    let arr_dyn: &dyn Array = arr.as_ref();
    let arr_datum: &dyn Datum = &arr_dyn;
    let scalar_datum: &dyn Datum = &scalar;
    let (lhs, rhs) = if lit_on_right {
        (arr_datum, scalar_datum)
    } else {
        (scalar_datum, arr_datum)
    };

    let out: ArrayRef = match op {
        Add => numeric::add_wrapping(lhs, rhs)?,
        Sub => numeric::sub_wrapping(lhs, rhs)?,
        Mul => numeric::mul_wrapping(lhs, rhs)?,
        Eq => Arc::new(cmp::eq(lhs, rhs)?),
        Ne => Arc::new(cmp::neq(lhs, rhs)?),
        Lt => Arc::new(cmp::lt(lhs, rhs)?),
        Le => Arc::new(cmp::lt_eq(lhs, rhs)?),
        Gt => Arc::new(cmp::gt(lhs, rhs)?),
        Ge => Arc::new(cmp::gt_eq(lhs, rhs)?),
        _ => unreachable!("filtered to arith/cmp above"),
    };
    Ok(Some(out))
}

pub(crate) fn eval_binary(op: BinaryOp, l: &ArrayRef, r: &ArrayRef) -> Result<ArrayRef, ExprError> {
    use BinaryOp::*;
    // SQL-style implicit numeric promotion: mixed Int64/Float64 operands are
    // promoted to Float64 so `qty * price` (int × float) works as expected.
    let (l, r) = coerce_numeric(l, r)?;
    let (l, r) = (&l, &r);
    let out: ArrayRef = match op {
        Eq => Arc::new(cmp::eq(&l.as_ref(), &r.as_ref())?),
        Ne => Arc::new(cmp::neq(&l.as_ref(), &r.as_ref())?),
        Lt => Arc::new(cmp::lt(&l.as_ref(), &r.as_ref())?),
        Le => Arc::new(cmp::lt_eq(&l.as_ref(), &r.as_ref())?),
        Gt => Arc::new(cmp::gt(&l.as_ref(), &r.as_ref())?),
        Ge => Arc::new(cmp::gt_eq(&l.as_ref(), &r.as_ref())?),
        // Wrapping integer arithmetic (no-op for floats, which stay IEEE). This is
        // the bit-for-bit match for the Cranelift JIT's `iadd/isub/imul`, which
        // wrap on i64 overflow; the *checked* `numeric::add/sub/mul` would error
        // and so diverge from the compiled tier. Semantics match Polars / Rust
        // release. (DuckDB-style promotion to a wider type on overflow is a
        // separate, output-type-changing enhancement.)
        Add => numeric::add_wrapping(&l.as_ref(), &r.as_ref())?,
        Sub => numeric::sub_wrapping(&l.as_ref(), &r.as_ref())?,
        Mul => numeric::mul_wrapping(&l.as_ref(), &r.as_ref())?,
        // Integer div/rem by zero is a hardware trap (the kernel aborts the
        // process), so guard it and return a clean error. Float division is IEEE
        // (inf/nan), so it needs no guard.
        Div => {
            reject_zero_divisor(r)?;
            numeric::div(&l.as_ref(), &r.as_ref())?
        }
        Mod => {
            reject_zero_divisor(r)?;
            numeric::rem(&l.as_ref(), &r.as_ref())?
        }
        // SQL three-valued logic: `FALSE AND NULL` is FALSE, `TRUE OR NULL` is
        // TRUE (a known-controlling operand wins over an unknown). Arrow's plain
        // `and`/`or` propagate the null instead, so use the Kleene kernels to match
        // DuckDB. With null-free operands these reduce to plain and/or, so the
        // JIT's bitwise band/bor (null-free only) stays bit-for-bit identical.
        And => Arc::new(boolean::and_kleene(as_bool(l, "and")?, as_bool(r, "and")?)?),
        Or => Arc::new(boolean::or_kleene(as_bool(l, "or")?, as_bool(r, "or")?)?),
        // SQL `||`: cast both operands to Utf8 and concatenate element-wise.
        // A null on either side yields a null (matching DuckDB's `||` operator).
        Concat => {
            use arrow::array::StringArray;
            use arrow::compute::kernels::concat_elements::concat_elements_utf8;
            let l = cast(l, &DataType::Utf8)?;
            let r = cast(r, &DataType::Utf8)?;
            // Casting to Utf8 always yields a StringArray, so the downcasts hold.
            let (ls, rs) = (
                l.as_any().downcast_ref::<StringArray>(),
                r.as_any().downcast_ref::<StringArray>(),
            );
            match (ls, rs) {
                (Some(ls), Some(rs)) => Arc::new(concat_elements_utf8(ls, rs)?),
                _ => {
                    return Err(ExprError::ExpectedString {
                        func: "concat".into(),
                        got: format!("{}, {}", l.data_type(), r.data_type()),
                    })
                }
            }
        }
        // Integer bitwise ops. Operands are coerced/cast to Int64.
        BitAnd | BitOr | BitXor | ShiftLeft | ShiftRight => {
            use arrow::compute::kernels::bitwise::{
                bitwise_and, bitwise_or, bitwise_shift_left, bitwise_shift_right, bitwise_xor,
            };
            let (li, ri) = (cast(l, &DataType::Int64)?, cast(r, &DataType::Int64)?);
            let (la, ra) = (
                li.as_any().downcast_ref::<Int64Array>().unwrap(),
                ri.as_any().downcast_ref::<Int64Array>().unwrap(),
            );
            match op {
                BitAnd => Arc::new(bitwise_and(la, ra)?),
                BitOr => Arc::new(bitwise_or(la, ra)?),
                BitXor => Arc::new(bitwise_xor(la, ra)?),
                ShiftLeft => Arc::new(bitwise_shift_left(la, ra)?),
                ShiftRight => Arc::new(bitwise_shift_right(la, ra)?),
                _ => unreachable!(),
            }
        }
        AddMonths => add_months(l, r)?,
    };
    Ok(out)
}

/// Reject a zero divisor for integer/decimal division and modulo, which would
/// otherwise trap the CPU and abort the process. Float divisors are left to IEEE.
fn reject_zero_divisor(divisor: &ArrayRef) -> Result<(), ExprError> {
    let has_zero = match divisor.data_type() {
        DataType::Int64 => divisor
            .as_any()
            .downcast_ref::<Int64Array>()
            .is_some_and(|a| a.iter().flatten().any(|v| v == 0)),
        DataType::Decimal128(_, _) => divisor
            .as_any()
            .downcast_ref::<arrow::array::Decimal128Array>()
            .is_some_and(|a| a.iter().flatten().any(|v| v == 0)),
        _ => false,
    };
    if has_zero {
        return Err(ExprError::DivideByZero);
    }
    Ok(())
}

/// Promote mixed Int64/Float64 operands to a common Float64 type (SQL semantics).
/// Same-typed operands (and non-numeric mixes) pass through unchanged.
pub(crate) fn coerce_numeric(
    l: &ArrayRef,
    r: &ArrayRef,
) -> Result<(ArrayRef, ArrayRef), ExprError> {
    use DataType::{Decimal128, Float64, Int64};
    match (l.data_type(), r.data_type()) {
        (Int64, Float64) => Ok((cast(l, &Float64)?, r.clone())),
        (Float64, Int64) => Ok((l.clone(), cast(r, &Float64)?)),
        // A numeric literal against a decimal column adopts the decimal's
        // precision/scale, so the comparison/arithmetic stays exact.
        (Decimal128(..), Int64 | Float64) => Ok((l.clone(), cast(r, l.data_type())?)),
        (Int64 | Float64, Decimal128(..)) => Ok((cast(l, r.data_type())?, r.clone())),
        _ => Ok((l.clone(), r.clone())),
    }
}

/// Downcast an array to `BooleanArray`, erroring with operator context.
pub(crate) fn as_bool<'a>(arr: &'a ArrayRef, op: &str) -> Result<&'a BooleanArray, ExprError> {
    arr.as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| ExprError::ExpectedBoolean {
            op: op.to_string(),
            got: arr.data_type().to_string(),
        })
}

#[cfg(test)]
mod scalar_path_tests {
    use super::*;
    use crate::Literal;
    use arrow::array::{Float64Array, Int64Array};
    use arrow::datatypes::{Field, Schema};

    fn batch(name: &str, col: ArrayRef) -> RecordBatch {
        let schema = Schema::new(vec![Field::new(name, col.data_type().clone(), true)]);
        RecordBatch::try_new(Arc::new(schema), vec![col]).unwrap()
    }

    /// The scalar fast path MUST equal the full-array path bit-for-bit, for every
    /// arith/cmp op, both operand orders, and Int/Float column-literal mixes
    /// (including nulls in the column).
    #[test]
    fn scalar_path_equals_array_path() {
        let int_col: ArrayRef = Arc::new(Int64Array::from(vec![Some(3), None, Some(-7), Some(0)]));
        let flt_col: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(2.5),
            Some(-1.0),
            None,
            Some(4.0),
        ]));
        let lits = [Literal::Int(2), Literal::Float(0.5), Literal::Int(0)];
        let ops = [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Eq,
            BinaryOp::Ne,
            BinaryOp::Lt,
            BinaryOp::Le,
            BinaryOp::Gt,
            BinaryOp::Ge,
        ];
        for (cname, col) in [("i", &int_col), ("f", &flt_col)] {
            let b = batch(cname, col.clone());
            for lit in &lits {
                for &op in &ops {
                    for lit_on_right in [true, false] {
                        let col_expr = Expr::Col {
                            name: cname.to_string(),
                        };
                        let lit_expr = Expr::Lit { value: lit.clone() };
                        let (l, r) = if lit_on_right {
                            (col_expr, lit_expr)
                        } else {
                            (lit_expr, col_expr)
                        };
                        let fast = try_scalar_binary(op, &l, &r, &b)
                            .unwrap()
                            .expect("fast path taken");
                        let la = l.eval(&b).unwrap();
                        let ra = r.eval(&b).unwrap();
                        let slow = eval_binary(op, &la, &ra).unwrap();
                        assert_eq!(
                            fast.as_ref(),
                            slow.as_ref(),
                            "mismatch op={op:?} lit={lit:?} on_right={lit_on_right} col={cname}"
                        );
                    }
                }
            }
        }
    }
}
