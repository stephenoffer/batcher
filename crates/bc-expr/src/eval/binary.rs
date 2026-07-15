//! Binary-operator evaluation for `Expr::Binary` plus the shared numeric/boolean
//! coercion helpers (split out of `lib.rs`).

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, Datum, Int64Array, RecordBatch, Scalar,
};
use arrow::compute::cast;
use arrow::compute::kernels::{boolean, cmp, numeric};
use arrow::datatypes::DataType;

use crate::eval::date::add_months;
use crate::{BinaryOp, Expr, ExprError, Literal};

/// Fast path for `<dictionary column> <cmp> <literal>` (in either operand order): compare the
/// **dictionary values** — one entry per *distinct* value — and gather one bit per row through
/// the keys, instead of decoding the column and comparing every row.
///
/// This is the same dict-acceleration `InList` already uses, extended to the comparison
/// operators, and it is the difference between a dictionary being an asset and a liability.
/// The scalar path decodes at the `Col` leaf (`decode_dict`), so before this a filter on a
/// dictionary column *materialized every row's value* and then compared all N of them — strictly
/// more work than if the column had never been dictionary-encoded at all. `assign_groups_dict`
/// records the same finding from the group-by side: pre-fast-path, a dictionary column measured
/// **~7x slower** than the decoded one. Both paths still do exactly one `take`; this one's
/// comparison runs over the cardinality (often a handful of entries) rather than the row count.
///
/// Measured, 6M rows × 25 distinct values (TPC-H `l_shipmode` / ClickBench's low-cardinality
/// string shape), `m = 'AIR'`: **7.4 ms vs 144.9 ms — 19.6x**, against the same column *already
/// decoded*. The old dictionary path was strictly worse still, since it paid the decode on top
/// of that 144.9 ms.
///
/// **Bit-identical to the decoded path**, which is what lets it live in the oracle: decoding is
/// `take(values, keys)`, and an elementwise comparison commutes with a gather —
/// `cmp(take(values, keys), lit) == take(cmp(values, lit), keys)`. Nulls carry through the same
/// identity: `take` maps a null key to a null output, and a null dictionary value to its own
/// null comparison bit. The operand order is preserved rather than flipped, so `lit < col` uses
/// the same kernel it always did and no operator needs inverting.
///
/// Returns `None` for any other shape (non-dictionary column, non-comparison operator, two
/// columns), so the caller falls back to the array path.
pub(crate) fn try_dict_compare(
    op: BinaryOp,
    left: &Expr,
    right: &Expr,
    batch: &RecordBatch,
) -> Result<Option<ArrayRef>, ExprError> {
    use BinaryOp::*;

    if !matches!(op, Eq | Ne | Lt | Le | Gt | Ge) {
        return Ok(None);
    }
    // Exactly one operand a plain column over a dictionary, the other a literal. The column is
    // read from the batch directly rather than through `eval`, which would decode it at the
    // leaf — the whole point is to never materialize the decoded column.
    let (dict, lit, col_on_left) = match (left, right) {
        (Expr::Col { name }, Expr::Lit { value }) => (dict_column(batch, name), value, true),
        (Expr::Lit { value }, Expr::Col { name }) => (dict_column(batch, name), value, false),
        _ => return Ok(None),
    };
    let Some(dict) = dict else { return Ok(None) };

    let dict = dict.as_any_dictionary();
    let values = dict.values();
    // One literal per *distinct* value, not per row.
    let lit_arr = lit.to_array(values.len());
    let over_values = if col_on_left {
        eval_binary(op, values, &lit_arr)?
    } else {
        eval_binary(op, &lit_arr, values)?
    };
    Ok(Some(arrow::compute::take(&over_values, dict.keys(), None)?))
}

/// The named column when it is dictionary-encoded, else `None` (including when absent — an
/// unknown column is the array path's error to raise, with its own message).
fn dict_column(batch: &RecordBatch, name: &str) -> Option<ArrayRef> {
    let arr = batch.column_by_name(name)?;
    matches!(arr.data_type(), DataType::Dictionary(_, _)).then(|| arr.clone())
}

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
        // Wrapping integer arithmetic (no-op for floats, which stay IEEE), bit-identical
        // to `eval_binary` and to the Cranelift JIT's `iadd/isub/imul`. See the parity
        // note there — a *checked* kernel would error on overflow and diverge from the
        // compiled tier, breaking the hard interpreter == JIT invariant.
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
    // Comparison kernels (`cmp::eq` …) require *identical* decimal precision AND scale;
    // arithmetic kernels align scales themselves. Two decimal columns of differing scale
    // (`Decimal128(10,1)` vs `Decimal128(10,2)`, e.g. `1.0 = 1.00`) would otherwise raise
    // "Invalid comparison operation" on a reachable data path — DuckDB compares them equal
    // by widening to a common scale. Align here only for the comparison arms, so `*`/`+`
    // keep their own scale-propagation rules untouched.
    let (l, r) = if matches!(op, Eq | Ne | Lt | Le | Gt | Ge) {
        align_decimals_for_cmp(&l, &r)?
    } else {
        (l, r)
    };
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
        // and so diverge from the compiled tier — a violation of the hard
        // interpreter == JIT invariant (CLAUDE.md #6). Semantics match Polars / Rust
        // release. (DuckDB-style overflow-to-error is desirable but subordinate to
        // that hard gate; matching it would require the JIT to also error on
        // overflow, an ABI/SIMD change deferred as a separate enhancement.)
        Add => numeric::add_wrapping(&l.as_ref(), &r.as_ref())?,
        Sub => numeric::sub_wrapping(&l.as_ref(), &r.as_ref())?,
        Mul => numeric::mul_wrapping(&l.as_ref(), &r.as_ref())?,
        // Integer div/rem by zero is a hardware trap (the kernel aborts the process).
        // DuckDB returns NULL for `i % 0` / integer `i / 0` rather than erroring, so
        // sanitize the divisor and null the offending rows. Float division is IEEE
        // (inf/nan), so it passes straight through.
        Div => int_div_or_mod(true, l, r)?,
        Mod => int_div_or_mod(false, l, r)?,
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
                bitwise_and, bitwise_or, bitwise_shift_left, bitwise_xor,
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
                // Arrow's `>>` masks the shift amount to its low 6 bits
                // (`wrapping_shr`), so a negative or ≥ 64 amount wraps to an in-range
                // shift — `-7 >> -1` returned `-1`. DuckDB defines an out-of-range
                // shift as 0, so shift arithmetically only for `0 ≤ s < 64` and
                // yield 0 otherwise.
                ShiftRight => Arc::new(arithmetic_shift_right(la, ra)),
                _ => unreachable!(),
            }
        }
        AddMonths => add_months(l, r)?,
    };
    Ok(out)
}

/// Arithmetic right shift with DuckDB out-of-range semantics: an amount outside
/// `0..64` yields 0 (arrow's `wrapping_shr` would instead mask it into range). Nulls on
/// either side propagate. Right shift is sign-extending, matching DuckDB's `>>` on BIGINT.
fn arithmetic_shift_right(values: &Int64Array, amounts: &Int64Array) -> Int64Array {
    (0..values.len())
        .map(|i| {
            if values.is_null(i) || amounts.is_null(i) {
                return None;
            }
            let (v, s) = (values.value(i), amounts.value(i));
            Some(if (0..64).contains(&s) { v >> s } else { 0 })
        })
        .collect()
}

/// Integer/decimal division (`is_div`) or modulo, with DuckDB zero-divisor semantics:
/// a zero divisor yields NULL for that row (not an error and not a CPU trap). The naked
/// kernel would trap the process on an integer zero divisor, so zero divisors are
/// replaced by 1 before the kernel runs and the corresponding output rows are then
/// nulled. Float divisors are IEEE (inf/nan) and pass straight through.
fn int_div_or_mod(is_div: bool, l: &ArrayRef, r: &ArrayRef) -> Result<ArrayRef, ExprError> {
    let kernel = |l: &ArrayRef, r: &ArrayRef| -> Result<ArrayRef, ExprError> {
        Ok(if is_div {
            numeric::div(&l.as_ref(), &r.as_ref())?
        } else {
            numeric::rem(&l.as_ref(), &r.as_ref())?
        })
    };
    // Only Int64/Decimal128 divisors can trap; anything else (notably Float64) is IEEE.
    let (safe_r, is_zero) = match r.data_type() {
        DataType::Int64 => {
            let a = r.as_any().downcast_ref::<Int64Array>().expect("int64");
            if !a.iter().flatten().any(|v| v == 0) {
                return kernel(l, r);
            }
            let safe: Int64Array = a
                .iter()
                .map(|o| o.map(|v| if v == 0 { 1 } else { v }))
                .collect();
            let zero: BooleanArray = a.iter().map(|o| o.map(|v| v == 0)).collect();
            (Arc::new(safe) as ArrayRef, zero)
        }
        DataType::Decimal128(_, _) => {
            use arrow::array::Decimal128Array;
            let a = r
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .expect("decimal128");
            if !a.iter().flatten().any(|v| v == 0) {
                return kernel(l, r);
            }
            // Rebuild with the same precision/scale, swapping 0 → 1.
            let safe: Decimal128Array = a
                .iter()
                .map(|o| o.map(|v| if v == 0 { 1 } else { v }))
                .collect::<Decimal128Array>()
                .with_data_type(r.data_type().clone());
            let zero: BooleanArray = a.iter().map(|o| o.map(|v| v == 0)).collect();
            (Arc::new(safe) as ArrayRef, zero)
        }
        _ => return kernel(l, r),
    };
    let out = kernel(l, &safe_r)?;
    // Null every row whose divisor was zero (a null `is_zero` element — a null divisor —
    // leaves the already-null kernel output untouched).
    Ok(arrow::compute::nullif(&out, &is_zero)?)
}

/// Widen two decimal operands to a common precision/scale so a comparison kernel (which
/// demands identical decimal types) can run — DuckDB compares `DECIMAL(10,1)` against
/// `DECIMAL(10,2)` by casting both to a common `DECIMAL`. The common scale is
/// `max(s1, s2)`; the common precision covers the larger integer part plus that scale,
/// capped at Decimal128's 38 digits. Non-decimal or already-identical operands (and any
/// pair that isn't two `Decimal128`s — e.g. Decimal256 or a mixed width) pass through
/// unchanged, deferring to the existing path.
fn align_decimals_for_cmp(
    l: &ArrayRef,
    r: &ArrayRef,
) -> Result<(ArrayRef, ArrayRef), ExprError> {
    use DataType::Decimal128;
    if let (Decimal128(p1, s1), Decimal128(p2, s2)) = (l.data_type(), r.data_type()) {
        if (p1, s1) == (p2, s2) {
            return Ok((l.clone(), r.clone()));
        }
        let scale = *s1.max(s2);
        // Integer-digit budget on each side is `precision - scale`; the common precision
        // is the larger budget plus the common scale.
        let int_digits = (*p1 as i16 - *s1 as i16).max(*p2 as i16 - *s2 as i16);
        let precision = ((int_digits + scale as i16).clamp(1, 38)) as u8;
        let common = Decimal128(precision, scale);
        return Ok((cast(l, &common)?, cast(r, &common)?));
    }
    Ok((l.clone(), r.clone()))
}

/// Promote mixed operands to a common type before a binary op (SQL semantics):
/// Int64/Float64 → Float64, numeric/decimal → the decimal type, and a string against a
/// binary-typed column → the binary type. Same-typed operands pass through unchanged.
pub(crate) fn coerce_numeric(
    l: &ArrayRef,
    r: &ArrayRef,
) -> Result<(ArrayRef, ArrayRef), ExprError> {
    use DataType::{
        Binary, Date32, Date64, Decimal128, Float64, Int64, LargeBinary, LargeUtf8, Timestamp, Utf8,
    };
    match (l.data_type(), r.data_type()) {
        (Int64, Float64) => Ok((cast(l, &Float64)?, r.clone())),
        (Float64, Int64) => Ok((l.clone(), cast(r, &Float64)?)),
        // An *integer* against a decimal adopts the decimal's precision/scale, so the
        // arithmetic/comparison stays exact (DuckDB widens INTEGER into the DECIMAL).
        (Decimal128(..), Int64) => Ok((l.clone(), cast(r, l.data_type())?)),
        (Int64, Decimal128(..)) => Ok((cast(l, r.data_type())?, r.clone())),
        // A *float* against a decimal promotes to Float64 (DuckDB: DOUBLE dominates
        // DECIMAL, casting the decimal up to DOUBLE). Casting the float *down* to the
        // decimal's scale instead silently truncates the float's sub-scale precision —
        // e.g. `0.3333333333 + d` collapsed to `0.33` — and defeated `a / b`, which
        // lowers to `div(cast(a, float64), b)` precisely to force true (double)
        // division but was then re-narrowed to a truncated decimal quotient.
        (Decimal128(..), Float64) => Ok((cast(l, &Float64)?, r.clone())),
        (Float64, Decimal128(..)) => Ok((l.clone(), cast(r, &Float64)?)),
        // A Utf8 date/time literal (`'2013-07-01'`) against a temporal column: cast the
        // string to the column's exact temporal type (`Date32`/`Date64`/`Timestamp(unit,
        // tz)`), which arrow parses from ISO-8601 — matching DuckDB, which casts the
        // string literal to the column's DATE/TIMESTAMP type for the comparison.
        (Date32 | Date64 | Timestamp(..), Utf8 | LargeUtf8) => {
            Ok((l.clone(), cast(r, l.data_type())?))
        }
        (Utf8 | LargeUtf8, Date32 | Date64 | Timestamp(..)) => {
            Ok((cast(l, r.data_type())?, r.clone()))
        }
        // A Utf8 string literal compared to a Binary-typed column — the shape ClickBench's
        // `hits` produces, since its string columns arrive as `Binary` (no UTF-8 logical
        // annotation) while a SQL literal like `''` is `Utf8`. Cast the Utf8 side to the
        // binary type: `Utf8 -> Binary` is a zero-copy, never-failing reinterpret (offsets
        // + bytes are identical), and a lexicographic byte compare equals a string compare
        // for valid UTF-8, so `=`/`<>`/`<`/`>` match DuckDB's VARCHAR semantics.
        (Binary, Utf8 | LargeUtf8) => Ok((l.clone(), cast(r, &Binary)?)),
        (Utf8 | LargeUtf8, Binary) => Ok((cast(l, &Binary)?, r.clone())),
        (LargeBinary, Utf8 | LargeUtf8) => Ok((l.clone(), cast(r, &LargeBinary)?)),
        (Utf8 | LargeUtf8, LargeBinary) => Ok((cast(l, &LargeBinary)?, r.clone())),
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
mod arith_semantics_tests {
    use super::*;
    use arrow::array::Int64Array;

    fn i64arr(v: Vec<Option<i64>>) -> ArrayRef {
        Arc::new(Int64Array::from(v))
    }
    fn as_i64(a: &ArrayRef) -> Vec<Option<i64>> {
        let a = a.as_any().downcast_ref::<Int64Array>().expect("i64");
        (0..a.len())
            .map(|i| (!a.is_null(i)).then(|| a.value(i)))
            .collect()
    }

    /// Scalar integer `+`/`-`/`*` WRAP on i64 overflow (Polars / Rust-release semantics),
    /// matching the Cranelift JIT's `iadd/isub/imul` bit-for-bit — the hard interpreter
    /// == JIT invariant (CLAUDE.md #6). DuckDB-style overflow-to-error is deferred so the
    /// two tiers cannot diverge; see the arithmetic comment in `eval_binary`.
    #[test]
    fn integer_arithmetic_overflow_wraps() {
        let max = i64arr(vec![Some(i64::MAX)]);
        assert_eq!(
            as_i64(&eval_binary(BinaryOp::Add, &max, &i64arr(vec![Some(7)])).unwrap()),
            vec![Some(i64::MAX.wrapping_add(7))]
        );
        assert_eq!(
            as_i64(&eval_binary(BinaryOp::Mul, &max, &i64arr(vec![Some(7)])).unwrap()),
            vec![Some(i64::MAX.wrapping_mul(7))]
        );
        // In-range arithmetic is exact.
        assert_eq!(
            as_i64(
                &eval_binary(
                    BinaryOp::Add,
                    &i64arr(vec![Some(2)]),
                    &i64arr(vec![Some(3)])
                )
                .unwrap()
            ),
            vec![Some(5)]
        );
    }

    /// Integer `%` / `/` by zero yields NULL for that row (DuckDB), not an error or a CPU
    /// trap; a non-zero divisor is unaffected and a null divisor stays null.
    #[test]
    fn integer_mod_div_by_zero_is_null() {
        let l = i64arr(vec![Some(7), Some(-7), Some(5), Some(9)]);
        let r = i64arr(vec![Some(0), Some(0), Some(2), None]);
        assert_eq!(
            as_i64(&eval_binary(BinaryOp::Mod, &l, &r).unwrap()),
            vec![None, None, Some(1), None]
        );
        assert_eq!(
            as_i64(&eval_binary(BinaryOp::Div, &l, &r).unwrap()),
            vec![None, None, Some(2), None]
        );
        // No zeros → straight kernel result.
        assert_eq!(
            as_i64(
                &eval_binary(
                    BinaryOp::Mod,
                    &i64arr(vec![Some(7)]),
                    &i64arr(vec![Some(3)])
                )
                .unwrap()
            ),
            vec![Some(1)]
        );
    }

    /// Right shift by a negative or ≥ 64 amount yields 0 (DuckDB), not the masked value
    /// arrow's `wrapping_shr` produced (`-7 >> -1` gave `-1`).
    #[test]
    fn right_shift_out_of_range_is_zero() {
        let v = i64arr(vec![
            Some(-7),
            Some(-7),
            Some(-7),
            Some(-7),
            Some(i64::MIN),
            Some(-7),
        ]);
        let s = i64arr(vec![Some(-1), Some(64), Some(1), Some(63), Some(63), None]);
        assert_eq!(
            as_i64(&eval_binary(BinaryOp::ShiftRight, &v, &s).unwrap()),
            // -1 → 0, 64 → 0, 1 → -4 (arithmetic), 63 → -1, MIN>>63 → -1, null → null
            vec![Some(0), Some(0), Some(-4), Some(-1), Some(-1), None]
        );
    }

    /// A `Float64` operand against a `Decimal128` operand promotes to `Float64`
    /// (DuckDB: DOUBLE dominates DECIMAL), preserving the float's sub-scale
    /// precision. Previously the float was narrowed *into* the decimal's scale,
    /// truncating `0.3333333333 + 1.00` to `1.33` instead of `1.3333333333`, and
    /// re-narrowing `a / b` (lowered as `div(cast(a,f64), b)`) to a scale-6 decimal
    /// quotient rather than the true double DuckDB returns.
    #[test]
    fn float_vs_decimal_promotes_to_float64() {
        use arrow::array::{Decimal128Array, Float64Array};
        let f: ArrayRef = Arc::new(Float64Array::from(vec![Some(0.3333333333f64)]));
        let d: ArrayRef = Arc::new(
            Decimal128Array::from(vec![Some(100i128)]) // 1.00
                .with_precision_and_scale(10, 2)
                .unwrap(),
        );
        // Both operand orders promote to Float64 and keep full precision.
        for (l, r) in [(&f, &d), (&d, &f)] {
            let out = eval_binary(BinaryOp::Add, l, r).unwrap();
            assert_eq!(out.data_type(), &DataType::Float64);
            let a = out.as_any().downcast_ref::<Float64Array>().unwrap();
            assert!((a.value(0) - 1.3333333333f64).abs() < 1e-12);
        }
        // Division stays true double division, not a truncated decimal quotient.
        let ten: ArrayRef = Arc::new(Float64Array::from(vec![Some(10.0f64)]));
        let three: ArrayRef = Arc::new(
            Decimal128Array::from(vec![Some(300i128)])
                .with_precision_and_scale(10, 2)
                .unwrap(),
        );
        let q = eval_binary(BinaryOp::Div, &ten, &three).unwrap();
        assert_eq!(q.data_type(), &DataType::Float64);
        let a = q.as_any().downcast_ref::<Float64Array>().unwrap();
        assert!((a.value(0) - 3.3333333333333335f64).abs() < 1e-12);
    }

    /// Comparing two decimal columns of *different* scale (`1.0` as `Decimal128(10,1)`
    /// vs `1.00` as `Decimal128(10,2)`) must compare equal (DuckDB widens to a common
    /// scale), not raise "Invalid comparison operation" as the bare kernel does.
    #[test]
    fn decimal_compare_different_scales() {
        use arrow::array::{BooleanArray, Decimal128Array};
        let a: ArrayRef = Arc::new(
            Decimal128Array::from(vec![Some(10i128), Some(15), Some(-20)]) // 1.0, 1.5, -2.0
                .with_precision_and_scale(10, 1)
                .unwrap(),
        );
        let b: ArrayRef = Arc::new(
            Decimal128Array::from(vec![Some(100i128), Some(140), Some(-200)]) // 1.00, 1.40, -2.00
                .with_precision_and_scale(10, 2)
                .unwrap(),
        );
        let eq = eval_binary(BinaryOp::Eq, &a, &b).unwrap();
        let eq = eq.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(
            (0..eq.len()).map(|i| eq.value(i)).collect::<Vec<_>>(),
            vec![true, false, true] // 1.0=1.00, 1.5≠1.40, -2.0=-2.00
        );
        let gt = eval_binary(BinaryOp::Gt, &a, &b).unwrap();
        let gt = gt.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(
            (0..gt.len()).map(|i| gt.value(i)).collect::<Vec<_>>(),
            vec![false, true, false] // 1.5 > 1.40
        );
    }
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

    /// A `Binary`-typed column compared to a `Utf8` string literal (ClickBench's `hits`
    /// shape) coerces to a byte compare instead of erroring on the type mismatch, and the
    /// result equals a plain string compare.
    #[test]
    fn binary_column_vs_utf8_literal_compares() {
        use arrow::array::{BinaryArray, BooleanArray, StringArray};
        let bin: ArrayRef = Arc::new(BinaryArray::from_opt_vec(vec![
            Some(b"a".as_ref()),
            Some(b""),
            None,
        ]));
        let lit: ArrayRef = Arc::new(StringArray::from(vec![Some(""), Some(""), Some("")]));
        let out = eval_binary(BinaryOp::Ne, &bin, &lit).unwrap();
        let got = out.as_any().downcast_ref::<BooleanArray>().unwrap();
        // "a" <> "" is true, "" <> "" is false, NULL <> "" is null.
        assert_eq!(got.value(0), true);
        assert_eq!(got.value(1), false);
        assert!(got.is_null(2));
    }

    /// A `Timestamp`/`Date32` column compared to a `Utf8` date-string literal coerces by
    /// parsing the literal to the column's temporal type (ClickBench's `EventDate >= '…'`).
    #[test]
    fn date_column_vs_utf8_literal_compares() {
        use arrow::array::{BooleanArray, Date32Array, StringArray};
        // 2013-07-01 is day 15887 from the epoch; use a value on each side of it.
        let dates: ArrayRef = Arc::new(Date32Array::from(vec![15886, 15887, 15888]));
        let lit: ArrayRef = Arc::new(StringArray::from(vec!["2013-07-01"; 3]));
        let out = eval_binary(BinaryOp::Ge, &dates, &lit).unwrap();
        let got = out.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(got.value(0), false); // 06-30 >= 07-01 → false
        assert_eq!(got.value(1), true); // 07-01 >= 07-01 → true
        assert_eq!(got.value(2), true); // 07-02 >= 07-01 → true
    }
}

#[cfg(test)]
mod dict_path_tests {
    use super::*;
    use crate::Literal;
    use arrow::array::{DictionaryArray, Int32Array, Int64Array, StringArray};
    use arrow::datatypes::{Field, Int32Type, Schema};

    fn batch(name: &str, col: ArrayRef) -> RecordBatch {
        let schema = Schema::new(vec![Field::new(name, col.data_type().clone(), true)]);
        RecordBatch::try_new(Arc::new(schema), vec![col]).unwrap()
    }

    /// The decoded column — what the engine compared before the dict path existed, and the
    /// oracle this path must reproduce exactly.
    fn decoded(dict: &ArrayRef) -> ArrayRef {
        let DataType::Dictionary(_, value) = dict.data_type() else {
            unreachable!("test passes a dictionary")
        };
        arrow::compute::cast(dict, value).unwrap()
    }

    const CMPS: [BinaryOp; 6] = [
        BinaryOp::Eq,
        BinaryOp::Ne,
        BinaryOp::Lt,
        BinaryOp::Le,
        BinaryOp::Gt,
        BinaryOp::Ge,
    ];

    /// The dict fast path MUST equal the decoded path bit-for-bit — every comparison operator,
    /// both operand orders, null keys *and* null dictionary values.
    ///
    /// This is the whole licence for the optimization: `Expr::eval` is the correctness oracle,
    /// so a faster leaf is only admissible if it is indistinguishable from the slow one. The
    /// identity it leans on is `cmp(take(values, keys), lit) == take(cmp(values, lit), keys)`.
    fn assert_dict_equals_decoded(cname: &str, dict: ArrayRef, lits: &[Literal]) {
        let dict_batch = batch(cname, dict.clone());
        let plain_batch = batch(cname, decoded(&dict));

        for lit in lits {
            for &op in &CMPS {
                for col_on_left in [true, false] {
                    let c = Expr::Col {
                        name: cname.to_string(),
                    };
                    let l = Expr::Lit { value: lit.clone() };
                    let (lhs, rhs) = if col_on_left { (c, l) } else { (l, c) };

                    // The fast path must actually be taken — otherwise this test proves nothing.
                    let fast = try_dict_compare(op, &lhs, &rhs, &dict_batch)
                        .unwrap()
                        .expect("dict fast path must engage for <dict col> cmp <literal>");
                    // ...and the same expression over the *decoded* column takes the array path.
                    let slow = Expr::Binary {
                        op,
                        left: Box::new(lhs.clone()),
                        right: Box::new(rhs.clone()),
                    }
                    .eval(&plain_batch)
                    .unwrap();

                    assert_eq!(
                        fast.as_ref(),
                        slow.as_ref(),
                        "dict path diverged from the decoded oracle: op={op:?} lit={lit:?} \
                         col_on_left={col_on_left} col={cname}"
                    );
                }
            }
        }
    }

    #[test]
    fn string_dict_compare_equals_the_decoded_oracle() {
        // Null *keys* (row 1) — the ordinary "this row is null" case.
        let dict: DictionaryArray<Int32Type> =
            vec![Some("b"), None, Some("a"), Some("c"), Some("b")]
                .into_iter()
                .collect();
        assert_dict_equals_decoded(
            "s",
            Arc::new(dict),
            &[
                Literal::Str("b".into()),
                Literal::Str("a".into()),
                Literal::Str("zzz".into()), // sorts after every entry
                Literal::Str("".into()),    // sorts before every entry
            ],
        );
    }

    #[test]
    fn a_null_inside_the_dictionary_values_still_matches_the_oracle() {
        // The subtler null: the *value* at key 1 is null, and three rows point at it. Decoding
        // gathers a null into each of those rows; the dict path instead compares null once and
        // gathers the resulting null bit. Both must yield null — SQL's `NULL <op> x = NULL`.
        let values = Arc::new(StringArray::from(vec![Some("a"), None, Some("c")]));
        let keys = Int32Array::from(vec![Some(0), Some(1), Some(2), Some(1), None]);
        let dict: ArrayRef = Arc::new(DictionaryArray::<Int32Type>::try_new(keys, values).unwrap());
        assert_dict_equals_decoded(
            "s",
            dict,
            &[Literal::Str("a".into()), Literal::Str("c".into())],
        );
    }

    #[test]
    fn numeric_dict_compare_equals_the_decoded_oracle() {
        // A dictionary need not be strings — and a numeric one would otherwise fall to
        // `try_scalar_binary`, which decodes the whole column first.
        let values = Arc::new(Int64Array::from(vec![Some(10), Some(-3), None, Some(0)]));
        let keys = Int32Array::from(vec![Some(0), Some(1), Some(2), Some(3), None, Some(0)]);
        let dict: ArrayRef = Arc::new(DictionaryArray::<Int32Type>::try_new(keys, values).unwrap());
        assert_dict_equals_decoded(
            "n",
            dict,
            &[
                Literal::Int(0),
                Literal::Int(10),
                Literal::Int(-3),
                Literal::Int(999),
            ],
        );
    }

    /// The path declines anything it cannot prove equivalent, rather than guessing.
    #[test]
    fn the_fast_path_declines_shapes_it_does_not_own() {
        let dict: DictionaryArray<Int32Type> = vec![Some("a"), Some("b")].into_iter().collect();
        let b = batch("s", Arc::new(dict));
        let col = Expr::Col { name: "s".into() };
        let lit = Expr::Lit {
            value: Literal::Str("a".into()),
        };

        // Not a comparison operator.
        assert!(try_dict_compare(BinaryOp::Add, &col, &lit, &b)
            .unwrap()
            .is_none());
        // Column vs column — no literal to fold over the dictionary values.
        assert!(try_dict_compare(BinaryOp::Eq, &col, &col, &b)
            .unwrap()
            .is_none());
        // A non-dictionary column falls through to the ordinary array path.
        let plain = batch("s", Arc::new(StringArray::from(vec!["a", "b"])));
        assert!(try_dict_compare(BinaryOp::Eq, &col, &lit, &plain)
            .unwrap()
            .is_none());
    }
}
