//! Binary-operator evaluation for `Expr::Binary` plus the shared numeric/boolean
//! coercion helpers (split out of `lib.rs`).

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, Datum, Int64Array, RecordBatch, Scalar,
};
use arrow::compute::cast;
use arrow::compute::kernels::{boolean, cmp, numeric};
use arrow::datatypes::DataType;
use bc_arrow::canon_float_array;

use crate::eval::temporal::date::add_months;
use crate::{BinaryOp, Expr, ExprError, Literal};

/// Whether comparing this type's values raw would disagree with the engine's float identity.
fn is_float_dtype(dt: &DataType) -> bool {
    matches!(dt, DataType::Float32 | DataType::Float64)
}

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
    // Exactly one operand a literal; the other is the column.
    let is_lit = |e: &Expr| matches!(e, Expr::Lit { .. });
    let (arr_expr, lit_expr, lit_on_right) = match (left, right) {
        (a, l) if is_lit(l) => (a, l, true),
        (l, a) if is_lit(l) => (a, l, false),
        _ => return Ok(None),
    };
    let Expr::Lit { value: lit } = lit_expr else {
        return Ok(None);
    };
    // A non-numeric literal has no arithmetic arm here (`Utf8 || Utf8` is `Concat`, which
    // keeps the array path), so reject that combination before evaluating the column
    // rather than after.
    let numeric_lit = matches!(lit, Literal::Int(_) | Literal::Float(_));
    let is_cmp = matches!(op, Eq | Ne | Lt | Le | Gt | Ge);
    if !numeric_lit && !is_cmp {
        return Ok(None);
    }

    let arr = arr_expr.eval(batch)?;
    let lit_arr = lit.to_array(1);
    let (arr, lit_arr) = if numeric_lit {
        // Only Int64/Float64 columns broadcast here; defer decimals to the array path's
        // coercion, which handles their wider promotion rules.
        if !matches!(arr.data_type(), Int64 | Float64) {
            return Ok(None);
        }
        // Mirror `coerce_numeric`: a mixed Int/Float pair promotes to Float64.
        match (arr.data_type(), lit_arr.data_type()) {
            (Int64, Float64) => (cast(&arr, &Float64)?, lit_arr),
            (Float64, Int64) => (arr, cast(&lit_arr, &Float64)?),
            _ => (arr, lit_arr),
        }
    } else {
        // A `Utf8`, `Boolean`, `Date32` or naive-`Timestamp` column against its own
        // literal type — `o_orderpriority = '1-URGENT'`, `l_shipdate < DATE '1995-03-15'`.
        // These are the most common predicates in the benchmark suite and they were paying
        // a full N-row materialization of the literal: `Literal::to_array(n)` builds a
        // `StringArray` of *n* copies of the needle, offsets and bytes, once per morsel per
        // evaluation. A one-element `Scalar` broadcasts instead.
        //
        // Served on an **exact** type match, because that is what makes it bit-identical
        // rather than merely equivalent. The array path applies three adjustments before
        // comparing, and an exact match makes each provably the identity:
        // `align_date_timestamp_for_cmp` only fires on a Date-vs-Timestamp pair,
        // `align_decimals_for_cmp` only on two decimals of differing scale, and
        // `canon_float_array` is an `Arc::clone` for anything that is not a float. So a
        // mixed pair (`Utf8` vs `LargeUtf8`, a tz-aware timestamp against a naive literal,
        // an `Int64` column against a `Date` literal) is declined here and keeps the array
        // path's coercion, unchanged.
        //
        // The one mixed pair served here is a **string literal against a temporal column**
        // (`EventDate >= '2013-07-01'`), which `coerce_numeric` handles by casting the
        // literal to the column's temporal type. That cast is elementwise, so casting the
        // one-element literal and broadcasting is the same value as casting the
        // materialized N-row copy — including the error, since an unparseable string fails
        // identically on one element or N. What it saves is real: the array path re-parses
        // the *same* ISO string once per row, which showed up as 6.3% of ClickBench q39 in
        // `arrow_cast::parse::parse_date`, and a bare date range over `hits` measured
        // 24.7 ms against 7.5 ms for the already-typed `DATE '...'` spelling.
        let temporal_vs_string = matches!(
            (arr.data_type(), lit_arr.data_type()),
            (
                DataType::Date32 | DataType::Date64 | DataType::Timestamp(..),
                DataType::Utf8 | DataType::LargeUtf8
            )
        );
        if temporal_vs_string {
            let typed = cast(&lit_arr, arr.data_type())?;
            (arr, typed)
        } else if arr.data_type() != lit_arr.data_type() {
            return Ok(None);
        } else {
            (arr, lit_arr)
        }
    };

    // Canonicalize both float operands for the comparison arms, exactly as the array path
    // does — this path must be bit-identical to it (see `canon_floats_for_cmp`). Arithmetic
    // is untouched: `-0.0 + x` and NaN propagation stay IEEE.
    let (arr, lit_arr) =
        if matches!(op, Eq | Ne | Lt | Le | Gt | Ge) && is_float_dtype(arr.data_type()) {
            (canon_float_array(&arr), canon_float_array(&lit_arr))
        } else {
            (arr, lit_arr)
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
    // DuckDB `DATE - DATE` is the integer count of days between the two dates, not an
    // interval. Arrow's `sub` over two `Date32` (days-since-epoch) arrays produces a
    // `Duration` — neither the type nor the value SQL wants — so subtract the day ordinals
    // directly to `Int64`, preserving nulls.
    if matches!(op, Sub)
        && matches!(l.data_type(), arrow::datatypes::DataType::Date32)
        && matches!(r.data_type(), arrow::datatypes::DataType::Date32)
    {
        return date32_diff_days(&l, &r);
    }
    // `DATE ± <integer days>` shifts the date by that many days (DuckDB `DATE - 5`,
    // `DATE + n`). Arrow's `sub`/`add` reject `Date32 - Int64`, so compute the new day
    // ordinal directly (`int + date` is commutative). Any other combination falls through
    // to the numeric kernels / the existing error.
    {
        use arrow::datatypes::DataType::{Date32, Int64};
        match (op, l.data_type(), r.data_type()) {
            (Add, Date32, Int64) | (Sub, Date32, Int64) => {
                return date32_offset_days(&l, &r, matches!(op, Sub));
            }
            (Add, Int64, Date32) => return date32_offset_days(&r, &l, false),
            _ => {}
        }
    }
    // Comparison kernels (`cmp::eq` …) require *identical* decimal precision AND scale;
    // arithmetic kernels align scales themselves. Two decimal columns of differing scale
    // (`Decimal128(10,1)` vs `Decimal128(10,2)`, e.g. `1.0 = 1.00`) would otherwise raise
    // "Invalid comparison operation" on a reachable data path — DuckDB compares them equal
    // by widening to a common scale. Align here only for the comparison arms, so `*`/`+`
    // keep their own scale-propagation rules untouched.
    let (l, r) = if matches!(op, Eq | Ne | Lt | Le | Gt | Ge) {
        let (l, r) = align_date_timestamp_for_cmp(&l, &r)?;
        let (l, r) = align_decimals_for_cmp(&l, &r)?;
        // Float identity is the engine's, not the raw bits' — see `canon_floats_for_cmp`.
        (canon_float_array(&l), canon_float_array(&r))
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
        // Python/Polars `//`: the quotient rounded toward NEGATIVE INFINITY, which
        // is deliberately NOT DuckDB's truncating integer division (`-7 // 3` is
        // `-3`, not `-2`). Zero divisor → NULL, as for `Div`/`Mod`.
        FloorDiv => floor_div(l, r)?,
        // SQL three-valued logic: `FALSE AND NULL` is FALSE, `TRUE OR NULL` is
        // TRUE (a known-controlling operand wins over an unknown). Arrow's plain
        // `and`/`or` propagate the null instead, so use the Kleene kernels to match
        // DuckDB. With null-free operands these reduce to plain and/or, so the
        // JIT's bitwise band/bor (null-free only) stays bit-for-bit identical.
        And => Arc::new(boolean::and_kleene(as_bool(l, "and")?, as_bool(r, "and")?)?),
        Or => Arc::new(boolean::or_kleene(as_bool(l, "or")?, as_bool(r, "or")?)?),
        // SQL `||`: cast both operands to Utf8 and concatenate element-wise.
        // A null on either side yields a null (matching DuckDB's `||` operator).
        //
        // Two *lists* concatenate as lists, not as their rendered text. `[1,2] || [3]`
        // is `[1,2,3]` in DuckDB, Spark and Polars alike; casting them to Utf8 first
        // produced the string `'[1, 2][3]'` — a wrong answer with no error, which is the
        // worst shape a defect can take. The list kernel is the same one `list_concat`
        // uses, so the operator and the function cannot disagree.
        Concat if matches!(l.data_type(), DataType::List(_) | DataType::LargeList(_)) => {
            return crate::eval::list_ops::list_set::eval_list_set(crate::ListSetOp::Concat, l, r);
        }
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
            use arrow::compute::kernels::bitwise::{bitwise_and, bitwise_or, bitwise_xor};
            let (li, ri) = (cast(l, &DataType::Int64)?, cast(r, &DataType::Int64)?);
            let (la, ra) = (
                li.as_any().downcast_ref::<Int64Array>().unwrap(),
                ri.as_any().downcast_ref::<Int64Array>().unwrap(),
            );
            match op {
                BitAnd => Arc::new(bitwise_and(la, ra)?),
                BitOr => Arc::new(bitwise_or(la, ra)?),
                BitXor => Arc::new(bitwise_xor(la, ra)?),
                // Arrow's `<<` masks the shift amount to its low 6 bits
                // (`wrapping_shl`), so `1 << 64` silently became `1 << 0 = 1` and a
                // *negative* amount wrapped into an in-range shift. Mirror the
                // `ShiftRight` out-of-range convention (an amount outside `0..64`
                // yields 0), so the two directions agree; in range, the wrapping
                // shift matches the engine's wrapping integer arithmetic and the JIT.
                ShiftLeft => Arc::new(arithmetic_shift_left(la, ra)),
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

/// Left shift with an out-of-range amount yielding 0 (arrow's `wrapping_shl` would
/// instead mask the amount to its low 6 bits, so `1 << 64` returned `1` and `x << -1`
/// wrapped to `x << 63`). An amount in `0..64` shifts normally, wrapping the overflowed
/// high bits out — the engine's wrapping integer-arithmetic convention (`add`/`sub`/`mul`
/// and the JIT wrap too). Nulls on either side propagate. This mirrors
/// [`arithmetic_shift_right`]'s out-of-range → 0 rule so the two shift directions agree;
/// DuckDB instead raises on an out-of-range or overflowing left shift, a deliberate
/// difference (the engine wraps rather than errors, per CLAUDE.md #6).
/// `l - r` for two `Date32` (days-since-epoch) columns, as the `Int64` count of days between
/// them (DuckDB `DATE - DATE`), preserving nulls. Arrow's arithmetic kernel would instead
/// yield a `Duration`, so this is special-cased in [`eval_binary`].
fn date32_diff_days(l: &ArrayRef, r: &ArrayRef) -> Result<ArrayRef, ExprError> {
    use arrow::array::Date32Array;
    let a = l
        .as_any()
        .downcast_ref::<Date32Array>()
        .expect("checked Date32 in eval_binary");
    let b = r
        .as_any()
        .downcast_ref::<Date32Array>()
        .expect("checked Date32 in eval_binary");
    let out: Int64Array = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| match (x, y) {
            (Some(x), Some(y)) => Some(i64::from(x) - i64::from(y)),
            _ => None,
        })
        .collect();
    Ok(Arc::new(out))
}

/// `date ± days` for a `Date32` column and an `Int64` day count, producing a `Date32`
/// (DuckDB `DATE + n` / `DATE - n`), preserving nulls. `subtract` chooses the direction.
/// The day ordinal is computed in i64 and truncated to i32 (arrow's `Date32` range).
fn date32_offset_days(
    dates: &ArrayRef,
    days: &ArrayRef,
    subtract: bool,
) -> Result<ArrayRef, ExprError> {
    use arrow::array::Date32Array;
    let d = dates
        .as_any()
        .downcast_ref::<Date32Array>()
        .expect("checked Date32 in eval_binary");
    let n = cast(days, &DataType::Int64)?;
    let n = n
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("cast to Int64");
    let out: Date32Array = (0..d.len())
        .map(|i| {
            if d.is_null(i) || n.is_null(i) {
                return None;
            }
            let base = i64::from(d.value(i));
            let shifted = if subtract {
                base - n.value(i)
            } else {
                base + n.value(i)
            };
            Some(shifted as i32)
        })
        .collect();
    Ok(Arc::new(out))
}

fn arithmetic_shift_left(values: &Int64Array, amounts: &Int64Array) -> Int64Array {
    (0..values.len())
        .map(|i| {
            if values.is_null(i) || amounts.is_null(i) {
                return None;
            }
            let (v, s) = (values.value(i), amounts.value(i));
            Some(if (0..64).contains(&s) { v << s } else { 0 })
        })
        .collect()
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

/// Floored division (`a // b`) — the quotient rounded toward **negative infinity**.
///
/// This is Python/Polars `//`, deliberately *not* SQL/DuckDB integer division, which
/// truncates toward zero: `-7 // 3` is `-3` here and `-2` under `Div`. That direction is
/// the documented contract of `Expr.__floordiv__`, so DuckDB is not the oracle for it.
///
/// Int64 ÷ Int64 is computed **in integers** and stays Int64. That is the whole reason
/// this is an op rather than sugar for `floor(a / b)`: routing through Float64 silently
/// loses precision above 2^53 (`i64::MAX // 3` came back as `3.07e18`) and turns a zero
/// divisor into `inf`/`NaN` instead of NULL.
///
/// Float64 is IEEE `(l / r).floor()`, so `/0.0` still yields ±inf / NaN exactly as `Div`
/// does. Any other numeric type (notably Decimal128) keeps the pre-existing behavior of
/// the old desugaring — computed as Float64 — rather than inventing floored-decimal
/// semantics here.
fn floor_div(l: &ArrayRef, r: &ArrayRef) -> Result<ArrayRef, ExprError> {
    use arrow::array::Float64Array;
    use arrow::datatypes::DataType::{Float64, Int64};

    match (l.data_type(), r.data_type()) {
        (Int64, Int64) => {
            let (a, b) = (
                l.as_any().downcast_ref::<Int64Array>().expect("int64"),
                r.as_any().downcast_ref::<Int64Array>().expect("int64"),
            );
            let out: Int64Array = (0..a.len())
                .map(|i| {
                    if a.is_null(i) || b.is_null(i) {
                        return None;
                    }
                    let (x, y) = (a.value(i), b.value(i));
                    // A zero divisor is NULL (never a CPU trap), matching `Div`/`Mod`.
                    if y == 0 {
                        return None;
                    }
                    // `wrapping_div`/`wrapping_rem` so the single overflowing input
                    // `i64::MIN / -1` wraps to `i64::MIN` instead of trapping; the
                    // remainder is then 0, so no correction applies and the result
                    // agrees with the wrapping `Add`/`Sub`/`Mul` convention this
                    // engine already uses (see the note on the arithmetic arms).
                    let q = x.wrapping_div(y);
                    let rem = x.wrapping_rem(y);
                    // Truncating quotient → floored quotient: when the remainder is
                    // non-zero and its sign differs from the divisor's, truncation
                    // rounded toward zero, i.e. one step UP from the floor.
                    Some(if rem != 0 && ((rem < 0) != (y < 0)) {
                        q.wrapping_sub(1)
                    } else {
                        q
                    })
                })
                .collect();
            Ok(Arc::new(out))
        }
        (Float64, Float64) => {
            let (a, b) = (
                l.as_any().downcast_ref::<Float64Array>().expect("float64"),
                r.as_any().downcast_ref::<Float64Array>().expect("float64"),
            );
            let out: Float64Array = (0..a.len())
                .map(|i| {
                    if a.is_null(i) || b.is_null(i) {
                        return None;
                    }
                    Some((a.value(i) / b.value(i)).floor())
                })
                .collect();
            Ok(Arc::new(out))
        }
        // Decimal128 and any other numeric pair: preserve what `floor(a / b)` produced
        // before this op existed by evaluating in Float64.
        _ => {
            let (a, b) = (cast(l, &Float64)?, cast(r, &Float64)?);
            floor_div(&a, &b)
        }
    }
}

/// Widen two decimal operands to a common precision/scale so a comparison kernel (which
/// demands identical decimal types) can run — DuckDB compares `DECIMAL(10,1)` against
/// `DECIMAL(10,2)` by casting both to a common `DECIMAL`. The common scale is
/// `max(s1, s2)`; the common precision covers the larger integer part plus that scale,
/// capped at Decimal128's 38 digits. Non-decimal or already-identical operands (and any
/// pair that isn't two `Decimal128`s — e.g. Decimal256 or a mixed width) pass through
/// unchanged, deferring to the existing path.
/// A `DATE` compared against a `TIMESTAMP` column: widen the date to midnight.
///
/// `ts = DATE '1995-01-02'` is a query DuckDB answers — it casts the DATE up to TIMESTAMP at
/// 00:00:00 and compares instants — and arrow's comparison kernels reject outright, raising
/// "Invalid comparison operation: Timestamp(Microsecond, None) == Date32". That gap is not
/// hypothetical: the fold rule that builds `InList` is a predicate-*shape* rewrite with no
/// access to the schema, so `ts IN (DATE …, DATE …)` reaches the engine as exactly this pair,
/// and `tests/differential/test_diff_in_list.py` pins it against DuckDB.
///
/// Casting the *date* up (rather than truncating the timestamp down to a date) is what makes
/// `ts = DATE '1995-01-02'` false for a timestamp at 12:00 on that day, which is DuckDB's
/// answer and SQL's. Widening never loses information, so no comparison can flip.
///
/// A tz-aware timestamp is handled by casting the date to the *naive* type and letting the
/// zone-stripping arm of `coerce_numeric` line the two timestamps up: the stored values are
/// UTC instants either way, and dropping the zone needs no timezone database (casting *to* a
/// named zone would fail on an arrow build without `chrono-tz`).
fn align_date_timestamp_for_cmp(
    l: &ArrayRef,
    r: &ArrayRef,
) -> Result<(ArrayRef, ArrayRef), ExprError> {
    use DataType::{Date32, Date64, Timestamp};
    let naive = |unit: &arrow::datatypes::TimeUnit| Timestamp(*unit, None);
    match (l.data_type(), r.data_type()) {
        (Timestamp(unit, tz), Date32 | Date64) => {
            let target = naive(unit);
            let left = if tz.is_some() {
                cast(l, &target)?
            } else {
                l.clone()
            };
            Ok((left, cast(r, &target)?))
        }
        (Date32 | Date64, Timestamp(unit, tz)) => {
            let target = naive(unit);
            let right = if tz.is_some() {
                cast(r, &target)?
            } else {
                r.clone()
            };
            Ok((cast(l, &target)?, right))
        }
        _ => Ok((l.clone(), r.clone())),
    }
}

fn align_decimals_for_cmp(l: &ArrayRef, r: &ArrayRef) -> Result<(ArrayRef, ArrayRef), ExprError> {
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
        // Two timestamps that differ in timezone (one tz-aware, one naive) — the shape
        // `tz_aware_col <op> naive_literal` produces: a Delta/event-time column is
        // `Timestamp(us, Some("UTC"))` while a bare `lit(datetime)` is `Timestamp(us, None)`.
        // The comparison kernels demand identical types, so this raised "Invalid comparison
        // operation: Timestamp(Microsecond, Some(...)) > Timestamp(Microsecond, None)" — a hard
        // crash on a common query. A tz-aware timestamp's stored values are UTC instants, so we
        // **strip the zone** (cast the aware side to the naive side's `Timestamp(unit, None)`) and
        // compare the raw instants: the naive literal is thereby read as that same UTC instant —
        // exactly how DuckDB compares a naive literal against a `TIMESTAMPTZ` in its (UTC) session
        // zone. Stripping the zone is a metadata drop that needs no timezone database, so it works
        // for a *named* zone (`"UTC"`) too — casting *to* a named zone would fail on an arrow build
        // without the `chrono-tz` feature. Casting to the naive side's type unifies the unit as well.
        (Timestamp(_, Some(_)), Timestamp(_, None)) => Ok((cast(l, r.data_type())?, r.clone())),
        (Timestamp(_, None), Timestamp(_, Some(_))) => Ok((l.clone(), cast(r, l.data_type())?)),
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
        // Two string columns of differing offset width (`Utf8` vs `LargeUtf8`), or two
        // binary columns of differing width (`Binary` vs `LargeBinary`). The comparison
        // kernels demand *identical* types, so a bare `largeutf8_col = 'x'` (the literal
        // is `Utf8`) raised "Invalid comparison operation" on a reachable data path, where
        // DuckDB treats both as one VARCHAR / BLOB domain and compares them. Widen the
        // narrower side to the wider one (`i32 → i64` offsets — always lossless), matching
        // DuckDB. Arithmetic never reaches here (strings/binaries don't arithmetic-coerce),
        // and `||` casts both to `Utf8` itself, so this only affects the comparison arms.
        (Utf8, LargeUtf8) => Ok((cast(l, &LargeUtf8)?, r.clone())),
        (LargeUtf8, Utf8) => Ok((l.clone(), cast(r, &LargeUtf8)?)),
        (Binary, LargeBinary) => Ok((cast(l, &LargeBinary)?, r.clone())),
        (LargeBinary, Binary) => Ok((l.clone(), cast(r, &LargeBinary)?)),
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

    /// Left shift by a negative or ≥ 64 amount yields 0, not the masked value arrow's
    /// `wrapping_shl` produced (`1 << 64` gave `1`, `1 << -1` gave `1 << 63`). In range,
    /// the shift wraps the overflowed high bits out (the engine's wrapping-arithmetic
    /// convention), consistent with `ShiftRight`'s out-of-range → 0 rule.
    #[test]
    fn left_shift_out_of_range_is_zero() {
        let v = i64arr(vec![Some(1), Some(1), Some(1), Some(1), Some(5), Some(3)]);
        let s = i64arr(vec![Some(64), Some(-1), Some(0), Some(3), Some(100), None]);
        assert_eq!(
            as_i64(&eval_binary(BinaryOp::ShiftLeft, &v, &s).unwrap()),
            // 64 → 0, -1 → 0, 0 → 1, 3 → 8, 100 → 0, null → null
            vec![Some(0), Some(0), Some(1), Some(8), Some(0), None]
        );
        // In-range shift that overflows the sign bit wraps (does not error), matching the
        // engine's wrapping integer arithmetic.
        assert_eq!(
            as_i64(
                &eval_binary(
                    BinaryOp::ShiftLeft,
                    &i64arr(vec![Some(1)]),
                    &i64arr(vec![Some(63)])
                )
                .unwrap()
            ),
            vec![Some(1i64 << 63)] // i64::MIN
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

    /// The same bit-for-bit obligation for the *non-numeric* literals, which broadcast as
    /// a `Scalar` instead of materializing N copies of the value.
    ///
    /// `o_orderpriority = '1-URGENT'` and `l_shipdate < DATE '1995-03-15'` are the two most
    /// common predicate shapes in the benchmark suite, so this path is hot; and a string
    /// literal is exactly where a broadcast is worth most, because materializing it copies
    /// offsets *and* bytes. Nulls in the column are included because a scalar comparison
    /// must null the same rows an array comparison does.
    #[test]
    fn scalar_path_equals_array_path_for_non_numeric_literals() {
        use arrow::array::{BooleanArray, Date32Array, StringArray, TimestampMicrosecondArray};

        let str_col: ArrayRef = Arc::new(StringArray::from(vec![
            Some("1-URGENT"),
            None,
            Some("5-LOW"),
            Some(""),
        ]));
        let bool_col: ArrayRef = Arc::new(BooleanArray::from(vec![
            Some(true),
            None,
            Some(false),
            Some(true),
        ]));
        let date_col: ArrayRef = Arc::new(Date32Array::from(vec![
            Some(9_131),
            None,
            Some(0),
            Some(-1),
        ]));
        let ts_col: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![
            Some(1_000),
            None,
            Some(0),
            Some(-5),
        ]));

        let cases: [(&str, &ArrayRef, Vec<Literal>); 4] = [
            (
                "s",
                &str_col,
                vec![
                    Literal::Str("1-URGENT".into()),
                    Literal::Str("".into()),
                    Literal::Str("zzz".into()),
                ],
            ),
            (
                "b",
                &bool_col,
                vec![Literal::Bool(true), Literal::Bool(false)],
            ),
            (
                "d",
                &date_col,
                vec![Literal::Date(9_131), Literal::Date(0), Literal::Date(-1)],
            ),
            (
                "t",
                &ts_col,
                vec![Literal::Timestamp(1_000), Literal::Timestamp(0)],
            ),
        ];
        let ops = [
            BinaryOp::Eq,
            BinaryOp::Ne,
            BinaryOp::Lt,
            BinaryOp::Le,
            BinaryOp::Gt,
            BinaryOp::Ge,
        ];

        for (cname, col, lits) in cases {
            let b = batch(cname, (*col).clone());
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
                            .expect("the scalar fast path must engage on an exact type match");
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

    /// A type *mismatch* must be declined, not served, because the array path's coercion is
    /// what gives it its meaning. These are the three shapes that would silently change
    /// behaviour if the exact-match gate were loosened to "both are strings" or "both are
    /// temporal": a `LargeUtf8` column, a tz-aware timestamp against a naive literal, and
    /// an integer column against a date literal.
    #[test]
    fn a_mismatched_literal_type_is_declined() {
        use arrow::array::{Int64Array, LargeStringArray, TimestampMicrosecondArray};
        use arrow::datatypes::TimeUnit;

        let large: ArrayRef = Arc::new(LargeStringArray::from(vec![Some("a"), None]));
        let tz: ArrayRef = Arc::new(
            TimestampMicrosecondArray::from(vec![Some(1), None]).with_timezone("UTC".to_string()),
        );
        let ints: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None]));
        assert_eq!(
            tz.data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
        );

        for (cname, col, lit) in [
            ("s", &large, Literal::Str("a".into())),
            ("t", &tz, Literal::Timestamp(1)),
            ("i", &ints, Literal::Date(1)),
        ] {
            let b = batch(cname, (*col).clone());
            let out = try_scalar_binary(
                BinaryOp::Eq,
                &Expr::Col {
                    name: cname.to_string(),
                },
                &Expr::Lit { value: lit.clone() },
                &b,
            )
            .unwrap();
            assert!(
                out.is_none(),
                "a {} column against {lit:?} must keep the array path",
                col.data_type()
            );
        }
    }

    /// A non-numeric literal has no arithmetic arm, so `Add`/`Sub`/`Mul` must decline
    /// rather than reach a kernel that would reject the pair with a different error than
    /// the array path raises.
    #[test]
    fn arithmetic_with_a_non_numeric_literal_is_declined() {
        use arrow::array::StringArray;
        let col: ArrayRef = Arc::new(StringArray::from(vec![Some("a")]));
        let b = batch("s", col);
        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul] {
            let out = try_scalar_binary(
                op,
                &Expr::Col { name: "s".into() },
                &Expr::Lit {
                    value: Literal::Str("a".into()),
                },
                &b,
            )
            .unwrap();
            assert!(out.is_none(), "{op:?} on a string literal must be declined");
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
        assert!(got.value(0));
        assert!(!got.value(1));
        assert!(got.is_null(2));
    }

    /// Two string columns of different offset width (`LargeUtf8` vs `Utf8`) — the shape a
    /// `largeutf8_col = 'lit'` produces, since a SQL string literal is `Utf8` — must compare
    /// by widening to a common type, not raise "Invalid comparison operation". Same for two
    /// binary columns of different width (`Binary` vs `LargeBinary`). DuckDB compares both.
    #[test]
    fn mixed_width_string_and_binary_columns_compare() {
        use arrow::array::{
            BinaryArray, BooleanArray, LargeBinaryArray, LargeStringArray, StringArray,
        };
        // LargeUtf8 column vs Utf8 literal, both operand orders.
        let lutf8: ArrayRef = Arc::new(LargeStringArray::from(vec![Some("a"), Some("b"), None]));
        let utf8: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), Some("a"), Some("a")]));
        for (l, r) in [(&lutf8, &utf8), (&utf8, &lutf8)] {
            let out = eval_binary(BinaryOp::Eq, l, r).expect("mixed-width string compares");
            let got = out.as_any().downcast_ref::<BooleanArray>().unwrap();
            assert!(got.value(0)); // "a" == "a"
            assert!(!got.value(1)); // "b" != "a" (or "a" != "b")
            assert!(got.is_null(2)); // NULL compares null
        }
        // Binary vs LargeBinary, both operand orders.
        let bin: ArrayRef = Arc::new(BinaryArray::from_opt_vec(vec![
            Some(b"a".as_ref()),
            Some(b"b"),
            None,
        ]));
        let lbin: ArrayRef = Arc::new(LargeBinaryArray::from_opt_vec(vec![
            Some(b"a".as_ref()),
            Some(b"a"),
            Some(b"a"),
        ]));
        for (l, r) in [(&bin, &lbin), (&lbin, &bin)] {
            let out = eval_binary(BinaryOp::Ne, l, r).expect("mixed-width binary compares");
            let got = out.as_any().downcast_ref::<BooleanArray>().unwrap();
            assert!(!got.value(0)); // "a" == "a" → Ne false
            assert!(got.value(1)); // "b" != "a" → Ne true
            assert!(got.is_null(2)); // NULL → null
        }
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
        assert!(!got.value(0)); // 06-30 >= 07-01 → false
        assert!(got.value(1)); // 07-01 >= 07-01 → true
        assert!(got.value(2)); // 07-02 >= 07-01 → true
    }

    /// `DATE - DATE` is the integer count of days between them (DuckDB), not an interval.
    #[test]
    fn date_minus_date_is_int64_day_count() {
        use arrow::array::{Date32Array, Int64Array};
        // 2023-05-15 is day 19492; 2023-05-10 is 19487 → difference 5. Nulls propagate.
        let a: ArrayRef = Arc::new(Date32Array::from(vec![Some(19492), Some(19487), None]));
        let b: ArrayRef = Arc::new(Date32Array::from(vec![Some(19487), Some(19492), Some(1)]));
        let out = eval_binary(BinaryOp::Sub, &a, &b).unwrap();
        let got = out
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64 day count");
        assert_eq!(got.value(0), 5);
        assert_eq!(got.value(1), -5);
        assert!(got.is_null(2));
    }

    /// `DATE ± <integer days>` shifts the date (DuckDB `DATE - 5`), staying a `Date32`.
    #[test]
    fn date_plus_minus_int_shifts_by_days() {
        use arrow::array::{Date32Array, Int64Array};
        let d: ArrayRef = Arc::new(Date32Array::from(vec![Some(19492), None]));
        let n: ArrayRef = Arc::new(Int64Array::from(vec![Some(5), Some(3)]));
        // date - 5 → 19487; null propagates.
        let sub = eval_binary(BinaryOp::Sub, &d, &n).unwrap();
        let sub = sub.as_any().downcast_ref::<Date32Array>().expect("Date32");
        assert_eq!(sub.value(0), 19487);
        assert!(sub.is_null(1));
        // date + 5 → 19497.
        let add = eval_binary(BinaryOp::Add, &d, &n).unwrap();
        assert_eq!(
            add.as_any().downcast_ref::<Date32Array>().unwrap().value(0),
            19497
        );
        // int + date is commutative → Date32.
        let radd = eval_binary(BinaryOp::Add, &n, &d).unwrap();
        assert_eq!(
            radd.as_any()
                .downcast_ref::<Date32Array>()
                .unwrap()
                .value(0),
            19497
        );
    }

    /// A tz-aware `Timestamp` column compared to a tz-naive `Timestamp` literal (the shape
    /// `utc_col > lit(naive_datetime)` produces) must compare the instants directly — no crash,
    /// and no value shift: the naive literal is read as the same UTC instant, matching DuckDB.
    #[test]
    fn tz_aware_timestamp_column_vs_naive_literal_compares() {
        use arrow::array::{BooleanArray, TimestampMicrosecondArray};
        let us = 1_000_000i64;
        let (jan1, mar1, jun1) = (1_609_459_200 * us, 1_614_556_800 * us, 1_622_505_600 * us);
        // UTC-aware column; naive literal at 2021-03-01.
        let aware: ArrayRef =
            Arc::new(TimestampMicrosecondArray::from(vec![jan1, jun1]).with_timezone("+00:00"));
        let naive: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![mar1, mar1]));
        let out = eval_binary(BinaryOp::Gt, &aware, &naive).unwrap();
        let got = out.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(!got.value(0), "2021-01-01 > 2021-03-01 must be false");
        assert!(got.value(1), "2021-06-01 > 2021-03-01 must be true");
        // Reversed operand order coerces the same way (naive on the left).
        let out2 = eval_binary(BinaryOp::Lt, &naive, &aware).unwrap();
        let got2 = out2.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(!got2.value(0)); // 2021-03-01 < 2021-01-01 → false
        assert!(got2.value(1)); // 2021-03-01 < 2021-06-01 → true
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

/// Floored division (`//`) — Python/Polars semantics, NOT DuckDB's truncating
/// integer division. These pin the two properties that motivated making it a real
/// op instead of desugaring to `floor(a / b)`: Int64 stays Int64 (exact past 2^53)
/// and a zero divisor is NULL rather than `inf`/`NaN`.
#[cfg(test)]
mod floor_div_tests {
    use super::*;
    use arrow::array::Float64Array;

    fn fd_i64(l: Vec<Option<i64>>, r: Vec<Option<i64>>) -> Vec<Option<i64>> {
        let (l, r): (ArrayRef, ArrayRef) =
            (Arc::new(Int64Array::from(l)), Arc::new(Int64Array::from(r)));
        let out = eval_binary(BinaryOp::FloorDiv, &l, &r).expect("floor_div");
        assert_eq!(
            out.data_type(),
            &DataType::Int64,
            "Int64 // Int64 must stay Int64 — a Float64 result is the precision bug"
        );
        let a = out.as_any().downcast_ref::<Int64Array>().expect("i64");
        (0..a.len())
            .map(|i| (!a.is_null(i)).then(|| a.value(i)))
            .collect()
    }

    fn fd_f64(l: Vec<Option<f64>>, r: Vec<Option<f64>>) -> Vec<Option<f64>> {
        let (l, r): (ArrayRef, ArrayRef) = (
            Arc::new(Float64Array::from(l)),
            Arc::new(Float64Array::from(r)),
        );
        let out = eval_binary(BinaryOp::FloorDiv, &l, &r).expect("floor_div");
        assert_eq!(out.data_type(), &DataType::Float64);
        let a = out.as_any().downcast_ref::<Float64Array>().expect("f64");
        (0..a.len())
            .map(|i| (!a.is_null(i)).then(|| a.value(i)))
            .collect()
    }

    /// All four sign combinations round toward NEGATIVE INFINITY, matching Python's
    /// `//`. Truncating (DuckDB `/`) would give -2 and -2 for the mixed-sign rows.
    #[test]
    fn rounds_toward_negative_infinity_in_every_sign_combination() {
        assert_eq!(
            fd_i64(
                vec![Some(7), Some(-7), Some(7), Some(-7)],
                vec![Some(3), Some(3), Some(-3), Some(-3)],
            ),
            // Python: 7//3=2, -7//3=-3, 7//-3=-3, -7//-3=2
            vec![Some(2), Some(-3), Some(-3), Some(2)],
        );
    }

    /// Exact division needs no correction in any sign combination.
    #[test]
    fn exact_division_is_unadjusted() {
        assert_eq!(
            fd_i64(
                vec![Some(6), Some(-6), Some(6), Some(-6), Some(0)],
                vec![Some(3), Some(3), Some(-3), Some(-3), Some(5)],
            ),
            vec![Some(2), Some(-2), Some(-2), Some(2), Some(0)],
        );
    }

    /// The regression this op exists for: past 2^53 a Float64 round-trip loses
    /// precision. `i64::MAX // 3` came back as `3.0744573456182584e18` (and
    /// `(2^53+1) // 3` was off by one) when `//` desugared to `floor(a / b)`.
    #[test]
    fn exact_at_i64_and_f64_mantissa_boundaries() {
        assert_eq!(
            fd_i64(
                vec![Some(i64::MAX), Some(9007199254740993), Some(i64::MIN)],
                vec![Some(3), Some(3), Some(3)],
            ),
            vec![
                Some(3074457345618258602),
                Some(3002399751580331),
                // i64::MIN/3 = -3074457345618258602.67 → floor = ...603
                Some(-3074457345618258603),
            ],
        );
    }

    /// `i64::MIN / -1` is the one input whose true quotient is unrepresentable.
    /// It WRAPS to `i64::MIN` rather than trapping, matching the wrapping
    /// convention the `Add`/`Sub`/`Mul` arms already use. The remainder is 0, so
    /// the floor correction does not fire and cannot turn it into a second wrap.
    #[test]
    fn i64_min_over_negative_one_wraps() {
        assert_eq!(
            fd_i64(vec![Some(i64::MIN)], vec![Some(-1)]),
            vec![Some(i64::MIN)]
        );
    }

    /// A zero divisor is NULL — never a CPU trap, and never `inf`/`NaN`. The old
    /// float desugaring returned `inf`; both DuckDB and Polars return NULL.
    #[test]
    fn zero_divisor_is_null() {
        assert_eq!(
            fd_i64(
                vec![Some(7), Some(-7), Some(0)],
                vec![Some(0), Some(0), Some(0)]
            ),
            vec![None, None, None],
        );
    }

    /// Nulls propagate from either side, exactly as for the other arithmetic ops.
    #[test]
    fn nulls_propagate_from_either_side() {
        assert_eq!(
            fd_i64(
                vec![None, Some(7), None, Some(9)],
                vec![Some(3), None, None, Some(2)],
            ),
            vec![None, None, None, Some(4)],
        );
    }

    /// Float64 is IEEE `(l / r).floor()`: a zero divisor yields ±inf / NaN just as
    /// `Div` does, rather than the integer arm's NULL.
    #[test]
    fn float_is_ieee_floor_of_the_quotient() {
        let out = fd_f64(
            vec![
                Some(7.0),
                Some(-7.0),
                Some(7.5),
                Some(1.0),
                Some(-1.0),
                Some(0.0),
                None,
            ],
            vec![
                Some(2.0),
                Some(2.0),
                Some(2.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(2.0),
            ],
        );
        assert_eq!(out[0], Some(3.0));
        // -3.5 floors to -4, not -3 (truncation would give -3).
        assert_eq!(out[1], Some(-4.0));
        assert_eq!(out[2], Some(3.0));
        assert_eq!(out[3], Some(f64::INFINITY));
        assert_eq!(out[4], Some(f64::NEG_INFINITY));
        assert!(out[5].expect("0.0/0.0 is NaN, not null").is_nan());
        assert_eq!(out[6], None);
    }

    /// A mixed Int64/Float64 pair promotes to Float64 (`coerce_numeric`), matching
    /// every other arithmetic op, so `int_col // 2.0` is a float floor.
    #[test]
    fn mixed_int_and_float_promotes_to_float() {
        let l: ArrayRef = Arc::new(Int64Array::from(vec![Some(-7)]));
        let r: ArrayRef = Arc::new(Float64Array::from(vec![Some(2.0)]));
        let out = eval_binary(BinaryOp::FloorDiv, &l, &r).expect("floor_div");
        assert_eq!(out.data_type(), &DataType::Float64);
        let a = out.as_any().downcast_ref::<Float64Array>().expect("f64");
        assert_eq!(a.value(0), -4.0);
    }

    /// An empty input yields an empty Int64 array, not an error.
    #[test]
    fn empty_input() {
        assert_eq!(fd_i64(vec![], vec![]), Vec::<Option<i64>>::new());
    }

    /// A string literal against a temporal column must give the **same** answer through the
    /// broadcasting scalar path as through the array path that materializes and casts the
    /// literal per row.
    ///
    /// `try_scalar_binary` now serves this mixed pair (it parses the ISO string once instead
    /// of once per row). The two paths are only equivalent because `cast` is elementwise, so
    /// this pins that claim across both operand orders, all six comparison operators, a
    /// Date32 and a Timestamp column, and rows on either side of the literal plus a null.
    #[test]
    fn temporal_column_against_a_string_literal_matches_the_array_path() {
        use crate::{BinaryOp, Expr, Literal};
        use arrow::array::{
            ArrayRef, BooleanArray, Date32Array, RecordBatch, TimestampMicrosecondArray,
        };
        use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
        use std::sync::Arc;

        // 2013-07-01 is day 15887; the timestamp column is the same instant in µs.
        let date: ArrayRef = Arc::new(Date32Array::from(vec![
            Some(15886),
            Some(15887),
            Some(15888),
            None,
        ]));
        let ts: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![
            Some(15886 * 86_400_000_000),
            Some(15887 * 86_400_000_000),
            Some(15888 * 86_400_000_000),
            None,
        ]));
        let cases: [(&str, ArrayRef, DataType, &str); 2] = [
            ("d", date, DataType::Date32, "2013-07-01"),
            (
                "t",
                ts,
                DataType::Timestamp(TimeUnit::Microsecond, None),
                "2013-07-01T00:00:00",
            ),
        ];

        for (name, col, dt, lit) in cases {
            let schema = Arc::new(Schema::new(vec![Field::new(name, dt, true)]));
            let batch = RecordBatch::try_new(schema, vec![col.clone()]).unwrap();
            for op in [
                BinaryOp::Eq,
                BinaryOp::Ne,
                BinaryOp::Lt,
                BinaryOp::Le,
                BinaryOp::Gt,
                BinaryOp::Ge,
            ] {
                for lit_on_right in [true, false] {
                    let c = Expr::Col { name: name.into() };
                    let l = Expr::Lit {
                        value: Literal::Str(lit.into()),
                    };
                    let (left, right) = if lit_on_right {
                        (c.clone(), l.clone())
                    } else {
                        (l.clone(), c.clone())
                    };
                    // The fast path under test.
                    let fast = super::try_scalar_binary(op, &left, &right, &batch)
                        .unwrap()
                        .expect("temporal-vs-string should take the scalar path");
                    // The reference: materialize the literal to N rows, as the array path does.
                    let n = batch.num_rows();
                    let (la, ra) = if lit_on_right {
                        (col.clone(), Literal::Str(lit.into()).to_array(n))
                    } else {
                        (Literal::Str(lit.into()).to_array(n), col.clone())
                    };
                    let slow = super::eval_binary(op, &la, &ra).unwrap();
                    let f = fast.as_any().downcast_ref::<BooleanArray>().unwrap();
                    let s = slow.as_any().downcast_ref::<BooleanArray>().unwrap();
                    assert_eq!(f, s, "col={name} op={op:?} lit_on_right={lit_on_right}");
                }
            }
        }
    }
}
