//! `<integer or temporal column> <cmp> <literal>`, and a two-sided range over one column, a
//! word of mask bits at a time.
//!
//! `arrow_ord::cmp` answers a column-against-scalar comparison through `collect_bool` and a
//! generic `ArrayOrd` operator; at this build's `x86-64-v2` floor that loop did not vectorize.
//! `l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1995-03-31'` over TPC-H's 6M rows spent
//! **59% of its CPU in `apply_op::<&[i32], is_lt>`**, and TPC-H q6 another 33% across the
//! `i32` and `i64` instances of it. [`fill`] packs 64 comparisons into a word with a loop of
//! fixed trip count, which the compiler unrolls into packed compares and a mask extract.
//!
//! A range costs more than its two comparisons when it is evaluated as two conjuncts: the
//! filter's short-circuit path evaluates the first bound at full width, gathers the surviving
//! rows, evaluates the second over them and scatters the mask back. [`try_prim_range`] answers
//! `lo <= x AND x <= hi` in one pass over the column instead — the numeric twin of
//! `string::try_string_range`, which does the same for a sargable `LIKE 'p%'`.
//!
//! Both are **bit-identical** to the generic path, nulls included, and decline everything they
//! cannot answer that way. Integer and temporal comparisons are plain integer order on both
//! paths (two values of one `DataType` share a unit and a timezone, so their raw integers order
//! exactly as the values do). A float range uses the NaN handling `binary::float_scalar_cmp`
//! documents: `!(v < lo)` for a lower bound, so a NaN row answers what the canonicalizing path
//! answers, and a NaN literal is declined. A null row is null in the output exactly where the
//! column is null, as it is from `arrow_ord::cmp` against a non-null scalar; and the Kleene
//! `AND` of two such comparisons is null in the same places, which is what the range returns.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, ArrowPrimitiveType, AsArray, BooleanArray, RecordBatch};
use arrow::buffer::BooleanBuffer;
use arrow::datatypes::{
    DataType, Date32Type, Date64Type, DurationMicrosecondType, DurationMillisecondType,
    DurationNanosecondType, DurationSecondType, Float32Type, Float64Type, Int16Type, Int32Type,
    Int64Type, Int8Type, Time32MillisecondType, Time32SecondType, Time64MicrosecondType,
    Time64NanosecondType, TimeUnit, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType, TimestampSecondType, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
};

use crate::eval::binary::scalar_operands;
use crate::{BinaryOp, Expr, ExprError};

/// One bit per value, `test` applied to each, packed 64 values to a word.
///
/// The obvious `bits |= u64::from(test(v)) << i` loop does not vectorize: each step is a
/// compare, a set, a shift and an OR, ~2 cycles a value, which is what `arrow_ord`'s
/// `collect_bool` costs too. Writing the 64 answers as *bytes* first lets the compiler run the
/// compares as packed SIMD, and eight bytes of 0/1 collapse into eight bits with one multiply:
/// for `w = sum(b_i << 8i)`, the top byte of `w * 0x0102_0408_1020_4080` is `sum(b_i << i)`,
/// because each `b_i` lands on its own bit of that byte and no two terms share one, so nothing
/// carries.
#[inline(always)]
pub(crate) fn fill<T: Copy>(values: &[T], test: impl Fn(T) -> bool) -> BooleanBuffer {
    let n = values.len();
    let mut words = Vec::with_capacity(n.div_ceil(64));
    let (chunks, rest) = values.as_chunks::<64>();
    let mut bytes = [0u8; 64];
    for chunk in chunks {
        for (b, &v) in bytes.iter_mut().zip(chunk) {
            *b = u8::from(test(v));
        }
        words.push(pack(&bytes));
    }
    if !rest.is_empty() {
        bytes = [0u8; 64];
        for (b, &v) in bytes.iter_mut().zip(rest) {
            *b = u8::from(test(v));
        }
        words.push(pack(&bytes));
    }
    BooleanBuffer::new(words.into(), 0, n)
}

/// Sixty-four 0/1 bytes as one word, byte `i` to bit `i` -- see [`fill`].
#[inline(always)]
fn pack(bytes: &[u8; 64]) -> u64 {
    let mut word = 0u64;
    for (k, eight) in bytes.as_chunks::<8>().0.iter().enumerate() {
        let w = u64::from_le_bytes(*eight);
        word |= (w.wrapping_mul(0x0102_0408_1020_4080) >> 56) << (8 * k);
    }
    word
}

/// `row OP lit` for the six comparisons over an ordered integer type.
fn int_cmp<T: Copy + PartialOrd>(values: &[T], lit: T, op: BinaryOp) -> BooleanBuffer {
    use BinaryOp::{Eq, Ge, Gt, Le, Lt, Ne};
    match op {
        Eq => fill(values, |v| v == lit),
        Ne => fill(values, |v| v != lit),
        Lt => fill(values, |v| v < lit),
        Le => fill(values, |v| v <= lit),
        Gt => fill(values, |v| v > lit),
        Ge => fill(values, |v| v >= lit),
        _ => unreachable!("callers pass a comparison"),
    }
}

/// Calls `$body` with `$t` bound to the primitive type of an integer or temporal `$dt`, or
/// evaluates `$none` for any other type. Floats are deliberately absent: their comparison has a
/// NaN rule the integer one does not.
macro_rules! with_int_type {
    ($dt:expr, $t:ident => $body:expr, _ => $none:expr) => {
        match $dt {
            DataType::Int8 => {
                type $t = Int8Type;
                $body
            }
            DataType::Int16 => {
                type $t = Int16Type;
                $body
            }
            DataType::Int32 => {
                type $t = Int32Type;
                $body
            }
            DataType::Int64 => {
                type $t = Int64Type;
                $body
            }
            DataType::UInt8 => {
                type $t = UInt8Type;
                $body
            }
            DataType::UInt16 => {
                type $t = UInt16Type;
                $body
            }
            DataType::UInt32 => {
                type $t = UInt32Type;
                $body
            }
            DataType::UInt64 => {
                type $t = UInt64Type;
                $body
            }
            DataType::Date32 => {
                type $t = Date32Type;
                $body
            }
            DataType::Date64 => {
                type $t = Date64Type;
                $body
            }
            DataType::Time32(TimeUnit::Second) => {
                type $t = Time32SecondType;
                $body
            }
            DataType::Time32(TimeUnit::Millisecond) => {
                type $t = Time32MillisecondType;
                $body
            }
            DataType::Time64(TimeUnit::Microsecond) => {
                type $t = Time64MicrosecondType;
                $body
            }
            DataType::Time64(TimeUnit::Nanosecond) => {
                type $t = Time64NanosecondType;
                $body
            }
            DataType::Timestamp(TimeUnit::Second, _) => {
                type $t = TimestampSecondType;
                $body
            }
            DataType::Timestamp(TimeUnit::Millisecond, _) => {
                type $t = TimestampMillisecondType;
                $body
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                type $t = TimestampMicrosecondType;
                $body
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                type $t = TimestampNanosecondType;
                $body
            }
            DataType::Duration(TimeUnit::Second) => {
                type $t = DurationSecondType;
                $body
            }
            DataType::Duration(TimeUnit::Millisecond) => {
                type $t = DurationMillisecondType;
                $body
            }
            DataType::Duration(TimeUnit::Microsecond) => {
                type $t = DurationMicrosecondType;
                $body
            }
            DataType::Duration(TimeUnit::Nanosecond) => {
                type $t = DurationNanosecondType;
                $body
            }
            _ => $none,
        }
    };
}

/// The comparison of an integer or temporal column with a same-typed, non-null literal, or
/// `None` for any other pair. `op` is read as `row OP literal` (the caller mirrors it for a
/// literal on the left).
pub(crate) fn int_scalar_cmp(op: BinaryOp, arr: &ArrayRef, lit_arr: &ArrayRef) -> Option<ArrayRef> {
    if arr.data_type() != lit_arr.data_type() || lit_arr.is_null(0) {
        return None;
    }
    let bits = with_int_type!(arr.data_type(), T => {
        let lit = lit_arr.as_primitive::<T>().value(0);
        int_cmp(arr.as_primitive::<T>().values(), lit, op)
    }, _ => return None);
    Some(Arc::new(BooleanArray::new(bits, arr.nulls().cloned())))
}

/// `<float column> <cmp> <float scalar>` in one IEEE pass — no canonicalized copy of the
/// column, and no `total_cmp`.
///
/// The engine's float identity folds `-0.0` into `0.0` and all NaNs into one value
/// (`bc_arrow::float_ident`), and every other float comparison here obtains it by rewriting
/// **both operands** and handing them to arrow's `cmp`, which ranks floats by `total_cmp`.
/// That is two full passes over the column — one to look for a `-0.0` or a NaN, plus a
/// second, scalar one to compare — and on a 60 M-row predicate the first alone profiled at
/// 10.9 % of the query while the comparison kernel took another 39 %.
///
/// Neither pass is necessary against a scalar, because the identity is already *implied* by
/// the right IEEE predicate. Two observations do it:
///
/// * **IEEE already folds the zeros.** `-0.0 == 0.0` is true and `-0.0 < x` iff `0.0 < x`
///   for every `x`, so a canonicalizing pass changes no comparison's answer. Only
///   `total_cmp` — which orders the two zeros apart, and which nothing here needs — made it
///   look necessary.
/// * **NaN is the only real difference, and it is a predicate choice.** Canonicalizing sends
///   every NaN to `+qNaN`, which `total_cmp` ranks above `+inf`, so against a non-NaN literal
///   a NaN row answers *false* to `<` and `<=` and *true* to `>` and `>=`. IEEE agrees on the
///   first pair (a NaN comparison is false) and disagrees on the second — which is exactly
///   what the **negated** form fixes: `!(v <= lit)` is true for a NaN and equals `v > lit`
///   for everything else. That is one unordered-or-greater compare, which is a single
///   instruction on every SIMD target, rather than a branch.
///
/// `Eq`/`Ne` need no adjustment: with a non-NaN literal, `canon(v) == canon(lit)` under
/// `total_cmp` holds exactly when `v == lit` under IEEE, NaN answering false to both.
///
/// # Declines
///
/// * A **NaN literal**. It is the one value whose canonical form changes the answer —
///   `canon(NaN) == canon(NaN)` is *true*, where IEEE `NaN == NaN` is false — so it keeps
///   the canonicalizing path rather than being special-cased here.
/// * A null literal, a non-float column, or a column and literal of different float widths.
///   All are the array path's to coerce; this fires only where the two already agree.
///
/// Nulls are carried through unchanged: the output is null exactly where the input is,
/// which is what `arrow_ord::cmp` produces for a null-free scalar operand.
pub(crate) fn float_scalar_cmp(
    op: BinaryOp,
    arr: &ArrayRef,
    lit_arr: &ArrayRef,
    lit_on_right: bool,
) -> Option<ArrayRef> {
    if arr.data_type() != lit_arr.data_type() || lit_arr.is_null(0) {
        return None;
    }
    // A literal on the *left* is the mirrored predicate on the right: `24 > x` is `x < 24`.
    // Mirroring the operator is exact for every arm, including the NaN ones, because it is
    // the same total order read in the other direction.
    let op = if lit_on_right { op } else { mirror_cmp(op) };
    let values = match arr.data_type() {
        DataType::Float64 => {
            let lit = lit_arr.as_primitive::<Float64Type>().value(0);
            if lit.is_nan() {
                return None;
            }
            float_cmp_bits(arr.as_primitive::<Float64Type>().values(), lit, op)
        }
        DataType::Float32 => {
            let lit = lit_arr.as_primitive::<Float32Type>().value(0);
            if lit.is_nan() {
                return None;
            }
            float_cmp_bits(arr.as_primitive::<Float32Type>().values(), lit, op)
        }
        _ => return None,
    };
    Some(Arc::new(BooleanArray::new(values, arr.nulls().cloned())))
}

/// The comparison read from the other side — `a < b` is `b > a`, and so on.
pub(crate) fn mirror_cmp(op: BinaryOp) -> BinaryOp {
    use BinaryOp::{Ge, Gt, Le, Lt};
    match op {
        Lt => Gt,
        Le => Ge,
        Gt => Lt,
        Ge => Le,
        other => other, // Eq / Ne are symmetric
    }
}

/// One IEEE comparison per value, packed 64 bits at a time. See [`float_scalar_cmp`] for why
/// `Gt`/`Ge` are written as negations — that is the whole of the NaN handling.
// `!(a <= b)` on a partially ordered type is exactly what this needs, and it is what the
// lint exists to question: the negation is the NaN handling, not a lazy spelling of `>`.
// `partial_cmp` — the lint's suggestion — would reintroduce the branch on `None` that the
// unordered-or-greater compare replaces with one instruction.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn float_cmp_bits<T: PartialOrd + Copy>(values: &[T], lit: T, op: BinaryOp) -> BooleanBuffer {
    use BinaryOp::{Eq, Ge, Gt, Le, Lt, Ne};
    match op {
        Lt => fill(values, |v| v < lit),
        Le => fill(values, |v| v <= lit),
        Gt => fill(values, |v| !(v <= lit)),
        Ge => fill(values, |v| !(v < lit)),
        Eq => fill(values, |v| v == lit),
        Ne => fill(values, |v| !(v == lit)),
        _ => unreachable!("float_scalar_cmp filters to the comparison arms"),
    }
}

/// A boolean column against a boolean literal, from the column's bits with no per-row work.
///
/// `false < true`, so every comparison with a known literal is the column, its negation, or a
/// constant: `b = true` is `b`, `b < true` is `!b`, `b >= false` is everything. A filter folded
/// into `COUNT` reaches here as `NULLIF(p, false)`, whose `p = false` went through `arrow_ord`
/// one bit at a time -- 19% of `COUNT(*) WHERE l_comment LIKE 'the%'`. Nulls stay null where the
/// column is, as from `arrow_ord::cmp` against a non-null scalar. `op` is read `row OP lit`.
pub(crate) fn bool_scalar_cmp(
    op: BinaryOp,
    arr: &ArrayRef,
    lit_arr: &ArrayRef,
) -> Option<ArrayRef> {
    use BinaryOp::{Eq, Ge, Gt, Le, Lt, Ne};
    if arr.data_type() != &DataType::Boolean
        || lit_arr.data_type() != &DataType::Boolean
        || lit_arr.is_null(0)
    {
        return None;
    }
    let b = arr.as_boolean();
    let lit = lit_arr.as_boolean().value(0);
    let n = b.len();
    let values = b.values();
    let bits = match (op, lit) {
        (Eq, true) | (Ne, false) | (Gt, false) | (Ge, true) => values.clone(),
        (Eq, false) | (Ne, true) | (Lt, true) | (Le, false) => !values,
        (Lt, false) | (Gt, true) => BooleanBuffer::new_unset(n),
        (Le, true) | (Ge, false) => BooleanBuffer::new_set(n),
        _ => return None,
    };
    Some(Arc::new(BooleanArray::new(bits, b.nulls().cloned())))
}

/// `left AND right` in one pass, when both are a comparison of the **same** column with a
/// literal, one bounding it below and one above — either order, each literal on either side —
/// and the column is an integer, temporal or float type the literals need no promotion to
/// meet. `None` for every other shape, which the caller evaluates as it always has.
pub(crate) fn try_prim_range(
    left: &Expr,
    right: &Expr,
    batch: &RecordBatch,
) -> Result<Option<ArrayRef>, ExprError> {
    let (Some((col_l, op_l, lit_l)), Some((col_r, op_r, lit_r))) =
        (bound_of(left), bound_of(right))
    else {
        return Ok(None);
    };
    if col_l != col_r {
        return Ok(None);
    }
    let left_is_lower = match (is_lower(op_l), is_lower(op_r)) {
        (Some(true), Some(false)) => true,
        (Some(false), Some(true)) => false,
        _ => return Ok(None),
    };
    // Only once the shape matched, so a declined `AND` has paid nothing.
    let arr = Expr::Col {
        name: col_l.to_owned(),
    }
    .eval(batch)?;
    // The literal of each bound, coerced exactly as the scalar comparison path coerces it —
    // left before right, so a literal that fails to coerce raises the error the two-kernel
    // evaluation would have raised first. A bound that would promote the *column* (an integer
    // column against a float literal) is declined: the generic path compares the promoted
    // values, and this kernel does not.
    let (Some((a, l)), Some((b, r))) = (
        scalar_operands(arr.clone(), lit_l)?,
        scalar_operands(arr.clone(), lit_r)?,
    ) else {
        return Ok(None);
    };
    let ((lo_op, lo), (hi_op, hi)) = if left_is_lower {
        ((op_l, l), (op_r, r))
    } else {
        ((op_r, r), (op_l, l))
    };
    let dt = arr.data_type();
    if a.data_type() != dt || b.data_type() != dt || lo.data_type() != dt || hi.data_type() != dt {
        return Ok(None);
    }
    if lo.is_null(0) || hi.is_null(0) {
        return Ok(None);
    }
    let bits = match dt {
        DataType::Float64 => float_range::<Float64Type>(&arr, &lo, &hi, lo_op, hi_op),
        DataType::Float32 => float_range::<Float32Type>(&arr, &lo, &hi, lo_op, hi_op),
        _ => with_int_type!(dt, T => {
            let values = arr.as_primitive::<T>().values();
            let (lo, hi) = (lo.as_primitive::<T>().value(0), hi.as_primitive::<T>().value(0));
            Some(int_range(values, lo, hi, matches!(lo_op, BinaryOp::Gt), matches!(hi_op, BinaryOp::Lt)))
        }, _ => None),
    };
    Ok(bits.map(|bits| Arc::new(BooleanArray::new(bits, arr.nulls().cloned())) as ArrayRef))
}

/// `lo <(=) v <(=) hi`, one pass, one strictness flag per side.
fn int_range<T: Copy + PartialOrd>(
    values: &[T],
    lo: T,
    hi: T,
    strict_lo: bool,
    strict_hi: bool,
) -> BooleanBuffer {
    match (strict_lo, strict_hi) {
        (false, false) => fill(values, |v| (v >= lo) & (v <= hi)),
        (false, true) => fill(values, |v| (v >= lo) & (v < hi)),
        (true, false) => fill(values, |v| (v > lo) & (v <= hi)),
        (true, true) => fill(values, |v| (v > lo) & (v < hi)),
    }
}

/// The float range, with each bound written the way `binary::float_cmp_bits` writes it, or
/// `None` for a NaN literal.
// The negations are the NaN handling — see `binary::float_scalar_cmp`.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn float_range<T: ArrowPrimitiveType>(
    arr: &ArrayRef,
    lo: &ArrayRef,
    hi: &ArrayRef,
    lo_op: BinaryOp,
    hi_op: BinaryOp,
) -> Option<BooleanBuffer>
where
    T::Native: PartialOrd + Copy,
{
    let (lo, hi) = (
        lo.as_primitive::<T>().value(0),
        hi.as_primitive::<T>().value(0),
    );
    #[allow(clippy::eq_op)]
    if lo != lo || hi != hi {
        return None; // NaN
    }
    let values = arr.as_primitive::<T>().values();
    Some(
        match (matches!(lo_op, BinaryOp::Gt), matches!(hi_op, BinaryOp::Lt)) {
            (false, false) => fill(values, |v| !(v < lo) & (v <= hi)),
            (false, true) => fill(values, |v| !(v < lo) & (v < hi)),
            (true, false) => fill(values, |v| !(v <= lo) & (v <= hi)),
            (true, true) => fill(values, |v| !(v <= lo) & (v < hi)),
        },
    )
}

/// Whether `a AND b` is a lower and an upper bound of one column against literals — the shape
/// [`try_prim_range`] (or, for strings, `string::try_string_range`) answers in one pass. The
/// types are not checked here; a kernel that declines leaves the `AND` to the generic path.
pub(crate) fn is_range_pair(a: &Expr, b: &Expr) -> bool {
    match (bound_of(a), bound_of(b)) {
        (Some((ca, oa, _)), Some((cb, ob, _))) => {
            ca == cb && matches!((is_lower(oa), is_lower(ob)), (Some(x), Some(y)) if x != y)
        }
        _ => false,
    }
}

/// `(column, op, literal)` for `col OP lit` or `lit OP col`, with `op` read as `col OP lit`.
fn bound_of(expr: &Expr) -> Option<(&str, BinaryOp, &crate::Literal)> {
    let Expr::Binary { op, left, right } = expr else {
        return None;
    };
    match (left.as_ref(), right.as_ref()) {
        (Expr::Col { name }, Expr::Lit { value }) => Some((name, *op, value)),
        (Expr::Lit { value }, Expr::Col { name }) => Some((name, mirror(*op)?, value)),
        _ => None,
    }
}

/// The operator read from the other side, for the four ordering comparisons only.
fn mirror(op: BinaryOp) -> Option<BinaryOp> {
    use BinaryOp::{Ge, Gt, Le, Lt};
    Some(match op {
        Lt => Gt,
        Le => Ge,
        Gt => Lt,
        Ge => Le,
        _ => return None,
    })
}

/// `Some(true)` for a lower bound, `Some(false)` for an upper one, `None` otherwise.
fn is_lower(op: BinaryOp) -> Option<bool> {
    match op {
        BinaryOp::Gt | BinaryOp::Ge => Some(true),
        BinaryOp::Lt | BinaryOp::Le => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        Date32Array, Datum, Float64Array, Int32Array, Int64Array, Scalar,
        TimestampMicrosecondArray, UInt8Array,
    };
    use arrow::compute::kernels::{boolean, cmp};
    use arrow::datatypes::{Field, Schema};
    use bc_arrow::canon_float_array;

    use crate::Literal;

    const OPS: [BinaryOp; 6] = [
        BinaryOp::Eq,
        BinaryOp::Ne,
        BinaryOp::Lt,
        BinaryOp::Le,
        BinaryOp::Gt,
        BinaryOp::Ge,
    ];

    /// The generic path's answer: canonicalize floats, then `arrow_ord::cmp`.
    fn oracle(op: BinaryOp, arr: &ArrayRef, lit: &ArrayRef, lit_on_right: bool) -> BooleanArray {
        let (a, l) = (canon_float_array(arr), canon_float_array(lit));
        let a: &dyn Array = a.as_ref();
        let s = Scalar::new(l);
        let (x, y): (&dyn Datum, &dyn Datum) = if lit_on_right { (&a, &s) } else { (&s, &a) };
        match op {
            BinaryOp::Eq => cmp::eq(x, y),
            BinaryOp::Ne => cmp::neq(x, y),
            BinaryOp::Lt => cmp::lt(x, y),
            BinaryOp::Le => cmp::lt_eq(x, y),
            BinaryOp::Gt => cmp::gt(x, y),
            _ => cmp::gt_eq(x, y),
        }
        .unwrap()
    }

    fn mirrored(op: BinaryOp, lit_on_right: bool) -> BinaryOp {
        if lit_on_right {
            op
        } else {
            mirror(op).unwrap_or(op)
        }
    }

    /// 202 rows (three full words and a partial one), nulls scattered, and the same column
    /// sliced at an odd offset; literals drawn from the column itself.
    fn columns() -> Vec<(ArrayRef, Vec<ArrayRef>)> {
        let ints: Vec<Option<i64>> = (0..200)
            .map(|i| (i % 13 != 5).then_some((i * 7919) % 101 - 50))
            .chain([Some(i64::MIN), Some(i64::MAX)])
            .collect();
        let i64s: ArrayRef = Arc::new(Int64Array::from(ints.clone()));
        let i32s: ArrayRef = Arc::new(Int32Array::from(
            ints.iter().map(|v| v.map(|x| x as i32)).collect::<Vec<_>>(),
        ));
        let u8s: ArrayRef = Arc::new(UInt8Array::from(
            ints.iter().map(|v| v.map(|x| x as u8)).collect::<Vec<_>>(),
        ));
        // Dates across the epoch, the far past and the far future.
        let dates: ArrayRef = Arc::new(Date32Array::from(
            ints.iter()
                .map(|v| v.map(|x| (x as i32).wrapping_mul(997)))
                .chain([Some(-719_528), Some(2_932_896), Some(11_016)])
                .collect::<Vec<_>>(),
        ));
        let ts: ArrayRef = Arc::new(
            TimestampMicrosecondArray::from(
                ints.iter()
                    .map(|v| v.map(|x| x.wrapping_mul(1_000_003)))
                    .collect::<Vec<_>>(),
            )
            .with_timezone("UTC"),
        );
        let mut out = Vec::new();
        for arr in [i64s, i32s, u8s, dates, ts] {
            let lits: Vec<ArrayRef> = [0usize, 3, 7, 199, 200]
                .iter()
                .map(|&i| arr.slice(i, 1))
                .filter(|l| !l.is_null(0))
                .collect();
            out.push((arr.slice(3, arr.len() - 3), lits.clone()));
            out.push((arr, lits));
        }
        out
    }

    #[test]
    fn int_scalar_cmp_matches_arrow_on_every_type_operator_and_side() {
        for (arr, lits) in columns() {
            for lit in &lits {
                for op in OPS {
                    for lit_on_right in [true, false] {
                        let got =
                            int_scalar_cmp(mirrored(op, lit_on_right), &arr, lit).expect("served");
                        assert_eq!(
                            got.as_boolean(),
                            &oracle(op, &arr, lit, lit_on_right),
                            "{op:?} {} lit_on_right={lit_on_right}",
                            arr.data_type()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn int_scalar_cmp_declines_a_float_a_null_literal_and_a_type_mismatch() {
        let f: ArrayRef = Arc::new(Float64Array::from(vec![1.0]));
        assert!(int_scalar_cmp(BinaryOp::Lt, &f, &f).is_none());
        let i: ArrayRef = Arc::new(Int64Array::from(vec![1]));
        let null: ArrayRef = Arc::new(Int64Array::from(vec![None::<i64>]));
        assert!(int_scalar_cmp(BinaryOp::Lt, &i, &null).is_none());
        let i32s: ArrayRef = Arc::new(Int32Array::from(vec![1]));
        assert!(int_scalar_cmp(BinaryOp::Lt, &i, &i32s).is_none());
    }

    #[test]
    fn bool_scalar_cmp_matches_arrow_on_every_operator_and_literal() {
        use arrow::array::BooleanArray as B;
        let col: ArrayRef = Arc::new(B::from(
            (0..150)
                .map(|i| (i % 7 != 3).then_some(i % 3 == 0))
                .collect::<Vec<_>>(),
        ));
        for arr in [col.clone(), col.slice(5, 101)] {
            for lit in [true, false] {
                let lit: ArrayRef = Arc::new(B::from(vec![lit]));
                for op in OPS {
                    for lit_on_right in [true, false] {
                        let got = bool_scalar_cmp(mirrored(op, lit_on_right), &arr, &lit)
                            .expect("served");
                        assert_eq!(got.as_boolean(), &oracle(op, &arr, &lit, lit_on_right));
                    }
                }
            }
        }
        let null_lit: ArrayRef = Arc::new(B::from(vec![None]));
        assert!(bool_scalar_cmp(BinaryOp::Eq, &col, &null_lit).is_none());
    }

    fn bound(col_left: bool, op: BinaryOp, lit: Literal) -> Expr {
        let col = Box::new(Expr::Col { name: "x".into() });
        let lit = Box::new(Expr::Lit { value: lit });
        let (left, right) = if col_left { (col, lit) } else { (lit, col) };
        Expr::Binary { op, left, right }
    }

    /// Kleene `AND` of the two comparisons, each answered by the generic kernel.
    fn range_oracle(l: &Expr, r: &Expr, batch: &RecordBatch) -> BooleanArray {
        let side = |e: &Expr| {
            let Expr::Binary { op, left, right } = e else {
                unreachable!()
            };
            let col = batch.column(0).clone();
            let (value, lit_on_right) = match (left.as_ref(), right.as_ref()) {
                (Expr::Col { .. }, Expr::Lit { value }) => (value, true),
                (Expr::Lit { value }, Expr::Col { .. }) => (value, false),
                _ => unreachable!(),
            };
            let (a, l) = scalar_operands(col, value).unwrap().unwrap();
            oracle(*op, &a, &l, lit_on_right)
        };
        boolean::and_kleene(&side(l), &side(r)).unwrap()
    }

    fn batch_of(arr: ArrayRef) -> RecordBatch {
        let schema = Schema::new(vec![Field::new("x", arr.data_type().clone(), true)]);
        RecordBatch::try_new(Arc::new(schema), vec![arr]).unwrap()
    }

    /// Every pair of literals as lower and upper bound, every strictness, each literal on
    /// either side, both `AND` orders — held to the two generic kernels.
    fn check_ranges(arr: ArrayRef, lits: &[Literal]) {
        let batch = batch_of(arr);
        let mut served = 0;
        for lo_lit in lits {
            for hi_lit in lits {
                for lo_op in [BinaryOp::Gt, BinaryOp::Ge] {
                    for hi_op in [BinaryOp::Lt, BinaryOp::Le] {
                        for (lo_left, hi_left) in [(true, true), (false, true), (true, false)] {
                            // A bound written with the literal on the left mirrors its operator.
                            let side = |left: bool, op: BinaryOp, lit: &Literal| {
                                let op = if left { op } else { mirror(op).unwrap() };
                                bound(left, op, lit.clone())
                            };
                            let lo = side(lo_left, lo_op, lo_lit);
                            let hi = side(hi_left, hi_op, hi_lit);
                            assert!(is_range_pair(&lo, &hi) && is_range_pair(&hi, &lo));
                            for (l, r) in [(&lo, &hi), (&hi, &lo)] {
                                if let Some(got) = try_prim_range(l, r, &batch).unwrap() {
                                    served += 1;
                                    assert_eq!(
                                        got.as_boolean(),
                                        &range_oracle(l, r, &batch),
                                        "{l:?} AND {r:?}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(served > 0, "the kernel never served this type");
    }

    #[test]
    fn a_fused_range_matches_two_kernels_over_integers_and_dates() {
        let ints: ArrayRef = Arc::new(Int64Array::from(
            (-70..70)
                .map(|i| (i % 9 != 0).then_some(i))
                .chain([Some(i64::MIN), Some(i64::MAX)])
                .collect::<Vec<_>>(),
        ));
        check_ranges(
            ints,
            &[
                Literal::Int(-5),
                Literal::Int(0),
                Literal::Int(5),
                Literal::Int(i64::MAX),
            ],
        );
        // Dates bounded by strings: the coercion `EventDate >= '2013-07-01'` takes.
        let dates: ArrayRef = Arc::new(Date32Array::from(
            (-400..400)
                .map(|i| (i % 11 != 0).then_some(i * 37 + 11_000))
                .collect::<Vec<_>>(),
        ));
        check_ranges(
            dates,
            &[
                Literal::Str("1999-12-31".into()),
                Literal::Str("2000-02-29".into()),
                Literal::Str("2001-03-01".into()),
            ],
        );
    }

    #[test]
    fn a_fused_float_range_matches_the_canonicalizing_kernels() {
        let mut vals: Vec<Option<f64>> = (-60..60).map(|i| Some(f64::from(i) / 8.0)).collect();
        vals.extend([
            Some(f64::NAN),
            Some(-f64::NAN),
            Some(-0.0),
            Some(0.0),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
            None,
        ]);
        let floats: ArrayRef = Arc::new(Float64Array::from(vals));
        check_ranges(
            floats,
            &[
                Literal::Float(-0.0),
                Literal::Float(0.0),
                Literal::Float(1.5),
                Literal::Float(f64::INFINITY),
            ],
        );
    }

    #[test]
    fn a_range_that_would_promote_the_column_or_has_a_nan_bound_is_declined() {
        let ints = batch_of(Arc::new(Int64Array::from(vec![1, 2, 3])));
        let lo = bound(true, BinaryOp::Ge, Literal::Float(1.5));
        let hi = bound(true, BinaryOp::Le, Literal::Int(3));
        assert!(try_prim_range(&lo, &hi, &ints).unwrap().is_none());
        let floats = batch_of(Arc::new(Float64Array::from(vec![1.0, f64::NAN])));
        let nan = bound(true, BinaryOp::Ge, Literal::Float(f64::NAN));
        let hi = bound(true, BinaryOp::Le, Literal::Float(3.0));
        assert!(try_prim_range(&nan, &hi, &floats).unwrap().is_none());
        // Two lower bounds are not a range.
        let lo2 = bound(true, BinaryOp::Gt, Literal::Int(0));
        assert!(!is_range_pair(
            &lo2,
            &bound(true, BinaryOp::Ge, Literal::Int(1))
        ));
    }
}
