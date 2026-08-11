//! Operand coercion — bringing two arrays to a type the arrow kernels will accept.
//!
//! Arrow's comparison and arithmetic kernels demand *identical* input types, while SQL
//! does not: `qty * price` mixes an integer with a float, `amount = 1.50` mixes two
//! decimal scales, and a directory of Parquet files mixes `timestamp[ms]` with
//! `timestamp[us]`. Everything in this module exists to close that gap before a kernel
//! sees the operands, so a valid query returns a value instead of `Invalid comparison
//! operation`.
//!
//! It sits beside `binary` rather than inside it because it is a different
//! responsibility — *what type should these two be* rather than *what does this operator
//! compute* — and because three callers want it: `eval_binary`, `coalesce`, and
//! `greatest`/`least`.
//!
//! The arms in [`coerce_numeric`] are the pairings that need a rule of their own, where
//! the answer is not simply "widen both to a common type": a string literal read as a
//! date, an aware timestamp compared against a naive one. Everything else falls through
//! to the shared [`crate::common_supertype`] lattice, which is the single answer the set
//! operations and the Python control plane use too.

use arrow::array::{Array, ArrayRef, BooleanArray};
use arrow::compute::cast;
use arrow::datatypes::DataType;

use crate::ExprError;

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
pub(crate) fn align_date_timestamp_for_cmp(
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

pub(crate) fn align_decimals_for_cmp(
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
///
/// The arms below are the *specific* pairings that need a rule of their own — a string
/// literal read as a date, an aware timestamp compared to a naive one — where the answer
/// is not simply "widen both to a common type". Anything they don't name falls through to
/// the shared `common_supertype` lattice, which is what lets an all-null column, two
/// timestamps of differing resolution, or two decimals of differing scale meet at a type
/// the kernels accept instead of raising `Invalid comparison operation`.
pub(crate) fn coerce_numeric(
    l: &ArrayRef,
    r: &ArrayRef,
) -> Result<(ArrayRef, ArrayRef), ExprError> {
    use DataType::{
        Binary, Date32, Date64, Decimal128, Float64, Int64, LargeBinary, LargeUtf8, Timestamp, Utf8,
    };
    // Identical operand types are the overwhelmingly common case and every arm below is a
    // *cross*-type rule, so this both short-circuits the match and keeps the lattice
    // fallthrough from re-casting an array to the type it already has.
    if l.data_type() == r.data_type() {
        return Ok((l.clone(), r.clone()));
    }
    match (l.data_type(), r.data_type()) {
        (Int64, Float64) => Ok((cast(l, &Float64)?, r.clone())),
        (Float64, Int64) => Ok((l.clone(), cast(r, &Float64)?)),
        // An *integer* against a decimal adopts the decimal's precision/scale, so the
        // arithmetic/comparison stays exact (DuckDB widens INTEGER into the DECIMAL).
        (Decimal128(..), Int64) => Ok((l.clone(), cast(r, l.data_type())?)),
        (Int64, Decimal128(..)) => Ok((cast(l, r.data_type())?, r.clone())),
        // Two decimals pass through **untouched**, so each operator keeps its own
        // scale-propagation rule. Unifying them here instead — which is what the
        // `common_supertype` fallthrough below does — is applied *before* the operator
        // runs, so the operator's rule then sees two already-widened operands and compounds
        // the widening: `decimal(10,2) * decimal(8,3)` unifies to two `decimal(11,3)` and
        // multiplies to `decimal(23,6)` where SQL (and DuckDB, and this engine's own
        // `plan.types.infer`) say `s1+s2 = 5`. Values survive that — a wider scale is
        // trailing zeros — but the *column type* does not, so `Dataset.schema` and the
        // collected batch disagreed on every decimal multiply.
        //
        // Nothing needs the unification: the arithmetic kernels align scales themselves,
        // and the comparison arms in `binary.rs` call `align_decimals_for_cmp` right after
        // this, which is where the "kernels demand identical decimal types" requirement is
        // actually met.
        (Decimal128(..), Decimal128(..)) => Ok((l.clone(), r.clone())),
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
        // Anything the arms above don't name: ask the shared lattice for a type both sides
        // widen into. When it has one, cast both; when it doesn't (Int64 against Utf8),
        // hand the operands back untouched so the kernel raises its own typed error rather
        // than a lenient cast silently nulling the non-conforming side.
        (lt, rt) => match crate::common_supertype(lt, rt) {
            Some(common) => Ok((cast(l, &common)?, cast(r, &common)?)),
            None => Ok((l.clone(), r.clone())),
        },
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
