"""The operators: arithmetic, comparison, the bit family, and the three the engine redefines.

Split from `exprs`, which is the dispatcher and the control flow — how a `CASE` selects a
branch and how a null propagates through it — while this module is the operator table and the
handful of entries that cannot simply be the library's own dunder.

Four of them cannot, and each returns a plausible number rather than an error if it is:

* a **float comparison** follows the engine's total order, where `NaN` is the largest value and
  equal to itself, against IEEE's, where every comparison involving it is false;
* **`%`** takes the sign of the dividend, where both libraries implement the floored form;
* **`DATE - DATE`** is a count of days, where both libraries return a duration;
* the **bit operators over a boolean** answer in an integer, where both libraries answer in a
  boolean, and the **shifts** are not implemented on an Arrow-typed column at all.

And a fifth that is not a redefinition but a disagreement between the two backends themselves:
**`AND`/`OR` are three-valued**, which pandas implements on an Arrow column and cuDF does not.
It is computed here rather than inherited, so both sides give SQL's answer.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

from batcher.core.gpu_plan.backend import Unsupported

if TYPE_CHECKING:
    from batcher.core.gpu_plan.backend import DfBackend

__all__ = ["BINOPS", "compare", "eval_binary"]


# Arithmetic, comparison and boolean operators that map straight onto the libraries'
# element-wise dunders. Both backends propagate nulls through arithmetic, which is what the CPU
# engine does. `and`/`or` are listed but not reached through this table: the two libraries
# disagree about three-valued logic, so `_kleene` states it (see there).
BINOPS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
    "mod": operator.mod,
    "floor_div": operator.floordiv,
    "gt": operator.gt,
    "lt": operator.lt,
    "ge": operator.ge,
    "le": operator.le,
    "eq": operator.eq,
    "ne": operator.ne,
    "and": operator.and_,
    "or": operator.or_,
    "bit_and": operator.and_,
    "bit_or": operator.or_,
    "bit_xor": operator.xor,
    "shift_left": operator.lshift,
    "shift_right": operator.rshift,
}


def eval_binary(ir, df, be, eval_expr):
    op = ir["op"]
    left = eval_expr(ir["left"], df, be)
    right = eval_expr(ir["right"], df, be)
    if op == "concat":
        # String concatenation, not addition: `+` on two string Series does concatenate on
        # both backends, but a scalar operand has to become a column first or pandas raises.
        return be.column(left, df) + be.column(right, df)
    if op == "add_months":
        raise Unsupported("add_months")  # calendar arithmetic differs across the backends
    if not be.is_series(left) and not be.is_series(right):
        raise Unsupported("constant-folded binary")  # nothing to align against
    if op == "sub" and be.is_date(left) and be.is_date(right):
        return date_difference(left, right, be)
    if op == "mod":
        return _truncated_mod(be.column(left, df), right, df, be)
    if op in _BITWISE and (be.is_boolean(left) or be.is_boolean(right)):
        return _boolean_bitwise(op, be.column(left, df), be.column(right, df), be)
    if op in _SHIFTS:
        return _shift(op, be.column(left, df), right, be)
    if op in _COMPARISONS and (be.is_float(left) or be.is_float(right)):
        return compare(op, be.column(left, df), be.column(right, df))
    if op in ("and", "or"):
        return _kleene(op, be.column(left, df), be.column(right, df))
    fn = BINOPS.get(op)
    if fn is None:
        raise Unsupported(f"binary op {op}")
    return fn(left, right)


def date_difference(left, right, be):
    """`DATE - DATE` as a count of days, which is the engine's answer and not the libraries'.

    Subtracting two dates gives a *duration* on both backends and an integer number of days in
    the engine. The values agree; the column does not, and a shard contributing `duration[s]`
    beside a CPU-fallback shard's `int64` is a concatenation nobody can complete.

    A timestamp difference is a duration on both sides and is left alone — this is the only
    arithmetic where the two disagree about the unit rather than the number.
    """
    import pyarrow as pa

    days = getattr((left - right).dt, "days", None)
    if days is None:
        raise Unsupported("date difference")
    return days.astype(be.dtype(pa.int64()))


_COMPARISONS = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})

#: The bit operators, whose result over a BOOLEAN operand is an **integer** in the engine (and
#: in DuckDB) and a boolean in both dataframe libraries, which route them to logical `and`/`or`.
#: The values agree; the column does not, so `col("flag").xor(other)` came back as `true`/`false`
#: beside a CPU-recovered shard's `1`/`0` and the two could not be concatenated.
_BITWISE = frozenset({"bit_and", "bit_or", "bit_xor"})

#: The shift operators. Neither backend implements `<<`/`>>` on an Arrow-typed column at all, so
#: the operator dunder raised `TypeError` — which is not an `Unsupported`, and so was read as
#: "the GPU backend is broken" and logged as such for what is an ordinary integer expression.
_SHIFTS = frozenset({"shift_left", "shift_right"})

#: The widest shift the translation is exact for. A count at or past the word width is
#: implementation-defined, and the power-of-two form below would silently answer with an
#: overflow rather than with whatever the engine does, so it is declined instead.
_SHIFT_LIMIT = 63


def _boolean_bitwise(op: str, left, right, be: DfBackend):
    """A bit operator over a boolean operand, in the integer the engine answers it in."""
    import pyarrow as pa

    i64 = be.dtype(pa.int64())
    return BINOPS[op](left.astype(i64), right.astype(i64))


def _shift(op: str, left, right, be: DfBackend):
    """`a << n` / `a >> n` as the multiplication and floor division they are defined to be.

    An arithmetic right shift *is* floor division by the same power of two on a two's-complement
    integer — `-7 >> 1` and `floor(-7 / 2)` are both `-4` — so this is the operator's definition
    rather than an approximation of it, and it uses only arithmetic both backends implement on
    an Arrow-typed column.

    Only a constant, in-range count is accepted. A per-row count would need a per-row power of
    two, and a count at or past the word width has no portable answer at all.
    """
    if be.is_series(right):
        raise Unsupported("shift by a non-constant count")
    if not isinstance(right, int) or isinstance(right, bool) or not 0 <= right <= _SHIFT_LIMIT:
        raise Unsupported(f"shift by {right!r}")
    scale = 1 << right
    try:
        if op == "shift_right":
            return left // scale
        # A left shift that leaves the word **wraps** in the engine, and the two backends
        # disagree about that: one raises and one wraps. Declining on the data rather than on
        # the type keeps them identical — the ordinary shift, which cannot overflow, still runs
        # on the device, and the one that would need two different answers runs on neither.
        if scale > 1 and _overflows_left_shift(left, scale):
            raise Unsupported("shift_left past the word width")
        return left * scale
    except (TypeError, ValueError, OverflowError, NotImplementedError) as exc:
        raise Unsupported(f"{op}: {exc}") from exc


def _overflows_left_shift(left, scale: int) -> bool:
    """Whether shifting `left` by `scale` would leave the signed 64-bit range.

    One reduction over the column, which is cheap beside the multiplication it guards and is
    the only way to ask the question before an answer the two backends disagree about.
    """
    bound = (1 << 63) // scale
    return bool((left.abs() >= bound).fillna(False).any())


def _kleene(op: str, left, right):
    """`AND`/`OR` under three-valued logic, stated rather than inherited from the library.

    SQL's rule, and the engine's: `true OR unknown` is true, because the answer does not depend
    on the unknown; `false OR unknown` is unknown. `false AND unknown` is false for the same
    reason, and `true AND unknown` is unknown.

    pandas implements exactly that on an Arrow-typed column and **cuDF does not** — `null | true`
    came back null there. Every consequence of that is a wrong answer rather than an error: a
    `filter(a | b)` drops rows the engine keeps, and a `CASE` whose `WHEN` reads
    `is_null(x) or is_nan(x)` takes the else arm for precisely the null rows it was written to
    catch. `cut` is built on that exact condition, and returned a bucket where the engine
    returns null, on the device only.

    Written with `fillna` and `where`, so the correction itself never depends on the operator
    it is correcting: both intermediates are null-free booleans by construction.
    """
    if op == "or":
        # True wherever either side is true; unknown where the other side is not true.
        decided = left.fillna(False) | right.fillna(False)
        unknown = (left.isna() & ~right.fillna(False)) | (right.isna() & ~left.fillna(False))
    else:
        # False wherever either side is false; unknown where the other side is not false.
        decided = left.fillna(True) & right.fillna(True)
        unknown = (left.isna() & right.fillna(True)) | (right.isna() & left.fillna(True))
    return decided.where(~unknown, None)


def compare(op: str, left, right):
    """A float comparison under the engine's total order, where `NaN` is the largest value.

    Both dataframe libraries use IEEE comparison, in which every comparison involving `NaN`
    is false — so `NaN > 1.0` is False there and True in the engine (and in DuckDB), and
    `NaN = NaN` is False there and True in the engine. Left alone, that silently drops or
    keeps the wrong rows in a `WHERE` over a column that happens to carry a `NaN`.

    Nulls still propagate: a comparison with a null operand is null, which the naive form
    already gives, so the corrections are applied only where neither side is null.
    """
    ln, rn = left != left, right != right  # NaN masks (null-safe: null yields null)
    ln, rn = ln.fillna(False), rn.fillna(False)
    naive = BINOPS[op](left, right)
    if op == "eq":
        return naive.where(~(ln & rn), True).where(~(ln ^ rn), False)
    if op == "ne":
        return naive.where(~(ln & rn), False).where(~(ln ^ rn), True)
    if op in ("gt", "ge"):
        # NaN exceeds every non-NaN value; two NaNs are equal, so `>` is false and `>=` true.
        out = naive.where(~(ln & ~rn), True).where(~(rn & ~ln), False)
        return out.where(~(ln & rn), op == "ge")
    out = naive.where(~(ln & ~rn), False).where(~(rn & ~ln), True)
    return out.where(~(ln & rn), op == "le")


def _truncated_mod(left, right, df, be):
    """`a % b` with the sign of the *dividend*, which is what the engine computes.

    Both `%` implementations available here are the floored form (Python's, whose result
    takes the sign of the divisor), and one of the two backends does not implement `%` on
    Arrow-typed columns at all. Deriving the truncated remainder from floored division uses
    only operators both support, and makes the two paths provably the same expression.
    """
    right = be.column(right, df)
    floored = left - (left // right) * right
    differs = ((floored != 0) & ((floored < 0) != (left < 0))).fillna(False)
    return floored - right * differs.astype("int64")
