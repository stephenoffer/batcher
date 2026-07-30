"""Scalar `Expr` IR → dataframe column, for the GPU (cuDF) and verification (pandas) backends.

This is the translator's vocabulary: every expression the GPU path can evaluate is a case
here, and everything else raises `Unsupported` so the caller drops the whole stage to the
native CPU engine. Coverage is what decides how much of a real query reaches the device — a
plan is GPU-eligible only if *every* expression in it is, so one missing case sends an
otherwise perfect chain back to the host.

Two rules govern what may be added. A case must be **result-identical to the CPU engine**
including its null behavior, which is not the same as its NaN behavior: Batcher follows Arrow,
where `null` and `NaN` are different values, and both backends here are loaded so they agree
(see `backend.DfBackend`). And a case must be *exact* — never an approximation that happens to
match on the common input, because a fallback costs time while a wrong answer costs trust.
"""

from __future__ import annotations

import datetime as _dt
import operator
from typing import TYPE_CHECKING, Any

from batcher.core.gpu_plan.backend import Unsupported

if TYPE_CHECKING:
    from batcher.core.gpu_plan.backend import DfBackend

__all__ = ["eval_expr", "literal_value"]

# Arithmetic, comparison and boolean operators that map straight onto the libraries'
# element-wise dunders. Both backends implement Kleene logic for `and`/`or` and propagate
# nulls through arithmetic, which is exactly what the CPU engine does.
_BINOPS = {
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

# Unary math functions, as `(device method, NumPy ufunc)`. Both names are spelled out rather
# than derived from the engine's name, because guessing either is a silent-wrong-answer bug:
# `trunc` was resolved by name to pandas' `Series.truncate`, which slices *rows by index* and
# has nothing to do with truncating a value — it returned a different table without raising.
# A `None` device method means the ufunc is used on both backends.
_MATH_FNS = {
    "abs": ("abs", "absolute"),
    "acos": ("acos", "arccos"),
    "asin": ("asin", "arcsin"),
    "atan": ("atan", "arctan"),
    "cbrt": (None, "cbrt"),
    "ceil": ("ceil", "ceil"),
    "cos": ("cos", "cos"),
    "cosh": (None, "cosh"),
    "degrees": (None, "degrees"),
    "exp": ("exp", "exp"),
    "floor": ("floor", "floor"),
    "ln": ("log", "log"),
    "log10": (None, "log10"),
    "log2": (None, "log2"),
    "radians": (None, "radians"),
    "sin": ("sin", "sin"),
    "sinh": (None, "sinh"),
    "sqrt": ("sqrt", "sqrt"),
    "tan": ("tan", "tan"),
    "tanh": (None, "tanh"),
    "trunc": (None, "trunc"),
}

# `.dt` attributes that carry the same name as the engine's date function.
_DATE_ATTRS = frozenset({"day", "hour", "microsecond", "minute", "month", "quarter", "second",
                         "year"})  # fmt: skip

# String functions that are a no-argument `.str` method of the same name on both backends.
_STR_METHODS = frozenset({"lower", "upper", "title", "reverse"})

# String functions taking a single `pattern` argument, mapped to their `.str` method.
_STR_PATTERN_METHODS = {
    "contains": "contains",
    "starts_with": "startswith",
    "ends_with": "endswith",
}


def literal_value(tagged: dict) -> Any:
    """A tagged IR literal as the Python scalar the dataframe libraries compare against.

    The tag carries the *type*, and dropping it is a silent-wrong-answer bug rather than a
    cosmetic one: a `date` literal rides the wire as days-since-epoch and a `timestamp` as
    microseconds, so handing the raw integer to a comparison against a datetime column either
    raises (pandas) or coerces to something meaningless (cuDF). Non-finite floats ride as
    names because JSON has no `NaN`/`Infinity` token.

    Args:
        tagged: The single-entry ``{"<kind>": <value>}`` dict from a `lit` node's ``value``.

    Returns:
        The Python scalar the literal denotes.

    Raises:
        Unsupported: For a literal kind the translator does not model.
    """
    if len(tagged) != 1:
        raise Unsupported(f"literal shape {sorted(tagged)}")
    kind, raw = next(iter(tagged.items()))
    if kind in ("int", "str", "bool"):
        return raw
    if kind == "float":
        return float(raw) if isinstance(raw, (int, float)) else _NON_FINITE[raw]
    if kind == "date":
        return _dt.date(1970, 1, 1) + _dt.timedelta(days=int(raw))
    if kind == "timestamp":
        return _dt.datetime(1970, 1, 1) + _dt.timedelta(microseconds=int(raw))
    raise Unsupported(f"literal kind {kind!r}")


_NON_FINITE = {
    "NaN": float("nan"),
    "inf": float("inf"),
    "+inf": float("inf"),
    "Infinity": float("inf"),
    "-inf": float("-inf"),
    "-Infinity": float("-inf"),
}


def eval_expr(ir: dict, df, be: DfBackend):
    """Evaluate one `Expr` IR node against dataframe `df`, returning a column or a scalar.

    A scalar comes back only for a bare literal; every caller that needs a column passes the
    result through `DfBackend.column`, so a literal is broadcast in the library's own layer
    rather than one Python object per row.

    Args:
        ir: The expression's JSON IR node.
        df: The dataframe the column names resolve against.
        be: The dataframe backend to compute on.

    Returns:
        A Series of `df`'s length, or a Python scalar for a bare literal.

    Raises:
        Unsupported: For any node outside the translated subset.
    """
    handler = _HANDLERS.get(ir.get("e"))
    if handler is None:
        raise Unsupported(f"expr {ir.get('e')}")
    return handler(ir, df, be)


def _col(ir, df, _be):
    name = ir["name"]
    if name not in df.columns:
        raise Unsupported(f"column {name!r} absent from the GPU frame")
    return df[name]


def _binary(ir, df, be):
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
    if op == "mod":
        return _truncated_mod(be.column(left, df), right, df, be)
    if op in _COMPARISONS and (be.is_float(left) or be.is_float(right)):
        return _compare(op, be.column(left, df), be.column(right, df))
    fn = _BINOPS.get(op)
    if fn is None:
        raise Unsupported(f"binary op {op}")
    return fn(left, right)


_COMPARISONS = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})


def _compare(op: str, left, right):
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
    naive = _BINOPS[op](left, right)
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


def _not(ir, df, be):
    return ~be.column(eval_expr(ir["input"], df, be), df)


def _is_null(ir, df, be):
    return be.column(eval_expr(ir["input"], df, be), df).isna()


def _is_not_null(ir, df, be):
    return be.column(eval_expr(ir["input"], df, be), df).notna()


def _is_nan(ir, df, be):
    # `x != x` is True exactly for NaN and null for null, which is what the engine returns —
    # `.isna()` would fold the two together and report NaN and null alike.
    x = be.column(eval_expr(ir["input"], df, be), df)
    return x != x


def _is_inf(ir, df, be):
    x = be.column(eval_expr(ir["input"], df, be), df)
    return (x == float("inf")) | (x == float("-inf"))


def _cast(ir, df, be):
    from batcher.plan.types.registry import DTYPE_REGISTRY

    target = DTYPE_REGISTRY.get(ir["dtype"])
    if target is None:
        raise Unsupported(f"cast to {ir['dtype']!r}")
    if ir.get("try_cast"):
        # A `try_cast` nulls the rows that fail rather than raising, and neither backend has
        # that mode — approximating it with a strict cast would raise on data the CPU engine
        # accepts.
        raise Unsupported("try_cast")
    x = be.column(eval_expr(ir["input"], df, be), df)
    import pyarrow as pa

    if pa.types.is_integer(target) and be.is_float(x):
        # Float → integer **rounds**, half to even, where a direct `astype` instead raises on
        # any value with a fractional part. That is not a cast this path can decline: it is
        # the ordinary spelling of bucketing a measure, so refusing it would send every such
        # query to the host.
        x = _ufunc("rint", x, be)
    return x.astype(be.dtype(target))


def _case(ir, df, be):
    """`CASE WHEN … THEN … ELSE …` as a fold of `where` from the last branch backwards.

    A null predicate must select the *else* arm, matching the engine (a `WHEN` that is
    unknown is not taken), so each branch's condition is filled with False before use.
    """
    otherwise = ir.get("otherwise")
    if otherwise is None:
        raise Unsupported("CASE without ELSE")  # the null-typed default has no column dtype
    out = be.column(eval_expr(otherwise, df, be), df)
    for branch in reversed(ir["branches"]):
        cond = be.column(eval_expr(branch["when"], df, be), df).fillna(False)
        out = be.column(eval_expr(branch["then"], df, be), df).where(cond, out)
    return out


def _coalesce(ir, df, be):
    inputs = [be.column(eval_expr(i, df, be), df) for i in ir["inputs"]]
    out = inputs[0]
    for nxt in inputs[1:]:
        out = out.where(out.notna(), nxt)
    return out


def _nullif(ir, df, be):
    left = be.column(eval_expr(ir["left"], df, be), df)
    right = be.column(eval_expr(ir["right"], df, be), df)
    return left.where((left != right).fillna(True), None)


def _extreme(ir, df, be, *, want_max: bool):
    """`GREATEST`/`LEAST` — the extreme of the *non-null* arguments, null only if all are.

    This is the engine's semantics (verified against it), and it is not SQL's: DuckDB's
    `GREATEST` returns NULL when any argument is NULL. Folding pairwise keeps it exact on
    both backends without an `axis=1` reduction cuDF only partly supports.
    """
    inputs = [be.column(eval_expr(i, df, be), df) for i in ir["inputs"]]
    out = inputs[0]
    for nxt in inputs[1:]:
        # Through `_compare`, so a float `NaN` is the largest value here too — the engine's
        # `greatest` over a NaN returns NaN, and an IEEE comparison would return the other
        # argument instead.
        op = "ge" if want_max else "le"
        picked = (
            _compare(op, out, nxt)
            if be.is_float(out) or be.is_float(nxt)
            else ((out >= nxt) if want_max else (out <= nxt))
        )
        out = out.where(picked.fillna(False), nxt).where(out.notna(), nxt).where(nxt.notna(), out)
    return out


def _math(ir, df, be):
    x = be.column(eval_expr(ir["input"], df, be), df)
    fn = ir["fn"]
    if fn == "sign":
        # Neither Series type has `.sign()`. The arithmetic form has to restore the null
        # itself: a comparison against a null yields null, and casting that to an integer
        # raises rather than propagating.
        pos = (x > 0).fillna(False).astype("int64")
        neg = (x < 0).fillna(False).astype("int64")
        return (pos - neg).astype(be.dtype(_float64())).where(x.notna(), None)
    if fn == "round":
        return _round(x, 0, be)
    names = _MATH_FNS.get(fn)
    if names is None:
        raise Unsupported(f"math fn {fn}")
    device_method, ufunc = names
    if be.is_gpu and device_method is not None:
        return getattr(x, device_method)()
    return _ufunc(ufunc, x, be)


def _ufunc(name: str, x, be):
    """Apply NumPy's `name` element-wise, keeping the null mask the ufunc would destroy.

    On the device the column is handed to the ufunc directly, which cuDF dispatches on the
    GPU — materializing it as a host array first would move the whole column off the device
    to compute something it can do in place.

    On the host backend the ufunc drops the Arrow extension type to a plain float array, in
    which a null becomes `NaN`; after that null and `NaN` are the same value and every one of
    them converts back to null. That turns `sqrt(NaN)` into null, which is not what the engine
    returns, so the input's own mask is re-applied to restore the distinction.
    """
    import numpy as np

    fn = getattr(np, name, None)
    if fn is None:
        raise Unsupported(f"math fn {name}")
    try:
        if be.is_gpu:
            return fn(x)
        raw = fn(x.to_numpy(dtype="float64", na_value=np.nan))
    except (TypeError, AttributeError, NotImplementedError, ValueError) as exc:
        raise Unsupported(f"math fn {name}: {exc}") from exc
    out = be.float_series(raw)
    out.index = x.index
    return out.where(x.notna(), None)


def _round(x, digits: int, be):
    """`round(x, digits)` rounding halves **away from zero**, as the engine does.

    Both backends round halves to *even* (NumPy's rule), so `round(-2.5)` is `-2.0` there and
    `-3.0` in the engine. The difference is invisible on most data and systematic on money,
    which is exactly the data most likely to be rounded.
    """
    scale = 10.0**digits
    scaled = x * scale if digits else x
    shifted = _ufunc("floor", _ufunc("absolute", scaled, be) + 0.5, be)
    signed = shifted.where((scaled >= 0).fillna(True), -shifted)
    return signed / scale if digits else signed


def _float64():
    import pyarrow as pa

    return pa.float64()


def _math2(ir, df, be):
    fn = ir["fn"]
    left = eval_expr(ir["left"], df, be)
    right = eval_expr(ir["right"], df, be)
    if fn == "pow":
        return be.column(left, df) ** right
    if fn == "round":
        # `round(x, digits)` — the digit count is a constant in every plan the engine builds,
        # and a per-row digit count has no Series form on either backend.
        if be.is_series(right):
            raise Unsupported("round with a non-constant digit count")
        return _round(be.column(left, df), int(right), be)
    raise Unsupported(f"math2 fn {fn}")


def _in_list(ir, df, be):
    values = [literal_value(v) for v in ir["set"]]
    return be.column(eval_expr(ir["input"], df, be), df).isin(values)


def _str(ir, df, be):
    x = be.column(eval_expr(ir["input"], df, be), df)
    fn = ir["fn"]
    if fn in _STR_METHODS:
        return getattr(x.str, fn)()
    if fn == "len":
        return x.str.len().astype(be.dtype(_int64()))
    if fn in _STR_PATTERN_METHODS:
        return getattr(x.str, _STR_PATTERN_METHODS[fn])(ir["pattern"])
    if fn == "replace":
        # The engine replaces every occurrence and treats the pattern as a literal.
        return x.str.replace(ir["pattern"], ir["replacement"], regex=False)
    if fn == "substr":
        # SQL's 1-based, inclusive `substring(s, start, length)`: a `start` below 1 spends
        # part of the length before the string begins, so `substr(s, 0, 1)` is the empty
        # string. Slicing from a 0-based `start` instead silently returns a shifted window.
        start, length = int(ir["start"]), int(ir["length"])
        return x.str.slice(max(start - 1, 0), max(start + length - 1, 0))
    if fn in ("trim", "l_trim", "r_trim"):
        return {"trim": x.str.strip, "l_trim": x.str.lstrip, "r_trim": x.str.rstrip}[fn]()
    raise Unsupported(f"str fn {fn}")


def _date(ir, df, be):
    fn = ir["fn"]
    if fn not in _DATE_ATTRS:
        raise Unsupported(f"date fn {fn}")
    x = be.column(eval_expr(ir["input"], df, be), df)
    return getattr(x.dt, fn).astype(be.dtype(_int64()))


def _int64():
    import pyarrow as pa

    return pa.int64()


_HANDLERS = {
    "col": _col,
    "lit": lambda ir, _df, _be: literal_value(ir["value"]),
    "binary": _binary,
    "not": _not,
    "is_null": _is_null,
    "is_not_null": _is_not_null,
    "is_nan": _is_nan,
    "is_inf": _is_inf,
    "cast": _cast,
    "case": _case,
    "coalesce": _coalesce,
    "nullif": _nullif,
    "greatest": lambda ir, df, be: _extreme(ir, df, be, want_max=True),
    "least": lambda ir, df, be: _extreme(ir, df, be, want_max=False),
    "math": _math,
    "math2": _math2,
    "in_list": _in_list,
    "str": _str,
    "date": _date,
}
