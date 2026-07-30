"""The named scalar-function families: math, two-argument math, strings, and dates.

Split from `exprs` on the seam that matters for reading them: that module is operators and
control flow — how values combine and how nulls propagate through a `CASE` — while this one is
a vocabulary, where each entry is an independent claim that one named function means the same
thing here as in the engine.

Almost every entry exists because the obvious translation is wrong in a way that produces a
plausible number rather than an error. `substr` is 1-based, `lpad` truncates as well as pads,
`day_of_week` counts from a different day, `round` breaks halves away from zero rather than to
even, and `position` reports 0 rather than -1 for "not found". Each is spelled out and pinned
by a case comparing it against the engine.

Two differences are known and deliberately left, because neither is a claim this vocabulary
can make and neither changes a value:

* the **transcendentals** (`sinh`, `tanh`, `acos` and their neighbours) can land one unit in
  the last place away from the engine's, because they are a different `libm`. On a real device
  they will differ again, and by more — a GPU's transcendental unit is its own implementation.
  IEEE does not require these to be correctly rounded, so agreeing to the last bit is not
  something any two implementations promise each other.
* `round` returns `+0.0` where the engine returns `-0.0`. The two compare equal, and every
  place in this package where the sign of zero could change an *answer* — a group key, a
  distinct row, a join key — folds them together on purpose (`ops.fold_zero`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.core.gpu_plan.backend import Unsupported

if TYPE_CHECKING:
    pass

__all__ = [
    "apply_ufunc",
    "eval_date",
    "eval_date_trunc",
    "eval_math",
    "eval_math2",
    "eval_str",
    "eval_strftime",
    "round_half_away",
]


def _int64():
    import pyarrow as pa

    return pa.int64()


def _float64():
    import pyarrow as pa

    return pa.float64()


def _float64():
    import pyarrow as pa

    return pa.float64()


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

# Date functions that are a `.dt` attribute of the same name. The ones that are NOT are
# handled explicitly below, because each disagrees with the engine on something: `day_of_week`
# counts from a different day, `week` is the ISO week rather than an attribute, and `epoch` is
# a unit conversion.
_DATE_ATTRS = frozenset({"day", "day_of_year", "days_in_month", "hour", "is_leap_year",
                         "microsecond", "minute", "month", "quarter", "second",
                         "year"})  # fmt: skip

# String functions that are a no-argument `.str` method, named where the two differ. Spelled
# out rather than passed through by name: a name that happens to exist on a Series is not the
# same as one that means what the engine means (see the `trunc` case in `_MATH_FNS`), and a
# name that exists on cuDF but not pandas is a path nothing verifies.
_STR_METHODS = {"lower": "lower", "upper": "upper", "initcap": "title"}

# String functions taking a single `pattern` argument, mapped to their `.str` method.
_STR_PATTERN_METHODS = {
    "contains": "contains",
    "starts_with": "startswith",
    "ends_with": "endswith",
}


def eval_math(ir, df, be, eval_expr):
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
        return round_half_away(x, 0, be)
    names = _MATH_FNS.get(fn)
    if names is None:
        raise Unsupported(f"math fn {fn}")
    device_method, ufunc = names
    if be.is_gpu and device_method is not None:
        return getattr(x, device_method)()
    return apply_ufunc(ufunc, x, be)


def apply_ufunc(name: str, x, be):
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
        # `errstate`: `sqrt(-1)` and `log(0)` are NaN and -inf here, which is what the engine
        # returns, so NumPy's warning about them is noise from a correct result. Left alone it
        # escapes from a translation into whatever the caller's warning filter does with it.
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            raw = fn(x.to_numpy(dtype="float64", na_value=np.nan))
    except (TypeError, AttributeError, NotImplementedError, ValueError) as exc:
        raise Unsupported(f"math fn {name}: {exc}") from exc
    out = be.float_series(raw)
    out.index = x.index
    return out.where(x.notna(), None)


def round_half_away(x, digits: int, be):
    """`round(x, digits)` rounding halves **away from zero**, as the engine does.

    Both backends round halves to *even* (NumPy's rule), so `round(-2.5)` is `-2.0` there and
    `-3.0` in the engine. The difference is invisible on most data and systematic on money,
    which is exactly the data most likely to be rounded.
    """
    scale = 10.0**digits
    scaled = x * scale if digits else x
    shifted = apply_ufunc("floor", apply_ufunc("absolute", scaled, be) + 0.5, be)
    signed = shifted.where((scaled >= 0).fillna(True), -shifted)
    return signed / scale if digits else signed


def _float64():
    import pyarrow as pa

    return pa.float64()


def eval_math2(ir, df, be, eval_expr):
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
        return round_half_away(be.column(left, df), int(right), be)
    raise Unsupported(f"math2 fn {fn}")


def eval_str(ir, df, be, eval_expr):
    x = be.column(eval_expr(ir["input"], df, be), df)
    fn = ir["fn"]
    if fn in _STR_METHODS:
        return getattr(x.str, _STR_METHODS[fn])()
    if fn in ("lpad", "rpad"):
        return _pad(x, int(ir["start"]), str(ir["pattern"]), left=fn == "lpad")
    if fn == "repeat":
        return x.str.repeat(int(ir["start"]))
    if fn == "right":
        return x.str.slice(-int(ir["start"]))
    if fn == "position":
        # SQL `position` is 1-based and reports 0 for "not found"; `find` is 0-based and -1.
        return (x.str.find(ir["pattern"]) + 1).astype(be.dtype(_int64()))
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


def _pad(x, width: int, fill: str, *, left: bool):
    """SQL `lpad`/`rpad`: pad to `width`, and **truncate** to it when the value is longer.

    The libraries' `rjust`/`ljust` only ever pad, so a value longer than the width came back
    unchanged where the engine returns its first `width` characters.
    """
    padded = x.str.pad(width, side="left" if left else "right", fillchar=fill)
    return padded.str.slice(0, width)


def eval_date(ir, df, be, eval_expr):
    fn = ir["fn"]
    x = be.column(eval_expr(ir["input"], df, be), df)
    if fn in _DATE_ATTRS:
        return getattr(x.dt, fn).astype(be.dtype(_int64()))
    if fn == "day_of_week":
        # The engine numbers from Sunday; both backends number from Monday. Same values, a
        # different origin — the sort of difference that is invisible until a Sunday. Written
        # as a shift and a single wrap rather than `% 7`, because one of the backends does not
        # implement `%` on Arrow-typed columns at all.
        shifted = x.dt.dayofweek + 1
        # `fillna(True)` keeps the null: a `where` whose condition is null takes the OTHER
        # branch, so a null timestamp would come back as Sunday rather than as missing.
        wrapped = shifted.where((shifted != 7).fillna(True), 0)
        return wrapped.astype(be.dtype(_int64())).where(x.notna(), None)
    if fn == "week":
        return _iso_week(x, be)
    if fn == "epoch":
        # Whole seconds since the Unix epoch, from the microseconds Arrow stores.
        import pyarrow as pa

        return (x.astype(be.dtype(pa.int64())) // 1_000_000).astype(be.dtype(_int64()))
    raise Unsupported(f"date fn {fn}")


#: `date_trunc` units that are a fixed duration, so truncating is flooring to a multiple of it.
#: The calendar units — week, month, quarter, year — are deliberately absent: their length
#: varies, so they are calendar arithmetic rather than a floor, and the two backends do not
#: offer the same construction for it. A `GROUP BY date_trunc('month', ...)` therefore still
#: runs on the CPU engine, which is a coverage gap and not a wrong answer.
_TRUNC_FREQ = {"second": "s", "minute": "min", "hour": "h", "day": "D"}


def eval_date_trunc(ir, df, be, eval_expr):
    """`date_trunc(unit, ts)` for the fixed-duration units, by flooring."""
    unit = ir.get("unit")
    freq = _TRUNC_FREQ.get(unit)
    if freq is None:
        raise Unsupported(f"date_trunc to a calendar unit ({unit})")
    x = be.column(eval_expr(ir["input"], df, be), df)
    floor = getattr(x.dt, "floor", None)
    if floor is None:
        raise Unsupported("date_trunc")
    try:
        return floor(freq)
    except (TypeError, ValueError, NotImplementedError) as exc:
        raise Unsupported(f"date_trunc {unit}: {exc}") from exc


def eval_strftime(ir, df, be, eval_expr):
    """`strftime(ts, format)` — the format is a constant in every plan the engine builds."""
    x = be.column(eval_expr(ir["input"], df, be), df)
    fmt = getattr(x.dt, "strftime", None)
    if fmt is None:
        raise Unsupported("strftime")
    try:
        return fmt(ir["format"])
    except (TypeError, ValueError, NotImplementedError) as exc:
        raise Unsupported(f"strftime: {exc}") from exc


def _iso_week(x, be):
    """The ISO-8601 week number, which is a calculation rather than a `.dt` attribute."""
    iso = getattr(x.dt, "isocalendar", None)
    if iso is None:
        raise Unsupported("date fn week")
    try:
        week = iso().week
    except (AttributeError, TypeError, NotImplementedError) as exc:
        raise Unsupported(f"date fn week: {exc}") from exc
    # `isocalendar` fills a null timestamp with 0 rather than propagating it, so the input's
    # own mask has to be put back — a week zero does not exist, and reading as one is worse
    # than reading as missing.
    return week.astype(be.dtype(_int64())).where(x.notna(), None)
