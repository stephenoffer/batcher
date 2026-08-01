"""The named scalar-function families: math, two-argument math, and dates.

Split from `exprs` on the seam that matters for reading them: that module is operators and
control flow — how values combine and how nulls propagate through a `CASE` — while this one is
a vocabulary, where each entry is an independent claim that one named function means the same
thing here as in the engine.

Almost every entry exists because the obvious translation is wrong in a way that produces a
plausible number rather than an error. `substr` is 1-based, `lpad` truncates as well as pads,
`day_of_week` counts from a different day and `round` breaks halves away from zero rather than
to even. Each is spelled out and pinned by a case comparing it against the engine. The string
family is the same idea and lives in `vocab.strings`.

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
from batcher.core.gpu_plan.temporal import (
    DATE_FNS,
    epoch_micros,
    eval_calendar_date,
    eval_calendar_trunc,
    isocalendar_field,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "apply_ufunc",
    "eval_date",
    "eval_date_trunc",
    "eval_math",
    "eval_math2",
    "eval_strftime",
    "round_half_away",
]


def _int64():
    import pyarrow as pa

    return pa.int64()


def _float64():
    import pyarrow as pa

    return pa.float64()


# Unary math functions, as the NumPy ufunc each one is. Spelled out rather than derived from
# the engine's name, because guessing is a silent-wrong-answer bug: `trunc` was resolved by
# name to pandas' `Series.truncate`, which slices *rows by index* and has nothing to do with
# truncating a value — it returned a different table without raising.
#
# The ufunc is used on **both** backends. This used to prefer a same-named method on the device
# (`x.sqrt()`, `x.exp()`, `x.sin()`), and cuDF no longer has any of them: every one of those
# calls raised `AttributeError` on a real GPU, which is not an `Unsupported` and so was reported
# as "the GPU backend is not usable" — for `sqrt`. The pandas verification backend could not see
# it, because the preference was `is_gpu`-gated and pandas took the ufunc branch. Measured on a
# T4 fleet: thirty-one functions, every one of them broken on the device and green in CI.
#
# cuDF dispatches a NumPy ufunc over a column on the GPU, so the ufunc branch is not a host
# fallback — it is the device path, and it is the one that was already being exercised.
_MATH_FNS = {
    "abs": "absolute",
    "acos": "arccos",
    "asin": "arcsin",
    "atan": "arctan",
    "cbrt": "cbrt",
    "ceil": "ceil",
    "cos": "cos",
    "cosh": "cosh",
    "degrees": "degrees",
    "exp": "exp",
    "floor": "floor",
    "ln": "log",
    "log10": "log10",
    "log2": "log2",
    "radians": "radians",
    "sin": "sin",
    "sinh": "sinh",
    "sqrt": "sqrt",
    "tan": "tan",
    "tanh": "tanh",
    "trunc": "trunc",
    # IEEE-754 `roundTiesToEven`, which is NumPy's `rint` exactly. Deliberately a different
    # function from `round`, which breaks halves away from zero — the engine ships both and
    # translating one as the other is a half-unit error on precisely the values people round.
    "rint": "rint",
}

#: Reciprocal trigonometric functions, as the function each is one over. Neither backend has
#: any of the three, and the engine computes them exactly this way (`1.0 / v.tan()`), so the
#: reciprocal is the translation rather than an approximation of it — including at the poles,
#: where both sides divide by zero and get an infinity.
_RECIPROCAL_FNS = {"cot": "tan", "sec": "cos", "csc": "sin"}

# `bit_count` and `factorial` are deliberately absent, and are the two integer-only members of
# the engine's math vocabulary. Both need a host-side construction — a SWAR popcount over the
# unsigned reinterpretation, or a table indexed per row — and neither backend dispatches one to
# the device, so translating them would move the column off the GPU to compute something and
# then move it back. `factorial` also has to *raise* above 20! rather than wrap, which is a
# data-dependent decline. Falling back to the CPU engine for both is the cheaper answer.

# Date functions that are a `.dt` attribute of the same name. The ones that are NOT are
# handled explicitly below, because each disagrees with the engine on something: `day_of_week`
# counts from a different day, `week` is the ISO week rather than an attribute, and `epoch` is
# a unit conversion.
_DATE_ATTRS = frozenset({"day", "day_of_year", "days_in_month", "hour",
                         "microsecond", "minute", "month", "quarter", "second",
                         "year"})  # fmt: skip

#: Date functions that are a `.dt` attribute returning a **boolean**, so they must not take the
#: int64 cast every other attribute takes. `is_leap_year` did, and came back as `1`/`0` where
#: the engine returns `true`/`false` — the right answer in the wrong column, which a fan-out
#: cannot concatenate against a CPU-recovered shard and which broke every `CASE` built on it
#: (`days_in_year` lowers to exactly that, and raised rather than returning a number).
_DATE_BOOL_ATTRS = frozenset({"is_leap_year"})


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
    if fn == "abs" and be.is_integer(x):
        # The one unary function whose result keeps an integer input's type. Both libraries'
        # own `.abs()` preserves it; the ufunc path below would widen it to double, which is
        # the right number in the wrong column.
        return x.abs()
    if fn in _RECIPROCAL_FNS:
        return 1.0 / apply_ufunc(_RECIPROCAL_FNS[fn], x, be)
    if fn == "even":
        return _even(x, be)
    ufunc = _MATH_FNS.get(fn)
    if ufunc is None:
        raise Unsupported(f"math fn {fn}")
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


def _even(x, be):
    """`even(x)` — round the magnitude **out** to an even integer, as the engine does.

    Halving and doubling are exact in binary floating point, so ceiling the halved magnitude
    and doubling it introduces no rounding of its own. The sign is reapplied afterwards rather
    than carried through, because ceiling is not symmetric about zero: `ceil(-1.5)` is `-1.0`,
    which would round a negative value *in* while its positive twin rounds out.
    """
    halved = apply_ufunc("absolute", x / 2.0, be)
    magnitude = apply_ufunc("ceil", halved, be) * 2.0
    # `fillna(True)`: a null comparison would take the negated branch, turning a null into a
    # value. The same guard `round_half_away` needs, for the same reason.
    return magnitude.where((x >= 0).fillna(True), -magnitude)


#: Two-argument math functions that are a NumPy ufunc of the same shape, named where the engine
#: and NumPy spell them differently. All four return a double on both sides.
_MATH2_UFUNCS = {"atan2": "arctan2", "hypot": "hypot", "next_after": "nextafter"}

#: Two-argument math functions the engine answers in **integer**, not double. DuckDB returns a
#: BIGINT for both, and the engine follows it: routing them through a double would mistype the
#: column and lose every value above 2^53 (`gcd(2^53+1, 3)` came back as 1.0 rather than 3).
_MATH2_INT_UFUNCS = {"gcd": "gcd", "lcm": "lcm"}


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
        x = be.column(left, df)
        if be.is_integer(x):
            # Rounding an integer leaves it an integer, and the engine says so: DuckDB returns
            # BIGINT for `round(bigint, n)`. Going through the float path returns the right
            # number in the wrong column, which a fan-out cannot concatenate — a shard that
            # fell back to the CPU engine contributes int64 beside this one's double — and
            # which corrupts anything above 2^53 on the way past.
            return _round_int(x, int(right))
        return round_half_away(x, int(right), be)
    if fn in _MATH2_UFUNCS:
        return _binary_ufunc(_MATH2_UFUNCS[fn], left, right, df, be)
    if fn in _MATH2_INT_UFUNCS:
        return _binary_ufunc(_MATH2_INT_UFUNCS[fn], left, right, df, be, integer=True)
    raise Unsupported(f"math2 fn {fn}")


def _round_int(x, digits: int):
    """`round(integer, digits)`, staying in int64 the whole way.

    A non-negative `digits` cannot change an integer, so it is the identity. A negative one
    rounds to a power of ten, with halves going away from zero the way `round` does everywhere
    else here — computed on the integers themselves, so a value past 2^53 survives it.
    """
    if digits >= 0:
        return x
    scale = 10 ** (-digits)
    half = scale // 2
    # Away from zero on a tie means adding half the scale *toward* the value's own sign before
    # truncating toward zero. Floor division alone would round a negative value the wrong way.
    shifted = x.abs() + half
    magnitude = (shifted // scale) * scale
    return magnitude.where((x >= 0).fillna(True), -magnitude)


def _binary_ufunc(name: str, left, right, df, be, *, integer: bool = False):
    """Apply NumPy's two-argument `name` to `left` and `right`, keeping both null masks.

    The unary `apply_ufunc` cannot be reused: it restores *one* input's mask, and here a null
    on either side makes the result null. The engine unions the two null buffers, so a row is
    null exactly when it is null on either side.
    """
    import numpy as np

    fn = getattr(np, name, None)
    if fn is None:
        raise Unsupported(f"math2 fn {name}")
    lc = be.column(left, df)
    rc = be.column(right, df)
    present = lc.notna() & rc.notna()
    try:
        if be.is_gpu:
            return fn(lc, rc).where(present, None)
        dtype = "int64" if integer else "float64"
        # `gcd`/`lcm` have no null-bearing form at all: NumPy's integer ufuncs cannot take a
        # `NaN` filler the way the float ones can, so the holes are filled with a value that
        # cannot fail (`0`, whose gcd is defined) and masked back out afterwards.
        fill = 0 if integer else np.nan
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            raw = fn(
                lc.to_numpy(dtype=dtype, na_value=fill), rc.to_numpy(dtype=dtype, na_value=fill)
            )
    except (TypeError, AttributeError, NotImplementedError, ValueError) as exc:
        raise Unsupported(f"math2 fn {name}: {exc}") from exc
    out = be.series(raw, dtype=be.dtype(_int64())) if integer else be.float_series(raw)
    out.index = lc.index
    return out.where(present, None)


def eval_date(ir, df, be, eval_expr):
    fn = ir["fn"]
    x = be.column(eval_expr(ir["input"], df, be), df)
    if fn in _DATE_ATTRS:
        return getattr(x.dt, fn).astype(be.dtype(_int64()))
    if fn in _DATE_BOOL_ATTRS:
        import pyarrow as pa

        return getattr(x.dt, fn).astype(be.dtype(pa.bool_())).where(x.notna(), None)
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
        # Whole seconds since the Unix epoch, from the microseconds Arrow stores. The cast to
        # `timestamp[us]` is what lets a DATE take this path: a date's own representation is a
        # count of *days*, which has no direct cast to int64 at all, so this used to raise and
        # send the whole plan to the CPU engine over an `epoch()` on an ordinary date column.
        # Flooring (not truncating) is the engine's rule for an instant before 1970.
        return (epoch_micros(x, be) // 1_000_000).astype(be.dtype(_int64()))
    if fn in DATE_FNS:
        return eval_calendar_date(x, fn, be)
    raise Unsupported(f"date fn {fn}")


#: `date_trunc` units that are a fixed duration, so truncating is flooring to a multiple of it
#: and the libraries' own `dt.floor` expresses it. The calendar units — week, month, quarter,
#: year — vary in length and are calendar arithmetic instead, handled in `temporal`.
_TRUNC_FREQ = {"second": "s", "minute": "min", "hour": "h", "day": "D"}


def eval_date_trunc(ir, df, be, eval_expr):
    """`date_trunc(unit, ts)`, by flooring for the fixed-duration units and by calendar
    arithmetic for the rest."""
    unit = ir.get("unit")
    freq = _TRUNC_FREQ.get(unit)
    x = be.column(eval_expr(ir["input"], df, be), df)
    if freq is None:
        # A `GROUP BY date_trunc('month', ...)` is the most ordinary shape a time series has,
        # and it used to send the entire plan to the CPU engine over this one call.
        return eval_calendar_trunc(x, unit, be)
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
    return isocalendar_field(x, be, "week", "date fn week")
