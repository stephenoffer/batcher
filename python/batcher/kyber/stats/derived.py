"""Bounds through a monotonic arithmetic projection — the one *non-constant* computed
column whose distribution the input still determines.

`project_columns` carries a plain `col(x)` through and drops anything computed, because the
output distribution of a general expression is unknown. A **monotonic** transform of a
single column is the exception: `x + c`, `x - c`, `c - x`, `x * c`, and `-x` map the input's
`[min, max]` interval onto a known output interval (order-preserving for a positive scale,
reversed for a negative one), and preserve the distinct count (they are injective for a
non-zero constant). So a `SELECT x + 10 AS y` followed by `WHERE y > 100` can still
interpolate a range selectivity from `x`'s footer bounds instead of falling to the blunt
constant.

Kept a `DEFAULT`-provenance estimate, never `EXACT`: integer arithmetic can overflow at the
extreme and float arithmetic rounds, so the derived interval must inform cost and pruning
without ever answering an exact `min`/`max`/`count` from metadata.

The *shape* statistics transform too, and exactly. A strictly monotonic `g` satisfies
`F_y(g(x)) = F_x(x)`, so a quantile grid maps through it value-by-value (reversed, with
`probs` complemented, when `g` decreases); and `g` is injective, so every value's frequency
is carried unchanged to its image. Dropping them meant `SELECT x * 100 AS cents ... WHERE
cents BETWEEN ...` fell from histogram interpolation to a flat constant on a column whose
distribution was fully known — which is the common shape, since a derived column exists to
be filtered on.
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal

from batcher.plan.expr_ir import Binary, Col, Expr, Greatest, Least, Lit, NullIf
from batcher.plan.stats import ColumnStat, Provenance, RelStats

__all__ = ["derived_projection_stat"]


def _minmax_bounds(inputs: list[Expr], child: RelStats, *, greatest: bool) -> ColumnStat | None:
    """Bounding-box bounds for `greatest(...)`/`least(...)` over its argument columns.

    Every output value equals one of the (non-null) inputs, and *which* one is known: a
    `least` is at most the smallest of the arguments' maxima, and a `greatest` is at least
    the largest of their minima. So the bounds are

    * `least`    -> `[min(all mins), min(all maxes)]`
    * `greatest` -> `[max(all mins), max(all maxes)]`

    and only one end of each comes from the union box the previous version used for both. The
    difference is not cosmetic: `least(price, 100)` over a price column spanning `[0, 10^6]`
    is bounded above by **100**, not by a million, and a downstream `WHERE capped > 500` is
    provably empty rather than estimated at a third of the table.

    Sound under nulls, which SQL ignores here: the result is one of the non-null arguments,
    so it lies inside the bounds of the arguments that were present. Requires every argument
    to be a bounded column or a numeric literal; anything else yields None.
    """
    mins: list[float] = []
    maxs: list[float] = []
    for arg in inputs:
        if isinstance(arg, Lit):
            v = _numeric(arg.value)
            if v is None:
                return None
            mins.append(float(v))
            maxs.append(float(v))
        elif isinstance(arg, Col):
            src = child.columns.get(arg.name)
            lo = _numeric(src.min) if src is not None else None
            hi = _numeric(src.max) if src is not None else None
            if lo is None or hi is None:
                return None
            mins.append(float(lo))
            maxs.append(float(hi))
        else:
            return None
    if not mins:
        return None
    if greatest:
        return ColumnStat(min=max(mins), max=max(maxs), provenance=Provenance.DEFAULT)
    return ColumnStat(min=min(mins), max=min(maxs), provenance=Provenance.DEFAULT)


def _numeric(value: object) -> float | int | Decimal | None:
    """`value` if it is a real number usable as an arithmetic bound, else None.

    `bool` is excluded (it is an `int` subclass but not a quantity here); `date`/`datetime`
    are excluded because a `Binary` add over them is not day arithmetic (that is a
    `DateOffset` node), so their bounds must not be shifted as if they were numbers.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        return value
    return None


def _commensurable(*values: float | int | Decimal) -> tuple:
    """`values` widened so arithmetic between any two of them is defined.

    Python refuses `Decimal + float` (and every other mixed operator) rather than coercing,
    so a DECIMAL column -- whose footer bounds arrive as `Decimal` -- against a float literal
    raises `TypeError` out of the arithmetic below. That is not a hypothetical shape: it is
    `l_quantity * 0.2`, and it made TPC-H q17 and TPC-DS q32/q92 fail to *plan*.

    Widening to float is the right resolution and not merely the convenient one. These bounds
    carry `Provenance.DEFAULT` -- they are an estimate feeding a selectivity interpolation, and
    nothing downstream reads them as an exact decimal. `Decimal` against `int` is exact in
    Python and is deliberately left alone, so the common integer case keeps its precision.

    Args:
        values: The numeric bounds and constant about to be combined.

    Returns:
        The values unchanged, or all of them as floats when the mix would not combine.
    """
    if any(isinstance(v, Decimal) for v in values) and any(isinstance(v, float) for v in values):
        return tuple(float(v) for v in values)
    return values


def derived_projection_stat(expr: Expr, child: RelStats) -> ColumnStat | None:
    """The output `ColumnStat` of a monotonic `col OP literal` projection, or None.

    Handles `x + c`, `c + x`, `x - c`, `c - x`, `x * c`, `c * x`, and `-x` (lowered as
    `0 - x`). Requires the column to carry both numeric bounds; the transform maps them onto
    the output interval, reversing the order for a negative scale, and carries the distinct
    count and null count (a non-null constant makes the result null exactly where the column
    is). Returns None for anything else, leaving the column unknown as before.
    """
    # `NULLIF(x, c)` (cleaning a sentinel value to NULL) keeps a *subset* of `x`'s values, so
    # `x`'s bounds and distinct count carry through — only nulls are added. A common data-cleaning
    # shape (`NULLIF(amount, -999)`) whose derived column is then range-filtered downstream.
    if isinstance(expr, NullIf) and isinstance(expr.left, Col):
        src = child.columns.get(expr.left.name)
        if src is not None:
            # A subset of `x`'s values, so every descriptive stat stays a valid (loose)
            # superset — carry them downgraded; only the null count is no longer known.
            return dataclasses.replace(src.downgrade(Provenance.DEFAULT), null_count=None)
        return None
    if isinstance(expr, (Greatest, Least)):
        return _minmax_bounds(expr.inputs, child, greatest=isinstance(expr, Greatest))
    if not isinstance(expr, Binary) or expr.op not in ("add", "sub", "mul"):
        return None
    col, lit, col_on_left = _column_and_literal(expr)
    if col is None:
        return None
    src = child.columns.get(col.name)
    if src is None:
        return None
    lo, hi = _numeric(src.min), _numeric(src.max)
    c = _numeric(lit.value)
    if lo is None or hi is None or c is None:
        return None
    lo, hi, c = _commensurable(lo, hi, c)
    bounds = _transform_bounds(expr.op, lo, hi, c, col_on_left)
    if bounds is None:
        return None
    new_min, new_max = bounds
    scale = _scale_of(expr.op, c, col_on_left)
    return ColumnStat(
        min=new_min,
        max=new_max,
        # A non-zero-scale transform is injective, so the distinct count survives. A `* 0`
        # is a constant and is handled by the constant folder, not here.
        ndv=src.ndv,
        null_count=src.null_count,
        provenance=Provenance.DEFAULT,
        quantiles=_transform_quantiles(src.quantiles, expr.op, c, col_on_left),
        mcv=_transform_mcv(src.mcv, expr.op, c, col_on_left),
        # An affine map of a number does not change how many bytes it occupies.
        avg_bytes=src.avg_bytes,
        # `mean(a·x + b) = a·mean(x) + b`, exactly. The total sum needs the row count to
        # shift, which is not carried here, so it is left unknown.
        # `float(src.mean)` rather than `src.mean`: `scale` is a float pair, and a DECIMAL
        # column's mean arrives as a `Decimal`, which Python refuses to multiply by a float.
        mean=None if src.mean is None or scale is None else scale[0] * float(src.mean) + scale[1],
    )


def _scale_of(op: str, c, col_on_left: bool) -> tuple[float, float] | None:
    """`(a, b)` for the affine map `y = a·x + b` this projection applies, or None."""
    if op == "add":
        return 1.0, float(c)
    if op == "sub":
        return (1.0, -float(c)) if col_on_left else (-1.0, float(c))
    if op == "mul":
        return (float(c), 0.0) if c != 0 else None
    return None


def _apply(scale: tuple[float, float], x: float) -> float:
    return scale[0] * x + scale[1]


def _transform_quantiles(grid, op: str, c, col_on_left: bool):
    """Map a quantile grid through the affine projection, or None when it cannot be.

    Exact for a strictly monotonic map: `P(y <= g(x)) = P(x <= x)` when `g` increases, so the
    boundary *values* move and the probabilities do not. A decreasing `g` reverses the order,
    which means reversing the value list and complementing the probabilities — the grid the
    interpolator expects is ascending in both.
    """
    scale = _scale_of(op, c, col_on_left)
    if not grid or scale is None:
        return None
    values = grid.get("values") or []
    probs = grid.get("probs") or []
    if len(values) != len(probs) or len(values) < 2:
        return None
    try:
        mapped = [_apply(scale, float(v)) for v in values]
    except (TypeError, ValueError):  # pragma: no cover - a non-numeric grid
        return None
    if scale[0] > 0:
        return {"probs": [float(p) for p in probs], "values": mapped}
    return {"probs": [1.0 - float(p) for p in reversed(probs)], "values": list(reversed(mapped))}


def _transform_mcv(mcv, op: str, c, col_on_left: bool):
    """Carry each most-common value's frequency to its image under the projection.

    The map is injective, so no two values collide and every frequency transfers unchanged.
    The table is keyed by the value's string form, so the image is re-rendered the same way —
    which is what lets a downstream `WHERE derived = k` still find the measured skew instead
    of falling back to a uniform `1/ndv`.
    """
    scale = _scale_of(op, c, col_on_left)
    if not mcv or scale is None:
        return None
    out: dict[str, float] = {}
    for key, freq in mcv.items():
        try:
            out[str(_apply(scale, float(key)))] = float(freq)
        except (TypeError, ValueError):
            return None  # a non-numeric key: the whole table is untransformable
    return out


def _column_and_literal(expr: Binary) -> tuple[Col | None, Lit, bool]:
    """`(col, literal, col_on_left)` for `col OP literal` / `literal OP col`, else `(None, …)`."""
    if isinstance(expr.left, Col) and isinstance(expr.right, Lit):
        return expr.left, expr.right, True
    if isinstance(expr.right, Col) and isinstance(expr.left, Lit):
        return expr.right, expr.left, False
    return None, Lit(None), True


def _transform_bounds(op: str, lo, hi, c, col_on_left: bool):
    """Map `[lo, hi]` through `col OP c` (or `c OP col`), or None for a non-monotone case."""
    if op == "add":
        return lo + c, hi + c
    if op == "sub":
        if col_on_left:  # x - c: shift down, order preserved
            return lo - c, hi - c
        return c - hi, c - lo  # c - x: order reversed
    # mul: a positive scale preserves order, a negative one reverses it; a zero scale is a
    # constant (handled by the folder), so it is left to that path.
    if c > 0:
        return lo * c, hi * c
    if c < 0:
        return hi * c, lo * c
    return None
