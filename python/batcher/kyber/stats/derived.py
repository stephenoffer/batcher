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
without ever answering an exact `min`/`max`/`count` from metadata. Quantiles and MCVs are
dropped rather than transformed — bounds and ndv are the sound, useful part.
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal

from batcher.plan.expr_ir import Binary, Col, Expr, Greatest, Least, Lit, NullIf
from batcher.plan.stats import ColumnStat, Provenance, RelStats

__all__ = ["derived_projection_stat"]


def _minmax_bounds(inputs: list[Expr], child: RelStats) -> ColumnStat | None:
    """Bounding-box bounds for `greatest(...)`/`least(...)` over its argument columns.

    Every output value equals one of the (non-null) inputs, so it lies in the union of the
    input ranges: `[min(all mins), max(all maxes)]`. That box is a valid superset for both
    `greatest` and `least` regardless of nulls (SQL ignores them), so a downstream range
    filter on the derived column stays sharp. Requires every argument to be a bounded column
    or a numeric literal; anything else (a nested expression with no bounds) yields None.
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
    return ColumnStat(min=min(mins), max=max(maxs), provenance=Provenance.DEFAULT)


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
        return _minmax_bounds(expr.inputs, child)
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
    bounds = _transform_bounds(expr.op, lo, hi, c, col_on_left)
    if bounds is None:
        return None
    new_min, new_max = bounds
    return ColumnStat(
        min=new_min,
        max=new_max,
        # A non-zero-scale transform is injective, so the distinct count survives. A `* 0`
        # is a constant and is handled by the constant folder, not here.
        ndv=src.ndv,
        null_count=src.null_count,
        provenance=Provenance.DEFAULT,
    )


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
