"""Tighten a filtered column's bounds to the values its own predicate admits.

A filter's output keeps each column's `min`/`max` as bounds, which is sound but loose for the
column the predicate constrains: after `d_year IN (2000, 2001, 2002, 2003)` every surviving
value lies in `[2000, 2003]`, yet the bounds still read the table's `[1900, 2100]`. That is not
merely imprecise. The next conjunct on the same column is then priced against the whole range
again, so a redundant `d_year <= 2003` (the planner derives such implied predicates) cuts the
estimate a second time for rows it removes none of. TPC-DS q23's date side stacked three such
cuts, estimated 41 dates where 1,461 survive, and a pushdown gated on that estimate grouped
2.3M fact rows into nearly as many groups.

Unlike `comonotone`, which infers a position under a uniformity assumption and so leaves the
bounds alone, this is a proof: a row the predicate keeps satisfies it. Layer: `kyber/stats`.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from batcher.plan.expr_ir import Binary, Col, Expr, InList, Lit
from batcher.plan.expr_rewrite import split_conjuncts
from batcher.plan.stats import ColumnStat

__all__ = ["bounded_by_predicate", "without_implied_ranges"]

_FLIP = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le", "eq": "eq"}


def bounded_by_predicate(stat: ColumnStat, predicate: Expr) -> ColumnStat:
    """`stat` with `min`/`max` tightened to what the single-column `predicate` admits.

    Only literals of exactly the bound's own Python type are used, so an `int` literal never
    becomes a `float` column's bound and a string never a date's. A conjunct the helper cannot
    read (`!=`, a function of the column, an `OR`) simply contributes nothing, which keeps the
    result a sound bound: it is the intersection of intervals each conjunct provably implies.

    Args:
        stat: The column's statistics on the filter's output.
        predicate: The filter's conjuncts on this column alone.

    Returns:
        `stat`, with narrower bounds where the predicate proves them.
    """
    if stat.min is None or stat.max is None or isinstance(stat.min, bool):
        return stat
    kind = type(stat.min)
    if type(stat.max) is not kind:
        return stat
    lo, hi = stat.min, stat.max
    for conjunct in split_conjuncts(predicate):
        interval = _interval(conjunct, kind)
        if interval is None:
            continue
        low, high = interval
        if low is not None and low > lo:
            lo = low
        if high is not None and high < hi:
            hi = high
    if lo > hi or (lo == stat.min and hi == stat.max):
        return stat  # a contradiction is `_join_keys_range_disjoint`'s business, not a bound
    return dataclasses.replace(stat, min=lo, max=hi)


def _interval(conjunct: Expr, kind: type) -> tuple[Any, Any] | None:
    """`(low, high)` a conjunct implies for its column (either may be None), or None."""
    if isinstance(conjunct, InList):
        if not isinstance(conjunct.input, Col) or not conjunct.values:
            return None
        if any(type(v) is not kind for v in conjunct.values):
            return None
        return min(conjunct.values), max(conjunct.values)
    if not isinstance(conjunct, Binary) or conjunct.op not in _FLIP:
        return None
    side = _column_and_literal(conjunct)
    if side is None:
        return None
    _, value, col_on_left = side
    if type(value) is not kind:
        return None
    op = conjunct.op if col_on_left else _FLIP[conjunct.op]
    if op == "eq":
        return value, value
    return (value, None) if op in ("gt", "ge") else (None, value)


def _column_and_literal(expr: Binary) -> tuple[str, Any, bool] | None:
    """`(column, literal value, column_on_left)` for `col OP lit` / `lit OP col`, else None."""
    if isinstance(expr.left, Col) and isinstance(expr.right, Lit):
        return expr.left.name, expr.right.value, True
    if isinstance(expr.right, Col) and isinstance(expr.left, Lit):
        return expr.right.name, expr.left.value, False
    return None


def without_implied_ranges(conjuncts: list[Expr]) -> list[Expr]:
    """`conjuncts` minus each range comparison a same-column `IN`/`=` already implies.

    `d_year IN (2000, 2001, 2002, 2003) AND d_year >= 2000 AND d_year <= 2003` keeps exactly the
    rows the `IN` keeps, but a product of three selectivities prices the two ranges against
    the column's whole `[1900, 2100]` span and cuts the estimate by a further 4x. The planner
    derives such range conjuncts from an `IN` (for zone-map pruning), so the shape is routine.

    Args:
        conjuncts: One `AND`'s conjuncts.

    Returns:
        The conjuncts that constrain anything the point sets beside them do not.
    """
    points: dict[str, tuple[Any, Any]] = {}
    kinds: dict[str, type] = {}
    mixed: set[str] = set()
    for c in conjuncts:
        name = _point_set_column(c)
        if name is None:
            continue
        kind = _value_type(c)
        # Two point sets over one column with different literal types (`x = 1 AND x = '1'`)
        # have no common order, so neither can bound the other; drop the column rather than
        # compare an int with a str.
        if kinds.setdefault(name, kind) is not kind:
            mixed.add(name)
            continue
        low, high = _interval(c, kind)
        prior = points.get(name)
        points[name] = (low, high) if prior is None else (max(prior[0], low), min(prior[1], high))
    for name in mixed:
        points.pop(name, None)
    if not points:
        return conjuncts
    kept = []
    for c in conjuncts:
        if _point_set_column(c) is None and _implied(c, points):
            continue
        kept.append(c)
    return kept


def _point_set_column(c: Expr) -> str | None:
    """The column of an `IN` list or a `col = lit`, the conjuncts that pin a value set."""
    if isinstance(c, InList) and isinstance(c.input, Col) and c.values:
        kind = type(c.values[0])
        if kind is not bool and all(type(v) is kind for v in c.values):
            return c.input.name
        return None
    if isinstance(c, Binary) and c.op == "eq":
        side = _column_and_literal(c)
        if side is not None and side[1] is not None and not isinstance(side[1], bool):
            return side[0]
    return None


def _value_type(c: Expr) -> type:
    if isinstance(c, InList):
        return type(c.values[0])
    side = _column_and_literal(c)
    return type(side[1]) if side is not None else type(None)


def _implied(c: Expr, points: dict[str, tuple[Any, Any]]) -> bool:
    """Whether range comparison `c` holds for every value its column's point set allows."""
    if not isinstance(c, Binary) or c.op not in ("lt", "le", "gt", "ge"):
        return False
    side = _column_and_literal(c)
    if side is None or side[0] not in points:
        return False
    name, value, col_on_left = side
    low, high = points[name]
    if type(value) is not type(low):
        return False
    op = c.op if col_on_left else _FLIP[c.op]
    if op == "ge":
        return low >= value
    if op == "gt":
        return low > value
    if op == "le":
        return high <= value
    return high < value
