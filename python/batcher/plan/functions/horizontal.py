"""Row-wise ("horizontal") reductions across several columns.

Where an aggregate collapses a column *down* to one value per group, these fold
*across* columns within each row: ``sum_horizontal(col("a"), col("b"))`` adds the
columns element-wise. The family mirrors the Polars ``*_horizontal`` functions and
composes existing `Expr` nodes — no new IR. The row-wise max/min also carry the SQL
names `greatest`/`least`; both spellings are kept for discoverability.
"""

from __future__ import annotations

import functools
import operator

from batcher.plan.expr_ir import (
    Expr,
    IntoExpr,
    coalesce,
    greatest,
    least,
    lit,
    nullif,
)
from batcher.plan.expr_ir.core import _wrap
from batcher.plan.expr_ir.nodes import Greatest, Least


def sum_horizontal(*exprs: IntoExpr) -> Expr:
    """Row-wise sum across the given columns, treating nulls as 0 (Polars
    ``sum_horizontal``). Complements `greatest`/`least` (row-wise max/min). An
    all-null row sums to 0. ``sum_horizontal(col("a"), col("b"), col("c"))``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1, None], "b": [10, 20]})
            >>> ds.select(s=bt.sum_horizontal(bt.col("a"), bt.col("b"))).to_pydict()
            {'s': [11, 20]}
    """
    if not exprs:
        raise ValueError("sum_horizontal() requires at least one argument")
    parts = [coalesce(_wrap(e), lit(0)) for e in exprs]
    return functools.reduce(operator.add, parts)


def mean_horizontal(*exprs: IntoExpr) -> Expr:
    """Row-wise mean across the given columns, ignoring nulls (Polars
    ``mean_horizontal``): the sum of the non-null values divided by how many were
    non-null. An all-null row yields null (no division by zero).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1.0, None], "b": [3.0, 20.0]})
            >>> ds.select(m=bt.mean_horizontal(bt.col("a"), bt.col("b"))).to_pydict()
            {'m': [2.0, 20.0]}
    """
    if not exprs:
        raise ValueError("mean_horizontal() requires at least one argument")
    wrapped = [_wrap(e) for e in exprs]
    total = functools.reduce(operator.add, [coalesce(e, lit(0)) for e in wrapped])
    count = functools.reduce(operator.add, [e.is_not_null().cast("int64") for e in wrapped])
    # Divide by NULLIF(count, 0): an all-null row has count 0 → null (no div-by-zero).
    return total / nullif(count, lit(0))


def min_horizontal(*exprs: IntoExpr) -> Least:
    """Row-wise minimum across the given columns, ignoring nulls (Polars
    ``min_horizontal``). The Polars-named spelling of `least`, completing the
    horizontal family alongside `sum_horizontal`/`mean_horizontal`; an all-null row
    yields null.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1, 9], "b": [4, 2]})
            >>> ds.select(lo=bt.min_horizontal(bt.col("a"), bt.col("b"))).to_pydict()
            {'lo': [1, 2]}
    """
    return least(*exprs)


def max_horizontal(*exprs: IntoExpr) -> Greatest:
    """Row-wise maximum across the given columns, ignoring nulls (Polars
    ``max_horizontal``). The Polars-named spelling of `greatest`; an all-null row
    yields null.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1, 9], "b": [4, 2]})
            >>> ds.select(hi=bt.max_horizontal(bt.col("a"), bt.col("b"))).to_pydict()
            {'hi': [4, 9]}
    """
    return greatest(*exprs)


def all_horizontal(*exprs: IntoExpr) -> Expr:
    """Row-wise boolean AND across the given predicate columns (Polars ``all_horizontal``).

    The horizontal counterpart to a chain of ``&`` — true only where every column is
    true. Follows SQL three-valued logic (a null makes the row null unless another
    column is false). The idiomatic way to combine many validation flags into one.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [True, True], "b": [True, False]})
            >>> ds.select(ok=bt.all_horizontal(bt.col("a"), bt.col("b"))).to_pydict()
            {'ok': [True, False]}
    """
    if not exprs:
        raise ValueError("all_horizontal() requires at least one argument")
    return functools.reduce(operator.and_, [_wrap(e) for e in exprs])


def any_horizontal(*exprs: IntoExpr) -> Expr:
    """Row-wise boolean OR across the given predicate columns (Polars ``any_horizontal``).

    The horizontal counterpart to a chain of ``|`` — true where any column is true.
    Follows SQL three-valued logic. The idiomatic way to test whether any of several
    flags is set.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [False, False], "b": [True, False]})
            >>> ds.select(hit=bt.any_horizontal(bt.col("a"), bt.col("b"))).to_pydict()
            {'hit': [True, False]}
    """
    if not exprs:
        raise ValueError("any_horizontal() requires at least one argument")
    return functools.reduce(operator.or_, [_wrap(e) for e in exprs])
