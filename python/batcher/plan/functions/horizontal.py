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
from collections.abc import Callable

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
    """Row-wise sum across the given columns, treating nulls as 0 (Polars ``sum_horizontal``).

    Complements `greatest`/`least` (row-wise max/min). An all-null row sums to 0, as in
    ``sum_horizontal(col("a"), col("b"), col("c"))``.

    Args:
        exprs: The columns to add together element-wise.

    Returns:
        A column holding the per-row sum across the given columns.

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
    """Row-wise mean across the given columns, ignoring nulls (Polars ``mean_horizontal``).

    The sum of the non-null values divided by how many were non-null. An all-null row
    yields null (no division by zero).

    Args:
        exprs: The columns to average element-wise.

    Returns:
        A column holding the per-row mean across the non-null values.

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
    """Row-wise minimum across the given columns, ignoring nulls (Polars ``min_horizontal``).

    The Polars-named spelling of `least`, completing the horizontal family alongside
    `sum_horizontal`/`mean_horizontal`; an all-null row yields null.

    Args:
        exprs: The columns to take the row-wise minimum of.

    Returns:
        A column holding the per-row minimum across the given columns.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1, 9], "b": [4, 2]})
            >>> ds.select(lo=bt.min_horizontal(bt.col("a"), bt.col("b"))).to_pydict()
            {'lo': [1, 2]}
    """
    return least(*exprs)


def max_horizontal(*exprs: IntoExpr) -> Greatest:
    """Row-wise maximum across the given columns, ignoring nulls (Polars ``max_horizontal``).

    The Polars-named spelling of `greatest`; an all-null row yields null.

    Args:
        exprs: The columns to take the row-wise maximum of.

    Returns:
        A column holding the per-row maximum across the given columns.

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

    Args:
        exprs: The boolean columns to AND together element-wise.

    Returns:
        A boolean column true only where every argument is true.

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

    Args:
        exprs: The boolean columns to OR together element-wise.

    Returns:
        A boolean column true where any argument is true.

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


def count_horizontal(*exprs: IntoExpr) -> Expr:
    """Row-wise count of non-null values across the given columns (Polars ``count_horizontal``).

    The horizontal companion to :func:`sum_horizontal`: how many of the arguments are
    non-null in each row. Composes as a sum of null-indicators, so an all-null row
    counts 0 (never null).

    Args:
        exprs: The columns whose per-row non-null values are counted.

    Returns:
        An Int64 column holding the per-row count of non-null arguments.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1, None], "b": [10, 20], "c": [None, 3]})
            >>> ds.select(n=bt.count_horizontal(bt.col("a"), bt.col("b"), bt.col("c"))).to_pydict()
            {'n': [2, 2]}
    """
    if not exprs:
        raise ValueError("count_horizontal() requires at least one argument")
    return functools.reduce(operator.add, [_wrap(e).is_not_null().cast("int64") for e in exprs])


def product_horizontal(*exprs: IntoExpr) -> Expr:
    """Row-wise product across the given columns, treating nulls as 1 (Polars-style).

    The multiplicative counterpart to :func:`sum_horizontal`: an all-null row yields 1
    (the empty product), and a single null factor is skipped rather than nulling the row.

    Args:
        exprs: The columns to multiply together element-wise.

    Returns:
        A column holding the per-row product across the non-null values.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [2, None], "b": [10, 20]})
            >>> ds.select(p=bt.product_horizontal(bt.col("a"), bt.col("b"))).to_pydict()
            {'p': [20, 20]}
    """
    if not exprs:
        raise ValueError("product_horizontal() requires at least one argument")
    parts = [coalesce(_wrap(e), lit(1)) for e in exprs]
    return functools.reduce(operator.mul, parts)


def reduce_horizontal(function: Callable[[Expr, Expr], Expr], *exprs: IntoExpr) -> Expr:
    """Reduce the given columns row-wise with a binary expression `function` (Polars ``reduce``).

    Left-folds `function` across the columns with no seed: the first column is the
    initial accumulator, then ``acc = function(acc, next)`` for each remaining column.
    `function` runs **once at plan-build time** on `Expr` operands to assemble the
    expression tree — it never sees a row — so any expression-valued combiner works
    (``lambda a, b: a + b``, ``lambda a, b: bt.max_horizontal(a, b)``, …). Use it for a
    horizontal reduction the named ``*_horizontal`` helpers don't cover.

    Args:
        function: A binary combiner mapping two `Expr` operands to one `Expr`.
        exprs: The columns to reduce, left to right.

    Returns:
        A column holding the per-row reduction.

    Raises:
        ValueError: If no columns are given.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1, 2], "b": [10, 20], "c": [100, 200]})
            >>> cols = [bt.col("a"), bt.col("b"), bt.col("c")]
            >>> ds.select(r=bt.reduce_horizontal(lambda x, y: x + y, *cols)).to_pydict()
            {'r': [111, 222]}
    """
    if not exprs:
        raise ValueError("reduce_horizontal() requires at least one argument")
    return functools.reduce(function, [_wrap(e) for e in exprs])


def fold_horizontal(
    acc: IntoExpr, function: Callable[[Expr, Expr], Expr], *exprs: IntoExpr
) -> Expr:
    """Fold the given columns row-wise with `function`, seeded by `acc` (Polars ``fold``).

    Like :func:`reduce_horizontal` but with an explicit initial accumulator `acc`, so a
    fold over zero columns is still well-defined and the accumulator can seed a running
    computation (``fold_horizontal(bt.lit(0), lambda a, b: a + b * b, *cols)`` for a sum
    of squares). `function` is applied at plan-build time on `Expr` operands, never per
    row.

    Args:
        acc: The initial accumulator expression.
        function: A binary combiner mapping ``(acc, next)`` to a new `Expr`.
        exprs: The columns to fold in, left to right.

    Returns:
        A column holding the per-row folded value.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> d = bt.from_pydict({"a": [1, 2], "b": [3, 4]})
            >>> c = [bt.col("a"), bt.col("b")]
            >>> d.select(r=bt.fold_horizontal(bt.lit(0), lambda s, x: s + x * x, *c)).to_pydict()
            {'r': [10, 20]}
    """
    result = _wrap(acc)
    for e in exprs:
        result = function(result, _wrap(e))
    return result
