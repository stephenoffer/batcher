"""Quantile, cardinality, and histogram aggregate shorthands.

The SQL-style top-level spellings of the distribution aggregates — the same shorthand
`sum('x')` is for `col('x').sum()`. `quantile` is exact; `approx_quantile` / `approx_median`
(DDSketch/KLL) and `approx_n_unique` (HyperLogLog) trade a little accuracy for bounded
memory, and `histogram` builds a value→count map. The sketch-backed ones are mergeable, so
the estimate is identical single-node and distributed.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import AggExpr, Expr
from batcher.plan.functions.aggregate import _as_column


def quantile(column: str | Expr, q: float) -> AggExpr:
    """Exact ``q``-quantile of a column (``q`` in ``[0, 1]``; ``0.5`` is the median).

    Args:
        column: The column to summarize, as a name or an expression.
        q: The quantile to compute, between 0 and 1.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "a"], "x": [10.0, 20.0, 30.0]})
            >>> ds.group_by("g").agg(q=bt.quantile("x", 0.5)).to_pydict()
            {'g': ['a'], 'q': [20.0]}
    """
    return _as_column(column).quantile(q)


def approx_quantile(column: str | Expr, q: float) -> AggExpr:
    """Approximate ``q``-quantile via a mergeable sketch (DuckDB ``approx_quantile``).

    Args:
        column: The column to summarize, as a name or an expression.
        q: The quantile to estimate, between 0 and 1.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "b"], "x": [10.0, 20.0]})
            >>> r = ds.group_by("g").agg(q=bt.approx_quantile("x", 0.5)).sort("g")
            >>> r.with_columns(q=bt.col("q").round()).to_pydict()
            {'g': ['a', 'b'], 'q': [10.0, 20.0]}
    """
    return _as_column(column).approx_quantile(q)


def approx_median(column: str | Expr) -> AggExpr:
    """Approximate median (the 0.5 quantile) via a mergeable sketch — bounded memory.

    Args:
        column: The column to summarize, as a name or an expression.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "b"], "x": [10.0, 20.0]})
            >>> r = ds.group_by("g").agg(m=bt.approx_median("x")).sort("g")
            >>> r.with_columns(m=bt.col("m").round()).to_pydict()
            {'g': ['a', 'b'], 'm': [10.0, 20.0]}
    """
    return _as_column(column).approx_median()


def approx_n_unique(column: str | Expr) -> AggExpr:
    """Approximate distinct count via a HyperLogLog sketch — bounded memory (~2% error).

    Args:
        column: The column to summarize, as a name or an expression.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [5, 5, 9]})
            >>> ds.group_by("g").agg(n=bt.approx_n_unique("x")).sort("g").to_pydict()
            {'g': ['a', 'b'], 'n': [1, 1]}
    """
    return _as_column(column).approx_n_unique()


def histogram(column: str | Expr) -> AggExpr:
    """Value → count map of a column's values in each group (DuckDB ``histogram``).

    Args:
        column: The column to tally, as a name or an expression.

    Returns:
        An aggregate expression producing a `Map` column; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, 1, 2]})
            >>> ds.agg(h=bt.histogram("x")).to_pydict()
            {'h': [[(1, 2), (2, 1)]]}
    """
    return _as_column(column).histogram()
