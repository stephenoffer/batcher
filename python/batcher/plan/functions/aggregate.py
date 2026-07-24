"""Aggregate free functions that compose existing mergeable aggregates.

`count_if` desugars to ``sum(iff(cond, 1, 0))`` — counting the rows where a
predicate holds reuses the mergeable `sum` aggregate, so it stays identical
single-node and distributed with no new engine state.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import AggExpr, Expr, IntoExpr, Lit
from batcher.plan.functions.scalar import iff


def corr(x: IntoExpr, y: IntoExpr) -> AggExpr:
    """Pearson correlation coefficient of two columns (DuckDB/Spark ``corr``).

    Mergeable (6-column sum-of-powers state), so identical single-node and
    distributed. Null when a group has fewer than 2 paired values or either column
    is constant. Symmetric in `x` and `y`.

    Args:
        x: The first column.
        y: The second column.

    Returns:
        The Pearson correlation coefficient of the two columns per group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 3.0], "y": [2.0, 6.0]})
            >>> ds.agg(c=bt.corr(bt.col("x"), bt.col("y"))).to_pydict()
            {'c': [1.0]}
    """
    return AggExpr("corr", _as_column(x), input2=_as_column(y))


def covar_pop(x: IntoExpr, y: IntoExpr) -> AggExpr:
    """Population covariance of two columns (DuckDB/Spark ``covar_pop``).

    Mergeable; null when a group has no paired values. Symmetric in `x` and `y`.

    Args:
        x: The first column.
        y: The second column.

    Returns:
        The population covariance of the two columns per group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 3.0], "y": [2.0, 6.0]})
            >>> ds.agg(c=bt.covar_pop(bt.col("x"), bt.col("y"))).to_pydict()
            {'c': [2.0]}
    """
    return AggExpr("covar_pop", _as_column(x), input2=_as_column(y))


def covar_samp(x: IntoExpr, y: IntoExpr) -> AggExpr:
    """Sample covariance of two columns (DuckDB/Spark ``covar_samp``).

    Mergeable; null when a group has fewer than 2 paired values. Symmetric in `x` and
    `y`.

    Args:
        x: The first column.
        y: The second column.

    Returns:
        The sample covariance of the two columns per group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 3.0], "y": [2.0, 6.0]})
            >>> ds.agg(c=bt.covar_samp(bt.col("x"), bt.col("y"))).to_pydict()
            {'c': [4.0]}
    """
    return AggExpr("covar_samp", _as_column(x), input2=_as_column(y))


def count_if(condition: Expr) -> AggExpr:
    """Count the rows in each group where a predicate is true (DuckDB/Spark ``count_if``).

    A NULL condition is treated as false (not counted), matching DuckDB. Use inside
    ``group_by(...).agg(...)`` or ``agg(...)``::

        ds.group_by("dept").agg(n_high=count_if(col("salary") > 100_000))

    Args:
        condition: The boolean predicate; rows where it is true are counted.

    Returns:
        The number of rows per group where ``condition`` is true.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "v": [10, 200, 5]})
            >>> ds.group_by("g").agg(n=bt.count_if(bt.col("v") > 100)).sort("g").to_pydict()
            {'g': ['a', 'b'], 'n': [1, 0]}
    """
    return iff(condition, Lit(1), Lit(0)).sum()


def _as_column(value: str | Expr) -> Expr:
    """A column reference from a name, else the expression as-is (Polars `pl.sum('a')`)."""
    from batcher.plan.expr_ir.constructors import col

    return col(value) if isinstance(value, str) else value


def sum(column: str | Expr) -> AggExpr:
    """Sum a column — the ``pl.sum('x')`` shorthand for ``col('x').sum()``.

    Args:
        column: The column to sum, as a name or an expression.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 3]})
            >>> ds.group_by("g").agg(bt.sum("x")).sort("g").to_pydict()
            {'g': ['a', 'b'], 'x': [3, 3]}
    """
    return _as_column(column).sum()


def mean(column: str | Expr) -> AggExpr:
    """Average a column — the ``pl.mean('x')`` shorthand for ``col('x').mean()``.

    Args:
        column: The column to average, as a name or an expression.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1.0, 3.0, 8.0]})
            >>> ds.group_by("g").agg(bt.mean("x")).sort("g").to_pydict()
            {'g': ['a', 'b'], 'x': [2.0, 8.0]}
    """
    return _as_column(column).mean()


def min(column: str | Expr) -> AggExpr:
    """Minimum of a column — the ``pl.min('x')`` shorthand for ``col('x').min()``.

    Args:
        column: The column to reduce, as a name or an expression.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [3, 1, 2]})
            >>> ds.group_by("g").agg(bt.min("x")).sort("g").to_pydict()
            {'g': ['a', 'b'], 'x': [1, 2]}
    """
    return _as_column(column).min()


def max(column: str | Expr) -> AggExpr:
    """Maximum of a column — the ``pl.max('x')`` shorthand for ``col('x').max()``.

    Args:
        column: The column to reduce, as a name or an expression.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [3, 1, 2]})
            >>> ds.group_by("g").agg(bt.max("x")).sort("g").to_pydict()
            {'g': ['a', 'b'], 'x': [3, 2]}
    """
    return _as_column(column).max()


def median(column: str | Expr) -> AggExpr:
    """Median of a column — the ``pl.median('x')`` shorthand for ``col('x').median()``.

    Args:
        column: The column to reduce, as a name or an expression.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "a", "b"], "x": [1, 2, 3, 9]})
            >>> ds.group_by("g").agg(bt.median("x")).sort("g").to_pydict()
            {'g': ['a', 'b'], 'x': [2.0, 9.0]}
    """
    return _as_column(column).median()


def std(column: str | Expr) -> AggExpr:
    """Sample standard deviation of a column — ``pl.std('x')`` for ``col('x').std()``.

    Args:
        column: The column to reduce, as a name or an expression.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b", "b"], "x": [1.0, 3.0, 10.0, 10.0]})
            >>> ds.group_by("g").agg(bt.std("x")).sort("g").to_pydict()
            {'g': ['a', 'b'], 'x': [1.4142135623730951, 0.0]}
    """
    return _as_column(column).std()


def var(column: str | Expr) -> AggExpr:
    """Sample variance of a column — the ``pl.var('x')`` shorthand for ``col('x').var()``.

    Args:
        column: The column to reduce, as a name or an expression.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b", "b"], "x": [1.0, 3.0, 10.0, 10.0]})
            >>> ds.group_by("g").agg(bt.var("x")).sort("g").to_pydict()
            {'g': ['a', 'b'], 'x': [2.0, 0.0]}
    """
    return _as_column(column).var()


def n_unique(column: str | Expr) -> AggExpr:
    """Count distinct values of a column — ``pl.n_unique('x')`` for ``col('x').n_unique()``.

    Args:
        column: The column to reduce, as a name or an expression.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 1, 5]})
            >>> ds.group_by("g").agg(bt.n_unique("x")).sort("g").to_pydict()
            {'g': ['a', 'b'], 'x': [1, 1]}
    """
    return _as_column(column).n_unique()


def product(column: str | Expr) -> AggExpr:
    """Multiply a column's values — the ``pl.product('x')`` shorthand for ``col('x').product()``.

    Args:
        column: The column to multiply, as a name or an expression.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [2, 3, 4]})
            >>> ds.group_by("g").agg(p=bt.product("x")).sort("g").to_pydict()
            {'g': ['a', 'b'], 'p': [6.0, 4.0]}
    """
    return _as_column(column).product()


def mode(column: str | Expr) -> AggExpr:
    """Most frequent value of a column (SQL ``MODE`` / DuckDB ``mode``; ties break low).

    Args:
        column: The column to summarize, as a name or an expression.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "a"], "x": [5, 5, 9]})
            >>> ds.group_by("g").agg(m=bt.mode("x")).to_pydict()
            {'g': ['a'], 'm': [5]}
    """
    return _as_column(column).mode()


def skewness(column: str | Expr) -> AggExpr:
    """Sample skewness — the third standardized moment (DuckDB ``skewness``).

    Args:
        column: The column to summarize, as a name or an expression.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
            >>> ds.agg(s=bt.skewness("x").round(4)).to_pydict()
            {'s': [2.2324]}
    """
    return _as_column(column).skewness()


def kurtosis(column: str | Expr) -> AggExpr:
    """Sample excess kurtosis — the fourth standardized moment (DuckDB ``kurtosis``).

    Args:
        column: The column to summarize, as a name or an expression.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
            >>> ds.agg(k=bt.kurtosis("x").round(4)).to_pydict()
            {'k': [4.9869]}
    """
    return _as_column(column).kurtosis()


def bool_and(column: str | Expr) -> AggExpr:
    """True when every non-null value is true (SQL ``BOOL_AND`` / ``EVERY``).

    Args:
        column: The boolean column to reduce, as a name or an expression.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "ok": [True, False, True]})
            >>> ds.group_by("g").agg(a=bt.bool_and("ok")).sort("g").to_pydict()
            {'g': ['a', 'b'], 'a': [False, True]}
    """
    return _as_column(column).bool_and()


def bool_or(column: str | Expr) -> AggExpr:
    """True when any non-null value is true (SQL ``BOOL_OR`` / ``SOME``).

    Args:
        column: The boolean column to reduce, as a name or an expression.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "ok": [True, False, False]})
            >>> ds.group_by("g").agg(o=bt.bool_or("ok")).sort("g").to_pydict()
            {'g': ['a', 'b'], 'o': [True, False]}
    """
    return _as_column(column).bool_or()


def bit_and(column: str | Expr) -> AggExpr:
    """Bitwise AND of the non-null integer values in each group (SQL ``BIT_AND``).

    Args:
        column: The integer column to reduce, as a name or an expression.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a"], "x": [6, 10]})
            >>> ds.group_by("g").agg(r=bt.bit_and("x")).to_pydict()
            {'g': ['a'], 'r': [2]}
    """
    return _as_column(column).bit_and()


def bit_or(column: str | Expr) -> AggExpr:
    """Bitwise OR of the non-null integer values in each group (SQL ``BIT_OR``).

    Args:
        column: The integer column to reduce, as a name or an expression.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a"], "x": [6, 10]})
            >>> ds.group_by("g").agg(r=bt.bit_or("x")).to_pydict()
            {'g': ['a'], 'r': [14]}
    """
    return _as_column(column).bit_or()


def bit_xor(column: str | Expr) -> AggExpr:
    """Bitwise XOR of the non-null integer values in each group (SQL ``BIT_XOR``).

    Args:
        column: The integer column to reduce, as a name or an expression.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a"], "x": [6, 10]})
            >>> ds.group_by("g").agg(r=bt.bit_xor("x")).to_pydict()
            {'g': ['a'], 'r': [12]}
    """
    return _as_column(column).bit_xor()


def array_agg(column: str | Expr) -> AggExpr:
    """Collect each group's values into a list (SQL ``ARRAY_AGG`` / Spark ``collect_list``).

    Args:
        column: The column to collect, as a name or an expression.

    Returns:
        An aggregate expression producing a `List` column; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [2, 3, 4]})
            >>> ds.group_by("g").agg(xs=bt.array_agg("x")).sort("g").to_pydict()
            {'g': ['a', 'b'], 'xs': [[2, 3], [4]]}
    """
    return _as_column(column).array_agg()
