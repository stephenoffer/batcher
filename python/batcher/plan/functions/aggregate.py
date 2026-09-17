"""Aggregate free functions that compose existing mergeable aggregates.

`count_if` desugars to ``sum(cast(cond AS BIGINT))`` — counting the rows where a
predicate holds reuses the mergeable `sum` aggregate, so it stays identical
single-node and distributed with no new engine state.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from batcher.plan.expr_ir.core import AggExpr, Expr, IntoExpr

__all__ = [
    "array_agg",
    "bit_and",
    "bit_or",
    "bit_xor",
    "bool_and",
    "bool_or",
    "corr",
    "count_distinct",
    "count_if",
    "covar_pop",
    "covar_samp",
    "kurtosis",
    "max",
    "mean",
    "median",
    "min",
    "mode",
    "product",
    "skew",
    "std",
    "sum",
    "var",
]


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
            >>> round(ds.agg(c=bt.corr(bt.col("x"), bt.col("y"))).to_pydict()["c"][0], 6)
            1.0
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
    # The boolean cast is the whole implementation: TRUE is 1, FALSE is 0 and NULL stays
    # NULL, so `sum` skips the unknown rows and answers NULL for a group with none known.
    return condition.cast("int64").sum()


def _as_column(value: str | Expr) -> Expr:
    """A column reference from a name, else the expression as-is (Polars `pl.sum('a')`)."""
    from batcher.plan.expr_ir.constructors import col

    return col(value) if isinstance(value, str) else value


def sum(column: str | Expr, *, empty_value: int | float | None = None) -> AggExpr | Expr:
    """Sum a column — the ``pl.sum('x')`` shorthand for ``col('x').sum()``.

    Args:
        column: The column to sum, as a name or an expression.
        empty_value: The sum of a group with no non-null value (Polars answers ``0``);
            ``None`` keeps SQL's null.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 3]})
            >>> ds.group_by("g").agg(bt.sum("x")).sort("g").to_pydict()
            {'g': ['a', 'b'], 'x': [3, 3]}
    """
    return _as_column(column).sum(empty_value=empty_value)


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


def max(column: str | Expr, *, nan_policy: str = "propagate") -> AggExpr | Expr:
    """Maximum of a column — the ``pl.max('x')`` shorthand for ``col('x').max()``.

    ``pl.max`` skips NaN; that is ``nan_policy="ignore"`` here (see :meth:`Expr.max`).

    Args:
        column: The column to reduce, as a name or an expression.
        nan_policy: ``"propagate"`` (NaN is the greatest value) or ``"ignore"``.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [3, 1, 2]})
            >>> ds.group_by("g").agg(bt.max("x")).sort("g").to_pydict()
            {'g': ['a', 'b'], 'x': [3, 2]}
    """
    return _as_column(column).max(nan_policy=nan_policy)


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


def std(column: str | Expr, *, ddof: int = 1) -> AggExpr | Expr:
    """Sample standard deviation of a column — ``pl.std('x')`` for ``col('x').std()``.

    Args:
        column: The column to reduce, as a name or an expression.
        ddof: Delta degrees of freedom; ``0`` is the population standard deviation.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b", "b"], "x": [1.0, 3.0, 10.0, 10.0]})
            >>> ds.group_by("g").agg(bt.std("x")).sort("g").to_pydict()
            {'g': ['a', 'b'], 'x': [1.4142135623730951, 0.0]}
    """
    return _as_column(column).std(ddof=ddof)


def var(column: str | Expr, *, ddof: int = 1) -> AggExpr | Expr:
    """Sample variance of a column — the ``pl.var('x')`` shorthand for ``col('x').var()``.

    Args:
        column: The column to reduce, as a name or an expression.
        ddof: Delta degrees of freedom; ``0`` is the population variance.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b", "b"], "x": [1.0, 3.0, 10.0, 10.0]})
            >>> ds.group_by("g").agg(bt.var("x")).sort("g").to_pydict()
            {'g': ['a', 'b'], 'x': [2.0, 0.0]}
    """
    return _as_column(column).var(ddof=ddof)


def count_distinct(column: str | Expr, *, count_nulls: bool = False) -> AggExpr | Expr:
    """Count distinct values of a column — ``pl.n_unique('x')`` for ``col('x').count_distinct()``.

    ``pl.n_unique`` counts null as a value, which is ``count_nulls=True`` here.

    Args:
        column: The column to reduce, as a name or an expression.
        count_nulls: Whether a null counts as a distinct value.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 1, 5]})
            >>> ds.group_by("g").agg(bt.count_distinct("x")).sort("g").to_pydict()
            {'g': ['a', 'b'], 'x': [1, 1]}
    """
    return _as_column(column).count_distinct(count_nulls=count_nulls)


def product(column: str | Expr, *, empty_value: float | None = None) -> AggExpr | Expr:
    """Multiply a column's values — the ``pl.product('x')`` shorthand for ``col('x').product()``.

    Args:
        column: The column to multiply, as a name or an expression.
        empty_value: The product of a group with no non-null value (Polars answers ``1``);
            ``None`` keeps SQL's null.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [2, 3, 4]})
            >>> ds.group_by("g").agg(p=bt.product("x")).sort("g").to_pydict()
            {'g': ['a', 'b'], 'p': [6.0, 4.0]}
    """
    return _as_column(column).product(empty_value=empty_value)


def mode(column: str | Expr, *, all_modes: bool = False) -> AggExpr:
    """Most frequent value of a column (SQL ``MODE`` / DuckDB ``mode``; ties break low).

    Args:
        column: The column to summarize, as a name or an expression.
        all_modes: Whether to return every tied value as an ascending list, as Polars does.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "a"], "x": [5, 5, 9]})
            >>> ds.group_by("g").agg(m=bt.mode("x")).to_pydict()
            {'g': ['a'], 'm': [5]}
    """
    return _as_column(column).mode(all_modes=all_modes)


def skew(column: str | Expr, *, bias: bool = False) -> AggExpr:
    """Sample skewness — the third standardized moment (DuckDB ``skewness``).

    Args:
        column: The column to summarize, as a name or an expression.
        bias: Whether to return the population skewness, as Spark, Polars and Daft do.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
            >>> ds.agg(s=bt.skew("x").round(4)).to_pydict()
            {'s': [2.2324]}
    """
    return _as_column(column).skew(bias=bias)


def kurtosis(column: str | Expr, *, bias: bool = False, fisher: bool = True) -> AggExpr | Expr:
    """Sample excess kurtosis — the fourth standardized moment (DuckDB ``kurtosis``).

    Args:
        column: The column to summarize, as a name or an expression.
        bias: Whether to return the population estimate, as Spark and Polars do.
        fisher: Whether to subtract 3, so a normal distribution scores 0.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
            >>> ds.agg(k=bt.kurtosis("x").round(4)).to_pydict()
            {'k': [4.9869]}
    """
    return _as_column(column).kurtosis(bias=bias, fisher=fisher)


def bool_and(column: str | Expr, *, empty_value: bool | None = None) -> AggExpr | Expr:
    """True when every non-null value is true (SQL ``BOOL_AND`` / ``EVERY``).

    Args:
        column: The boolean column to reduce, as a name or an expression.
        empty_value: The result for a group with no non-null value (Polars' ``all``
            answers ``True``); ``None`` keeps SQL's null.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "ok": [True, False, True]})
            >>> ds.group_by("g").agg(a=bt.bool_and("ok")).sort("g").to_pydict()
            {'g': ['a', 'b'], 'a': [False, True]}
    """
    return _as_column(column).bool_and(empty_value=empty_value)


def bool_or(column: str | Expr, *, empty_value: bool | None = None) -> AggExpr | Expr:
    """True when any non-null value is true (SQL ``BOOL_OR`` / ``SOME``).

    Args:
        column: The boolean column to reduce, as a name or an expression.
        empty_value: The result for a group with no non-null value (Polars' ``any``
            answers ``False``); ``None`` keeps SQL's null.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "ok": [True, False, False]})
            >>> ds.group_by("g").agg(o=bt.bool_or("ok")).sort("g").to_pydict()
            {'g': ['a', 'b'], 'o': [True, False]}
    """
    return _as_column(column).bool_or(empty_value=empty_value)


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


def array_agg(
    column: str | Expr,
    *,
    order_by: IntoExpr | Iterable[IntoExpr] | None = None,
    descending: bool | Sequence[bool] = False,
    nulls_last: bool | Sequence[bool] = True,
    ignore_nulls: bool = False,
) -> AggExpr | Expr:
    """Collect each group's values into a list (SQL ``ARRAY_AGG`` / Spark ``collect_list``).

    The element order is unspecified unless `order_by` fixes it; see
    :meth:`Expr.array_agg <batcher.Expr.array_agg>` for the ordering and tie rules.

    Args:
        column: The column to collect, as a name or an expression.
        order_by: The key or keys that order each list's elements.
        descending: Order from the largest key, for every key or per key.
        nulls_last: Place elements whose key is null after the others, for every key or
            per key.
        ignore_nulls: Whether to leave nulls out of each list, as Spark's ``collect_list``
            and ``array_agg`` do; SQL and DuckDB keep them.

    Returns:
        An aggregate expression producing a `List` column; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [2, 3, 4], "t": [1, 0, 0]})
            >>> ds.group_by("g").agg(xs=bt.array_agg("x", order_by="t")).sort("g").to_pydict()
            {'g': ['a', 'b'], 'xs': [[3, 2], [4]]}
    """
    return _as_column(column).array_agg(
        order_by=order_by, descending=descending, nulls_last=nulls_last, ignore_nulls=ignore_nulls
    )
