"""pandas-compatible names for `Expr` methods that Batcher spells differently.

A data scientist arriving from pandas types ``isna``, ``fillna``, ``astype``,
``nunique``, ``cumsum``. Batcher's primary spelling is the SQL/Polars one
(``is_null``, ``fill_null``, ``cast``, ``n_unique``, ``cum_sum``), and that stays
canonical — these are the migration surface, mirroring the pandas aliases already
on `Dataset` (``ds.fillna``, ``ds.astype``, ``ds.dropna``).

Each function delegates to the primary, so the built plan is identical and there is
exactly one implementation to keep correct. They are bound onto `Expr` by
`compat.bind_compat_methods`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from batcher.plan.expr_ir.core import (
        AggExpr,
        Cast,
        Coalesce,
        Expr,
        IntoExpr,
        IsNotNull,
        IsNull,
        MathExpr,
    )
    from batcher.plan.expr_ir.nodes import WindowExpr

__all__ = [
    "all",
    "any",
    "astype",
    "cumcount",
    "cummax",
    "cummin",
    "cumsum",
    "fillna",
    "isin",
    "isna",
    "isnull",
    "kurt",
    "log",
    "notna",
    "notnull",
    "nunique",
    "prod",
    "rename",
    "skew",
]


def astype(self: Expr, dtype: str) -> Cast:
    """Cast to an Arrow type — the pandas ``astype`` spelling of :meth:`cast`.

    Args:
        dtype: Target Arrow type name (e.g. ``"int64"``, ``"float64"``, ``"string"``).

    Returns:
        A new expression of the converted values.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, 2]})
            >>> ds.select(r=bt.col("x").astype("float64")).to_pydict()
            {'r': [1.0, 2.0]}
    """
    return self.cast(dtype)


def isna(self: Expr) -> IsNull:
    """True where the value is NULL — the pandas ``isna`` spelling of :meth:`is_null`.

    Returns:
        A boolean expression, true where the value is null.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, None, 3]})
            >>> ds.select(r=bt.col("x").isna()).to_pydict()
            {'r': [False, True, False]}
    """
    return self.is_null()


def isnull(self: Expr) -> IsNull:
    """True where the value is NULL — the pandas ``isnull`` alias of :meth:`is_null`.

    Returns:
        A boolean expression, true where the value is null.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, None, 3]})
            >>> ds.select(r=bt.col("x").isnull()).to_pydict()
            {'r': [False, True, False]}
    """
    return self.is_null()


def notna(self: Expr) -> IsNotNull:
    """True where the value is non-NULL — the pandas ``notna`` spelling of :meth:`is_not_null`.

    Returns:
        A boolean expression, true where the value is non-null.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, None, 3]})
            >>> ds.select(r=bt.col("x").notna()).to_pydict()
            {'r': [True, False, True]}
    """
    return self.is_not_null()


def notnull(self: Expr) -> IsNotNull:
    """True where the value is non-NULL — the pandas ``notnull`` alias of :meth:`is_not_null`.

    Returns:
        A boolean expression, true where the value is non-null.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, None, 3]})
            >>> ds.select(r=bt.col("x").notnull()).to_pydict()
            {'r': [True, False, True]}
    """
    return self.is_not_null()


def fillna(self: Expr, value: IntoExpr) -> Coalesce:
    """Replace nulls with `value` — the pandas ``fillna`` spelling of :meth:`fill_null`.

    Args:
        value: The replacement value or expression.

    Returns:
        A new expression with nulls replaced.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, None, 3]})
            >>> ds.select(r=bt.col("x").fillna(0)).to_pydict()
            {'r': [1, 0, 3]}
    """
    return self.fill_null(value)


def isin(self: Expr, values: Iterable[IntoExpr]) -> Expr:
    """Membership test — the pandas ``isin`` spelling of :meth:`is_in`.

    Args:
        values: The scalars or expressions to test membership against.

    Returns:
        A boolean expression, true where the value is in `values`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, 2, 3]})
            >>> ds.select(r=bt.col("x").isin([1, 3])).to_pydict()
            {'r': [True, False, True]}
    """
    return self.is_in(values)


def nunique(self: Expr) -> AggExpr:
    """Count distinct values — the pandas ``nunique`` spelling of :meth:`n_unique`.

    Returns:
        An aggregate expression counting the distinct non-null values.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10]})
            >>> ds.group_by("g").agg(r=bt.col("x").nunique()).sort("g").to_pydict()
            {'g': ['a', 'b'], 'r': [2, 1]}
    """
    return self.n_unique()


def rename(self: Expr, name: str) -> Expr:
    """Bind an output name — the pandas ``rename`` spelling of :meth:`alias`.

    Args:
        name: The output name to bind to this expression.

    Returns:
        The expression tagged with `name` for a positional `select`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, 2]})
            >>> ds.select((bt.col("x") * 2).rename("doubled")).to_pydict()
            {'doubled': [2, 4]}
    """
    return self.alias(name)


def skew(self: Expr) -> AggExpr:
    """Sample skewness — the pandas ``skew`` spelling of :meth:`skewness`.

    Returns:
        An aggregate expression computing the skewness.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a"] * 4, "x": [1, 2, 3, 10]})
            >>> ds.group_by("g").agg(r=bt.col("x").skew()).to_pydict()
            {'g': ['a'], 'r': [1.763632614803888]}
    """
    return self.skewness()


def kurt(self: Expr) -> AggExpr:
    """Sample kurtosis — the pandas ``kurt`` spelling of :meth:`kurtosis`.

    Returns:
        An aggregate expression computing the kurtosis.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a"] * 5, "x": [1, 2, 3, 4, 10]})
            >>> ds.group_by("g").agg(r=bt.col("x").kurt()).to_pydict()
            {'g': ['a'], 'r': [3.152000000000001]}
    """
    return self.kurtosis()


def cumsum(
    self: Expr,
    *,
    partition_by: Iterable[IntoExpr] = (),
    order_by: Iterable[IntoExpr] = (),
) -> WindowExpr:
    """Running total — the pandas ``cumsum`` spelling of :meth:`cum_sum`.

    Args:
        partition_by: Restart the accumulation within each group.
        order_by: Order the rows before accumulating.

    Returns:
        A window expression computing the running sum.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, 2, 3, 4]})
            >>> ds.with_columns(cs=bt.col("x").cumsum()).to_pydict()
            {'x': [1, 2, 3, 4], 'cs': [1, 3, 6, 10]}
    """
    return self.cum_sum(partition_by=partition_by, order_by=order_by)


def cummax(
    self: Expr,
    *,
    partition_by: Iterable[IntoExpr] = (),
    order_by: Iterable[IntoExpr] = (),
) -> WindowExpr:
    """Running maximum — the pandas ``cummax`` spelling of :meth:`cum_max`.

    Args:
        partition_by: Restart the accumulation within each group.
        order_by: Order the rows before accumulating.

    Returns:
        A window expression computing the running maximum.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, 3, 2, 5]})
            >>> ds.with_columns(cm=bt.col("x").cummax()).to_pydict()
            {'x': [1, 3, 2, 5], 'cm': [1, 3, 3, 5]}
    """
    return self.cum_max(partition_by=partition_by, order_by=order_by)


def cummin(
    self: Expr,
    *,
    partition_by: Iterable[IntoExpr] = (),
    order_by: Iterable[IntoExpr] = (),
) -> WindowExpr:
    """Running minimum — the pandas ``cummin`` spelling of :meth:`cum_min`.

    Args:
        partition_by: Restart the accumulation within each group.
        order_by: Order the rows before accumulating.

    Returns:
        A window expression computing the running minimum.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [5, 3, 4, 1]})
            >>> ds.with_columns(cm=bt.col("x").cummin()).to_pydict()
            {'x': [5, 3, 4, 1], 'cm': [5, 3, 3, 1]}
    """
    return self.cum_min(partition_by=partition_by, order_by=order_by)


def cumcount(
    self: Expr,
    *,
    partition_by: Iterable[IntoExpr] = (),
    order_by: Iterable[IntoExpr] = (),
) -> WindowExpr:
    """Running count — the pandas ``cumcount`` spelling of :meth:`cum_count`.

    Args:
        partition_by: Restart the count within each group.
        order_by: Order the rows before counting.

    Returns:
        A window expression computing the running count.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [5, 3, 4]})
            >>> ds.with_columns(cc=bt.col("x").cumcount()).to_pydict()
            {'x': [5, 3, 4], 'cc': [1, 2, 3]}
    """
    return self.cum_count(partition_by=partition_by, order_by=order_by)


def prod(self: Expr) -> AggExpr:
    """Product of the values — the pandas/numpy ``prod`` spelling of :meth:`product`.

    Returns:
        An aggregate expression multiplying the non-null values.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "a"], "x": [2, 3, 4]})
            >>> ds.group_by("g").agg(r=bt.col("x").prod()).to_pydict()
            {'g': ['a'], 'r': [24.0]}
    """
    return self.product()


def any(self: Expr) -> AggExpr:
    """True if any value is true — the pandas ``any`` spelling of :meth:`bool_or`.

    Returns:
        An aggregate boolean expression, true if at least one value is true.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a"], "b": [True, False]})
            >>> ds.group_by("g").agg(r=bt.col("b").any()).to_pydict()
            {'g': ['a'], 'r': [True]}
    """
    return self.bool_or()


def all(self: Expr) -> AggExpr:
    """True if every value is true — the pandas ``all`` spelling of :meth:`bool_and`.

    Returns:
        An aggregate boolean expression, true only if all values are true.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a"], "b": [True, False]})
            >>> ds.group_by("g").agg(r=bt.col("b").all()).to_pydict()
            {'g': ['a'], 'r': [False]}
    """
    return self.bool_and()


def log(self: Expr) -> MathExpr:
    """Natural logarithm — the numpy ``log`` spelling of :meth:`ln`.

    Follows the numpy convention where ``log`` is the natural (base-e) logarithm.
    Use :meth:`log10` or :meth:`log2` for the other bases.

    Returns:
        A Float64 expression of the natural logarithm.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [2.718281828459045]})
            >>> ds.select(r=bt.col("x").log()).to_pydict()
            {'r': [1.0]}
    """
    return self.ln()
