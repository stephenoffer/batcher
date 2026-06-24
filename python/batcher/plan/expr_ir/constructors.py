"""Module-level expression constructors (the user-facing entry points).

`col`, `lit`, `when`, `coalesce`, `nullif`, `atan2`, `greatest`, `least`, and
`count` build expression trees out of the node classes in `core`. These are the
free functions users call directly (e.g. `col("x")`, `when(c).then(v)`).
"""

from __future__ import annotations

import functools
import operator

from batcher.plan.expr_ir.core import (
    AggExpr,
    Coalesce,
    Expr,
    IntoExpr,
    Lit,
    Math2Expr,
    _wrap,
)
from batcher.plan.expr_ir.nodes import Array, CaseBuilder, Col, Greatest, Least, NullIf


def when(cond: Expr) -> CaseBuilder:
    """Begin a CASE expression.

    Returns a builder you chain with ``.then(value)`` and finish with
    ``.otherwise(default)``; add further ``.when(...).then(...)`` pairs for more
    branches. The first matching condition wins, evaluated row by row.

    Args:
        cond: A boolean expression selecting the rows this branch applies to.

    Returns:
        A `CaseBuilder`; call ``.then(...).otherwise(...)`` to produce the expression.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [-1, 0, 5]})
            >>> grade = bt.when(bt.col("x") > 0).then(bt.lit("pos")).otherwise(bt.lit("non-pos"))
            >>> ds.select(grade=grade).to_pydict()
            {'grade': ['non-pos', 'non-pos', 'pos']}
    """
    return CaseBuilder().when(cond)


def array(*elements: IntoExpr) -> Array:
    """A list literal built per row from the element expressions (SQL ``ARRAY[...]``).

    Each output row is a list of the per-row element values, coerced to a common
    type. Use it to pack several columns into one list column — a feature vector,
    an embedding, or a set passed to a list operation.

    Args:
        *elements: One or more expressions, one per list position.

    Returns:
        An expression producing a `List` column.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1], "b": [2]})
            >>> ds.select(pair=bt.array(bt.col("a"), bt.col("b"))).to_pydict()
            {'pair': [[1, 2]]}
    """
    if not elements:
        raise ValueError("array() requires at least one element")
    return Array([_wrap(e) for e in elements])


def coalesce(*exprs: IntoExpr) -> Coalesce:
    """First non-null among the arguments, per row (SQL ``COALESCE``).

    Evaluates the arguments left to right and returns the first that is not null,
    or null if all are. The usual use is a fallback for a nullable column, e.g.
    ``coalesce(col("discount"), lit(0))`` to treat a missing discount as zero.

    Args:
        *exprs: One or more expressions, tested in order.

    Returns:
        An expression equal to the first non-null argument.
    """
    if not exprs:
        raise ValueError("coalesce() requires at least one argument")
    return Coalesce([_wrap(e) for e in exprs])


def nullif(left: IntoExpr, right: IntoExpr) -> NullIf:
    """Null where ``left == right``, else ``left`` (SQL ``NULLIF``).

    Returns null when the two arguments are equal, otherwise the left value. Useful
    for turning a sentinel into a real null (``nullif(col("x"), lit(-1))``) or
    guarding a divisor against zero (``a / nullif(b, lit(0))`` yields null, not an
    error, when ``b`` is 0).

    Args:
        left: The value returned when the two differ.
        right: The value that, when equal to ``left``, produces null.

    Returns:
        An expression that is null on equality, else ``left``.
    """
    return NullIf(_wrap(left), _wrap(right))


def atan2(y: IntoExpr, x: IntoExpr) -> Math2Expr:
    """Two-argument arctangent of ``y / x`` (→ Float64).

    Computes the angle of the point ``(x, y)`` from the positive x-axis, using the
    signs of both arguments to place it in the correct quadrant, so the result
    spans the full ``[-π, π]`` range (unlike single-argument ``atan``).

    Args:
        y: The ordinate (numerator).
        x: The abscissa (denominator).

    Returns:
        A Float64 expression of the angle in radians.
    """
    return Math2Expr("atan2", _wrap(y), _wrap(x))


def greatest(*exprs: IntoExpr) -> Greatest:
    """The largest argument per row, ignoring nulls (SQL ``GREATEST``).

    Compares the arguments value by value within each row and returns the maximum,
    skipping nulls; a row that is null in every argument yields null. This is a
    row-wise (horizontal) max across columns, not an aggregate down a column — for
    that, use ``col("x").max()`` inside ``agg``.

    Args:
        *exprs: One or more expressions to compare.

    Returns:
        An expression equal to the per-row maximum.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1, 9], "b": [4, 2]})
            >>> ds.select(hi=bt.greatest(bt.col("a"), bt.col("b"))).to_pydict()
            {'hi': [4, 9]}
    """
    if not exprs:
        raise ValueError("greatest() requires at least one argument")
    return Greatest([_wrap(e) for e in exprs])


def least(*exprs: IntoExpr) -> Least:
    """The smallest argument per row, ignoring nulls (SQL ``LEAST``).

    The row-wise (horizontal) minimum across the given expressions, skipping nulls;
    an all-null row yields null. The counterpart to `greatest`.

    Args:
        *exprs: One or more expressions to compare.

    Returns:
        An expression equal to the per-row minimum.
    """
    if not exprs:
        raise ValueError("least() requires at least one argument")
    return Least([_wrap(e) for e in exprs])


def sum_horizontal(*exprs: IntoExpr) -> Expr:
    """Row-wise sum across the given columns, treating nulls as 0 (Polars
    ``sum_horizontal``). Complements `greatest`/`least` (row-wise max/min). An
    all-null row sums to 0. ``sum_horizontal(col("a"), col("b"), col("c"))``."""
    if not exprs:
        raise ValueError("sum_horizontal() requires at least one argument")
    parts = [coalesce(_wrap(e), Lit(0)) for e in exprs]
    return functools.reduce(operator.add, parts)


def mean_horizontal(*exprs: IntoExpr) -> Expr:
    """Row-wise mean across the given columns, ignoring nulls (Polars
    ``mean_horizontal``): the sum of the non-null values divided by how many were
    non-null. An all-null row yields null (no division by zero)."""
    if not exprs:
        raise ValueError("mean_horizontal() requires at least one argument")
    wrapped = [_wrap(e) for e in exprs]
    total = functools.reduce(operator.add, [coalesce(e, Lit(0)) for e in wrapped])
    count = functools.reduce(operator.add, [e.is_not_null().cast("int64") for e in wrapped])
    # Divide by NULLIF(count, 0): an all-null row has count 0 → null (no div-by-zero).
    return total / nullif(count, Lit(0))


def col(name: str) -> Col:
    """Reference an input column by name.

    ``col`` is the starting point for almost every expression: it names a column in
    the dataset, and the operators (``+``, ``==``, ``&`` …) and methods (``.sum()``,
    ``.cast(...)``, ``.str.upper()`` …) on the result build the computation that
    runs in the Rust engine. It is lazy and does no work itself.

    Args:
        name: The name of an existing column.

    Returns:
        An expression that evaluates to that column's values.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"price": [10, 20], "qty": [2, 3]})
            >>> ds.select(total=bt.col("price") * bt.col("qty")).to_pydict()
            {'total': [20, 60]}
    """
    return Col(name)


def count() -> AggExpr:
    """``COUNT(*)`` — the number of rows in each group.

    Use inside ``group_by(...).agg(...)`` to count rows per group, or with no
    grouping to count the whole dataset. It counts rows, not non-null values, so it
    takes no column; for non-null counts use ``col("x").count()``.

    Returns:
        An aggregate expression; pass it to ``.agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b"]})
            >>> ds.group_by("g").agg(n=bt.count()).sort("g").to_pydict()
            {'g': ['a', 'b'], 'n': [2, 1]}
    """
    return AggExpr("count_star", None)


def lit(value: int | float | bool | str) -> Lit:
    """A constant literal expression.

    Wraps a Python scalar so it can be combined with column expressions — a default
    in ``when(...).otherwise(bt.lit(0))``, an offset like ``bt.col("x") + bt.lit(1)``,
    or a fallback in ``coalesce(col("x"), bt.lit(0))``. Bare Python scalars are
    accepted in most places too; ``lit`` is the explicit form.

    Args:
        value: The constant value (int, float, bool, or str).

    Returns:
        An expression that evaluates to ``value`` on every row.
    """
    return Lit(value)
