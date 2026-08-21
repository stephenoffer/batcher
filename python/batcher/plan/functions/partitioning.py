"""Lakehouse partition transforms — the derived value a partitioned table stores.

A partitioned table does not store the partition *column*; it stores a deterministic
function of it. Iceberg calls these functions *partition transforms* and writes
``days(ts)``, ``months(ts)``, ``bucket(16, id)`` or ``truncate(4, name)`` into the
partition spec, and a Hive-style layout spells the same idea as a directory name. The
reader side already understands them (`io.formats.lakehouse.iceberg`); this module is
the *writer* and *query* side — the transform as an ordinary `Expr`, so the value a
table will be partitioned by is computable, groupable and filterable before it is
written, and a `GROUP BY` on it needs no shuffle over a table already laid out that way.

Every transform here is exact against the Iceberg specification, which matters more
than it looks: a partition value computed a different way from the one the table was
written with sends a row to the wrong partition, and nothing errors. Two places the
specification is easy to get wrong, and both are pinned by tests:

* the time transforms count from the **epoch** and are **negative** before 1970 —
  ``days`` of ``1969-06-01`` is ``-214``, not an error and not zero;
* ``truncate`` on an integer floors toward **negative infinity**, so
  ``truncate(-7, 5)`` is ``-10``. Writing it as ``v - (v % W)`` gives ``-5``, because
  the engine's ``%`` takes the dividend's sign the way C and SQL do. The floored
  remainder ``((v % W) + W) % W`` is what the specification means, and it is the
  difference between a row landing in the ``-10`` partition and the ``-5`` one.

Two transforms are deliberately absent rather than approximated.

``bucket`` is pinned by the specification to the 32-bit x86 variant of MurmurHash3 over
the value's canonical byte encoding, and the engine's hash family (`hash64`,
`xxhash64`, `crc32`) is none of those. A composition here would produce plausible
bucket numbers that put rows in different files than the table's own writer does, which
is worse than not having it. It needs the kernel.

``truncate`` **on text** is `.str.substr(1, width)` and is spelled that way rather than
folded into `partition_truncate`. The numeric and text readings are not one operation —
a modulus and a prefix — so choosing between them means knowing the argument's type,
which an expression builder does not: a bare `col("x")` has no type until a source is
bound. Guessing would silently take a modulus of a string column's length or a prefix
of a number. The SQL translator, which *does* have the column type in scope, dispatches
both readings under Iceberg's own one-word spelling.
"""

from __future__ import annotations

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.core import Expr, IntoExpr, _col_or_expr

__all__ = [
    "partition_days",
    "partition_hours",
    "partition_months",
    "partition_truncate",
    "partition_years",
]

#: Months in a year — the `months` transform's radix, named so the arithmetic below
#: reads the way the specification states it rather than as a bare 12.
_MONTHS_PER_YEAR = 12
#: Seconds in an hour, for the `hours` transform.
_SECONDS_PER_HOUR = 3600
#: The year the epoch transforms count from.
_EPOCH_YEAR = 1970


def partition_years(expr: IntoExpr) -> Expr:
    """Years since 1970 — the Iceberg ``years(ts)`` partition transform.

    Negative before 1970, as the specification requires: 1969 is ``-1``.

    Args:
        expr: A date or timestamp expression, or a column name.

    Returns:
        A new Int64 expression: the whole years between 1970 and this value.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> import datetime as dt
            >>> ds = bt.from_pydict({"t": [dt.date(2024, 3, 5), dt.date(1969, 6, 1)]})
            >>> ds.select(p=bt.partition_years("t")).to_pydict()
            {'p': [54, -1]}
    """
    return _col_or_expr(expr).dt.year() - _EPOCH_YEAR


def partition_months(expr: IntoExpr) -> Expr:
    """Months since 1970-01 — the Iceberg ``months(ts)`` partition transform.

    Negative before 1970: 1969-06 is ``-7``.

    Args:
        expr: A date or timestamp expression, or a column name.

    Returns:
        A new Int64 expression: the whole months between 1970-01 and this value.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> import datetime as dt
            >>> ds = bt.from_pydict({"t": [dt.date(2024, 3, 5), dt.date(1969, 6, 1)]})
            >>> ds.select(p=bt.partition_months("t")).to_pydict()
            {'p': [650, -7]}
    """
    value = _col_or_expr(expr)
    return (value.dt.year() - _EPOCH_YEAR) * _MONTHS_PER_YEAR + (value.dt.month() - 1)


def partition_days(expr: IntoExpr) -> Expr:
    """Days since 1970-01-01 — the Iceberg ``days(ts)`` partition transform.

    Negative before 1970: 1969-06-01 is ``-214``. This is the same integer a ``date``
    column already holds, which is why the transform is a cast rather than a subtraction.

    Args:
        expr: A date or timestamp expression, or a column name.

    Returns:
        A new Int64 expression: the whole days between the epoch and this value.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> import datetime as dt
            >>> ds = bt.from_pydict({"t": [dt.date(2024, 3, 5), dt.date(1969, 6, 1)]})
            >>> ds.select(p=bt.partition_days("t")).to_pydict()
            {'p': [19787, -214]}
    """
    return _col_or_expr(expr).cast("date").cast("int64")


def partition_hours(expr: IntoExpr) -> Expr:
    """Hours since 1970-01-01T00:00 — the Iceberg ``hours(ts)`` partition transform.

    Floored rather than truncated, so the sequence stays monotone across the epoch: the
    hour before midnight on 1970-01-01 is ``-1``.

    Args:
        expr: A timestamp expression, or a column name.

    Returns:
        A new Int64 expression: the whole hours between the epoch and this value.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> import datetime as dt
            >>> rows = [dt.datetime(1970, 1, 1, 5), dt.datetime(1969, 12, 31, 23)]
            >>> ds = bt.from_pydict({"t": rows})
            >>> ds.select(p=bt.partition_hours("t")).to_pydict()
            {'p': [5, -1]}
    """
    return _col_or_expr(expr).dt.epoch() // _SECONDS_PER_HOUR


def partition_truncate(expr: IntoExpr, width: int) -> Expr:
    """Round a number down to a multiple of `width` — Iceberg ``truncate(W, v)``.

    The largest multiple of `width` at or below the value, floored toward negative
    infinity: ``partition_truncate(-7, 5)`` is ``-10``, not ``-5``. That half is what a
    naive ``v - v % W`` gets wrong, because the engine's ``%`` takes the dividend's sign.

    This is the **numeric** reading. Iceberg's ``truncate`` also applies to text, where it
    means the first `width` characters; spell that ``col("s").str.substr(1, width)``. The
    two are not one operation, and an expression builder cannot tell which one a caller
    means without the column's type.

    Args:
        expr: An integer or decimal expression, or a column name.
        width: The bucket width. Must be positive.

    Returns:
        A new expression of the input's type: the value rounded down to a multiple of
        `width`.

    Raises:
        PlanError: If `width` is not positive.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"n": [17, -7]})
            >>> ds.select(p=bt.partition_truncate("n", 5)).to_pydict()
            {'p': [15, -10]}
    """
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise PlanError(f"partition_truncate() width must be a positive integer, got {width!r}")
    value = _col_or_expr(expr)
    return value - ((value % width) + width) % width
