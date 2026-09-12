"""Operator-mix: set operations over the TPC-H ``lineitem`` / ``orders`` key columns.

``UNION``, ``INTERSECT`` and ``EXCEPT`` are public API with no case in the suite. They are
not variations on one another: ``UNION ALL`` is a concatenation that must not deduplicate,
``UNION`` is a distinct over the concatenation, and ``INTERSECT``/``EXCEPT`` are semi- and
anti-joins on *every* column. Each reaches a different path, so each gets a case.

Both inputs are real and differently sized -- ``lineitem`` holds about four keys per
``orders`` row -- so the deduplicating forms do real work rather than returning their input.
Every case counts its result, keeping the correctness gate to one row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from registry import suite

from .base import sql_fanout

if TYPE_CHECKING:
    from context import Context

setops = suite("ops-setops", dataset="operators")


@setops.case("op-union-all")
def union_all(ctx: Context):
    """COUNT over UNION ALL -- concatenation, no deduplication."""
    sql = (
        "SELECT COUNT(*) AS n FROM ("
        "SELECT l_orderkey AS k FROM lineitem UNION ALL SELECT o_orderkey AS k FROM orders"
        ") u"
    )
    return sql_fanout(ctx, sql)


@setops.case("op-union-distinct")
def union_distinct(ctx: Context):
    """COUNT over UNION -- a distinct across two relations of very different size."""
    sql = (
        "SELECT COUNT(*) AS n FROM ("
        "SELECT l_orderkey AS k FROM lineitem UNION SELECT o_orderkey AS k FROM orders"
        ") u"
    )
    return sql_fanout(ctx, sql)


@setops.case("op-intersect")
def intersect(ctx: Context):
    """COUNT over INTERSECT -- a distinct semi-join on the whole row."""
    sql = (
        "SELECT COUNT(*) AS n FROM ("
        "SELECT l_orderkey AS k FROM lineitem INTERSECT SELECT o_orderkey AS k FROM orders"
        ") u"
    )
    return sql_fanout(ctx, sql)


@setops.case("op-except")
def except_(ctx: Context):
    """COUNT over EXCEPT -- a distinct anti-join on the whole row."""
    sql = (
        "SELECT COUNT(*) AS n FROM ("
        "SELECT o_orderkey AS k FROM orders EXCEPT SELECT l_orderkey AS k FROM lineitem"
        ") u"
    )
    return sql_fanout(ctx, sql)
