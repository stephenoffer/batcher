"""Operator-mix: scalar expression evaluation over TPC-H ``lineitem``.

The suite timed relational *operators* and never the expressions inside them, so the
Tier-1 Cranelift JIT -- whose whole purpose is a compiled scalar pipeline reused across
morsels -- had no case of its own. These isolate the evaluator from the operator around it:
each is a filter or a global reduction, so the relational work is a single pass and what
varies is the expression tree on top of it.

* ``op-expr-arith`` -- a five-operation float chain, the JIT's supported subset exactly.
* ``op-expr-case`` -- a branching ``CASE`` chain, which the JIT does not compile and the
  interpreter must evaluate branch-wise.
* ``op-expr-conditional`` -- ``COALESCE``/``NULLIF``, the null-handling path.
* ``op-expr-date-part`` -- temporal extraction feeding a group key.
* ``op-expr-date-arith`` -- interval arithmetic in a predicate.
* ``op-expr-cast-chain`` -- explicit casts across three widths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from registry import suite

from .base import sql_fanout, with_native

if TYPE_CHECKING:
    from context import Context

expressions = suite("ops-expressions", dataset="operators")


@expressions.case("op-expr-arith")
def expr_arith(ctx: Context):
    """A five-operation float chain reduced to one scalar -- the JIT's supported subset."""
    sql = (
        "SELECT SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax) + l_quantity * 2) AS s "
        "FROM lineitem"
    )

    def pyarrow(t: pa.Table) -> pa.Table:
        v = pc.add(
            pc.multiply(
                pc.multiply(t["l_extendedprice"], pc.subtract(1.0, t["l_discount"])),
                pc.add(1.0, t["l_tax"]),
            ),
            pc.multiply(t["l_quantity"], 2),
        )
        return pa.table({"s": pa.array([pc.sum(v).as_py()])})

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow)


@expressions.case("op-expr-case")
def expr_case(ctx: Context):
    """A four-arm CASE bucketing every row, grouped -- the branching path the JIT declines."""
    sql = (
        "SELECT CASE WHEN l_discount < 0.03 THEN 'a' WHEN l_discount < 0.06 THEN 'b' "
        "WHEN l_discount < 0.09 THEN 'c' ELSE 'd' END AS bucket, "
        "COUNT(*) AS n, SUM(l_quantity) AS q "
        "FROM lineitem GROUP BY 1"
    )
    return sql_fanout(ctx, sql)


@expressions.case("op-expr-conditional")
def expr_conditional(ctx: Context):
    """COALESCE over NULLIF -- the null-producing and null-absorbing pair in one expression."""
    sql = "SELECT SUM(COALESCE(NULLIF(l_discount, 0.00), 1.00)) AS s FROM lineitem"

    def pyarrow(t: pa.Table) -> pa.Table:
        nulled = pc.if_else(
            pc.equal(t["l_discount"], 0.00), pa.scalar(None, pa.float64()), t["l_discount"]
        )
        filled = pc.fill_null(nulled, 1.00)
        return pa.table({"s": pa.array([pc.sum(filled).as_py()])})

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow)


@expressions.case("op-expr-date-part")
def expr_date_part(ctx: Context):
    """GROUP BY year(l_shipdate) -- temporal extraction producing the key."""
    sql = (
        "SELECT EXTRACT(YEAR FROM l_shipdate) AS y, COUNT(*) AS n "
        "FROM lineitem GROUP BY EXTRACT(YEAR FROM l_shipdate)"
    )

    def pyarrow(t: pa.Table) -> pa.Table:
        y = pc.year(t["l_shipdate"])
        a = pa.table({"y": y}).group_by("y").aggregate([([], "count_all")])
        return pa.table({"y": pc.cast(a["y"], pa.int64()), "n": a["count_all"]})

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow)


@expressions.case("op-expr-date-arith")
def expr_date_arith(ctx: Context):
    """A predicate over interval arithmetic on two date columns."""
    sql = "SELECT COUNT(*) AS n FROM lineitem WHERE l_receiptdate > l_commitdate + INTERVAL '7' DAY"
    return sql_fanout(ctx, sql)


@expressions.case("op-expr-cast-chain")
def expr_cast_chain(ctx: Context):
    """Explicit casts across three widths inside a reduction."""
    sql = (
        "SELECT SUM(CAST(CAST(l_quantity AS INTEGER) AS BIGINT)) AS s, "
        "SUM(CAST(l_extendedprice AS BIGINT)) AS p FROM lineitem"
    )
    return sql_fanout(ctx, sql)
