"""Defining an expression once and using it several times.

An `Expr` is a value, so it can be assigned, passed and reused. That is what lets a business
rule live in one place instead of being retyped into four queries — and it costs nothing,
because the plan is built from the same node either way.

    python examples/expr_logic/expression_reuse.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")

    # The business rules, defined once.
    revenue = (col("l_extendedprice") * (1 - col("l_discount"))).alias("revenue")
    is_bulk = col("l_quantity") > 30
    is_returned = col("l_returnflag") == "R"

    # Reused in a projection...
    projected = lineitem.with_columns(revenue=revenue, bulk=is_bulk)
    assert "revenue" in projected.columns

    # ...in a filter...
    bulk_lines = lineitem.filter(is_bulk)
    assert 0 < bulk_lines.count() < lineitem.count()

    # ...and inside an aggregate.
    summary = lineitem.agg(
        total=revenue.sum(),
        bulk_lines=bt.count_if(is_bulk),
        returned_revenue=bt.when(is_returned).then(revenue).otherwise(0.0).sum(),
    ).to_pydict()
    print({name: round(value[0], 2) for name, value in summary.items()})

    # The same definition gives the same answer wherever it is used.
    assert summary["bulk_lines"][0] == bulk_lines.count()
    assert summary["returned_revenue"][0] <= summary["total"][0]

    # Composing expressions builds bigger ones, still without executing anything.
    combined = is_bulk & is_returned
    both = lineitem.filter(combined).count()
    assert both <= bulk_lines.count()
    print("bulk and returned:", both)

    # A helper that returns an expression is the natural unit of reuse.
    def over(threshold: float):
        return col("l_extendedprice") > threshold

    for threshold in (10_000.0, 50_000.0):
        count = lineitem.filter(over(threshold)).count()
        print(f"  over {threshold:>10,.0f}: {count:>7} lines")
        assert count <= lineitem.count()
    assert lineitem.filter(over(50_000.0)).count() <= lineitem.filter(over(10_000.0)).count()


if __name__ == "__main__":
    main()
