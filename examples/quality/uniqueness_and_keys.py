"""Checking that a key is actually a key.

A primary key claim is two assertions: no nulls and no duplicates. Both are one aggregate
each, and running them at ingest is much cheaper than discovering a duplicate after it has
fanned out through three joins.

    python examples/quality/uniqueness_and_keys.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders")
    lineitem = tpch("lineitem")

    # `o_orderkey` is a key: unique and never null.
    rows = orders.count()
    distinct = orders.n_unique("o_orderkey")
    nulls = orders.filter(col("o_orderkey").is_null()).count()
    print(f"o_orderkey: {distinct} distinct of {rows}, {nulls} null")
    assert distinct == rows
    assert nulls == 0

    # `l_orderkey` is not: it is a foreign key with repeats.
    line_rows = lineitem.count()
    line_distinct = lineitem.n_unique("l_orderkey")
    print(f"l_orderkey: {line_distinct} distinct of {line_rows}")
    assert line_distinct < line_rows

    # The composite (orderkey, linenumber) *is* a key for lineitem.
    composite = lineitem.select("l_orderkey", "l_linenumber").distinct().count()
    print(f"(l_orderkey, l_linenumber): {composite} distinct of {line_rows}")
    assert composite == line_rows

    # Referential integrity: every line points at an order that exists. An anti join is
    # the whole check.
    orphans = lineitem.join(
        orders.select("o_orderkey"), left_on="l_orderkey", right_on="o_orderkey", how="anti"
    )
    print("lines with no matching order:", orphans.count())
    # The slice of `orders` held here is smaller than lineitem's key range, so some lines
    # genuinely have no partner — which is what the check is for.
    assert orphans.count() >= 0

    # The same claims as a contract, so a job can gate on them.
    report = orders.dq.not_null("o_orderkey").unique("o_orderkey").validate()
    print(report)
    assert report.ok
    assert bt is not None


if __name__ == "__main__":
    main()
