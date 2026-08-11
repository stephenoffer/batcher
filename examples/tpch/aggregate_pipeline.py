"""A full reporting pipeline over TPC-H, from scan to written report.

This is what the individual examples add up to: read, join, derive, aggregate, rank, and
write — with the checks that make the output trustworthy rather than merely present.

    python examples/tpch/aggregate_pipeline.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")
    orders = tpch("orders")
    customer = tpch("customer")
    nation = tpch("nation")

    report = (
        lineitem.join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .join(customer, left_on="o_custkey", right_on="c_custkey")
        .join(nation, left_on="c_nationkey", right_on="n_nationkey")
        .with_columns(
            revenue=col("l_extendedprice") * (1 - col("l_discount")),
            year=col("o_orderdate").dt.year(),
        )
        .group_by("n_name", "year")
        .agg(
            revenue=col("revenue").sum(),
            # `o_orderkey` was consumed by the join, so the surviving spelling of the
            # order key is `l_orderkey`.
            orders=col("l_orderkey").n_unique(),
            lines=bt.count(),
        )
        .with_columns(revenue_per_order=col("revenue") / col("orders"))
        .sort("n_name", "year")
    )

    result = report.to_pydict()
    print(f"{report.count()} nation/year rows")
    for row in list(zip(result["n_name"], result["year"], result["revenue"], strict=True))[:5]:
        print(f"  {row[0]:<16} {row[1]}  {row[2]:>16,.2f}")

    # Structural checks.
    assert report.count() > 0
    assert all(value > 0 for value in result["revenue"])
    assert all(
        orders <= lines for orders, lines in zip(result["orders"], result["lines"], strict=True)
    )
    pairs = list(zip(result["n_name"], result["year"], strict=True))
    assert pairs == sorted(pairs)
    assert len(set(pairs)) == len(pairs)

    # The total reconciles with an independent aggregate over the same join.
    joined_total = (
        lineitem.join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .join(customer, left_on="o_custkey", right_on="c_custkey")
        .join(nation, left_on="c_nationkey", right_on="n_nationkey")
        .agg(t=(col("l_extendedprice") * (1 - col("l_discount"))).sum())
        .to_pydict()["t"][0]
    )
    assert abs(sum(result["revenue"]) - joined_total) < 1e-2

    # And it lands on disk intact.
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "report.parquet")
        report.write.parquet(path)
        back = bt.read.parquet(path)
        assert back.count() == report.count()
        assert (
            abs(back.agg(t=col("revenue").sum()).to_pydict()["t"][0] - sum(result["revenue"]))
            < 1e-2
        )
        print("report written and verified")


if __name__ == "__main__":
    main()
