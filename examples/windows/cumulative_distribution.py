"""Building a cumulative share, the Pareto "80% of revenue" chart.

A running total over a descending sort, divided by the grand total, gives the cumulative
share. The row where it crosses 0.8 is the answer to "how many customers make up 80% of
revenue", and it is one window away.

    python examples/windows/cumulative_distribution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    per_customer = (
        tpch("orders")
        .group_by("o_custkey")
        .agg(revenue=col("o_totalprice").sum())
        .sort("revenue", descending=True)
    )

    ranked = per_customer.with_columns(
        rank=bt.row_number().over(order_by=[("revenue", True)]),
        running=col("revenue").sum().over(order_by=[("revenue", True)], frame=(None, 0)),
        total=col("revenue").sum().over(),
    ).with_columns(share=col("running") / col("total"))

    result = ranked.sort("rank").to_pydict()
    print("top 3 shares:", [round(value, 5) for value in result["share"][:3]])

    # The cumulative share climbs from the first customer's share to exactly 1.
    assert result["share"] == sorted(result["share"])
    assert abs(result["share"][-1] - 1.0) < 1e-9
    assert result["rank"] == list(range(1, len(result["rank"]) + 1))

    # Where does 80% fall?
    crossing = ranked.filter(col("share") >= 0.8).sort("rank").head(1).to_pydict()
    customers = crossing["rank"][0]
    total_customers = per_customer.count()
    print(
        f"{customers} of {total_customers} customers make up 80% of revenue "
        f"({customers / total_customers:.1%})"
    )
    assert 0 < customers <= total_customers


if __name__ == "__main__":
    main()
