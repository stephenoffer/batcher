"""Several different windows in one projection.

Each window expression carries its own partition and frame, so a single `select` can compute
a per-customer rank, a global percentile and a running total at once. They share one pass
over the sorted data rather than costing one each.

    python examples/windows/multiple_windows.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_custkey", "o_orderkey", "o_orderdate", "o_totalprice")

    enriched = orders.with_columns(
        # Per customer, by date.
        nth_order=bt.row_number().over(
            partition_by=["o_custkey"], order_by=["o_orderdate", "o_orderkey"]
        ),
        customer_total=col("o_totalprice").sum().over(partition_by=["o_custkey"]),
        # Global, by price.
        global_rank=bt.rank().over(order_by=[("o_totalprice", True)]),
        # Per customer, cumulative.
        running=col("o_totalprice")
        .sum()
        .over(
            partition_by=["o_custkey"],
            order_by=["o_orderdate", "o_orderkey"],
            frame=(None, 0),
        ),
    )

    sample_customer = orders.head(1).to_pydict()["o_custkey"][0]
    rows = enriched.filter(col("o_custkey") == sample_customer).sort("nth_order").to_pydict()
    print("customer", sample_customer)
    for index in range(len(rows["nth_order"])):
        print(
            f"  #{rows['nth_order'][index]} {rows['o_orderdate'][index]} "
            f"{rows['o_totalprice'][index]:>12,.2f} running {rows['running'][index]:>13,.2f}"
        )

    # The per-customer sequence is 1..n.
    assert rows["nth_order"] == list(range(1, len(rows["nth_order"]) + 1))

    # The running total ends at the customer total.
    assert abs(rows["running"][-1] - rows["customer_total"][0]) < 1e-6

    # The customer total is constant across the partition.
    assert len(set(rows["customer_total"])) == 1

    # The global rank is over every row, so it exceeds the per-customer count.
    assert max(enriched.select("global_rank").to_pydict()["global_rank"]) >= orders.count() - 5


if __name__ == "__main__":
    main()
