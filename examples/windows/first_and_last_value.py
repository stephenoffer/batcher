"""Reaching the endpoints of a window: first_value, last_value, nth_value.

`last_value` is where Batcher deliberately parts company with SQL. The SQL default frame
ends at the current row, so `last_value` there returns the current row unless you widen
the frame — a famous trap. Batcher defaults to the whole partition instead, so it returns
the partition's last value, and you narrow the frame when you want the running form.

    python examples/windows/first_and_last_value.py
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

    daily = (
        orders.group_by("o_orderdate")
        .agg(revenue=col("o_totalprice").sum())
        .sort("o_orderdate")
        .head(20)
    )

    edges = daily.with_columns(
        opening=bt.first_value(col("revenue")).over(order_by=["o_orderdate"]),
        # Widen the frame explicitly, or this returns the current row.
        closing=bt.last_value(col("revenue")).over(order_by=["o_orderdate"], frame=(None, None)),
        # No frame: the whole partition, so this agrees with `closing` above.
        default_closing=bt.last_value(col("revenue")).over(order_by=["o_orderdate"]),
        # Narrowed to end at the current row — the running "latest so far".
        running_latest=bt.last_value(col("revenue")).over(
            order_by=["o_orderdate"], frame=(None, 0)
        ),
        third=bt.nth_value(col("revenue"), 3).over(order_by=["o_orderdate"], frame=(None, None)),
    ).sort("o_orderdate")

    result = edges.to_pydict()
    print("opening:", round(result["opening"][0]), "closing:", round(result["closing"][0]))

    # The opening and closing values are the same on every row of the partition.
    assert len(set(result["opening"])) == 1
    assert len(set(result["closing"])) == 1
    assert abs(result["opening"][0] - result["revenue"][0]) < 1e-6
    assert abs(result["closing"][0] - result["revenue"][-1]) < 1e-6

    # The default frame is the whole partition, so leaving it off changes nothing.
    assert result["default_closing"] == result["closing"]
    # Narrowing the frame to end at the current row gives the running form, which is
    # the current row on every row.
    assert result["running_latest"] == result["revenue"]

    # nth_value picks a specific position in the frame.
    assert abs(result["third"][0] - result["revenue"][2]) < 1e-6


if __name__ == "__main__":
    main()
