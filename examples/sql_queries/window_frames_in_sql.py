"""Window frames spelled out in SQL.

`ROWS BETWEEN` is the frame, and writing it explicitly is worth the words: readers assume a
default and the defaults differ between engines. Being explicit is how a ported query keeps
meaning the same thing.

    python examples/sql_queries/window_frames_in_sql.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    daily = (
        tpch("orders")
        .group_by("o_orderdate")
        .agg(revenue=col("o_totalprice").sum())
        .sort("o_orderdate")
        .head(40)
    )

    framed = bt.sql(
        """
        SELECT
            o_orderdate,
            revenue,
            SUM(revenue) OVER (
                ORDER BY o_orderdate ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS running,
            AVG(revenue) OVER (
                ORDER BY o_orderdate ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ) AS weekly
        FROM daily
        ORDER BY o_orderdate
        """,
        daily=daily,
    ).to_pydict()

    print([round(value) for value in framed["running"][:4]])

    # The running total is non-decreasing and ends at the grand total.
    assert framed["running"] == sorted(framed["running"])
    assert abs(framed["running"][-1] - sum(framed["revenue"])) < 1e-3

    # The seven-row mean is checkable by hand from the seventh row onwards.
    window = framed["revenue"][0:7]
    assert abs(framed["weekly"][6] - sum(window) / 7) < 1e-6

    # And the DataFrame spelling agrees.
    equivalent = (
        daily.with_columns(
            running=col("revenue").sum().over(order_by=["o_orderdate"], frame=(None, 0)),
            weekly=col("revenue").mean().over(order_by=["o_orderdate"], frame=(-6, 0)),
        )
        .sort("o_orderdate")
        .to_pydict()
    )

    assert all(
        abs(a - b) < 1e-6 for a, b in zip(framed["running"], equivalent["running"], strict=True)
    )
    assert all(
        abs(a - b) < 1e-6 for a, b in zip(framed["weekly"], equivalent["weekly"], strict=True)
    )


if __name__ == "__main__":
    main()
