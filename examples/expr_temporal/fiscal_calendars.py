"""Reporting on a fiscal year that does not start in January.

A fiscal calendar is a shifted calendar, so every fiscal period is a calendar period computed
on a shifted date. Shifting the date once and reusing the ordinary date parts is much less
error-prone than writing fiscal arithmetic for each part.

    python examples/expr_temporal/fiscal_calendars.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderdate", "o_totalprice")

    # A fiscal year starting in April: shift back three months and use the ordinary parts.
    fiscal = orders.with_columns(
        shifted=col("o_orderdate").dt.offset_by("-3mo"),
    ).with_columns(
        fiscal_year=col("shifted").dt.year(),
        fiscal_quarter=col("shifted").dt.quarter(),
        calendar_year=col("o_orderdate").dt.year(),
        month=col("o_orderdate").dt.month(),
    )

    sample = fiscal.head(5).to_pydict()
    for row in zip(
        sample["o_orderdate"],
        sample["calendar_year"],
        sample["fiscal_year"],
        sample["fiscal_quarter"],
        strict=True,
    ):
        print(f"  {row[0]}  calendar {row[1]}  fiscal {row[2]} Q{row[3]}")

    values = fiscal.to_pydict()

    # January to March belong to the previous fiscal year.
    for index in range(len(values["month"])):
        month = values["month"][index]
        expected = values["calendar_year"][index] - (1 if month <= 3 else 0)
        assert values["fiscal_year"][index] == expected

    # April is fiscal Q1, and the quarters are still 1..4.
    assert set(values["fiscal_quarter"]) <= {1, 2, 3, 4}
    april_quarters = {
        values["fiscal_quarter"][index]
        for index in range(len(values["month"]))
        if values["month"][index] == 4
    }
    assert april_quarters == {1}

    # The report reconciles: every order lands in exactly one fiscal period.
    report = (
        fiscal.group_by("fiscal_year", "fiscal_quarter")
        .agg(orders=bt.count(), revenue=col("o_totalprice").sum())
        .sort("fiscal_year", "fiscal_quarter")
    )
    result = report.to_pydict()
    print(f"{report.count()} fiscal periods")
    assert sum(result["orders"]) == orders.count()


if __name__ == "__main__":
    main()
