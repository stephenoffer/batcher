"""Pulling the components out of a date column.

Every one of these is a column expression, so extracting the year to group by it costs one
vectorized pass and no materialized intermediate. The alternative — formatting to a string
and slicing it — is slower and loses the type.

    python examples/expr_temporal/date_parts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderkey", "o_orderdate").head(1_000)

    parts = orders.select(
        "o_orderdate",
        year=col("o_orderdate").dt.year(),
        quarter=col("o_orderdate").dt.quarter(),
        month=col("o_orderdate").dt.month(),
        day=col("o_orderdate").dt.day(),
        weekday=col("o_orderdate").dt.weekday(),
        week=col("o_orderdate").dt.weekofyear(),
        ordinal=col("o_orderdate").dt.ordinal_day(),
        iso_year=col("o_orderdate").dt.iso_year(),
    )

    result = parts.head(3).to_pydict()
    print(result)

    # Every part sits in its natural range.
    full = parts.to_pydict()
    assert all(1990 <= value <= 2000 for value in full["year"])
    assert all(1 <= value <= 4 for value in full["quarter"])
    assert all(1 <= value <= 12 for value in full["month"])
    assert all(1 <= value <= 31 for value in full["day"])
    assert all(1 <= value <= 366 for value in full["ordinal"])
    assert all(1 <= value <= 53 for value in full["week"])

    # Quarter and month agree: quarter is the month divided into three-month blocks.
    assert all(
        quarter == (month - 1) // 3 + 1
        for quarter, month in zip(full["quarter"], full["month"], strict=True)
    )

    # And the parts reconstruct the date they came from.
    assert all(
        date.year == year and date.month == month and date.day == day
        for date, year, month, day in zip(
            full["o_orderdate"], full["year"], full["month"], full["day"], strict=True
        )
    )


if __name__ == "__main__":
    main()
