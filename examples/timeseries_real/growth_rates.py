"""Period-over-period change, and why the first period is null.

Growth needs a previous value, and the first period has none. A null there is correct; a
zero is a lie that shows up as a spike on every chart. Assert that the first row is null
rather than filling it.

    python examples/timeseries_real/growth_rates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderdate", "o_totalprice")

    monthly = (
        orders.with_columns(period=col("o_orderdate").dt.truncate("month"))
        .group_by("period")
        .agg(revenue=col("o_totalprice").sum())
        .sort("period")
    )

    growth = (
        monthly.with_columns(
            previous=col("revenue").shift(1).over(order_by=["period"]),
        )
        .with_columns(
            change=col("revenue") - col("previous"),
            pct_change=(col("revenue") - col("previous")) / col("previous"),
        )
        .sort("period")
    )

    result = growth.to_pydict()
    print(result["period"][:3])
    print([None if v is None else round(v, 4) for v in result["pct_change"][:4]])

    # The first period has no predecessor.
    assert result["previous"][0] is None
    assert result["pct_change"][0] is None

    # Every later row's percentage change reconciles with the two revenues.
    for index in range(1, len(result["period"])):
        expected = (result["revenue"][index] - result["revenue"][index - 1]) / result["revenue"][
            index - 1
        ]
        assert abs(result["pct_change"][index] - expected) < 1e-9

    # A year-over-year comparison is the same shape with a bigger lag.
    yearly = (
        monthly.with_columns(year_ago=col("revenue").shift(12).over(order_by=["period"]))
        .sort("period")
        .to_pydict()
    )
    assert all(value is None for value in yearly["year_ago"][:12])
    assert yearly["year_ago"][12] == yearly["revenue"][0]


if __name__ == "__main__":
    main()
