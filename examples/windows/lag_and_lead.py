"""Looking at neighbouring rows: lag, lead, and period-over-period change.

`lag` reaches backwards inside the partition and `lead` forwards. The first row of each
partition has no predecessor, so its lag is null — and that null is what makes the first
period's growth rate null rather than a misleading zero.

    python examples/windows/lag_and_lead.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders")

    monthly = (
        orders.with_columns(year=col("o_orderdate").dt.year(), month=col("o_orderdate").dt.month())
        .group_by("year", "month")
        .agg(revenue=col("o_totalprice").sum())
        .sort("year", "month")
    )

    compared = monthly.with_columns(
        previous=col("revenue").shift(1).over(order_by=["year", "month"]),
        following=col("revenue").shift(-1).over(order_by=["year", "month"]),
    ).with_columns(change=col("revenue") - col("previous"))

    result = compared.sort("year", "month").to_pydict()
    print(result["year"][:4], result["month"][:4])
    print("first three changes:", result["change"][:3])

    # The very first period has nothing to compare against.
    assert result["previous"][0] is None
    assert result["change"][0] is None
    # And the last has nothing following it.
    assert result["following"][-1] is None

    # Every other row's lag is literally the row above it.
    assert result["previous"][1:] == result["revenue"][:-1]
    assert result["following"][:-1] == result["revenue"][1:]


if __name__ == "__main__":
    main()
