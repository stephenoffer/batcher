"""Relative position: percent_rank and cume_dist.

Both answer "where does this row sit in its partition", on a 0-1 scale, and they differ
at the ends. `percent_rank` starts at 0 for the smallest row; `cume_dist` ends at 1 for
the largest. Reporting one as the other shifts every percentile by a row.

    python examples/windows/percent_rank_and_cume_dist.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    customer = tpch("customer").select("c_custkey", "c_acctbal").head(1_000)

    placed = customer.with_columns(
        pct=bt.percent_rank().over(order_by=[("c_acctbal", False)]),
        cume=bt.cume_dist().over(order_by=[("c_acctbal", False)]),
    ).sort("c_acctbal")

    result = placed.to_pydict()
    print("lowest:", round(result["pct"][0], 4), round(result["cume"][0], 4))
    print("highest:", round(result["pct"][-1], 4), round(result["cume"][-1], 4))

    # Both are proportions.
    assert all(0.0 <= value <= 1.0 for value in result["pct"])
    assert all(0.0 < value <= 1.0 for value in result["cume"])

    # The ends are where they differ.
    assert result["pct"][0] == 0.0
    assert abs(result["cume"][-1] - 1.0) < 1e-9

    # Both are non-decreasing along the ordering.
    assert result["pct"] == sorted(result["pct"])
    assert result["cume"] == sorted(result["cume"])

    # The top 10% by balance, expressed with the relative rank.
    top = placed.filter(col("cume") > 0.9).count()
    assert abs(top - 100) <= 5


if __name__ == "__main__":
    main()
