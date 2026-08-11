"""Ranking without a window: `top_k`, `nlargest` and a self-join.

A window is the general tool, but a plain top-N does not need one. Knowing the cheaper forms
matters because a window sorts within partitions, and for a single global top-N that is more
work than the answer requires.

    python examples/expr_logic/window_free_ranking.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderkey", "o_totalprice")

    by_top_k = orders.top_k(10, by="o_totalprice").to_pydict()
    by_sort = orders.sort("o_totalprice", descending=True).head(10).to_pydict()
    by_nlargest = orders.nlargest(10, "o_totalprice").to_pydict()

    print("top prices:", [round(value) for value in by_top_k["o_totalprice"][:5]])

    # All three agree, and all three are descending.
    assert by_top_k["o_totalprice"] == by_sort["o_totalprice"]
    assert by_nlargest["o_totalprice"] == by_sort["o_totalprice"]
    assert by_sort["o_totalprice"] == sorted(by_sort["o_totalprice"], reverse=True)

    # The other end.
    smallest = orders.bottom_k(10, by="o_totalprice").to_pydict()
    by_nsmallest = orders.nsmallest(10, "o_totalprice").to_pydict()
    assert smallest["o_totalprice"] == by_nsmallest["o_totalprice"]
    assert smallest["o_totalprice"] == sorted(smallest["o_totalprice"])

    # A threshold instead of a count, when "the top 1%" is the question.
    cutoff = orders.agg(t=bt.quantile(col("o_totalprice"), 0.99)).to_pydict()["t"][0]
    top_percent = orders.filter(col("o_totalprice") >= cutoff)
    share = top_percent.count() / orders.count()
    print(f"top 1% by threshold: {top_percent.count()} rows ({share:.4%})")
    assert 0.005 < share < 0.02

    # The threshold form scales where a sort does not: it never orders the whole table.
    assert top_percent.count() > 10


if __name__ == "__main__":
    main()
