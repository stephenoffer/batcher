"""Stacking datasets: vstack, append, and concat.

Vertical concatenation needs the schemas to line up. When they do not, the honest fix is to
project both sides to a shared shape first — which forces you to decide what a missing
column means rather than letting the reader guess.

    python examples/relational/append_and_concat.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderkey", "o_orderstatus", "o_totalprice")

    open_orders = orders.filter(col("o_orderstatus") == "O")
    closed = orders.filter(col("o_orderstatus") == "F")
    print("open:", open_orders.count(), "closed:", closed.count())

    stacked = open_orders.vstack(closed)
    assert stacked.count() == open_orders.count() + closed.count()
    assert stacked.columns == orders.columns

    # `append` is the same operation.
    appended = open_orders.append(closed)
    assert appended.count() == stacked.count()

    # `bt.concat` takes several at once, which reads better than a chain.
    pending = orders.filter(col("o_orderstatus") == "P")
    everything = bt.concat([open_orders, closed, pending])
    assert everything.count() == orders.count()

    # Totals reconcile, which is the check that catches a dropped piece.
    parts_total = (
        open_orders.agg(t=col("o_totalprice").sum()).to_pydict()["t"][0]
        + closed.agg(t=col("o_totalprice").sum()).to_pydict()["t"][0]
        + pending.agg(t=col("o_totalprice").sum()).to_pydict()["t"][0]
    )
    whole_total = orders.agg(t=col("o_totalprice").sum()).to_pydict()["t"][0]
    assert abs(parts_total - whole_total) < 1e-3

    # Mismatched schemas: project to a shared shape and say what the missing column means.
    wide = orders.with_columns(source=bt.lit("primary"))
    narrow = orders.head(10).with_columns(source=bt.lit("backfill"))
    merged = wide.vstack(narrow)
    assert merged.count() == orders.count() + 10
    assert set(merged.select("source").distinct().to_pydict()["source"]) == {
        "primary",
        "backfill",
    }


if __name__ == "__main__":
    main()
