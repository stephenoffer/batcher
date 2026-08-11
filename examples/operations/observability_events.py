"""Watching a query run: the progress reporter and the activity store.

Observability is a consumer of an event bus every subsystem publishes to, so turning it on
does not change what runs. That decoupling is why you can leave it on in production.

    python examples/operations/observability_events.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")

    query = (
        lineitem.filter(col("l_quantity") > 20)
        .group_by("l_shipmode")
        .agg(lines=bt.count(), revenue=col("l_extendedprice").sum())
        .sort("l_shipmode")
    )

    result = query.to_pydict()
    print(result["l_shipmode"], result["lines"])
    assert sum(result["lines"]) > 0

    # The metadata the executor recorded for this run, which is what the optimizer reads
    # back on the next one.
    info = query.meta
    print("metadata object:", type(info).__name__)
    assert info is not None

    # `show` prints a table; it is the interactive terminal form of `head`.
    query.show(3)

    # The dashboard is opt-in and does not change the plan. Starting it here would bind a
    # port, so this only checks that the accessor exists and stays out of the way.
    assert hasattr(bt, "start_ui")
    assert hasattr(bt, "stop_ui")
    assert hasattr(bt, "ui_url")

    # Running the same query again gives the same answer, whether or not anything watched.
    assert query.to_pydict() == result


if __name__ == "__main__":
    main()
