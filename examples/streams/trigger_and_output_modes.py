"""Trigger and output mode: how often a streaming query fires, and what it emits.

The trigger decides when a batch runs. The output mode decides what goes to the sink each
time: everything, only what changed, or only what is new. Getting the mode wrong is how a
sink ends up with a full snapshot every minute.

    python examples/streams/trigger_and_output_modes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    # The surface, without a broker: triggers and modes are plain values you construct
    # and hand to a streaming write.
    every_minute = bt.Trigger.processing_time("1 minute")
    print("trigger:", every_minute)
    assert every_minute is not None

    once = bt.Trigger.once() if hasattr(bt.Trigger, "once") else None
    print("once trigger:", once)

    modes = [name for name in dir(bt.OutputMode) if not name.startswith("_")]
    print("output modes:", modes)
    assert modes

    # What each mode means, demonstrated on bounded data: the aggregate a streaming query
    # maintains is the same one a batch query computes.
    orders = tpch("orders").select("o_orderdate", "o_totalprice")
    running = (
        orders.with_columns(window=col("o_orderdate").dt.truncate("month"))
        .group_by("window")
        .agg(revenue=col("o_totalprice").sum(), orders=bt.count())
        .sort("window")
    )
    complete = running.to_pydict()
    print(f"{len(complete['window'])} windows in the complete view")

    # "Complete" is every window; "append" would be only the windows that closed since
    # the last trigger. Both describe the same underlying aggregate.
    assert sum(complete["orders"]) == orders.count()

    latest_window = complete["window"][-1]
    appended = running.filter(col("window") == latest_window).to_pydict()
    assert len(appended["window"]) == 1
    assert appended["orders"][0] <= sum(complete["orders"])


if __name__ == "__main__":
    main()
