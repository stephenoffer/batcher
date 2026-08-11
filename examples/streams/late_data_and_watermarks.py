"""Late-arriving events, and the watermark that decides when a window closes.

A watermark is a promise about how late an event can be. Everything later than it is late
data, and what you do with it — drop, or reopen the window — is a decision the framework
cannot make for you. Counting how much arrives late is the first step.

    python examples/streams/late_data_and_watermarks.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    # Order date is the event time; receipt date stands in for arrival time.
    events = (
        tpch("lineitem")
        .select("l_orderkey", "l_shipdate", "l_receiptdate", "l_quantity")
        .head(50_000)
    )

    lateness = events.with_columns(delay=col("l_receiptdate") - col("l_shipdate"))
    bounds = lateness.agg(typical=col("delay").median(), worst=col("delay").max()).to_pydict()
    print("delay: median", bounds["typical"][0], "worst", bounds["worst"][0])

    # A watermark of 7 days: anything arriving later is late.
    allowed = 7
    on_time = lateness.filter(col("delay") <= allowed)
    late = lateness.filter(col("delay") > allowed)
    print(f"on time {on_time.count()}, late {late.count()}")
    assert on_time.count() + late.count() == events.count()
    assert late.count() > 0

    # The windowed aggregate over the on-time events only.
    windowed = (
        on_time.with_columns(window=col("l_shipdate").dt.truncate("month"))
        .group_by("window")
        .agg(qty=col("l_quantity").sum(), events=bt.count())
        .sort("window")
    )
    closed = windowed.to_pydict()
    print(f"{len(closed['window'])} windows, {sum(closed['events'])} events counted")
    assert sum(closed["events"]) == on_time.count()

    # What a wider watermark would have added, which is the cost of closing early.
    everything = (
        lateness.with_columns(window=col("l_shipdate").dt.truncate("month"))
        .group_by("window")
        .agg(qty=col("l_quantity").sum())
        .sort("window")
        .to_pydict()
    )
    dropped = sum(everything["qty"]) - sum(closed["qty"])
    print(f"quantity dropped by the 7-day watermark: {dropped}")
    assert dropped > 0
    assert bt.lit is not None
    assert dt is not None


if __name__ == "__main__":
    main()
