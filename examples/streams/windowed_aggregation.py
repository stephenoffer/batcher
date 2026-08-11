"""Time windows over an event stream, computed as a grouped aggregate.

A tumbling window is a truncation of the timestamp used as a group key. Seeing it that way
is useful: windowing is not a separate engine feature, it is a key derived from time, and
everything you know about group-by still applies.

    python examples/streams/windowed_aggregation.py
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

    # Tumbling monthly windows: every event falls in exactly one.
    tumbling = (
        orders.with_columns(window=col("o_orderdate").dt.truncate("month"))
        .group_by("window")
        .agg(events=bt.count(), value=col("o_totalprice").sum())
        .sort("window")
    )
    result = tumbling.to_pydict()
    print(result["window"][:3], result["events"][:3])

    assert sum(result["events"]) == orders.count()
    assert result["window"] == sorted(result["window"])

    # A sliding view over the tumbling windows: a three-window trailing total. This is
    # the "hopping window" of a streaming engine, expressed as a frame.
    smoothed = tumbling.with_columns(
        trailing=col("value").sum().over(order_by=["window"], frame=(-2, 0)),
        trailing_events=col("events").sum().over(order_by=["window"], frame=(-2, 0)),
    ).sort("window")
    rolled = smoothed.to_pydict()
    print([round(value) for value in rolled["trailing"][:4]])

    # The first window's trailing total is just itself; the third onwards spans three.
    assert abs(rolled["trailing"][0] - rolled["value"][0]) < 1e-6
    assert rolled["trailing_events"][2] == sum(rolled["events"][0:3])

    # A trailing sum over positive values never falls below the current value.
    assert all(
        trailing >= value - 1e-6
        for value, trailing in zip(rolled["value"], rolled["trailing"], strict=True)
    )


if __name__ == "__main__":
    main()
