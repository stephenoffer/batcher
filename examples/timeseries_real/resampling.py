"""Resampling a real order stream to daily, weekly and monthly grain.

Resampling is a truncation used as a group key. The one thing to watch is that periods
with no events are simply absent — there is no row for a day nothing happened, which is
usually not what a chart wants.

    python examples/timeseries_real/resampling.py
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

    grains = {}
    for label, unit in (("daily", "day"), ("weekly", "week"), ("monthly", "month")):
        series = (
            orders.with_columns(period=col("o_orderdate").dt.truncate(unit))
            .group_by("period")
            .agg(events=bt.count(), value=col("o_totalprice").sum())
            .sort("period")
        )
        grains[label] = series.to_pydict()
        print(f"{label:<8} {len(grains[label]['period']):>5} periods")

    # A coarser grain has fewer periods and the same total.
    assert len(grains["monthly"]["period"]) < len(grains["weekly"]["period"])
    assert len(grains["weekly"]["period"]) < len(grains["daily"]["period"])
    for label in grains:
        assert sum(grains[label]["events"]) == orders.count()

    totals = [sum(grains[label]["value"]) for label in grains]
    assert max(totals) - min(totals) < 1.0

    # Gaps are real: the daily series has fewer rows than the calendar span has days.
    first = grains["daily"]["period"][0]
    last = grains["daily"]["period"][-1]
    span = (last - first).days + 1
    print(f"{len(grains['daily']['period'])} days with orders across a {span}-day span")
    assert len(grains["daily"]["period"]) <= span


if __name__ == "__main__":
    main()
