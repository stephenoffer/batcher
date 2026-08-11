"""Dropping duplicate events without keeping every key forever.

A stream cannot remember every id it has seen, so deduplication needs a bound. Bounding it by
a watermark is the standard answer: remember keys inside the window and forget them once
nothing that late can still arrive.

    python examples/streams/deduplication_in_stream.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    base = tpch("orders").select("o_orderkey", "o_orderdate", "o_totalprice").head(20_000)

    # A stream with genuine duplicates: the same events replayed.
    stream = base.union(base.head(5_000))
    print("events:", stream.count(), "distinct keys:", stream.n_unique("o_orderkey"))
    assert stream.count() == 25_000
    assert stream.n_unique("o_orderkey") == 20_000

    # Unbounded deduplication: correct, and needs every key in memory.
    deduped = stream.drop_duplicates(subset=["o_orderkey"])
    assert deduped.count() == 20_000

    # Bounded deduplication: remember keys only within a watermark window.
    windowed = stream.with_columns(window=col("o_orderdate").dt.truncate("month"))
    per_window = windowed.drop_duplicates(subset=["window", "o_orderkey"])
    print("after windowed dedup:", per_window.count())

    # A duplicate inside the same window is removed; the bound is what keeps state finite.
    assert per_window.count() == 20_000

    # The totals confirm nothing else was lost.
    original_total = base.agg(t=col("o_totalprice").sum()).to_pydict()["t"][0]
    deduped_total = deduped.agg(t=col("o_totalprice").sum()).to_pydict()["t"][0]
    assert abs(original_total - deduped_total) < 1e-3

    # And the duplicated stream really would have double-counted.
    stream_total = stream.agg(t=col("o_totalprice").sum()).to_pydict()["t"][0]
    print(f"raw stream total {stream_total:,.0f} vs deduplicated {deduped_total:,.0f}")
    assert stream_total > deduped_total
    assert bt is not None


if __name__ == "__main__":
    main()
