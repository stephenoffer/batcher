"""When to stream and when to collect.

`collect` builds the whole result in memory, which is right when the result is small — a
grouped summary usually is. `iter_batches` keeps peak memory at one batch, which is right
when the result is the size of the input.

    python examples/perf/streaming_versus_collect.py
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

    # Small result: collect it.
    summary = lineitem.group_by("l_shipmode").agg(lines=bt.count()).sort("l_shipmode").collect()
    print("summary rows:", summary.num_rows)
    assert summary.num_rows < 20

    # Large result: stream it. Peak memory is one batch, not 200,000 rows.
    projection = lineitem.select("l_orderkey", "l_extendedprice")
    seen = 0
    largest_batch = 0
    for batch in projection.iter_batches(batch_size=8_192):
        seen += batch.num_rows
        largest_batch = max(largest_batch, batch.num_rows)
    print(f"streamed {seen} rows, largest batch {largest_batch}")

    assert seen == projection.count()
    assert largest_batch <= 8_192

    # The same total either way, so the choice is about memory, not correctness.
    streamed_total = 0.0
    for batch in projection.iter_batches(batch_size=8_192):
        streamed_total += sum(batch.column("l_extendedprice").to_pylist())
    collected_total = projection.agg(t=col("l_extendedprice").sum()).to_pydict()["t"][0]
    assert abs(streamed_total - collected_total) < 1e-3

    # A `limit` in front of a `collect` is the third option, and the cheapest of all when
    # you only need a look at the data.
    assert projection.head(5).collect().num_rows == 5


if __name__ == "__main__":
    main()
