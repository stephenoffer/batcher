"""Reading a large result without materializing it: iter_batches.

`collect` builds the whole table in memory. `iter_batches` hands you one Arrow batch at a
time, so peak memory is one batch rather than the result. Use it whenever the result is
larger than the machine, or when the consumer is itself incremental.

    python examples/io/streaming_reads_iter_batches.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem").select("l_orderkey", "l_quantity", "l_extendedprice")

    total_rows = 0
    total_quantity = 0.0
    batches = 0
    for batch in lineitem.iter_batches(batch_size=16_384):
        batches += 1
        total_rows += batch.num_rows
        total_quantity += sum(batch.column("l_quantity").to_pylist())

    print(f"{batches} batches, {total_rows} rows")

    # Streaming sees exactly the rows a collect would.
    assert total_rows == lineitem.count()
    assert batches > 1

    # And computes the same aggregate, one batch at a time.
    in_engine = lineitem.agg(q=col("l_quantity").sum()).to_pydict()["q"][0]
    assert abs(total_quantity - in_engine) < 1e-6

    # `iter_slices` is the same idea at a chosen row count. It also yields Arrow
    # record batches, so the row count is `num_rows` rather than a Dataset `count()`.
    slices = 0
    for piece in lineitem.iter_slices(50_000):
        slices += 1
        assert piece.num_rows <= 50_000
    print(f"{slices} slices")
    assert slices >= 4

    # The aggregate is better done in the engine: this loop exists to show the boundary,
    # not to recommend summing in Python.
    print("engine total:", in_engine)


if __name__ == "__main__":
    main()
