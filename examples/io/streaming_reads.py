"""Reading in bounded memory: iter_batches, limits, and lazy metadata.

``collect()`` materializes. ``iter_batches()`` does not: it streams Arrow batches through
the pipeline so a table larger than memory still works. The metadata shortcuts go further
and answer some questions without reading any data at all.

    python examples/io/streaming_reads.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import batcher as bt
from batcher import col


def main() -> None:
    big = bt.from_pydict({"i": list(range(1000)), "grp": ["a", "b"] * 500})

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data"
        big.write.parquet(str(path))
        ds = bt.read.parquet(str(path))

        # Stream in batches; memory stays bounded by the batch size, not the table.
        total = 0
        batches = 0
        for batch in ds.iter_batches(batch_size=128):
            total += batch.num_rows
            batches += 1
        print("rows:", total, "batches:", batches)
        assert total == 1000
        assert batches >= 8

        # A filter pushes into the scan, so streaming reads less.
        filtered = sum(b.num_rows for b in ds.filter(col("i") < 10).iter_batches())
        assert filtered == 10

        # `limit` stops early rather than reading everything and discarding.
        head = ds.limit(5).to_pydict()
        assert len(head["i"]) == 5

        # Metadata answers without a scan.
        assert ds.count() == 1000
        print("schema:", ds.schema)
        assert "i" in ds.schema.names

        # The lazy chain is still just a plan until a terminal call.
        plan = ds.filter(col("grp") == "a").select("i")
        assert plan.count() == 500

        # `to_arrow` when you do want it all in memory as one table.
        table = ds.limit(3).to_arrow()
        assert table.num_rows == 3


if __name__ == "__main__":
    main()
