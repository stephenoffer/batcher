"""Processing only what is new since the last run.

A high-water mark is the whole mechanism: record the largest key you processed, and next time
read only what is above it. The two things that break it are a non-monotonic key and a
half-committed run, so the mark is written after the output, never before.

    python examples/relational/incremental_processing.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    source = tpch("orders").select("o_orderkey", "o_totalprice").sort("o_orderkey")

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "processed"
        output.mkdir()

        watermark = 0
        batches = 0
        for _ in range(3):
            new_rows = source.filter(col("o_orderkey") > watermark).head(20_000)
            if new_rows.count() == 0:
                break

            # Write the output first...
            target = str(output / f"batch-{batches}.parquet")
            new_rows.write.parquet(target)

            # ...then advance the mark. A crash between the two costs a re-run of one
            # batch, which is idempotent; the other order loses data.
            watermark = new_rows.agg(m=col("o_orderkey").max()).to_pydict()["m"][0]
            batches += 1
            print(f"batch {batches}: {new_rows.count()} rows, watermark now {watermark}")

        assert batches == 3

        processed = bt.read.parquet(str(output / "*.parquet"))
        print("total processed:", processed.count())
        assert processed.count() == 60_000

        # No row processed twice, which is what the strict inequality buys.
        keys = processed.to_pydict()["o_orderkey"]
        assert len(keys) == len(set(keys))

        # And nothing below the mark was skipped.
        expected = source.filter(col("o_orderkey") <= watermark).count()
        assert processed.count() == expected


if __name__ == "__main__":
    main()
