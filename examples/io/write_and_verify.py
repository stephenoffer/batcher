"""Writing a result and proving what landed on disk.

A write that reports success and a file that holds the right rows are two different claims.
Reading the output back and comparing it against the source is the only version of "the
job worked" that means anything.

    python examples/io/write_and_verify.py
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
    report = (
        tpch("lineitem")
        .group_by("l_shipmode", "l_returnflag")
        .agg(lines=bt.count(), revenue=col("l_extendedprice").sum())
        .sort("l_shipmode", "l_returnflag")
    )
    expected = report.to_pydict()

    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "report.parquet")
        manifest = report.write.parquet(path)
        print("manifest:", type(manifest).__name__)

        back = bt.read.parquet(path).sort("l_shipmode", "l_returnflag")
        actual = back.to_pydict()

        # Row count, schema and values all have to match.
        assert back.count() == report.count()
        assert back.columns == report.columns
        assert actual["l_shipmode"] == expected["l_shipmode"]
        assert actual["lines"] == expected["lines"]
        assert all(
            abs(a - b) < 1e-6 for a, b in zip(expected["revenue"], actual["revenue"], strict=True)
        )

        # A totals check that would catch a partial write even if the row count matched.
        assert abs(sum(actual["revenue"]) - sum(expected["revenue"])) < 1e-3
        assert sum(actual["lines"]) == tpch("lineitem").count()

        print(f"verified {back.count()} rows on disk")


if __name__ == "__main__":
    main()
