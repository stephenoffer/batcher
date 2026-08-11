"""A write that either lands completely or not at all.

A reader must never see half a write. For a file sink that means writing to a staging path
and renaming; for a transactional sink it means a commit. Checking that the reader's row
count is only ever one of the two valid values is the property that matters.

    python examples/io/write_modes_and_atomicity.py
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
    orders = tpch("orders").select("o_orderkey", "o_totalprice")

    with tempfile.TemporaryDirectory() as directory:
        table = str(Path(directory) / "orders_delta")

        orders.head(1_000).write.delta(table)
        first = bt.read.delta(table).count()

        orders.head(5_000).write.delta(table, mode="overwrite")
        second = bt.read.delta(table).count()

        print(f"{first} then {second} rows")
        assert first == 1_000
        assert second == 5_000

        # Each write is one commit, so the log has one entry per write and a reader is
        # always looking at one of them, never at a mixture.
        log = sorted((Path(table) / "_delta_log").glob("*.json"))
        print(f"{len(log)} commits")
        assert len(log) == 2

        # Every historical version is a complete, consistent snapshot.
        assert bt.read.delta(table, version=0).count() == 1_000
        assert bt.read.delta(table, version=1).count() == 5_000

        # The file sink's manifest reports what it wrote, which is the same check for a
        # non-transactional target.
        parquet = str(Path(directory) / "orders.parquet")
        manifest = orders.head(2_000).write.parquet(parquet)
        print("manifest:", type(manifest).__name__)
        back = bt.read.parquet(parquet)
        assert back.count() == 2_000

        # And the totals reconcile, which catches a truncated write a row count would not.
        expected = orders.head(2_000).agg(t=col("o_totalprice").sum()).to_pydict()["t"][0]
        actual = back.agg(t=col("o_totalprice").sum()).to_pydict()["t"][0]
        assert abs(expected - actual) < 1e-3


if __name__ == "__main__":
    main()
