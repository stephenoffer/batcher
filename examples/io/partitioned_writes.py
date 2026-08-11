"""Partitioned output: writing a directory tree a reader can prune.

`partition_by` writes one directory per key value, Hive-style. The payoff is on the read
side: a filter on the partition column skips whole directories without opening a file.
The cost is small files, so partition on something with tens of values, not millions.

Read the result with `read.parquet_dataset`, not `read.parquet`. The partition value
lives in the directory name rather than in the files, so the plain reader returns every
row with that column missing — and warns you that it did.

    python examples/io/partitioned_writes.py
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
    lineitem = tpch("lineitem").select("l_orderkey", "l_shipmode", "l_quantity", "l_extendedprice")

    with tempfile.TemporaryDirectory() as directory:
        root = str(Path(directory) / "lineitem")
        lineitem.write.parquet(root, partition_by=["l_shipmode"])

        # One directory per distinct ship mode.
        written = sorted(p.name for p in Path(root).iterdir() if p.is_dir())
        print(written)
        assert len(written) == lineitem.n_unique("l_shipmode")
        assert all(name.startswith("l_shipmode=") for name in written)

        # The dataset reader reconstructs the partition column from the path.
        back = bt.read.parquet_dataset(root)
        assert back.count() == lineitem.count()
        assert "l_shipmode" in back.columns

        # A filter on the partition key prunes directories rather than rows.
        air = back.filter(col("l_shipmode") == "AIR")
        direct = lineitem.filter(col("l_shipmode") == "AIR")
        print("AIR rows:", air.count())
        assert air.count() == direct.count()

        totals = air.agg(total=col("l_extendedprice").sum()).to_pydict()["total"][0]
        expected = direct.agg(total=col("l_extendedprice").sum()).to_pydict()["total"][0]
        assert abs(totals - expected) < 1e-3


if __name__ == "__main__":
    main()
