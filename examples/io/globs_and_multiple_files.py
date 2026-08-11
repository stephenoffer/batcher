"""Reading many files as one dataset, and what the glob can and cannot cross.

A `*` matches within one path segment. Crossing directories needs `**`, which is the
difference between reading one partition and reading the table. Format inference stops at
the first `*`, so a globbed path needs a typed reader.

    python examples/io/globs_and_multiple_files.py
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
    orders = tpch("orders").select("o_orderkey", "o_orderdate", "o_totalprice").head(3_000)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        # Three files in two directories, so the two glob depths differ. The names
        # are deliberately not `key=value`: that is a Hive partition layout, and
        # reading one with `read.parquet` drops the partition column (with a
        # warning). Partitioned reads are `read.parquet_dataset`, in their own
        # example; this one is only about how far a glob reaches.
        for index, part in enumerate([orders.head(1000), orders.slice(1000, 1000)]):
            target = root / "batch-a" / f"part-{index}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            part.write.parquet(str(target))
        tail = root / "batch-b" / "part-0.parquet"
        tail.parent.mkdir(parents=True, exist_ok=True)
        orders.slice(2000, 1000).write.parquet(str(tail))

        # One segment: only the first directory.
        one_year = bt.read.parquet(str(root / "batch-a" / "*.parquet"))
        print("first directory:", one_year.count())
        assert one_year.count() == 2000

        # Two levels: every file under the root.
        everything = bt.read.parquet(str(root / "**" / "*.parquet"))
        print("all rows:", everything.count())
        assert everything.count() == 3000

        # The files are read as one dataset, so aggregates span them.
        total = everything.agg(total=col("o_totalprice").sum()).to_pydict()["total"][0]
        expected = orders.agg(total=col("o_totalprice").sum()).to_pydict()["total"][0]
        assert abs(total - expected) < 1e-3


if __name__ == "__main__":
    main()
