"""Reading a directory of files as one relation, and controlling what is included.

A directory read takes every file the reader understands. That is convenient until someone
drops a `_SUCCESS` marker or a temp file beside the data — which is why a glob that names
the extension is the safer habit.

    python examples/io/reading_a_directory.py
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
    supplier = tpch("supplier").select("s_suppkey", "s_name", "s_acctbal")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "supplier"
        root.mkdir()
        for index in range(4):
            supplier.slice(index * 500, 500).write.parquet(str(root / f"part-{index}.parquet"))

        # A glob naming the extension.
        by_glob = bt.read.parquet(str(root / "*.parquet"))
        print("rows:", by_glob.count())
        assert by_glob.count() == 2_000

        # The directory form reads the same files.
        by_directory = bt.read.parquet(str(root))
        assert by_directory.count() == by_glob.count()

        # Aggregates span the files, because they are one relation.
        total = by_glob.agg(t=col("s_acctbal").sum()).to_pydict()["t"][0]
        expected = supplier.head(2_000).agg(t=col("s_acctbal").sum()).to_pydict()["t"][0]
        assert abs(total - expected) < 1e-6

        # A narrower glob reads a subset.
        subset = bt.read.parquet(str(root / "part-0*.parquet"))
        assert subset.count() == 500

        # A marker file beside the data is why the extension in the glob matters.
        (root / "_SUCCESS").write_text("")
        assert bt.read.parquet(str(root / "*.parquet")).count() == 2_000


if __name__ == "__main__":
    main()
