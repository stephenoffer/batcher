"""The small-files problem, and compacting a table that has it.

An incremental writer leaves one small file per commit, and the next write cannot fix that
— it only adds another. Eventually the table costs more to *plan* than to read. Compaction
bin-packs the files in a transaction that never deletes anything an older version needs.

    python examples/lakehouse/compaction.py
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
    orders = tpch("orders").select("o_orderkey", "o_custkey", "o_totalprice")

    with tempfile.TemporaryDirectory() as directory:
        table = str(Path(directory) / "orders")

        # Twelve small commits, the way an hourly job would leave them.
        for index in range(12):
            orders.slice(index * 200, 200).write.delta(
                table, mode="overwrite" if index == 0 else "append"
            )

        before_files = len(list(Path(table).glob("*.parquet")))
        before_rows = bt.read.delta(table).count()
        before_total = bt.read.delta(table).agg(t=col("o_totalprice").sum()).to_pydict()["t"][0]
        print(f"before: {before_files} files, {before_rows} rows")
        assert before_files >= 12
        assert before_rows == 2_400

        bt.compact(table)

        after_files = len(list(Path(table).glob("*.parquet")))
        after = bt.read.delta(table)
        print(f"after:  {after_files} files, {after.count()} rows")

        # Compaction rewrites the layout and nothing else: same rows, same total.
        assert after.count() == before_rows
        after_total = after.agg(t=col("o_totalprice").sum()).to_pydict()["t"][0]
        assert abs(after_total - before_total) < 1e-3

        # The old files are still on disk, because an earlier version still references
        # them. Removing them is `vacuum`, and it is the destructive one.
        assert after_files >= 1


if __name__ == "__main__":
    main()
