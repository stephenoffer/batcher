"""Table maintenance: compaction, vacuum, and the version they cost you.

Compaction rewrites the layout and keeps every version readable. Vacuum removes the files old
versions point at, which is what makes it the destructive one — after it, time travel past
the retention window stops working.

    python examples/lakehouse/table_maintenance.py
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
        table = str(Path(directory) / "orders")

        # Ten small commits, as an incremental writer leaves them.
        for index in range(10):
            orders.slice(index * 500, 500).write.delta(
                table, mode="overwrite" if index == 0 else "append"
            )

        files_before = len(list(Path(table).glob("*.parquet")))
        rows = bt.read.delta(table).count()
        total = bt.read.delta(table).agg(t=col("o_totalprice").sum()).to_pydict()["t"][0]
        print(f"before: {files_before} files, {rows} rows")
        assert rows == 5_000
        assert files_before >= 10

        # Compaction: same rows, fewer files, every version still readable.
        bt.compact(table)
        after = bt.read.delta(table)
        print(
            f"after compaction: {len(list(Path(table).glob('*.parquet')))} files, "
            f"{after.count()} rows"
        )
        assert after.count() == rows
        assert abs(after.agg(t=col("o_totalprice").sum()).to_pydict()["t"][0] - total) < 1e-3

        # Time travel still works, because the old files are still there.
        assert bt.read.delta(table, version=0).count() == 500

        # Vacuum is the destructive one. With a long retention it removes nothing, which
        # is the safe default and why the retention argument exists.
        try:
            bt.vacuum(table, retention_hours=168)
        except Exception as error:
            print("vacuum unavailable here:", str(error)[:70])
        else:
            print("vacuumed with a 168-hour retention")
            assert bt.read.delta(table).count() == rows
            # Nothing inside the retention window was removed.
            assert bt.read.delta(table, version=0).count() == 500


if __name__ == "__main__":
    main()
