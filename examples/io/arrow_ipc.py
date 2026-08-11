"""Arrow IPC: the format with no conversion cost.

Parquet is a storage format and has to be decoded. Arrow IPC is the in-memory layout
written straight to disk, so reading it is close to a memory map. Use it for handoffs
between processes and for intermediate results; use Parquet for anything you keep.

    python examples/io/arrow_ipc.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch


def main() -> None:
    lineitem = tpch("lineitem").select("l_orderkey", "l_quantity", "l_shipmode")

    with tempfile.TemporaryDirectory() as directory:
        arrow_path = str(Path(directory) / "lineitem.arrow")
        parquet_path = str(Path(directory) / "lineitem.parquet")

        lineitem.write.arrow(arrow_path)
        lineitem.write.parquet(parquet_path)

        from_arrow = bt.read.arrow(arrow_path)
        from_parquet = bt.read.parquet(parquet_path)

        assert from_arrow.count() == from_parquet.count() == lineitem.count()
        # Neither format promises to hand rows back in write order, so compare on a
        # defined order. The sort key has to be a *total* one: ordering by a subset of
        # the columns leaves ties free to fall either way, and the comparison then
        # fails on data that is in fact identical.
        keys = ["l_orderkey", "l_quantity", "l_shipmode"]
        assert from_arrow.sort(*keys).to_pydict() == from_parquet.sort(*keys).to_pydict()

        # Uncompressed Arrow is larger on disk; that is the trade for not decoding.
        arrow_size = sum(p.stat().st_size for p in Path(arrow_path).rglob("*") if p.is_file())
        if arrow_size == 0:
            arrow_size = Path(arrow_path).stat().st_size
        print(f"arrow {arrow_size / 1024:.0f} KiB")
        assert arrow_size > 0

        # A zero-copy handoff to pyarrow, with no serialization in between.
        table = lineitem.to_arrow()
        print(table.schema)
        assert table.num_rows == lineitem.count()
        assert bt.from_arrow(table).count() == lineitem.count()


if __name__ == "__main__":
    main()
