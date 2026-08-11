"""ORC and Avro: the two formats you meet in someone else's warehouse.

ORC is columnar like Parquet and shows up wherever Hive did. Avro is row-oriented, which
makes it good for streams and bad for analytics — reading one column means reading every
row in full. Both are read and written through the same reader and writer surface.

    python examples/io/orc_and_avro.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch


def main() -> None:
    supplier = tpch("supplier").select("s_suppkey", "s_name", "s_nationkey", "s_acctbal")
    expected = supplier.sort("s_suppkey").to_pydict()

    with tempfile.TemporaryDirectory() as directory:
        for fmt in ("orc", "avro", "parquet"):
            target = str(Path(directory) / f"supplier.{fmt}")
            getattr(supplier.write, fmt)(target)
            back = getattr(bt.read, fmt)(target)

            print(f"{fmt}: {back.count()} rows, {back.width} columns")
            assert back.count() == supplier.count()
            assert set(back.columns) == set(supplier.columns)

            # The values survive every format identically — that is the contract.
            restored = back.sort("s_suppkey").to_pydict()
            assert restored["s_name"] == expected["s_name"]
            assert restored["s_nationkey"] == expected["s_nationkey"]
            assert all(
                abs(left - right) < 1e-9
                for left, right in zip(expected["s_acctbal"], restored["s_acctbal"], strict=True)
            )


if __name__ == "__main__":
    main()
