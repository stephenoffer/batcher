"""Save modes: overwrite, and why a plain file sink has no append.

`mode="append"` works for transactional sinks — Delta, Iceberg, Hudi — because there is a
table to commit to. A plain Parquet sink has no such thing, so appending would mean
rewriting the whole output, and the writer refuses rather than doing that silently.

The two honest alternatives are below: write each batch to its own path under a directory
and read the directory back as one relation, or use a sink where a commit is a real thing.

    python examples/io/save_modes_and_transactional_append.py
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
    first = supplier.head(100)
    second = supplier.slice(100, 100)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        # Overwrite is the file sink's one mode: it replaces the target outright.
        target = str(root / "suppliers.parquet")
        first.write.parquet(target, mode="overwrite")
        assert bt.read.parquet(target).count() == 100
        second.write.parquet(target, mode="overwrite")
        assert bt.read.parquet(target).count() == 100

        # Asking a file sink to append is an error, with the two alternatives named.
        try:
            second.write.parquet(target, mode="append")
        except Exception as error:
            print("refused:", str(error)[:90])
        else:
            raise AssertionError("a plain parquet sink must not accept mode='append'")

        # Alternative one: one file per batch, read back as a single relation.
        accumulating = root / "suppliers"
        accumulating.mkdir()
        first.write.parquet(str(accumulating / "batch-0.parquet"))
        second.write.parquet(str(accumulating / "batch-1.parquet"))
        combined = bt.read.parquet(str(accumulating / "*.parquet"))
        print("accumulated rows:", combined.count())
        assert combined.count() == 200

        # Alternative two: a transactional sink, where append is a commit.
        table = str(root / "suppliers_delta")
        first.write.delta(table)
        second.write.delta(table, mode="append")
        assert bt.read.delta(table).count() == 200

        # Both routes agree, and neither duplicates a row.
        keys = sorted(bt.read.delta(table).to_pydict()["s_suppkey"])
        assert len(keys) == len(set(keys)) == 200
        totals = combined.agg(total=col("s_acctbal").sum()).to_pydict()["total"][0]
        committed = bt.read.delta(table).agg(total=col("s_acctbal").sum()).to_pydict()["total"][0]
        assert abs(totals - committed) < 1e-6


if __name__ == "__main__":
    main()
