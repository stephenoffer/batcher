"""Adding a column to a table that already has data.

A transactional table can widen its schema in a commit, so old rows keep their old shape
and read back with a null for the new column. That is the behaviour you want; the failure
mode to avoid is a writer that silently drops the column instead.

    python examples/lakehouse/schema_evolution_on_write.py
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
        table = str(Path(directory) / "suppliers")

        supplier.head(500).write.delta(table)
        assert bt.read.delta(table).width == 3

        # A later batch carries an extra column.
        widened = supplier.slice(500, 500).with_columns(s_tier=bt.lit("standard"))
        widened.write.delta(table, mode="append", merge_schema=True)

        combined = bt.read.delta(table)
        print("columns after evolution:", combined.columns)
        assert "s_tier" in combined.columns
        assert combined.count() == 1_000

        # Old rows carry a null for the new column; new rows carry the value.
        missing = combined.filter(col("s_tier").is_null()).count()
        present = combined.filter(col("s_tier").is_not_null()).count()
        print(f"{missing} rows predate the column, {present} carry it")
        assert missing == 500
        assert present == 500

        # The old data is otherwise untouched.
        original = supplier.head(500).sort("s_suppkey").to_pydict()
        after = combined.filter(col("s_tier").is_null()).sort("s_suppkey").to_pydict()
        assert after["s_name"] == original["s_name"]


if __name__ == "__main__":
    main()
