"""Files whose schemas disagree: what the reader does, and what you must do.

The reader takes its schema from the first file it opens. A column that appears only in
later files is **not** unified in — it is dropped, and the rows still arrive. That is a
silent narrowing, so a pipeline that gains a column upstream keeps working and keeps
ignoring it.

The reliable handling is to read the generations separately and union them, which forces
you to say what the missing value means.

    python examples/io/schema_evolution.py
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
        root = Path(directory) / "orders"
        root.mkdir()

        old_file = root / "part-00.parquet"
        new_file = root / "part-01.parquet"

        # The old generation: two columns.
        orders.head(500).write.parquet(str(old_file))
        # The new generation: a third column was added upstream.
        orders.slice(500, 500).with_columns(o_channel=bt.lit("web")).write.parquet(str(new_file))

        # Reading both at once keeps every row and silently drops the new column.
        combined = bt.read.parquet(str(root / "*.parquet"))
        print("combined columns:", combined.columns)
        assert combined.count() == 1_000
        assert "o_channel" not in combined.columns

        # Read each generation on its own to see what is really there.
        old_side = bt.read.parquet(str(old_file))
        new_side = bt.read.parquet(str(new_file))
        assert "o_channel" not in old_side.columns
        assert "o_channel" in new_side.columns

        # Union them after deciding what the column means where it is absent. `union`
        # needs both sides to agree, which is exactly the forcing function you want.
        aligned = old_side.with_columns(o_channel=bt.lit("unknown")).union(
            new_side.select("o_orderkey", "o_totalprice", "o_channel")
        )
        print("aligned columns:", aligned.columns)
        assert aligned.count() == 1_000
        assert "o_channel" in aligned.columns

        counts = aligned.value_counts("o_channel").sort("o_channel").to_pydict()
        print(counts)
        assert dict(zip(counts["o_channel"], counts["count"], strict=True)) == {
            "unknown": 500,
            "web": 500,
        }

        # The totals are unaffected either way — only the column list narrowed.
        assert (
            abs(
                combined.agg(total=col("o_totalprice").sum()).to_pydict()["total"][0]
                - aligned.agg(total=col("o_totalprice").sum()).to_pydict()["total"][0]
            )
            < 1e-3
        )


if __name__ == "__main__":
    main()
