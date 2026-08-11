"""Delta Lake: transactional appends and overwrites over a real table.

Every write is one commit. That is what makes a half-finished write invisible to readers:
they keep seeing the previous version until the commit lands, so there is no window where
a query sees part of a batch.

    python examples/io/delta_roundtrip.py
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
    orders = tpch("orders").select("o_orderkey", "o_custkey", "o_totalprice", "o_orderstatus")

    with tempfile.TemporaryDirectory() as directory:
        table = str(Path(directory) / "orders_delta")

        # First commit.
        orders.head(1_000).write.delta(table)
        assert bt.read.delta(table).count() == 1_000

        # Append: a second commit, both visible.
        orders.slice(1_000, 500).write.delta(table, mode="append")
        assert bt.read.delta(table).count() == 1_500

        # Overwrite: the table's contents are replaced, not deleted and rewritten.
        orders.head(200).write.delta(table, mode="overwrite")
        current = bt.read.delta(table)
        print("after overwrite:", current.count())
        assert current.count() == 200

        # The data is queryable as an ordinary Dataset.
        summary = (
            current.group_by("o_orderstatus")
            .agg(orders=bt.count(), value=col("o_totalprice").sum())
            .sort("o_orderstatus")
            .to_pydict()
        )
        print(summary)
        assert sum(summary["orders"]) == 200

        # The transaction log is a directory of JSON commits, one per write.
        log = Path(table) / "_delta_log"
        commits = sorted(log.glob("*.json"))
        print(f"{len(commits)} commits")
        assert len(commits) == 3


if __name__ == "__main__":
    main()
