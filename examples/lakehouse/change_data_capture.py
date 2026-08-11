"""Applying a change feed: inserts, updates and deletes in one commit.

A CDC batch is not an append — it carries operations. Applying it means matching on the key
and doing different things per operation, which a `MERGE` expresses in one commit rather
than as three passes that can each half-fail.

    python examples/lakehouse/change_data_capture.py
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
    customer = tpch("customer").select("c_custkey", "c_name", "c_acctbal")

    with tempfile.TemporaryDirectory() as directory:
        table = str(Path(directory) / "customers")
        customer.head(1_000).write.delta(table)
        assert bt.read.delta(table).count() == 1_000

        # A change batch: 50 updates to existing keys and 50 new keys.
        updates = customer.slice(950, 100).with_columns(c_acctbal=col("c_acctbal") * 2.0)
        updates.write.delta(table, merge_on="c_custkey")

        after = bt.read.delta(table)
        print("rows after the change batch:", after.count())
        assert after.count() == 1_050

        keys = after.to_pydict()["c_custkey"]
        assert len(keys) == len(set(keys))

        # The updated rows really did change.
        touched = set(updates.to_pydict()["c_custkey"])
        existing = sorted(touched & set(customer.head(1_000).to_pydict()["c_custkey"]))
        assert len(existing) == 50

        original = (
            customer.head(1_000)
            .filter(col("c_custkey").is_in(existing))
            .sort("c_custkey")
            .to_pydict()
        )
        current = after.filter(col("c_custkey").is_in(existing)).sort("c_custkey").to_pydict()
        assert all(
            abs(new - old * 2.0) < 1e-6
            for old, new in zip(original["c_acctbal"], current["c_acctbal"], strict=True)
        )

        # Deletes are a filter plus an overwrite, which is still one commit.
        survivors = after.filter(~col("c_custkey").is_in(existing))
        survivors.write.delta(table, mode="overwrite")
        assert bt.read.delta(table).count() == 1_000


if __name__ == "__main__":
    main()
