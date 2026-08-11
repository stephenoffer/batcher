"""Snapshot isolation: a reader sees one version, whatever the writer is doing.

A query pins a version when it starts, so a write that lands mid-read does not change what
the reader sees. That is what makes a long report safe to run against a live table, and it is
observable by pinning the version explicitly.

    python examples/lakehouse/snapshot_isolation.py
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

        orders.head(2_000).write.delta(table)
        snapshot = bt.read.delta(table, version=0)
        snapshot_count = snapshot.count()
        snapshot_total = snapshot.agg(t=col("o_totalprice").sum()).to_pydict()["t"][0]
        print(f"version 0: {snapshot_count} rows")

        # A writer lands two more commits.
        orders.slice(2_000, 1_000).write.delta(table, mode="append")
        orders.slice(3_000, 1_000).write.delta(table, mode="append")

        latest = bt.read.delta(table)
        print(f"latest: {latest.count()} rows")
        assert latest.count() == 4_000

        # The pinned snapshot is unchanged, in count and in value.
        assert snapshot.count() == snapshot_count
        assert (
            abs(snapshot.agg(t=col("o_totalprice").sum()).to_pydict()["t"][0] - snapshot_total)
            < 1e-6
        )

        # Every intermediate version is a complete snapshot, not a delta.
        assert bt.read.delta(table, version=1).count() == 3_000
        assert bt.read.delta(table, version=2).count() == 4_000

        # And the snapshots nest, because these were appends.
        early_keys = set(bt.read.delta(table, version=0).to_pydict()["o_orderkey"])
        late_keys = set(latest.to_pydict()["o_orderkey"])
        assert early_keys < late_keys
        print(
            f"version 0's {len(early_keys)} keys are a strict subset of the latest {len(late_keys)}"
        )


if __name__ == "__main__":
    main()
