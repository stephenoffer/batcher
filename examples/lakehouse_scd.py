"""Lakehouse round-trip plus an SCD type-2 history build.

Two self-contained workflows over a temp directory: a Delta Lake write/append/read
with time travel and a `MERGE` upsert, then a slowly-changing-dimension (type 2)
build that keeps full history for a customer dimension. Both assert on their own
output, so this doubles as a runnable check of the table-format surface.

Run it directly::

    python examples/lakehouse_scd.py
"""

from __future__ import annotations

import os
import tempfile

import batcher as bt


def delta_roundtrip(work: str) -> None:
    """Write, append, time-travel, and upsert a Delta table."""
    table_uri = os.path.join(work, "events")

    # Version 0: the initial overwrite. Version 1: an atomic append.
    bt.from_pydict({"id": [1, 2, 3], "amount": [10, 20, 30]}).write.delta(
        table_uri, mode="overwrite"
    )
    bt.from_pydict({"id": [4], "amount": [40]}).write.delta(table_uri, mode="append")

    latest = bt.read.delta(table_uri).sort("id").to_pydict()
    assert latest["id"] == [1, 2, 3, 4]
    assert latest["amount"] == [10, 20, 30, 40]

    # Time travel: read the table as it was at the first commit.
    v0 = bt.read.delta(table_uri, version=0).sort("id").to_pydict()
    assert v0["id"] == [1, 2, 3]

    # MERGE upsert: id=2 is updated in place, id=5 is inserted.
    bt.from_pydict({"id": [2, 5], "amount": [999, 50]}).write.delta(table_uri, merge_on="id")
    merged = bt.read.delta(table_uri).sort("id").to_pydict()
    assert merged["id"] == [1, 2, 3, 4, 5]
    assert merged["amount"] == [10, 999, 30, 40, 50]

    print("delta:", merged)


def scd_type2_history(work: str) -> None:
    """Build a type-2 dimension that keeps every version of a changed key."""
    dim = os.path.join(work, "customer_dim.parquet")

    # First snapshot opens a current version for each key.
    bt.from_pydict({"id": [1, 2], "city": ["NYC", "LA"]}).scd.type2(
        dim, keys="id", track=["city"], as_of="2024-01-01"
    )
    # Second snapshot: id=1 moved to SF (expire + append); id=2 is unchanged.
    bt.from_pydict({"id": [1, 2], "city": ["SF", "LA"]}).scd.type2(
        dim, keys="id", track=["city"], as_of="2024-06-01"
    )

    history = bt.read.parquet(dim).sort("id", "valid_from")
    out = history.select("id", "city", "valid_from", "is_current").to_pydict()

    # id=1 has two versions (expired NYC, current SF); id=2 has one open version.
    assert out["id"] == [1, 1, 2]
    assert out["city"] == ["NYC", "SF", "LA"]
    assert out["valid_from"] == ["2024-01-01", "2024-06-01", "2024-01-01"]
    assert out["is_current"] == [False, True, True]

    print("scd type-2:", out)


def main() -> None:
    work = tempfile.mkdtemp()
    delta_roundtrip(work)
    scd_type2_history(work)


if __name__ == "__main__":
    main()
