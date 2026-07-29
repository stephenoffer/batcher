"""Deduplication: exact keys, whole rows, and keeping a chosen survivor.

"Remove duplicates" is under-specified until you say *which* copy survives. Keeping an
arbitrary one is how a pipeline becomes non-deterministic; keeping the latest by a
timestamp is almost always what was meant.

    python examples/dataset/deduplication.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    events = bt.from_pydict(
        {
            "id": [1, 1, 2, 2, 3],
            "version": [1, 2, 1, 3, 1],
            "payload": ["a", "b", "c", "d", "e"],
        }
    )

    # Whole-row distinct: only fully identical rows collapse. Nothing here does.
    assert events.distinct().count() == 5
    dupes = bt.from_pydict({"v": [1, 1, 2]})
    assert dupes.distinct().count() == 2
    assert dupes.unique().count() == 2

    # Distinct on a subset of columns.
    by_id = events.drop_duplicates(["id"]).to_pydict()
    print("one per id:", by_id)
    assert sorted(by_id["id"]) == [1, 2, 3]

    # The version that matters: keep the *latest* row per key, deterministically.
    latest = (
        events.sort("id", "version")
        .with_columns(rank=bt.row_number().over(partition_by=["id"], order_by=[("version", True)]))
        .filter(col("rank") == 1)
        .drop("rank")
        .sort("id")
        .to_pydict()
    )
    print("latest per id:", latest)
    assert latest["id"] == [1, 2, 3]
    assert latest["version"] == [2, 3, 1]
    assert latest["payload"] == ["b", "d", "e"]

    # Counting duplicates before deciding what to do about them.
    counts = events.group_by("id").agg(n=bt.count()).filter(col("n") > 1).sort("id").to_pydict()
    print("duplicated keys:", counts)
    assert counts["id"] == [1, 2]

    # The metadata accessor answers the same question without a full aggregate.
    assert events.meta.col("id").has_duplicates()
    assert not events.meta.col("payload").has_duplicates()
    assert events.meta.col("payload").is_key()

    # A composite key that *is* unique.
    assert events.meta.is_key(["id", "version"])

    # `value_counts` is the quick frequency view.
    freq = events.value_counts("id").sort("id").to_pydict()
    print(freq)
    assert len(freq["id"]) == 3


if __name__ == "__main__":
    main()
