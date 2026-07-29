"""Writing and reading Parquet, with partitioning and column pruning.

Parquet is the default for a reason: the footer carries statistics, so a filtered read
skips row groups without decoding them and ``count()`` is answered from metadata alone.
Partitioning on a column you always filter by turns that skipping into directory pruning.

    python examples/io/parquet_roundtrip.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import batcher as bt
from batcher import col


def main() -> None:
    events = bt.from_pydict(
        {
            "day": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "user": ["a", "b", "a", "c"],
            "amount": [10, 20, 30, 40],
        }
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # A plain write, then read it back.
        flat = root / "flat"
        events.write.parquet(str(flat))
        back = bt.read.parquet(str(flat)).to_pydict()
        assert sorted(back["amount"]) == [10, 20, 30, 40]

        # `count()` comes from the footer, so no data is decoded.
        n = bt.read.parquet(str(flat)).count()
        print("count:", n)
        assert n == 4

        # Reading a subset of columns never touches the other column chunks.
        pruned = bt.read.parquet(str(flat)).select("amount").to_pydict()
        assert list(pruned) == ["amount"]

        # Partitioned by `day`: one Hive-style directory per value.
        parts = root / "parts"
        events.write.parquet(str(parts), partition_by=["day"])
        dirs = sorted(p.name for p in parts.iterdir() if p.is_dir())
        print("partitions:", dirs)
        assert dirs == ["day=2024-01-01", "day=2024-01-02"]

        # Reading it back needs the *partition-aware* reader. A plain `read.parquet` only
        # reads files, and the partition key lives in the directory name, so `day` would be
        # missing -- it warns loudly rather than losing the column silently.
        assert "day" not in bt.read.parquet(str(parts)).schema.names

        partitioned = bt.read.parquet_dataset(str(parts))
        assert "day" in partitioned.schema.names

        # A filter on the partition column now prunes whole directories.
        one_day = partitioned.filter(col("day") == "2024-01-02").to_pydict()
        print(one_day)
        assert sorted(one_day["amount"]) == [30, 40]
        assert set(one_day["day"]) == {"2024-01-02"}

        # Save modes: `overwrite` replaces, the default refuses to clobber.
        events.write.parquet(str(flat), mode="overwrite")
        assert bt.read.parquet(str(flat)).count() == 4


if __name__ == "__main__":
    main()
