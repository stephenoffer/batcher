"""MERGE INTO: upserting keyed rows into a Delta table.

An upsert is one commit that updates matched rows and inserts the rest. Doing it as a
delete followed by an append is two commits with a window in between where readers see
neither version, which is exactly what a transactional table is for avoiding.

    python examples/lakehouse/delta_upserts.py
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

        # A batch that overlaps the existing keys and extends past them.
        updates = customer.slice(900, 300).with_columns(c_acctbal=col("c_acctbal") + 100.0)
        updates.write.delta(table, merge_on="c_custkey")

        merged = bt.read.delta(table)
        print("after upsert:", merged.count())

        # 1,000 existing + 200 genuinely new = 1,200. No duplicates.
        assert merged.count() == 1_200
        keys = sorted(merged.to_pydict()["c_custkey"])
        assert len(keys) == len(set(keys))

        # The 100 overlapping rows were updated, not duplicated.
        original = customer.slice(900, 100).sort("c_custkey").to_pydict()
        after = (
            merged.filter(col("c_custkey").is_in(original["c_custkey"]))
            .sort("c_custkey")
            .to_pydict()
        )
        assert all(
            abs(new - old - 100.0) < 1e-6
            for old, new in zip(original["c_acctbal"], after["c_acctbal"], strict=True)
        )

        # And rows the batch never mentioned are untouched: derive the set rather than
        # assuming the keys are a contiguous range.
        touched = set(updates.to_pydict()["c_custkey"])
        untouched = merged.join(updates.select("c_custkey"), on="c_custkey", how="anti")
        print("rows the merge never mentioned:", untouched.count())
        assert untouched.count() == 1_000 - len(touched & set(keys[:1_000]))
        assert (
            untouched.to_pydict()["c_acctbal"]
            == (
                customer.head(1_000)
                .join(updates.select("c_custkey"), on="c_custkey", how="anti")
                .sort("c_custkey")
                .to_pydict()["c_acctbal"]
            )
        )


if __name__ == "__main__":
    main()
