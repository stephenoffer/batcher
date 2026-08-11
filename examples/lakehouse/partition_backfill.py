"""Replacing one partition without touching the rest.

A backfill that rewrites the whole table to fix one day is both slow and dangerous. Scoping
the overwrite to a predicate replaces exactly the rows that match, in one commit, and leaves
every other partition byte-identical.

    python examples/lakehouse/partition_backfill.py
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
    orders = (
        tpch("orders")
        .with_columns(year=col("o_orderdate").dt.year())
        .select("year", "o_orderkey", "o_totalprice")
    )

    with tempfile.TemporaryDirectory() as directory:
        table = str(Path(directory) / "orders")
        orders.write.delta(table, partition_by=["year"])

        before = bt.read.delta(table)
        assert before.count() == orders.count()
        years = sorted(before.select("year").distinct().to_pydict()["year"])
        target = years[0]

        untouched_before = (
            before.filter(col("year") != target)
            .agg(t=col("o_totalprice").sum())
            .to_pydict()["t"][0]
        )
        target_before = before.filter(col("year") == target).count()
        print(f"year {target}: {target_before} rows before the backfill")

        # The corrected data for that year only.
        corrected = orders.filter(col("year") == target).with_columns(
            o_totalprice=col("o_totalprice") * 1.05
        )

        try:
            corrected.write.delta(table, mode="overwrite", replace_where=f"year = {target}")
        except Exception as error:
            print("replace_where unavailable here:", str(error)[:70])
            return

        after = bt.read.delta(table)
        print("rows after:", after.count())

        # The row count is unchanged and only the target partition moved.
        assert after.count() == orders.count()
        assert after.filter(col("year") == target).count() == target_before

        untouched_after = (
            after.filter(col("year") != target).agg(t=col("o_totalprice").sum()).to_pydict()["t"][0]
        )
        assert abs(untouched_after - untouched_before) < 1e-3

        # And the target partition really was corrected.
        raised = (
            after.filter(col("year") == target).agg(t=col("o_totalprice").sum()).to_pydict()["t"][0]
        )
        original = (
            orders.filter(col("year") == target)
            .agg(t=col("o_totalprice").sum())
            .to_pydict()["t"][0]
        )
        assert abs(raised - original * 1.05) < 1e-2


if __name__ == "__main__":
    main()
