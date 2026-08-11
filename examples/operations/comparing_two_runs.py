"""Diffing the output of two pipeline versions.

Before a refactor ships, the question is whether it changed anything. Three anti joins and a
control total answer it, and they answer it in a form that says *what* changed rather than
just that something did.

    python examples/operations/comparing_two_runs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")

    def version_a() -> bt.Dataset:
        return (
            lineitem.filter(col("l_quantity") > 25)
            .group_by("l_shipmode")
            .agg(revenue=col("l_extendedprice").sum(), lines=bt.count())
        )

    def version_b() -> bt.Dataset:
        # A refactor: the filter moved and the aggregate is spelled differently.
        return (
            lineitem.with_columns(big=col("l_quantity") > 25)
            .filter(col("big"))
            .group_by("l_shipmode")
            .agg(revenue=col("l_extendedprice").sum(), lines=bt.count())
        )

    old, new = version_a(), version_b()

    # 1. Row counts.
    print("rows:", old.count(), "->", new.count())
    assert old.count() == new.count()

    # 2. Keys present in one and not the other.
    old_keys = old.select("l_shipmode")
    new_keys = new.select("l_shipmode")
    assert old_keys.join(new_keys, on="l_shipmode", how="anti").count() == 0
    assert new_keys.join(old_keys, on="l_shipmode", how="anti").count() == 0

    # 3. Per-key differences, which is the report a reviewer wants.
    compared = old.join(
        new.rename({"revenue": "new_revenue", "lines": "new_lines"}), on="l_shipmode"
    ).with_columns(delta=(col("revenue") - col("new_revenue")).abs())
    changed = compared.filter(col("delta") > 1e-6)
    print("keys with a changed value:", changed.count())
    assert changed.count() == 0

    # 4. Control total.
    old_total = old.agg(t=col("revenue").sum()).to_pydict()["t"][0]
    new_total = new.agg(t=col("revenue").sum()).to_pydict()["t"][0]
    assert abs(old_total - new_total) < 1e-3
    print(f"control total unchanged: {old_total:,.2f}")

    # A version that genuinely differs is caught by the same four checks.
    def version_c() -> bt.Dataset:
        return (
            lineitem.filter(col("l_quantity") > 30)
            .group_by("l_shipmode")
            .agg(revenue=col("l_extendedprice").sum(), lines=bt.count())
        )

    third = version_c()
    assert third.agg(t=col("revenue").sum()).to_pydict()["t"][0] != old_total


if __name__ == "__main__":
    main()
