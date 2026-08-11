"""Completeness: did every expected group arrive.

A count that looks right can still be missing a whole partition, because the rows that would
have made it wrong are the ones that are absent. Checking against the *expected* set of
groups is the version of the check that can fail.

    python examples/quality/completeness_checks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders")
    nation = tpch("nation")
    customer = tpch("customer")

    expected_nations = set(nation.to_pydict()["n_name"])
    print("expected nations:", len(expected_nations))

    by_nation = (
        orders.join(customer, left_on="o_custkey", right_on="c_custkey")
        .join(nation, left_on="c_nationkey", right_on="n_nationkey")
        .group_by("n_name")
        .agg(orders=bt.count())
    )
    present = set(by_nation.to_pydict()["n_name"])
    missing = expected_nations - present
    print("nations with no orders:", missing or "none")

    # The completeness assertion, which a row count could not make.
    assert present <= expected_nations
    assert not missing

    # A deliberately incomplete run, to show the check has teeth.
    partial = (
        orders.join(customer, left_on="o_custkey", right_on="c_custkey")
        .join(nation.filter(col("n_nationkey") < 20), left_on="c_nationkey", right_on="n_nationkey")
        .group_by("n_name")
        .agg(orders=bt.count())
    )
    partial_present = set(partial.to_pydict()["n_name"])
    partial_missing = expected_nations - partial_present
    print(f"the partial run is missing {len(partial_missing)} nations")
    assert partial_missing

    # The row counts alone do not reveal it, which is the whole point.
    print(f"full {by_nation.count()} groups, partial {partial.count()} groups")
    assert partial.count() < by_nation.count()

    # Per-period completeness, the same shape over time.
    years = orders.with_columns(year=col("o_orderdate").dt.year()).n_unique("year")
    per_year = (
        orders.with_columns(year=col("o_orderdate").dt.year())
        .group_by("year")
        .agg(orders=bt.count())
    )
    assert per_year.count() == years
    assert all(value > 0 for value in per_year.to_pydict()["orders"])


if __name__ == "__main__":
    main()
