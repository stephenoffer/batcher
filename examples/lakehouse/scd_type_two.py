"""Slowly changing dimensions: keeping the history of a changed row.

Type 2 closes the old version and opens a new one rather than overwriting, so a query can
ask what a customer's segment was *at the time* an order was placed. That is the whole
reason to carry the extra rows.

    python examples/lakehouse/scd_type_two.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    customer = tpch("customer").select("c_custkey", "c_mktsegment").head(500)

    # The dimension as it stood, and a later snapshot where some segments changed.
    first = customer.with_columns(effective=bt.lit(dt.date(2024, 1, 1)))
    changed = customer.head(100).with_columns(
        c_mktsegment=bt.lit("HOUSEHOLD"),
        effective=bt.lit(dt.date(2024, 6, 1)),
    )

    history = first.union(changed).sort("c_custkey", "effective")

    # Close each version when the next one for the same key begins.
    versioned = history.with_columns(
        valid_to=col("effective")
        .shift(-1)
        .over(partition_by=["c_custkey"], order_by=["effective"]),
    ).with_columns(is_current=col("valid_to").is_null())

    result = versioned.to_pydict()
    print(versioned.head(4).to_pydict())

    # Exactly one current row per customer.
    current = versioned.filter(col("is_current"))
    assert current.count() == customer.count()
    assert current.n_unique("c_custkey") == customer.count()

    # The changed customers have two versions, the rest one.
    counts = versioned.group_by("c_custkey").agg(versions=bt.count()).to_pydict()
    assert sorted(set(counts["versions"])) == [1, 2]
    assert sum(1 for value in counts["versions"] if value == 2) == 100

    # A closed version is closed by the start of its successor, never overlapping.
    closed = versioned.filter(col("valid_to").is_not_null()).to_pydict()
    assert all(
        start < end for start, end in zip(closed["effective"], closed["valid_to"], strict=True)
    )
    assert len(result["c_custkey"]) == 600


if __name__ == "__main__":
    main()
