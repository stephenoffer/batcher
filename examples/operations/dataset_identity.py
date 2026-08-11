"""Comparing two Datasets, and what "equal" means for a lazy plan.

Two Datasets can hold the same rows and different plans, or the same plan and different
sources. `equals` compares the data, which means it executes; comparing the plans is a
different question and is what a cache key needs.

    python examples/operations/dataset_identity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderkey", "o_totalprice")

    # Two different plans producing the same rows.
    by_filter = orders.filter(col("o_totalprice") > 100_000).sort("o_orderkey")
    by_two_filters = (
        orders.filter(col("o_totalprice") > 50_000)
        .filter(col("o_totalprice") > 100_000)
        .sort("o_orderkey")
    )

    print("plan A:", by_filter.explain().splitlines()[0])
    print("plan B:", by_two_filters.explain().splitlines()[0])

    # The data is the same.
    assert by_filter.count() == by_two_filters.count()
    assert by_filter.equals(by_two_filters)

    # And a genuinely different result is not equal.
    other = orders.filter(col("o_totalprice") > 200_000).sort("o_orderkey")
    assert not by_filter.equals(other)

    # Immutability: deriving a new Dataset never changes the one it came from.
    before = orders.count()
    _ = orders.filter(col("o_totalprice") > 100_000)
    assert orders.count() == before

    # `copy` is explicit about that, and produces an independent handle.
    duplicate = orders.copy()
    assert duplicate.count() == orders.count()
    assert duplicate.columns == orders.columns

    # Same rows, different column order, is not the same Dataset.
    reordered = by_filter.select("o_totalprice", "o_orderkey")
    assert reordered.columns != by_filter.columns
    assert reordered.count() == by_filter.count()


if __name__ == "__main__":
    main()
