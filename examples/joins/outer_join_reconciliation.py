"""A full outer join, and reading the three populations it produces.

An outer join answers "what is on each side" in one pass, and the three cases — left only,
right only, both — are distinguished by which side's columns are null. Naming them
explicitly is what turns the result into a reconciliation report.

    python examples/joins/outer_join_reconciliation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderkey", "o_totalprice")

    left = orders.filter(col("o_orderkey") % 2 == 0).select(
        col("o_orderkey").alias("key"), col("o_totalprice").alias("left_value")
    )
    right = orders.filter(col("o_orderkey") % 3 == 0).select(
        col("o_orderkey").alias("key"), col("o_totalprice").alias("right_value")
    )

    outer = left.join(right, on="key", how="outer")

    classified = outer.with_columns(
        status=bt.when(col("left_value").is_null())
        .then(bt.lit("right only"))
        .when(col("right_value").is_null())
        .then(bt.lit("left only"))
        .otherwise(bt.lit("both"))
    )

    counts = classified.value_counts("status").sort("status").to_pydict()
    print(counts)
    assert set(counts["status"]) == {"both", "left only", "right only"}
    assert sum(counts["count"]) == outer.count()

    by_status = dict(zip(counts["status"], counts["count"], strict=True))

    # The three populations reconcile against each side's own count.
    assert by_status["both"] + by_status["left only"] == left.count()
    assert by_status["both"] + by_status["right only"] == right.count()

    # And where both sides matched, the values agree — which is the point of a
    # reconciliation rather than just a row count.
    matched = classified.filter(col("status") == bt.lit("both")).to_pydict()
    assert all(
        abs(a - b) < 1e-9
        for a, b in zip(matched["left_value"], matched["right_value"], strict=True)
    )


if __name__ == "__main__":
    main()
