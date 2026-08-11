"""TPC-H Q11 — the parts holding most of the inventory value, against a computed threshold.

The `HAVING` clause compares each group against a fraction of a *global* total, so the
query needs that scalar before it can filter. Computing it as its own one-row aggregate
and reading the value out is the direct way to express that dependency.

    python examples/tpch/q11_important_stock.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    partsupp = tpch("partsupp")
    supplier = tpch("supplier")
    nation = tpch("nation")

    german = nation.filter(col("n_name") == "GERMANY").select("n_nationkey")
    german_stock = (
        partsupp.join(supplier, left_on="ps_suppkey", right_on="s_suppkey")
        .join(german, left_on="s_nationkey", right_on="n_nationkey")
        .with_columns(value=col("ps_supplycost") * col("ps_availqty"))
    )

    # The scalar the HAVING clause compares against. One row, read out into Python — this
    # is a plan boundary, not a per-row operation, so it costs one pass and no more.
    total = german_stock.agg(total=col("value").sum()).to_pydict()["total"][0]
    threshold = total * 0.0001
    print(f"total German stock value {total:,.2f}; threshold {threshold:,.2f}")

    result = (
        german_stock.group_by("ps_partkey")
        .agg(value=col("value").sum())
        .filter(col("value") > threshold)
        .sort("value", descending=True)
        .to_pydict()
    )

    print(f"{len(result['ps_partkey'])} parts above the threshold")
    assert result["value"] == sorted(result["value"], reverse=True)
    assert all(value > threshold for value in result["value"])
    # Every group is a slice of the same total, so no group can exceed it.
    assert all(value <= total for value in result["value"])


if __name__ == "__main__":
    main()
