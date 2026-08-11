"""The suite in miniature: one script touching every subsystem.

If this passes, the engine reads, plans, optimizes, executes, aggregates, joins, windows,
writes and governs. It is the smoke test to run first when something is wrong, because it
localizes the failure to a subsystem in one run.

    python examples/operations/release_check.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import resolve_device, tpch, tpch_uri
from batcher import col


def main() -> None:
    checks: dict[str, bool] = {}

    # IO: a real read from object storage.
    remote = bt.read.parquet(tpch_uri("nation"))
    checks["s3 read"] = remote.count() == 25

    # IO: the local cache.
    lineitem = tpch("lineitem")
    checks["parquet scan"] = lineitem.count() > 0

    # Plan: metadata without execution.
    checks["schema"] = lineitem.width == 16 and "filter" in (
        lineitem.filter(col("l_quantity") > 1).explain().lower()
    )

    # Relational core.
    checks["filter"] = lineitem.filter(col("l_quantity") > 45).count() < lineitem.count()
    checks["join"] = (
        lineitem.join(tpch("orders"), left_on="l_orderkey", right_on="o_orderkey").count() > 0
    )
    grouped = lineitem.group_by("l_shipmode").agg(n=bt.count())
    checks["aggregate"] = sum(grouped.to_pydict()["n"]) == lineitem.count()
    checks["window"] = lineitem.with_columns(
        r=bt.row_number().over(partition_by=["l_orderkey"], order_by=["l_linenumber"])
    ).filter(col("r") == 1).count() == lineitem.n_unique("l_orderkey")
    sorted_head = lineitem.sort("l_extendedprice", descending=True).head(5).to_pydict()
    checks["sort"] = sorted_head["l_extendedprice"] == sorted(
        sorted_head["l_extendedprice"], reverse=True
    )

    # SQL.
    checks["sql"] = (
        bt.sql("SELECT COUNT(*) AS n FROM lineitem", lineitem=lineitem).to_pydict()["n"][0]
        == lineitem.count()
    )

    # Expressions.
    checks["expressions"] = (
        lineitem.select(x=col("l_comment").str.len_chars())
        .agg(m=col("x").max())
        .to_pydict()["m"][0]
        > 0
    )

    # Data quality.
    checks["data quality"] = lineitem.dq.not_null("l_orderkey").validate().ok

    # Execution tiers.
    device = resolve_device()
    checks["backend parity"] = (
        grouped.sort("l_shipmode").collect(backend=device).to_pydict()
        == grouped.sort("l_shipmode").collect(backend="cpu").to_pydict()
    )

    # Partition independence.
    checks["partition parity"] = (
        grouped.sort("l_shipmode").collect(num_partitions=1).to_pydict()
        == grouped.sort("l_shipmode").collect(num_partitions=8).to_pydict()
    )

    # Spill.
    checks["spill parity"] = (
        grouped.sort("l_shipmode").collect(spill=True).to_pydict()
        == grouped.sort("l_shipmode").collect().to_pydict()
    )

    # Write path.
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "out.parquet")
        grouped.write.parquet(path)
        checks["write"] = bt.read.parquet(path).count() == grouped.count()

    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'} {name}")
    failed = [name for name, passed in checks.items() if not passed]
    assert not failed, failed
    print(f"all {len(checks)} subsystem checks passed")


if __name__ == "__main__":
    main()
