"""Reconciling a transformed dataset against its source.

Four checks, in the order that localizes a failure fastest: row count, key set, per-key
values, and the control total. A mismatch at any level tells you where to look, which a
single end-to-end comparison does not.

    python examples/quality/reconciliation_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    source = tpch("lineitem").select("l_orderkey", "l_linenumber", "l_extendedprice")

    # A transformation that is meant to preserve everything but the column names.
    transformed = source.rename(
        {"l_orderkey": "order_id", "l_linenumber": "line_no", "l_extendedprice": "amount"}
    )

    # 1. Row count.
    print("rows:", source.count(), "->", transformed.count())
    assert source.count() == transformed.count()

    # 2. Key set: an anti join in each direction.
    left_keys = source.select(
        col("l_orderkey").alias("order_id"), col("l_linenumber").alias("line_no")
    )
    right_keys = transformed.select("order_id", "line_no")
    missing = left_keys.join(right_keys, on=["order_id", "line_no"], how="anti").count()
    extra = right_keys.join(left_keys, on=["order_id", "line_no"], how="anti").count()
    print("missing:", missing, "extra:", extra)
    assert missing == 0
    assert extra == 0

    # 3. Per-key values.
    paired = left_keys.join(
        source.select(
            col("l_orderkey").alias("order_id"),
            col("l_linenumber").alias("line_no"),
            col("l_extendedprice").alias("source_amount"),
        ),
        on=["order_id", "line_no"],
    ).join(transformed, on=["order_id", "line_no"])
    mismatches = paired.filter((col("source_amount") - col("amount")).abs() > 1e-9).count()
    print("value mismatches:", mismatches)
    assert mismatches == 0

    # 4. Control total, which catches a systematic shift the per-key check would too but
    # is cheap enough to run on its own in production.
    source_total = source.agg(t=col("l_extendedprice").sum()).to_pydict()["t"][0]
    target_total = transformed.agg(t=col("amount").sum()).to_pydict()["t"][0]
    print(f"control total: {source_total:,.2f} vs {target_total:,.2f}")
    assert abs(source_total - target_total) < 1e-3

    # A deliberately broken transformation fails at the level that localizes it.
    broken = transformed.filter(col("amount") > 1_000)
    assert broken.count() < source.count()
    assert bt is not None


if __name__ == "__main__":
    main()
