"""A parity matrix: every operator shape, on both tiers.

This is the device tier's release check. Each row is a query shape, and each is compared
schema-first against the CPU oracle. On a machine with no accelerator every row runs on the
CPU engine twice, which still proves the *harness* works — and that is worth having ready.

    python examples/gpu/cpu_gpu_parity_matrix.py
    python examples/gpu/cpu_gpu_parity_matrix.py --device cpu
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import has_gpu, resolve_device, tpch
from batcher import col


def main() -> None:
    device = resolve_device()
    print(f"device: {device} (accelerator visible: {has_gpu()})")

    lineitem = tpch("lineitem")
    orders = tpch("orders")

    shapes = {
        "projection": lineitem.select("l_orderkey", "l_quantity").head(1_000),
        "filter": lineitem.filter(col("l_quantity") > 45).select("l_orderkey").head(1_000),
        "arithmetic": lineitem.select(net=col("l_extendedprice") * (1 - col("l_discount"))).head(
            1_000
        ),
        "integer abs": lineitem.select(m=(col("l_linenumber") - 3).abs()).head(1_000),
        "date part": lineitem.select(y=col("l_shipdate").dt.year()).head(1_000),
        "date passthrough": lineitem.select("l_shipdate").head(1_000),
        "grouped sum": lineitem.group_by("l_shipmode")
        .agg(t=col("l_extendedprice").sum())
        .sort("l_shipmode"),
        "grouped count": lineitem.group_by("l_returnflag").agg(n=bt.count()).sort("l_returnflag"),
        "join": lineitem.join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .select("l_orderkey", "o_orderstatus")
        .head(1_000),
        "sort + limit": lineitem.sort("l_extendedprice", descending=True)
        .select("l_extendedprice")
        .head(20),
        "string filter": lineitem.filter(col("l_comment").str.contains("final"))
        .select("l_orderkey")
        .head(1_000),
        "distinct": lineitem.select("l_shipmode").distinct().sort("l_shipmode"),
    }

    failures: list[str] = []
    for name, query in shapes.items():
        on_device = query.collect(backend=device)
        on_cpu = query.collect(backend="cpu")
        same_schema = on_device.schema == on_cpu.schema
        same_values = on_device.to_pydict() == on_cpu.to_pydict()
        status = "ok  " if (same_schema and same_values) else "FAIL"
        print(f"  {status} {name:<18} {on_device.num_rows:>6} rows")
        if not (same_schema and same_values):
            failures.append(name)

    assert not failures, failures
    print(f"{len(shapes)} shapes agree on schema and values")


if __name__ == "__main__":
    main()
