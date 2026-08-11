"""A pipeline where some stages run on a device and some do not.

The device tier declines what it cannot translate and the CPU engine picks it up, so a plan
with one unsupported operator is not a plan that fails. What matters is that the seam is
invisible in the result, which is what this asserts.

    python examples/gpu/mixed_device_pipeline.py
    python examples/gpu/mixed_device_pipeline.py --device cpu
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import resolve_device, tpch
from batcher import col


def main() -> None:
    device = resolve_device()
    print("device:", device)

    lineitem = tpch("lineitem")
    orders = tpch("orders")

    # Stage one: numeric work, the device tier's home ground.
    numeric = lineitem.filter(col("l_quantity") > 20).with_columns(
        revenue=col("l_extendedprice") * (1 - col("l_discount"))
    )

    # Stage two: string work, which the tier may decline.
    textual = numeric.filter(col("l_comment").str.contains("final"))

    # Stage three: a join and an aggregate.
    combined = (
        textual.join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .group_by("o_orderpriority")
        .agg(revenue=col("revenue").sum(), lines=bt.count())
        .sort("o_orderpriority")
    )

    on_device = combined.collect(backend=device)
    on_cpu = combined.collect(backend="cpu")

    print(f"{on_device.num_rows} priorities")
    print(on_device.to_pydict()["lines"])

    # The seam is invisible: same schema, same rows, same values.
    assert on_device.schema == on_cpu.schema
    assert on_device.num_rows == on_cpu.num_rows
    assert on_device.to_pydict() == on_cpu.to_pydict()

    # Each stage independently, so a divergence would be attributable.
    for name, stage in (("numeric", numeric), ("textual", textual)):
        device_count = stage.count()
        assert device_count > 0, name
        staged = stage.select("l_orderkey").collect(backend=device)
        staged_cpu = stage.select("l_orderkey").collect(backend="cpu")
        assert staged.num_rows == staged_cpu.num_rows, name
        assert staged.schema == staged_cpu.schema, name
        print(f"  {name:<8} {staged.num_rows:>7} rows, agrees with the CPU engine")


if __name__ == "__main__":
    main()
