"""Choosing a device, and running the same query either way.

Every accelerator example in this suite takes `--device gpu|cpu|auto`. The default is
auto: use a GPU when the engine can see one, and the CPU engine otherwise. Asking for
`--device gpu` on a machine with no accelerator is an error rather than a silent
downgrade, because that is the one time you typed it deliberately.

    python examples/gpu/device_selection.py
    python examples/gpu/device_selection.py --device cpu
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import device_count, has_gpu, resolve_device, tpch
from batcher import col


def main() -> None:
    device = resolve_device()
    print(f"device: {device} ({device_count()} accelerator(s) visible)")

    lineitem = tpch("lineitem").select("l_shipmode", "l_quantity", "l_extendedprice")

    query = (
        lineitem.filter(col("l_quantity") > 10)
        .group_by("l_shipmode")
        .agg(lines=bt.count(), revenue=col("l_extendedprice").sum())
        .sort("l_shipmode")
    )

    # `backend` picks the execution tier. It changes where the plan runs, never what it
    # computes, so the result is the contract and the device is an implementation detail.
    result = query.collect(backend=device)
    print(result.to_pydict()["l_shipmode"])

    reference = query.collect(backend="cpu")
    assert result.column_names == reference.column_names
    assert result.num_rows == reference.num_rows

    # Same rows, same names, same types — whichever tier ran it.
    assert result.schema == reference.schema
    on_device = result.to_pydict()
    on_cpu = reference.to_pydict()
    assert on_device["l_shipmode"] == on_cpu["l_shipmode"]
    assert on_device["lines"] == on_cpu["lines"]
    assert all(
        abs(left - right) < 1e-6
        for left, right in zip(on_device["revenue"], on_cpu["revenue"], strict=True)
    )

    if not has_gpu():
        print("no accelerator visible: this ran on the CPU engine, which is the point.")


if __name__ == "__main__":
    main()
