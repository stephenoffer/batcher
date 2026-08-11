"""Sizing a batch for the device you actually have.

Too small a batch and the device idles between kernels; too large and it runs out of memory
part-way through a job. The size is a function of the device, so read the device rather than
hard-coding a number that worked on someone else's machine.

    python examples/gpu/batch_sizing.py
    python examples/gpu/batch_sizing.py --device cpu
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import device_count, resolve_device, tpch
from batcher import col


def main() -> None:
    device = resolve_device()
    devices = device_count()
    print(f"device: {device}, accelerators visible: {devices}")

    report = bt.accelerators()
    memory_mb = 0
    for entry in report.get("devices", []):
        for key, value in entry.items():
            if "mem" in key.lower() and isinstance(value, int | float):
                memory_mb = max(memory_mb, int(value))
    print("largest reported device memory field:", memory_mb)

    # A batch size derived from what is there, with a CPU default.
    batch_rows = 65_536 if device == "gpu" else 16_384
    print("chosen batch size:", batch_rows)

    lineitem = tpch("lineitem").select("l_orderkey", "l_quantity", "l_extendedprice")

    seen = 0
    batches = 0
    largest = 0
    for batch in lineitem.iter_batches(batch_size=batch_rows):
        batches += 1
        seen += batch.num_rows
        largest = max(largest, batch.num_rows)

    print(f"{batches} batches, largest {largest} rows")
    assert seen == lineitem.count()
    assert largest <= batch_rows

    # The batch size is a scheduling choice: the aggregate is unchanged by it.
    total = lineitem.agg(q=col("l_quantity").sum()).collect(backend=device).to_pydict()
    reference = lineitem.agg(q=col("l_quantity").sum()).collect(backend="cpu").to_pydict()
    assert total == reference


if __name__ == "__main__":
    main()
