"""What the engine can see about the accelerators on this machine.

Every field here is empty on a machine with no GPU, which is the honest answer rather than
a guess. Reading them before a job is how you find out that a device has less memory than
the batch you were about to send it.

    python examples/gpu/device_sizing.py
    python examples/gpu/device_sizing.py --device cpu
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
    report = bt.accelerators()

    print("backend:", report.get("backend"))
    print("devices:", device_count())
    assert set(report) >= {"backend", "devices"}
    assert isinstance(report["devices"], list)

    for entry in report["devices"]:
        print("  ", entry)

    # The visible device count and the resolved device agree.
    assert (device == "gpu") == has_gpu()

    # `show_accelerators` prints the same thing in a readable form.
    bt.show_accelerators()

    # None of it changes a result: the same query runs the same on any of these.
    total = tpch("lineitem").agg(q=col("l_quantity").sum()).collect(backend=device).to_pydict()
    reference = tpch("lineitem").agg(q=col("l_quantity").sum()).collect(backend="cpu").to_pydict()
    assert total == reference
    print("engine check:", total)


if __name__ == "__main__":
    main()
