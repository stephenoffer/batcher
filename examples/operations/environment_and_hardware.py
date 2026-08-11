"""What the engine can see about the machine it is on.

Core count, memory and the container's limits are what size a job. Reading them from the
engine rather than from `os` matters inside a container, where the host's core count is not
the one the cgroup will let you use.

    python examples/operations/environment_and_hardware.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col
from batcher.config import get_option


def main() -> None:
    print("engine version:", bt.engine_version())

    # The host's view, which is the one that is wrong inside a container.
    host_cores = os.cpu_count()
    print("os.cpu_count():", host_cores)
    assert host_cores and host_cores > 0

    # The engine's knobs, which are what actually size the work.
    morsel = get_option("execution.morsel_rows")
    print("morsel_rows:", morsel)
    assert isinstance(morsel, int)
    assert morsel > 0

    # The accelerator report, empty on a CPU-only machine.
    report = bt.accelerators()
    print("accelerator backend:", report.get("backend"))
    print("devices:", len(report.get("devices", [])))
    assert "devices" in report

    # Site information: empty on a laptop, populated on a managed cluster. Empty is the
    # honest answer rather than a guess.
    site = report.get("site")
    print("site:", site)

    # None of it changes an answer.
    total = tpch("lineitem").agg(q=col("l_quantity").sum()).to_pydict()
    print("engine check:", total)
    assert total["q"][0] > 0


if __name__ == "__main__":
    main()
