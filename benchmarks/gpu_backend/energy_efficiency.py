"""Measure a GPU stage's energy efficiency: joules, tokens/joule, and the idle share.

Throughput alone cannot rank two GPU deployments. A device that finishes 20% sooner while
drawing twice the power has made a power-capped fleet slower for everyone queued behind it, and
only an energy figure says so. This script produces the three numbers that do rank them:

* **joules** for the stage, measured from board readings where NVML answers;
* **work per joule** (rows, or tokens for a generative stage), the figure that stays comparable
  across device generations;
* **the idle share**, the energy spent holding devices the pipeline was not feeding, which is
  the part a scheduler can actually reclaim.

Run it on a GPU host:

    python benchmarks/gpu_backend/energy_efficiency.py --rows 20000000 --width 8

It prints what it measured. Nothing here is a committed claim: an energy number is a property
of one machine at one moment (a device's draw depends on its power limit, its cooling, and its
clocks), so results belong in `benchmarks/BENCHMARK_RESULTS.md` with the hardware named, not in
this file's docstring.

Without NVML the run still works and reports modelled energy from the device table, marked as
such — an estimate labelled an estimate is useful, and one presented as a measurement is not.
"""

from __future__ import annotations

import argparse
import sys

import batcher as bt
from batcher._internal.hardware import accelerator_backend, nvml_available
from batcher.core.energy import energy_scope, measure_stage
from batcher.observe import format_device_table, format_energy_report
from batcher.plan.energy import GridProfile


def _run(rows: int, width: int, devices: int, accelerator: str) -> None:
    """Run one aggregate over `rows` and report what it drew."""
    frame = bt.from_pydict({f"c{i}": list(range(min(rows, 1_000_000))) for i in range(width)})
    with (
        energy_scope() as ledger,
        measure_stage("Aggregate#1", accelerator_type=accelerator, device_count=devices) as meter,
    ):
        out = frame.group_by("c0").agg(total=bt.col("c1").sum()).collect()
        meter.add_rows(out.num_rows)
    print(format_energy_report(ledger, GridProfile(region="local")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_000_000, help="input rows")
    parser.add_argument("--width", type=int, default=8, help="numeric columns")
    parser.add_argument("--devices", type=int, default=1, help="devices the stage holds")
    parser.add_argument(
        "--accelerator",
        default="",
        help="device model (a ray.util.accelerators name); empty reads the local fleet",
    )
    args = parser.parse_args(argv)

    accelerator = args.accelerator
    if not accelerator:
        rows = bt.accelerators()["devices"]
        accelerator = rows[0].get("name", "").replace(" ", "_").upper() if rows else ""

    print(f"backend: {accelerator_backend()}   telemetry: {'yes' if nvml_available() else 'no'}")
    print(format_device_table())
    print()
    _run(args.rows, args.width, args.devices, accelerator)
    return 0


if __name__ == "__main__":
    sys.exit(main())
