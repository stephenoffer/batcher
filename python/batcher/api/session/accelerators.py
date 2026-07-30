"""Accelerator reporting (`accelerators`, `show_accelerators`).

The second question on a GPU bug report, after "which build", is "what hardware, and what was
it doing". `versions` answers the first. This answers the second in one call: which devices
this process can see, what the cluster's fleet looks like, what it is drawing, and whether the
driver is clamping anything. `measure_energy` answers the third — what a pipeline cost to run.

It reports rather than decides. Every figure comes from a source that already exists — the
device table, live telemetry, the cluster topology, the configured power envelope — and each
one is omitted when its source cannot answer, so a CPU-only host produces a small honest
report rather than a large one full of zeros.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from batcher.plan.energy import EnergyLedger

__all__ = ["accelerators", "measure_energy", "show_accelerators"]


def _device_rows() -> list[dict]:
    """Per-device rows: nameplate figures for what is attached, live readings where available."""
    from batcher._internal.device_specs import device_spec, resolve_device_name
    from batcher._internal.hardware import device_telemetry, gpu_inventory

    live = {d.index: d for d in device_telemetry()}
    rows: list[dict] = []
    for index, device in enumerate(gpu_inventory()):
        name = str(device.get("name") or "")
        row: dict = {"index": index, "name": name}
        memory = int(device.get("memory_bytes") or 0)
        if memory:
            row["memory_gib"] = round(memory / (1 << 30), 1)
        spec = device_spec(resolve_device_name(device.get("accelerator_type") or name))
        if spec is not None:
            row["tdp_watts"] = spec.tdp_watts
            row["nvlink_domain"] = spec.nvlink_domain
        reading = live.get(index)
        if reading is not None:
            row["power_watts"] = round(reading.power_watts, 1)
            row["sm_utilization"] = round(reading.sm_utilization, 3)
            row["temperature_c"] = round(reading.temperature_c, 1)
            if reading.throttle_reasons:
                row["throttled"] = list(reading.throttle_reasons)
            if reading.ecc_uncorrected:
                row["ecc_uncorrected"] = reading.ecc_uncorrected
        rows.append(row)
    return rows


def accelerators() -> dict:
    """Report the accelerators this process and its cluster can see.

    The machine-readable form of `show_accelerators`: paste it into a bug report, or assert
    on it in a deployment check. Keys are present only when their source could answer, so a
    CPU-only host reports a backend and an empty device list rather than a page of zeros.

    Returns:
        A mapping with `backend` (the compute backend this host would use), `devices` (one
        row per local device, carrying nameplate figures and any live readings), `fleet` (the
        cluster's shape when Ray is up: device and node counts, the widest coherent NVLink
        domain, the device models present, and how many racks and power zones they span), and
        `power` (the configured budget and, where telemetry is available, the measured draw).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> report = bt.accelerators()
            >>> sorted(report)
            ['backend', 'devices', 'power']
            >>> isinstance(report["devices"], list)
            True
    """
    from batcher._internal.hardware import accelerator_backend, nvml_available, total_power_watts
    from batcher.config import active_config

    report: dict = {"backend": accelerator_backend(), "devices": _device_rows()}

    from batcher.dist.executors.ray_runtime.fabric import topology_summary

    fleet = topology_summary()
    if fleet.get("gpu_nodes"):
        report["fleet"] = fleet

    energy = active_config().accelerator.energy
    power: dict = {}
    if energy.power_budget_watts > 0:
        power["budget_watts"] = energy.power_budget_watts
    if nvml_available():
        power["draw_watts"] = round(total_power_watts(), 1)
    report["power"] = power
    return report


def show_accelerators() -> None:
    """Print the accelerator report: local devices, the cluster fleet, and the power envelope.

    The human-readable form of `accelerators`, in the shape of `show_versions`. A device that
    the driver is clamping, or one reporting uncorrectable ECC errors, is called out by name,
    because those are the two conditions that make a run slow or wrong without appearing
    anywhere in its own timings.

    Returns:
        None. The report is written to stdout.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.show_accelerators()  # doctest: +SKIP
    """
    from batcher.observe import format_device_table

    report = accelerators()
    print(f"backend: {report['backend']}")
    devices = report["devices"]
    if not devices:
        print("devices: none visible to this process")
    else:
        print(format_device_table())
        for row in devices:
            if "power_watts" not in row:
                print(f"gpu {row['index']}  {row['name']}  (no live telemetry)")
    fleet = report.get("fleet")
    if fleet:
        models = ", ".join(fleet["device_models"]) or "unlabelled"
        print(
            f"fleet: {fleet['gpus']} device(s) on {fleet['gpu_nodes']} node(s), "
            f"widest fabric domain {fleet['largest_domain']}, models {models}"
        )
        if fleet["racks"] or fleet["power_zones"]:
            print(f"       {fleet['racks']} rack(s), {fleet['power_zones']} power zone(s)")
    power = report["power"]
    if power:
        parts = [f"{k.replace('_', ' ')} {v}" for k, v in sorted(power.items())]
        print("power: " + ", ".join(parts))


@contextlib.contextmanager
def measure_energy() -> Iterator[EnergyLedger]:
    """Collect the energy every accelerator stage inside the block drew.

    A GPU-hour is what a fleet is billed; joules are what it buys. This is how a pipeline
    reports the second: each accelerator stage that runs inside the block records what it
    drew, measured from device readings where NVML is available and modelled from the device
    table where it is not, and the ledger tells the two apart.

    The ledger is filled as the block runs, so read it after the block. Render it with
    :func:`batcher.observe.format_energy_report`, or take the ratios off it directly.
    Recording is skipped entirely when `accelerator.energy.accounting` is off.

    Returns:
        A context manager yielding the `EnergyLedger` the block's stages record into.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> with bt.measure_energy() as energy:
            ...     _ = bt.from_pydict({"x": [1, 2, 3]}).to_pydict()
            >>> energy.total_joules >= 0.0
            True
    """
    from batcher.core.energy import energy_scope

    with energy_scope() as ledger:
        yield ledger
