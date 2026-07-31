"""One row per local device: nameplate figures, live readings, and what is wrong with it.

Split out of the report itself because the two answer different questions. This assembles
what is true of a *device* — its model's figures, its telemetry, the link it came up on,
the faults its counters carry, the settings it arrived with. The report next door decides
which of those a reader is shown and in what order.
"""

from __future__ import annotations

__all__ = ["device_rows"]


def device_rows() -> list[dict]:
    """Per-device rows: nameplate figures for what is attached, live readings where available."""
    from batcher._internal.device_specs import device_spec, resolve_device_name
    from batcher._internal.hardware import device_telemetry, gpu_inventory
    from batcher._internal.hardware.faults import device_faults, device_modes

    live = {d.index: d for d in device_telemetry()}
    faults = {f.index: f for f in device_faults()}
    modes = {m.index: m for m in device_modes()}
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
            # The two links, because "why is this stage slow" is usually one of them: the host
            # link decides whether the data can arrive fast enough to be worth a device, and
            # the fabric decides how wide a collective can go before it leaves the fast path.
            if spec.host_link:
                row["host_link"] = spec.host_link
                row["host_link_gbps"] = spec.host_link_gbps
            if spec.nvlink_gbps:
                row["nvlink_gbps"] = spec.nvlink_gbps
        reading = live.get(index)
        if reading is not None:
            row["power_watts"] = round(reading.power_watts, 1)
            row["sm_utilization"] = round(reading.sm_utilization, 3)
            row["temperature_c"] = round(reading.temperature_c, 1)
            if reading.throttle_reasons:
                row["throttled"] = list(reading.throttle_reasons)
            if reading.ecc_uncorrected:
                row["ecc_uncorrected"] = reading.ecc_uncorrected
        _add_measured_link(row, index)
        _add_faults(row, faults.get(index))
        _add_modes(row, modes.get(index))
        rows.append(row)
    return rows


def _add_measured_link(row: dict, index: int) -> None:
    """Add what the device's host link *negotiated*, where it differs from the nameplate.

    The nameplate `host_link` above says what the model ships with. This says what this board
    came up at, and only when the two disagree — a link at full capability adds nothing a
    reader needs, while one at half width is the whole explanation for a transfer-bound stage
    that used to be fast.
    """
    from batcher._internal.hardware.fabric.device_links import (
        device_pcie_links,
        nearest_rdma_device,
    )

    links = device_pcie_links()
    if index >= len(links):
        return
    link = links[index]
    nic = nearest_rdma_device(index)
    if nic:
        # Which NIC this device should reach the fabric through. A transfer routed via a NIC
        # on the other root complex crosses the inter-socket link twice on its way off the
        # node, and nothing else in the report pairs the two halves.
        row["nearest_nic"] = nic
    if link.numa_node >= 0:
        # Which socket the device hangs off: the host half of its pipeline belongs there too.
        row["numa_node"] = link.numa_node
    if link.degraded:
        row["link_degraded"] = f"gen{link.gen} x{link.width} of gen{link.max_gen} x{link.max_width}"
        row["link_efficiency"] = round(link.degradation_ratio, 3)


def _add_faults(row: dict, faults) -> None:
    """Add the memory-fault counters that predict a device failing, where any are non-zero."""
    if faults is None or not faults.readable:
        return
    if faults.remap_failure:
        row["remap_failure"] = True
    if faults.needs_reset:
        row["reset_pending"] = True
    if faults.remapped_uncorrectable:
        row["remapped_uncorrectable"] = faults.remapped_uncorrectable
    if faults.pcie_replay:
        row["pcie_replay"] = faults.pcie_replay


def _add_modes(row: dict, modes) -> None:
    """Add the device's configuration, where it is costing something.

    Only the findings. A well-configured device contributes nothing here, which is what keeps
    the row readable and makes the day it says `ecc_disabled` worth noticing.
    """
    if modes is None:
        return
    if modes.mig_enabled:
        # Not a finding: partitioning is usually deliberate. It is reported unconditionally
        # because it changes what every other number on the row means — a process handed one
        # instance has a fraction of the memory and a fraction of the SMs.
        row["mig_instances"] = modes.mig_instances
    if modes.findings:
        row["config"] = list(modes.findings)
