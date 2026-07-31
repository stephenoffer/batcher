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
    from batcher._internal.hardware.amd import amd_devices
    from batcher._internal.hardware.faults import device_faults, device_modes

    live = {d.index: d for d in device_telemetry()}
    faults = {f.index: f for f in device_faults()}
    modes = {m.index: m for m in device_modes()}
    # NVML covers NVIDIA and nothing else, so an MI300X node reached here with every reading
    # empty and read as a healthy idle host. The AMD readings land in the *same* row keys, so
    # every consumer downstream — the printed report, `accelerator_problems`, the Prometheus
    # gauges — works on an AMD node without knowing one exists.
    amd = {d.index: d for d in amd_devices()}
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
        else:
            _add_amd_reading(row, amd.get(index))
        _add_measured_link(row, index)
        _add_faults(row, faults.get(index))
        _add_modes(row, modes.get(index))
        rows.append(row)
    return rows


def _add_amd_reading(row: dict, device) -> None:
    """Fill an AMD device's row from sysfs, using the keys the NVIDIA path already uses.

    Only when NVML reported nothing for this index, so a host with both vendors keeps NVML's
    richer reading for the NVIDIA half rather than having it overwritten by a card that
    happens to sit at the same position.

    Two figures have no NVIDIA counterpart and get their own keys. `hbm_uncorrectable` is an
    unrepairable error in the memory controller, which is what a fatal Xid means on the other
    vendor and carries the same consequence. `serial_number` is here because an AMD board
    publishes one and it is what an RMA is filed against.
    """
    from batcher._internal.hardware.amd import throttled_amd_devices

    if device is None:
        return
    if device.name and not row.get("name"):
        row["name"] = device.name
    if device.memory_total_bytes and not row.get("memory_gib"):
        row["memory_gib"] = round(device.memory_total_bytes / (1 << 30), 1)
    if device.power_watts:
        row["power_watts"] = round(device.power_watts, 1)
    if device.busy_percent:
        row["sm_utilization"] = round(device.busy_percent / 100.0, 3)
    if device.temperature_c:
        row["temperature_c"] = round(device.temperature_c, 1)
    if device.serial_number:
        row["serial_number"] = device.serial_number
    if device.compute_partition:
        # AMD's MIG. Reported unconditionally, like MIG, because it changes what every other
        # figure on the row means: a CPX board presents eight slices, not one device.
        row["partition"] = device.compute_partition
        if device.memory_partition:
            row["partition"] += f"/{device.memory_partition}"
    if device.uncorrectable_errors:
        row["ecc_uncorrected"] = device.uncorrectable_errors
    if device.memory_uncorrectable_errors:
        row["hbm_uncorrectable"] = device.memory_uncorrectable_errors
    # The reason, not just the fact: an AMD board publishes its own cap and its own critical
    # temperature, so the report can say which of the two is holding the clock down instead of
    # printing a bare "throttled" the reader then has to go and diagnose.
    if throttled_amd_devices((device,)):
        reasons = []
        if device.power_cap_watts > 0.0 and device.power_headroom <= 0.02:
            reasons.append(f"at the {device.power_cap_watts:.0f} W board cap")
        if 0.0 < device.thermal_headroom_c <= 3.0:
            reasons.append(f"{device.thermal_headroom_c:.0f} C below the critical limit")
        row["throttled"] = reasons or ["clock limited"]


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
