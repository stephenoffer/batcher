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
        `power` (the configured budget, the measured draw where telemetry is available, and
        the full-load draw per power zone). A device row carries its host link and fabric
        bandwidth where the model is recognized, which is what makes a transfer-bound stage
        diagnosable from the report alone. On a GPU cloud two more keys appear: `site` (the
        provider, instance type, region, scheduler, and the local scratch volume in force) and
        `fabric` (RDMA port state and aggregate rate, and NVLink link counts). Both are omitted
        where nothing could be read, so the report stays the same size on a laptop. On a live
        Ray cluster with accelerator nodes, `fleet` also carries `health`: one short probe per
        GPU node, since NVML and the kernel log each answer only about the host they run on, so
        a fleet's sick node is invisible from the driver otherwise.

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
    _add_site(report)
    _add_fabric(report)

    from batcher.dist.executors.ray_runtime.fabric import topology_summary

    fleet = topology_summary()
    if fleet.get("gpu_nodes"):
        _add_fleet_health(fleet)
        report["fleet"] = fleet

    energy = active_config().accelerator.energy
    power: dict = {}
    if energy.power_budget_watts > 0:
        power["budget_watts"] = energy.power_budget_watts
    if nvml_available():
        power["draw_watts"] = round(total_power_watts(), 1)
    if fleet.get("power_zones"):
        # Per zone, because the breaker a rack trips is a zone's, not the fleet's: a hall
        # inside its total budget can still have one busway over its own.
        from batcher.dist.executors.ray_runtime.fabric import power_zone_load

        zones = {z: round(w, 1) for z, w in power_zone_load().items() if z}
        if zones:
            power["by_zone_watts"] = zones
    report["power"] = power
    return report


def _add_fleet_health(fleet: dict) -> None:
    """Add the per-node device health of an accelerator fleet, when there is one to ask.

    This is the only part of the report that leaves the driver: NVML, the kernel log, and
    `/sys` each answer about the host they run on, so a fleet's sick node cannot be seen from
    here without asking it. One short task per accelerator node, bounded by the probe timeout,
    and skipped entirely off a cluster — which makes it a cost this report pays only when it
    is the question being asked.

    Nodes that answered clean contribute a count; the ones that did not are listed, because
    "which node do I drain" is the reason to run this.
    """
    from batcher.dist.executors.ray_runtime.hardware_probe import (
        cluster_device_health,
        unhealthy_nodes,
    )

    records = cluster_device_health()
    if not records:
        return
    fleet["health"] = {
        "nodes_probed": len(records),
        "unhealthy": [
            {
                "node_id": r["node_id"],
                "quarantined": r.get("quarantined", []),
                "degraded": r.get("degraded", []),
                "reset_pending": r.get("reset_pending", []),
                "degraded_links": r.get("degraded_links", []),
                "reasons": r.get("reasons", []),
            }
            for r in unhealthy_nodes(records)
        ],
    }


#: What each configuration finding means and what to do about it. Spelled out rather than
#: printed as a reason code, because the reader of this report is an operator deciding whether
#: to reconfigure a node, and `ecc_disabled` on its own does not say why that matters.
_CONFIG_ADVICE = {
    "ecc_disabled": "ECC is OFF: an uncorrectable memory error will not be reported at all",
    "compute_mode_exclusive_process": (
        "compute mode is exclusive: a second worker cannot open this device"
    ),
    "compute_mode_exclusive_thread": (
        "compute mode is exclusive: a second worker cannot open this device"
    ),
    "compute_mode_prohibited": "compute mode is prohibited: no process can use this device",
    "power_at_floor": "power limit is at the part's floor: the device is permanently clamped",
    "persistence_off": (
        "persistence mode is off: every task pays the driver's device initialization again"
    ),
}


def _show_silent_faults(devices: list[dict]) -> None:
    """Call out, by device, the conditions that cost throughput without costing correctness.

    The same treatment thermal clamping and ECC already get, for the two faults that are just
    as invisible from a job's own timings: a host link that renegotiated low, and memory that
    has repaired itself as far as it can. Neither fails a query, neither appears in a profile,
    and both are worth a line in a report someone pastes into a bug.
    """
    for row in devices:
        if row.get("link_degraded"):
            print(
                f"gpu {row['index']}  host link at {row['link_degraded']} "
                f"({row['link_efficiency']:.0%} of nameplate bandwidth)"
            )
        if row.get("remap_failure"):
            print(f"gpu {row['index']}  memory row remapping has FAILED: device needs replacing")
        elif row.get("reset_pending"):
            print(f"gpu {row['index']}  memory repair pending: schedule a device reset")
        for finding in row.get("config", ()):
            print(f"gpu {row['index']}  {_CONFIG_ADVICE.get(finding, finding)}")


def _add_site(report: dict) -> None:
    """Add where this process is running, when the environment says anything at all.

    Omitted entirely on a laptop and in CI, where every field would be empty: a report that
    prints "provider: unknown, scheduler: none" has told the reader nothing and cost them a
    line. On a GPU cloud it is the first thing that explains a default — which mount the spill
    went to, which node list a job was given.
    """
    from batcher._internal.site import scheduler_kind, site_profile, site_summary

    if not (site_profile().known or scheduler_kind() != "none"):
        return
    site = site_summary()
    from batcher._internal.site import local_scratch_root

    scratch = local_scratch_root()
    if scratch:
        site["scratch_dir"] = scratch
    report["site"] = site


def _add_fabric(report: dict) -> None:
    """Add the node's interconnect, when there is one to report.

    Two facts a cross-node stage is bounded by and nothing else in this report carries: what
    the NICs actually sustain, and whether the device fabric is up. Both are omitted on a node
    with neither, which is every machine without RDMA hardware.
    """
    from batcher._internal.hardware.fabric import nvlink_summary, rdma_summary

    fabric: dict = {}
    rdma = rdma_summary()
    if rdma["ports"]:
        from batcher._internal.hardware.fabric import fabric_error_total

        errors = fabric_error_total()
        # Only when non-zero. A clean fabric's counters are a row of zeros, and printing them
        # trains a reader to skip the section that matters on the day they are not zero.
        if any(errors.values()):
            rdma["errors"] = {k: v for k, v in errors.items() if v}
        fabric["rdma"] = rdma
    nvlink = nvlink_summary()
    if nvlink["links"]:
        fabric["nvlink"] = nvlink
    if fabric:
        from batcher.kyber.cost.fabric import net_weight_summary

        # What the fabric measurement did to the plan ranking. Otherwise invisible: two
        # clusters produce different plans for the same query and the report explains neither.
        fabric["cost"] = net_weight_summary()
        report["fabric"] = fabric


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
        _show_silent_faults(devices)
    site = report.get("site")
    if site:
        line = f"site: {site['provider']}"
        if site.get("instance_type"):
            line += f" {site['instance_type']}"
        if site.get("region"):
            line += f" in {site['region']}"
        print(f"{line}, scheduled by {site['scheduler']}")
        if site.get("scratch_dir"):
            print(f"      local scratch {site['scratch_dir']}")
    fabric = report.get("fabric")
    if fabric:
        rdma = fabric.get("rdma")
        if rdma:
            layers = ", ".join(f"{n} x {k}" for k, n in sorted(rdma["link_layers"].items()))
            print(
                f"fabric: {rdma['active_ports']}/{rdma['ports']} RDMA port(s) up "
                f"({rdma['bandwidth_gbps']:.0f} Gb/s, {layers or 'unreported'})"
            )
        nvlink = fabric.get("nvlink")
        if nvlink:
            print(
                f"        NVLink {nvlink['active_links']}/{nvlink['links']} link(s) up "
                f"across {nvlink['devices']} device(s)"
            )
    fleet = report.get("fleet")
    if fleet:
        models = ", ".join(fleet["device_models"]) or "unlabelled"
        print(
            f"fleet: {fleet['gpus']} device(s) on {fleet['gpu_nodes']} node(s), "
            f"widest fabric domain {fleet['largest_domain']}, models {models}"
        )
        if fleet["racks"] or fleet["power_zones"]:
            print(f"       {fleet['racks']} rack(s), {fleet['power_zones']} power zone(s)")
        health = fleet.get("health")
        if health:
            unhealthy = health["unhealthy"]
            print(f"       health: {len(unhealthy)} of {health['nodes_probed']} node(s) degraded")
            for node in unhealthy:
                reasons = ", ".join(node["reasons"]) or "degraded link"
                print(f"       node {node['node_id'][:12]}: {reasons}")
    power = report["power"]
    if power:
        parts = [
            f"{k.replace('_', ' ')} {v}" for k, v in sorted(power.items()) if k != "by_zone_watts"
        ]
        if parts:
            print("power: " + ", ".join(parts))
        for zone, watts in sorted(power.get("by_zone_watts", {}).items()):
            print(f"       zone {zone}: {watts} W at full load")


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

    On the way out, every *measured* stage is folded into the learned statistics, so the next
    run's device choice is made against what this fleet delivers rather than against a
    datasheet ratio. Modelled stages are not: learning from them would teach the optimizer its
    own assumptions back.

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
        try:
            yield ledger
        finally:
            _learn_from(ledger)


def _learn_from(ledger: EnergyLedger) -> None:
    """Fold a completed run's measured efficiency into the learned statistics.

    The conductor's half of the loop the architecture describes: Core measured what each stage
    drew, and this is where that measurement reaches Kyber, so the next run's device choice is
    made against what this fleet actually delivers rather than against a datasheet ratio.
    Only *measured* records are folded — a modelled figure is the datasheet restated, and
    learning from it would teach the optimizer its own assumptions.

    Best-effort: a missing hub, an unreadable backend, or a failed write is skipped rather
    than raised, because a learning path must never fail a query.
    """
    if not ledger.stages:
        return
    try:
        from batcher.core.runtime import default_hub
        from batcher.kyber.gpu import record_measured_efficiency

        hub = default_hub()
        for stage in ledger.stages:
            if not stage.measured or not stage.accelerator_type:
                continue
            if stage.tokens > 0:
                record_measured_efficiency(
                    hub, stage.accelerator_type, stage.joules, stage.tokens, kind="tokens"
                )
            elif stage.rows > 0:
                record_measured_efficiency(
                    hub, stage.accelerator_type, stage.joules, stage.rows, kind="rows"
                )
    except Exception as exc:  # pragma: no cover - learning must never break a query
        from batcher._internal.logging import note_suppressed

        note_suppressed("api", "record measured efficiency", exc)
