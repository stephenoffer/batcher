"""Assembling the accelerator report, and saying the part a reader would otherwise miss.

`rows` says what is true of each device. This decides what a reader is shown: the site the
process is on, the fabric it is wired to, the fleet's sick nodes, and — called out by name
rather than left in a table — the conditions that cost throughput without costing
correctness, which are the ones a job's own timings never reveal.
"""

from __future__ import annotations

from batcher.api.session.accelerators.node import node_problems
from batcher.api.session.accelerators.rows import device_rows
from batcher.api.session.accelerators.wires import wire_problems

__all__ = ["accelerator_problems", "accelerators", "show_accelerators"]


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
            >>> sorted({"backend", "devices", "power"} & set(report))  # always present
            ['backend', 'devices', 'power']
            >>> isinstance(report["devices"], list)
            True
    """
    from batcher._internal.hardware import accelerator_backend, nvml_available, total_power_watts
    from batcher.config import active_config

    report: dict = {"backend": accelerator_backend(), "devices": device_rows()}
    _add_site(report)
    _add_container(report)
    _add_fabric(report)

    from batcher.dist.executors.ray_runtime.fabric import topology_summary

    fleet = topology_summary()
    if fleet.get("gpu_nodes"):
        _add_fleet_health(fleet)
        report["fleet"] = fleet
    from batcher.api.session.accelerators.planning import add_planning

    add_planning(report)

    energy = active_config().accelerator.energy
    power: dict = {}
    if energy.power_budget_watts > 0:
        power["budget_watts"] = energy.power_budget_watts
    # Both vendors, and the gate has to admit both too: it was `nvml_available()`, so an
    # Instinct node reported no draw at all rather than the draw it was pulling. A power
    # envelope that under-states a real breaker is the one error in this area with a physical
    # consequence. Still gated, because "unknown" and "zero watts" must not look the same.
    from batcher._internal.hardware.amd import amd_power_watts
    from batcher._internal.hardware.amd import readable as amd_readable

    if nvml_available() or amd_readable():
        power["draw_watts"] = round(total_power_watts() + amd_power_watts(), 1)
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
                # What repairs each condemned device, and the node-level faults that are not
                # about a device at all. Without the first, "quarantined" leaves an operator
                # to look up whether the board comes back after a reset — and for an
                # exhausted row remapper it never does. Without the second, the most common
                # way a node goes bad (the kernel OOM-killing its workers, a spill
                # filesystem remounted read-only) has no entry in the drain list, because
                # every device on such a node reads perfectly healthy.
                "remedies": r.get("remedies", {}),
                "node_faults": r.get("node_faults", {}),
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


def accelerator_problems() -> list[str]:
    """Everything wrong with this node's accelerators, as one list of sentences.

    The machine-readable form of what `show_accelerators` calls out, for the deployment check
    that runs before a fleet takes work rather than the operator reading a report after it
    went slow. Each entry is a complete sentence naming the device and the condition, so a
    failing check can be pasted into an alert without a lookup table.

    Returns:
        The problems, most serious first, empty on a node with nothing wrong *and* on one that
        could read nothing — `bt.accelerators()` is where those are told apart, and a check
        that treats an unreadable node as broken would fail a fleet the day a base image
        stopped shipping `pynvml`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.accelerator_problems()
            []
    """
    report = accelerators()
    out: list[str] = []
    for row in report.get("devices", []):
        index = row.get("index")
        if row.get("remap_failure"):
            out.append(f"gpu {index}: memory row remapping has failed, the device needs replacing")
        if row.get("hbm_uncorrectable"):
            # AMD's equivalent of a fatal Xid: the memory controller could not repair the
            # error, so the data is gone and a reset does not clear it.
            out.append(
                f"gpu {index}: {row['hbm_uncorrectable']} unrepairable HBM error(s), "
                "the device needs replacing"
            )
        if row.get("ecc_uncorrected"):
            out.append(f"gpu {index}: {row['ecc_uncorrected']} uncorrectable ECC error(s)")
        for finding in row.get("config", ()):
            out.append(f"gpu {index}: {_CONFIG_ADVICE.get(finding, finding)}")
        if row.get("reset_pending"):
            out.append(f"gpu {index}: a memory repair is waiting for a device reset")
        if row.get("link_degraded"):
            out.append(
                f"gpu {index}: host link at {row['link_degraded']}, "
                f"{row['link_efficiency']:.0%} of nameplate bandwidth"
            )
        if row.get("throttled"):
            out.append(f"gpu {index}: clocks clamped ({', '.join(row['throttled'])})")
    out.extend(f"container: {finding}" for finding in report.get("container", []))
    wires = report.get("fabric") or {}
    fabric = wires.get("rdma") or {}
    if fabric.get("ports", 0) > fabric.get("active_ports", 0):
        down = fabric["ports"] - fabric["active_ports"]
        out.append(f"fabric: {down} of {fabric['ports']} RDMA port(s) are not carrying traffic")
    out.extend(wire_problems(wires))
    out.extend(node_problems())
    if len(report.get("devices", [])) > 1:
        # Only where a collective can actually happen. On a single-device host the settings
        # are inert, and reporting them would put four lines of advice about multi-GPU
        # failure modes in front of everyone who ran this on a laptop.
        from batcher.carbonite.resilience import collective_findings

        out.extend(collective_findings())
    for node in ((report.get("fleet") or {}).get("health") or {}).get("unhealthy", []):
        reasons = ", ".join(node.get("reasons", ())) or "a degraded device"
        out.append(f"node {node['node_id'][:12]}: {reasons}")
    return out


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
        if row.get("hbm_uncorrectable"):
            print(f"gpu {row['index']}  unrepairable HBM error: device needs replacing")
        if row.get("remap_failure"):
            print(f"gpu {row['index']}  memory row remapping has FAILED: device needs replacing")
        elif row.get("reset_pending"):
            print(f"gpu {row['index']}  memory repair pending: schedule a device reset")
        for finding in row.get("config", ()):
            print(f"gpu {row['index']}  {_CONFIG_ADVICE.get(finding, finding)}")
        if row.get("partition"):
            print(
                f"gpu {row['index']}  partitioned as {row['partition']}:"
                " every figure above is a slice, not the board"
            )
        if row.get("mig_instances"):
            print(
                f"gpu {row['index']}  partitioned into {row['mig_instances']} MIG instance(s):"
                " every figure above is a slice, not the board"
            )


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


def _add_container(report: dict) -> None:
    """Add the container limits that will cost this job something, when there are any.

    Its own section rather than part of `site`, because a container is a container whether or
    not the platform identified itself, and this is the section an operator acts on: each
    finding names the flag that fixes it.
    """
    from batcher._internal.site import container_findings

    findings = container_findings()
    if findings:
        report["container"] = list(findings)


def _add_fabric(report: dict) -> None:
    """Add the node's interconnect, when there is one to report.

    Three facts a cross-node stage is bounded by and nothing else in this report carries: what
    the RDMA ports sustain, what the Ethernet sustains where there is no RDMA, and whether the
    device fabric is up. Ethernet is reported only when there is no RDMA and only when an
    interface publishes a rate, because on a node with a fast fabric the management NIC beside
    it is not what a shuffle takes and printing it invites the wrong conclusion.
    """
    from batcher._internal.hardware.fabric import (
        ethernet_summary,
        nvlink_summary,
        rdma_summary,
    )

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
    else:
        ethernet = ethernet_summary()
        if ethernet["total_gbps"]:
            fabric["ethernet"] = ethernet
    nvlink = nvlink_summary()
    if nvlink["links"]:
        fabric["nvlink"] = nvlink
    from batcher.api.session.accelerators.wires import add_wires

    # Which NIC each device leaves through, and which pairs copy without host memory: the two
    # facts a multi-GPU stage is bounded by that the capability figures above cannot express.
    add_wires(fabric)
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
    report = accelerators()
    print(f"backend: {report['backend']}")
    _print_devices(report)
    _print_site(report)
    for finding in report.get("container", []):
        print(f"container: {finding}")
    _print_fabric(report)
    _print_fleet(report)
    _print_problems()
    _print_power(report)


def _print_devices(report: dict) -> None:
    """The local device table, plus any device the driver is clamping or failing silently."""
    from batcher.observe import format_device_table

    devices = report["devices"]
    if not devices:
        print("devices: none visible to this process")
        return
    print(format_device_table())
    for row in devices:
        if "power_watts" not in row:
            print(f"gpu {row['index']}  {row['name']}  (no live telemetry)")
    _show_silent_faults(devices)


def _print_site(report: dict) -> None:
    """Where this process is running: provider, instance shape, scheduler, local scratch."""
    site = report.get("site")
    if not site:
        return
    line = f"site: {site['provider']}"
    if site.get("instance_type"):
        line += f" {site['instance_type']}"
    if site.get("region"):
        line += f" in {site['region']}"
    if site.get("virtualized"):
        # Worth saying: in a VM an empty fabric or device probe has not proved the host
        # has none, it has proved the hypervisor did not pass one through.
        line += " (virtual machine)"
    print(f"{line}, scheduled by {site['scheduler']}")
    if site.get("scratch_dir"):
        print(f"      local scratch {site['scratch_dir']}")


def _print_fabric(report: dict) -> None:
    """The interconnect: RDMA or Ethernet, NVLink, and what it did to the plan ranking."""
    fabric = report.get("fabric")
    if not fabric:
        return
    rdma = fabric.get("rdma")
    if rdma:
        layers = ", ".join(f"{n} x {k}" for k, n in sorted(rdma["link_layers"].items()))
        print(
            f"fabric: {rdma['active_ports']}/{rdma['ports']} RDMA port(s) up "
            f"({rdma['bandwidth_gbps']:.0f} Gb/s, {layers or 'unreported'})"
        )
    ethernet = fabric.get("ethernet")
    if ethernet:
        print(
            f"fabric: no RDMA; {ethernet['up']}/{ethernet['interfaces']} Ethernet link(s) "
            f"up ({ethernet['total_gbps']:.0f} Gb/s via {ethernet['fastest']})"
        )
    nvlink = fabric.get("nvlink")
    if nvlink:
        print(
            f"        NVLink {nvlink['active_links']}/{nvlink['links']} link(s) up "
            f"across {nvlink['devices']} device(s)"
        )
    cost = fabric.get("cost") or {}
    if cost.get("derived_net_weight") is not None:
        # What the measurement did to the plan ranking. Two clusters producing different
        # plans for the same query is otherwise unexplained by anything printed here.
        print(
            f"        a shuffled byte is priced at {cost['net_weight']:.1f}x a local one "
            f"(measured, against the default 2.0)"
        )


def _print_fleet(report: dict) -> None:
    """The cluster's devices, how they are laid out, and which nodes are degraded."""
    fleet = report.get("fleet")
    if not fleet:
        return
    models = ", ".join(fleet["device_models"]) or "unlabelled"
    print(
        f"fleet: {fleet['gpus']} device(s) on {fleet['gpu_nodes']} node(s), "
        f"widest fabric domain {fleet['largest_domain']}, models {models}"
    )
    if fleet["racks"] or fleet["power_zones"]:
        print(f"       {fleet['racks']} rack(s), {fleet['power_zones']} power zone(s)")
    health = fleet.get("health")
    if not health:
        return
    unhealthy = health["unhealthy"]
    print(f"       health: {len(unhealthy)} of {health['nodes_probed']} node(s) degraded")
    for node in unhealthy:
        reasons = ", ".join(node["reasons"]) or "degraded link"
        print(f"       node {node['node_id'][:12]}: {reasons}")


def _print_problems() -> None:
    """The closing summary, because a reader who scrolled past a device table wants the
    count and the list, not to have reconstructed it from the lines above."""
    problems = accelerator_problems()
    if not problems:
        return
    print(f"problems: {len(problems)}")
    for problem in problems:
        print(f"  - {problem}")


def _print_power(report: dict) -> None:
    """The power envelope, in total and per zone at full load."""
    power = report["power"]
    if not power:
        return
    parts = [f"{k.replace('_', ' ')} {v}" for k, v in sorted(power.items()) if k != "by_zone_watts"]
    if parts:
        print("power: " + ", ".join(parts))
    for zone, watts in sorted(power.get("by_zone_watts", {}).items()):
        print(f"       zone {zone}: {watts} W at full load")
