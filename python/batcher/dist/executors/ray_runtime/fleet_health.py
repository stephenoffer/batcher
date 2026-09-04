"""Live device health across the fleet — every accelerator node, never cached.

Split from `hardware_probe`, which it imports, and separate from it in the two ways that
decide how each is measured. The hardware *profile* is a property of an instance type, so it
is sampled one representative node per shape and memoized by topology; a device *fault* is a
property of one board, so this probes **every** accelerator node — sampling would report the
fleet healthy on the strength of its healthy nodes — and it is only ever reused for
`_HEALTH_SAMPLE_TTL_S`, because the whole value is that a device which faulted a minute ago
is seen now.

The dependency points one way: this module reads `hardware_probe`'s probe plumbing
(`_cancel_pending`, `_PROBE_TIMEOUT_S`) and its profiles; nothing there imports this.
"""

from __future__ import annotations

import time

from batcher._internal.logging import note_suppressed
from batcher.dist.executors.ray_runtime.hardware_probe import (
    _PROBE_TIMEOUT_S,
    _cancel_pending,
)
from batcher.dist.executors.ray_runtime.scheduling import probe_options

__all__ = [
    "cluster_device_health",
    "reset_fleet_health",
    "sampled_device_health",
    "unhealthy_gpus_by_node",
    "unhealthy_nodes",
]

def _device_health_on_this_worker() -> dict:
    """Run on a GPU worker: that node's device verdicts and interconnect state.

    Everything here is invisible from the driver. NVML answers only about the host it runs on,
    the kernel log only about that host's driver, and `/sys` only about that host's wires — so
    on a fleet the difference between "no device is sick" and "no device that the driver can
    see is sick" is the difference between a report and a guess.
    """
    from batcher._internal.hardware.amd import ecc_faulted_amd_devices
    from batcher._internal.hardware.fabric import (
        degraded_device_links,
        fabric_error_total,
        nvlink_summary,
    )
    from batcher._internal.hardware.faults import (
        device_remedy,
        faulted_devices,
        misconfigured_devices,
        node_fault_counts,
        node_faults,
        node_faults_readable,
        worst_severity,
        xid_application_faults,
        xid_fatal,
        xid_readable,
        xid_unclassified,
    )
    from batcher.carbonite.accel import (
        assess_fleet,
        device_affinity_summary,
        device_reset_candidates,
    )

    verdicts = assess_fleet()
    return {
        # The fabric's own error history, which is how a failing cable announces itself: the
        # port stays `ACTIVE` and the errors climb, so a node whose counters stand out against
        # its neighbours has hardware to check before it drops a stage.
        "fabric_errors": fabric_error_total(),
        "devices": len(verdicts),
        "quarantined": [v.uuid or v.device_index for v in verdicts if not v.schedulable],
        "degraded": [v.uuid or v.device_index for v in verdicts if v.state == "degraded"],
        "reasons": sorted({r for v in verdicts for r in v.reasons}),
        "reset_pending": list(device_reset_candidates()),
        # The memory faults behind those verdicts, and the settings that cost this node
        # something without failing anything. Neither is a drain reason on its own — a
        # device with ECC off is working, it is simply not reporting — but both are what an
        # operator reconciles a slow node against.
        "faulted": [f.uuid or f.index for f in faulted_devices()]
        + [d.unique_id or d.index for d in ecc_faulted_amd_devices()],
        "config_findings": sorted({f for m in misconfigured_devices() for f in m.findings}),
        # How this worker's host half is placed against the device it feeds, and whether the
        # device is its own. Both are per-worker facts the driver cannot see, and both explain
        # a node that is slower than its identical neighbours without being faulty.
        "affinity": device_affinity_summary(),
        "degraded_links": [link.address for link in degraded_device_links()],
        "nvlink": nvlink_summary(),
        "xid_readable": xid_readable(),
        # Workload-caused Xids, kept apart from the hardware ones the verdicts act on. A
        # device here needs no operator action — the job that faulted on it does — and a
        # drain list that mixed the two would take healthy boards out over someone's
        # out-of-bounds write, one retry at a time.
        "xid_application": sorted(
            {code for codes in xid_application_faults().values() for code in codes}
        ),
        # Codes this build classifies as neither hardware nor workload. Nothing acts on them
        # — inventing a severity for an unseen code is how a driver release quarantines a
        # fleet — but they are the most interesting line in the log on a node that keeps
        # failing, because the vendor documents them and this build does not. Dropped
        # silently, they become months of "those nodes are just flaky".
        "xid_unclassified": sorted(
            {code for codes in xid_unclassified().values() for code in codes}
        ),
        # What to *do* about each condemned device, per PCI address. A drain list that says
        # "quarantined" and nothing else leaves an operator to look up whether the board comes
        # back after a reset — and for an exhausted row remapper it never does, so a slot sits
        # down while its ticket reads "pending reset".
        "remedies": {
            address: device_remedy(codes) for address, codes in sorted(xid_fatal().items())
        },
        # The node faults that are not about the device at all, and that leave no trace
        # anywhere a Python traceback can reach: the OOM killer having already fired here, a
        # filesystem remounted read-only under the spill directory, a PCIe link retraining.
        # A node failing every task for one of these looks identical, from the driver, to a
        # node with a bad GPU.
        "node_faults": node_fault_counts(node_faults()),
        "node_fault_severity": worst_severity(node_faults()),
        "kernel_log_readable": node_faults_readable(),
        # Whether this node can still write where it spills. Every stateful operator spills,
        # so a node whose scratch filesystem filled or went read-only fails every task placed
        # on it — with every GPU on it reading perfectly healthy, and with the scheduler
        # still seeing a free slot, which is what turns it into a retry storm.
        "scratch": _scratch_status(),
    }


def _scratch_status() -> str:
    """This node's spill directory as `"ok"`, `"warn"`, `"failed"`, or `"unknown"`.

    Reuses Carbonite's node readiness check rather than restating it, so the answer a fleet
    report gives and the answer a worker's own check gives cannot diverge — two different
    notions of "can this node spill" is exactly the kind of drift that makes a health report
    stop being believed.
    """
    from batcher.carbonite.resilience import preflight_check
    from batcher.config import active_config

    report = preflight_check(scratch_path=active_config().memory.spill_dir or "")
    return next((c.status for c in report.checks if c.name == "scratch"), "unknown")


#: How long a fleet-health sample is reused before every accelerator node is asked again.
#:
#: The probe is a task per GPU node, and its callers are not all reports: the collective
#: placement filter runs per placement decision, so an unsampled probe would put a fleet-wide
#: round trip on a scheduling path — the exact cost the representative-sampling in
#: `cluster_hardware_profiles` above exists to avoid.
#:
#: Thirty seconds is chosen against what is being measured, not against the callers. A
#: quarantined device stays quarantined until an operator resets or replaces it, which is
#: minutes at best; a device that faults *during* the window is caught on the next sample and
#: costs one stage's placement, against a probe on every placement forever.
_HEALTH_TTL_S = 30.0

_HEALTH_SAMPLE: dict[str, object] = {"expires": 0.0, "value": ()}


def reset_fleet_health() -> None:
    """Drop the fleet-health sample, so the next call re-probes every node.

    For a test, and for an operator who has just reset a device and wants the next report to
    say so rather than repeating a thirty-second-old verdict.
    """
    _HEALTH_SAMPLE.update(expires=0.0, value=())


def cluster_device_health() -> tuple[dict, ...]:
    """One device-health record per accelerator node, sampled.

    The fleet-wide view of the faults that do not fail a job. A node whose NVLink is down, whose
    host link renegotiated, or which holds a device the driver has condemned keeps accepting
    work and returning correct answers at a fraction of the rate — and on a hundred-node fleet
    nobody finds it by reading timings.

    Returns:
        One record per GPU node that answered, each carrying the node id, its device verdicts
        and reasons, its degraded links, and its NVLink summary. Empty when Ray is down, when
        the cluster has no accelerator nodes, or when no worker answered inside the timeout —
        a slow node must not stall the caller, and an unanswered probe is reported as absence
        rather than as health.
    """

    now = time.monotonic()
    if now < float(_HEALTH_SAMPLE["expires"]):  # type: ignore[arg-type]
        return _HEALTH_SAMPLE["value"]  # type: ignore[return-value]
    probed = _probe_fleet_health()
    # Only a successful probe is cached. An unreadable fleet is not a fact worth holding for
    # thirty seconds — the cluster may be seconds from coming up — and caching it would make
    # a transient failure decide the next half-minute of placements.
    if probed:
        _HEALTH_SAMPLE.update(expires=now + _HEALTH_TTL_S, value=probed)
    return probed


def sampled_device_health() -> tuple[dict, ...]:
    """The fleet-health sample **if one is already in hand**, without probing for it.

    `cluster_device_health` fans a task out to every accelerator node, which is the right cost
    for a health report and the wrong one for a path that runs per planned query. This is the
    read for such a path: it returns what the last sample said while that sample is still fresh,
    and an empty tuple otherwise.

    That asymmetry is deliberate rather than a compromise. The alternative designs are both
    worse: probing from the planner puts a fleet-wide round trip on every optimize, and caching
    a *stale* verdict forever makes a device that recovered stay condemned. Reporting "no
    information" until something else has paid for the sample keeps planning free and keeps the
    verdict fresh, and every consumer already treats absence as "assume healthy" — which is
    exactly the behavior that held before health reached the plan at all.

    Returns:
        The current health records, or empty when none has been sampled recently.
    """

    if time.monotonic() < float(_HEALTH_SAMPLE["expires"]):  # type: ignore[arg-type]
        return _HEALTH_SAMPLE["value"]  # type: ignore[return-value]
    return ()


def unhealthy_gpus_by_node(records: tuple[dict, ...] | None = None) -> dict[str, int]:
    """How many devices each node has out of rotation, from a health sample.

    Quarantined and degraded devices are counted together and deduplicated: a board the fleet
    will not schedule on and a board running at a fraction of its rate are different conditions,
    but a fan-out sized against either one asks for capacity it will not get. Deduplicated
    because a device is routinely reported under both.

    Args:
        records: Health records, or `None` to use the already-sampled ones (never a fresh probe;
            see `sampled_device_health`).

    Returns:
        Node id to the count of devices out of rotation. Nodes absent from the map — including
        every node when nothing has been sampled — are treated as fully healthy by the caller,
        which is the behavior that held before this existed.
    """
    out: dict[str, int] = {}
    for record in sampled_device_health() if records is None else records:
        node_id = str(record.get("node_id") or "")
        if not node_id:
            continue
        down = {str(d) for d in (record.get("quarantined") or ())}
        down |= {str(d) for d in (record.get("degraded") or ())}
        if down:
            out[node_id] = len(down)
    return out


def _probe_fleet_health() -> tuple[dict, ...]:
    """The unsampled fan-out behind `cluster_device_health`."""
    try:
        import ray

        if not ray.is_initialized():
            return ()
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        nodes = [
            n
            for n in ray.nodes()
            if n.get("Alive", True) and float((n.get("Resources") or {}).get("GPU", 0.0)) > 0
        ]
        if not nodes:
            return ()
        probe = ray.remote(**probe_options())(_device_health_on_this_worker)
        refs = {
            probe.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False)
            ).remote(): n["NodeID"]
            for n in nodes
            if n.get("NodeID")
        }
        ready, pending = ray.wait(list(refs), num_returns=len(refs), timeout=_PROBE_TIMEOUT_S)
        _cancel_pending(ray, pending)
        out = []
        for ref in ready:
            # Per node: the one whose probe raises (NVML throwing, a wedged driver) is
            # disproportionately the sick one. Letting it escape to the caller's `except`
            # discarded every record so far and returned `()`, which `unhealthy_nodes()` reads
            # as "nothing to drain" — one sick node made the fleet report clean.
            try:
                record = ray.get(ref)
            except Exception as exc:
                note_suppressed("dist", "probe a worker's device health", exc)
                continue
            if isinstance(record, dict):
                out.append({"node_id": refs[ref], **record})
        return tuple(out)
    except Exception as exc:
        note_suppressed("dist", "probe the fleet's device health", exc)
        return ()


def unhealthy_nodes(records: tuple[dict, ...] | None = None) -> tuple[dict, ...]:
    """The nodes holding a device that should not be scheduled, or one running degraded.

    The list an operator drains. Ordered as the probe returned them, which is node order.

    Args:
        records: Health records, or `None` to probe the fleet.

    Returns:
        The subset with a quarantined device, a degraded device, a pending reset, a degraded
        host link, a partially-down NVLink fabric, an RDMA port that has dropped, or a *node*
        fault the kernel called fatal. Empty on a healthy fleet *and* on one that could not be
        probed; `cluster_device_health()` returning nothing is what distinguishes them.
    """
    probed = cluster_device_health() if records is None else records
    return tuple(
        r
        for r in probed
        if r.get("quarantined")
        or r.get("degraded")
        or r.get("reset_pending")
        or r.get("degraded_links")
        or (r.get("nvlink") or {}).get("degraded_devices")
        # A link that has actually dropped, as opposed to one merely accumulating symbol
        # errors: the first cost a stage its in-flight transfers, the second is a warning.
        or (r.get("fabric_errors") or {}).get("link_downed")
        # A node whose kernel has already OOM-killed a process here, or remounted the spill
        # filesystem read-only, fails every task placed on it while every device on it reads
        # perfectly healthy — so without this the drain list has no entry for the most common
        # way a node goes bad.
        or r.get("node_fault_severity") == "fatal"
        # Same shape, different cause: a node that cannot write where it spills fails every
        # stateful operator placed on it and reads healthy by every other measure here.
        or r.get("scratch") == "failed"
    )
