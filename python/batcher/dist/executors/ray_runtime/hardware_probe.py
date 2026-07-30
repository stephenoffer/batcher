"""Worker-side hardware facts Ray's topology cannot report, collected by a probe.

`ray.nodes()` reports cores, memory, and custom resources, but not everything a plan is sized
against. The L3 cache is the case that bites: Kyber's broadcast-join threshold is sized to the
cache the hash table must stay resident in (`kyber.rules.selection`), and on a distributed run
that number was simply never collected — `cluster_hardware_profile` left it `0`, so every
cluster query fell back to the config default regardless of the workers' real cache.

The only way to learn a worker's cache is to ask the worker, so this runs a tiny remote task
that returns `l3_cache_bytes()` from the node itself. Two properties make it honest on a
heterogeneous cluster:

* it probes **one worker per distinct node shape** (grouped by cores / GPUs / accelerator
  type), not one worker full stop, so a cluster of two instance types is measured as two, not
  assumed uniform from a single sample; and
* it takes the **minimum** across those shapes, because a broadcast table sized to the largest
  cache would spill out of the smallest node's cache the plan might land on.

Everything is best-effort and cached by cluster shape: the probe runs once per distinct
topology, and any failure (Ray down, a worker that can't answer, a timeout) returns `0` — the
exact value the field held before, so a cluster that can't be probed plans as it always did.
"""

from __future__ import annotations

from batcher._internal.logging import get_logger, note_suppressed

__all__ = [
    "cluster_device_health",
    "cluster_hardware_profiles",
    "cluster_is_heterogeneous",
    "cluster_l3_cache_bytes",
    "reset_fleet_health",
    "unhealthy_nodes",
    "warn_once_if_fleet_is_mixed",
]

# Worker hardware profiles per topology signature, so the probe runs once per distinct cluster
# shape rather than on every query. Autoscaling changes the signature and re-probes.
_PROFILES_BY_TOPOLOGY: dict[tuple, tuple[dict, ...]] = {}

# Bound on how long the driver waits for the probe tasks before giving up and returning `0`.
# Sizing a threshold is not worth stalling a query for, so the wait is short and the fallback
# is the prior behavior.
_PROBE_TIMEOUT_S = 5.0


def _profile_on_this_worker() -> dict:
    """Run on a worker: that node's measured hardware profile. Layer-0 only.

    The whole profile rather than one number, because the probe's cost is the round trip and
    every additional field is free once the task has been scheduled. Cores, memory, cache
    hierarchy, NUMA nodes and the fingerprint all describe how a plan should be sized for
    *this* node shape, and none of them can be read from the driver.
    """
    from batcher._internal.hardware import hardware_profile

    return hardware_profile().to_dict()


def cluster_hardware_profiles() -> tuple[dict, ...]:
    """One measured hardware profile per distinct worker node shape, cached by topology.

    The cluster's real composition, as opposed to the driver's own machine — which is what
    every other in-process hardware reading describes, and which on a cluster is frequently a
    small head node that runs none of the work.

    Best-effort and empty on any failure (Ray absent or down, the probe unschedulable, a
    worker that cannot answer within the timeout), so a cluster that cannot be probed plans
    exactly as it did before.

    Returns:
        A profile dict per node shape, in no particular order; empty when unprobeable.
    """
    try:
        import ray

        if not ray.is_initialized():
            return ()
        nodes = [n for n in ray.nodes() if n.get("Alive", True)]
        reps = _representative_node_ids(nodes)
        if not reps:
            return ()
        signature = tuple(sorted(reps))
        cached = _PROFILES_BY_TOPOLOGY.get(signature)
        if cached is not None:
            return cached
        result = _probe_representatives(ray, reps)
        _PROFILES_BY_TOPOLOGY[signature] = result
        return result
    except Exception as exc:  # pragma: no cover - Ray optional / probe unschedulable
        note_suppressed("dist", "probe ray node hardware", exc)
        return ()


def cluster_is_heterogeneous() -> bool:
    """Whether the cluster's workers span more than one machine class.

    The fact that decides whether anything measured on one worker generalizes to another. On a
    uniform fleet a coefficient learned anywhere is true everywhere, and learning converges as
    fast as the whole cluster can produce feedback. On a mixed fleet it is true only on the
    nodes that share its fingerprint, which is why feedback is scoped by fingerprint rather
    than pooled — see `metadata.hardware_scope`.

    Worth surfacing because a mixed cluster is invisible from the driver and is the usual
    explanation for a model that will not converge: an autoscaling group quietly substituting
    a newer instance generation makes every node's history half about a machine it is not.

    Returns:
        `True` when two probed node shapes report different fingerprints. `False` when the
        cluster is uniform, single-shape, or unprobeable — never a guess.
    """
    profiles = cluster_hardware_profiles()
    return len({p.get("fingerprint", "") for p in profiles}) > 1


def cluster_l3_cache_bytes() -> int:
    """L3 cache of the cluster's smallest-cache node shape in bytes, or `0` when unknowable.

    The minimum across node shapes, because a broadcast table sized to the largest cache would
    spill out of the smallest node's cache the plan might land on. Derived from the same
    per-shape probe as `cluster_hardware_profiles`, so it costs no extra round trip.

    Best-effort: returns `0` (the historical "unknown", which leaves the broadcast threshold at
    its config default) on any failure rather than a fabricated or driver-local figure.

    Returns:
        Binding worker L3 cache in bytes, or `0` when the cluster can't be probed.
    """
    sizes = [
        int(caches.get("l3", 0))
        for p in cluster_hardware_profiles()
        if isinstance(caches := p.get("caches", {}), dict)
    ]
    # A shape reporting `0` (undetectable cache) is dropped rather than dragging the minimum to
    # zero; if none report a positive figure the whole probe is unknown.
    positive = [s for s in sizes if s > 0]
    return min(positive) if positive else 0


def _representative_node_ids(nodes: list[dict]) -> list[str]:
    """One node id per distinct resource shape (cores / GPUs / accelerator type).

    Nodes with identical advertised resources are the same instance type, so they share every
    hardware fact the probe reads — cache, NUMA layout, vector width, scratch device. Probing
    one representative of each shape therefore measures the cluster's real heterogeneity
    without an O(nodes) fan-out on a large fleet.
    """
    by_shape: dict[tuple, str] = {}
    for n in nodes:
        res = n.get("Resources", {})
        cpus = float(res.get("CPU", 0.0))
        if cpus <= 0:
            continue
        node_id = n.get("NodeID")
        if not node_id:
            continue
        labels = n.get("Labels", {}) or {}
        shape = (cpus, float(res.get("GPU", 0.0)), labels.get("ray.io/accelerator-type"))
        by_shape.setdefault(shape, node_id)  # first node of each shape represents it
    return list(by_shape.values())


def _probe_representatives(ray, node_ids: list[str]) -> tuple[dict, ...]:
    """Schedule `_profile_on_this_worker` pinned to each representative node.

    A hard node-affinity pin is what makes the sample cover each distinct shape rather than
    landing wherever the scheduler prefers. A worker that does not answer within the timeout is
    simply absent from the result — a slow node must not stall a query for a sizing input.
    """
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    probe = ray.remote(num_cpus=0)(_profile_on_this_worker)
    refs = [
        probe.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False)
        ).remote()
        for node_id in node_ids
    ]
    ready, _ = ray.wait(refs, num_returns=len(refs), timeout=_PROBE_TIMEOUT_S)
    return tuple(p for p in ray.get(ready) if isinstance(p, dict) and p)


# Set once the mixed-fleet warning has been emitted. A cluster's composition does not change
# between queries often enough to be worth saying twice, and a per-query warning on a
# long-running session is noise that trains the reader to ignore it.
_MIXED_FLEET_WARNED = False


def warn_once_if_fleet_is_mixed() -> None:
    """Say so, once, when the cluster's workers span more than one machine class.

    A mixed fleet is invisible from the driver and is the usual explanation for a learned model
    that will not converge: everything Batcher learns from measurement — per-row costs, memory
    per group, batch sizes — is true of the machine that measured it, so on a mixed fleet each
    node's history is partly about hardware it is not. Feedback is scoped by hardware
    fingerprint so the models stay separate and correct, and the cost of that correctness is
    that each shape converges on its own share of the traffic rather than on all of it.

    That is the right trade and it is not a fault, so this is informational rather than a
    warning about a defect. It exists because the alternative is a user watching plans improve
    more slowly than expected with nothing anywhere to explain why.

    """
    global _MIXED_FLEET_WARNED
    if _MIXED_FLEET_WARNED or not cluster_is_heterogeneous():
        return
    _MIXED_FLEET_WARNED = True
    get_logger("dist").info(
        "cluster mixes machine classes; learned costs, memory models and batch sizes are kept "
        "per hardware fingerprint, so each node shape converges on its own share of the runs"
    )


# --- Fleet device health ------------------------------------------------------------------
#
# Separate from the hardware profiles above in two ways that matter. It probes *every*
# accelerator node rather than one representative of each shape, because a fault is a property
# of one board and not of an instance type — sampling would report the fleet healthy on the
# strength of its healthy nodes. And it is never cached: the whole value is that a device that
# faulted a minute ago is seen now.


def _device_health_on_this_worker() -> dict:
    """Run on a GPU worker: that node's device verdicts and interconnect state.

    Everything here is invisible from the driver. NVML answers only about the host it runs on,
    the kernel log only about that host's driver, and `/sys` only about that host's wires — so
    on a fleet the difference between "no device is sick" and "no device that the driver can
    see is sick" is the difference between a report and a guess.
    """
    from batcher._internal.hardware.fabric import (
        degraded_device_links,
        fabric_error_total,
        nvlink_summary,
    )
    from batcher._internal.hardware.faults import xid_readable
    from batcher.carbonite.accel import assess_fleet, device_reset_candidates

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
        "degraded_links": [link.address for link in degraded_device_links()],
        "nvlink": nvlink_summary(),
        "xid_readable": xid_readable(),
    }


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
    import time

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
        probe = ray.remote(num_cpus=0)(_device_health_on_this_worker)
        refs = {
            probe.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False)
            ).remote(): n["NodeID"]
            for n in nodes
            if n.get("NodeID")
        }
        ready, _ = ray.wait(list(refs), num_returns=len(refs), timeout=_PROBE_TIMEOUT_S)
        out = []
        for ref in ready:
            record = ray.get(ref)
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
        host link, a partially-down NVLink fabric, or an RDMA port that has dropped. Empty on
        a healthy fleet *and* on one that could not be probed; `cluster_device_health()`
        returning nothing is what distinguishes them.
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
    )
