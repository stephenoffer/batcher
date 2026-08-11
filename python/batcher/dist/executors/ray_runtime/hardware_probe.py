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

import contextlib

from batcher._internal.logging import get_logger, note_suppressed

__all__ = [
    "cluster_device_health",
    "cluster_hardware_profiles",
    "cluster_is_heterogeneous",
    "cluster_l3_cache_bytes",
    "cluster_measured_gpu_memory_bytes",
    "cluster_storage_class",
    "cluster_worker_fingerprint",
    "reset_fleet_health",
    "reset_hardware_probe_cache",
    "sampled_device_health",
    "unhealthy_gpus_by_node",
    "unhealthy_nodes",
    "warn_once_if_fleet_is_mixed",
]

# Worker hardware profiles per topology signature, so the probe runs once per distinct cluster
# shape rather than on every query. Autoscaling changes the signature and re-probes.
#
# **Only a successful probe is memoized.** Caching a failure here is what made a transient
# miss permanent: on an autoscaling fleet the first distributed query of a session routinely
# races worker start-up, the probe's short wait expires, and the empty result was then stored
# against the topology — so every later query in that session planned with default cache
# sizing even though the workers were up and would have answered in milliseconds. Observed on
# a 9-node cluster whose workers were scaling from idle.
_PROFILES_BY_TOPOLOGY: dict[tuple, tuple[dict, ...]] = {}

# Failed attempts per topology, so a genuinely unprobeable fleet still stops paying the wait.
_FAILED_ATTEMPTS: dict[tuple, int] = {}

# How many times a topology may fail before its emptiness is taken as settled.
#
# The cold-start race this exists for is over after the first query — by the second the
# workers are live — so a couple of retries recover it. The bound is what keeps a fleet that
# truly cannot answer (a worker image without the engine) from paying the wait on every query
# for the life of the session.
_MAX_PROBE_ATTEMPTS = 3

_UNPROBEABLE_WARNED = False


def reset_hardware_probe_cache() -> None:
    """Forget every memoized worker profile, failure count, and one-shot warning.

    Two callers need this and neither had it. A **test** substituting a fake topology otherwise
    inherits whatever a previous test memoized against a signature it happens to reproduce. And
    an **operator** who has just fixed the reason a fleet could not answer — the usual one being
    a worker image on a different Batcher build than the driver — is otherwise stuck: after
    `_MAX_PROBE_ATTEMPTS` the emptiness is taken as settled *for the life of the process*, and
    the fix is invisible until the driver restarts. That is the right default (a fleet that
    cannot answer must stop being asked on every query) and the wrong terminal state.

    Deliberately separate from `reset_fleet_health`, which drops the *device-health* sample: one
    is machine shape, which changes when the cluster does, and the other is device condition,
    which changes on its own.
    """
    global _UNPROBEABLE_WARNED, _MIXED_FLEET_WARNED
    _PROFILES_BY_TOPOLOGY.clear()
    _FAILED_ATTEMPTS.clear()
    _UNPROBEABLE_WARNED = False
    _MIXED_FLEET_WARNED = False


def _note_fleet_unprobeable(shapes: int) -> None:
    """Say, once per process, that no worker answered — naming a cause the driver cannot see."""
    global _UNPROBEABLE_WARNED
    _UNPROBEABLE_WARNED = True
    get_logger("dist").warning(
        "no worker answered the hardware probe on any of %d node shape(s); cache-sized and "
        "device-sized planning falls back to defaults. The usual cause is a worker environment "
        "running a different Batcher build than the driver",
        shapes,
    )


# Bound on how long the driver waits for the probe tasks before giving up and returning `0`.
# Sizing a threshold is not worth stalling a query for, so the wait is short and the fallback
# is the prior behavior.
_PROBE_TIMEOUT_S = 5.0


def _cancel_pending(ray, pending) -> None:
    """Cancel probe tasks that never scheduled, so they do not queue forever.

    Both fleet probes pin their task to a specific node with
    `NodeAffinitySchedulingStrategy(..., soft=False)`, which is semantically right — the probe
    measures *that* node, so landing anywhere else would be a wrong reading — and is also the
    one node-affinity mode Ray never gives up on. A hard pin at a node that is gone, drained,
    or simply full leaves the task `PENDING_NODE_ASSIGNMENT` indefinitely; bounding the
    `ray.wait` bounds the *caller*, not the task.

    That matters because the probes are not one-shot. `cluster_hardware_profile` runs per
    planned distributed query and `cluster_device_health` per drain check, so on a churning
    fleet each unreachable node leaks one immortal pending task per call, for the life of the
    driver. They ask for no resources (`num_cpus=0`), so nothing is *reserved* — what
    accumulates is scheduler queue entries and a `ray status` pending list that describes
    nothing anyone is waiting for.

    Best-effort, and the same idiom `gpu.cudf_probe` already applies for the same reason.
    """
    for ref in pending:
        with contextlib.suppress(Exception):
            ray.cancel(ref, force=True)


def _profile_on_this_worker() -> dict:
    """Run on a worker: that node's measured hardware profile. Layer-0 only.

    The whole profile rather than one number, because the probe's cost is the round trip and
    every additional field is free once the task has been scheduled. Cores, memory, cache
    hierarchy, NUMA nodes and the fingerprint all describe how a plan should be sized for
    *this* node shape, and none of them can be read from the driver.

    Device memory is added on top of the profile because it is the one hardware fact the
    control plane otherwise has to *guess*. Ray reports a device count and a model label and
    never a byte figure, so `accelerators.binding_gpu_memory_bytes` recovers the size by looking
    the label up in a nameplate table — which reports `0` for an unlabelled fleet, an on-prem
    part the table has never heard of, a MIG instance (whose usable memory is a fraction of the
    board's), and any device newer than the table. Here the driver is already talking to the
    node that holds the device, and the node can simply say. A measurement, where the existing
    path had a lookup with a documented hole in it.
    """
    from batcher._internal.accelerators import gpu_inventory
    from batcher._internal.hardware import hardware_profile

    profile = hardware_profile().to_dict()
    devices = gpu_inventory()
    sized = [size for d in devices if (size := int(d.get("memory_bytes") or 0)) > 0]
    profile["gpu_count"] = len(devices)
    # The smallest device on the node, for the reason every binding figure is the weakest: a
    # node with a big card and a small one can only host a shard the small one holds. `0` when
    # nothing reported a size, which every reader treats as "unknown" and not as "no memory".
    profile["gpu_memory_bytes"] = min(sized, default=0)
    return profile


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
        reps = _representative_node_ids(_alive_node_records(ray))
        if not reps:
            return ()
        signature = tuple(sorted(reps))
        cached = _PROFILES_BY_TOPOLOGY.get(signature)
        if cached is not None:
            return cached
        if _FAILED_ATTEMPTS.get(signature, 0) >= _MAX_PROBE_ATTEMPTS:
            return ()  # settled: this fleet does not answer, and re-asking only costs the wait
        result = _probe_representatives(ray, reps)
        if not result:
            # Do NOT memoize this. The overwhelmingly common cause is a worker that has not
            # finished starting, which the next query will find running — see the note on
            # `_PROFILES_BY_TOPOLOGY`. Count it instead, so a fleet that never answers still
            # stops paying the wait after `_MAX_PROBE_ATTEMPTS`.
            _FAILED_ATTEMPTS[signature] = _FAILED_ATTEMPTS.get(signature, 0) + 1
            # A fleet where *no* shape answered differs from one never asked, and only this
            # shows it: each node's failure is a DEBUG note, so a wholly mute cluster leaves
            # no mark. Warned once per process, on the last attempt — before that the miss is
            # very likely transient and saying so would train the reader to ignore it.
            if _FAILED_ATTEMPTS[signature] >= _MAX_PROBE_ATTEMPTS and not _UNPROBEABLE_WARNED:
                _note_fleet_unprobeable(len(reps))
            return ()
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


def cluster_storage_class() -> str:
    """The **worst** spill-device class across probed node shapes, `""` when unknowable.

    The worst rather than the commonest, for the reason every binding-node field here takes the
    weakest: a plan whose spill is affordable on the slowest volume it might land on is
    affordable on every node, and the reverse is what produces a query that runs fine on most
    of the fleet and falls over on the rest.

    The spread the ordering encodes is large — a rotational volume costs about thirty times
    local flash for an external merge's concurrent run reads, a network volume about ten — so
    pricing a distributed spill against the *driver's* NVMe is the same class of error as
    pricing it against the driver's RAM. Derived from the same per-shape probe as
    `cluster_hardware_profiles`, so it costs no extra round trip.

    Returns:
        The binding worker's device class, or `""` when the cluster can't be probed.
    """
    from batcher._internal.hardware.storage import (
        SPILL_DEVICE_FACTOR,
        SPILL_DEVICE_FACTOR_DEFAULT,
    )

    classes = [
        found
        for p in cluster_hardware_profiles()
        if (found := str(p.get("storage_class", "") or "")) and found != "unknown"
    ]
    if not classes:
        return ""
    return max(classes, key=lambda c: SPILL_DEVICE_FACTOR.get(c, SPILL_DEVICE_FACTOR_DEFAULT))


def cluster_measured_gpu_memory_bytes() -> int:
    """VRAM of the smallest device the *workers themselves* reported, or `0` when unprobed.

    The measured counterpart of `accelerators.binding_gpu_memory_bytes`, which recovers a size
    from the `ray.io/accelerator-type` label through a nameplate table. That lookup is the only
    thing available when the workers cannot be reached, and it is blind in four situations that
    are not rare:

    * an **unlabelled fleet** — on-prem, or a Ray deployment that does not set node labels;
    * a **part the table has not seen**, which by contract reports unknown rather than guessing;
    * a **MIG instance**, whose usable memory is a seventh or a half of the board the label
      names, so the table's figure is not merely unknown but wrong and too large;
    * a **variant sharing one label**, where the table deliberately records the *smallest*
      shipping configuration — correct as a bound, and up to 2x under the real device.

    In every one of those the workers know the answer exactly, and the probe is already talking
    to them. The minimum across node shapes is taken for the usual reason: a shard sized to the
    largest device out-of-memories on every other one.

    Returns:
        Binding measured VRAM in bytes, or `0` when the cluster can't be probed or reports no
        device — which the caller must read as "fall back to the label lookup", not as "no VRAM".
    """
    sized = [
        size
        for p in cluster_hardware_profiles()
        if (size := int(p.get("gpu_memory_bytes", 0) or 0)) > 0
    ]
    return min(sized) if sized else 0


def cluster_worker_fingerprint() -> str:
    """The hardware-scoping key every probed worker shares, or `""` when they differ.

    The key under which anything learned in *machine units* about this fleet is stored: a cost
    coefficient in nanoseconds per row, a measured CPU utilization, a spill threshold. Kyber
    runs on the driver, which on a cluster executes none of the work, so reading those back
    under the driver's own key describes the wrong machine — and on a fat head node beside small
    workers it is wrong by the whole reason the scoping exists.

    `""` on a mixed fleet, following the same rule as `accelerator_type`: there is no single
    honest answer, and every consumer then falls back to its local key, which is what it did
    before this existed. A mixed fleet is *reported* by `warn_once_if_fleet_is_mixed`, so the
    condition is visible rather than silently degrading.

    Returns:
        The shared worker fingerprint, or `""` when the fleet is mixed or unprobeable.
    """
    keys = {str(p.get("fingerprint", "") or "") for p in cluster_hardware_profiles()}
    keys.discard("")
    return keys.pop() if len(keys) == 1 else ""


#: Granularity the node memory figure is bucketed to before it keys a node shape. Nodes of one
#: instance type report the same RAM to within whatever the kubelet and the object store
#: reserved, so raw bytes would give each node its own shape and turn the representative sample
#: back into an O(nodes) fan-out. A gibibyte separates every instance family that shares a core
#: count (32 / 64 / 128 GiB at sixteen vCPUs) while merging none of them.
_MEMORY_SHAPE_BUCKET = 1 << 30


def _memory_bucket(node_bytes: float) -> int:
    """`node_bytes` rounded to the nearest `_MEMORY_SHAPE_BUCKET` — nearest, never floored.

    Flooring puts a node reporting *just under* a round capacity a whole bucket away from its
    identical peers, and reporting just under is the normal case rather than the exception:
    Ray's `memory` resource is the node's RAM less its object-store reservation, so two nodes
    of one instance type routinely straddle a boundary. `profile._nearest_power_of_two` makes
    exactly this choice for exactly this reason.
    """
    return round(max(0.0, node_bytes) / _MEMORY_SHAPE_BUCKET)


def _alive_node_records(ray) -> list[dict]:
    """Alive node records, from the active `topology_scope()` snapshot when one is held.

    The profile build is wrapped in a scope precisely so its four topology readers share one
    GCS round trip and describe one cluster; reading live here would have left this one outside
    both guarantees, so an autoscale landing mid-build could have the probe sampling a node
    shape that the shape and core-count fields did not contain.
    """
    from batcher.dist.executors.ray_runtime.scaling import _TOPOLOGY

    snapshot = _TOPOLOGY.get()
    nodes = snapshot.alive_nodes if snapshot is not None else ray.nodes()
    return [n for n in nodes if n.get("Alive", True)]


def _worker_nodes(nodes: list[dict]) -> list[dict]:
    """`nodes` minus the Ray head — unless that would leave nothing.

    The probe describes the machines that will *run* the plan, and the head runs none of it:
    worker actors are never placed there. Including it was not a harmless extra sample, because
    every consumer of these profiles takes a binding or an agreement across them:

    * `cluster_l3_cache_bytes` takes the **minimum**, so a modest head node beside large workers
      pinned the broadcast threshold to the head's cache — the exact defaulting this probe
      exists to remove, arriving through the probe itself;
    * `cluster_storage_class` takes the **worst**, so a head on a network root volume priced
      every worker's spill at ten times its real cost;
    * `cluster_worker_fingerprint` requires **agreement**, and a head node is a different
      machine class from its workers on essentially every cluster anyone runs — a fat head
      beside small workers is the shape the surrounding code repeatedly names as the normal
      one. So it returned `""` almost always, and `""` means "fall back to the driver's own
      key": every cost coefficient, CPU share, and learned threshold measured on the workers
      was filed where nothing would read it, silently, on the fleets this was written for.
    * `warn_once_if_fleet_is_mixed` then reported a uniform cluster as mixed, teaching a reader
      to ignore the one message that explains a model failing to converge.

    Survivors-or-nothing, matching `scaling._worker_eligible`: a single-node cluster is its head
    and must still be described.
    """
    from batcher.dist.executors.ray_runtime.scaling import _HEAD_MARKER

    workers = [n for n in nodes if _HEAD_MARKER not in (n.get("Resources") or {})]
    return workers or nodes


def _representative_node_ids(nodes: list[dict]) -> list[str]:
    """One node id per distinct worker shape (cores / memory / GPUs / accelerator type).

    Nodes with identical advertised resources are the same instance type, so they share every
    hardware fact the probe reads — cache, NUMA layout, vector width, scratch device. Probing
    one representative of each shape therefore measures the cluster's real heterogeneity
    without an O(nodes) fan-out on a large fleet.

    **Memory is part of the shape**, because core count alone does not identify an instance
    type and the families that share one are exactly the families a mixed fleet mixes: at
    sixteen vCPUs, AWS alone offers a 32 GiB compute-optimized, a 64 GiB general-purpose and a
    128 GiB memory-optimized node, and on a Graviton fleet the same core count is a different
    vendor and vector width again. All of them collapsed into one shape, so one of them was
    probed and the rest were *assumed* to match it — reporting a uniform cluster, suppressing
    the mixed-fleet notice, and handing the whole fleet one machine class's L3, scratch device,
    and fingerprint. Memory is bucketed rather than compared exactly, for the reason every
    capacity in this engine is; see `_MEMORY_SHAPE_BUCKET`.

    The Ray head is excluded — see [`_worker_nodes`].
    """
    by_shape: dict[tuple, str] = {}
    for n in _worker_nodes(nodes):
        res = n.get("Resources", {})
        cpus = float(res.get("CPU", 0.0))
        if cpus <= 0:
            continue
        node_id = n.get("NodeID")
        if not node_id:
            continue
        labels = n.get("Labels", {}) or {}
        shape = (
            cpus,
            _memory_bucket(float(res.get("memory", 0.0))),
            float(res.get("GPU", 0.0)),
            labels.get("ray.io/accelerator-type"),
        )
        by_shape.setdefault(shape, node_id)  # first node of each shape represents it
    return list(by_shape.values())


def _probe_representatives(ray, node_ids: list[str]) -> tuple[dict, ...]:
    """Schedule `_profile_on_this_worker` pinned to each representative node.

    A hard node-affinity pin is what makes the sample cover each distinct shape rather than
    landing wherever the scheduler prefers. A worker that does not answer within the timeout is
    simply absent from the result — a slow node must not stall a query for a sizing input.

    A worker that *raises* is absent for the same reason, which is why each ref resolves on its
    own. One `ray.get(ready)` over the list raises on the first failed task and the caller's
    `except` turns that into `()`, so one unanswerable node discarded every healthy node's
    profile with it. `dist.executors.map._live_actors` resolves per ref too.
    """
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    probe = ray.remote(num_cpus=0)(_profile_on_this_worker)
    refs = [
        probe.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False)
        ).remote()
        for node_id in node_ids
    ]
    ready, pending = ray.wait(refs, num_returns=len(refs), timeout=_PROBE_TIMEOUT_S)
    _cancel_pending(ray, pending)
    out: list[dict] = []
    for ref in ready:
        try:
            profile = ray.get(ref)
        except Exception as exc:
            note_suppressed("dist", "probe a worker's hardware profile", exc)
            continue
        if isinstance(profile, dict) and profile:
            out.append(profile)
    return tuple(out)


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
    import time

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
        probe = ray.remote(num_cpus=0)(_device_health_on_this_worker)
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
