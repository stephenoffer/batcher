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
import time

from batcher._internal.logging import get_logger, note_suppressed
from batcher.dist.executors.ray_runtime.scheduling import probe_options

__all__ = [
    "cluster_hardware_profiles",
    "cluster_is_heterogeneous",
    "cluster_l3_cache_bytes",
    "cluster_measured_gpu_memory_bytes",
    "cluster_storage_class",
    "cluster_worker_fingerprint",
    "reset_hardware_probe_cache",
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

# In-flight probes per topology, `{signature: (refs, submitted_at)}`, and how long one may
# stay outstanding before it is written off. See `_probe_representatives` for both.
_PENDING_BY_TOPOLOGY: dict[tuple, tuple[list, float]] = {}
_PROBE_PATIENCE_S = 600.0

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
    with contextlib.suppress(Exception):  # ray optional; the refs go either way
        import ray

        for refs, _ in _PENDING_BY_TOPOLOGY.values():
            _cancel_pending(ray, refs)
    _PENDING_BY_TOPOLOGY.clear()
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
        result = _probe_representatives(ray, reps, signature)
        if not result:
            # Do NOT memoize this. The overwhelmingly common cause is a worker that has not
            # finished starting, which the next query will find running — see the note on
            # `_PROFILES_BY_TOPOLOGY`. Count it instead, so a fleet that never answers still
            # stops paying the wait after `_MAX_PROBE_ATTEMPTS`.
            if signature in _PENDING_BY_TOPOLOGY:
                return ()  # still outstanding: undecided, not failed
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


def _probe_representatives(ray, node_ids: list[str], signature: tuple) -> tuple[dict, ...]:
    """Schedule `_profile_on_this_worker` pinned to each representative node.

    A hard node-affinity pin is what makes the sample cover each distinct shape rather than
    landing wherever the scheduler prefers. A worker that does not answer within the timeout is
    simply absent from *this* result — a slow node must not stall a query for a sizing input —
    but its task is **kept**, not cancelled, so the next call can collect it for nothing.

    **A probe that is merely slow is not a probe that failed**, and treating the two the same
    is what made this permanently blind on the one path where it matters. When Ray was
    initialized by someone other than Batcher the package cannot ride on a job-level
    `runtime_env`, so `probe_options` attaches a per-task one — and Ray then has to
    *materialize* that env on the node (unpack ~90 MB) before the task body runs. That is tens
    of seconds the first time and milliseconds afterwards, so every one of the
    `_MAX_PROBE_ATTEMPTS` tries expired at exactly the five-second deadline, the fleet was
    recorded as settled-unprobeable, and `cluster_l3_cache_bytes()` stayed `0` for the life of
    the driver. Measured on a 64 x 16-core cluster, foreign-`ray.init` path: `profiles: 0 in
    5006 ms` on every attempt, against an unbounded wait that answered in 70 s. Everything
    under it then takes its documented unprobeable fallback, silently — the fan-out stays one
    worker per node, the broadcast threshold defaults, and a worker's coefficients are filed
    under the driver's machine class.

    Holding the refs turns the wait into a *deadline* rather than a verdict: the query that
    paid it moves on with the fallback exactly as before, and the next query collects the
    answer for free (`timeout=0`). At most one outstanding set per topology, never resubmitted
    while it is outstanding — which is also what keeps `_cancel_pending`'s leak (one immortal
    `PENDING_NODE_ASSIGNMENT` task per unreachable node per call) bounded to one. The failure
    budget therefore counts **submissions**, not polls: charging a free poll spent the budget
    in three queries and re-created the very blindness this removes, which is what the first
    version of it did. `_PROBE_PATIENCE_S` bounds the one submission that can be outstanding,
    so a task pinned to a node that will never schedule it is still written off.

    A worker that *raises* is absent for the same reason, which is why each ref resolves on its
    own. One `ray.get(ready)` over the list raises on the first failed task and the caller's
    `except` turns that into `()`, so one unanswerable node discarded every healthy node's
    profile with it. `dist.executors.map._live_actors` resolves per ref too.

    Args:
        ray: The imported `ray` module (passed so a test can substitute it).
        node_ids: One representative node per distinct worker shape.
        signature: The topology key these refs belong to.

    Returns:
        A profile dict per node that answered, empty when none did yet.
    """
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    # Never two outstanding sets for one topology: a resubmit while the first is still pending
    # is exactly the unbounded leak `_cancel_pending` was written against.
    held = _PENDING_BY_TOPOLOGY.pop(signature, None)
    if held is not None:
        refs, submitted_at = held
        wait_s: float = 0.0  # already paid for; collect whatever finished since
    else:
        probe = ray.remote(**probe_options())(_profile_on_this_worker)
        refs = [
            probe.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False)
            ).remote()
            for node_id in node_ids
        ]
        submitted_at = time.monotonic()
        wait_s = _PROBE_TIMEOUT_S
    ready, pending = ray.wait(refs, num_returns=len(refs), timeout=wait_s)
    out: list[dict] = []
    for ref in ready:
        try:
            profile = ray.get(ref)
        except Exception as exc:
            note_suppressed("dist", "probe a worker's hardware profile", exc)
            continue
        if isinstance(profile, dict) and profile:
            out.append(profile)
    if pending and time.monotonic() - submitted_at < _PROBE_PATIENCE_S:
        _PENDING_BY_TOPOLOGY[signature] = (list(pending), submitted_at)
    elif pending:
        _cancel_pending(ray, pending)  # written off: it is not going to schedule
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
