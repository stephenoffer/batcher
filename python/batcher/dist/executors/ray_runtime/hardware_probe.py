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

__all__ = ["cluster_l3_cache_bytes"]

# Binding L3 per topology signature, so the probe runs once per distinct cluster shape rather
# than on every query. Autoscaling changes the signature and re-probes.
_L3_BY_TOPOLOGY: dict[tuple, int] = {}

# Bound on how long the driver waits for the probe tasks before giving up and returning `0`.
# Sizing a threshold is not worth stalling a query for, so the wait is short and the fallback
# is the prior behavior.
_PROBE_TIMEOUT_S = 5.0


def _l3_on_this_worker() -> int:
    """Run on a worker: this node's L3 cache in bytes (`0` if undetectable). Layer-0 only."""
    from batcher._internal.hardware import l3_cache_bytes

    return int(l3_cache_bytes())


def cluster_l3_cache_bytes() -> int:
    """L3 cache of the cluster's smallest-cache node shape in bytes, or `0` when unknowable.

    Probes one worker per distinct node shape and returns the minimum, cached by topology.
    Best-effort: returns `0` (the historical "unknown", which leaves the broadcast threshold at
    its config default) on any failure rather than a fabricated or driver-local figure.

    Returns:
        Binding worker L3 cache in bytes, or `0` when the cluster can't be probed.
    """
    try:
        import ray

        if not ray.is_initialized():
            return 0
        nodes = [n for n in ray.nodes() if n.get("Alive", True)]
        reps = _representative_node_ids(nodes)
        if not reps:
            return 0
        signature = tuple(sorted(reps))
        cached = _L3_BY_TOPOLOGY.get(signature)
        if cached is not None:
            return cached
        result = _probe_representatives(ray, reps)
        _L3_BY_TOPOLOGY[signature] = result
        return result
    except Exception:  # pragma: no cover - Ray optional / probe unschedulable
        return 0


def _representative_node_ids(nodes: list[dict]) -> list[str]:
    """One node id per distinct resource shape (cores / GPUs / accelerator type).

    Nodes with identical advertised resources are the same instance type and so have the same
    cache; probing one representative of each shape measures the cluster's real heterogeneity
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


def _probe_representatives(ray, node_ids: list[str]) -> int:
    """Schedule `_l3_on_this_worker` pinned to each representative node; return the min > 0.

    A hard node-affinity pin is what makes the sample cover each distinct shape rather than
    landing wherever the scheduler prefers. A shape that reports `0` (undetectable cache) is
    dropped rather than dragging the minimum to zero; if none report a positive figure the whole
    probe is unknown (`0`).
    """
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    probe = ray.remote(num_cpus=0)(_l3_on_this_worker)
    refs = [
        probe.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False)
        ).remote()
        for node_id in node_ids
    ]
    ready, _ = ray.wait(refs, num_returns=len(refs), timeout=_PROBE_TIMEOUT_S)
    values = [v for v in ray.get(ready) if isinstance(v, int) and v > 0]
    return min(values) if values else 0
