"""Which workers are on a node that is going away.

Consulted at a stage boundary so the orchestrator can migrate a worker's shuffle output
to a survivor *before* it is reclaimed, turning a reactive recompute into a zero-loss
proactive migration. Split from the policy builders because it reads live cluster and
actor state, where they are pure `active_config()` -> policy functions.
"""

from __future__ import annotations

from batcher._internal.logging import note_suppressed
from batcher.config import active_config

__all__ = ["draining_workers"]

# How long a full drain answer is reused. Matches the cluster-level drain TTL in `scaling`:
# both exist so a barrier poll does not turn a stage-boundary question into a per-second RPC
# storm, and a drain notice arrives tens of seconds ahead of the reclamation it announces.
_DRAIN_POLL_TTL_S = 2.0
#: `(actor handles, monotonic_deadline, worker ids)` for the last full answer.
_draining_cache: tuple[tuple, float, frozenset[int]] | None = None


def draining_workers(actors, workers: int) -> set[int]:
    """Worker ids whose node is going away, for proactive migration.

    A draining worker will be reclaimed shortly, so the caller migrates its shuffle output
    to a survivor *before* it dies (a zero-loss proactive recompute) instead of paying a
    reactive recompute after a failed fetch. Two independent signals answer that, and they
    cover different failures:

    * **Ray's own drain list** (`scaling.draining_node_ids`) — what the *cluster* knows. It
      is set when the autoscaler scales a node in, when KubeRay drains a pod, or when a
      node provider reports a reclamation. Consulted on every cluster, because none of
      those are spot-specific: an ordinary autoscaling cluster scaling in mid-query hits
      exactly this, and it was previously invisible here, so the fleet only learned the
      node was gone by failing to fetch from it. One GCS read, shared with the topology
      snapshot, so a stable cluster pays approximately nothing.
    * **The worker's own preemption monitor** (`is_draining()`) — what the *node* knows. A
      cloud metadata reclamation notice or a `SIGTERM` reaches the node before the cluster
      hears about it, so this is the earlier signal where it applies. It costs a round trip
      per worker, and the monitors only run under the spot profile, so it stays gated
      there.

    A worker that errors on the ping is already gone, so it is reported as draining (it
    needs migrating regardless).

    The whole answer is TTL-cached. This began as a stage-boundary call, where cost did not
    matter, and is now also polled from inside the shuffle barrier twice a second — where
    the per-worker ping is `workers` round trips *per poll*, i.e. hundreds a second on a
    large fleet, spent hardest when a node is already going away. A drain notice precedes
    reclamation by tens of seconds, so seconds of staleness lose nothing.
    """
    global _draining_cache
    import time

    key = tuple(actors[:workers])
    cached = _draining_cache
    now = time.monotonic()
    if cached is not None and cached[0] == key and now < cached[1]:
        return set(cached[2])  # a copy: callers mutate the result

    out = _nodes_draining(actors, workers)
    if active_config().distributed.resilience == "spot":
        import ray

        refs = [actors[i].is_draining.remote() for i in range(workers)]
        for i, ref in enumerate(refs):
            try:
                if ray.get(ref):
                    out.add(i)
            except Exception:
                out.add(i)  # unreachable already ⇒ migrate it proactively
    _draining_cache = (key, now + _DRAIN_POLL_TTL_S, frozenset(out))
    return out


#: Single-slot memo of `(actor handles) -> node id per worker`. One fleet is active at a
#: time, so one slot is the whole working set and nothing accumulates as fleets churn.
_node_id_cache: tuple[tuple, list[str]] | None = None


def _worker_node_ids(actors, workers: int) -> list[str]:
    """The node each worker actor runs on, read once per fleet.

    An actor does not migrate: its node is fixed for its lifetime, so this is immutable data
    behind a remote call. That matters because the caller is polled from inside the shuffle
    barrier — twice a second, for the barrier's whole duration — and reading it live there
    is `workers` RPCs per poll. On a hundred-worker fleet that is two hundred round trips a
    second, spent re-deriving a constant, and spent *hardest* exactly when a node is
    draining and the cluster is already under stress.

    Keyed by the actor handles themselves rather than by list identity, so a rebuilt fleet
    is a miss rather than a stale hit.
    """
    global _node_id_cache
    import ray

    key = tuple(actors[:workers])
    cached = _node_id_cache
    if cached is not None and cached[0] == key:
        return cached[1]
    nodes = ray.get([actors[i].node_id.remote() for i in range(workers)])
    _node_id_cache = (key, nodes)
    return nodes


def _reset_drain_caches() -> None:
    """Drop the memoized node ids and drain answer (tests, and any fresh-probe caller)."""
    global _node_id_cache, _draining_cache
    _node_id_cache = None
    _draining_cache = None


def _nodes_draining(actors, workers: int) -> set[int]:
    """Worker ids sitting on a node Ray has marked for drain; empty when unknowable.

    Skips the per-worker `node_id()` lookup entirely when the cluster reports nothing
    draining, which is the overwhelmingly common case — so a healthy cluster pays one
    TTL-cached GCS read. When something *is* draining the node ids come from a per-fleet
    memo rather than a fresh round trip per worker (see `_worker_node_ids`).
    """
    try:
        from batcher.dist.executors.ray_runtime.scaling import draining_node_ids

        draining = draining_node_ids()
        if not draining:
            return set()
        nodes = _worker_node_ids(actors, workers)
        return {i for i, node in enumerate(nodes) if node in draining}
    except Exception as exc:
        # Proactive migration is an optimization over the reactive recompute path, so a
        # topology or actor read that fails must degrade to that path, never fail a query.
        note_suppressed("dist", "map draining nodes to workers", exc)
        return set()
