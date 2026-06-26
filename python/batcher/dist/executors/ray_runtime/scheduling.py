"""The metadata-driven scheduling envelope and placement-group machinery.

Carries Carbonite's per-task resource grant (`SchedulingEnvelope`) as an ambient
`ContextVar` so it reaches the Ray-remote wrap step and placement decisions without
threading through every operator function, and turns that grant into Ray
`.options(...)` kwargs and gang-scheduled placement groups.
"""

from __future__ import annotations

import contextlib
import contextvars

from batcher.config import active_config
from batcher.plan.resource import SchedulingEnvelope

# The scheduling grant in force for the current distributed execution. Ambient so it
# reaches the Ray-remote wrap step without threading through every operator function.
_ENVELOPE: contextvars.ContextVar[SchedulingEnvelope | None] = contextvars.ContextVar(
    "batcher_scheduling_envelope", default=None
)


def _placement_timeout_s() -> float:
    """How long to wait for a placement group to be schedulable before falling back
    to default scheduling. Generous (a real cluster may need to autoscale up), but
    bounded so an over-subscribed request degrades gracefully instead of hanging."""
    return active_config().distributed.placement_timeout_s


def current_envelope() -> SchedulingEnvelope | None:
    """The scheduling envelope in force for the current execution, if any."""
    return _ENVELOPE.get()


def set_scheduling_envelope(env: SchedulingEnvelope | None) -> contextvars.Token:
    """Install `env` as the ambient grant; returns a token to `reset` it after."""
    return _ENVELOPE.set(env)


def reset_scheduling_envelope(token: contextvars.Token) -> None:
    _ENVELOPE.reset(token)


def task_options(env: SchedulingEnvelope | None) -> dict:
    """Ray `.options(...)`/`ray.remote(...)` resource kwargs from an envelope.

    `num_gpus` is included only when positive so CPU-only tasks never request a GPU
    (which would make them unschedulable on a GPU-less cluster). `memory` is included
    only when sized (a soft scheduling hint)."""
    if env is None:
        return {}
    opts: dict = {"num_cpus": env.num_cpus}
    if env.memory_bytes > 0:
        opts["memory"] = int(env.memory_bytes)
    if env.num_gpus > 0:
        opts["num_gpus"] = env.num_gpus
        if env.accelerator_type is not None:
            opts["accelerator_type"] = env.accelerator_type
    return opts


def _bundle(env: SchedulingEnvelope | None) -> dict:
    """One placement-group bundle = the resources for a single worker slot."""
    bundle: dict = {"CPU": env.num_cpus if env else 1.0}
    if env and env.num_gpus > 0:
        bundle["GPU"] = env.num_gpus
    if env and env.memory_bytes > 0:
        bundle["memory"] = int(env.memory_bytes)
    return bundle


def create_worker_placement(workers: int, env: SchedulingEnvelope | None):
    """Gang-schedule a placement group of `workers` bundles, SPREAD across nodes.

    One bundle per worker slot (sized from the Carbonite envelope) reserved
    all-at-once, so the whole shuffle fleet exists before the shuffle starts — no
    partial-fleet deadlock — and `SPREAD` distributes the bundles over nodes for even
    data placement and locality. Returns the ready placement group, or `None` when
    placement is unavailable (single worker) or the cluster can't satisfy the request
    within the timeout (the caller then falls back to default scheduling rather than
    hanging — the over-subscription case Phase 4's autoscaler handles properly).
    """
    if workers <= 1:
        return None
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group

    pg = placement_group([_bundle(env) for _ in range(workers)], strategy="SPREAD")
    ready, _ = ray.wait([pg.ready()], timeout=_placement_timeout_s())
    if not ready:
        with contextlib.suppress(Exception):
            remove_placement_group(pg)
        return None
    return pg


def placement_actor_options(pg, index: int) -> dict:
    """Actor `.options(...)` placing worker `index` on bundle `index` of `pg`.

    Carries the envelope's per-task resources and binds the actor to its bundle via
    `PlacementGroupSchedulingStrategy`; with no PG it falls back to the plain
    resource options (default scheduling).
    """
    opts = task_options(current_envelope())
    if pg is None:
        return opts
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    opts["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
        placement_group=pg, placement_group_bundle_index=index
    )
    return opts


def release_placement(pg) -> None:
    """Remove a placement group created for a finished distributed execution."""
    if pg is None:
        return
    import ray

    with contextlib.suppress(Exception):
        ray.util.remove_placement_group(pg)
