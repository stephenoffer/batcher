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
    only when sized (a soft scheduling hint). A `runtime_env` that ships the driver's
    batcher package is attached when the job didn't already ship it (see
    `worker_runtime_env`), so every batcher task/actor can `import batcher` regardless
    of who initialized Ray."""
    opts: dict = {}
    if env is not None:
        opts["num_cpus"] = env.num_cpus
        if env.memory_bytes > 0:
            opts["memory"] = int(env.memory_bytes)
        if env.num_gpus > 0:
            opts["num_gpus"] = env.num_gpus
            if env.accelerator_type is not None:
                opts["accelerator_type"] = env.accelerator_type
    rt = worker_runtime_env()
    if rt is not None:
        opts["runtime_env"] = rt
    return opts


# Whether the Ray *job* already ships batcher to workers (batcher initialized Ray
# itself — a local cluster shares the driver's modules, a remote one got the
# self-shipped `runtime_env`). When False (a foreign `ray.init` ran before batcher),
# batcher attaches its package to each remote via `worker_runtime_env` instead.
_JOB_SHIPS_BATCHER = True


def set_job_ships_batcher(value: bool) -> None:
    """Record whether the active Ray job already makes batcher importable on workers."""
    global _JOB_SHIPS_BATCHER
    _JOB_SHIPS_BATCHER = value


# Cache the uploaded-package runtime_env for the process (one GCS upload, reused by
# every task/actor). Keyed by nothing — the driver's batcher package is fixed per run.
_WORKER_RT_ENV: dict | None = None
_WORKER_RT_ENV_DONE = False


def worker_runtime_env() -> dict | None:
    """A per-remote Ray `runtime_env` shipping the driver's batcher, or `None`.

    Returns `{"py_modules": ["gcs://...zip"]}` only when the job does **not** already
    ship batcher (a foreign `ray.init` ran first, so batcher couldn't set a job-level
    `runtime_env`) and the cluster image isn't trusted. Ray rejects a local directory
    in a *task/actor*-level `runtime_env` (dir uploads are job-level only), so the
    driver's batcher package is uploaded to the GCS once via Ray's own packaging
    helper and referenced by its content-addressed URI thereafter — one cached
    transfer per process, attachable to any number of remotes. This guarantees
    `import batcher` on every worker independent of Ray init order (the gap that made
    a user's own `ray.init()` silently break distributed runs). Returns `None` for the
    common case where batcher initialized Ray itself.
    """
    global _WORKER_RT_ENV, _WORKER_RT_ENV_DONE
    if _JOB_SHIPS_BATCHER or active_config().distributed.trust_cluster_image:
        return None
    if _WORKER_RT_ENV_DONE:
        return _WORKER_RT_ENV
    import os

    from ray._private.runtime_env.py_modules import upload_py_modules_if_needed

    import batcher

    pkg = os.path.dirname(os.path.abspath(batcher.__file__))
    # include_gitignore=False → upload the dir verbatim (the maturin-built native
    # `.so` may be gitignored; it must reach the worker for `import batcher` to work).
    rt = upload_py_modules_if_needed({"py_modules": [pkg]}, include_gitignore=False)
    _WORKER_RT_ENV = rt
    _WORKER_RT_ENV_DONE = True
    return _WORKER_RT_ENV


def _bundle(env: SchedulingEnvelope | None) -> dict:
    """One placement-group bundle = the resources for a single worker slot."""
    bundle: dict = {"CPU": env.num_cpus if env else 1.0}
    if env and env.num_gpus > 0:
        bundle["GPU"] = env.num_gpus
    if env and env.memory_bytes > 0:
        bundle["memory"] = int(env.memory_bytes)
    return bundle


def _resolve_placement_strategy(env: SchedulingEnvelope | None) -> str:
    """The placement strategy for the fleet, resolving the envelope's preference against
    the live cluster.

    Carbonite sets a *preference* (`SPREAD` by default, `PACK`/`STRICT_PACK` for a
    small-shuffle breaker or a co-located GPU collective). A SPREAD-family preference
    buys nothing on a single-node cluster — every bundle lands on the one node anyway,
    and PACK skips the (pointless) spread bookkeeping — so it degrades to PACK when Ray
    reports a single alive node. A PACK-family preference is honored as-is. Defaults to
    SPREAD with no envelope.
    """
    pref = env.placement_strategy if env is not None else "SPREAD"
    if pref in ("PACK", "STRICT_PACK"):
        return pref
    try:
        import ray

        nodes = sum(1 for n in ray.nodes() if n.get("Alive", True))
    except Exception:
        nodes = 0
    return "PACK" if nodes == 1 else pref


def create_worker_placement(workers: int, env: SchedulingEnvelope | None):
    """Gang-schedule a placement group of `workers` bundles across nodes.

    One bundle per worker slot (sized from the Carbonite envelope) reserved
    all-at-once, so the whole shuffle fleet exists before the shuffle starts — no
    partial-fleet deadlock. The strategy is resolved from the envelope's preference
    against the live cluster (`_resolve_placement_strategy`): SPREAD distributes the
    bundles over nodes for even data placement (the default), PACK co-locates them for a
    small shuffle / GPU collective / single-node cluster. Returns the ready placement
    group, or `None` when placement is unavailable (single worker) or the cluster can't
    satisfy the request within the timeout (the caller then falls back to default
    scheduling rather than hanging — the over-subscription case the autoscaler handles).
    """
    if workers <= 1:
        return None
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group

    pg = placement_group(
        [_bundle(env) for _ in range(workers)], strategy=_resolve_placement_strategy(env)
    )
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
