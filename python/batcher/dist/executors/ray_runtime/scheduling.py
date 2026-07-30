"""The metadata-driven scheduling envelope and placement-group machinery.

Carries Carbonite's per-task resource grant (`SchedulingEnvelope`) as an ambient
`ContextVar` so it reaches the Ray-remote wrap step and placement decisions without
threading through every operator function, and turns that grant into Ray
`.options(...)` kwargs and gang-scheduled placement groups.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging

from batcher._internal.logging import note_suppressed
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
    bounded so the *reservation* cannot hang.

    Note what this does and does not buy. Timing out here bounds the wait for the
    bundles; it does not bound the query. The fallback tasks ask for the same per-task
    CPUs the bundles did, so the reason the group was unsatisfiable is usually the
    reason they will not schedule either — and unlike this call, the barrier that
    gathers them (`gather_with_backups`) has no deadline. Measured on a 96-CPU node
    with two concurrent distributed sessions: all 96 CPUs held inside one session's
    placement group, `{'CPU': 48.0}: 2+ pending tasks/actors` for the other, and its
    window shuffle stuck in `ray.wait` for 17+ minutes with no error. The barrier now
    says so after two minutes; making it *fail* instead of wait is an open decision,
    because a legitimately slow first task looks identical from inside `ray.wait`.
    """
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
        # Custom accelerator resources: Ray reports NVIDIA/AMD/Intel/MetaX as `GPU`
        # (covered by `num_gpus`) but everything else as a named resource — `TPU`,
        # `neuron_cores`, `HPU`, `NPU` — as does any resource an operator defined on their
        # own cluster. Merged, not assigned, so it composes with the CPU-only node selector
        # below rather than one silently overwriting the other.
        if env.resources:
            opts["resources"] = {**opts.get("resources", {}), **dict(env.resources)}
        # Applied for any accelerator, not just GPUs: a TPU/Trainium node has `num_gpus == 0`,
        # so gating this on GPUs dropped the device-model pin on exactly the hardware that
        # most needs it, letting the task land anywhere in the cluster.
        if env.accelerator_type is not None and (env.num_gpus > 0 or env.resources):
            opts["accelerator_type"] = env.accelerator_type
        # Hard-restrict a CPU-only fleet to CPU-only nodes when the cluster opts in and can
        # host it (a no-op otherwise). Keeps a CPU shuffle from stealing an inference
        # stage's GPU-node cores; additive to Ray's soft GPU-node avoidance.
        from batcher.dist.executors.ray_runtime.scaling import node_class_selector

        sel = node_class_selector(env.prefer_cpu_only_nodes, env.n_tasks, env.num_cpus)
        if sel:
            opts["resources"] = {**opts.get("resources", {}), **sel["resources"]}
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
    from ray._private.runtime_env.py_modules import upload_py_modules_if_needed

    from batcher._internal.paths import package_dir

    pkg = package_dir()
    # include_gitignore=False → upload the dir verbatim (the maturin-built native
    # `.so` may be gitignored; it must reach the worker for `import batcher` to work).
    rt = upload_py_modules_if_needed({"py_modules": [pkg]}, include_gitignore=False)
    _WORKER_RT_ENV = rt
    _WORKER_RT_ENV_DONE = True
    return _WORKER_RT_ENV


def _fleet_node_class_resources(env: SchedulingEnvelope | None) -> dict:
    """The node-class selector's bundle resources for a whole fleet — computed **once**.

    `node_class_selector` reads the live topology (`ray.nodes()`), so it MUST NOT be called
    per bundle/per worker: at W workers on N nodes that is an O(W x N) cost (thousands of
    `ray.nodes()` RPCs on the driver). The selector is fleet-uniform, so a caller building a
    W-bundle placement group or launching W actors computes this once and reuses it.
    """
    if env is None:
        return {}
    from batcher.dist.executors.ray_runtime.scaling import node_class_selector

    sel = node_class_selector(env.prefer_cpu_only_nodes, env.n_tasks, env.num_cpus)
    return sel.get("resources", {}) if sel else {}


def _bundle(env: SchedulingEnvelope | None, node_class: dict | None = None) -> dict:
    """One placement-group bundle = the resources for a single worker slot.

    `node_class` is the precomputed fleet node-class selector (see
    `_fleet_node_class_resources`); it is threaded in rather than recomputed here so a
    W-bundle fleet reads the topology once, not W times. Falls back to computing it for a
    lone-bundle caller that passes nothing.
    """
    bundle: dict = {"CPU": env.num_cpus if env else 1.0}
    if env and env.num_gpus > 0:
        bundle["GPU"] = env.num_gpus
    if env and env.memory_bytes > 0:
        bundle["memory"] = int(env.memory_bytes)
    # Custom accelerator resources belong in the bundle for the same reason the node-class
    # selector below does: a bundle reserves by resource, so a `TPU`/`neuron_cores`/`HPU`
    # request that lives only in `.options()` reserves nothing. The gang would then be
    # placed on whatever nodes satisfied CPU alone, and each task would afterwards demand
    # an accelerator its own bundle never held — pending forever on a CPU node, or
    # oversubscribing the one accelerator node the group happened to land on.
    if env and env.resources:
        bundle.update(dict(env.resources))
    # A PG bundle is matched by resource, so the CPU-only restriction must live in the
    # bundle (not just `.options`) for the gang to land on CPU-only nodes.
    extra = _fleet_node_class_resources(env) if node_class is None else node_class
    if extra:
        bundle.update(extra)
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
    # A GPU-collective stage runs its own multi-GPU collective (NCCL/etc.) internally, so
    # its actors must be co-located — gang-schedule them STRICT_PACK regardless of the
    # shuffle-volume preference. (STRICT_PACK on a single node is a no-op.)
    if env is not None and env.gpu_collective:
        return "STRICT_PACK"
    pref = env.placement_strategy if env is not None else "SPREAD"
    if pref in ("PACK", "STRICT_PACK"):
        return pref
    from batcher.dist.executors.ray_runtime.scaling import alive_node_count

    nodes = alive_node_count()  # snapshot-aware: no extra `ray.nodes()` RPC inside a scope
    return "PACK" if nodes == 1 else pref


def _report_collective_fabric(workers: int, env: SchedulingEnvelope | None) -> None:
    """Log where a gang-scheduled collective sits relative to the fleet's fabric domains.

    STRICT_PACK already puts a collective's actors on one node. What it cannot do is make a
    node wide enough: a world size above the widest NVLink domain the fleet has runs its
    all-reduce over PCIe or the network at a fraction of the fabric rate. That is invisible
    from the job's own timings — the run is simply slower — so it is recorded here, where the
    world size and the topology are both known, rather than left to be rediscovered.

    Best-effort and never raises: this is an observation about a placement that has already
    been decided, and a fleet whose topology cannot be read keeps the placement it had.
    """
    if env is None or not env.gpu_collective or workers <= 1:
        return
    if not active_config().accelerator.fabric_aware_placement:
        return
    try:
        from batcher._internal.logging import get_logger, log_kv
        from batcher.dist.executors.ray_runtime.fabric import largest_local_domain

        widest = largest_local_domain()
        if widest <= 0:
            return  # unreadable or unlabelled topology: no observation to make
        log = get_logger("dist")
        if workers > widest:
            log_kv(
                log,
                logging.WARNING,
                "collective wider than the fleet's fabric domain",
                world_size=workers,
                widest_domain=widest,
                effect="all-reduce leaves NVLink for the host bus or network",
            )
        else:
            log_kv(log, logging.DEBUG, "collective fits one fabric domain", world_size=workers)
    except Exception as exc:  # observation only: never fail a placement over it
        note_suppressed("dist", "report collective fabric", exc)


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

    # Resolve the topology-dependent bits ONCE for the whole fleet (each reads `ray.nodes()`);
    # building W bundles must not re-read the cluster W times (O(workers x nodes)).
    node_class = _fleet_node_class_resources(env)
    strategy = _resolve_placement_strategy(env)
    _report_collective_fabric(workers, env)
    pg = placement_group([_bundle(env, node_class) for _ in range(workers)], strategy=strategy)
    ready, _ = ray.wait([pg.ready()], timeout=_placement_timeout_s())
    if not ready:
        with contextlib.suppress(Exception):
            remove_placement_group(pg)
        return None
    return pg


def placement_actor_options(pg, index: int, base: dict | None = None) -> dict:
    """Actor `.options(...)` placing worker `index` on bundle `index` of `pg`.

    Carries the envelope's per-task resources and binds the actor to its bundle via
    `PlacementGroupSchedulingStrategy`; with no PG it falls back to the plain
    resource options (default scheduling).

    `base` is the fleet-uniform `task_options(...)` result computed **once** by the caller
    (`fleet_actor_options`): `task_options` reads the live topology for the node-class
    selector, so recomputing it per worker is an O(workers x nodes) cost at scale. Falls
    back to computing it here for a lone-actor caller that passes nothing.
    """
    opts = dict(base) if base is not None else task_options(current_envelope())
    if pg is None:
        return opts
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    opts["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
        placement_group=pg, placement_group_bundle_index=index
    )
    return opts


def fleet_actor_options(pg, workers: int) -> list[dict]:
    """Per-worker actor `.options(...)` for a whole fleet, resolving the shared parts once.

    `task_options(current_envelope())` reads the live topology (node-class selector) and is
    fleet-uniform, so it is computed a single time here and only the per-bundle index varies
    — turning a W-actor launch from O(workers x nodes) topology reads into one.
    """
    base = task_options(current_envelope())
    return [placement_actor_options(pg, i, base) for i in range(workers)]


def release_placement(pg) -> None:
    """Remove a placement group created for a finished distributed execution."""
    if pg is None:
        return
    import ray

    with contextlib.suppress(Exception):
        ray.util.remove_placement_group(pg)
