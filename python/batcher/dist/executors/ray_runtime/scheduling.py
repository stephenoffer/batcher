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

    Under a wall-clock lease the budget shrinks to what is left. Spending a job's last
    minute waiting for a gang that would be reclaimed the moment it formed leaves nothing
    to run in; giving up sooner falls back to default scheduling, which at least starts.
    `drain_lead_s` is held back so the fallback still has the migration window the drain
    path assumes.
    """
    dc = active_config().distributed
    from batcher.config.deadline import remaining_budget

    return remaining_budget(dc.placement_timeout_s, reserve_s=dc.drain_lead_s)


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


def job_ships_batcher() -> bool:
    """Whether the active Ray job already makes batcher importable on its workers.

    True exactly when Batcher performed the `ray.init` itself, so it doubles as the honest
    witness for "did this process start the cluster or attach to one that was running" — which
    is what `lifecycle._report_attachment` reports. An accessor rather than a direct read of
    the global, because `set_job_ships_batcher` rebinds it and a from-import would freeze the
    value another module saw at import time.
    """
    return _JOB_SHIPS_BATCHER


# Cache the uploaded-package runtime_env (one GCS upload, reused by every task/actor),
# keyed by the **Ray session** it was uploaded into.
#
# It used to be keyed by nothing, on the reasoning that "the driver's batcher package is
# fixed per run". The package is; the *cluster it was uploaded to* is not. The cached value
# is a content-addressed `gcs://_ray_pkg_<hash>.zip` URI, which is meaningful only inside
# the GCS that stored it. A driver that outlives one Ray session — a cluster restart, a
# `ray.shutdown()` and reconnect, a notebook switching between a local and a remote address —
# then attaches a URI pointing into a GCS that no longer holds the package, and every task
# and actor fails in runtime_env setup rather than in anything the user wrote.
_WORKER_RT_ENV: dict | None = None
_WORKER_RT_ENV_SESSION: str | None = None


def ray_session_key() -> str | None:
    """An identifier that changes when the Ray session does, or `None` if Ray is down.

    Used to scope process-global caches that are only valid within one Ray session. The
    job id serves: `ray.init` mints a new one per session, so an equality test against a
    stored key detects a reconnect without reaching into Ray's private node state.
    """
    try:
        import ray

        if not ray.is_initialized():
            return None
        return ray.get_runtime_context().get_job_id()
    except Exception as e:
        # Ray's runtime-context API is the only way to ask, and it is not a stable public
        # contract across versions. Losing the key must not break scheduling: `None` means
        # "unknown session", which makes the caches below re-derive rather than serve a
        # value that may belong to a different cluster.
        note_suppressed("dist", "read ray session key", e)
        return None


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

    The upload is cached per Ray session, so a reconnect re-uploads into the new cluster's
    GCS instead of handing every remote a URI the new cluster cannot resolve.
    """
    global _WORKER_RT_ENV, _WORKER_RT_ENV_SESSION
    if _JOB_SHIPS_BATCHER or active_config().distributed.trust_cluster_image:
        return None
    session = ray_session_key()
    if _WORKER_RT_ENV is not None and session == _WORKER_RT_ENV_SESSION:
        return _WORKER_RT_ENV
    from ray._private.runtime_env.py_modules import upload_py_modules_if_needed

    from batcher._internal.paths import package_dir

    pkg = package_dir()
    # include_gitignore=False → upload the dir verbatim (the maturin-built native
    # `.so` may be gitignored; it must reach the worker for `import batcher` to work).
    rt = upload_py_modules_if_needed({"py_modules": [pkg]}, include_gitignore=False)
    _WORKER_RT_ENV = rt
    _WORKER_RT_ENV_SESSION = session
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


def _resolve_placement_strategy(env: SchedulingEnvelope | None, workers: int | None = None) -> str:
    """The placement strategy for the fleet, resolving the envelope's preference against
    the live cluster.

    Carbonite sets a *preference* (`SPREAD` by default, `PACK`/`STRICT_PACK` for a
    small-shuffle breaker or a co-located GPU collective). This reconciles it with the
    cluster, in both directions:

    * A SPREAD-family preference buys nothing on a single-node cluster — every bundle lands
      on the one node anyway, and PACK skips the (pointless) spread bookkeeping — so it
      degrades to PACK when Ray reports a single alive node.
    * A plain `PACK` preference asks to co-locate the fleet, and Carbonite decides that
      against `cpu_budget`, which is the *driver's* core count because Carbonite has no
      live topology. On a cluster whose nodes are smaller than the driver, that is a
      request to pack a gang no node can hold. Ray's PACK is best-effort so it does not
      hang, but it spends the attempt and then lands the fleet unevenly — the bundles pile
      onto whichever nodes fit until they do not. Downgrading to SPREAD when no single node
      can host the gang asks for the arrangement that is actually available.

    `STRICT_PACK` is never downgraded: it is requested only for a GPU collective, whose
    actors must be co-located to run their own NCCL ring at all. If no node can host that
    world size the gang is genuinely unsatisfiable, and `_report_collective_fabric` says so
    rather than this silently spreading a collective that cannot work spread out.

    Defaults to SPREAD with no envelope.
    """
    # A GPU-collective stage runs its own multi-GPU collective (NCCL/etc.) internally, so
    # its actors must be co-located — gang-schedule them STRICT_PACK regardless of the
    # shuffle-volume preference. (STRICT_PACK on a single node is a no-op.)
    if env is not None and env.gpu_collective:
        return "STRICT_PACK"
    pref = env.placement_strategy if env is not None else "SPREAD"
    from batcher.dist.executors.ray_runtime.scaling import cluster_node_count

    if pref == "STRICT_PACK":
        return pref
    if pref == "PACK":
        return pref if _gang_fits_one_node(env, workers) else "SPREAD"
    # The whole cluster's node count, head included — not the worker-eligible one. The
    # question here is "is there more than one machine to spread over", and the bundles this
    # resolves a strategy for carry no head-excluding resource, so the head is one of the
    # machines they can land on. Asking the worker-eligible count made a head-plus-one-worker
    # cluster read as single-node and pinned its whole fleet to one machine.
    nodes = cluster_node_count()  # snapshot-aware: no extra `ray.nodes()` RPC inside a scope
    return "PACK" if nodes == 1 else pref


def _gang_fits_one_node(env: SchedulingEnvelope | None, workers: int | None = None) -> bool:
    """Whether some single node could host the whole gang at this envelope's grant.

    `workers` is the number of bundles actually being reserved and is what the question is
    about. It is not always `env.n_tasks`: the fleet path spawns a worker count of its own
    (a reused warm fleet, or a count `clamp_workers` reduced) against whatever envelope is
    ambient, so reading the gang size off the envelope tests a fleet that is not the one
    being placed — and gets the answer wrong in both directions, spreading a gang that would
    have fitted or packing one that will not. Falls back to `env.n_tasks` only when the
    caller does not know.

    True when the topology is unreadable, so an unmeasurable cluster keeps the preference
    it was given rather than being second-guessed on no evidence.
    """
    if env is None:
        return True
    from batcher.dist.executors.ray_runtime.scaling import node_classes

    try:
        nodes = node_classes()
    except Exception as exc:  # pragma: no cover - topology read is best-effort
        note_suppressed("dist", "read node classes for the pack decision", exc)
        return True
    if not nodes:
        return True
    # `capacity.placeable_workers` answers a different question — how many fit across the
    # *whole* cluster. PACK needs the widest single node, so the same per-node rule is
    # applied here and the maximum taken instead of the sum.
    widest = 0
    for node in nodes:
        fits = int(float(node["cpus"]) // max(env.num_cpus, 1e-9))
        if env.num_gpus > 0:
            fits = min(fits, int(float(node["gpus"]) // env.num_gpus))
        node_memory = float(node.get("memory", 0.0))
        if env.memory_bytes > 0 and node_memory > 0:
            fits = min(fits, int(node_memory // env.memory_bytes))
        widest = max(widest, fits)
    needed = max(1, int(workers) if workers is not None else env.n_tasks)
    return widest >= needed


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


def _collective_bundles(
    workers: int, env: SchedulingEnvelope | None, node_class: dict
) -> list[dict] | None:
    """The fabric-aware bundle layout for a GPU collective, or `None` to use uniform bundles.

    `plan_collective` already knew how to lay a collective out — inside one coherent domain
    where it fits, filling the largest domain first where it does not, and skipping nodes a
    residency rule or a power-zone budget has excluded. It had no caller: this path built
    `workers` identical bundles and left every one of those constraints to be discovered by
    the placement failing or by the stage running slowly.

    Its bundles carry devices, not worker slots, so they are only usable when the two agree —
    one device per worker. A stage that packs several workers onto a device, or asks for
    several devices each, keeps the uniform layout rather than being reshaped into a gang of a
    different width. The node-class resources are merged in either way, since a CPU-only
    restriction that lives outside the bundle reserves nothing.

    Args:
        workers: Bundles being reserved.
        env: The scheduling envelope, or `None`.
        node_class: Precomputed node-class bundle resources.

    Returns:
        The bundles, or `None` when this is not a collective, the plan produced none, or the
        stage's device-per-worker shape is not the one the plan describes.
    """
    if env is None or not env.gpu_collective or workers <= 1 or env.num_gpus != 1:
        return None
    try:
        from batcher.dist.executors.ray_runtime.fabric import plan_collective

        placement = plan_collective(workers, cpus_per_device=max(env.num_cpus, 1.0))
    except Exception as exc:  # pragma: no cover - a placement hint never fails a placement
        note_suppressed("dist", "plan the collective's bundle layout", exc)
        return None
    if not placement.bundles or sum(b.get("GPU", 0.0) for b in placement.bundles) != workers:
        # A short plan means the fleet cannot host the collective. Reserving the *partial* gang
        # it describes would succeed and then hang the stage on a world size it never gets, so
        # the uniform request is made instead and Ray's own pending path reports it.
        return None
    bundles = [dict(b) for b in placement.bundles]
    if node_class:
        for bundle in bundles:
            bundle.update(node_class)
    if env.memory_bytes > 0:
        for bundle in bundles:
            bundle["memory"] = int(env.memory_bytes * bundle.get("GPU", 1.0))
    return bundles


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

    On a cluster spanning availability zones the bundles additionally carry a one-zone label
    selector (`_fleet_zone_selector`), because a shuffle's bytes are billed and delayed by
    the zone boundary they cross and the bundles are interchangeable.
    """
    if workers <= 1:
        return None
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group

    # Resolve the topology-dependent bits ONCE for the whole fleet (each reads `ray.nodes()`);
    # building W bundles must not re-read the cluster W times (O(workers x nodes)).
    node_class = _fleet_node_class_resources(env)
    strategy = _resolve_placement_strategy(env, workers)
    _report_collective_fabric(workers, env)
    bundles = _collective_bundles(workers, env, node_class) or [
        _bundle(env, node_class) for _ in range(workers)
    ]
    zone = _fleet_zone_selector(len(bundles), env)
    pg = _reserve(placement_group, bundles, strategy, zone)
    ready, _ = ray.wait([pg.ready()], timeout=_placement_timeout_s())
    if not ready:
        with contextlib.suppress(Exception):
            remove_placement_group(pg)
        _report_placement_timeout(workers, env, strategy)
        return None
    _report_placement(len(bundles), strategy, zone)
    return pg


def _report_placement(bundles: int, strategy: str, zone: dict[str, str]) -> None:
    """Record where the fleet was reserved, on the same bus the fan-out chain reports to.

    Placement is the other half of "why did my query run the way it did", and it was
    invisible in exactly the cases a reader asks about: a SPREAD that quietly became PACK on a
    one-node cluster, a fleet pinned into one availability zone, a collective that got
    STRICT_PACK. Reported as a `Decision` so it lands in `explain(analyze=True)` and the
    dashboard beside Kyber's and Carbonite's, rather than only in a log nobody enabled.

    Never raises: this describes a reservation that has already succeeded.
    """
    try:
        from batcher._internal import events
        from batcher.plan.profile import Decision

        where = f" in {next(iter(zone.values()))}" if zone else ""
        events.publish(
            events.DECISION,
            **Decision(
                subsystem="core",
                category="placement",
                summary=f"reserved {bundles} bundle(s) {strategy}{where}",
                detail={"bundles": bundles, "strategy": strategy, "zone": dict(zone)},
            ).to_dict(),
        )
    except Exception as exc:  # pragma: no cover - observation must never fail a placement
        note_suppressed("dist", "report the fleet placement", exc)


def _reserve(placement_group, bundles: list[dict], strategy: str, zone: dict[str, str]):
    """Create the group, with the zone selector when the Ray in use accepts one.

    `bundle_label_selector` is newer than the rest of the placement API, so a cluster running
    an older Ray rejects the keyword. That must cost the zone preference and nothing else —
    the fleet still has to be reserved — so the unpinned form is the fallback rather than an
    error.
    """
    if not zone:
        return placement_group(bundles, strategy=strategy)
    try:
        return placement_group(
            bundles, strategy=strategy, bundle_label_selector=[dict(zone)] * len(bundles)
        )
    except TypeError as exc:
        note_suppressed("dist", "pin the fleet to one availability zone", exc)
        return placement_group(bundles, strategy=strategy)


def _fleet_zone_selector(workers: int, env: SchedulingEnvelope | None) -> dict[str, str]:
    """The one-zone bundle label selector for this fleet, or `{}`.

    Gated on `distributed.zone_aware_placement` and on the fleet being one whose traffic
    crosses the zone boundary at all. A GPU collective is excluded: it is already STRICT_PACK
    onto a single node, so it is inside one zone by construction, and adding a selector to it
    could only narrow which node that is.
    """
    if env is not None and env.gpu_collective:
        return {}
    if not active_config().distributed.zone_aware_placement:
        return {}
    try:
        from .capacity import Demand, preferred_fleet_zone

        return preferred_fleet_zone(workers, Demand.from_envelope(env, count=workers))
    except Exception as exc:  # pragma: no cover - a cost hint never fails a placement
        note_suppressed("dist", "choose an availability zone for the fleet", exc)
        return {}


def _report_placement_timeout(workers: int, env: SchedulingEnvelope | None, strategy: str) -> None:
    """Say why the gang did not form, at the moment the reservation is given up on.

    The fallback to default scheduling is silent, and that silence is expensive: the tasks
    it falls back to ask for the same per-task resources the bundles did, so whatever made
    the group unsatisfiable usually makes them unschedulable too — and the barrier that
    gathers them has no deadline. The query then hangs with nothing anywhere saying a
    reservation was attempted, let alone why it failed.

    When the ask is one no node can host, that is said outright, because no amount of
    waiting or autoscaling fixes a bundle wider than the widest machine. Otherwise the
    timeout is reported as what it is — a cluster that was busy for longer than the budget
    — which is ordinary on a shared cluster and must not read as an error.
    """
    from batcher._internal.logging import get_logger, log_kv

    from .capacity import Demand, describe_pending_demand

    try:
        reason = describe_pending_demand(Demand.from_envelope(env, count=workers))
    except Exception as exc:  # pragma: no cover - a diagnostic never fails a placement
        note_suppressed("dist", "diagnose the placement timeout", exc)
        reason = None
    log_kv(
        get_logger("dist"),
        logging.WARNING,
        "placement group did not form within the timeout; falling back to default scheduling",
        workers=workers,
        strategy=strategy,
        reason=reason or "cluster busy for longer than placement_timeout_s",
    )


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
