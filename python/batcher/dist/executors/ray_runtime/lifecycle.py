"""Ray lifecycle + single-node fallback for the distributed executor.

`_ensure_ray` initializes Ray and wraps the module-level task functions (which live
in the per-operator `executors.*` modules) as `ray.remote`, preserving picklability
— the tasks stay top-level module functions; only their module-bound names are
rebound to the remote wrappers.

The wrapping carries the **metadata-driven scheduling envelope** Carbonite produced:
each task is wrapped with `ray.remote(num_cpus=, memory=, num_gpus=)` from the
ambient `SchedulingEnvelope`, so worker placement reflects estimated per-task CPU,
memory, and (for GPU map/inference tasks) GPU demand instead of Ray's implicit
one-CPU default. The envelope is ambient (a `ContextVar` set by
`execute_distributed`) so it reaches the wrap step without threading through every
operator function; tasks are re-wrapped when the resource grant changes.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import threading

import pyarrow as pa

from batcher._internal.paths import package_dir
from batcher.config import active_config
from batcher.io.source import Source, read_source
from batcher.plan.logical import LogicalPlan

from .policies import actor_fault_options, fault_options
from .scaling import cluster_topology
from .scheduling import current_envelope, set_job_ships_batcher, task_options


def engine_config_json(num_cpus: float | None = None) -> str:
    """The driver's active `EngineConfig` (morsel size, parallelism) as JSON, to
    ship into remote tasks.

    A Ray worker's `active_config()` sees only that process's own default (the
    driver's `config_context` does not cross the process boundary), so the driver
    must capture this here and pass it as a task argument to every worker-side
    `execute_plan` — otherwise distributed runs silently ignore the session config.

    `num_cpus` pins the rayon width to *this task's* CPU grant, for a caller whose tasks are
    not all sized by the ambient envelope. The map path is exactly that: it sizes each task's
    `num_cpus` from its own partition (`map._adaptive_task_cpus`), so the envelope's
    per-worker grant is not what any individual map task actually holds. Omit it and the
    envelope's grant is used (the shuffle operators, whose tasks are uniform).

    When a `SchedulingEnvelope` is in force (the ambient Carbonite grant for the
    current distributed execution), its per-task `memory_bytes` is folded into
    `memory_budget_bytes` so each worker's `execute_plan` spills its reducer bucket
    within its share of the envelope instead of OOMing — the distributed arm of the
    "Carbonite protects against OOM" invariant. The tighter of the envelope grant
    and any global cap wins; with no global cap, the envelope alone enables spill
    (so distributed survival does not require the user to set `max_memory_bytes`).
    """
    base = active_config().engine_config_json()
    env = current_envelope()
    cfg = json.loads(base)

    # Pin the worker's rayon width to the CPUs it was actually GRANTED. The driver's
    # config says `parallelism: 0` — "use every core" — which is right on the driver and
    # badly wrong on a remote worker: a Ray task/actor holding a 1-CPU bundle would still
    # open a rayon pool over the whole NODE. On a 16-core node running 16 such workers
    # that is 256 threads contending for 16 cores, with 16 duplicate sets of per-thread
    # allocator arenas and morsel buffers — the thrash (and the worker OOM-kills) that
    # made a distributed shuffle slower than one node. Sizing the pool to the grant is the
    # Carbonite contract: a worker uses exactly the resources it was admitted for.
    # `parallelism` never changes a result, only how many threads compute it.
    if not cfg.get("parallelism"):
        if num_cpus is not None:
            grant = num_cpus
        elif env is not None:
            grant = env.num_cpus
        else:
            grant = 1.0
        cfg["parallelism"] = max(1, int(grant))

    if env is not None and env.memory_bytes > 0:
        existing = int(cfg.get("memory_budget_bytes", 0) or 0)
        cfg["memory_budget_bytes"] = (
            env.memory_bytes if existing <= 0 else min(existing, env.memory_bytes)
        )
    return json.dumps(cfg)


# Names of the module-level Ray task functions, keyed by the module they live in.
# `_ensure_ray` rebinds each `<module>.<name> = ray.remote(<module>.<name>)`.
_TASK_FUNCS: dict[str, tuple[str, ...]] = {
    "batcher.dist.executors.map": ("_map_udf_task", "_map_agg_task", "_MapActor"),
    "batcher.dist.executors.aggregate": ("_map_task", "_reduce_task"),
    "batcher.dist.executors.join": (
        "_join_map_task",
        "_join_reduce_task",
        "_broadcast_join_task",
        "_join_detect_task",
    ),
    "batcher.dist.executors.sort": ("_sample_task", "_range_task", "_sort_reduce_task"),
    "batcher.dist.executors.window": ("_map_task", "_reduce_task"),
    "batcher.dist.executors.write": ("_write_shard", "_write_plan_shard"),
    "batcher.dist.streaming.microbatch": ("_stage_shard",),
}

# Unwrapped originals (so re-wrapping with a new grant never double-wraps) and the
# resource signature the task fns are currently wrapped with.
_originals: dict[tuple[str, str], object] = {}
_wrapped_resources: tuple | None = None
# Guards the module-global rebind in `_wrap_tasks`: two concurrent distributed
# queries with different envelopes must not interleave their re-wraps and hand one
# query's tasks the other's resource grant (N12).
_wrap_lock = threading.Lock()


def _ray_init_kwargs(
    workers: int, *, force_attach: bool = False, force_local: bool = False
) -> dict:
    """`ray.init(**kwargs)` for the active config — attach to a cluster or spin local.

    Attach to a *running* cluster when an address is configured (`Config` or the
    `RAY_ADDRESS` env var Ray itself honors) OR a managed control plane is detected
    (`detect_managed_cluster` — e.g. an Anyscale workspace), shipping `batcher` + its
    native extension via `runtime_env` so workers can run the data plane. On a managed
    workspace that exports neither `RAY_ADDRESS` nor Ray's current-cluster pointer, a bare
    `ray.init()` would silently start a *local* single-node Ray and strand a distributed job
    on one node — so a managed signal routes to `address="auto"` here, and `_ensure_ray`
    falls back to a local start only if no cluster turns out to be reachable. Only when no
    address is configured *and* no cluster is detected/reachable do we start a *local*
    cluster capped at `workers` CPUs (the single-node / test path); against a real cluster we
    leave fan-out to the scheduler/autoscaler and never pin `num_cpus`.

    `force_attach` forces the discoverable-cluster attach (`address="auto"`); `force_local`
    forces the local start (the reachability fallback). Both take precedence over detection."""
    import os

    from batcher.config.profiles import detect_managed_cluster

    dc = active_config().distributed
    kwargs: dict = {
        "include_dashboard": dc.dashboard,
        "logging_level": "ERROR",
        "ignore_reinit_error": True,
        "namespace": dc.namespace,
    }
    # `RAY_ADDRESS`'s *value* is the address to attach to, not merely a signal that one
    # exists. Collapsing it to `"auto"` throws away the only disambiguation Ray offers when a
    # host has more than one live instance — a managed cluster plus a stray local Ray started
    # by a colocated test run — and `ray.init(address="auto")` then dies with "Found multiple
    # active Ray instances ... set the RAY_ADDRESS environment variable", which is precisely
    # what the user did. `"auto"` remains the default when it names no specific address.
    env_address = os.environ.get("RAY_ADDRESS")
    attach = not force_local and (force_attach or env_address or detect_managed_cluster())
    if not force_local and dc.ray_address:
        kwargs["address"] = dc.ray_address
    elif attach:
        kwargs["address"] = env_address or "auto"
    else:
        kwargs["num_cpus"] = workers
        # Only a locally started Ray accepts an object-store size; attaching to an
        # existing cluster (the branches above) would be rejected by Ray.
        if dc.object_store_memory_bytes is not None:
            kwargs["object_store_memory"] = int(dc.object_store_memory_bytes)
    # Ship the data plane to workers. An explicit `runtime_env` wins; otherwise, when
    # attaching to a *cluster* (not a local single-process Ray), auto-ship the batcher
    # package if it is a source/editable install the worker image won't already carry —
    # the flight workers import `batcher` + its native extension to run, and die with
    # `ModuleNotFoundError` without it. A no-op for a normal site-packages install.
    if dc.runtime_env is not None:
        kwargs["runtime_env"] = dc.runtime_env
    elif "address" in kwargs:
        shipped = _self_ship_runtime_env()
        if shipped is not None:
            kwargs["runtime_env"] = shipped
    return kwargs


# Build output a managed workspace's *injected* `working_dir` may contain, excluded from
# the Ray upload. None of it is source, all of it is regenerable, and any one of these can
# on its own exceed Ray's 512 MiB package cap and fail `ray.init` (a `cargo` target/ runs
# to gigabytes). Patterns are Ray `excludes` globs, matched relative to the working dir.
_BUILD_ARTIFACT_EXCLUDES = (
    "**/target/debug/**",  # cargo
    "**/target/release/**",
    "**/docs/_build/**",  # sphinx
    "**/.git/**",
    "**/__pycache__/**",
    "**/.pytest_cache/**",
    "**/node_modules/**",
    "**/.venv/**",
)


def _self_ship_runtime_env() -> dict | None:
    """A Ray `runtime_env` that uploads the driver's batcher package to workers.

    Returns `{"py_modules": [<batcher pkg dir>]}` so the driver's *exact* batcher
    package (abi3 native extension included) runs on every worker — Ray caches the
    upload in the object store, so it is a one-time ~10MB transfer per package
    content, not per query. This is correctness-first: a driver that pip-installed
    batcher and attached to an arbitrary cluster cannot assume that cluster's image
    carries a *compatible* batcher, and shipping guarantees driver==worker code. (The
    old heuristic skipped shipping for any site-/dist-packages install, which produced
    a silent `ModuleNotFoundError` on workers for the common local-install →
    remote-cluster case.)

    The env also pins `pip: None`. Batcher ships its own package here and relies on each
    worker's base environment for every other dependency (Batcher's "ship my code, trust
    the cluster image" contract). Pinning `pip` off makes that explicit and, critically,
    stops a managed runtime-env hook (some platforms inject the workspace's
    `requirements.txt` as the job's pip) from re-installing dependencies per job — a
    redundant round-trip that also hard-fails when that list names a local editable such as
    `batcher-engine` itself (unresolvable on any index, and already shipped here via
    `py_modules`). A worker's base env already carries the workspace deps, so nothing is
    lost; a job that genuinely needs extra per-worker pip deps sets `distributed.runtime_env`
    (the explicit-override branch above), which wins outright.

    The env also carries `excludes` (see `_BUILD_ARTIFACT_EXCLUDES`). Batcher does not set
    a `working_dir` — but a managed workspace *injects* one (its whole project directory),
    and Ray zips it on every `ray.init`. A project that has been built in place carries its
    build output there, and Ray hard-caps that upload at 512 MiB: a `cargo` `target/` (GBs)
    or a Sphinx `docs/_build/` is enough to fail `ray.init` outright with "Package size
    exceeds the maximum size of 512.00MiB" — before any work starts, from a *distributed
    query*, for a reason that has nothing to do with the query. Excluding build output from
    an upload that exists to ship *source* is right regardless, and it is the same
    defence-in-depth as pinning `pip` off above: neutralize what the platform injects.

    Returns `{"pip": None}` (neutralize the injected pip, ship nothing) when
    `distributed.trust_cluster_image` is set — a production image that bakes a matching
    batcher into every node and wants to skip the upload.
    """
    if active_config().distributed.trust_cluster_image:
        return {"pip": None, "excludes": list(_BUILD_ARTIFACT_EXCLUDES)}
    return {
        "py_modules": [package_dir()],
        "pip": None,
        "excludes": list(_BUILD_ARTIFACT_EXCLUDES),
    }


@contextlib.contextmanager
def _platform_env_hook_disabled():
    """Run Batcher's `ray.init` with the platform's runtime-env hook out of the way.

    Ray applies `RAY_RUNTIME_ENV_HOOK` to the `runtime_env` *after* the caller builds it, so
    a managed platform's hook silently rewrites the env `_self_ship_runtime_env` just
    constructed. Two of those rewrites are fatal, and that function's defences cannot stop
    them precisely because they run downstream of it:

    - **`pip`.** We pin `pip: None` — ship our package, trust the cluster image. Ray's
      `RuntimeEnv` drops falsey values, so the hook sees no `pip` key and substitutes the
      workspace's tracked requirements. When that list names the local editable project
      itself (`batcher-engine`, resolvable on no index), *every* worker's runtime-env build
      fails and no task in the job can start.
    - **`working_dir`.** We set `excludes` for build output, but the hook zips the project
      directory itself before Ray ever applies them. A `cargo` `target/` (gigabytes) then
      hangs `ray.init` or blows Ray's 512 MiB package cap — from a distributed query, for a
      reason that has nothing to do with the query.

    Both were live on the cluster this was written against: dist mode could not run a single
    Ray task. Batcher fully determines what its workers need (its own package, via
    `py_modules`), so the hook has nothing to add here and a demonstrated ability to break
    the job. Disabling it is scoped to our `ray.init` call and restored immediately after, so
    any other Ray user in the process still gets the platform's behavior. An unimportable
    hook — Batcher running *outside* the runtime that exported it — is handled by the same
    removal, which is why this subsumes the narrower "drop a hook whose module is missing".
    """
    import os

    saved = {v: os.environ.pop(v, None) for v in ("RAY_RUNTIME_ENV_HOOK",)}
    try:
        yield
    finally:
        for var, value in saved.items():
            if value is not None:
                os.environ[var] = value


# Once we've decided whether the job ships batcher (on the first `_ensure_ray`), the
# answer is fixed for the process — a later `_ensure_ray` must not flip it just because
# Ray now reports initialized.
_ship_decided = False


def _ensure_ray(workers: int) -> None:
    global _ship_decided
    import ray

    if not ray.is_initialized():
        with _platform_env_hook_disabled():
            try:
                ray.init(**_ray_init_kwargs(workers))
            except ValueError as e:
                # A cluster is already running but no address was configured (a managed
                # workspace that doesn't export `RAY_ADDRESS`): Ray auto-attaches and then
                # rejects the local-only `num_cpus`/object-store hints. Retry as a plain
                # attach so the distributed path works on the cluster instead of failing.
                if "existing cluster" not in str(e):
                    raise
                ray.init(**_ray_init_kwargs(workers, force_attach=True))
            except ConnectionError:
                # A managed-cluster signal routed us to `address="auto"` but no cluster is
                # actually reachable (e.g. a local dev run inside a workspace whose cluster
                # is down): start a local single-node Ray instead of failing the job.
                if not ray.is_initialized():
                    ray.init(**_ray_init_kwargs(workers, force_local=True))
        # Batcher initialized Ray: a local cluster shares the driver's modules and a
        # remote one carries the self-shipped runtime_env, so the job makes batcher
        # importable — no per-remote shipping needed.
        set_job_ships_batcher(True)
        _ship_decided = True
    elif not _ship_decided:
        # A foreign `ray.init` ran before batcher (e.g. the user attached to the
        # cluster themselves): batcher couldn't set the job runtime_env, so it must
        # ship its package on each remote instead. A no-op when trust_cluster_image.
        set_job_ships_batcher(False)
        _ship_decided = True
    _wrap_tasks(ray, task_options(current_envelope()))


def resolve_transport(transport: str, workers: int) -> str:
    """Resolve `transport == "auto"` to a concrete shuffle transport.

    Flight (Carbonite) on a genuine multi-node cluster — the disk shuffle writes to
    a driver-local `work_dir` that worker nodes can't reach, so disk is correct only
    on a single node or a configured shared filesystem. Explicit `"flight"`/`"disk"`
    pass through unchanged.
    """
    if transport != "auto":
        return transport
    if active_config().distributed.shared_filesystem:
        return "disk"
    _ensure_ray(workers)
    return "flight" if cluster_topology()["nodes"] > 1 else "disk"


def _wrap_tasks(ray, resources: dict) -> None:
    """(Re)wrap the module task fns as `ray.remote(**resources, **fault_kwargs)`.

    Each remote also carries its kind's fault-tolerance budget (config-driven): task
    functions get task retries (`max_retries`/`retry_exceptions`), actor classes get
    restart/retry (`max_restarts`/`max_task_retries`). Idempotent per signature: the
    unwrapped originals are cached on first sight, and tasks are re-wrapped only when
    the resource grant *or* the fault config changes — so successive queries with
    different envelopes each get correctly-resourced, correctly-resilient tasks."""
    global _wrapped_resources
    import importlib
    import inspect

    task_fault = fault_options()
    actor_fault = actor_fault_options()
    signature = (
        tuple(sorted(resources.items())),
        tuple(sorted(task_fault.items())),
        tuple(sorted(actor_fault.items())),
    )
    with _wrap_lock:
        if _originals and signature == _wrapped_resources:
            return
        for mod_name, fn_names in _TASK_FUNCS.items():
            module = importlib.import_module(mod_name)
            for fn_name in fn_names:
                key = (mod_name, fn_name)
                original = _originals.get(key)
                if original is None:
                    original = getattr(module, fn_name)
                    _originals[key] = original
                fault = actor_fault if inspect.isclass(original) else task_fault
                wrapper = ray.remote(**resources, **fault)(original)
                setattr(module, fn_name, wrapper)
        _wrapped_resources = signature


def _single_node(plan: LogicalPlan, sources: list[Source]) -> pa.Table:
    """Fallback: optimize + run on the multi-core single-node engine."""
    from batcher import core, kyber

    physical = kyber.optimize(plan)
    resolved = [
        read_source(src, physical.source_projections.get(i), physical.source_predicates.get(i))
        for i, src in enumerate(sources)
    ]
    batches = core.execute_local(physical, resolved)
    if not batches:
        # `Table.from_batches([], schema=None)` raises, and a null-typed guess would make an
        # empty result's schema differ from a non-empty one. The plan knows its own types.
        from batcher.dist.executors.plan_analysis import empty_result_table

        return empty_result_table(plan, plan.available_columns())
    return pa.Table.from_batches(batches, schema=batches[0].schema)


def _rmtree(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)
