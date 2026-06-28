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

import json
import shutil
import threading

import pyarrow as pa

from batcher.config import active_config
from batcher.io.source import Source, read_source
from batcher.plan.logical import LogicalPlan

from .policies import actor_fault_options, fault_options
from .scaling import cluster_topology
from .scheduling import current_envelope, task_options


def engine_config_json() -> str:
    """The driver's active `EngineConfig` (morsel size, parallelism) as JSON, to
    ship into remote tasks.

    A Ray worker's `active_config()` sees only that process's own default (the
    driver's `config_context` does not cross the process boundary), so the driver
    must capture this here and pass it as a task argument to every worker-side
    `execute_plan` — otherwise distributed runs silently ignore the session config.

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
    if env is None or env.memory_bytes <= 0:
        return base
    cfg = json.loads(base)
    existing = int(cfg.get("memory_budget_bytes", 0) or 0)
    cfg["memory_budget_bytes"] = (
        env.memory_bytes if existing <= 0 else min(existing, env.memory_bytes)
    )
    return json.dumps(cfg)


# Names of the module-level Ray task functions, keyed by the module they live in.
# `_ensure_ray` rebinds each `<module>.<name> = ray.remote(<module>.<name>)`.
_TASK_FUNCS: dict[str, tuple[str, ...]] = {
    "batcher.dist.executors.map": ("_map_udf_task", "_MapActor"),
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
}

# Unwrapped originals (so re-wrapping with a new grant never double-wraps) and the
# resource signature the task fns are currently wrapped with.
_originals: dict[tuple[str, str], object] = {}
_wrapped_resources: tuple | None = None
# Guards the module-global rebind in `_wrap_tasks`: two concurrent distributed
# queries with different envelopes must not interleave their re-wraps and hand one
# query's tasks the other's resource grant (N12).
_wrap_lock = threading.Lock()


def _ray_init_kwargs(workers: int) -> dict:
    """`ray.init(**kwargs)` for the active config — attach to a cluster or spin local.

    Attach to a *running* cluster when an address is configured (`Config` or the
    `RAY_ADDRESS` env var Ray itself honors), shipping `batcher` + its native
    extension via `runtime_env` so workers can run the data plane. Only when no
    address is given do we start a *local* cluster capped at `workers` CPUs (the
    single-node / test path); against a real cluster we leave fan-out to the
    scheduler/autoscaler and never pin `num_cpus`."""
    import os

    dc = active_config().distributed
    kwargs: dict = {
        "include_dashboard": dc.dashboard,
        "logging_level": "ERROR",
        "ignore_reinit_error": True,
        "namespace": dc.namespace,
    }
    if dc.ray_address:
        kwargs["address"] = dc.ray_address
    elif os.environ.get("RAY_ADDRESS"):
        kwargs["address"] = "auto"
    else:
        kwargs["num_cpus"] = workers
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


def _self_ship_runtime_env() -> dict | None:
    """A Ray `runtime_env` that uploads the batcher package to workers, or `None`.

    Returns `{"py_modules": [<batcher pkg dir>]}` when batcher is imported from a
    source/editable tree (the dev install, or a freshly-built wheel laid out in the
    repo) — the worker image then can't be assumed to carry it, so the package dir is
    uploaded via Ray's job-level `py_modules` (cached in the object store, abi3 native
    extension included). Returns `None` for a normal site-/dist-packages install: the
    cluster image already provides batcher, so shipping it would just churn an upload.
    """
    import os

    import batcher

    pkg = os.path.dirname(os.path.abspath(batcher.__file__))
    installed = f"{os.sep}site-packages{os.sep}" in pkg or f"{os.sep}dist-packages{os.sep}" in pkg
    return None if installed else {"py_modules": [pkg]}


def _neutralize_broken_runtime_env_hook() -> None:
    """Drop a `RAY_RUNTIME_ENV_HOOK`/`RAY_RUNTIME_ENV_PLUGINS` whose module is missing.

    A deployment env may export a runtime-env hook (e.g. Anyscale's
    `cgroup_runtime_plugin`) that Ray imports during `ray.init`. When Batcher runs
    *outside* that runtime the module is absent and `ray.init` raises
    `ModuleNotFoundError` before any work starts. A hook pointing at an
    unimportable module is broken regardless of context, so removing it is strictly
    safer than crashing — and a no-op where the module is present (a real cluster)."""
    import importlib.util
    import os

    for var in ("RAY_RUNTIME_ENV_HOOK", "RAY_RUNTIME_ENV_PLUGINS"):
        value = os.environ.get(var)
        if not value:
            continue
        # The leading dotted path before the first `.` / `[` names the module to import
        # (`mod._hook`, or `[{"class": "mod.Plugin"}]` JSON). Probe just the base module.
        head = value.lstrip("[{\"' ").split(".")[0].split("[")[0]
        if head and importlib.util.find_spec(head) is None:
            os.environ.pop(var, None)


def _ensure_ray(workers: int) -> None:
    import ray

    if not ray.is_initialized():
        _neutralize_broken_runtime_env_hook()
        ray.init(**_ray_init_kwargs(workers))
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
    schema = batches[0].schema if batches else None
    return pa.Table.from_batches(batches, schema=schema)


def _rmtree(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)
