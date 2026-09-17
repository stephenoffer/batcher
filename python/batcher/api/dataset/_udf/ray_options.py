"""Resolve the Ray Data resource parameters of the UDF verbs onto what the scheduler honours.

`map`, `flat_map`, `map_batches` and the callable form of `filter` take Ray Data's whole
resource parameter set, so a pipeline ported from Ray keeps its call sites. Each parameter
lands on a field the map scheduler already acts on (`num_gpus`, `concurrency`,
`accelerator_type`, the custom `resources`), or it is refused with a `PlanError` naming it.
Nothing in between: a resource request that is accepted and then not applied reads as a
scheduled job and behaves as an unscheduled one, which is the failure this module exists to
rule out.

What is refused, and why, is stated once here so every verb says the same thing:

- ``num_cpus`` and ``memory``: a CPU map task's share is sized from its own partition by the
  scheduler, and an accelerator stage reserves no CPU at all, so a fixed per-worker request
  has no field to land on.
- ``ray_remote_args_fn``: per-task options are fixed when the stage is planned.
- ``ray_remote_args`` keys other than ``num_gpus``, ``resources`` and ``accelerator_type``.
- a ``concurrency`` whose initial pool size is above its minimum: the autoscaling pool starts
  at the minimum, which is also Ray Data's default initial size.
- a ``compute`` strategy that asks for something the pool cannot express (a task-pool size, a
  per-actor in-flight cap, an unbounded actor pool).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from batcher._internal.errors import PlanError

__all__ = ["Placement", "normalize_concurrency", "resolve_placement"]

#: The Ray Data spellings of a resource request, as they appear in a `PlanError`.
_NO_CPU_FIELD = (
    "the map scheduler sizes a CPU task's share from its own partition and an accelerator "
    "stage reserves no CPU, so a fixed per-worker request has no field to land on. Use "
    "num_workers= for in-worker parallelism"
)
_NO_MEMORY_FIELD = (
    "a map stage has no per-worker memory reservation. Declare a model's footprint with "
    "map_batches(model_memory_gb=...), which budgets host RAM per worker"
)
#: `ray_remote_args` keys that map onto a `MapBatches` field.
_REMOTE_KEYS = ("num_gpus", "resources", "accelerator_type")


@dataclass(frozen=True, slots=True)
class Placement:
    """The scheduling request a UDF verb resolves to, in the fields `MapBatches` carries."""

    num_gpus: float
    concurrency: int | tuple[int, int] | None
    accelerator_type: str | None
    resources: dict[str, float] | None


def _refuse(verb: str, param: str, value: object, why: str) -> PlanError:
    """The one message shape for a parameter the scheduler cannot honour."""
    return PlanError(f"{verb}({param}={value!r}) cannot be honoured: {why}.")


def normalize_concurrency(verb: str, concurrency: object) -> int | tuple[int, int] | None:
    """Validate a pool size: an int, ``(min, max)``, or Ray Data's ``(min, max, initial)``.

    Args:
        verb: The method name, for the message.
        concurrency: The caller's value.

    Returns:
        The size in the form the scheduler reads: `None`, an int, or ``(min, max)``.

    Raises:
        PlanError: If the value is malformed, or a three-tuple names an initial size the pool
            cannot promise.
    """
    if concurrency is None:
        return None
    if isinstance(concurrency, bool) or not isinstance(concurrency, int | tuple):
        raise PlanError(
            f"{verb}(concurrency=...) must be a positive int, (min, max) or (min, max, "
            f"initial), got {concurrency!r}"
        )
    if isinstance(concurrency, int):
        if concurrency <= 0:
            raise PlanError(f"{verb}(concurrency=...) must be positive, got {concurrency}")
        return concurrency
    if len(concurrency) not in (2, 3) or not all(
        isinstance(v, int) and not isinstance(v, bool) for v in concurrency
    ):
        raise PlanError(
            f"{verb}(concurrency=...) tuple must be (min, max) or (min, max, initial) of "
            f"ints, got {concurrency!r}"
        )
    lo, hi = concurrency[0], concurrency[1]
    if not 0 < lo <= hi:
        raise PlanError(
            f"{verb}(concurrency=...) tuple must satisfy 0 < min <= max, got {concurrency!r}"
        )
    if len(concurrency) == 2:
        return (lo, hi)
    initial = concurrency[2]
    if not lo <= initial <= hi:
        raise PlanError(
            f"{verb}(concurrency=...) initial size must lie in [min, max], got {concurrency!r}"
        )
    if initial != lo:
        raise _refuse(
            verb,
            "concurrency",
            concurrency,
            "the autoscaling pool starts at min and grows with queued work, so an initial "
            "size above min cannot be promised. Pass (min, max), or raise min",
        )
    return lo if lo == hi else (lo, hi)


def _merge(verb: str, name: str, explicit: Any, default: Any, remote: Any) -> Any:
    """One value from an explicit argument and its `ray_remote_args` twin, refusing a clash."""
    if explicit != default and explicit != remote:
        raise PlanError(
            f"{verb}() got {name}={explicit!r} and ray_remote_args[{name!r}]={remote!r}; "
            "pass it once"
        )
    return remote


def _from_remote_args(
    verb: str,
    remote: object,
    num_gpus: float,
    accelerator_type: str | None,
    resources: dict[str, float] | None,
) -> tuple[float, str | None, dict[str, float] | None]:
    """Fold a `ray_remote_args` dict into the explicit request, refusing what cannot land."""
    if remote is None:
        return num_gpus, accelerator_type, resources
    if not isinstance(remote, dict):
        raise PlanError(
            f"{verb}(ray_remote_args=...) must be a dict of Ray options, got "
            f"{type(remote).__name__}"
        )
    for key, value in remote.items():
        if key == "num_cpus":
            raise _refuse(verb, "ray_remote_args['num_cpus']", value, _NO_CPU_FIELD)
        if key == "memory":
            raise _refuse(verb, "ray_remote_args['memory']", value, _NO_MEMORY_FIELD)
        if key not in _REMOTE_KEYS:
            raise _refuse(
                verb,
                f"ray_remote_args[{key!r}]",
                value,
                f"only {list(_REMOTE_KEYS)} reach the map scheduler",
            )
    if "num_gpus" in remote:
        num_gpus = _merge(verb, "num_gpus", num_gpus, 0.0, remote["num_gpus"])
    if "accelerator_type" in remote:
        accelerator_type = _merge(
            verb, "accelerator_type", accelerator_type, None, remote["accelerator_type"]
        )
    if "resources" in remote:
        extra = remote["resources"]
        if not isinstance(extra, dict):
            raise PlanError(f"{verb}(ray_remote_args['resources']) must be a dict")
        merged = dict(resources or {})
        for name, amount in extra.items():
            _merge(verb, f"resources[{name!r}]", merged.get(name), None, amount)
            merged[name] = amount
        resources = merged
    return num_gpus, accelerator_type, resources


def _from_compute(
    verb: str, fn: object, compute: object, concurrency: int | tuple[int, int] | None
) -> int | tuple[int, int] | None:
    """Fold a Ray Data `compute` strategy (a string or strategy object) into `concurrency`."""
    if compute is None:
        return concurrency
    is_class = isinstance(fn, type)
    if hasattr(compute, "min_size") and hasattr(compute, "max_size"):  # ActorPoolStrategy
        if getattr(compute, "max_tasks_in_flight_per_actor", None) is not None:
            raise _refuse(
                verb,
                "compute",
                compute,
                "the pool sets each actor's in-flight depth from measured utilization",
            )
        if compute.max_size is None or compute.max_size == float("inf"):
            raise _refuse(
                verb, "compute", compute, "an unbounded actor pool has no size; give max_size"
            )
        initial = getattr(compute, "initial_size", None)
        spec = (compute.min_size, compute.max_size) + (() if initial is None else (initial,))
        pool = normalize_concurrency(verb, spec)
        if concurrency is not None and concurrency != pool:
            raise PlanError(f"{verb}() got both compute={compute!r} and concurrency; pass one")
        return pool
    if compute == "actors":
        if not is_class and concurrency is None:
            raise PlanError(
                f"{verb}(compute='actors') needs a pool size for a function: pass "
                "concurrency=n (a class already runs on actors)"
            )
        return concurrency
    is_tasks = compute == "tasks" or hasattr(compute, "size")  # TaskPoolStrategy
    if not is_tasks:
        raise PlanError(
            f"{verb}(compute=...) must be 'tasks', 'actors', or a Ray Data "
            f"ActorPoolStrategy/TaskPoolStrategy, got {compute!r}"
        )
    if getattr(compute, "size", None) is not None:
        raise _refuse(
            verb, "compute", compute, "the task fan-out is sized from the data, not capped"
        )
    if is_class or concurrency is not None:
        raise _refuse(
            verb,
            "compute",
            compute,
            "a class fn or an explicit concurrency runs on a long-lived actor pool, not tasks",
        )
    return None


def resolve_placement(
    verb: str,
    fn: object,
    *,
    num_cpus: float | None,
    num_gpus: float,
    memory: float | None,
    compute: object,
    concurrency: object,
    ray_remote_args: dict[str, Any] | None,
    ray_remote_args_fn: object,
    accelerator_type: str | None = None,
    resources: dict[str, float] | None = None,
) -> Placement:
    """Resolve a verb's Ray Data resource parameters to the scheduler's request.

    Args:
        verb: The method name, for messages.
        fn: The user's function or class (a class forces an actor pool).
        num_cpus: Ray's per-worker CPU request; only `None` can be honoured.
        num_gpus: GPUs per worker.
        memory: Ray's per-worker memory request; only `None` can be honoured.
        compute: A Ray Data compute strategy or ``"tasks"``/``"actors"``, or `None`.
        concurrency: An int, ``(min, max)``, or ``(min, max, initial)``.
        ray_remote_args: Extra Ray options; ``num_gpus``/``resources``/``accelerator_type``.
        ray_remote_args_fn: Ray's per-task options callback; only `None` can be honoured.
        accelerator_type: A device-model pin.
        resources: Custom Ray resources per worker.

    Returns:
        The `Placement` the stage runs with.

    Raises:
        PlanError: If any parameter cannot be honoured, naming it.
    """
    if num_cpus is not None:
        raise _refuse(verb, "num_cpus", num_cpus, _NO_CPU_FIELD)
    if memory is not None:
        raise _refuse(verb, "memory", memory, _NO_MEMORY_FIELD)
    if ray_remote_args_fn is not None:
        raise _refuse(
            verb,
            "ray_remote_args_fn",
            ray_remote_args_fn,
            "per-task Ray options are fixed when the stage is planned. Pass them statically "
            "with num_gpus=, concurrency= or ray_remote_args=",
        )
    num_gpus, accelerator_type, resources = _from_remote_args(
        verb, ray_remote_args, num_gpus, accelerator_type, resources
    )
    pool = _from_compute(verb, fn, compute, normalize_concurrency(verb, concurrency))
    return Placement(num_gpus, pool, accelerator_type, resources)
