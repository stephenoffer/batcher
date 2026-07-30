"""What the *engine process* detected about its CPU, as opposed to what this interpreter sees.

Every other probe in this package reads `/proc` and `/sys` from the Python process. That is
usually the same machine the data plane runs on, and "usually" is doing real work in that
sentence. Two cases break it, and both are ordinary:

* **A cgroup applied after the interpreter started.** A Ray worker's CPU quota lands on the
  actor once it is placed, which is after `import batcher`. Anything memoized before that
  describes a machine the engine is no longer running on.
* **A heterogeneous cluster.** The driver plans against its own cache sizes and vector width
  while the work executes on workers that have neither.

So the engine detects its own hardware locally (`bc_arrow::topology`, `bc_arrow::isa`) and this
module reads those numbers back. When the two disagree, the engine's answer is the one that
governed how the data plane actually sized itself, and therefore the one worth acting on.

Every function degrades to an empty answer when the engine is not built, or when it is built
from a revision that predates these entry points. That second case is not hypothetical — an
installed extension module routinely lags the source tree — so callers get "I don't know"
rather than an `AttributeError`.
"""

from __future__ import annotations

import functools
from typing import Any

from batcher._internal.native import engine_or_none

__all__ = [
    "call_engine",
    "engine_hardware",
    "engine_numa_map",
    "engine_pinning_order",
]


def call_engine(name: str, *args: Any) -> Any:
    """Call `name` on the compiled engine, or return `None` when it cannot be reached.

    Covers both "not built" and "built from an older revision", which is why the attribute is
    looked up rather than accessed: an installed `.so` frequently lags the source tree, and an
    `AttributeError` out of a hardware probe would be a crash in place of a shrug.

    Args:
        name: The engine entry point to call.
        *args: Positional arguments forwarded to it.

    Returns:
        Whatever the entry point returned, or `None` when it could not be called.
    """
    eng = engine_or_none()
    if eng is None:
        return None
    fn = getattr(eng, name, None)
    if fn is None:
        return None
    try:
        return fn(*args)
    except Exception:
        # A hardware probe is advisory by construction. Nothing here is worth failing a
        # query over, and every caller has a defined answer for "unknown".
        return None


@functools.lru_cache(maxsize=1)
def engine_hardware() -> dict[str, Any]:
    """The CPU topology and ISA the engine detected in its own process.

    Keys, when the engine can answer: `logical_cores`, `physical_cores`, `smt_width`,
    `has_smt`, `numa_nodes`, `is_numa`, `compute_threads`, `l1d_bytes`, `l2_bytes`,
    `l3_bytes`, `cache_line`, `isa_tier`, `isa_features`, `vector_bytes`, and
    `avx512_is_cheap`.

    `compute_threads` is the physical core count, which is the right fan-out for work that
    saturates execution units rather than stalling on memory; `logical_cores` is the right one
    for memory-bound work, where the SMT sibling is what hides the stall.

    Returns:
        The engine's hardware facts, or an empty dict when the engine cannot report them.
    """
    return call_engine("engine_hardware") or {}


@functools.lru_cache(maxsize=1)
def engine_pinning_order() -> tuple[int, ...]:
    """The CPU ids the engine pins worker threads to, in worker order.

    Position `i` is where worker `i` lands. The order fills distinct physical cores before any
    SMT sibling and strides across NUMA nodes, and it only ever names CPUs in this process's
    affinity mask.

    Empty means the engine will not pin — either the topology is unreadable or pinning is off.
    A placement decision made up here can be checked against it instead of guessing what the
    data plane will do.

    Returns:
        CPU ids in worker order, empty when the engine will not pin.
    """
    return tuple(call_engine("engine_pinning_order") or ())


@functools.lru_cache(maxsize=1)
def engine_numa_map() -> dict[int, tuple[int, ...]]:
    """NUMA node id to the usable CPU ids on it, as the engine sees them.

    Empty when NUMA is not exposed, which callers read as "one node" — and correctly so for a
    process pinned to a single socket, whose memory is all local no matter how many nodes the
    host has.

    Returns:
        Node id to its usable CPU ids, empty when NUMA is not exposed.
    """
    pairs = call_engine("engine_numa_map") or []
    return {node: tuple(cpus) for node, cpus in pairs}
