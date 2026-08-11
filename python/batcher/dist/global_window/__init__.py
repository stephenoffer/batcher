"""The *global* (no-``PARTITION BY``) ordered window, in bounded memory and across a cluster.

Every other window shape splits by its partition keys. A global window has one partition over
every row, so it splits by its *order* instead: range-partition into buckets that are ordered
relative to each other, window each bucket independently, then shift each bucket's result by
the prior buckets' contribution.

`offsets` owns that algebra and the set of functions it covers. `stream` schedules it one
bucket at a time on one node (bounded memory); `flight` and `disk` schedule every bucket at
once across Ray workers, over an Arrow Flight shuffle and driver-local IPC files respectively
-- the same two transports the distributed sort offers, so the route never depends on which
one the cluster picked. One algebra, three schedules, which is what makes the distributed
result the single-node result.

**The three schedules load lazily; only `offsets` is eager.** `dist.executor` imports
`supports_ordered_bucket_offsets` at module scope precisely because the predicate sees
nothing but `plan` -- but importing a submodule runs this file first, and eagerly naming
`flight` here reached `flight_aggregate` -> `flight_worker`, whose `import ray` is at module
scope for its `@ray.remote` actors. That put a **0.44 s `import ray` on every local
`collect()`**, which `tests/unit/test_cold_start_imports.py` exists to catch: a local
relational query must not drag the cluster runtime into the process. The `__getattr__` below
keeps the public names exactly where they were and defers the cost to the first caller that
actually schedules a distributed bucket.
"""

from __future__ import annotations

from typing import Any

from batcher.dist.global_window.offsets import supports_ordered_bucket_offsets

#: Public name -> the submodule that defines it, resolved on first access.
_LAZY = {
    "execute_global_window_disk": "disk",
    "execute_global_window_flight": "flight",
    "stream_spilling_global_window": "stream",
}

__all__ = [
    "execute_global_window_disk",
    "execute_global_window_flight",
    "stream_spilling_global_window",
    "supports_ordered_bucket_offsets",
]


def __getattr__(name: str) -> Any:
    """Import the schedule that defines `name` on first access (PEP 562)."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value  # cache, so the indirection is paid once
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
