"""Memoize the NORMALIZE phase's output, so a re-plan does not re-derive it.

The optimizer memo (`memo`) re-plans a key when what it learned moves, and a re-plan runs every
phase again from the raw plan. NORMALIZE -- constant folding, expression simplification,
canonicalization -- reads no statistics at all: over the 99 TPC-DS queries at sf0.01, every
optimization they ran, it made no estimator call, while it took 7.3 s of the optimizers' 25.8 s
(28%). Its output is therefore a pure function of the plan, the sources it is bound to, the
config and the hardware, which is exactly `cache_key(..., learned=False)`, and a re-plan of the
same plan can start from the normalized plan the first optimization produced.

That is the cost of convergence, not of a cold query: TPC-DS q64 at sf10 re-planned on each of
its second to fourth runs for ~1.3 s against a 0.3 s execution, q61 for ~0.2-0.36 s against
0.07 s. Re-plans are how measurements reach a plan, so the fix is to make them cheaper rather
than rarer.

Entries hold the bound sources alive for the reason `memo.store` does: an in-memory source is
keyed by object identity, and a freed source's recycled `id()` must not find another's entry.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from batcher.plan.logical import LogicalPlan

__all__ = ["clear", "lookup", "store"]

_MEMO: OrderedDict[str, tuple[LogicalPlan, tuple]] = OrderedDict()
_LOCK = threading.Lock()


def lookup(key: str | None) -> LogicalPlan | None:
    """The normalized plan stored under `key`, or `None`."""
    if key is None:
        return None
    with _LOCK:
        entry = _MEMO.get(key)
        if entry is None:
            return None
        _MEMO.move_to_end(key)
        return entry[0]


def store(key: str | None, plan: LogicalPlan, sources: list | None, max_entries: int) -> None:
    """Remember `plan` as `key`'s normalized form, evicting the oldest past `max_entries`."""
    if key is None or max_entries <= 0:
        return
    keepalive = tuple(s for s in (sources or ()) if not getattr(s, "derivation", None))
    with _LOCK:
        _MEMO[key] = (plan, keepalive)
        _MEMO.move_to_end(key)
        while len(_MEMO) > max_entries:
            _MEMO.popitem(last=False)


def clear() -> None:
    """Drop every entry. Called by `plan_cache.clear`."""
    with _LOCK:
        _MEMO.clear()
