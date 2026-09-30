"""The memo itself: exact-key entries, their learning rounds, lookup, store, clear.

See the package docstring for what a key captures. Layer: kyber (optimizer).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from batcher.kyber.plan_cache.keys import _BUCKET_STATE, _split

__all__ = ["clear", "lookup", "misses", "store"]

# Entry: exact key -> (result, keepalive, deps, (fingerprint rounds, dependency rounds), learned
# fingerprint). `keepalive` pins
# the source objects whose `id()` the key used, so the ids cannot be recycled while the entry
# lives (see the module docs). `rounds` counts how many times this key was re-planned because
# what it learned from moved (see `cache_key` and `lookup`).
_CACHE: OrderedDict[str, tuple[Any, tuple, Any, tuple[int, int], str]] = OrderedDict()

# Rounds a key has been re-planned, carried from the lookup that dropped its entry to the store
# that replaces it. Bounded by the entries dropped since their re-plan was stored, which is at
# most the number of queries planning at once.
_ROUNDS: dict[str, tuple[int, int]] = {}

# The memo is process-global and `execution.max_concurrent_queries` lets several queries plan
# at once, so its LRU bookkeeping is shared mutable state. The individual dict operations are
# each atomic, but the *pairs* are not: `lookup` reads an entry and then reorders it, and an
# eviction landing between the two raises `KeyError` out of `move_to_end` — into planning,
# where nothing is catching it, on a path whose entire job is to be a transparent speed-up.
# `store`'s eviction loop has the same shape against `popitem`. Every critical section here is
# a handful of O(1) dict operations, so one lock costs nothing measurable next to the
# optimization it is memoizing.
_LOCK = threading.Lock()

# Keyed lookups that found no entry to serve, ever: a plan was derived rather than replayed.
# Read by a caller that needs to know whether a run it timed paid for planning (see `misses`).
_MISSES = 0


def misses() -> int:
    """How many keyed lookups have returned no plan, a monotonic count for this process.

    A run during which this does not move served every plan it asked for from the memo, so its
    wall time is the query's steady state rather than one that paid for derivation and
    learning. The adaptive-route bandit (`api.adaptive.gating.record_adaptive_route`) records
    only such runs. A lookup with no key is not counted: a query that can never be memoized
    pays for its planning on every run, so that cost is its steady state.
    """
    return _MISSES


def clear() -> None:
    """Drop every cached plan. For tests and for a hub reset.

    The bucket deadband goes with them: it is state *about* the keys, so leaving it behind
    would let one test's coefficients hold another's bucket.
    """
    with _LOCK:
        _CACHE.clear()
        _BUCKET_STATE.clear()
        _ROUNDS.clear()


def lookup(key: str | None, holds: Callable[[Any, int], bool] | None = None) -> Any | None:
    """The cached optimizer result for `key`, or `None`. Refreshes its LRU position.

    `holds` re-validates the entry's stored dependencies (see `optimizer.plan_deps`), given
    them and how many times this key has already been re-planned: an entry whose dependencies
    no longer hold is dropped and reported as a miss, so the caller re-plans against the
    measurements it would otherwise have ignored. A changed *learned* fingerprint (see
    `cache_key`) re-plans the same way, but only while the key is within its learning rounds;
    past them the entry is served, a semantically correct plan built from slightly older
    numbers, and only its own dependencies moving can still replace it.
    """
    global _MISSES
    if key is None:
        return None
    with _LOCK:
        exact, learned_fp = _split(key)
        entry = _CACHE.get(exact)
        if entry is None:
            _MISSES += 1
            return None
        _, _, deps, (fp_rounds, dep_rounds), stored_fp = entry
        from batcher.kyber.optimizer.plan_deps import FINGERPRINT_ROUNDS

        # The two causes are rationed separately. Counted together, a round spent re-planning
        # for another query's learning (a fingerprint change) left none for this plan's *own*
        # first measurements, which then arrived and were ignored for good.
        relearn = stored_fp != learned_fp and fp_rounds < FINGERPRINT_ROUNDS
        moved = holds is not None and not holds(deps, dep_rounds)
        if relearn or moved:
            del _CACHE[exact]
            _ROUNDS[exact] = (fp_rounds + relearn, dep_rounds + moved)
            _MISSES += 1
            return None
        _CACHE.move_to_end(exact)
        return entry[0]


def store(
    key: str | None, result: Any, sources: list | None, max_entries: int, deps: Any = ()
) -> None:
    """Cache `result` under `key`, evicting the least recently used entry past the cap.

    `deps` is what `lookup`'s `holds` will be asked about; the default holds for any check.
    """
    if key is None or max_entries <= 0:
        return
    # The keepalive pins the sources whose `id()` the key used, so an address cannot be
    # recycled underneath a live entry. A derivation-keyed source is named by *how it was
    # derived* rather than by its address, so pinning it would buy nothing and cost the whole
    # materialized intermediate staying resident until the entry is evicted.
    keepalive = tuple(s for s in (sources or ()) if not getattr(s, "derivation", None))
    exact, learned_fp = _split(key)
    with _LOCK:
        _CACHE[exact] = (result, keepalive, deps, _ROUNDS.pop(exact, (0, 0)), learned_fp)
        _CACHE.move_to_end(exact)
        while len(_CACHE) > max_entries:
            _CACHE.popitem(last=False)
