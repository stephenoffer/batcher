"""The memo itself: exact-key entries, their learning rounds, lookup, store, clear.

See the package docstring for what a key captures. Layer: kyber (optimizer).

## The regret guard

A re-plan is the learning loop acting on what it measured, and nothing promised that acting on
it would help. Correcting one estimate can expose another, and the cost model prices a tree it
has never seen run: on JOB q5a the first plan ran in 43 ms, the plan chosen after two runs of
learning ran in 230 ms, and the memo then served that plan on every run that followed --
learning that made the query five times slower and stayed. So the memo keeps the plan a
re-plan displaced, together with the best execution time measured for it
(`record_outcome`, fed by the conductor), and when the replacement's first measured execution
is clearly slower it puts the displaced plan back and **pins** the key: the plan that measured
better is served from then on, and later learning no longer re-plans it. This is the
plan-regression correction SQL Server ships as `FORCE_LAST_GOOD_PLAN`, and it makes the loop
monotone in what was measured rather than in what was estimated.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from batcher.kyber.plan_cache.keys import _BUCKET_STATE, _BUCKETED, _split

__all__ = ["clear", "lookup", "misses", "record_outcome", "served", "store"]

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

# The regret guard's state (see the module notes), all keyed by exact key and pruned with it.
_BEST_MS: dict[str, float] = {}  # best measured execution of the plan now cached
_DISPLACED: dict[str, tuple[tuple, float]] = {}  # the entry a re-plan replaced, and its best
_PINNED: set[str] = set()  # keys whose re-plan lost to the plan it replaced
# `id(result) -> (result, exact key)` for the results handed out recently, so an execution
# time reported against a result finds its key. The strong reference keeps the id unique.
_SERVED: OrderedDict[int, tuple[Any, str]] = OrderedDict()
_SERVED_MAX = 512
# How much slower a replacement's measured execution must be than the plan it displaced before
# the displaced plan is restored, as a ratio and as an absolute floor: a millisecond of jitter
# on a millisecond query is not evidence.
_REGRESSION_RATIO = 1.25
_REGRESSION_MIN_MS = 2.0
# Re-plans a key may take before it keeps the plan it has (`lookup`), and how many it took.
_MAX_REPLANS = 3
_REPLANS: dict[str, int] = {}


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
        _BUCKETED.clear()
        _ROUNDS.clear()
        _BEST_MS.clear()
        _DISPLACED.clear()
        _PINNED.clear()
        _SERVED.clear()
        _REPLANS.clear()


def lookup(key: str | None, holds: Callable[[Any, int], bool] | None = None) -> Any | None:
    """The cached optimizer result for `key`, or `None`. Refreshes its LRU position.

    `holds` re-validates the entry's stored dependencies (see `optimizer.plan_deps`), given
    them and how many times this key has already been re-planned: an entry whose dependencies
    no longer hold is dropped and reported as a miss, so the caller re-plans against the
    measurements it would otherwise have ignored. A changed *learned* fingerprint (see
    `cache_key`) re-plans the same way, but only while the key is within its learning rounds;
    past them the entry is served, a semantically correct plan built from slightly older
    numbers, and only its own dependencies moving can still replace it. A key the regret guard
    pinned (see the module notes) is served regardless, and so is a key that has already been
    re-planned `_MAX_REPLANS` times.

    That cap is what stops a key re-planning on every run. A dependency that moves by a full
    bucket invalidates whatever its rounds, and on a many-way join each new plan runs operators
    nobody had measured, whose measurements then move the next plan's dependencies: JOB q29b
    chose a new plan on every one of seven runs and paid ~350 ms of planning each time, against
    ~55 ms of execution. Three re-plans spend the first runs' measurements, which is what the
    rounds exist for; past them the plan stands, and the regret guard has already reverted any
    of those re-plans that measured slower than the plan before it.
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
        if (relearn or moved) and exact not in _PINNED:
            replans = _REPLANS.get(exact, 0) + 1
            if replans > _MAX_REPLANS:
                _PINNED.add(exact)
                _CACHE.move_to_end(exact)
                return entry[0]
            _REPLANS[exact] = replans
            del _CACHE[exact]
            if exact in _BEST_MS:
                # Kept so the replacement can be held to what this plan measured.
                _DISPLACED[exact] = (entry, _BEST_MS.pop(exact))
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
            evicted, _ = _CACHE.popitem(last=False)
            _BEST_MS.pop(evicted, None)
            _DISPLACED.pop(evicted, None)
            _PINNED.discard(evicted)
            _REPLANS.pop(evicted, None)


def served(result: Any, key: str | None) -> None:
    """Remember that `result` was handed out for `key`, so its execution can be reported.

    Args:
        result: The object the caller will report an execution time against.
        key: The key it was looked up or stored under.
    """
    if key is None:
        return
    exact, _ = _split(key)
    with _LOCK:
        _SERVED[id(result)] = (result, exact)
        _SERVED.move_to_end(id(result))
        while len(_SERVED) > _SERVED_MAX:
            _SERVED.popitem(last=False)


def record_outcome(result: Any, elapsed_ms: float) -> None:
    """Report how long executing a served plan took; restore a displaced plan that ran faster.

    The first measured execution of a plan that displaced another is compared with the best
    the displaced plan measured. When it is slower by more than `_REGRESSION_RATIO` (and by
    more than `_REGRESSION_MIN_MS`), the displaced plan goes back into the memo and the key is
    pinned to it. Otherwise the replacement stands and the displaced plan is forgotten.

    Args:
        result: The plan object `served` was told about.
        elapsed_ms: Its execution time, excluding planning.
    """
    with _LOCK:
        hit = _SERVED.get(id(result))
        if hit is None:
            return
        exact = hit[1]
        best = _BEST_MS.get(exact)
        _BEST_MS[exact] = elapsed_ms if best is None else min(best, elapsed_ms)
        displaced = _DISPLACED.pop(exact, None)
        if displaced is None:
            return
        entry, prior_ms = displaced
        slower = elapsed_ms > prior_ms * _REGRESSION_RATIO
        if slower and elapsed_ms - prior_ms > _REGRESSION_MIN_MS:
            _CACHE[exact] = entry
            _CACHE.move_to_end(exact)
            _BEST_MS[exact] = prior_ms
            _PINNED.add(exact)
