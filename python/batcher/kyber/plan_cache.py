"""Memoize the optimizer — the same query, planned once.

Optimization is a pure function of `(logical plan, bound sources, config, learned stats)`.
It is also, on a join-heavy query, the single most expensive thing Batcher does: TPC-H Q8
spends 63 ms in Kyber against 40 ms in the engine and 22 ms for DuckDB's entire query. A
BI dashboard, a scheduled report, and a benchmark harness all re-issue the identical
statement; re-deriving the identical plan each time is pure waste. Every serious engine
caches plans (Spark, Presto, Snowflake); Batcher did not.

**The key is exact, not structural.** `kyber.signature.plan_signature` deliberately
*normalizes literals* so learned statistics generalize across `x > 5` and `x > 6` — which
makes it lethal as a cache key. This module keys on the plan's lowered IR verbatim, so two
queries share an entry only when they would lower to the same bytes.

Three more things go into the key, each because it can change the plan Kyber chooses:

* **the bound sources**, by data-stable identity. A file source identifies by path; an
  in-memory source's `identity()` is only shape-based (schema + row count), so two
  different relations collide on it. That collision is not merely suboptimal — Kyber's
  zone-map pruning folds a filter to `FALSE` from a source's `min`/`max`, so a plan built
  for one relation could return the *wrong answer* for another. In-memory sources are
  therefore keyed by object identity, and the entry pins them alive so a freed `id()`
  cannot be reused underneath it.
* **the optimizer config**, which decides selectivity constants, cost weights, and which
  rules run at all.
* **the learned statistics**, by the `kyber.learning.generation` counter rather than by
  content. Fingerprinting the content does not work: the feedback loop rewrites the stats
  after *every* execution — the exponential average keeps drifting and the q-error history
  keeps growing — so a content hash never repeats and the cache never hits (measured: 0
  hits in 8 identical runs). The generation instead advances only when the loop learns
  something a plan could turn on: a column measured for the first time, or a cardinality
  that corrected its prior by more than 10%. That is the same judgement the adaptive
  executor makes — re-optimize when reality disagreed with the estimate, not because a
  smoothed average moved in its fourth decimal. The `MetadataHub` itself is keyed by object
  identity, so resetting it invalidates every entry.

A hit returns exactly what a miss computes. Correctness does not depend on the key being
*complete* in the "captures every input" sense — an over-broad key would return a plan
optimized for slightly different statistics, which is still a **semantically correct** plan
(that is what the optimizer's differential tests guarantee), merely a possibly worse one. It
depends on the key capturing everything that can change a plan's *meaning*: the plan itself
and the data its pruning decisions read. Both are keyed exactly.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any

from batcher.config import Config
from batcher.kyber import learning

__all__ = ["cache_key", "clear", "lookup", "record_write", "store"]

# Entry: key -> (result, keepalive). `keepalive` pins the source objects whose `id()` the
# key used, so the ids cannot be recycled while the entry lives (see the module docs).
_CACHE: OrderedDict[str, tuple[Any, tuple]] = OrderedDict()


def clear() -> None:
    """Drop every cached plan. For tests and for a hub reset."""
    _CACHE.clear()


def lookup(key: str | None) -> Any | None:
    """The cached optimizer result for `key`, or `None`. Refreshes its LRU position."""
    if key is None:
        return None
    entry = _CACHE.get(key)
    if entry is None:
        return None
    _CACHE.move_to_end(key)
    return entry[0]


def store(key: str | None, result: Any, sources: list | None, max_entries: int) -> None:
    """Cache `result` under `key`, evicting the least recently used entry past the cap."""
    if key is None or max_entries <= 0:
        return
    _CACHE[key] = (result, tuple(sources or ()))
    _CACHE.move_to_end(key)
    while len(_CACHE) > max_entries:
        _CACHE.popitem(last=False)


def cache_key(
    plan_key: str, sources: list | None, config: Config, hub: Any, kind: str = "full"
) -> str | None:
    """A key identifying this exact optimization, or `None` when it must not be cached.

    `plan_key` is the plan's content fingerprint (`LogicalPlan.content_key()`), which the
    plan node computes and memoizes once — so re-issuing an identical query (or an adaptive
    re-optimization of the same subtree) does not re-serialize the whole IR here on every
    lookup, which was essentially all of the lookup cost. The remaining inputs — sources,
    the optimizer config (its `repr` memoized per config identity), the hub, and the learned
    generation — are cheap to fold in. `kind` names which optimizer entry point produced the
    value (`optimize_full` and `optimize_logical` share this memo but return different shapes,
    so they must not collide). The parts are joined with reserved delimiters none of them
    contain (hex digests, `id:`/`obj:<int>` source keys, a fixed `kind`), so the flat string
    is as injective as hashing the tuple was — without the per-lookup serialization.
    """
    source_ids = _source_keys(sources)
    if source_ids is None:
        return None
    # Injectivity: the first five fields are all `|`-free (a fixed `kind`, two hex digests,
    # two integers), so a `|`-split recovers them and everything after the fifth `|` is the
    # source component. That component is `repr(source_ids)` — unambiguous for a list of
    # strings even when a source identity (a file path) contains `|` or `,`, which a naive
    # delimiter-join would let collide two different source sets onto one key.
    return "|".join(
        (
            kind,
            plan_key,
            _config_key(config),
            str(id(hub)),
            str(learning.generation()),
            repr(source_ids),
        )
    )


# The optimizer config is stable for the life of an active `Config`, but its `repr` (the
# part of the key that reflects selectivity constants, cost weights, and which rules run)
# is ~1 KB to build. Memoize it by the config object's identity: a `config_context` swap
# makes a new object and re-derives it; the rare id-reuse after GC can only *collide* two
# configs onto one key, which per this module's contract returns a still-correct (merely
# possibly-worse) plan, never a wrong answer.
_CONFIG_KEY_CACHE: dict[int, str] = {}


def _config_key(config: Config) -> str:
    """A short, stable key for `config.optimizer`, memoized by config identity."""
    oid = id(config.optimizer)
    key = _CONFIG_KEY_CACHE.get(oid)
    if key is None:
        key = hashlib.blake2b(repr(config.optimizer).encode(), digest_size=8).hexdigest()
        if len(_CONFIG_KEY_CACHE) > 64:  # bound it; configs are few and long-lived
            _CONFIG_KEY_CACHE.clear()
        _CONFIG_KEY_CACHE[oid] = key
    return key


def _source_keys(sources: list | None) -> list[str] | None:
    """One data-stable key per source, or `None` if any source cannot be keyed safely."""
    keys: list[str] = []
    for source in sources or ():
        identity_fn = getattr(source, "identity", None)
        if not callable(identity_fn):
            return None  # an unkeyable source: never cache a plan built over it
        # A shape-based identity (in-memory data) collides across different relations, and
        # zone-map pruning reads a source's actual bounds — so key those by object identity.
        if getattr(source, "stable_stats_identity", True):
            keys.append(f"id:{identity_fn()}")
        else:
            keys.append(f"obj:{id(source)}")
    return keys


# --------------------------------------------------------------------------- #
# Invalidation
# --------------------------------------------------------------------------- #

# Fields that only *count* observations. They tick on every execution — `n_obs` from 1 to 2 is
# a 100% "change" — but no plan reads them as a value; they weight the averages beside them.
# Comparing them would make every write look material and defeat the memo entirely (measured:
# 6 hits in 8 identical runs became 0).
_BOOKKEEPING_FIELDS = frozenset({"n_obs", "n", "total", "flips"})


def record_write(hub: Any, namespace: str, key: str, value: object) -> None:
    """Write a learned value, invalidating memoized plans when it *materially* changed.

    Every value `kyber.learned_tuning` stores feeds a plan decision — which join strategy the
    bandit prefers, whether adaptive re-optimization pays off, how many partitions a breaker
    wants — so a plan memoized before the value moved is stale, and the contract that "plans
    improve the more a query runs" is broken. Routing all writes through one place is
    deliberate: the first version of this cache let the join-strategy bandit learn a better arm
    while the cache kept serving the old plan.

    But these are *measurements*, rewritten on every execution. Invalidating on their drift
    would mean never reusing a plan. So the write is compared against its prior and only a
    change large enough to flip a decision advances the generation. Over-bumping costs a
    re-plan; under-bumping leaves a stale plan, so anything unrecognized is treated as material.
    """
    prior = hub.get_keyed_param(namespace, key)
    if _materially_differs(prior, value):
        learning.bump_generation()
    hub.put_keyed_param(namespace, key, value)


def _materially_differs(prior: object, value: object) -> bool:
    """Whether `value` differs from `prior` by enough to change a plan decision."""
    if prior is None or type(prior) is not type(value):
        return True
    if isinstance(value, dict):
        keys = {k for k in value if k not in _BOOKKEEPING_FIELDS}
        if keys != {k for k in prior if k not in _BOOKKEEPING_FIELDS}:
            return True
        return any(_materially_differs(prior[k], value[k]) for k in keys)
    if isinstance(value, bool):
        return value != prior
    if isinstance(value, (int, float)):
        return learning.is_material_change(float(prior), float(value))
    return prior != value
