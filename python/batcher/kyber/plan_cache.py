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

from batcher._internal.mathx import safe_div
from batcher.config import Config
from batcher.kyber import learning
from batcher.plan.source_stats import source_stats_key

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
    plan_key: str,
    sources: list | None,
    config: Config,
    hub: Any,
    kind: str = "full",
    source_stats: list | None = None,
    hardware: Any = None,
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
    # Injectivity: the first eight fields are all `|`-free (a fixed `kind`, three hex digests,
    # three integers, a comma-joined hardware fingerprint), so a `|`-split recovers them and
    # everything after the eighth `|` is the
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
            _calibration_epoch(hub),
            _source_stats_key(source_stats),
            _hardware_key(hardware),
            repr(source_ids),
        )
    )


def _hardware_key(hardware: Any) -> str:
    """A compact fingerprint of the target hardware, or ``"-"`` when unspecified.

    Folded into the key because a plan is now a function of the hardware too: the same query
    planned against a 16 MiB-L3 driver and 64 MiB-L3 cluster workers picks a different broadcast
    threshold, so reusing the driver's cached plan for the cluster run would ship the wrong one.
    Keyed only on the fields that actually steer a decision (`|`-free integers), so an
    unchanged machine keeps hitting its cached plan.
    """
    if hardware is None:
        return "-"
    h = hardware
    return (
        f"{h.cpu_cores},{h.memory_bytes},{h.l3_cache_bytes},{h.gpu_memory_bytes},{h.worker_count}"
    )


def _calibration_epoch(hub: Any) -> str:
    """Which cost-coefficient refit a plan was chosen under, or `"-"` without a hub.

    `learning.generation()` above covers everything written through `record_write`, but the
    **cost calibration** and **CPU-share** refits do not go through it — they read
    `hub.op_stats_by_kind()` directly and re-fit every `_RECALIBRATE_AFTER` feedback rows. So
    a plan (and the `ResourceBounds`/cpu shares annotated onto it) stayed frozen at whatever
    coefficients were in force when it was first memoized, however much the engine's measured
    per-row costs had since moved — the very staleness this module exists to prevent, entering
    by a door the generation counter does not watch.

    Keyed by the refit *epoch* rather than the raw version, so it changes exactly when a refit
    can change the coefficients and not once per recorded operator — which would miss on every
    single execution and defeat the memo entirely.
    """
    version = getattr(hub, "version", None)
    if version is None:
        return "-"
    from batcher.kyber.calibration import _RECALIBRATE_AFTER

    return str(int(version) // _RECALIBRATE_AFTER)


def _source_stats_key(source_stats: list | None) -> str:
    """A digest of the collected `SourceStatistics`, or `"-"` when none were supplied.

    The source *identity* is not enough. Zone-map pruning folds a filter to `FALSE` from a
    source's footer `min`/`max`, and those bounds arrive in `source_stats` — collected at
    plan-build time, not derived from the source object. The same source list can therefore
    be optimized twice with *different* statistics (a footer re-collected after an append; a
    caller that passes them on one path and `None` on another), and without this field the
    second call would be served the first call's pruned plan — the very wrong-answer this
    module's `_source_keys` docstring warns about, entering by the other door.

    Folded in as a hex digest so it stays `|`-free and the key's injectivity argument holds.
    """
    if not source_stats:
        return "-"
    return hashlib.blake2b(repr(tuple(source_stats)).encode(), digest_size=8).hexdigest()


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
    """One data-stable key per source, or `None` if any source cannot be keyed safely.

    `plan.source_stats.source_stats_key` is the single definition of that key — a
    data-stable identity where one exists, object identity for shape-keyed in-memory data
    (whose `identity()` collides across different relations), `None` when a source cannot
    key itself. The learned column statistics are filed under exactly the same key, so a
    plan and the statistics it was chosen under can never disagree about which source is
    which.
    """
    keys: list[str] = []
    for source in sources or ():
        key = source_stats_key(source)
        if key is None:
            return None  # an unkeyable source: never cache a plan built over it
        keys.append(key)
    return keys


# --------------------------------------------------------------------------- #
# Invalidation
# --------------------------------------------------------------------------- #

# Fields that only *count* observations. They tick on every execution — `n_obs` from 1 to 2 is
# a 100% "change" — but no plan reads them as a value; they weight the averages beside them.
# Comparing them would make every write look material and defeat the memo entirely (measured:
# 6 hits in 8 identical runs became 0).
# The OLS sufficient statistics (`sx`/`sy`/`sxx`/`sxy`, from `learned_tuning.crossover`) and
# the bandit arm accumulators (`sum`/`sumsq`, from `record_arm`) belong here for the same
# reason `n` does: every one of them grows monotonically with each observation, so comparing
# them raw made *every* join run look material and flushed the whole plan cache — the exact
# "6 hits in 8 identical runs became 0" regression this list was created to fix, still live
# for the accumulators sitting beside the counter that was fixed. What a plan actually reads
# is their per-observation quotient, compared via `_DERIVED_RATIOS` below. (`xmin`/`xmax` stay
# compared directly: they are bounds, not accumulators, and move only on a genuinely new
# extreme — which does change the fit's applicable range.)
_BOOKKEEPING_FIELDS = frozenset(
    {"n_obs", "n", "total", "flips", "sx", "sy", "sxx", "sxy", "sum", "sumsq"}
)

# Pairs whose *ratio* is a decision even though both fields are bookkeeping. `flips/total` is
# the adaptive gate: `learned_tuning.record_adaptive_flip` writes **only** those two counters,
# so with both listed above the key-set comparison below sees an empty set, `any(())` is False,
# and the write can never bump the generation — `learned_adaptive_helps` flips False -> True and
# a memoized plan is served forever, which is precisely the staleness routing every write
# through one place was meant to prevent. Comparing the raw counters instead would bump on
# every execution (`total` 1 -> 2 is a 100% "change") and defeat the memo, so the ratio — the
# number a plan actually reads — is what gets compared.
_DERIVED_RATIOS: tuple[tuple[str, str], ...] = (
    ("flips", "total"),  # the adaptive gate's flip fraction
    ("sum", "n"),  # a bandit arm's mean reward — what `ucb1_best_arm` ranks by
    ("sumsq", "n"),  # its second moment, which the UCB confidence width reads
    # The OLS fit's per-observation moments. `_fit`'s intercept and slope are functions of
    # exactly these, so when none of them has moved materially neither has the crossover the
    # plan was chosen under — and when one has, the plan is genuinely stale.
    ("sx", "n"),
    ("sy", "n"),
    ("sxx", "n"),
    ("sxy", "n"),
)


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


def _ratio_differs(prior: dict, value: dict) -> bool:
    """Whether a `_DERIVED_RATIOS` pair moved enough to change the decision it encodes.

    Both fields of such a pair are bookkeeping counters, so neither is compared on its own;
    their quotient is the value a plan reads. A run that ticks `total` without moving the
    ratio stays a cache hit, while a run that moves it materially invalidates.
    """
    for numerator, denominator in _DERIVED_RATIOS:
        if not all(f in prior and f in value for f in (numerator, denominator)):
            continue
        if learning.is_material_change(
            _ratio(prior[numerator], prior[denominator]),
            _ratio(value[numerator], value[denominator]),
        ):
            return True
    return False


def _ratio(numerator: object, denominator: object) -> float:
    """`numerator / denominator` as a float, or 0.0 for a zero/unusable denominator."""
    try:
        den = float(denominator)  # type: ignore[arg-type]
        return safe_div(float(numerator), den)  # type: ignore[arg-type]
    except (TypeError, ValueError):  # pragma: no cover - non-numeric bookkeeping
        return 0.0


def _materially_differs(prior: object, value: object) -> bool:
    """Whether `value` differs from `prior` by enough to change a plan decision."""
    if prior is None or type(prior) is not type(value):
        return True
    if isinstance(value, dict):
        keys = {k for k in value if k not in _BOOKKEEPING_FIELDS}
        if keys != {k for k in prior if k not in _BOOKKEEPING_FIELDS}:
            return True
        if _ratio_differs(prior, value):
            return True
        return any(_materially_differs(prior[k], value[k]) for k in keys)
    if isinstance(value, bool):
        return value != prior
    if isinstance(value, (int, float)):
        return learning.is_material_change(float(prior), float(value))
    return prior != value
