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
import math
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from batcher._internal.logging import note_suppressed
from batcher._internal.mathx import safe_div
from batcher.config import Config
from batcher.kyber import learning
from batcher.plan.source_stats import source_stats_key

__all__ = ["cache_key", "clear", "lookup", "record_write", "store"]

# Entry: key -> (result, keepalive). `keepalive` pins the source objects whose `id()` the
# key used, so the ids cannot be recycled while the entry lives (see the module docs).
_CACHE: OrderedDict[str, tuple[Any, tuple]] = OrderedDict()


def clear() -> None:
    """Drop every cached plan. For tests and for a hub reset.

    The bucket deadband goes with them: it is state *about* the keys, so leaving it behind
    would let one test's coefficients hold another's bucket.
    """
    _CACHE.clear()
    _BUCKET_STATE.clear()


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
    # The keepalive pins the sources whose `id()` the key used, so an address cannot be
    # recycled underneath a live entry. A derivation-keyed source is named by *how it was
    # derived* rather than by its address, so pinning it would buy nothing and cost the whole
    # materialized intermediate staying resident until the entry is evicted.
    _CACHE[key] = (
        result,
        tuple(s for s in (sources or ()) if not getattr(s, "derivation", None)),
    )
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
    learned: bool = True,
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

    `learned=False` drops the four **learned** fields — the generation, the calibration
    fingerprint, the measured read costs and the source statistics — leaving a key that moves
    only when the plan, the config, the hub or the sources do. It is emphatically **not** for
    the optimizer memo, whose whole purpose is to re-plan when the numbers move. It is for a
    caller whose decision is orders of magnitude less sensitive than a plan and whose analysis
    is orders of magnitude more expensive than a lookup, which in this codebase is exactly one
    caller: `api.subplan_reuse`, whose verdict is "does this subtree repeat, and is
    materializing it worth an engine round trip". Keyed with the learned fields it never once
    hit inside a mixed workload — every query in the suite moves the generation for every other
    — and the 400 ms analysis ran on every execution forever.
    """
    source_ids = _source_keys(sources)
    if source_ids is None:
        return None
    if not learned:
        return "|".join(
            (
                kind,
                plan_key,
                _config_key(config),
                str(id(hub)),
                _hardware_key(hardware),
                repr(source_ids),
            )
        )
    # Injectivity: the first nine fields are all `|`-free (a fixed `kind`, three hex digests,
    # three integers, and two comma-joined vectors whose free-form members are sanitized to an
    # alphanumeric alphabet by `_sanitized`), so a `|`-split recovers them and everything after
    # the ninth `|` is the source component. That component is
    # `repr(source_ids)` — unambiguous for a list of strings even when a source identity (a
    # file path) contains `|` or `,`, which a naive delimiter-join would let collide two
    # different source sets onto one key.
    return "|".join(
        (
            kind,
            plan_key,
            _config_key(config),
            str(id(hub)),
            str(learning.generation()),
            _calibration_epoch(hub),
            _read_cost_key(hub, sources),
            _source_stats_key(source_stats),
            _hardware_key(hardware),
            repr(source_ids),
        )
    )


# Half-octave buckets: a read-cost factor is folded into the key at
# ``round(log2(factor) * _READ_COST_BUCKETS)``, so a plan is re-optimized when a source's
# measured throughput moves by roughly 40% relative to its neighbours and not when the
# smoothed figure twitches. This is the same trade `_calibration_epoch` makes and for the
# same reason: keyed on the raw value the memo would miss on every query and cease to exist,
# keyed on nothing the plan would freeze at whichever factors were in force when it was first
# cached — which for a cold store is all-1.0, i.e. the measurement would never be spent.
_READ_COST_BUCKETS = 2


def _read_cost_key(hub: Any, sources: list | None) -> str:
    """A coarse fingerprint of the per-source read-cost factors, or ``"-"``.

    The third door past the generation counter, after cost calibration and CPU shares: the
    cost model prices a `Scan`'s bytes by the source's *measured* throughput relative to the
    plan's median (`metadata.io_stats.relative_read_cost`), and that is re-derived from the
    hub on every optimize. A plan memoized without it keeps whichever join order and build
    side the throughputs implied when it was first cached, however far they have since moved.

    Bucketed rather than exact — see `_READ_COST_BUCKETS`. A cold store, a single source, or
    an unidentifiable one all produce the same all-1.0 vector and therefore the same key, so
    nothing that was cacheable before becomes uncacheable.
    """
    if hub is None or not sources:
        return "-"
    try:
        from batcher.metadata.io_stats import relative_read_cost
        from batcher.plan.source_stats import source_identity

        factors = relative_read_cost(hub, [source_identity(s) for s in sources])
    except Exception as exc:
        # A learned read must never break the memo, but it must not fail *invisibly* either.
        # Degrading to "-" is the all-1.0 vector: every source looks equally cheap, so plans
        # silently stop specializing on read cost and simply get a little worse. That is the
        # failure mode `note_suppressed` exists for — "the difference between 'this
        # optimization did not apply' and 'this optimization has been broken since March'".
        note_suppressed("kyber", "read learned relative read cost", exc)
        return "-"
    return ",".join(
        str(round(math.log2(f) * _READ_COST_BUCKETS)) if f > 0.0 else "0" for f in factors
    )


def _hardware_key(hardware: Any) -> str:
    """A compact fingerprint of the target hardware, or ``"-"`` when unspecified.

    Folded into the key because a plan is now a function of the hardware too: the same query
    planned against a 16 MiB-L3 driver and 64 MiB-L3 cluster workers picks a different broadcast
    threshold, so reusing the driver's cached plan for the cluster run would ship the wrong one.
    Keyed only on the fields that actually steer a decision, so an unchanged machine keeps
    hitting its cached plan. Three of those are **not** numbers, and leaving them out was a
    silent hole: each is independent of every scalar here, so two profiles differing only in
    one of them produced an identical key and one fleet's plan was served to the other.

    * `storage_class` prices a spilled byte (`kyber.storage_cost`), across a **thirtyfold**
      range. Two fleets identical in cores, RAM, cache, VRAM and worker count — one on local
      NVMe, one on a network volume — shared a key, so whichever planned first decided for
      both whether an out-of-core plan was acceptable at all.
    * `accelerator_type` steers the device-tier decisions in `kyber.gpu.policy`: the host-link
      efficiency and the MIG profile a small model is packed into. VRAM alone does not imply
      either, which is exactly why the field exists separately from `gpu_memory_bytes`.
    * `fingerprint` selects *which* learned cost coefficients and CPU shares the optimizer
      loaded (`optimizer.facade` passes it to `calibrate` and `load_cpu_utilization`). Two
      fleets of the same shape and different silicon — a Graviton fleet and an x86 one at the
      same core count and RAM — read different coefficients and rank plans differently, and
      `_calibration_epoch` cannot see the difference because it keys on the hub's version
      rather than on whose measurements were read.

    The fleet's **shape** is keyed too, and by structure rather than by identity. Thirty-two
    devices on four nodes and thirty-two on thirty-two produce identical values for every scalar
    field above while ranking a shuffle a factor of eight apart, so a key built from the scalars
    alone would serve one fleet's plan to the other. Keyed on the *shape* — node count, device
    density, domain width, rack spread — and not on node ids, so an autoscaler replacing a node
    with an identical one keeps its cached plans instead of re-optimizing the workload from
    scratch on every reschedule.
    """
    if hardware is None:
        return "-"
    h = hardware
    scalars = (
        f"{h.cpu_cores},{h.memory_bytes},{h.l3_cache_bytes},{h.gpu_memory_bytes},{h.worker_count}"
    )
    # Sanitized to the key's alphabet rather than trusted: these three are the only free-form
    # strings in the whole key, and a `,` or `|` arriving in a device model from a node label
    # nobody controls would make two different profiles' keys ambiguous under the `|`-split
    # this function's contract promises.
    identity = ",".join(
        _sanitized(getattr(h, name, ""))
        for name in ("storage_class", "accelerator_type", "fingerprint")
    )
    return f"{scalars},{identity},{_cluster_key(getattr(h, 'cluster', None))}"


def _sanitized(value: Any) -> str:
    """A free-form field reduced to the key's delimiter-free alphabet, `"-"` when empty."""
    text = str(value or "")
    return "".join(c if (c.isalnum() or c in "._") else "_" for c in text) or "-"


def _cluster_key(cluster: Any) -> str:
    """A structural fingerprint of the fleet, or ``"-"`` when its shape is unknown.

    Deliberately lossy: the counts that change a ranking, not the nodes that carry them. Two
    fleets that agree on every figure here rank every plan identically, which is exactly the
    condition for sharing a memoized plan.
    """
    if cluster is None or not getattr(cluster, "known", False):
        return "-"
    return (
        f"{cluster.node_count}n{len(cluster.gpu_nodes)}g{cluster.total_gpus}d"
        f"{cluster.max_gpus_per_node}x{cluster.largest_nvlink_domain}v"
        f"{cluster.racks}r{len(cluster.device_models)}m"
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

    The epoch is read from the two refit memos themselves — the hub version each fit was
    computed at — because that is the only thing that advances exactly when a fit is replaced.
    It used to be `version // _RECALIBRATE_AFTER`, which reads as the same quantity and is a
    different clock: the refit throttle counts feedback rows *since the last refit* while the
    bucket counts from zero, so on a query recording ~35 operators the bucket rolled over every
    second execution regardless of whether anything had been re-fit. Measured on TPC-H q8 at
    sf10 the plan cache alternated hit/miss forever over a completely stable set of
    coefficients — and a miss there costs 350 ms against a hit's 160 ms, so half of the warm
    query was a re-plan that should have been a lookup.
    """
    if getattr(hub, "version", None) is None:
        return "-"
    from batcher.kyber import calibration, cpu_shares

    coeffs = _bucketed(calibration.live_coefficients(hub), (id(hub), "coeffs"))
    shares = _bucketed(cpu_shares.live_shares(hub), (id(hub), "shares"))
    return f"{coeffs},{shares}"


#: How far past a bucket's edge a coefficient must move before the key follows it.
#:
#: Bucketing alone is not enough, and the reason is the shape of the data rather than the
#: width of the bucket. A coefficient is re-fit from measured operator times, so it does not
#: settle on a value — it wanders inside a band. A coefficient whose band *straddles* an edge
#: therefore alternates buckets forever however wide the buckets are, and the key alternates
#: with it. Measured on TPC-DS q80 with half-octave buckets already in force: `hash_build_row`
#: crossed 8 <-> 9 and `hash_probe_row` 0 <-> 2 run after run, so the plan cache alternated
#: hit/miss indefinitely on an identical query (~140 ms of re-optimization on every miss, and
#: the *reuse* verdict cache keyed on the same string never hit at all — 400 ms more).
#:
#: A quarter-bucket deadband turns "which bucket is nearest" into "has it left the bucket it
#: was in", which is the question a cache key wants: a wandering value keeps its bucket, and a
#: genuinely drifting one still moves as soon as it is properly inside the next.
_BUCKET_HYSTERESIS = 0.25

#: Last bucket emitted per `(state key, coefficient name)`, so the deadband above has
#: something to be sticky about. Bounded and cleared wholesale — a dropped entry costs one
#: bucket re-derivation, and a wrong one costs a slightly stale plan, exactly as the bucketing
#: itself does.
_BUCKET_STATE: dict[tuple, int] = {}
_BUCKET_STATE_MAX = 4096

#: Buckets per octave for the quantities `_bucketed` fingerprints — the fitted cost
#: coefficients and the measured CPU shares — as against a read-cost factor.
#:
#: One, where `_READ_COST_BUCKETS` is two, and the difference is which quantity is being
#: fingerprinted. A read-cost factor is a ratio between sources that settles once measured; a
#: cost coefficient sits inside a feedback loop — it is fit from the operators the plan ran,
#: and the plan it produces decides which operators run next. On TPC-DS q77 that loop does not
#: reach a fixed point at all: with the anchor settled and a genuine shift landing in one refit
#: (`calibration._tracked`), `filter_row`, `hash_probe_row` and `project_row` still walk ~10% a
#: run in one direction, so half-octave buckets kept re-keying the memo on drift that changes
#: no plan. An octave is also exactly where `_tracked` stops damping, which is what makes the
#: pair coherent: **a coefficient the estimator treats as genuinely changed is a coefficient the
#: key follows, and nothing smaller is.**
_COEFF_BUCKETS = 1


def _sticky_bucket(state_key: tuple, name: str, value: float) -> int:
    """`value`'s octave bucket, kept at its previous one inside the deadband."""
    if value <= 0.0:
        return 0
    raw = math.log2(value) * _COEFF_BUCKETS
    key = (*state_key, name)
    previous = _BUCKET_STATE.get(key)
    if previous is not None and abs(raw - previous) < 0.5 + _BUCKET_HYSTERESIS:
        return previous
    bucket = round(raw)
    if len(_BUCKET_STATE) >= _BUCKET_STATE_MAX:
        _BUCKET_STATE.clear()
    _BUCKET_STATE[key] = bucket
    return bucket


def _bucketed(fit: object, state_key: tuple = ()) -> str:
    """A coefficient set as half-octave buckets, so drift does not move the key.

    The same device as `_read_cost_key`, applied to the fit itself instead of to *when* it was
    made, and for a sharper reason. A refit-version epoch is only stable while refits are
    rare; the throttle counts feedback rows, so a query recording more operators than
    `calibration._RECALIBRATE_AFTER` in one execution refits on **every** execution and the
    epoch advances every execution. The key then never repeats and the memo cannot hit, which
    is not a small loss: measured on tpcds-q83, `hit=0 / miss=1` on every run of an identical
    query, 190 ms of re-optimization against 20 ms in the engine; on tpcds-q80 the epoch
    climbed by 76 per run forever. Fingerprinting the *values* is stable by construction —
    it moves when a coefficient crosses a bucket and not when a refit merely happened.

    Buckets are ``round(log2(v) * _COEFF_BUCKETS)``, so a coefficient must roughly double or
    halve to change the key — the same trade the read-cost factors take at half that width, for
    the reason `_COEFF_BUCKETS` gives: keyed on the raw value the memo would miss on every
    query, keyed on nothing a plan would freeze at whichever coefficients were in force when it
    was first cached.

    Bucketing is necessary and **not sufficient**: see `_BUCKET_HYSTERESIS` for why a
    coefficient that wanders across a bucket edge defeats it, and what the deadband adds.
    `state_key` names whose buckets are being kept sticky; the default `()` is for callers
    with nothing to be sticky about (tests, and any use where one fit is fingerprinted once).
    """
    if fit is None:
        return "-"
    values: list[tuple[str, float]]
    if isinstance(fit, dict):
        values = sorted((str(k), float(v)) for k, v in fit.items())
    else:
        import dataclasses

        values = sorted(
            (f.name, float(getattr(fit, f.name)))
            for f in dataclasses.fields(fit)
            if isinstance(getattr(fit, f.name), (int, float))
            and not isinstance(getattr(fit, f.name), bool)
        )
    return ";".join(f"{n}:{_sticky_bucket(state_key, n, v)}" for n, v in values)


# Digest memo for `_source_stats_key`, keyed by the statistics object's identity.
#
# The entry **retains the object it is keyed on**, which is what makes an `id()` key sound
# here: CPython reuses the address of a freed object immediately, so an id-keyed memo that
# does not hold a reference can serve one relation's digest for another's — and this digest
# gates zone-map pruning, so that is a wrong answer rather than a worse plan. Holding the key
# object alive makes the id unique for as long as the entry exists.
#
# `SourceStatistics` is `frozen=True, slots=True` with no `__weakref__`, so it can neither be
# weak-referenced nor carry a cached attribute; an id map that owns its keys is the remaining
# option. Bounded and cleared wholesale like `_CONFIG_KEY_CACHE`, so the retention cannot grow
# without limit — a dropped entry costs one recomputation, never a wrong key.
_STATS_DIGEST_CACHE: dict[int, tuple[object, str]] = {}
_STATS_DIGEST_CACHE_MAX = 256
#: Digest of a source that reported no statistics at all. Fixed-width, like a real digest, so
#: the concatenation in `_source_stats_key` stays positional and therefore injective.
_NO_STATS_DIGEST = "0" * 16


def _stats_digest(stats: object) -> str:
    """A fixed-width hex digest of one source's `SourceStatistics`, memoized by identity."""
    if stats is None:
        return _NO_STATS_DIGEST
    cached = _STATS_DIGEST_CACHE.get(id(stats))
    if cached is not None and cached[0] is stats:
        return cached[1]
    digest = hashlib.blake2b(repr(stats).encode(), digest_size=8).hexdigest()
    if len(_STATS_DIGEST_CACHE) >= _STATS_DIGEST_CACHE_MAX:
        _STATS_DIGEST_CACHE.clear()
    _STATS_DIGEST_CACHE[id(stats)] = (stats, digest)
    return digest


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

    Memoized per statistics *object*, because building the digest is `repr` of every column's
    `ColumnStat` — a Python-level dataclass `__repr__` per column, each spelling out two or
    three `Provenance` enum members — and the conductor hands the same objects back on every
    execution of a query. On ClickBench's 105-column `hits` that repr was the single largest
    item in a `collect()` that the engine finishes in a fifth of the time.
    """
    if not source_stats:
        return "-"
    return "".join(_stats_digest(s) for s in source_stats)


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

    An **ephemeral** source is refused for the same reason as an unkeyable one, and the
    reason is worth stating because it is not that its key is wrong — it is that the key is
    correct and unrepeatable. An adaptive stage boundary wraps its intermediate as an
    in-memory source, keyed by object identity, and that object is gone by the next
    execution. A plan filed under it can therefore never be *read* again, and storing it
    anyway costs on three counts: it burns an LRU slot and evicts an entry that would have
    hit (measured on TPC-H Q8 at sf10, where the reusable entry survived only every other
    run), and `store` pins the source tuple alive as a keepalive — so the stage's whole
    materialized intermediate, tens or hundreds of megabytes of it, stays resident until
    eviction. Not caching costs one re-plan of a subtree that was going to be re-planned
    regardless.
    """
    keys: list[str] = []
    for source in sources or ():
        key = source_stats_key(source)
        if key is None:
            return None  # an unkeyable source: never cache a plan built over it
        if getattr(source, "ephemeral", False) and not getattr(source, "derivation", None):
            # A key that cannot recur: the entry could never be read again. An ephemeral
            # relation *with* a derivation key is the exception the rule was missing — it
            # names how the engine derived it, so the next run of the same query derives the
            # same relation and asks for the same key.
            return None
        keys.append(key)
    return keys


# --------------------------------------------------------------------------- #
# Invalidation
# --------------------------------------------------------------------------- #

# Fields that only *count* observations. They tick on every execution — `n_obs` from 1 to 2 is
# a 100% "change" — but no plan reads them as a value; they weight the averages beside them.
# Comparing them would make every write look material and defeat the memo entirely (measured:
# 6 hits in 8 identical runs became 0).
# The OLS co-moments (`m2x`/`m2y`/`cxy`, from `kyber.ols`) and the bandit arm's squared-error
# accumulator (`m2`, from `record_arm`) belong here for the same reason `n` does: each grows
# with every observation, so comparing them raw makes *every* join run look material and
# flushes the whole plan cache. What a plan actually reads is a quotient of them, compared via
# `_DERIVED_RATIOS` below. (`xmin`/`xmax` stay compared directly: they are bounds, not
# accumulators, and move only on a genuinely new extreme — which does change the fit's
# applicable range. `mx`/`my`/`mean` stay compared directly too: those *are* the decision.)
#
# **This list is keyed on field names, so it goes stale silently when a writer changes shape,
# and it had.** `kyber.ols` was rewritten from power sums (`sx`/`sy`/`sxx`/`sxy`) to a centered
# Welford form, and `record_arm` from `sum`/`sumsq` to a discounted Welford `(n, mean, m2)` —
# and this list still named the retired fields, so none of the live accumulators was recognized
# and the "6 hits in 8 identical runs became 0" regression it exists to prevent was fully back.
# Measured on TPC-H at scale 1: **seven of the twenty-two queries never hit the plan cache at
# all** (q4, q12, q13, q14, q15, q19, q22), and they were exactly the queries whose control
# plane dominated their wall clock — q19 spent 73% of 40 ms re-optimizing, q15 85% of 9 ms.
# In the traces the bandit's `mean` was *bit-identical* between runs while `m2` drifted, so the
# decision had not moved and the memo was flushed anyway.
#
# Anything added to `kyber.ols` or `bandit._welford_update` has to be classified here too;
# `tests/unit/test_plan_cache_accumulators.py` fails if a writer emits a field this file has
# never heard of, so the next rename cannot repeat this quietly.
_BOOKKEEPING_FIELDS = frozenset({"n_obs", "n", "m2", "m2x", "m2y", "cxy"})

# Pairs whose *ratio* is a decision even though both fields are bookkeeping. A bandit arm is
# the canonical case: `record_arm` writes only accumulators, so with each of them listed above
# the key-set comparison below sees an empty set, `any(())` is False, and the write could never
# bump the generation — the arm's ranking would move while a memoized plan chosen under the old
# one was served forever, which is precisely the staleness routing every write through one place
# was meant to prevent. Comparing the raw counters instead would bump on every execution (`n`
# 1 -> 2 is a 100% "change") and defeat the memo, so the ratio — the number a plan actually
# reads — is what gets compared.
_DERIVED_RATIOS: tuple[tuple[str, str], ...] = (
    # The bandit's per-observation variance, which `ucb1_best_arm`'s UCB-V radius reads. Its
    # `mean` is compared directly, being the value the arms are ranked by.
    ("m2", "n"),
    # The OLS fit. `fit_ols`'s slope is `cxy / m2x`, and its R² gate factors exactly into these
    # two quotients: `cxy**2 / (m2x*m2y) == (cxy/m2x) * (cxy/m2y)`. So when neither has moved
    # materially, neither the slope nor the fit's credibility has, and the crossover the plan
    # was chosen under is unchanged. (`m2y/n` — the response variance — is deliberately *not*
    # here: an observation landing exactly on the fitted line moves it, while moving no term of
    # the fit, so comparing it invalidates on a sample that confirms the model.)
    ("cxy", "m2x"),
    ("cxy", "m2y"),
)


def record_write(
    hub: Any,
    namespace: str,
    key: str,
    value: object,
    *,
    decides: Callable[[object], object] | None = None,
) -> None:
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

    `decides` closes the gap between those two sentences. It maps a stored value to **the
    decision a plan actually reads from it**, and when supplied the write invalidates only if
    that decision changed — which is the exact condition, not a proxy for it.

    The value-drift proxy is not merely imprecise here, it is self-sustaining. A bandit arm's
    reward is the query's own latency, so a slow run writes a mean that differs by more than
    the materiality threshold, which invalidates the plan, which makes the next run pay the
    optimizer again, which keeps it slow. Measured on a two-table star join over TPC-DS at
    scale 1: `optimize` ran twice per query at ~12 ms against ~2 ms of execution, and the
    query sat at **26 ms for twelve consecutive runs** before the arm's discounted mean
    settled inside the threshold — then dropped to **4.7 ms** and stayed there. The arm the
    bandit would have chosen was `broadcast` on every one of those runs.
    """
    prior = hub.get_keyed_param(namespace, key)
    material = (
        _materially_differs(prior, value) if decides is None else decides(prior) != decides(value)
    )
    if material:
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
