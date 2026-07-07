"""Learned strategy + parameter tuning — self-tuning physical decisions from measured runs.

Kyber's physical choices (which join algorithm, the broadcast byte threshold, the sort-merge
row crossover, whether to pre-aggregate, how many partitions, a per-task CPU share) ship as
*static* constants tuned for one cluster. This module closes the same learning loop the GPU
crossover (`gpu/adaptive.py`) and cost calibration (`calibration.py`) close: every decision reads
a **measured/learned** signal from the `MetadataHub`, keyed by plan signature, and on a cold store
falls back to the current default so a first run is byte-identical.

Every decision here ranges over **semantically-equivalent** alternatives — hash vs broadcast vs
sort-merge all emit the same relation, a partition count only shards, a CPU share only schedules —
so the tuned choice changes *performance*, never the *result*. That invariance is what lets Kyber
learn aggressively: the worst a wrong learned value can do is cost throughput, never correctness.

Two reusable primitives back the family:

* a **UCB1 bandit** (`ucb1_best_arm` / `learned_arm`) — regret-minimizing selection over a fixed
  arm set from measured per-arm latencies; deterministic (no RNG, ties broken by arm name), so a
  plan is reproducible. It generalizes the two-arm GPU crossover to N discrete algorithm arms.
* an **OLS line-crossover** (`_fit` / `_solve_crossover`) — the exact machinery `gpu/adaptive.py`
  uses, fitting `t ≈ a + b·x` per algorithm and solving for the x (bytes or rows) where the
  cheaper-below algorithm is overtaken by the cheaper-above one, clamped to a band around the
  default so one noisy early fit can't send a threshold to an absurd value.

Everything is best-effort: a malformed bucket, a degenerate fit, or a cold store yields the
default (or `None`), never an exception into planning or execution. **Core measures, Kyber
consumes** — the `record_*` functions fold one observation into O(1) sufficient statistics; the
`learned_*` functions read them back and decide.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from batcher.config import active_config

if TYPE_CHECKING:
    from batcher.metadata import MetadataHub

__all__ = [
    "learned_adaptive_helps",
    "learned_arm",
    "learned_broadcast_max_bytes",
    "learned_build_sides",
    "learned_join_strategy",
    "learned_partial_agg",
    "learned_partition_count",
    "learned_signature_rows",
    "learned_sort_merge_min_rows",
    "record_adaptive_flip",
    "record_arm",
    "record_broadcast_timing",
    "record_group_reduction",
    "record_join_sides",
    "record_join_strategy",
    "record_partition_rows",
    "record_sort_merge_timing",
    "ucb1_best_arm",
]

# --- namespaces (one hub learned-params bucket per decision family) ---------------------------
_NS_ARM = "tuning.join_arm"  # per-signature bandit arm statistics
_NS_BCAST = "tuning.broadcast_xover"  # OLS buckets: "broadcast" vs "shuffle", x = build bytes
_NS_SMJ = "tuning.sortmerge_xover"  # OLS buckets: "hash" vs "sort_merge", x = build rows
_NS_SIDES = "tuning.join_sides"  # per-signature measured (left_rows, right_rows)
_NS_PART = "tuning.partition_rows"  # per-signature measured shuffle rows
_NS_GROUP = "tuning.group_reduction"  # per-signature measured groups / input rows
_NS_ADAPT = "tuning.adaptive_flip"  # per-signature re-optimization flip counts

# The discrete join-algorithm arms the bandit ranges over — all equivalent relations.
_JOIN_ARMS: tuple[str, ...] = ("hash", "broadcast", "sort_merge")

# UCB exploration weight and the warm-up floor below which the bandit defers to the cost model.
_UCB_C = 1.0
_MIN_ARM_TOTAL = 3
# OLS confidence gate (mirrors gpu/adaptive: enough spread-out samples per algorithm) and the
# band a learned threshold is clamped to around its default.
_MIN_SAMPLES = 8
_BAND = 8.0
_SMOOTH_ALPHA = 0.5  # exp-smoothing of a per-signature scalar toward its latest observation


# Reusable primitive 1 — a deterministic UCB1 bandit over a fixed arm set.
def record_arm(
    hub: MetadataHub | None, namespace: str, key: str, arm: str, reward_ms: float
) -> None:
    """Fold one measured `reward_ms` for `arm` into the per-`key` bandit statistics.

    Stores `(n, sum, sumsq)` per arm under one keyed param so a record touches only its own
    signature. `reward_ms` is a latency (lower is better); the bandit minimizes it.
    """
    if hub is None or reward_ms <= 0.0 or not arm:
        return
    try:
        stats = dict(hub.get_keyed_param(namespace, key) or {})
        a = dict(stats.get(arm, {}))
        n = int(a.get("n", 0)) + 1
        stats[arm] = {
            "n": n,
            "sum": float(a.get("sum", 0.0)) + reward_ms,
            "sumsq": float(a.get("sumsq", 0.0)) + reward_ms * reward_ms,
        }
        hub.put_keyed_param(namespace, key, stats)
    except Exception:  # pragma: no cover - learning must never break a query
        return


def ucb1_best_arm(arm_stats: dict, arms: tuple[str, ...], *, c: float = _UCB_C) -> str | None:
    """The UCB1-optimal arm (minimizing latency) over `arms`, or `None` if none is tried.

    Deterministic: an untried arm is explored first (lowest name), then the arm with the smallest
    lower-confidence bound `mean - c·√(2·ln N / n)` wins, ties broken by name. No RNG, so a plan is
    reproducible run to run — the "fixed seed" a bandit needs here is simply the absence of one.
    """
    tried = {a: s for a, s in arm_stats.items() if a in arms and int(s.get("n", 0)) > 0}
    if not tried:
        return None
    total = sum(int(s["n"]) for s in tried.values())
    untried = sorted(a for a in arms if a not in tried)
    # Give an untried arm a turn once the tried arms have a little evidence — bounded exploration.
    if untried and total >= len(tried):
        return untried[0]
    best: str | None = None
    best_lcb = math.inf
    for a in sorted(tried):
        s = tried[a]
        n = int(s["n"])
        mean = float(s["sum"]) / n
        lcb = mean - c * math.sqrt(2.0 * math.log(max(2, total)) / n)
        if lcb < best_lcb:
            best_lcb, best = lcb, a
    return best


def learned_arm(
    hub: MetadataHub | None,
    namespace: str,
    key: str,
    arms: tuple[str, ...],
    *,
    min_total: int = _MIN_ARM_TOTAL,
) -> str | None:
    """Bandit arm for `key`, or `None` when there is too little evidence (defer to the cost model).

    Reads the per-key arm statistics and returns the UCB1 choice once at least `min_total`
    observations have accrued across the arms; below that the caller keeps its default.
    """
    if hub is None:
        return None
    try:
        stats = hub.get_keyed_param(namespace, key) or {}
        total = sum(int(s.get("n", 0)) for s in stats.values() if isinstance(s, dict))
        if total < min_total:
            return None
        return ucb1_best_arm(stats, arms)
    except Exception:  # pragma: no cover
        return None


# Reusable primitive 2 — an OLS two-line crossover (generalizing gpu/adaptive.py).
def _fold_ols(hub: MetadataHub, namespace: str, bucket: str, x: float, y: float) -> None:
    s = hub.get_keyed_param(namespace, bucket) or {}
    hub.put_keyed_param(
        namespace,
        bucket,
        {
            "n": int(s.get("n", 0)) + 1,
            "sx": float(s.get("sx", 0.0)) + x,
            "sy": float(s.get("sy", 0.0)) + y,
            "sxx": float(s.get("sxx", 0.0)) + x * x,
            "sxy": float(s.get("sxy", 0.0)) + x * y,
            "xmin": min(float(s.get("xmin", x)), x),
            "xmax": max(float(s.get("xmax", x)), x),
        },
    )


def _fit(s: dict) -> tuple[float, float] | None:
    """`(intercept, slope)` from OLS sufficient statistics, or `None` without enough spread."""
    n = int(s.get("n", 0))
    if n < _MIN_SAMPLES or float(s.get("xmax", 0.0)) <= float(s.get("xmin", 0.0)):
        return None
    sx, sy, sxx, sxy = s.get("sx", 0.0), s.get("sy", 0.0), s.get("sxx", 0.0), s.get("sxy", 0.0)
    denom = n * sxx - sx * sx
    if denom <= 0.0:
        return None
    slope = (n * sxy - sx * sy) / denom
    return (sy - slope * sx) / n, slope


def _solve_crossover(
    hub: MetadataHub | None,
    namespace: str,
    cheap_below: str,
    cheap_above: str,
    default: float,
) -> float | None:
    """The x where `cheap_above` overtakes `cheap_below`, clamped to a band around `default`.

    `cheap_below` is the algorithm that wins for small x (lower fixed cost, higher per-x slope);
    `cheap_above` wins for large x. A crossover exists only in that regime — any other shape means
    "no useful threshold in the data", so we return `None` and the caller keeps its default.
    """
    if hub is None:
        return None
    try:
        below = _fit(hub.get_keyed_param(namespace, cheap_below) or {})
        above = _fit(hub.get_keyed_param(namespace, cheap_above) or {})
    except Exception:  # pragma: no cover
        return None
    if below is None or above is None:
        return None
    a_b, b_b = below
    a_a, b_a = above
    # cheap_below must be cheaper for small x (lower intercept) yet grow faster (steeper slope).
    if not (b_b > b_a and a_a > a_b):
        return None
    xover = (a_a - a_b) / (b_b - b_a)
    if xover <= 0.0:
        return None
    return min(max(xover, default / _BAND), default * _BAND)


# Decision family — join strategy (bandit over hash / broadcast / sort_merge).
def record_join_strategy(
    hub: MetadataHub | None, signature: str, strategy: str, wall_ms: float
) -> None:
    """Record one join run's `(strategy, wall_ms)` under its plan signature."""
    record_arm(hub, _NS_ARM, signature, strategy, wall_ms)


def learned_join_strategy(
    hub: MetadataHub | None, signature: str, arms: tuple[str, ...] = _JOIN_ARMS
) -> str | None:
    """The measured-fastest join algorithm for this signature, or `None` cold (cost model decides).

    A regret-minimizing bandit over the equivalent algorithm arms: it converges to whichever
    strategy is genuinely fastest on *this* hardware/data, correcting a mis-ranked static cost
    guess. Every arm yields the identical relation, so the choice is result-invariant.
    """
    return learned_arm(hub, _NS_ARM, signature, arms)


# Decision family — learned broadcast and sort-merge thresholds (OLS crossover).
def record_broadcast_timing(
    hub: MetadataHub | None, strategy: str, build_bytes: float, wall_ms: float
) -> None:
    """Fold a `(build_bytes, wall_ms)` join run into the broadcast-vs-shuffle crossover buckets."""
    if (
        hub is None
        or build_bytes <= 0.0
        or wall_ms <= 0.0
        or strategy not in ("broadcast", "shuffle")
    ):
        return
    try:
        _fold_ols(hub, _NS_BCAST, strategy, float(build_bytes), float(wall_ms))
    except Exception:  # pragma: no cover
        return


def learned_broadcast_max_bytes(hub: MetadataHub | None, default: int | None = None) -> int | None:
    """The measured build-byte threshold below which broadcasting beats shuffling, or `None`.

    Broadcast wins for a small build (low fixed cost) but its replication cost grows with build
    bytes, so it is the cheaper-below arm; shuffle is cheaper-above. Solving their crossover learns
    `broadcast_max_bytes` from real timings instead of the static 10 MiB guess.
    """
    base = float(default if default is not None else active_config().optimizer.broadcast_max_bytes)
    xover = _solve_crossover(hub, _NS_BCAST, "broadcast", "shuffle", base)
    return int(xover) if xover is not None else None


def record_sort_merge_timing(
    hub: MetadataHub | None, strategy: str, build_rows: float, wall_ms: float
) -> None:
    """Fold a `(build_rows, wall_ms)` join run into the hash-vs-sort_merge crossover buckets."""
    if hub is None or build_rows <= 0.0 or wall_ms <= 0.0 or strategy not in ("hash", "sort_merge"):
        return
    try:
        _fold_ols(hub, _NS_SMJ, strategy, float(build_rows), float(wall_ms))
    except Exception:  # pragma: no cover
        return


def learned_sort_merge_min_rows(hub: MetadataHub | None, default: float) -> float | None:
    """The measured build-row crossover above which sort-merge beats hash, or `None`.

    Hash wins for a small build (cheap hash table) but degrades as the build strains memory, so it
    is the cheaper-below arm; sort-merge's bounded-memory merge is cheaper-above. Solving the
    crossover learns `SORT_MERGE_MIN_ROWS` from real hash-vs-SMJ timings.
    """
    return _solve_crossover(hub, _NS_SMJ, "hash", "sort_merge", default)


# Decision family — per-signature priors (build sides, partitions, pre-aggregation).
def _smooth(prior: float, observed: float, n_obs: int) -> float:
    alpha = max(_SMOOTH_ALPHA, 1.0 / (n_obs + 1))
    return alpha * observed + (1.0 - alpha) * prior


def _record_scalar(
    hub: MetadataHub | None, namespace: str, key: str, field: str, value: float
) -> None:
    if hub is None or value < 0.0:
        return
    try:
        entry = dict(hub.get_keyed_param(namespace, key) or {})
        n = int(entry.get("n_obs", 0))
        prior = entry.get(field)
        entry[field] = float(value) if prior is None else _smooth(float(prior), float(value), n)
        entry["n_obs"] = n + 1
        hub.put_keyed_param(namespace, key, entry)
    except Exception:  # pragma: no cover
        return


def record_join_sides(
    hub: MetadataHub | None, signature: str, left_rows: float, right_rows: float
) -> None:
    """Record a join's measured left/right input sizes, keyed by signature."""
    if hub is None:
        return
    try:
        entry = dict(hub.get_keyed_param(_NS_SIDES, signature) or {})
        n = int(entry.get("n_obs", 0))
        for field, value in (("left", left_rows), ("right", right_rows)):
            prior = entry.get(field)
            entry[field] = float(value) if prior is None else _smooth(float(prior), float(value), n)
        entry["n_obs"] = n + 1
        hub.put_keyed_param(_NS_SIDES, signature, entry)
    except Exception:  # pragma: no cover
        return


def learned_build_sides(hub: MetadataHub | None, signature: str) -> tuple[float, float] | None:
    """The measured `(left_rows, right_rows)` for this join, or `None` cold.

    Seeds build-side selection from what the two sides *actually* were last time, so a join whose
    estimate is wrong (correlated predicates, skew) still builds the truly-smaller side. Only the
    build orientation changes — the relation does not.
    """
    if hub is None:
        return None
    try:
        entry = hub.get_keyed_param(_NS_SIDES, signature) or {}
        left, right = entry.get("left"), entry.get("right")
        if left is None or right is None:
            return None
        return float(left), float(right)
    except Exception:  # pragma: no cover
        return None


def record_partition_rows(hub: MetadataHub | None, signature: str, rows: float) -> None:
    """Record a breaker's measured shuffle row count, keyed by signature."""
    _record_scalar(hub, _NS_PART, signature, "rows", rows)


def learned_partition_count(
    hub: MetadataHub | None, signature: str, target_rows: int
) -> int | None:
    """A partition prior from measured shuffle rows (`ceil(rows / target_rows)`), or `None`.

    Fan-out from the *measured* volume this breaker actually shuffled, not a cold estimate, so a
    recurring stage shards to fit memory on the first re-run. A partition count only shards data,
    so any value produces the identical result.
    """
    if hub is None or target_rows <= 0:
        return None
    try:
        entry = hub.get_keyed_param(_NS_PART, signature) or {}
        rows = entry.get("rows")
        if rows is None or float(rows) <= 0.0:
            return None
        return max(1, math.ceil(float(rows) / target_rows))
    except Exception:  # pragma: no cover
        return None


def record_group_reduction(
    hub: MetadataHub | None, signature: str, groups: float, input_rows: float
) -> None:
    """Record an aggregate's measured cardinality reduction (`groups / input_rows`)."""
    if input_rows <= 0.0:
        return
    _record_scalar(hub, _NS_GROUP, signature, "ratio", max(0.0, min(1.0, groups / input_rows)))


def learned_partial_agg(
    hub: MetadataHub | None, signature: str, *, engage_below: float = 0.5
) -> bool | None:
    """Whether to engage partial pre-aggregation, from the group-reduction ratio, or `None`.

    Partial pre-aggregation pays off exactly when a group-by collapses many rows into few groups
    (a low measured `groups/input` ratio); when almost every row is its own group it is wasted
    work. Learning the ratio per signature beats DuckDB's static "always pre-aggregate" guess.
    Engaging or skipping the pre-agg is an algebraic identity — the final aggregate is unchanged.
    """
    if hub is None:
        return None
    try:
        entry = hub.get_keyed_param(_NS_GROUP, signature) or {}
        ratio = entry.get("ratio")
        return None if ratio is None else float(ratio) <= engage_below
    except Exception:  # pragma: no cover
        return None


# Decision family — learned selectivity-primed estimate and the adaptive-re-opt gate.
def learned_signature_rows(hub: MetadataHub | None, signature: str) -> float | None:
    """The measured output rows recorded for a (sub)plan signature, or `None` if never seen.

    Reads the same `kyber.stats` feedback `learning.record_execution` writes, so a recurring
    subplan's estimate starts from its measured size rather than a default — priming selectivity
    and join-order costing for the intermediate, not just the whole query. Estimate-only: it steers
    cost, never the result.
    """
    if hub is None:
        return None
    try:
        from batcher.kyber.learning import load_learned_stats

        rows = load_learned_stats(hub).get(signature, {}).get("rows")
        return float(rows) if rows is not None else None
    except Exception:  # pragma: no cover
        return None


def record_adaptive_flip(hub: MetadataHub | None, signature: str, flipped: bool) -> None:
    """Record whether a stage's adaptive re-optimization changed the plan (`flipped`)."""
    if hub is None:
        return
    try:
        entry = dict(hub.get_keyed_param(_NS_ADAPT, signature) or {})
        entry["flips"] = int(entry.get("flips", 0)) + (1 if flipped else 0)
        entry["total"] = int(entry.get("total", 0)) + 1
        hub.put_keyed_param(_NS_ADAPT, signature, entry)
    except Exception:  # pragma: no cover
        return


def learned_adaptive_helps(
    hub: MetadataHub | None, signature: str, *, min_total: int = 3, threshold: float = 0.25
) -> bool:
    """Whether stage-by-stage re-optimization has historically helped this signature.

    Turns the adaptive gate itself into a learned decision: enable per-stage re-opt only for shapes
    where measured re-optimization actually *flipped* a plan often enough (a flip fraction above
    `threshold`), so cheap stable queries skip the re-opt overhead. `False` cold — the caller keeps
    its own default heuristic. Re-optimization only re-plans equivalent algebra, so gating it never
    changes a result; this only trades planning overhead.
    """
    if hub is None:
        return False
    try:
        entry = hub.get_keyed_param(_NS_ADAPT, signature) or {}
        total = int(entry.get("total", 0))
        if total < min_total:
            return False
        return int(entry.get("flips", 0)) / total >= threshold
    except Exception:  # pragma: no cover
        return False
