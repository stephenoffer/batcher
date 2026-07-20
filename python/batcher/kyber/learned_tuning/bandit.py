"""A deterministic UCB1 bandit over a fixed arm set — and the join-strategy choice on it.

Regret-minimizing selection from measured per-arm latencies, with no RNG anywhere (ties break
by arm name), so a plan is reproducible run to run. It generalizes the two-arm GPU crossover
(`gpu/adaptive.py`) to N discrete algorithm arms, and its one caller here is the join-strategy
decision: hash vs broadcast vs sort-merge, all of which emit the same relation.

The family contract — result-invariance, cold-store fallback, best-effort — is in the package
docstring.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from batcher.kyber import plan_cache

if TYPE_CHECKING:
    from batcher.metadata import MetadataHub

__all__ = [
    "learned_arm",
    "learned_join_strategy",
    "record_arm",
    "record_join_strategy",
    "ucb1_best_arm",
]

# Learned-tuning values all feed *plan* decisions, so writing one may invalidate a memoized
# plan. `plan_cache.record_write` applies the materiality test and does the write; routing
# every write through it means a writer cannot forget to invalidate.

# v2: arm rewards are now size-normalized (ms per million input rows), not raw wall ms.
# A fresh namespace so a hub carrying ms-scale history cannot mix the two scales.
_NS_ARM = "tuning.join_arm_v2"  # per-signature bandit arm statistics

# The discrete join-algorithm arms the bandit ranges over — all equivalent relations.
_JOIN_ARMS: tuple[str, ...] = ("hash", "broadcast", "sort_merge")

# UCB exploration weight (dimensionless — the radius is scaled by the measured reward spread)
# and the warm-up floor below which the bandit defers to the cost model.
_UCB_C = 1.0
_MIN_ARM_TOTAL = 3
# Floor on the reward scale, as a fraction of the pooled mean reward. A history with no
# observed spread (every run identical, or one sample per arm) would otherwise get a zero
# exploration radius and freeze on whichever arm was measured first.
_UCB_SCALE_FLOOR = 0.25


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
        plan_cache.record_write(hub, namespace, key, stats)
    except Exception:  # pragma: no cover - learning must never break a query
        return


def _reward_scale(tried: dict) -> float:
    """The pooled spread of the observed rewards — the unit the confidence radius is measured in.

    Textbook UCB1 assumes rewards in `[0, 1]`, which makes its `sqrt(2*ln N / n)` radius directly
    comparable to a mean. These rewards are latencies in arbitrary units, so a bare radius is
    dimensionally meaningless: against a 500 ms mean it is a 0.2% nudge and the bandit collapses to
    greedy, while against a per-row mean of 1e-5 ms the same radius is pure exploration. Scaling by
    the pooled standard deviation (recovered from the `sumsq` each `record_arm` already stores)
    makes `c` dimensionless, so the arm ranking is invariant to whether the reward is recorded in
    ms, microseconds, or ms-per-row, and the exploration rate is a real fraction of the spread.

    This does **not** rescue an arm whose recorded mean is genuinely far worse: the pooled spread
    decays as `1/sqrt(N)`, so the radius shrinks faster than such a gap closes, and UCB is right to
    stop paying for it. An arm made to *look* far worse by a larger input is a non-stationarity in
    the reward, and is fixed where the reward is formed — see `record_join_strategy`.

    Floored at `_UCB_SCALE_FLOOR * |mean|` so a history with no observed spread still explores.
    """
    total = sum(int(s["n"]) for s in tried.values())
    if total <= 0:
        return 0.0
    s_sum = sum(float(s.get("sum", 0.0)) for s in tried.values())
    s_sumsq = sum(float(s.get("sumsq", 0.0)) for s in tried.values())
    mean = s_sum / total
    variance = max(0.0, s_sumsq / total - mean * mean)
    return max(math.sqrt(variance), _UCB_SCALE_FLOOR * abs(mean))


def ucb1_best_arm(arm_stats: dict, arms: tuple[str, ...], *, c: float = _UCB_C) -> str | None:
    """The UCB1-optimal arm (minimizing latency) over `arms`, or `None` if none is tried.

    Deterministic: an untried arm is explored first (lowest name), then the arm with the smallest
    lower-confidence bound `mean - c*sd*sqrt(2*ln N / n)` wins, ties broken by name. No RNG, so a
    plan is reproducible run to run — the "fixed seed" a bandit needs is simply the absence of one.

    `sd` is the pooled reward spread (`_reward_scale`), which is what keeps the radius commensurate
    with the mean it is subtracted from.
    """
    tried = {a: s for a, s in arm_stats.items() if a in arms and int(s.get("n", 0)) > 0}
    if not tried:
        return None
    total = sum(int(s["n"]) for s in tried.values())
    untried = sorted(a for a in arms if a not in tried)
    # Give an untried arm a turn once the tried arms have a little evidence — bounded exploration.
    if untried and total >= len(tried):
        return untried[0]
    scale = _reward_scale(tried)
    best: str | None = None
    best_lcb = math.inf
    for a in sorted(tried):
        s = tried[a]
        n = int(s["n"])
        mean = float(s["sum"]) / n
        lcb = mean - c * scale * math.sqrt(2.0 * math.log(max(2, total)) / n)
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


# Decision family — join strategy (bandit over hash / broadcast / sort_merge).
def record_join_strategy(
    hub: MetadataHub | None, signature: str, strategy: str, wall_ms: float, input_rows: float = 0.0
) -> None:
    """Record one join run's strategy and its **size-normalized** cost under `signature`.

    UCB1 assumes each arm's reward is drawn from a fixed distribution. Raw wall time is not:
    the same signature runs over 1M rows today and 50M tomorrow, so an arm that happened to be
    sampled on a large input carries a permanently inflated mean and is never chosen again —
    the bandit locks onto whichever arm the first small run tried. Dividing by the join's input
    size makes the reward a per-row cost, which *is* approximately stationary across sizes, and
    is the quantity the arms actually differ in.

    Args:
        hub: The metadata hub to record into; `None` is a no-op.
        signature: The join's plan signature — the bandit key.
        strategy: The arm that ran (`hash` / `broadcast` / `sort_merge`).
        wall_ms: Measured wall time for the run.
        input_rows: Total rows entering the join (both sides). Non-positive falls back to the
            raw `wall_ms`, which is correct when every run of this signature is the same size.
    """
    mrows = max(0.0, input_rows) / 1e6
    reward = wall_ms / mrows if mrows > 0.0 else wall_ms
    record_arm(hub, _NS_ARM, signature, strategy, reward)


def learned_join_strategy(
    hub: MetadataHub | None, signature: str, arms: tuple[str, ...] = _JOIN_ARMS
) -> str | None:
    """The measured-fastest join algorithm for this signature, or `None` cold (cost model decides).

    A regret-minimizing bandit over the equivalent algorithm arms: it converges to whichever
    strategy is genuinely fastest on *this* hardware/data, correcting a mis-ranked static cost
    guess. Every arm yields the identical relation, so the choice is result-invariant.
    """
    return learned_arm(hub, _NS_ARM, signature, arms)
