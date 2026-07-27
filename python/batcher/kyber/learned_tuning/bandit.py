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

from batcher._internal.logging import note_suppressed
from batcher.kyber import plan_cache
from batcher.metadata.hardware_scope import scoped

if TYPE_CHECKING:
    from batcher.metadata import MetadataHub

__all__ = [
    "JOIN_ARMS",
    "learned_adaptive_route",
    "learned_arm",
    "learned_join_strategy",
    "record_adaptive_route",
    "record_arm",
    "record_join_strategy",
    "ucb1_best_arm",
]

# Learned-tuning values all feed *plan* decisions, so writing one may invalidate a memoized
# plan. `plan_cache.record_write` applies the materiality test and does the write; routing
# every write through it means a writer cannot forget to invalidate.

# v3: arm statistics are stored as a discounted Welford state `(n, mean, m2)` rather than
# `(n, sum, sumsq)`. A fresh namespace so a hub carrying the older shape cannot be misread.
#
# The change is two fixes in one record. `sumsq` recovers the variance as `E[x²] - E[x]²`, a
# subtraction of two nearly equal large numbers: with rewards around 1000 and a spread of 1,
# both terms are ~1e6 and the difference is noise — the same catastrophic cancellation the
# `var`/`covar` aggregates were rewritten to avoid. And an undiscounted sum remembers every
# run forever, so a machine that got faster, or data that changed shape, leaves an arm with a
# permanently stale mean that the bandit will not revisit.
_NS_ARM = "tuning.join_arm_v3"  # per-signature bandit arm statistics
_NS_ROUTE = "tuning.adaptive_route_v1"  # per-signature staged-vs-one-shot arm statistics
_NS_ROUTE_COLD = "tuning.adaptive_route_cold_v1"  # which route arms have spent their cold sample

# The discrete join-algorithm arms the bandit ranges over — all equivalent relations.
JOIN_ARMS: tuple[str, ...] = ("hash", "broadcast", "sort_merge")
# The two execution routes for a plan: re-optimize between stages, or plan once and run.
# Equivalent relations again — staging only re-plans, it never changes the algebra.
_ROUTE_ARMS: tuple[str, ...] = ("one_shot", "staged")
# One observation is enough to start ranking, unlike the join bandit's three. Each sample here
# is a whole query rather than one operator inside one, so the evidence per sample is far
# stronger; and with only two arms, "start ranking" means "explore the arm never tried" —
# `ucb1_best_arm` takes an untried arm first. So the second run of a signature measures the
# other route and the third onward picks the winner, which bounds the exploration cost at one
# run of the losing arm. (The floor is compared against a *discounted* count, so 2 would not
# even be reached by two observations: `_ARM_DISCOUNT` makes them sum to 1.975.)
_MIN_ROUTE_TOTAL = 1

# UCB exploration weight (dimensionless — the radius is scaled by the measured reward spread)
# and the warm-up floor below which the bandit defers to the cost model.
_UCB_C = 1.0
_MIN_ARM_TOTAL = 3
# Floor on the reward scale, as a fraction of the pooled mean reward. A history with no
# observed spread (every run identical, or one sample per arm) would otherwise get a zero
# exploration radius and freeze on whichever arm was measured first.
_UCB_SCALE_FLOOR = 0.25
# Per-observation discount applied to every arm's accumulated evidence — the discounted-UCB
# of Garivier & Moulines, which is the standard answer to a *non-stationary* bandit.
#
# UCB1 assumes each arm's reward distribution is fixed forever. A join's is not: the hardware
# changes, the data grows a skew, a release makes one strategy faster. With undiscounted
# statistics an arm measured badly a thousand runs ago carries that mean with a confidence
# radius that has shrunk to nothing, so the bandit can never re-examine it — it is not
# converged, it is stuck. Discounting gives the evidence an effective horizon of about
# `1/(1-gamma)` observations (~40 here), after which a genuinely changed arm is re-explored,
# while still averaging over enough runs to be immune to a single slow one.
_ARM_DISCOUNT = 0.975


# Reusable primitive 1 — a deterministic UCB1 bandit over a fixed arm set.
def record_arm(
    hub: MetadataHub | None,
    namespace: str,
    key: str,
    arm: str,
    reward_ms: float,
    *,
    invalidates_plans: bool = True,
) -> None:
    """Fold one measured `reward_ms` for `arm` into the per-`key` bandit statistics.

    Stores a discounted Welford state `(n, mean, m2)` per arm under one keyed param, so a
    record touches only its own signature. `reward_ms` is a latency (lower is better); the
    bandit minimizes it.

    `invalidates_plans=False` writes without advancing the learned generation. It is for the
    one bandit whose decision is made *outside* the optimizer — the execution route, chosen by
    `resolve_adaptive` before `optimize_full` runs and absent from the value the plan cache
    stores. Left at the default it is a slow leak of exactly the kind this module's `record_write`
    contract exists to prevent: an arm's mean moves on **every** execution, so every execution
    would invalidate every memoized plan. Measured on TPC-H q8 at sf10, that alone halved the
    plan cache's hit rate, and a hit is worth 160 ms against 350 ms there.
    """
    if hub is None or reward_ms <= 0.0 or not arm:
        return
    try:
        stored = hub.get_keyed_param(scoped(namespace), key) or {}
        # Every arm's evidence decays on every observation, not just the arm that ran. Decaying
        # only the observed arm would make the discount a function of how often an arm happens
        # to be chosen, so a rarely-picked arm would keep its ancient mean at full strength —
        # exactly the arm whose evidence is most likely to be stale.
        stats = {a: _decayed(v) for a, v in stored.items() if isinstance(v, dict)}
        stats[arm] = _welford_update(stats.get(arm), reward_ms)
        if invalidates_plans:
            plan_cache.record_write(hub, scoped(namespace), key, stats)
        else:
            hub.put_keyed_param(scoped(namespace), key, stats)
    except Exception as exc:  # pragma: no cover - learning must never break a query
        note_suppressed("kyber", "record a bandit arm observation", exc)
        return


def _decayed(state: dict) -> dict:
    """One arm's statistics after a single discount step.

    The mean is a location and is *not* decayed — decaying it would drag every arm toward
    zero. What decays is the *weight of evidence* behind it: the effective sample count and
    the accumulated squared deviation, which together are what the confidence radius is
    computed from. So an arm not chosen for a long time keeps its estimate but loses its
    certainty, which is what lets the bandit go back and check.
    """
    return {
        "n": float(state.get("n", 0.0)) * _ARM_DISCOUNT,
        "mean": float(state.get("mean", 0.0)),
        "m2": float(state.get("m2", 0.0)) * _ARM_DISCOUNT,
    }


def _welford_update(state: dict | None, reward: float) -> dict:
    """Fold one reward into a (possibly discounted, possibly absent) Welford state.

    `m2` accumulates the squared deviations from the running mean directly, so the variance
    is never recovered by subtracting two large nearly-equal numbers.
    """
    n = float((state or {}).get("n", 0.0)) + 1.0
    mean = float((state or {}).get("mean", 0.0))
    m2 = float((state or {}).get("m2", 0.0))
    if n <= 1.0:
        return {"n": 1.0, "mean": reward, "m2": 0.0}
    delta = reward - mean
    mean += delta / n
    return {"n": n, "mean": mean, "m2": m2 + delta * (reward - mean)}


def _arm_variance(state: dict) -> float | None:
    """An arm's own reward variance, or None when a single observation cannot establish one."""
    n = float(state.get("n", 0.0))
    if n <= 1.0:
        return None
    return max(0.0, float(state.get("m2", 0.0)) / n)


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
    total = sum(float(s["n"]) for s in tried.values())
    if total <= 0.0:
        return 0.0
    # Pool the per-arm Welford states with Chan's parallel formula, which is the same
    # cancellation-free combine the variance aggregate uses. Pooling the *within-arm* spreads
    # and the *between-arm* mean differences separately is also what makes this a spread and
    # not a measure of how far apart the arms are.
    pooled_n = 0.0
    pooled_mean = 0.0
    pooled_m2 = 0.0
    for a in sorted(tried):
        s = tried[a]
        nb = float(s["n"])
        mb = float(s.get("mean", 0.0))
        m2b = float(s.get("m2", 0.0))
        if pooled_n <= 0.0:
            pooled_n, pooled_mean, pooled_m2 = nb, mb, m2b
            continue
        n = pooled_n + nb
        delta = mb - pooled_mean
        pooled_mean += delta * nb / n
        pooled_m2 += m2b + delta * delta * pooled_n * nb / n
        pooled_n = n
    variance = max(0.0, pooled_m2 / pooled_n) if pooled_n > 0.0 else 0.0
    return max(math.sqrt(variance), _UCB_SCALE_FLOOR * abs(pooled_mean))


def ucb1_best_arm(arm_stats: dict, arms: tuple[str, ...], *, c: float = _UCB_C) -> str | None:
    """The UCB1-optimal arm (minimizing latency) over `arms`, or `None` if none is tried.

    Deterministic: an untried arm is explored first (lowest name), then the arm with the smallest
    lower-confidence bound `mean - c*sd*sqrt(2*ln N / n)` wins, ties broken by name. No RNG, so a
    plan is reproducible run to run — the "fixed seed" a bandit needs is simply the absence of one.

    `sd` is the pooled reward spread (`_reward_scale`), which is what keeps the radius commensurate
    with the mean it is subtracted from.
    """
    tried = {
        a: s
        for a, s in arm_stats.items()
        if a in arms and isinstance(s, dict) and float(s.get("n", 0.0)) > 0.0
    }
    if not tried:
        return None
    total = sum(float(s["n"]) for s in tried.values())
    untried = sorted(a for a in arms if a not in tried)
    # Give an untried arm a turn once the tried arms have a little evidence — bounded exploration.
    if untried and total >= len(tried):
        return untried[0]
    pooled = _reward_scale(tried)
    best: str | None = None
    best_lcb = math.inf
    for a in sorted(tried):
        s = tried[a]
        n = float(s["n"])
        mean = float(s.get("mean", 0.0))
        # UCB-V: the radius uses **this arm's own** spread, falling back to the pooled one
        # only where a single observation cannot supply it. The pooled spread charges a
        # consistently-fast, low-variance arm the same exploration cost as an erratic one, so
        # a strategy that wins every single time keeps being second-guessed by whichever arm
        # happens to be noisy — which is the opposite of what the noise says.
        arm_variance = _arm_variance(s)
        scale = pooled if arm_variance is None else math.sqrt(arm_variance)
        # Floored against the *pooled* spread, not against the arm's own mean. An arm whose
        # every run came back identical has no observed spread, but "no spread observed" is
        # not "no spread exists", so it still needs some radius. Taking a fraction of what the
        # workload's rewards actually vary by keeps that residual exploration on the same
        # scale as the evidence — and, unlike a fraction of the mean, it stays small enough
        # that a genuinely consistent arm is cheaper to be confident about than an erratic one.
        scale = max(scale, _UCB_SCALE_FLOOR * pooled)
        lcb = mean - c * scale * math.sqrt(2.0 * math.log(max(2.0, total)) / n)
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
        stats = hub.get_keyed_param(scoped(namespace), key) or {}
        total = sum(float(s.get("n", 0.0)) for s in stats.values() if isinstance(s, dict))
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
    hub: MetadataHub | None, signature: str, arms: tuple[str, ...] = JOIN_ARMS
) -> str | None:
    """The measured-fastest join algorithm for this signature, or `None` cold (cost model decides).

    A regret-minimizing bandit over the equivalent algorithm arms: it converges to whichever
    strategy is genuinely fastest on *this* hardware/data, correcting a mis-ranked static cost
    guess. Every arm yields the identical relation, so the choice is result-invariant.
    """
    return learned_arm(hub, _NS_ARM, signature, arms)


# Decision family — execution route (bandit over staged / one-shot re-optimization).
def record_adaptive_route(
    hub: MetadataHub | None, signature: str, route: str, wall_ms: float
) -> None:
    """Record what one run of `signature` cost on the route it took.

    Both routes compute the identical relation — stage-by-stage re-optimization only re-plans
    equivalent algebra — so this is a pure latency choice and the bandit may range over it
    freely.

    **The first observation of each arm is discarded**, and that is not a tuning knob but a
    correction for what it would otherwise be measuring. A shape's first execution on a given
    route pays one-time costs that recur for *neither* route afterwards: sketching the source
    columns the estimator needs, deriving the plan, warming the join-strategy bandit sitting
    under it. On TPC-H q18 at sf10 that is the difference between ~8,000 ms and ~300 ms — 25x,
    swamping the difference between the arms by an order of magnitude. Whichever arm happened
    to run first would carry that penalty forever, so the bandit would be ranking the order the
    arms were tried in rather than the routes themselves. Discarding it symmetrically costs one
    extra run per arm and makes the first *recorded* number a steady-state one.

    Args:
        hub: The metadata hub to record into; `None` is a no-op.
        signature: The plan signature — the bandit key.
        route: The arm that ran (`staged` / `one_shot`).
        wall_ms: Measured wall time for the whole query.
    """
    if hub is None or wall_ms <= 0.0 or not route:
        return
    try:
        spent = dict(hub.get_keyed_param(scoped(_NS_ROUTE_COLD), signature) or {})
        if not spent.get(route):
            spent[route] = 1
            hub.put_keyed_param(scoped(_NS_ROUTE_COLD), signature, spent)
            return
    except Exception as exc:  # pragma: no cover - learning must never break a query
        note_suppressed("kyber", "read the route cold-sample marker", exc)
        return
    record_arm(hub, _NS_ROUTE, signature, route, wall_ms, invalidates_plans=False)


def learned_adaptive_route(hub: MetadataHub | None, signature: str) -> str | None:
    """The measured-cheaper execution route for this signature, or `None` cold.

    Whether to re-optimize between stages is a *cost* question, and it had been answered with
    a proxy: the loop recorded whether any stage's measured cardinality missed its estimate (a
    "flip") and stayed staged where flips were common. The proxy does not answer the question
    in either direction.

    It cannot say *stop*, because an accurate per-stage estimate does not mean the one-shot
    plan would have been the same — the whole point of staging is that the stage boundary is
    where the exact size becomes available at all. And it cannot say *go*, because the
    structural heuristic that turns staging on fires on nearly every multi-join query at scale
    while nothing measured whether it paid. Staging runs one breaker per stage, so it
    materializes every join separately and gives up both operator fusion and the streaming
    executor's width: measured on TPC-H sf10, once the learned statistics are warm, q8 cost
    887 ms staged against 142 ms one-shot, q17 476 against 105, q2 205 against 32, and q5 fell
    to **1.9x parallelism on a 96-core machine** where the one-shot plan reaches 22.6x.

    So the decision is made the way every other genuinely-two-sided one here is made — by
    measuring both arms and minimizing regret. That matters more than the direction the
    current numbers point, because the direction is not a constant: staging is the only
    distributed route for some shapes, it is what earns the statistics a cold shape has not
    learned yet, and the cost of a mis-estimated plan grows with the data. A measured router
    follows those; a hardcoded verdict, in either direction, does not.

    The arms are equivalent — staging only re-plans, never changes the algebra — so the choice
    is result-invariant, and exploration is bounded at roughly one run of the losing arm per
    signature. Raw wall time is the right reward here, unlike the join bandit's size-normalized
    one: the two routes are compared on the *same* query, so a size that drifts over time moves
    both arms together.
    """
    return learned_arm(hub, _NS_ROUTE, signature, _ROUTE_ARMS, min_total=_MIN_ROUTE_TOTAL)
