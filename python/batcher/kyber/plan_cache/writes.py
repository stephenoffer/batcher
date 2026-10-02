"""Advance the learning generation only when a write could change a plan.

Layer: kyber (optimizer).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from batcher._internal.mathx import safe_div
from batcher.kyber import learning

__all__ = ["record_write"]


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
