"""An OLS two-line crossover — where one algorithm overtakes another, learned from timings.

The exact machinery `gpu/adaptive.py` uses: fit `t ~ a + b*x` per algorithm from O(1) sufficient
statistics, then solve for the x (build bytes or build rows) where the cheaper-below algorithm is
overtaken by the cheaper-above one. The result is clamped to a band around the shipped default, so
one noisy early fit cannot send a threshold to an absurd value, and an ambiguous fit yields `None`
and the caller keeps that default.

Its two callers are the broadcast byte threshold and the sort-merge row crossover. The family
contract is in the package docstring.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from batcher._internal.logging import note_suppressed
from batcher.config import active_config
from batcher.kyber import plan_cache
from batcher.kyber.ols import fit_ols, ols_update
from batcher.metadata.hardware_scope import scoped

if TYPE_CHECKING:
    from batcher.metadata import MetadataHub

__all__ = [
    "learned_broadcast_max_bytes",
    "learned_sort_merge_min_rows",
    "record_broadcast_timing",
    "record_sort_merge_timing",
]

_NS_BCAST = "tuning.broadcast_xover"  # OLS buckets: "broadcast" vs "shuffle", x = build bytes
_NS_SMJ = "tuning.sortmerge_xover"  # OLS buckets: "hash" vs "sort_merge", x = build rows

# The band a learned threshold is clamped to around its default. (The sample-count and
# spread gates that decide whether a fit is trustworthy at all live in `kyber.ols`.)
_BAND = 8.0


# Reusable primitive 2 — an OLS two-line crossover (generalizing gpu/adaptive.py).
def _fold_ols(
    hub: MetadataHub,
    namespace: str,
    bucket: str,
    x: float,
    y: float,
    *,
    below: str = "",
    above: str = "",
) -> None:
    """Fold one `(x, y)` observation into a bucket's fitted line. Best-effort.

    Scoped to the machine class. A crossover is the row count at which one strategy overtakes
    another, and that point is a ratio of two per-row costs — so it moves with the hardware
    even when the ranking does not. Learn "the GPU wins above 200k rows" on an A100 beside a
    slow host CPU, and the same fit is badly wrong on a fast CPU beside a T4.

    `below`/`above` name the crossover this bucket feeds, so the plan cache is invalidated on
    **the threshold moving** rather than on the fit's accumulators drifting. A plan reads the
    threshold and nothing else, and the accumulators move on every timed join, so the drift
    test invalidated *every memoized plan in the process* — the learned generation is global —
    on a run that changed no decision at all.
    """
    s = hub.get_keyed_param(scoped(namespace), bucket) or {}
    updated = ols_update(s, x, y)
    decides = None
    if below and above:

        def decides(candidate: object, _b: str = below, _a: str = above) -> object:
            return _crossover_step(hub, namespace, bucket, candidate, _b, _a)

    plan_cache.record_write(hub, scoped(namespace), bucket, updated, decides=decides)


#: Relative step at which a moved crossover counts as a different decision. The same
#: materiality the rest of the learning loop uses (`learning.is_material_change`), applied to
#: a *continuous* threshold: quantizing it to geometric steps of this size is what turns
#: "the number moved" into "the number left the band a plan was chosen in".
_XOVER_STEP = 0.10


def _crossover_step(
    hub: MetadataHub, namespace: str, bucket: str, candidate: object, below: str, above: str
) -> object:
    """The crossover this fit implies, quantized to `_XOVER_STEP` — or `None` when unusable.

    `candidate` stands in for `bucket`'s stored value, so the decision is evaluated against
    what the store *would* hold. Both buckets are read, because a crossover is a property of
    the pair.

    Deliberately **unclamped**, where the reader clamps to a band around its default. The
    reader's clamp is monotone, so the unclamped value moves whenever the clamped one does:
    reading it here can only invalidate a plan the clamp would have spared, never miss one it
    would not. Over-invalidating costs a re-plan; under-invalidating serves a stale plan, and
    this side of that trade is the safe one — which is also what lets this stay free of the
    rule module that owns the default.
    """
    try:
        fits = {
            name: _fit(
                (candidate if name == bucket else hub.get_keyed_param(scoped(namespace), name))
                or {}
            )
            for name in (below, above)
        }
    except Exception:  # pragma: no cover - a decision test must never break a query
        return None
    xover = _crossover_of(fits.get(below), fits.get(above), None)
    if xover is None or xover <= 0.0:
        return None
    return math.floor(math.log(xover) / math.log(1.0 + _XOVER_STEP))


_fit = fit_ols


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
        below = _fit(hub.get_keyed_param(scoped(namespace), cheap_below) or {})
        above = _fit(hub.get_keyed_param(scoped(namespace), cheap_above) or {})
    except Exception:  # pragma: no cover
        return None
    return _crossover_of(below, above, default)


def _crossover_of(
    below: tuple[float, float] | None,
    above: tuple[float, float] | None,
    default: float | None,
) -> float | None:
    """Where two fitted lines cross, clamped to the band around `default`, or `None`.

    Split from `_solve_crossover` so the plan cache's decision test can evaluate the same
    threshold against a *candidate* fit it has not written yet. `default=None` skips the
    clamp — see `_crossover_step` for why that direction is the safe one.
    """
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
    if default is None:
        return xover
    return min(max(xover, default / _BAND), default * _BAND)


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
        _fold_ols(
            hub,
            _NS_BCAST,
            strategy,
            float(build_bytes),
            float(wall_ms),
            below="broadcast",
            above="shuffle",
        )
    except Exception as exc:  # pragma: no cover - a learned write must never break a query
        # Noted, not swallowed. This recorder had *no caller at all* until recently, so
        # `learned_broadcast_max_bytes` returned `None` forever and the threshold never moved
        # off its static default — with nothing in the log to say so. A silent failure here
        # reproduces exactly that state, and it is what `note_suppressed` exists to separate:
        # "this optimization did not apply" against "it has been broken since March".
        note_suppressed("kyber", "record a broadcast-vs-shuffle timing", exc)


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
        _fold_ols(
            hub,
            _NS_SMJ,
            strategy,
            float(build_rows),
            float(wall_ms),
            below="hash",
            above="sort_merge",
        )
    except Exception as exc:  # pragma: no cover - a learned write must never break a query
        note_suppressed("kyber", "record a hash-vs-sort-merge timing", exc)


def learned_sort_merge_min_rows(hub: MetadataHub | None, default: float) -> float | None:
    """The measured build-row crossover above which sort-merge beats hash, or `None`.

    Hash wins for a small build (cheap hash table) but degrades as the build strains memory, so it
    is the cheaper-below arm; sort-merge's bounded-memory merge is cheaper-above. Solving the
    crossover learns `SORT_MERGE_MIN_ROWS` from real hash-vs-SMJ timings.
    """
    return _solve_crossover(hub, _NS_SMJ, "hash", "sort_merge", default)
