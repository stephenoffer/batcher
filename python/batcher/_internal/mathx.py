"""Small, exact numeric helpers shared across every subsystem — the one home for the idioms.

These are the arithmetic one-liners that were being retyped inline in `kyber`, `core`,
`carbonite`, `dist`, and `ml`: clamp a fraction into [0, 1], divide with a zero-denominator
fallback, ceil-divide a count into shards, test an IR literal for NaN. Each is trivial to
write and just as trivial to get subtly wrong — a swapped `min`/`max`, a missing
zero-guard, `math.ceil` on floats where integer ceil-division was meant — and because the
four layer-3 subsystems cannot import one another, the only way the idiom was being shared
was copy-paste. This module is layer 0 (`_internal`), so every layer can import it and the
idiom lives once.

Every function here is pure, dependency-free, and defined to match the exact behavior of
the inline form it replaces, so a call-site migration is behavior-preserving.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import median
from typing import TypeVar, overload

_Number = TypeVar("_Number", int, float)

__all__ = [
    "blend",
    "ceil_div",
    "clamp",
    "clamp01",
    "clamp_factor",
    "is_concentrated",
    "is_nan",
    "safe_div",
]


def clamp(value: _Number, low: _Number, high: _Number) -> _Number:
    """`value` bounded to `[low, high]` — i.e. ``max(low, min(high, value))``.

    The `max`-outer form is deliberate and matches the inline idiom it replaces: when
    ``low <= high`` it is equivalent to the `min`-outer spelling for every non-NaN input.
    """
    return max(low, min(high, value))


def clamp01(value: float) -> float:
    """`value` bounded to the unit interval ``[0.0, 1.0]``.

    The canonical clamp for a probability, selectivity, or fraction.
    """
    return max(0.0, min(1.0, value))


def clamp_factor(value: float, reference: float, factor: float) -> float:
    """`value` bounded to within a multiplicative `factor` of `reference`.

    Bounds `value` to ``[reference / factor, reference * factor]`` — the shape used to keep a
    learned correction, a calibrated estimate, or a memory prediction from straying more than
    `factor`-fold away from a trusted reference. `factor` must be ``>= 1``.
    """
    return max(reference / factor, min(reference * factor, value))


@overload
def safe_div(numerator: float, denominator: float, default: float = ...) -> float: ...
@overload
def safe_div(numerator: float, denominator: float, default: None) -> float | None: ...
def safe_div(numerator: float, denominator: float, default: float | None = 0.0) -> float | None:
    """`numerator / denominator`, or `default` when `denominator` is zero (or otherwise falsy).

    Guards the ubiquitous ratio-with-empty-denominator case (hit rate, selectivity, average)
    without a scattered ``if d else 0.0`` at every call site. `default` may be `None` for callers
    that need "no ratio" to be distinguishable from zero; the return type tracks that per overload.
    """
    return numerator / denominator if denominator else default


def ceil_div(numerator: int, denominator: int) -> int:
    """Integer ceiling division: the smallest ``k`` with ``k * denominator >= numerator``.

    Uses ``-(-numerator // denominator)`` — exact for integers, and free of the float rounding
    that ``math.ceil(numerator / denominator)`` can introduce for large counts. The standard
    way to size ``ceil(rows / per_shard)`` shards.
    """
    return -(-numerator // denominator)


def is_nan(value: object) -> bool:
    """Whether `value` is a float NaN. A non-float is never NaN, so it is safe on any object.

    The precise test for "is this IR literal / bookkeeping scalar an actual NaN", where a bare
    ``math.isnan`` would raise on the `int`, `str`, or `None` a literal slot can also hold.
    """
    return isinstance(value, float) and math.isnan(value)


def blend(prior: float, observed: float, alpha: float) -> float:
    """A static exponential blend of `observed` into `prior` with weight `alpha`.

    Returns ``alpha * observed + (1 - alpha) * prior``. This is the *fixed-weight* EWMA step
    (a new reading always carries weight `alpha`); it is distinct from the observation-count
    decay used when an estimate must converge from cold. `alpha` is in ``[0, 1]``.
    """
    return alpha * observed + (1.0 - alpha) * prior


def is_concentrated(xs: Sequence[float], max_rel_spread: float) -> bool:
    """Whether `xs` cluster tightly enough that their centre means something.

    A confidence gate for a learned value fitted from repeated observations. One sample is
    trivially concentrated; otherwise the full spread must sit within `max_rel_spread` of the
    median. Constant samples always pass, and a non-positive median (nothing measured) passes
    rather than blocking, so the gate can only ever *withhold* a value it has reason to doubt.

    It exists because a mean is only a summary when the thing summarized is one thing. Two
    populations accidentally sharing a key — the same operator shape measured over two very
    different relations, say — average to a number that is wrong for both and is *worse* than
    the structural estimate it replaces. A wide spread is the signature of that, whatever
    caused it.

    Args:
        xs: The observations.
        max_rel_spread: Permitted `(max - min)` as a multiple of the median.

    Returns:
        `True` when the samples are concentrated enough to summarize.
    """
    if len(xs) <= 1:
        return True
    mid = median(xs)
    return mid <= 0.0 or (max(xs) - min(xs)) <= max_rel_spread * mid
