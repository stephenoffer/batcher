"""Distributional primitives shared by the cardinality and selectivity estimators.

The estimators above this module ask the same handful of *distributional* questions over
and over: how often does a non-skewed value occur, how many distinct values survive a
selection, how much do two value ranges overlap, how do two histograms join. Answering
each of those in one place — with the sharpest closed form that is actually justified —
is what keeps the row estimator, the column-stat propagator, and the predicate
selectivity model from drifting into three different approximations of the same quantity.

Everything here is pure: statistics in, a number out. The trust/provenance discipline
stays with the callers (`stats.columns`, `stats.estimator`); this module only computes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

__all__ = [
    "distinct_after_selection",
    "geometric_mean",
    "join_match_fraction",
    "mcv_join_rows",
    "merge_quantile_grids",
    "overlap_fraction",
    "residual_eq_frequency",
    "residual_mass",
    "union_ndv",
]


def residual_mass(mcv: Mapping[str, float] | None) -> float:
    """The probability mass *not* covered by a column's most-common-value table.

    `1 - Σ f(v)` over the measured top values, clamped to `[0, 1]`. This is the mass the
    uniformity assumption may legitimately be applied to; applying it to the whole column
    (as a bare `1/ndv` does) double-counts the skew the MCV table already accounts for.

    Args:
        mcv: The measured `{str(value): frequency}` table, or None.

    Returns:
        The residual (non-MCV) probability mass, in `[0, 1]`.
    """
    if not mcv:
        return 1.0
    covered = sum(f for f in mcv.values() if f > 0.0)
    return max(0.0, min(1.0, 1.0 - covered))


def residual_eq_frequency(
    ndv: float | None,
    mcv: Mapping[str, float] | None,
    default: float,
) -> float:
    """`P(col = v)` for a value that is *known not to be* one of the top values.

    The uniform `1/ndv` is the right answer only for a column with no measured skew. Once
    an MCV table exists it says two things: the top `k` values carry `Σ f` of the mass, and
    the remaining `d - k` values share what is left. So a value absent from the table occurs

    ``(1 - Σ f) / (d - k)``

    times, which is strictly below `1/d` whenever the column is skewed at all — often by an
    order of magnitude. Estimating such a value at `1/d` is the single largest systematic
    over-estimate in equality selectivity on real (Zipfian) data, and it compounds: the
    over-estimate flows into join cardinality, build-side sizing, and join order.

    Falls back to `1/d` when there is no MCV table, and to `default` when the distinct count
    is unknown. Degenerate cases (the MCV table covering every distinct value, or all of the
    mass) return the smallest positive frequency the table itself implies rather than 0, so a
    predicate on a value the table happens not to list never estimates to exactly nothing.

    Args:
        ndv: The column's distinct count, if known.
        mcv: The measured most-common-value table, if any.
        default: The cold-start equality selectivity to use when `ndv` is unknown.

    Returns:
        The estimated frequency of a non-most-common value, in `[0, 1]`.
    """
    if not ndv or ndv <= 0:
        return default
    uniform = 1.0 / ndv
    if not mcv:
        return uniform
    remaining_values = ndv - len(mcv)
    residual = residual_mass(mcv)
    if remaining_values <= 0.0 or residual <= 0.0:
        # The table already enumerates the whole column. A value outside it is then at most
        # as frequent as the rarest value the table lists, which is a far better bound than
        # the uniform average and stays strictly positive.
        return min(uniform, min(mcv.values(), default=uniform))
    return min(uniform, residual / remaining_values)


def distinct_after_selection(ndv: float, rows_in: float, rows_out: float) -> float:
    """Expected distinct values surviving a selection that keeps `rows_out` of `rows_in`.

    Carrying a column's distinct count unchanged through a filter is wrong in the direction
    that matters: a 1%-selective predicate over a 10M-row table with 1M distinct values does
    *not* leave 1M distinct values in the 100K surviving rows — it leaves about 95K. The
    error then propagates into every group-by, `DISTINCT`, and join estimate above the
    filter, and it grows with how selective the filter is.

    The classical answer is Yao's formula: drawing `k` of `n` rows without replacement from
    `d` equal-sized value groups leaves

    ``d · (1 - C(n - n/d, k) / C(n, k))``

    distinct values in expectation. Cardenas' approximation replaces the ratio of binomials
    with ``(1 - k/n)^(n/d)``, which is accurate to well under a percent across the whole
    range and needs no factorials:

    ``d · (1 - (1 - k/n)^(n/d))``

    Both limits are the ones intuition demands — the full relation keeps all `d` values, and
    a single surviving row keeps exactly one — and the result is always bounded by
    `min(d, k)`, the two trivial bounds. Uses the exact `-expm1(x·log1p(-s))` form so a
    highly selective predicate (`s` near 0) does not lose the answer to cancellation.

    Args:
        ndv: The distinct count before the selection.
        rows_in: Rows before the selection.
        rows_out: Rows after the selection.

    Returns:
        The expected surviving distinct count, in `[1, min(ndv, rows_out)]`.
    """
    if ndv <= 0.0 or rows_in <= 0.0 or rows_out <= 0.0:
        return 0.0
    ceiling = min(ndv, rows_out)
    if rows_out >= rows_in:
        return ceiling
    selectivity = rows_out / rows_in
    rows_per_value = rows_in / ndv
    if rows_per_value <= 1.0:
        # Every value is unique (or nearly so), so the surviving distinct count is just the
        # surviving row count — Cardenas degenerates to exactly that, but stating it avoids
        # the pow() entirely for the common unique-key case.
        return max(1.0, ceiling)
    # -expm1(x) == 1 - e^x, evaluated without cancellation for small x.
    survived = -math.expm1(rows_per_value * math.log1p(-selectivity))
    return max(1.0, min(ceiling, ndv * survived))


def union_ndv(ndvs: Sequence[float], rows: float | None = None) -> float | None:
    """Distinct count of a union of branches, from each branch's distinct count.

    The Fréchet bounds are ``max_i d_i <= d_union <= Σ_i d_i``: total overlap at one end,
    disjoint value sets at the other. Dropping the statistic entirely (the alternative) is
    not neutral — a relation with no ndv makes every join above it fall back to
    ``max(|L|, |R|)``, so a `UNION ALL` of two partitions blinds the rest of the plan.

    The estimate models the branches as independent subsets of a common domain of size
    ``D = max_i d_i``, so folding a branch in gives ``|A or B| = |A| + |B| - |A|·|B|/D``.
    That is the inclusion-exclusion identity under independence, it reproduces both Fréchet
    bounds at the extremes, and it stays below `Σ d_i`, which keeps it on the safe side: a
    *smaller* ndv raises a downstream join estimate (over-budget) rather than deflating it.

    Args:
        ndvs: Each branch's distinct count. Non-positive entries are ignored.
        rows: The union's row count, which caps the result when known.

    Returns:
        The estimated distinct count, or None when no branch has one.
    """
    known = [d for d in ndvs if d and d > 0.0]
    if not known:
        return None
    domain = max(known)
    combined = 0.0
    for d in known:
        combined = combined + d - (combined * d / domain)
    if rows is not None:
        # `rows == 0` is a *known* row count, not a missing one, and an empty union has no
        # distinct values. The `rows > 0` guard skipped the cap there and the `max(1.0, ...)`
        # floor then reported at least one, so `union_ndv([1e9], 0)` returned 1e9 while every
        # positive `rows` capped correctly. No caller can reach it today -- `columns.py` passes
        # `total_rows or None`, and `estimator.py` only sees `total == 0` when every branch is
        # empty, which returns None above -- but the parameter is documented as capping whenever
        # the count is known, and a floor of one distinct value is only right for a non-empty
        # relation.
        if rows <= 0.0:
            return 0.0
        combined = min(combined, rows)
    return max(1.0, combined)


def overlap_fraction(
    a: tuple[float, float] | None,
    b: tuple[float, float] | None,
) -> float | None:
    """The fraction of `a`'s value range that also lies inside `b`'s.

    Two join keys whose `[min, max]` ranges only partly overlap can only match inside the
    intersection. Disjointness (fraction 0) is the case the estimator already special-cases;
    *partial* overlap is the common one — a fact table spanning three years joined to a
    dimension covering one — and treating it as full overlap over-estimates the join by the
    reciprocal of this fraction.

    Both ranges must already be mapped to a common ordinal. A degenerate (single-point)
    range is treated as fully overlapping when the point lies inside the other range, which
    is exact.

    Args:
        a: The `(min, max)` range whose fraction is measured.
        b: The range it is intersected with.

    Returns:
        The overlapping fraction of `a`, in `[0, 1]`, or None when either range is unknown.
    """
    if a is None or b is None:
        return None
    a_lo, a_hi = a
    b_lo, b_hi = b
    if a_hi < a_lo or b_hi < b_lo:
        return None
    lo = max(a_lo, b_lo)
    hi = min(a_hi, b_hi)
    if hi < lo:
        return 0.0
    width = a_hi - a_lo
    if width <= 0.0:
        return 1.0  # a is a single point, and it lies inside b
    return max(0.0, min(1.0, (hi - lo) / width))


def join_match_fraction(left_ndv: float, right_ndv: float) -> float:
    """The fraction of the left side's distinct keys that find a match on the right.

    The containment assumption — the smaller key domain is a subset of the larger — gives
    ``min(1, d_R / d_L)``. It is what semi/anti/outer estimation is built on, and stating it
    once keeps semi and anti complementary: an anti-join keeps exactly the left rows a
    semi-join drops, so both must be derived from the same fraction.

    Args:
        left_ndv: Distinct keys on the left.
        right_ndv: Distinct keys on the right.

    Returns:
        The matching fraction, in `[0, 1]`.
    """
    if left_ndv <= 0.0:
        return 1.0
    return max(0.0, min(1.0, right_ndv / left_ndv))


def mcv_join_rows(
    left_rows: float,
    right_rows: float,
    left_mcv: Mapping[str, float] | None,
    right_mcv: Mapping[str, float] | None,
    left_ndv: float | None,
    right_ndv: float | None,
) -> float | None:
    """Equi-join output rows, decomposing the key distribution into skew and residual.

    Selinger's ``|L|·|R| / max(d_L, d_R)`` assumes both key columns are uniform. On a skewed
    key that is not a small error: one value held by 47% of the left rows and 30% of the
    right produces ``0.14·|L|·|R|`` output rows *by itself*, which the uniform estimate
    misses entirely — and a hash join sized from it OOMs.

    The output size is ``|L|·|R|·Σ_v p_L(v)·p_R(v)`` over the shared key domain, where `p` is
    a value's per-row probability. Modelling each side as "the measured top values, plus a
    uniform residual over the values the table does not list" splits that sum into four terms,
    all of which are needed:

    * **both** sides list `v` — ``f_L(v)·f_R(v)``, an exact contribution, since both
      frequencies were measured;
    * **left only** — the value is not among the right's heavy hitters, so it draws from the
      right's residual: ``f_L(v)·m_R/n_R``, with ``n = d - k`` unlisted values holding mass `m`;
    * **right only** — the mirror image;
    * **neither** — ``min(n_L, n_R)`` residual values coincide under containment, each
      contributing ``(m_L/n_L)·(m_R/n_R)``, which sums to ``m_L·m_R / max(n_L, n_R)``.

    Dropping the two cross terms (the first attempt here) under-counts a join whose skew is
    *anti-correlated* — a hot left value that is rare on the right still joins against every
    one of its right-side rows — and that shape is common: a fact table skewed toward one
    customer joined to a dimension skewed toward a different one.

    The model's one real assumption is that a value listed on one side but not the other has
    only residual frequency on that side. A frequency table records heavy hitters, so a value
    just below the threshold is priced slightly low; that is the standard trade every
    MCV-based join estimator makes, and it is far smaller than the error from assuming
    uniformity over a skewed key.

    Args:
        left_rows: Left input rows.
        right_rows: Right input rows.
        left_mcv: Left key's measured frequency table.
        right_mcv: Right key's measured frequency table.
        left_ndv: Left key's distinct count.
        right_ndv: Right key's distinct count.

    Returns:
        The estimated output rows, or None when either side has no usable MCV table.
    """
    if not left_mcv or not right_mcv:
        return None
    m_left, m_right = residual_mass(left_mcv), residual_mass(right_mcv)
    n_left = _residual_domain(left_ndv, len(left_mcv))
    n_right = _residual_domain(right_ndv, len(right_mcv))
    # Per-value probability inside each side's residual, 0 when the MCV table already
    # enumerates the whole domain (there is then nothing left for an unlisted value).
    unit_left = m_left / n_left if n_left else 0.0
    unit_right = m_right / n_right if n_right else 0.0

    probability = 0.0
    for value, f_left in left_mcv.items():
        f_right = right_mcv.get(value)
        probability += f_left * (f_right if f_right is not None else unit_right)
    for value, f_right in right_mcv.items():
        if value not in left_mcv:
            probability += f_right * unit_left
    if n_left and n_right:
        probability += m_left * m_right / max(n_left, n_right)
    return min(probability * left_rows * right_rows, left_rows * right_rows)


def _residual_domain(ndv: float | None, listed: int) -> float:
    """How many distinct values a column has outside its most-common-value table."""
    if ndv is None:
        return 0.0
    return max(0.0, ndv - listed)


def merge_quantile_grids(
    grids: Sequence[Mapping[str, Any] | None],
    weights: Sequence[float],
    points: int = 11,
) -> dict[str, list[float]] | None:
    """Merge per-branch quantile grids into one, weighted by branch row count.

    A union's CDF is the row-weighted mixture of its branches' CDFs — ``F(x) = Σ nᵢ Fᵢ(x) / N``
    — which is an *exact* identity, not an approximation. Only the re-gridding is approximate:
    the mixture is evaluated on a uniform probability grid by inverting it with a bisection on
    the value axis, so the result is a quantile grid of the same shape the branches carry.

    Dropping quantiles at a union (the alternative) means a range predicate above a
    partition-union falls back from histogram interpolation to a flat constant, which is
    precisely where a partitioned fact table needs it most.

    Args:
        grids: Each branch's `{"probs": [...], "values": [...]}` grid, or None.
        weights: Each branch's row count, positionally aligned with `grids`.
        points: How many probability points the merged grid carries.

    Returns:
        The merged grid, or None when no branch has a usable one.
    """
    usable = [
        (g, w)
        for g, w in zip(grids, weights, strict=False)
        if g and w > 0.0 and len(g.get("values", ())) >= 2
    ]
    if not usable:
        return None
    total = sum(w for _, w in usable)
    lo = min(float(g["values"][0]) for g, _ in usable)
    hi = max(float(g["values"][-1]) for g, _ in usable)
    if not math.isfinite(lo) or not math.isfinite(hi) or hi < lo:
        return None

    def mixture_cdf(x: float) -> float:
        return sum(w * _grid_cdf(g, x) for g, w in usable) / total

    probs = [i / (points - 1) for i in range(points)]
    values = [_invert_monotone(mixture_cdf, p, lo, hi) for p in probs]
    # The inverse is monotone by construction, but bisection on a flat region can emit a
    # non-monotone pair at the resolution limit; a running max restores the invariant the
    # interpolator relies on.
    for i in range(1, len(values)):
        values[i] = max(values[i], values[i - 1])
    return {"probs": probs, "values": values}


def _grid_cdf(grid: Mapping[str, Any], x: float) -> float:
    """`P(v <= x)` from one quantile grid, linearly interpolated between boundaries."""
    values = [float(v) for v in grid["values"]]
    probs = [float(p) for p in grid["probs"]]
    if x <= values[0]:
        return 0.0 if x < values[0] else probs[0]
    if x >= values[-1]:
        return 1.0
    for i in range(len(values) - 1):
        lo, hi = values[i], values[i + 1]
        if lo <= x <= hi:
            if hi == lo:
                return probs[i]
            return probs[i] + (x - lo) / (hi - lo) * (probs[i + 1] - probs[i])
    return 1.0


def _invert_monotone(cdf, target: float, lo: float, hi: float, iterations: int = 40) -> float:
    """The smallest `x` in `[lo, hi]` with `cdf(x) >= target`, by bisection."""
    if target <= 0.0:
        return lo
    if target >= 1.0:
        return hi
    for _ in range(iterations):
        mid = lo + (hi - lo) / 2.0
        if cdf(mid) < target:
            lo = mid
        else:
            hi = mid
    return hi


def geometric_mean(values: Iterable[float]) -> float | None:
    """The geometric mean of strictly positive values, computed in log space.

    The right average for a *ratio* — a q-error, a correction factor, a speedup. An
    arithmetic mean of ratios is asymmetric: over-estimating by 4x and under-estimating by 4x
    average to 2.125 rather than 1, so a sequence of symmetric errors accumulates a bias
    toward over-correction. The log-space form also avoids the overflow a running product
    reaches after a few dozen samples.

    Args:
        values: Strictly positive ratios.

    Returns:
        Their geometric mean, or None when no value is positive.
    """
    logs = [math.log(v) for v in values if v > 0.0]
    if not logs:
        return None
    return math.exp(sum(logs) / len(logs))
