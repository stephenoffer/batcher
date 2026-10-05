"""`merge_quantile_grids` reads each grid's CDF by bisection, exactly as the linear scan did.

The merge evaluates the mixture CDF thousands of times per call, and `_grid_cdf` was rewritten
from a per-call float conversion plus a linear scan into a bisection over pre-converted lists.
That is a speed change only: the segment chosen, and so the arithmetic, must be the same, which
this holds bit for bit against the scan it replaced -- including on repeated boundary values,
where the *first* matching segment decides the answer.
"""

from __future__ import annotations

import random

import pytest

from batcher.kyber.stats.distribution import _grid_cdf, merge_quantile_grids

pytestmark = pytest.mark.unit


def _scan_cdf(grid, x):
    """The linear-scan reading `_grid_cdf` replaced, kept as the oracle."""
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


def _grid(rng: random.Random, n: int) -> dict:
    # Repeats on purpose: a heavy value spans several quantiles.
    values = sorted(rng.choice([rng.uniform(-50, 50), 7.0, 7.0, 0.0]) for _ in range(n))
    return {"probs": [i / (n - 1) for i in range(n)], "values": values}


@pytest.mark.parametrize("seed", range(20))
def test_bisection_reads_the_same_segment_as_the_scan(seed):
    rng = random.Random(seed)
    grid = _grid(rng, rng.randint(2, 40))
    values = [float(v) for v in grid["values"]]
    probs = [float(p) for p in grid["probs"]]
    points = [*values, values[0] - 1, values[-1] + 1]
    points += [rng.uniform(values[0] - 5, values[-1] + 5) for _ in range(200)]
    for x in points:
        assert _grid_cdf(values, probs, x) == _scan_cdf(grid, x), (grid, x)


def test_the_merge_is_monotone_and_spans_its_branches():
    rng = random.Random(7)
    grids = [_grid(rng, 21), _grid(rng, 33), None]
    merged = merge_quantile_grids(grids, [1000.0, 250.0, 99.0], 17)
    assert merged is not None
    assert merged["values"] == sorted(merged["values"])
    assert merged["values"][0] == min(g["values"][0] for g in grids if g)
    assert merged["values"][-1] <= max(g["values"][-1] for g in grids if g)
