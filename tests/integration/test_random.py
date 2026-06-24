"""`Dataset.with_random` — seeded, reproducible pseudo-random columns.

Keyed by (seed, row index), so it is deterministic (no wall-clock seed) and stable
across runs — verified by distribution moments rather than a DuckDB oracle.
"""

from __future__ import annotations

import statistics

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.integration


def _many(n: int = 20000):
    return bt.from_arrow(pa.table({"v": list(range(n))}))


def test_uniform_reproducible_and_in_range():
    ds = bt.from_pydict({"x": [1, 2, 3, 4, 5]})
    a = ds.with_random(seed=7).to_pydict()["random"]
    b = ds.with_random(seed=7).to_pydict()["random"]
    assert a == b  # same seed → identical
    assert all(0.0 <= v < 1.0 for v in a)
    assert ds.with_random(seed=8).to_pydict()["random"] != a  # different seed differs


def test_uniform_distribution_moments():
    u = _many().with_random("u", seed=1).collect().to_pydict()["u"]
    assert abs(statistics.mean(u) - 0.5) < 0.02  # uniform mean 0.5
    assert abs(statistics.pstdev(u) - (1 / 12) ** 0.5) < 0.02  # uniform stdev ~0.289


def test_normal_distribution_moments():
    z = _many().with_random("z", seed=2, normal=True).collect().to_pydict()["z"]
    assert abs(statistics.mean(z)) < 0.05  # standard normal mean 0
    assert abs(statistics.pstdev(z) - 1.0) < 0.05  # standard normal stdev 1


def test_random_leaves_no_helper_column():
    out = bt.from_pydict({"x": [1, 2]}).with_random().to_pydict()
    assert set(out) == {"x", "random"}  # the internal row-index helper is dropped
