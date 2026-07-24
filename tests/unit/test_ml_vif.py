"""Variance inflation factor.

VIF is read off the diagonal of the inverted correlation matrix; the test checks it equals the
independent regression definition (``1 / (1 - R^2)`` from regressing each column on the others)
and that it flags a deliberately collinear column.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher.ml.stats import variance_inflation_factor

pytestmark = pytest.mark.unit


def _vif_by_regression(x: np.ndarray, j: int) -> float:
    y = x[:, j]
    others = np.delete(x, j, axis=1)
    design = np.column_stack([others, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    r2 = 1 - np.sum(residual**2) / np.sum((y - y.mean()) ** 2)
    return 1 / (1 - r2)


def test_matches_the_regression_definition() -> None:
    rng = np.random.default_rng(0)
    x0 = rng.normal(0, 1, 500)
    x1 = rng.normal(0, 1, 500)
    x2 = x0 + x1 + rng.normal(0, 0.3, 500)
    x = np.column_stack([x0, x1, x2])
    ds = bt.from_pydict({f"x{i}": x[:, i].tolist() for i in range(3)})
    got = variance_inflation_factor(ds, ["x0", "x1", "x2"])
    for j, name in enumerate(["x0", "x1", "x2"]):
        assert got[name] == pytest.approx(_vif_by_regression(x, j), rel=1e-6)


def test_flags_a_collinear_column() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 300)
    b = a + rng.normal(0, 0.05, 300)  # nearly a copy of a
    c = rng.normal(0, 1, 300)  # independent
    ds = bt.from_pydict({"a": a.tolist(), "b": b.tolist(), "c": c.tolist()})
    vif = variance_inflation_factor(ds, ["a", "b", "c"])
    assert vif["a"] > 10
    assert vif["c"] < 2


def test_independent_columns_have_vif_near_one() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, (500, 3))
    ds = bt.from_pydict({f"x{i}": x[:, i].tolist() for i in range(3)})
    vif = variance_inflation_factor(ds, ["x0", "x1", "x2"])
    assert all(v == pytest.approx(1.0, abs=0.2) for v in vif.values())
