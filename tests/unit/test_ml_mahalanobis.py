"""Multivariate Mahalanobis outlier scoring.

Checked cell for cell against `scipy.spatial.distance.mahalanobis` with the sample covariance,
and pinned to the property that motivates it: a row unremarkable on each column alone but far
from the joint center scores highest.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError
from batcher.ml.outliers import mahalanobis_distance

pytestmark = pytest.mark.unit

scipy_distance = pytest.importorskip("scipy.spatial.distance")


def test_matches_scipy() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (300, 3)) @ rng.normal(0, 1, (3, 3))
    ds = bt.from_pydict({f"c{i}": x[:, i].tolist() for i in range(3)})
    got = np.array(mahalanobis_distance(ds, ["c0", "c1", "c2"]).to_pydict()["mahalanobis"])
    mean = x.mean(0)
    inverse = np.linalg.pinv(np.cov(x.T))
    ref = np.array([scipy_distance.mahalanobis(row, mean, inverse) for row in x])
    assert np.allclose(got, ref, atol=1e-8)


def test_flags_a_joint_outlier_the_margins_miss() -> None:
    # Each column alone is ordinary, but the last row breaks the x-y correlation.
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 3.0]
    y = [1.0, 2.0, 3.0, 4.0, 5.0, -3.0]
    ds = bt.from_pydict({"x": x, "y": y})
    scored = mahalanobis_distance(ds, ["x", "y"]).to_pydict()["mahalanobis"]
    assert scored[5] == max(scored)


def test_appends_a_named_column() -> None:
    ds = bt.from_pydict({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 4.0]})
    out = mahalanobis_distance(ds, ["a", "b"], output_column="score")
    assert "score" in out.columns and out.count() == 3


def test_names_a_missing_column() -> None:
    ds = bt.from_pydict({"a": [1.0, 2.0]})
    with pytest.raises(ColumnNotFoundError):
        mahalanobis_distance(ds, ["a", "nope"])


def test_distance_is_nonnegative() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (100, 2))
    ds = bt.from_pydict({"a": x[:, 0].tolist(), "b": x[:, 1].tolist()})
    scored = mahalanobis_distance(ds, ["a", "b"]).to_pydict()["mahalanobis"]
    assert all(v >= 0 for v in scored)
