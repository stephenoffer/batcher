"""Hinge and squared-hinge margin losses.

Both are single-pass aggregates over a label and a decision score. `hinge_loss` is checked
against scikit-learn; the squared variant against its closed form, plus the property that a
correctly-classified point with margin at least 1 contributes zero.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt

pytestmark = pytest.mark.unit

sk_metrics = pytest.importorskip("sklearn.metrics")


def _agg(ds: bt.Dataset, expr) -> float:
    return ds.agg(m=expr).to_pydict()["m"][0]


def test_hinge_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    score = rng.normal(0, 1.5, 400)
    ds = bt.from_pydict({"y": y.tolist(), "s": score.tolist()})
    assert _agg(ds, bt.hinge_loss("y", "s")) == pytest.approx(sk_metrics.hinge_loss(y, score))


def test_squared_hinge_matches_the_closed_form() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    score = rng.normal(0, 1.5, 400)
    ds = bt.from_pydict({"y": y.tolist(), "s": score.tolist()})
    signed = 2 * y - 1
    expected = np.mean(np.maximum(0, 1 - signed * score) ** 2)
    assert _agg(ds, bt.squared_hinge_loss("y", "s")) == pytest.approx(expected)


def test_confident_correct_predictions_incur_no_loss() -> None:
    # Positives scored well above +1, negatives well below -1: every margin is satisfied.
    ds = bt.from_pydict({"y": [1, 1, 0, 0], "s": [3.0, 2.0, -3.0, -2.0]})
    assert _agg(ds, bt.hinge_loss("y", "s")) == pytest.approx(0.0)
    assert _agg(ds, bt.squared_hinge_loss("y", "s")) == pytest.approx(0.0)


def test_squared_hinge_punishes_a_deep_miss_harder() -> None:
    # A single point 3 on the wrong side: hinge contributes 4, squared contributes 16.
    ds = bt.from_pydict({"y": [1], "s": [-3.0]})
    assert _agg(ds, bt.hinge_loss("y", "s")) == pytest.approx(4.0)
    assert _agg(ds, bt.squared_hinge_loss("y", "s")) == pytest.approx(16.0)


def test_composes_with_group_by() -> None:
    ds = bt.from_pydict({"g": ["a", "a", "b", "b"], "y": [1, 0, 1, 0], "s": [0.5, -0.5, 2.0, -2.0]})
    out = ds.group_by("g").agg(h=bt.hinge_loss("y", "s")).sort("g").to_pydict()
    assert len(out["h"]) == 2
