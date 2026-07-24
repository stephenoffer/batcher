"""The geometric-mean classification score and the mean absolute scaled error.

Both are pinned to their closed forms over numpy: the geometric mean to
``sqrt(recall * specificity)`` from the confusion counts, and MASE to the model MAE over the
seasonal-naive MAE. MASE is fed a shuffled input to prove it honors the time order.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher.ml.timeseries import mean_absolute_scaled_error

pytestmark = pytest.mark.unit


def test_geometric_mean_matches_the_confusion_counts() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = rng.integers(0, 2, 400)
    ds = bt.from_pydict({"y": y.tolist(), "p": p.tolist()})
    tp = int(((y == 1) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    ref = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))
    got = ds.agg(g=bt.geometric_mean_score("y", "p")).to_pydict()["g"][0]
    assert got == pytest.approx(ref)


def test_geometric_mean_is_zero_when_a_class_is_never_recalled() -> None:
    # Predict everything positive: specificity is 0, so the geometric mean collapses to 0.
    ds = bt.from_pydict({"y": [1, 1, 0, 0], "p": [1, 1, 1, 1]})
    assert ds.agg(g=bt.geometric_mean_score("y", "p")).to_pydict()["g"][0] == pytest.approx(0.0)


def test_geometric_mean_is_one_for_a_perfect_classifier() -> None:
    ds = bt.from_pydict({"y": [1, 0, 1, 0], "p": [1, 0, 1, 0]})
    assert ds.agg(g=bt.geometric_mean_score("y", "p")).to_pydict()["g"][0] == pytest.approx(1.0)


@pytest.mark.parametrize("seasonality", [1, 7])
def test_mase_matches_the_reference(seasonality: int) -> None:
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0, 1, 200)) + 50
    p = y + rng.normal(0, 0.5, 200)
    ds = bt.from_pydict({"t": list(range(200)), "y": y.tolist(), "p": p.tolist()})
    numerator = np.mean(np.abs(y - p))
    denominator = np.mean(np.abs(y[seasonality:] - y[:-seasonality]))
    got = mean_absolute_scaled_error(ds, "y", "p", order_by="t", seasonality=seasonality)
    assert got == pytest.approx(numerator / denominator)


def test_mase_is_zero_for_a_perfect_forecast() -> None:
    ds = bt.from_pydict({"t": [0, 1, 2, 3], "y": [1.0, 2.0, 3.0, 4.0], "p": [1.0, 2.0, 3.0, 4.0]})
    assert mean_absolute_scaled_error(ds, "y", "p", order_by="t") == pytest.approx(0.0)


def test_mase_honors_the_time_order() -> None:
    rng = np.random.default_rng(2)
    n = 100
    y = np.cumsum(rng.normal(0, 1, n)) + 20
    p = y + rng.normal(0, 0.3, n)
    order = rng.permutation(n)
    ds = bt.from_pydict({"t": order.tolist(), "y": y[order].tolist(), "p": p[order].tolist()})
    numerator = np.mean(np.abs(y - p))
    denominator = np.mean(np.abs(np.diff(y)))
    assert mean_absolute_scaled_error(ds, "y", "p", order_by="t") == pytest.approx(
        numerator / denominator
    )
