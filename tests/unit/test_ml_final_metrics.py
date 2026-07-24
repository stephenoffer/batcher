"""The closing batch of error, rate, and dispersion measures.

Each is checked against its numpy/scipy reference: the signed percentage error and normalized
RMSE against their closed forms, the false-negative rate against ``1 - recall``, the mean absolute
deviation against numpy, and the normalized entropy against Shannon entropy over its maximum.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import batcher as bt
from batcher.ml.stats import mean_abs_deviation, normalized_entropy

pytestmark = pytest.mark.unit


def _agg(ds: bt.Dataset, expr) -> float:
    return ds.agg(m=expr).to_pydict()["m"][0]


def test_mean_percentage_error_is_signed() -> None:
    y = np.array([10.0, 20.0, 30.0, 40.0])
    p = np.array([12.0, 18.0, 33.0, 38.0])
    ds = bt.from_pydict({"y": y.tolist(), "p": p.tolist()})
    assert _agg(ds, bt.mean_percentage_error("y", "p")) == pytest.approx(np.mean((y - p) / y))


def test_mean_percentage_error_skips_zero_actuals() -> None:
    ds = bt.from_pydict({"y": [0.0, 10.0, 20.0], "p": [5.0, 11.0, 18.0]})
    # The zero-actual row is dropped from both numerator and denominator.
    expected = np.mean([(10 - 11) / 10, (20 - 18) / 20])
    assert _agg(ds, bt.mean_percentage_error("y", "p")) == pytest.approx(expected)


def test_normalized_rmse_matches_the_closed_form() -> None:
    y = np.array([10.0, 20.0, 30.0, 40.0])
    p = np.array([12.0, 18.0, 33.0, 38.0])
    ds = bt.from_pydict({"y": y.tolist(), "p": p.tolist()})
    expected = np.sqrt(np.mean((y - p) ** 2)) / np.mean(y)
    assert _agg(ds, bt.normalized_rmse("y", "p")) == pytest.approx(expected)


def test_false_negative_rate_is_one_minus_recall() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = rng.integers(0, 2, 300)
    ds = bt.from_pydict({"y": y.tolist(), "p": p.tolist()})
    fnr = _agg(ds, bt.false_negative_rate("y", "p"))
    assert fnr == pytest.approx(1 - _agg(ds, bt.recall("y", "p")))


def test_mean_abs_deviation_matches_numpy() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    ds = bt.from_pydict({"x": x.tolist()})
    assert mean_abs_deviation(ds, "x") == pytest.approx(np.mean(np.abs(x - x.mean())))


def test_normalized_entropy_is_one_for_a_uniform_column() -> None:
    ds = bt.from_pydict({"c": ["a", "a", "b", "b", "c", "c", "d", "d"]})
    assert normalized_entropy(ds, "c") == pytest.approx(1.0)


def test_normalized_entropy_matches_entropy_over_log_k() -> None:
    from batcher.ml.stats import entropy

    ds = bt.from_pydict({"c": ["a", "a", "a", "b", "c"]})
    assert normalized_entropy(ds, "c") == pytest.approx(entropy(ds, "c", base=math.e) / math.log(3))


def test_normalized_entropy_is_nan_for_a_constant_column() -> None:
    ds = bt.from_pydict({"c": ["a", "a", "a"]})
    assert math.isnan(normalized_entropy(ds, "c"))
