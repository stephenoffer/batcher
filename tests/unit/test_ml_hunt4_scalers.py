"""Wave-4 ml bug-hunt: numeric fidelity of the fit/transform preprocessors.

Every fit-then-transform is checked against a hand recomputation with numpy, across
the edge cases (large magnitude, constant column, nulls) where a naive statistic
formula silently diverges.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher.ml.preprocessors import StandardScaler

pytestmark = pytest.mark.unit


def _pop_std(vals: list[float]) -> float:
    return float(np.std(vals))  # population (ddof=0), what StandardScaler targets


def test_standard_scaler_large_magnitude_column_is_numerically_stable() -> None:
    """StandardScaler must not lose the variance to catastrophic cancellation.

    Regression for the naive ``E[x^2] - E[x]^2`` variance: on a large-magnitude
    column the squares overflow float64's 2**53 exact-integer range, so the naive
    formula reported ``std = sqrt(2)`` for values whose true population std is
    ``sqrt(1.25)``. The stable (Welford) variance aggregate gets it right.
    """
    vals = [1e8, 1e8 + 1.0, 1e8 + 2.0, 1e8 + 3.0]
    ds = bt.from_pydict({"x": vals})
    scaler = StandardScaler(["x"]).fit(ds)

    assert scaler.scale_["x"] == pytest.approx(_pop_std(vals), rel=1e-9)
    # The old naive formula produced sqrt(2) here — assert we are NOT that.
    assert abs(scaler.scale_["x"] - np.sqrt(2.0)) > 0.2

    out = scaler.transform(ds).to_pydict()["x"]
    mean = float(np.mean(vals))
    manual = [(v - mean) / _pop_std(vals) for v in vals]
    assert out == pytest.approx(manual, abs=1e-9)


def test_standard_scaler_matches_numpy_across_scales() -> None:
    """Fitted scale_ equals numpy population std for small and large columns alike."""
    for vals in (
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1e9 + i for i in range(10)],
        [-3.0, -1.0, 1.0, 3.0],
    ):
        ds = bt.from_pydict({"x": [float(v) for v in vals]})
        scaler = StandardScaler(["x"]).fit(ds)
        assert scaler.scale_["x"] == pytest.approx(_pop_std(vals), rel=1e-9)


def test_standard_scaler_ignores_nulls_like_numpy() -> None:
    """Nulls are dropped from the statistic (population std over the non-null values)."""
    vals = [1.0, None, 3.0, None, 5.0]
    present = [1.0, 3.0, 5.0]
    ds = bt.from_pydict({"x": vals})
    scaler = StandardScaler(["x"]).fit(ds)
    assert scaler.scale_["x"] == pytest.approx(_pop_std(present), rel=1e-9)


def test_standard_scaler_constant_column_scales_by_one() -> None:
    """A zero-variance column keeps scale 1.0 (centered value, no divide-by-zero)."""
    ds = bt.from_pydict({"x": [5.0, 5.0, 5.0]})
    scaler = StandardScaler(["x"]).fit(ds)
    assert scaler.scale_["x"] == 1.0
    assert scaler.transform(ds).to_pydict()["x"] == [0.0, 0.0, 0.0]


def test_standard_scaler_single_row_scales_by_one() -> None:
    """One-row column has zero variance → scale 1.0 (not a NaN/None crash)."""
    ds = bt.from_pydict({"x": [42.0]})
    scaler = StandardScaler(["x"]).fit(ds)
    assert scaler.scale_["x"] == 1.0
    assert scaler.transform(ds).to_pydict()["x"] == [0.0]
