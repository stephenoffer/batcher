"""The preprocessing surface is reachable, and the leakage-free workflow it exists for works.

The `Preprocessor` family lived in `batcher.ml.preprocessors` without being re-exported
from `batcher.ml`, so the documented surface could not reach it. These pin the export
and the workflow it serves: split, `fit` on **train only**, `transform` both parts with
that same state. Fitting on the whole frame would leak test statistics into the training
features — the check below is that `transform` uses the train-fitted state, not the
statistics of whatever frame it is handed.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytestmark = pytest.mark.unit


def test_preprocessors_are_reexported_from_batcher_ml():
    import batcher.ml as ml
    import batcher.ml.preprocessors as pp

    missing = set(pp.__all__) - set(ml.__all__)
    assert not missing, f"not re-exported from batcher.ml: {sorted(missing)}"
    for name in pp.__all__:
        assert getattr(ml, name) is getattr(pp, name)


def test_standard_scaler_transform_uses_the_fitted_state_not_the_new_frame():
    """The anti-leakage contract: state comes from `fit`, never from the frame passed
    to `transform`."""
    from batcher.ml import StandardScaler

    train = bt.from_pydict({"x": [0.0, 10.0]})  # mean 5, population std 5
    scaler = StandardScaler(["x"]).fit(train)

    # A frame with completely different statistics must be scaled by the *train* ones:
    # x = 5 sits at the train mean, so it must map to 0.0 regardless of this frame.
    other = bt.from_pydict({"x": [5.0, 105.0]})
    got = scaler.transform(other).to_pydict()["x"]
    assert got[0] == pytest.approx(0.0)
    assert got[1] == pytest.approx((105.0 - 5.0) / 5.0)


def test_fit_on_train_transform_both_splits():
    from batcher.ml import SimpleImputer, StandardScaler

    ds = bt.from_pydict({"x": [float(i) for i in range(100)], "y": list(range(100))})
    train, test = ds.ml.train_test_split(0.3, seed=11)

    imputer = SimpleImputer(["x"]).fit(train)
    scaler = StandardScaler(["x"]).fit(imputer.transform(train))

    def prep(part):
        return scaler.transform(imputer.transform(part))

    train_out, test_out = prep(train).to_pydict(), prep(test).to_pydict()
    assert len(train_out["x"]) == train.count()
    assert len(test_out["x"]) == test.count()
    # The train split, scaled by its own statistics, is centered; the test split is not
    # forced to be — that asymmetry is exactly what "no leakage" means.
    assert sum(train_out["x"]) / len(train_out["x"]) == pytest.approx(0.0, abs=1e-9)


def test_simple_imputer_fills_nulls_with_the_fitted_mean():
    from batcher.ml import SimpleImputer

    train = bt.from_pydict({"x": [1.0, 3.0, None]})  # mean of non-null = 2.0
    got = SimpleImputer(["x"]).fit_transform(train).to_pydict()["x"]
    assert got == pytest.approx([1.0, 3.0, 2.0])


def test_one_hot_encoder_expands_categories():
    from batcher.ml import OneHotEncoder

    ds = bt.from_pydict({"c": ["a", "b", "a"]})
    out = OneHotEncoder(["c"]).fit_transform(ds).to_pydict()
    assert out["c_a"] == [1, 0, 1]
    assert out["c_b"] == [0, 1, 0]
