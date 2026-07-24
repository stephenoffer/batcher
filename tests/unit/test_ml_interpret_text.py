"""Model interpretation and text surface features.

The interpretation tests use a linear model with a known coefficient structure, so the
*answer* is known rather than merely reproducible: permutation importance must rank the
strong features above the near-useless one, and partial dependence must trace the model's
actual slope. The text-feature tests pin each surface statistic to a hand-counted value.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml.interpret import partial_dependence, permutation_importance
from batcher.ml.preprocessors import TextStatFeaturizer

pytestmark = pytest.mark.unit


# --- permutation importance ------------------------------------------------------------


@pytest.fixture(scope="module")
def linear_model() -> tuple[Any, bt.Dataset, list[str]]:
    """A linear model where feature ``a`` matters most, ``b`` less, ``c`` almost not at all."""
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    features = rng.normal(size=(400, 3))
    target = 3.0 * features[:, 0] - 2.0 * features[:, 1] + 0.01 * features[:, 2]
    model = LinearRegression().fit(features, target)
    ds = bt.from_pydict(
        {
            "a": features[:, 0].tolist(),
            "b": features[:, 1].tolist(),
            "c": features[:, 2].tolist(),
            "y": target.tolist(),
        }
    )
    return model, ds, ["a", "b", "c"]


def _predictor(model: Any, features: list[str]):
    return lambda ds: ds.ml.predict(model, features=features)


def _double_x(ds: bt.Dataset) -> bt.Dataset:
    return ds.with_columns(prediction=bt.col("x") * bt.lit(2.0))


def _identity_x(ds: bt.Dataset) -> bt.Dataset:
    return ds.with_columns(prediction=bt.col("x"))


def test_permutation_importance_ranks_the_useful_features_first(linear_model) -> None:
    model, ds, features = linear_model
    imp = permutation_importance(
        ds, _predictor(model, features), features, y_true="y", n_repeats=3
    ).to_pydict()
    # a (coef 3) above b (coef 2) above c (coef 0.01).
    assert imp["feature"] == ["a", "b", "c"]


def test_a_useless_feature_has_near_zero_importance(linear_model) -> None:
    model, ds, features = linear_model
    imp = permutation_importance(
        ds, _predictor(model, features), features, y_true="y", n_repeats=3
    ).to_pydict()
    by_feature = dict(zip(imp["feature"], imp["importance"], strict=True))
    assert by_feature["c"] < 0.1
    assert by_feature["a"] > 1.0


def test_permutation_importance_is_reproducible(linear_model) -> None:
    model, ds, features = linear_model
    predict = _predictor(model, features)
    first = permutation_importance(ds, predict, features, y_true="y", seed=7).to_pydict()
    second = permutation_importance(ds, predict, features, y_true="y", seed=7).to_pydict()
    assert first["importance"] == pytest.approx(second["importance"])


def test_permutation_importance_needs_a_feature(linear_model) -> None:
    model, ds, features = linear_model
    with pytest.raises(PlanError, match="at least one feature"):
        permutation_importance(ds, _predictor(model, features), [], y_true="y")


def test_permutation_importance_names_a_missing_column(linear_model) -> None:
    model, ds, features = linear_model
    with pytest.raises(ColumnNotFoundError):
        permutation_importance(ds, _predictor(model, features), ["nope"], y_true="y")


# --- partial dependence ----------------------------------------------------------------


def test_partial_dependence_traces_a_known_slope() -> None:
    # A model that is exactly 2*x: the partial-dependence curve must have slope 2.
    ds = bt.from_pydict({"x": [1.0, 2.0, 3.0], "other": [5.0, 1.0, 9.0]})
    curve = partial_dependence(ds, _double_x, "x", grid=[0.0, 1.0, 2.0]).to_pydict()
    assert curve["value"] == [0.0, 1.0, 2.0]
    assert curve["mean_prediction"] == [0.0, 2.0, 4.0]


def test_partial_dependence_averages_over_the_other_features(linear_model) -> None:
    model, ds, features = linear_model
    curve = partial_dependence(ds, _predictor(model, features), "a", grid_points=5).to_pydict()
    # The linear model's dependence on `a` is monotone increasing (coefficient +3).
    assert curve["mean_prediction"] == sorted(curve["mean_prediction"])


def test_partial_dependence_derives_a_grid_from_the_feature_range() -> None:
    ds = bt.from_pydict({"x": [0.0, 10.0], "other": [1.0, 2.0]})
    curve = partial_dependence(ds, _identity_x, "x", grid_points=3).to_pydict()
    assert curve["value"] == [0.0, 5.0, 10.0]


def test_partial_dependence_rejects_a_degenerate_grid() -> None:
    ds = bt.from_pydict({"x": [1.0, 2.0]})
    with pytest.raises(PlanError, match="grid_points"):
        partial_dependence(ds, _identity_x, "x", grid_points=1)


def test_partial_dependence_names_a_missing_feature() -> None:
    ds = bt.from_pydict({"x": [1.0]})
    with pytest.raises(ColumnNotFoundError):
        partial_dependence(ds, _identity_x, "nope")


# --- text surface features -------------------------------------------------------------


def test_text_features_count_characters_and_words() -> None:
    ds = bt.from_pydict({"t": ["Hello World 42"]})
    out = TextStatFeaturizer("t", features=["char_count", "word_count"]).fit_transform(ds)
    got = out.to_pydict()
    assert got["t_char_count"] == [14]
    assert got["t_word_count"] == [3]


def test_text_digit_ratio_is_the_fraction_of_digits() -> None:
    ds = bt.from_pydict({"t": ["ABC12"]})
    out = TextStatFeaturizer("t", features=["digit_ratio"]).fit_transform(ds)
    assert out.to_pydict()["t_digit_ratio"] == [pytest.approx(0.4)]


def test_text_upper_ratio_flags_shouting() -> None:
    ds = bt.from_pydict({"t": ["HELLO", "hello"]})
    out = TextStatFeaturizer("t", features=["upper_ratio"]).fit_transform(ds).to_pydict()
    assert out["t_upper_ratio"] == [pytest.approx(1.0), pytest.approx(0.0)]


def test_text_features_handle_an_empty_string() -> None:
    ds = bt.from_pydict({"t": [""]})
    out = (
        TextStatFeaturizer("t", features=["char_count", "word_count", "digit_ratio"])
        .fit_transform(ds)
        .to_pydict()
    )
    assert out["t_char_count"] == [0]
    assert out["t_word_count"] == [0]
    assert out["t_digit_ratio"] == [pytest.approx(0.0)]


def test_text_punctuation_count() -> None:
    ds = bt.from_pydict({"t": ["a, b! c."]})
    out = TextStatFeaturizer("t", features=["punctuation_count"]).fit_transform(ds)
    assert out.to_pydict()["t_punctuation_count"] == [3]


def test_text_features_are_stateless() -> None:
    pre = TextStatFeaturizer("t", features=["char_count"])
    first = pre.transform(bt.from_pydict({"t": ["abc"]})).to_pydict()
    second = pre.fit_transform(bt.from_pydict({"t": ["abc"]})).to_pydict()
    assert first == second


def test_text_featurizer_can_drop_the_original() -> None:
    ds = bt.from_pydict({"t": ["abc"]})
    out = TextStatFeaturizer("t", features=["char_count"], drop_original=True).fit_transform(ds)
    assert out.columns == ["t_char_count"]


def test_text_featurizer_rejects_an_unknown_feature() -> None:
    with pytest.raises(PlanError, match="unknown text feature"):
        TextStatFeaturizer("t", features=["sentiment"])
