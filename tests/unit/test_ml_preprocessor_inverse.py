"""`inverse_transform` on the reversible preprocessors, a reasoned refusal elsewhere (AP-408)."""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.preprocessors import (
    KBinsDiscretizer,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    SimpleImputer,
    StandardScaler,
)

pytestmark = pytest.mark.unit

_X = [3.5, -1.0, 0.0, 12.25, None, 7.0]


def _ds() -> bt.Dataset:
    return bt.from_pydict({"x": _X, "keep": list(range(len(_X)))})


@pytest.mark.parametrize(
    "make",
    [
        lambda: StandardScaler("x"),
        lambda: StandardScaler("x", with_mean=False),
        lambda: MinMaxScaler("x"),
        lambda: MinMaxScaler("x", feature_range=(-2.0, 5.0)),
        lambda: MaxAbsScaler("x"),
        lambda: RobustScaler("x"),
    ],
)
def test_scaler_inverse_round_trips_within_float_tolerance(make):
    pre = make().fit(_ds())
    back = pre.inverse_transform(pre.transform(_ds())).to_pydict()
    assert back["keep"] == list(range(len(_X)))
    for got, want in zip(back["x"], _X, strict=True):
        assert (got is None and want is None) or got == pytest.approx(want, abs=1e-12)


def test_a_non_in_place_scaler_restores_the_source_column_from_its_output():
    pre = StandardScaler("x", output_columns="z").fit(_ds())
    scaled = pre.transform(_ds()).drop("x")
    back = pre.inverse_transform(scaled).to_pydict()
    assert back["x"][:4] == pytest.approx(_X[:4])


def test_a_constant_column_inverts_to_its_constant():
    ds = bt.from_pydict({"x": [4.0, 4.0]})
    for pre in (MinMaxScaler("x"), StandardScaler("x"), MaxAbsScaler("x"), RobustScaler("x")):
        pre.fit(ds)
        assert pre.inverse_transform(pre.transform(ds)).to_pydict()["x"] == [4.0, 4.0]


def test_ordinal_inverse_restores_categories_and_nulls_the_unknown_code():
    train = bt.from_pydict({"c": ["b", "a", "c", "a"]})
    pre = OrdinalEncoder("c").fit(train)
    assert pre.inverse_transform(pre.transform(train)).to_pydict() == {"c": ["b", "a", "c", "a"]}
    unseen = bt.from_pydict({"c": ["z", None, "c"]})
    assert pre.inverse_transform(pre.transform(unseen)).to_pydict() == {"c": [None, None, "c"]}


def test_ordinal_inverse_decodes_a_list_column_element_by_element():
    train = bt.from_pydict({"c": [["b", "a"], ["c"], None]})
    pre = OrdinalEncoder("c").fit(train)
    back = pre.inverse_transform(pre.transform(train)).to_pydict()
    assert back == {"c": [["b", "a"], ["c"], None]}


def test_label_encoder_inverse_decodes_predicted_indices():
    enc = LabelEncoder("y", output_column="y_code").fit(
        bt.from_pydict({"y": ["dog", "cat", "emu"]})
    )
    preds = bt.from_pydict({"y_code": [2, 0, 1, -1]})
    assert enc.inverse_transform(preds).to_pydict() == {
        "y_code": [2, 0, 1, -1],
        "y": ["emu", "cat", "dog", None],
    }


@pytest.mark.parametrize(
    ("pre", "reason"),
    [
        (KBinsDiscretizer("x", n_bins=2), "same bin index"),
        (SimpleImputer("x"), "indistinguishable"),
        (Normalizer(["x", "keep"]), "norm"),
    ],
)
def test_a_lossy_preprocessor_refuses_and_says_why(pre, reason):
    pre.fit(_ds())
    with pytest.raises(PlanError, match=reason):
        pre.inverse_transform(_ds())


def test_one_hot_refusal_names_the_collapse():
    ds = bt.from_pydict({"c": ["a", "b"]})
    with pytest.raises(PlanError, match="all-zero row"):
        OneHotEncoder("c").fit(ds).inverse_transform(ds)
