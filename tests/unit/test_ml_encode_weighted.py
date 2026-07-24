"""Rank/label preprocessors, weighted statistics, and the Hamming loss.

The weighted statistics are checked against `numpy.average` and a frequency-weighted
`numpy.cov`, which is the convention they follow. The rank and binarizer preprocessors are
pinned to their structural definitions — a rank is the fraction below, a binarizer is
one-vs-rest — and to the property that makes each worth having.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.preprocessors import (
    LabelBinarizer,
    MultiLabelBinarizer,
    RankTransformer,
)

pytestmark = pytest.mark.unit


# --- RankTransformer -------------------------------------------------------------------


def test_rank_transformer_maps_to_percentile() -> None:
    ds = bt.from_pydict({"x": [10.0, 40.0, 20.0, 1000.0]})
    got = RankTransformer("x").fit_transform(ds).to_pydict()["x"]
    assert got == [0.0, pytest.approx(2 / 3), pytest.approx(1 / 3), 1.0]


def test_rank_transformer_is_immune_to_an_outlier() -> None:
    # The extreme value becomes simply rank 1, not a huge number.
    ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 1e12]})
    assert max(RankTransformer("x").fit_transform(ds).to_pydict()["x"]) == 1.0


def test_rank_transformer_is_monotone() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=100)
    ds = bt.from_pydict({"x": values.tolist()})
    ranked = RankTransformer("x").fit_transform(ds).to_pydict()["x"]
    # The rank order matches the value order.
    assert [r for _, r in sorted(zip(values, ranked, strict=True))] == sorted(ranked)


# --- LabelBinarizer --------------------------------------------------------------------


def test_label_binarizer_expands_one_column_per_class() -> None:
    ds = bt.from_pydict({"label": ["cat", "dog", "cat"]})
    out = LabelBinarizer("label").fit_transform(ds).to_pydict()
    assert out["label_cat"] == [1, 0, 1]
    assert out["label_dog"] == [0, 1, 0]


def test_label_binarizer_can_drop_the_original() -> None:
    ds = bt.from_pydict({"y": ["a", "b"]})
    out = LabelBinarizer("y", drop_original=True).fit_transform(ds)
    assert out.columns == ["y_a", "y_b"]


def test_label_binarizer_indicators_sum_to_one_per_row() -> None:
    ds = bt.from_pydict({"y": ["a", "b", "c", "a"]})
    out = LabelBinarizer("y").fit_transform(ds).to_pydict()
    per_row = [out["y_a"][i] + out["y_b"][i] + out["y_c"][i] for i in range(4)]
    assert per_row == [1, 1, 1, 1]


def test_label_binarizer_takes_one_column() -> None:
    with pytest.raises(PlanError, match="one column"):
        LabelBinarizer(["a", "b"])


# --- MultiLabelBinarizer ---------------------------------------------------------------


def test_multilabel_binarizer_marks_membership() -> None:
    ds = bt.from_pydict({"tags": [["a", "b"], ["b"], ["c"]]})
    out = MultiLabelBinarizer("tags").fit_transform(ds).to_pydict()
    assert out["tags_a"] == [1, 0, 0]
    assert out["tags_b"] == [1, 1, 0]
    assert out["tags_c"] == [0, 0, 1]


def test_multilabel_binarizer_allows_many_labels_per_row() -> None:
    ds = bt.from_pydict({"t": [["x", "y", "z"]]})
    out = MultiLabelBinarizer("t").fit_transform(ds).to_pydict()
    assert out["t_x"] == [1] and out["t_y"] == [1] and out["t_z"] == [1]


def test_multilabel_binarizer_honors_an_explicit_label_set() -> None:
    ds = bt.from_pydict({"t": [["a"], ["b"]]})
    out = MultiLabelBinarizer("t", labels=["a", "b", "c"]).fit_transform(ds)
    assert {"t_a", "t_b", "t_c"} <= set(out.columns)


# --- weighted statistics ---------------------------------------------------------------


@pytest.fixture(scope="module")
def weighted() -> tuple[np.ndarray, np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    y = 2.0 * x + rng.normal(0, 0.5, 400)
    w = rng.random(400) + 0.1
    ds = bt.from_pydict({"x": x.tolist(), "y": y.tolist(), "w": w.tolist()})
    return x, y, w, ds


def _agg(ds: bt.Dataset, expr) -> float:
    return ds.agg(m=expr).collect().column("m")[0].as_py()


def test_weighted_mean_matches_numpy(weighted) -> None:
    x, _, w, ds = weighted
    assert _agg(ds, bt.weighted_mean("x", "w")) == pytest.approx(np.average(x, weights=w))


def test_weighted_var_matches_numpy(weighted) -> None:
    x, _, w, ds = weighted
    mean = np.average(x, weights=w)
    expected = np.average((x - mean) ** 2, weights=w)
    assert _agg(ds, bt.weighted_var("x", "w")) == pytest.approx(expected)


def test_weighted_std_is_the_root_of_the_variance(weighted) -> None:
    _, _, _, ds = weighted
    got = _agg(ds, bt.weighted_std("x", "w"))
    assert got == pytest.approx(_agg(ds, bt.weighted_var("x", "w")) ** 0.5)


def test_weighted_correlation_matches_numpy(weighted) -> None:
    x, y, w, ds = weighted
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    sx = np.sqrt(np.average((x - mx) ** 2, weights=w))
    sy = np.sqrt(np.average((y - my) ** 2, weights=w))
    assert _agg(ds, bt.weighted_correlation("x", "y", "w")) == pytest.approx(cov / (sx * sy))


def test_weighted_reduces_to_unweighted_when_weights_are_equal() -> None:
    ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "w": [1.0, 1.0, 1.0, 1.0]})
    got = _agg(ds, bt.weighted_mean("x", "w"))
    assert got == pytest.approx(2.5)


def test_weighted_statistics_compose_with_group_by() -> None:
    ds = bt.from_pydict(
        {"g": ["a", "a", "b", "b"], "x": [1.0, 3.0, 10.0, 30.0], "w": [1.0, 1.0, 1.0, 1.0]}
    )
    got = ds.group_by("g").agg(m=bt.weighted_mean("x", "w")).sort("g").to_pydict()
    assert got["m"] == [2.0, 20.0]


# --- hamming_loss ----------------------------------------------------------------------


def test_hamming_loss_is_the_fraction_wrong() -> None:
    ds = bt.from_pydict({"y": [1, 0, 1, 1], "p": [1, 0, 0, 1]})
    assert _agg(ds, bt.hamming_loss("y", "p")) == pytest.approx(0.25)


def test_hamming_loss_is_one_minus_accuracy(weighted) -> None:
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    p = rng.integers(0, 2, 300)
    ds = bt.from_pydict({"y": y.tolist(), "p": p.tolist()})
    got = _agg(ds, bt.hamming_loss("y", "p"))
    assert got == pytest.approx(1.0 - _agg(ds, bt.accuracy("y", "p")))
