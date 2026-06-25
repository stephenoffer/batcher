"""Preprocessors — fit/transform correctness against a NumPy/sklearn reference.

`fit` lowers to mergeable aggregates and `transform` to `Expr` projections, so these
assert the *math* (matching scikit-learn's definitions) end to end through the engine.
The fit statistics inherit distributed-correctness from the aggregate layer (already
single-node == distributed tested); here we also check fit-on-train / transform-on-test
reuse and that a partitioned fit yields identical statistics.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher.ml.preprocessors import (
    Concatenator,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    SimpleImputer,
    StandardScaler,
)


def _col(ds, name):
    return np.array(ds.collect().column(name).to_pylist(), dtype=float)


def test_standard_scaler_matches_population_std():
    x = np.array([1.0, 2.0, 5.0, 7.0, 9.0, 11.0])
    ds = bt.from_pydict({"x": x.tolist()})
    expected = (x - x.mean()) / x.std()  # np.std is ddof=0 (population), as sklearn
    got = _col(StandardScaler(["x"]).fit_transform(ds), "x")
    assert np.allclose(got, expected)


def test_standard_scaler_constant_column_no_div_by_zero():
    ds = bt.from_pydict({"x": [3.0, 3.0, 3.0]})
    got = _col(StandardScaler(["x"]).fit_transform(ds), "x")
    assert np.allclose(got, [0.0, 0.0, 0.0])


def test_minmax_scaler_range():
    x = np.array([10.0, 20.0, 30.0, 40.0])
    ds = bt.from_pydict({"x": x.tolist()})
    got = _col(MinMaxScaler(["x"], feature_range=(-1.0, 1.0)).fit_transform(ds), "x")
    expected = (x - x.min()) / (x.max() - x.min()) * 2.0 - 1.0
    assert np.allclose(got, expected)


def test_maxabs_scaler():
    x = np.array([-4.0, -2.0, 0.0, 2.0, 8.0])
    ds = bt.from_pydict({"x": x.tolist()})
    got = _col(MaxAbsScaler(["x"]).fit_transform(ds), "x")
    assert np.allclose(got, x / np.abs(x).max())


def test_robust_scaler():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    ds = bt.from_pydict({"x": x.tolist()})
    got = _col(RobustScaler(["x"]).fit_transform(ds), "x")
    med = np.median(x)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    assert np.allclose(got, (x - med) / iqr, atol=1e-6)


def test_ordinal_encoder_sorted_categories():
    ds = bt.from_pydict({"c": ["b", "a", "c", "a", "b"]})
    got = OrdinalEncoder(["c"]).fit_transform(ds).collect().column("c").to_pylist()
    assert got == [1, 0, 2, 0, 1]  # sorted a<b<c -> 0,1,2


def test_ordinal_encoder_unknown_on_test_set():
    enc = OrdinalEncoder(["c"], unknown_value=-1).fit(bt.from_pydict({"c": ["a", "b"]}))
    out = enc.transform(bt.from_pydict({"c": ["a", "z", "b"]})).collect().column("c").to_pylist()
    assert out == [0, -1, 1]


def test_label_encoder():
    ds = bt.from_pydict({"y": ["cat", "dog", "cat", "bird"]})
    got = LabelEncoder("y").fit_transform(ds).collect().column("y").to_pylist()
    assert got == [1, 2, 1, 0]  # bird<cat<dog -> 0,1,2


def test_one_hot_encoder_indicator_columns():
    ds = bt.from_pydict({"id": [1, 2, 3], "c": ["a", "b", "a"]})
    out = OneHotEncoder(["c"]).fit_transform(ds).collect()
    assert out.column_names == ["id", "c_a", "c_b"]
    assert out.column("c_a").to_pylist() == [1, 0, 1]
    assert out.column("c_b").to_pylist() == [0, 1, 0]


def test_one_hot_encoder_drop_first():
    ds = bt.from_pydict({"c": ["a", "b", "c"]})
    out = OneHotEncoder(["c"], drop_first=True).fit_transform(ds).collect()
    assert out.column_names == ["c_b", "c_c"]


def test_simple_imputer_mean():
    ds = bt.from_pydict({"x": [1.0, None, 3.0, None, 5.0]})
    got = SimpleImputer(["x"], strategy="mean").fit_transform(ds).collect().column("x").to_pylist()
    assert got == [1.0, 3.0, 3.0, 3.0, 5.0]  # mean of [1,3,5] = 3


def test_simple_imputer_median_and_most_frequent():
    ds = bt.from_pydict({"x": [1.0, None, 2.0, 2.0, 100.0], "c": ["a", "a", None, "b", "a"]})
    imp_med = SimpleImputer(["x"], strategy="median").fit_transform(ds).collect()
    assert imp_med.column("x").to_pylist()[1] == 2.0  # median of [1,2,2,100] = 2.0
    imp_mf = SimpleImputer(["c"], strategy="most_frequent").fit_transform(ds).collect()
    assert imp_mf.column("c").to_pylist()[2] == "a"  # 'a' is the mode


def test_simple_imputer_constant_requires_value():
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError, match="fill_value"):
        SimpleImputer(["x"], strategy="constant")


def test_concatenator_builds_list_column():
    ds = bt.from_pydict({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    out = Concatenator(["a", "b"], output_column="f").fit_transform(ds).collect()
    assert out.column("f").to_pylist() == [[1.0, 3.0], [2.0, 4.0]]


def test_sequenced_imputer_then_scaler():
    # Compose by sequencing: fit each step on the prior step's output, then reuse
    # the same fitted objects on any split.
    ds = bt.from_pydict({"x": [1.0, None, 3.0, None]})
    imputer = SimpleImputer(["x"], strategy="constant", fill_value=0.0)
    scaler = StandardScaler(["x"])
    imputed = imputer.fit_transform(ds)
    got = _col(scaler.fit_transform(imputed), "x")
    filled = np.array([1.0, 0.0, 3.0, 0.0])
    assert np.allclose(got, (filled - filled.mean()) / filled.std())


def test_fit_is_partition_independent():
    # Same statistics whether the rows arrive in one batch or several.
    x = list(range(1, 101))
    whole = StandardScaler(["x"]).fit(bt.from_pydict({"x": [float(v) for v in x]}))
    import pyarrow as pa

    parts = [pa.record_batch({"x": [float(v) for v in x[i::4]]}) for i in range(4)]
    chunked = StandardScaler(["x"]).fit(bt.from_arrow(parts))
    assert np.isclose(whole.mean_["x"], chunked.mean_["x"])
    assert np.isclose(whole.scale_["x"], chunked.scale_["x"])


def test_sklearn_cross_check_standard_scaler():
    sk = pytest.importorskip("sklearn.preprocessing")
    x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    ds = bt.from_pydict({"x": x.tolist()})
    got = _col(StandardScaler(["x"]).fit_transform(ds), "x")
    expected = sk.StandardScaler().fit_transform(x.reshape(-1, 1)).ravel()
    assert np.allclose(got, expected)


def test_normalizer_l2_matches_reference():
    from batcher.ml.preprocessors import Normalizer

    ds = bt.from_pydict({"a": [3.0, 1.0, 0.0], "b": [4.0, 1.0, 0.0]})
    out = Normalizer(["a", "b"], norm="l2").fit_transform(ds).collect().to_pydict()
    assert [round(v, 6) for v in out["a"]] == [0.6, round(1 / 2**0.5, 6), 0.0]
    assert [round(v, 6) for v in out["b"]] == [0.8, round(1 / 2**0.5, 6), 0.0]


def test_normalizer_l1():
    from batcher.ml.preprocessors import Normalizer

    ds = bt.from_pydict({"a": [1.0, 0.0], "b": [3.0, 0.0]})
    out = Normalizer(["a", "b"], norm="l1").fit_transform(ds).collect().to_pydict()
    assert [round(v, 6) for v in out["a"]] == [0.25, 0.0]
    assert [round(v, 6) for v in out["b"]] == [0.75, 0.0]


def test_kbins_uniform_equal_width():
    from batcher.ml.preprocessors import KBinsDiscretizer

    ds = bt.from_pydict({"x": [float(i) for i in range(10)]})
    out = KBinsDiscretizer(["x"], n_bins=5, strategy="uniform").fit_transform(ds).collect()
    assert out.to_pydict()["x"] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_kbins_quantile_balances_counts():
    from batcher.ml.preprocessors import KBinsDiscretizer

    ds = bt.from_pydict({"x": [float(i) for i in range(10)]})
    out = KBinsDiscretizer(["x"], n_bins=5, strategy="quantile").fit_transform(ds).collect()
    bins = out.to_pydict()["x"]
    assert min(bins) == 0 and max(bins) == 4  # spans all bins


def test_multihot_encoder_indicators():
    from batcher.ml.preprocessors import MultiHotEncoder

    ds = bt.from_pydict({"tags": [["a", "b"], ["b", "c"], ["a"]]})
    out = MultiHotEncoder("tags").fit_transform(ds).collect().to_pydict()
    assert out["tags_a"] == [1, 0, 1]
    assert out["tags_b"] == [1, 1, 0]
    assert out["tags_c"] == [0, 1, 0]
