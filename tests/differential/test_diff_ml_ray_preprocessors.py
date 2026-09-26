"""The Ray Data parameters on Batcher's preprocessors, against Ray Data 2.58 run live.

Each case fits the Ray preprocessor and the Batcher one on the same rows, with the parameter
spelled the way the port writes it, and compares the transformed columns value by value:
`output_columns=` on the scalers, the normalizer and the imputer, `LabelEncoder`'s
`output_column=`, `OrdinalEncoder`'s element-wise list encoding, `Tokenizer`'s default
single-space split, and `UniformKBinsDiscretizer`'s per-column `bins`, `right=` and
`output_columns=` as `KBinsDiscretizer(strategy="uniform")`.

The fixtures stay inside what both engines define. Ray refuses to fit an `OrdinalEncoder`
over a null and its tokenizer calls `str.split` on a null, and Ray's discretizer turns a
value outside the fitted range into NaN where Batcher clamps it, so those edges are pinned
Batcher-side in `tests/unit/test_ml_preprocessor_ray_params.py` rather than compared here.
Ray Data starts a local cluster, which costs tens of seconds, so one cluster serves the
module.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import batcher as bt
from batcher.ml.preprocessors import (
    KBinsDiscretizer,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OrdinalEncoder,
    SimpleImputer,
    StandardScaler,
    Tokenizer,
)

ray = pytest.importorskip("ray")
rp = pytest.importorskip("ray.data.preprocessors")

pytestmark = pytest.mark.differential

NUM = {
    "a": [0.2, 1.4, 2.5, 6.2, 9.7, 2.1, -3.0],
    "b": [10.0, 15.0, 13.0, 12.0, 23.0, 25.0, 10.0],
}


@pytest.fixture(scope="module")
def ray_ctx():
    started = not ray.is_initialized()
    if started:
        ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
    yield
    if started:
        ray.shutdown()


def _ray_out(pre, data: dict) -> pd.DataFrame:
    return pre.fit_transform(ray.data.from_pandas(pd.DataFrame(data))).to_pandas()


def _bt_out(pre, data: dict) -> dict:
    return pre.fit_transform(bt.from_pydict(data)).to_pydict()


def _same_floats(got: list, want) -> None:
    want = [None if (w is None or (isinstance(w, float) and math.isnan(w))) else w for w in want]
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        if w is None:
            assert g is None
        else:
            assert g == pytest.approx(float(w), rel=1e-12, abs=1e-12)


SCALERS = [
    ("StandardScaler", lambda o: rp.StandardScaler(["a", "b"], output_columns=o),
     lambda o: StandardScaler(["a", "b"], output_columns=o)),
    ("MinMaxScaler", lambda o: rp.MinMaxScaler(["a", "b"], output_columns=o),
     lambda o: MinMaxScaler(["a", "b"], output_columns=o)),
    ("MaxAbsScaler", lambda o: rp.MaxAbsScaler(["a", "b"], output_columns=o),
     lambda o: MaxAbsScaler(["a", "b"], output_columns=o)),
    ("Normalizer-l2", lambda o: rp.Normalizer(["a", "b"], output_columns=o),
     lambda o: Normalizer(["a", "b"], output_columns=o)),
    ("Normalizer-l1", lambda o: rp.Normalizer(["a", "b"], norm="l1", output_columns=o),
     lambda o: Normalizer(["a", "b"], norm="l1", output_columns=o)),
]  # fmt: skip


@pytest.mark.parametrize("name, ray_pre, bt_pre", SCALERS, ids=[s[0] for s in SCALERS])
def test_output_columns_matches_ray(ray_ctx, name, ray_pre, bt_pre) -> None:
    outs = ["a_out", "b_out"]
    want = _ray_out(ray_pre(outs), NUM)
    got = _bt_out(bt_pre(outs), NUM)
    assert list(got) == list(want.columns) == ["a", "b", "a_out", "b_out"]
    for column in ("a", "b", "a_out", "b_out"):
        _same_floats(got[column], want[column].tolist())


def test_simple_imputer_output_columns_matches_ray(ray_ctx) -> None:
    data = {"a": [1.0, None, 3.0, float("nan"), 8.0]}
    for strategy in ("mean", "most_frequent"):
        want = _ray_out(rp.SimpleImputer(["a"], strategy=strategy, output_columns=["f"]), data)
        # Ray reads NaN as missing; Batcher's null is the missing value, so NaN is written
        # as null on the Batcher side to compare the same input.
        clean = {"a": [None if v is None or math.isnan(v) else v for v in data["a"]]}
        got = _bt_out(SimpleImputer("a", strategy=strategy, output_columns="f"), clean)
        assert list(got) == list(want.columns) == ["a", "f"]
        _same_floats(got["f"], want["f"].tolist())


def test_label_encoder_output_column_matches_ray(ray_ctx) -> None:
    data = {"y": ["dog", "cat", "bird", "dog", "cat"]}
    want = _ray_out(rp.LabelEncoder("y", output_column="code"), data)
    got = _bt_out(LabelEncoder("y", output_column="code"), data)
    assert got["y"] == want["y"].tolist()
    assert got["code"] == want["code"].tolist()


def test_ordinal_encoder_lists_and_output_columns_match_ray(ray_ctx) -> None:
    data = {"t": [["b", "a"], ["c"], [], ["a", "a"]], "s": ["x", "y", "x", "z"]}
    want = _ray_out(rp.OrdinalEncoder(["t", "s"], output_columns=["tc", "sc"]), data)
    got = _bt_out(OrdinalEncoder(["t", "s"], output_columns=["tc", "sc"]), data)
    assert got["tc"] == [list(map(int, codes)) for codes in want["tc"]]
    assert got["sc"] == want["sc"].tolist()
    assert got["t"] == [list(v) for v in want["t"]]


def test_tokenizer_default_split_matches_ray(ray_ctx) -> None:
    data = {"t": ["a  b", "", " x", "the quick fox "]}
    want = _ray_out(rp.Tokenizer(["t"], output_columns=["toks"]), data)
    got = _bt_out(Tokenizer("t", output_column="toks"), data)
    assert got["toks"] == [list(v) for v in want["toks"]]
    assert got["t"] == data["t"]


@pytest.mark.parametrize("right", [True, False])
def test_uniform_kbins_matches_ray(ray_ctx, right) -> None:
    # Values chosen to land exactly on inner edges, where `right` decides the bin.
    data = {"a": [0.0, 2.5, 5.0, 7.5, 10.0, 3.3], "b": [10.0, 15.0, 13.0, 12.0, 25.0, 20.0]}
    bins = {"a": 4, "b": 3}
    ray_pre = rp.UniformKBinsDiscretizer(
        ["a", "b"], bins=bins, right=right, output_columns=["a_bin", "b_bin"]
    )
    want = _ray_out(ray_pre, data)
    bt_pre = KBinsDiscretizer(
        ["a", "b"], n_bins=bins, strategy="uniform", right=right, output_columns=["a_bin", "b_bin"]
    )
    got = _bt_out(bt_pre, data)
    for column in ("a_bin", "b_bin"):
        assert got[column] == np.asarray(want[column], dtype=np.int64).tolist()
    assert got["a"] == data["a"]
