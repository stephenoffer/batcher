"""A preprocessor must not invent a value for a missing one.

`KBinsDiscretizer` did. Its bin index is a `CASE` chain of ``value < edge`` tests, and a
null compares false against every edge, so every missing value fell through to the
`otherwise` and landed in the **top** bin. Nothing errored and nothing warned, so a model
trained on it learned from fabricated values sitting at one end of the feature's range.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.ml.preprocessors import (
    Binarizer,
    KBinsDiscretizer,
    LogTransformer,
    MinMaxScaler,
    Normalizer,
    RankTransformer,
    StandardScaler,
)


def test_binning_keeps_a_null_null() -> None:
    ds = bt.from_pydict({"v": [0.0, 2.0, 6.0, 8.0, 10.0, None]})
    out = KBinsDiscretizer(["v"], n_bins=2, strategy="uniform").fit_transform(ds).to_pydict()
    assert out["v"] == [0, 0, 1, 1, 1, None]


def test_binning_of_clean_data_is_unchanged() -> None:
    """The fix must not move a real value between bins."""
    ds = bt.from_pydict({"v": [0.0, 2.0, 6.0, 8.0, 10.0]})
    out = KBinsDiscretizer(["v"], n_bins=2, strategy="uniform").fit_transform(ds).to_pydict()
    assert out["v"] == [0, 0, 1, 1, 1]


def test_binning_null_is_not_the_top_bin_at_any_width() -> None:
    """The old bug always produced ``n_bins - 1``, so a two-bin test alone could not see it."""
    fit_on = bt.from_pydict({"v": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]})
    binner = KBinsDiscretizer(["v"], n_bins=5, strategy="uniform").fit(fit_on)
    out = binner.transform(bt.from_pydict({"v": [0.0, None, 10.0]})).to_pydict()
    assert out["v"][1] is None
    assert out["v"][0] != out["v"][2]  # the real values still separate


def test_the_bin_column_stays_an_integer_column() -> None:
    ds = bt.from_pydict({"v": [0.0, None, 10.0]})
    binned = KBinsDiscretizer(["v"], n_bins=2, strategy="uniform").fit_transform(ds)
    assert binned.schema.field("v").type == "int64"


@pytest.mark.parametrize("cls", [StandardScaler, MinMaxScaler, Normalizer])
def test_the_scalers_already_agree(cls) -> None:
    """Pins the convention the binner now follows, rather than assuming it."""
    ds = bt.from_pydict({"v": [1.0, 2.0, 3.0, None]})
    assert cls(["v"]).fit_transform(ds).to_pydict()["v"][3] is None


def test_binarizer_does_not_call_a_null_below_threshold() -> None:
    """A null compared false against the threshold, so `otherwise` claimed it as a 0."""
    ds = bt.from_pydict({"v": [1.0, 2.0, 3.0, 4.0, None]})
    out = Binarizer("v", threshold=2.0).fit_transform(ds).to_pydict()["v"]
    assert out == [0, 0, 1, 1, None]


def test_binarizer_of_clean_data_is_unchanged() -> None:
    ds = bt.from_pydict({"v": [-1.0, 1.0]})
    assert Binarizer("v").fit_transform(ds).to_pydict()["v"] == [0, 1]


def test_rank_does_not_put_a_null_at_the_top() -> None:
    """A null ranked last, so every missing value came out at percentile 1.0."""
    ds = bt.from_pydict({"v": [1.0, 2.0, 3.0, 4.0, None]})
    out = RankTransformer("v").fit_transform(ds).to_pydict()["v"]
    assert out == [0.0, 0.25, 0.5, 0.75, None]


def test_rank_of_clean_data_is_unchanged() -> None:
    ds = bt.from_pydict({"v": [3.0, 1.0, 2.0]})
    assert RankTransformer("v").fit_transform(ds).to_pydict()["v"] == [1.0, 0.0, 0.5]


def test_log_emits_null_not_nan_as_documented() -> None:
    """The class docstring has always promised null; the code emitted `nan`.

    The difference is not cosmetic: a NaN escapes `is_null()`, so a downstream null check
    calls the column clean, and it propagates through `mean`/`sum` to poison every aggregate.
    """
    ds = bt.from_pydict({"v": [1.0, 2.0, None]})
    out = LogTransformer("v").fit_transform(ds).to_pydict()["v"]
    assert out[2] is None


def test_log_emits_null_for_an_undefined_logarithm() -> None:
    ds = bt.from_pydict({"v": [1.0, -5.0, None]})
    out = LogTransformer("v", offset=0.0).fit_transform(ds).to_pydict()["v"]
    assert out == [0.0, None, None]


def test_a_nulled_column_still_reports_its_nulls() -> None:
    """The property NaN broke: `null_count` must see what the transform produced."""
    ds = bt.from_pydict({"v": [1.0, 2.0, None]})
    logged = LogTransformer("v").fit_transform(ds)
    assert logged.filter(bt.col("v").is_null()).count() == 1
