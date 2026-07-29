"""A featurizer pointed at the wrong column type says so, in its own words.

`DateTimeFeaturizer`, `CyclicalEncoder`, and `TextStatFeaturizer` reached the engine with a
column they could not use and came back with a raw kernel message — ``Compute error: Hour
does not support: Float64``, ``string function Len expected a Utf8 argument, got Float64`` —
naming an internal function and neither the preprocessor nor the column. The schema already
knows, and reading it costs no scan.
"""

from __future__ import annotations

import datetime as dt

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.preprocessors import CyclicalEncoder, DateTimeFeaturizer, TextStatFeaturizer


@pytest.fixture
def numeric() -> bt.Dataset:
    return bt.from_pydict({"v": [1.0, 2.0]})


@pytest.fixture
def stamped() -> bt.Dataset:
    return bt.from_pydict({"t": [dt.datetime(2024, 3, 16, 14, 30), dt.datetime(2024, 7, 4)]})


@pytest.mark.parametrize("cls", [DateTimeFeaturizer, CyclicalEncoder])
def test_calendar_featurizers_reject_a_numeric_column(numeric: bt.Dataset, cls) -> None:
    with pytest.raises(PlanError) as caught:
        cls("v").fit_transform(numeric)
    message = str(caught.value)
    assert cls.__name__ in message
    assert "'v'" in message and "double" in message


def test_text_featurizer_rejects_a_numeric_column(numeric: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="TextStatFeaturizer needs a string column"):
        TextStatFeaturizer("v").fit_transform(numeric)


def test_calendar_featurizer_accepts_a_timestamp(stamped: bt.Dataset) -> None:
    out = DateTimeFeaturizer("t", parts=["hour"]).fit_transform(stamped)
    assert out.to_pydict()["t_hour"] == [14, 0]


def test_cyclical_encoder_accepts_a_timestamp(stamped: bt.Dataset) -> None:
    assert list(CyclicalEncoder("t", parts=["hour"]).fit_transform(stamped).columns) == [
        "t",
        "t_hour_sin",
        "t_hour_cos",
    ]


def test_text_featurizer_accepts_a_string() -> None:
    txt = bt.from_pydict({"s": ["hello world", "x"]})
    assert TextStatFeaturizer("s").fit_transform(txt).to_pydict()["s_word_count"] == [2, 1]


def test_a_string_column_is_still_allowed_for_calendar_parts() -> None:
    """It may hold parseable timestamps, and the schema cannot tell — so it must not be rejected."""
    strings = bt.from_pydict({"s": ["2024-03-16T14:30:00", "2024-07-04T00:00:00"]})
    assert DateTimeFeaturizer("s", parts=["year"]).fit_transform(strings).to_pydict()["s_year"] == [
        2024,
        2024,
    ]
