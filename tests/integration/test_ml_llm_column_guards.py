"""`generate`/`extract`/`classify` report an unknown prompt or image column consistently.

The `template` argument already produced a clear "references column(s) not in the data"
error naming what is available. The `prompt_column` and `image_column` beside it did not:
they reached the UDF and failed with a bare pyarrow ``KeyError: 'Field "nope" does not exist
in schema'``. Two arguments of the same call reporting the same mistake two different ways
is what makes an API feel arbitrary.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError


def _engine():
    return lambda prompts: ["r"] * len(prompts)


@pytest.fixture
def ds() -> bt.Dataset:
    return bt.from_pydict({"id": [1, 2], "q": ["a", "b"]})


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda d: d.ml.generate(_engine, prompt_column="nope"), id="generate"),
        pytest.param(
            lambda d: d.ml.classify(_engine, labels=["x"], prompt_column="nope"), id="classify"
        ),
        pytest.param(
            lambda d: d.ml.extract(_engine, schema={"a": "int64"}, prompt_column="nope"),
            id="extract",
        ),
    ],
)
def test_unknown_prompt_column_is_typed_and_lists_alternatives(ds: bt.Dataset, call) -> None:
    with pytest.raises(ColumnNotFoundError) as caught:
        call(ds)
    assert "'q'" in str(caught.value)


def test_unknown_image_column_is_caught(ds: bt.Dataset) -> None:
    with pytest.raises(ColumnNotFoundError, match="image_column"):
        ds.ml.generate(_engine, prompt_column="q", image_column="nope")


def test_a_template_supersedes_the_prompt_column(ds: bt.Dataset) -> None:
    """The template builds the prompt, so an unread `prompt_column` must not be required."""
    out = ds.ml.generate(_engine, prompt_column="not_read", template="Q: {q}")
    assert out.to_pydict()["response"] == ["r", "r"]


def test_a_template_still_reports_its_own_unknown_columns(ds: bt.Dataset) -> None:
    """The template check runs at execution rather than build time; it must still fire."""
    with pytest.raises(Exception, match="nope"):
        ds.ml.generate(_engine, prompt_column="q", template="Q: {nope}").to_pydict()


def _json_engine(payload: str):
    return lambda: lambda prompts: [payload] * len(prompts)


def test_extract_warns_when_a_field_eats_the_prompt_column(ds: bt.Dataset) -> None:
    """The extraction replacing its own input leaves no prompts and raises nothing."""
    from batcher._internal.errors import DataWarning

    with pytest.warns(DataWarning, match="prompt column 'q'"):
        out = ds.ml.extract(
            _json_engine('{"q": "z"}'), prompt_column="q", schema={"q": "string"}
        ).to_pydict()
    assert out["q"] == ["z", "z"]  # the documented replace semantics still apply


def test_extract_warns_on_any_existing_column(ds: bt.Dataset) -> None:
    with pytest.warns(Warning, match=r"\['id'\]"):
        ds.ml.extract(
            _json_engine('{"id": 9}'), prompt_column="q", schema={"id": "int64"}
        ).to_pydict()


def test_extract_is_silent_when_nothing_collides(ds: bt.Dataset) -> None:
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as seen:
        _warnings.simplefilter("always")
        ds.ml.extract(
            _json_engine('{"name": "x"}'), prompt_column="q", schema={"name": "string"}
        ).to_pydict()
    assert not [w for w in seen if "already exist" in str(w.message)]


def test_valid_calls_are_untouched(ds: bt.Dataset) -> None:
    assert ds.ml.generate(_engine, prompt_column="q").to_pydict()["response"] == ["r", "r"]
