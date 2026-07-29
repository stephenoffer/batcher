"""A serving backend that answers wrongly is caught before its answer is used.

`serving_udf` sends a batch to a remote model and lines the response up with the rows that
produced it. The failure that matters is an **under-returning** backend — a truncated
response, a partially-failed batch — because that is the one standing between the caller and
a result whose predictions belong to different rows. It surfaced as
``ValueError: Arrays were not all the same length: 1 vs 3`` from inside the output assembly,
naming neither the backend nor the output that came up short.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import BackendError, ColumnNotFoundError
from batcher.ml import serving_udf


class Good:
    def predict(self, inputs: dict) -> dict:
        return {"y": inputs["x"] * 2}


class Short:
    """Answers fewer rows than it was sent."""

    def predict(self, inputs: dict) -> dict:
        return {"y": inputs["x"][:1]}


class NotADict:
    def predict(self, inputs: dict) -> list:
        return [1, 2, 3]


class WrongName:
    def predict(self, inputs: dict) -> dict:
        return {"z": inputs["x"]}


@pytest.fixture
def ds() -> bt.Dataset:
    return bt.from_pydict({"x": [1.0, 2.0, 3.0]})


def _run(ds: bt.Dataset, client, **kwargs):
    udf = serving_udf(lambda: client, input_columns=["x"], **kwargs)
    return ds.map_batches(udf, output_columns=["x", "y"]).to_pydict()


def test_a_working_backend_is_untouched(ds: bt.Dataset) -> None:
    assert _run(ds, Good()) == {"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]}


def test_an_under_returning_backend_is_caught(ds: bt.Dataset) -> None:
    with pytest.raises(BackendError, match="returned 1 rows for output 'y' but was sent 3"):
        _run(ds, Short())


def test_the_message_says_why_it_matters(ds: bt.Dataset) -> None:
    with pytest.raises(BackendError, match="wrong inputs"):
        _run(ds, Short())


def test_a_non_dict_response_is_caught(ds: bt.Dataset) -> None:
    with pytest.raises(BackendError, match="must return a"):
        _run(ds, NotADict())


def test_a_missing_declared_output_is_caught(ds: bt.Dataset) -> None:
    with pytest.raises(BackendError, match="missing output 'y'"):
        _run(ds, WrongName(), output_columns=["y"])


def test_an_unknown_input_column_names_the_batch(ds: bt.Dataset) -> None:
    """`serving_udf` builds a callable, so the first batch is the earliest honest moment."""
    udf = serving_udf(lambda: Good(), input_columns=["nope"])
    with pytest.raises(ColumnNotFoundError, match=r"input_columns=\['nope'\]"):
        ds.map_batches(udf, output_columns=["x", "y"]).to_pydict()


def test_a_raising_backend_still_reports_the_retries(ds: bt.Dataset) -> None:
    class Down:
        def predict(self, inputs: dict) -> dict:
            raise RuntimeError("backend down")

    with pytest.raises(BackendError, match="after 3 attempt"):
        _run(ds, Down())
