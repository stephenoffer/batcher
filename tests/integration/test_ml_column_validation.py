"""The `ds.ml` loaders reject an unknown column where it was written, not on first pull.

They return generators, so a mistyped column name used to raise a bare pyarrow
``KeyError: 'Field "NOPE" does not exist in schema'`` on the first `next()` — which for a
training loader means inside the first step of the training loop, naming neither the
parameter nor the method nor the columns that do exist.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError


@pytest.fixture
def ds() -> bt.Dataset:
    return bt.from_pydict({"id": [1, 2, 3], "t": ["a", "b", "c"]})


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda d: d.ml.to_numpy_batches(columns=["nope"]), id="to_numpy_batches"),
        pytest.param(lambda d: d.ml.iter_torch_batches(columns=["nope"]), id="iter_torch_batches"),
        pytest.param(lambda d: d.ml.to_torch(columns=["nope"]), id="to_torch"),
        pytest.param(lambda d: d.ml.to_tf(columns=["nope"]), id="to_tf"),
        pytest.param(
            lambda d: d.ml.to_torch_dataloader(columns=["nope"]), id="to_torch_dataloader"
        ),
        pytest.param(lambda d: d.ml.stream_loader(columns=["nope"]), id="stream_loader"),
        pytest.param(lambda d: d.ml.download("nope"), id="download"),
    ],
)
def test_unknown_column_raises_at_the_call_site(ds: bt.Dataset, call) -> None:
    with pytest.raises(ColumnNotFoundError) as caught:
        call(ds)
    message = str(caught.value)
    assert "nope" in message
    assert "'id'" in message and "'t'" in message  # the available columns are listed


def test_the_message_names_the_parameter(ds: bt.Dataset) -> None:
    with pytest.raises(ColumnNotFoundError, match="url_column"):
        ds.ml.download("nope")


def test_a_valid_selection_is_untouched(ds: bt.Dataset) -> None:
    assert next(iter(ds.ml.to_numpy_batches(columns=["id"])))["id"].tolist() == [1, 2, 3]


def test_no_selection_is_untouched(ds: bt.Dataset) -> None:
    assert sorted(next(iter(ds.ml.to_numpy_batches()))) == ["id", "t"]
