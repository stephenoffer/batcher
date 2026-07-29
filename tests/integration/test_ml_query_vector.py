"""A vector search rejects a query that cannot match the column it searches.

Searching with an embedding of the wrong width is the vector-search mistake: the query came
from a different model, or from the same model at a different Matryoshka dimension. The
engine's answer was a raw ``RuntimeError`` out of the kernel naming two "list lengths", and
an empty query got ``ValueError: array() requires at least one element``, which names
nothing at all.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError


@pytest.fixture
def tensor_ds() -> bt.Dataset:
    """An embedding column as `map_batches` produces it: a fixed-shape tensor."""
    return bt.from_pydict({"id": [1, 2, 3]}).map_batches(
        lambda b: {"id": b.column("id").to_pylist(), "embedding": np.eye(3, 4, dtype="float32")},
        output_columns=["id", "embedding"],
    )


@pytest.fixture
def fsl_ds() -> bt.Dataset:
    """An embedding column as Arrow types it on read: a fixed-size list."""
    return bt.from_arrow(
        pa.table(
            {
                "id": [1, 2],
                "embedding": pa.array([[1.0, 0.0], [0.0, 1.0]], type=pa.list_(pa.float64(), 2)),
            }
        )
    )


def test_wrong_width_against_a_tensor_column(tensor_ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="2-dimensional query but 'embedding' holds 4"):
        tensor_ds.ml.nearest_neighbors([1.0, 0.0], k=2)


def test_wrong_width_against_a_fixed_size_list(fsl_ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="3-dimensional query but 'embedding' holds 2"):
        fsl_ds.ml.similarity_to([1.0, 0.0, 0.0])


def test_matching_width_is_untouched(tensor_ds: bt.Dataset) -> None:
    assert tensor_ds.ml.nearest_neighbors([1.0, 0.0, 0.0, 0.0], k=2).to_pydict()["id"] == [1, 2]


def test_empty_query_is_rejected(fsl_ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="empty query vector"):
        fsl_ds.ml.nearest_neighbors([], k=1)


def test_unknown_column_is_named(fsl_ds: bt.Dataset) -> None:
    with pytest.raises(ColumnNotFoundError, match="nope"):
        fsl_ds.ml.nearest_neighbors([1.0, 0.0], column="nope")


def test_a_variable_length_list_column_is_not_width_checked() -> None:
    """No static width to check against, so the query passes through as before."""
    loose = bt.from_pydict({"id": [1], "embedding": [[1.0, 0.0]]})
    assert loose.ml.similarity_to([1.0, 0.0]).to_pydict()["id"] == [1]
