"""A similarity join between differently-sized vectors is caught before it runs.

"Embedding dimension mismatch between indexing and query" is a named symptom in the RAG
guides: the index is built with one model and the query embedded with another, and the two
sides differ only by a number nobody looks at.

The engine already refused — but as a `RuntimeError` from inside a Rust kernel ("string
function list.CosineSimilarity: list dimensions must be equal"), after the whole scan, in
vocabulary that belongs to the engine. Both widths are in the schema whenever the vectors
came from `ds.ml.embed`, so it is knowable before a row is read.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit


def _vectors(values: list[float], width: int, ids: list[int]) -> bt.Dataset:
    col = pa.FixedSizeListArray.from_arrays(pa.array(values, type=pa.float64()), width)
    return bt.from_arrow(pa.table({"id": pa.array(ids), "v": col}))


def _four(ids=(1, 2)) -> bt.Dataset:
    return _vectors([1.0, 0, 0, 0, 0, 1.0, 0, 0], 4, list(ids))


def _eight(ids=(9,)) -> bt.Dataset:
    return _vectors([1.0, 0, 0, 0, 0, 0, 0, 0], 8, list(ids))


def test_a_dimension_mismatch_is_refused_before_execution() -> None:
    with pytest.raises(PlanError, match="4-dimensional"):
        _four().ml.similarity_join(_eight(), left_on="v", threshold=0.8)


def test_the_message_names_the_actual_cause() -> None:
    """ "dimensions must be equal" is a restatement; "two different embedding models" is a
    diagnosis the user can act on."""
    with pytest.raises(PlanError, match="two different"):
        _four().ml.similarity_join(_eight(), left_on="v", threshold=0.8)


def test_it_is_a_typed_error_not_a_bare_runtime_one() -> None:
    """A `RuntimeError` from the engine cannot be caught by a user handling Batcher's
    exceptions, and leaks kernel vocabulary into a user-facing failure."""
    with pytest.raises(PlanError):
        _four().ml.similarity_join(_eight(), left_on="v", threshold=0.8)


def test_it_fails_at_build_time_not_at_collect() -> None:
    """The whole point is failing before the scan — a lazy plan that only raises at
    `collect()` has already read the data."""
    with pytest.raises(PlanError):
        _four().ml.similarity_join(_eight(), left_on="v", threshold=0.8)  # no terminal op


def test_matching_dimensions_are_untouched() -> None:
    joined = _four().ml.similarity_join(_four(ids=(3, 4)), left_on="v", threshold=0.5)
    assert joined.to_pydict() is not None


def test_a_plain_list_column_is_left_to_the_engine() -> None:
    """A `list` column declares no width, so there is nothing to compare here; the engine
    still checks per row. Guessing from the first row would reject a ragged column that the
    engine would accept."""
    left = bt.from_arrow(pa.table({"id": [1], "v": [[1.0, 0.0, 0.0, 0.0]]}))
    right = bt.from_arrow(pa.table({"id": [2], "v": [[1.0, 0.0, 0.0, 0.0]]}))
    left.ml.similarity_join(right, left_on="v", threshold=0.8)  # builds without raising
