"""The wrong-dimension rate of a vector column — the index-corruption check.

A mixed-dimension embedding column is the failure that leaves no trace: both dimensions are
valid embeddings, the rows either fail to insert or drop silently, and the queries that should
have matched them come back with a plausible distance to something else.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit


def _rate(vectors, expected):
    return (
        bt.from_pydict({"v": vectors})
        .agg(d=bt.embedding_dim_drift("v", expected))
        .to_pydict()["d"][0]
    )


def test_a_uniform_column_has_no_drift():
    assert _rate([[1.0, 2.0], [3.0, 4.0]], 2) == 0.0


def test_a_mixed_column_reports_the_wrong_rows():
    assert _rate([[1.0, 2.0, 3.0], [1.0, 2.0], [0.0, 1.0, 0.0]], 3) == pytest.approx(1 / 3)


def test_every_row_wrong_reports_one():
    assert _rate([[1.0], [2.0]], 4) == 1.0


def test_an_empty_vector_counts_as_wrong():
    """A failed embedding is an empty list, not a null, and must not pass the check."""
    assert _rate([[], [1.0, 2.0]], 2) == 0.5


def test_a_null_row_is_not_counted_as_the_right_dimension():
    """Null is not a valid vector; treating it as conforming would hide a failed encode."""
    got = _rate([None, [1.0, 2.0]], 2)
    assert got == 0.5


def test_the_check_breaks_down_by_source():
    ds = bt.from_pydict(
        {
            "model": ["v1", "v1", "v2", "v2"],
            "v": [[1.0, 2.0], [3.0, 4.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        }
    )
    got = ds.group_by("model").agg(d=bt.embedding_dim_drift("v", 2)).to_pydict()
    by_model = dict(zip(got["model"], got["d"], strict=True))
    assert by_model["v1"] == 0.0
    assert by_model["v2"] == 1.0


def test_a_non_positive_dimension_is_rejected():
    with pytest.raises(PlanError):
        bt.embedding_dim_drift("v", 0)
