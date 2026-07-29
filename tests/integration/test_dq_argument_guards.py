"""The data-quality checks reject an unusable argument where it was written.

A quality gate runs in front of a training pipeline, so its own failures should be the
easiest thing in the pipeline to read. Three were not: a numeric `in_range` against a text
column came back as ``Invalid comparison operation: Utf8 >= Int64``, a malformed regex as a
bare ``invalid regular expression: [`` with no indication of which check carried it, and an
empty `accepted_values` allow-list silently rejected every row.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError


@pytest.fixture
def ds() -> bt.Dataset:
    return bt.from_pydict({"id": [1, 2, 3], "v": [10.0, -5.0, 200.0], "c": ["a", "b", "zz"]})


def test_numeric_bounds_against_a_text_column(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="numeric bounds against a string column"):
        ds.dq.in_range("c", 0, 1)


def test_numeric_bounds_against_a_numeric_column_are_fine(ds: bt.Dataset) -> None:
    assert ds.dq.in_range("v", 0, 100).drop().to_pydict()["v"] == [10.0]


def test_swapped_bounds_still_reported(ds: bt.Dataset) -> None:
    """Pins the guard that already existed, so the new one cannot shadow it."""
    with pytest.raises(PlanError, match="swap the arguments"):
        ds.dq.in_range("v", 100, 0)


@pytest.mark.parametrize("pattern", ["[", "(?P<"])
def test_a_malformed_regex_names_the_check_and_column(ds: bt.Dataset, pattern: str) -> None:
    with pytest.raises(PlanError, match="not a valid regular expression"):
        ds.dq.matches("c", pattern)


def test_a_valid_regex_is_untouched(ds: bt.Dataset) -> None:
    assert ds.dq.matches("c", "^a$").drop().to_pydict()["c"] == ["a"]


def test_an_empty_allow_list_is_rejected(ds: bt.Dataset) -> None:
    """It makes every row a violation, so `.drop()` silently empties the dataset."""
    with pytest.raises(PlanError, match="must be non-empty"):
        ds.dq.accepted_values("c", [])


def test_a_populated_allow_list_is_untouched(ds: bt.Dataset) -> None:
    assert ds.dq.accepted_values("c", ["a", "b"]).drop().to_pydict()["c"] == ["a", "b"]


def test_the_checks_still_compose(ds: bt.Dataset) -> None:
    gate = ds.dq.not_null("id").in_range("v", 0, 100).accepted_values("c", ["a", "b"])
    assert gate.drop().to_pydict()["id"] == [1]
