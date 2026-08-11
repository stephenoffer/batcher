"""`from_items`/`from_iter` name the constructor that takes the shape you actually have.

These two are the scripting entry points, so they are where a Python user arrives first and
where a wrong guess is most likely. Each shape below is a plausible first attempt with a
one-line fix, and each one used to surface as a pyarrow type-inference error that named
neither the constructor, nor the item, nor the fix.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit

_TABLE = pa.table({"x": [1, 2, 3]})


def test_row_tuples_point_at_from_records():
    """The `cursor.fetchall()` shape, and the most common thing to try."""
    with pytest.raises(PlanError, match="from_records"):
        bt.from_items([(1, "a"), (2, "b")])


def test_arrow_batches_point_at_from_batches():
    with pytest.raises(PlanError, match="from_batches"):
        bt.from_items(list(_TABLE.to_batches()))


def test_arrow_tables_point_at_from_batches_too():
    with pytest.raises(PlanError, match="from_batches"):
        bt.from_items([_TABLE])


def test_from_iter_names_itself_not_from_items():
    """`from_iter` delegates, but the user did not call the thing it delegates to."""
    with pytest.raises(PlanError, match=r"from_iter\(\)"):
        bt.from_iter(iter(_TABLE.to_batches()))


def test_partly_dict_items_say_why_they_cannot_share_a_schema():
    with pytest.raises(PlanError, match="not all dicts"):
        bt.from_items([{"a": 1}, 2])


def test_an_unrecognized_shape_still_quotes_the_underlying_reason():
    """A shape with no better constructor must not be answered with a shrug."""
    with pytest.raises(PlanError, match="could not build a column from items of type object"):
        bt.from_items([object(), object()])


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        pytest.param([1, 2, 3], {"item": [1, 2, 3]}, id="scalars"),
        pytest.param([{"a": 1}, {"a": 2}], {"a": [1, 2]}, id="dicts"),
        pytest.param([], {"item": []}, id="empty"),
    ],
)
def test_the_shapes_that_worked_still_work(items, expected):
    assert bt.from_items(items).to_pydict() == expected


def test_from_iter_still_drains_a_generator():
    assert bt.from_iter(x * x for x in range(4)).to_pydict() == {"item": [0, 1, 4, 9]}
