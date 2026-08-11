"""A Python value Arrow cannot type is diagnosed on the way *in*, not just on the way out.

`map_batches` has named the offending column and the fix for a while; the constructors did
not, and they are where a Python user meets the problem first. A UUID primary key or an enum
member in a fifty-column dict produced pyarrow's own message — the value's repr, its class,
and nothing about which column it came from or what to do with it.

The two sides share one diagnosis (`interop.diagnostics`), so a fix learned in one place is
learned in both. These tests pin the ingest half, and that the two agree.
"""

from __future__ import annotations

import enum
import pathlib
import uuid

import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit


class Color(enum.Enum):
    RED = "red"


class _Opaque:
    """An object with no Arrow representation and no known remedy."""


def test_from_pydict_names_the_column_and_the_uuid_fix():
    with pytest.raises(PlanError, match=r"column 'id'.*str\(u\)"):
        bt.from_pydict({"id": [uuid.uuid4()], "n": [1]})


def test_from_pylist_names_the_column_too():
    """Row-oriented input has to be pivoted before a *column* can be named at all."""
    with pytest.raises(PlanError, match=r"column 'id'"):
        bt.from_pylist([{"id": uuid.uuid4(), "n": 1}])


def test_an_enum_member_is_told_to_pass_its_value():
    with pytest.raises(PlanError, match=r"pass its `\.value`"):
        bt.from_pylist([{"c": Color.RED}])


def test_a_path_is_told_to_pass_its_string():
    with pytest.raises(PlanError, match=r"pass `str\(path\)`"):
        bt.from_pydict({"p": [pathlib.Path("/tmp")]})


def test_an_unknown_object_still_gets_the_generic_remedy():
    with pytest.raises(PlanError, match="Arrow-native type"):
        bt.from_pydict({"o": [_Opaque()]})


def test_the_constructor_is_named_so_the_error_says_where_it_came_from():
    with pytest.raises(PlanError, match=r"from_pydict\(\)"):
        bt.from_pydict({"id": [uuid.uuid4()]})


def test_both_sides_of_the_boundary_give_the_same_remedy():
    """A fix learned for a returned column must be the fix for a supplied one."""
    ingest = _message(lambda: bt.from_pydict({"id": [uuid.uuid4()]}))
    emitted = _message(
        lambda: (
            bt.from_pydict({"x": [1]})
            .map_batches(lambda b: {"id": [uuid.uuid4()]}, output_columns=["id"])
            .to_pydict()
        )
    )
    remedy = "For a uuid.UUID, pass `str(u)`"
    assert remedy in ingest
    assert remedy in emitted


def test_a_shape_mismatch_is_not_reported_as_a_missing_conversion():
    """A wrong-length column is a different mistake and must not borrow this message."""
    with pytest.raises(PlanError, match="from_pydict"):
        bt.from_pydict({"a": [1, 2], "b": [1]})


def test_the_valid_shapes_are_untouched():
    assert bt.from_pydict({"a": [1, 2]}).to_pydict() == {"a": [1, 2]}
    assert bt.from_pylist([{"a": 1}, {"a": 2}]).to_pydict() == {"a": [1, 2]}


def _message(call) -> str:
    with pytest.raises(PlanError) as caught:
        call()
    return str(caught.value)
