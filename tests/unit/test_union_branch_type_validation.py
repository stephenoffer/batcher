"""A union whose branches disagree on a column's type says so while building the plan.

The name and arity checks already raised a typed `PlanError` at build time. A *type*
mismatch was the one shape that got through: it reached the engine and surfaced mid-query
as a bare `RuntimeError` reading ``set operation (UNION/INTERSECT/EXCEPT) column 0 has
incompatible branch types Int64 and Utf8``. The same user mistake, reported two very
different ways, one of them after the scan had already run.

Only scalar pairs are judged, and that limit is deliberate. `plan.types.promote` is the
flat mirror of the engine's `common_supertype` and is not nested-aware, so it answers
`None` for `list<int32>` against `list<int64>` -- a union the engine performs perfectly
well. Raising on that would break working queries, so a nested type on either side is left
to the engine exactly as before. These tests pin both halves: the mismatch that must raise,
and the widening and nested cases that must not.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit


def _ds(dtype):
    return bt.from_arrow(pa.table({"a": pa.array([], dtype)}))


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (pa.int64(), pa.string()),
        (pa.bool_(), pa.string()),
        (pa.date32(), pa.string()),
        (pa.binary(), pa.string()),
    ],
)
def test_incompatible_scalar_types_raise_planerror_at_build(left, right):
    with pytest.raises(PlanError, match="disagree on the type of column 'a'"):
        _ds(left).union(_ds(right))


def test_error_names_both_types_and_suggests_a_cast():
    with pytest.raises(PlanError) as excinfo:
        _ds(pa.int64()).union(_ds(pa.string()))
    message = str(excinfo.value)
    assert "int64" in message and "string" in message
    assert ".cast(" in message


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (pa.int32(), pa.int64()),
        (pa.int64(), pa.float64()),
        (pa.string(), pa.large_string()),
        (pa.timestamp("s"), pa.timestamp("us")),
        (pa.decimal128(10, 2), pa.float64()),
        (pa.null(), pa.int64()),
    ],
)
def test_widening_pairs_are_still_accepted(left, right):
    """Every pair with a common supertype must keep planning and running."""
    assert _ds(left).union(_ds(right)).collect().to_pydict() == {"a": []}


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (pa.list_(pa.int32()), pa.list_(pa.int64())),
        (pa.struct([("x", pa.int32())]), pa.struct([("x", pa.int64())])),
    ],
)
def test_nested_types_are_left_to_the_engine(left, right):
    """`promote` cannot judge these, and the engine unions them -- so do not raise."""
    assert _ds(left).union(_ds(right)).collect().to_pydict() == {"a": []}


def test_column_name_mismatch_still_reports_the_name_error():
    """The pre-existing check must fire first; a type error here would mislead."""
    with pytest.raises(PlanError, match="identical columns"):
        bt.from_pydict({"a": [1]}).union(bt.from_pydict({"b": [1]}))


def test_matching_types_union_normally():
    got = bt.from_pydict({"a": [1]}).union(bt.from_pydict({"a": [2]})).collect().to_pydict()
    assert sorted(got["a"]) == [1, 2]
