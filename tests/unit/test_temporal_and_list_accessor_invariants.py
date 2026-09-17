"""Null and boundary behaviour for the zero-argument `.dt` and `.list` accessors.

The companion to `test_text_metric_invariants.py`, covering the other two large namespaces
the DuckDB oracle does not fully reach. 56 `.dt` methods and 39 `.list` ones take no
arguments, and the properties below are what they must hold whatever they compute.

The null rule is the same one and matters for the same reason: a temporal component that
returned `0` for a null timestamp claims the year is zero, and a list metric that returned
`0` for a null list claims the list is empty. Both are wrong answers that look like data.

Empty is deliberately *not* folded into that rule, because for a list the two are different
questions and the engine is right to answer them differently. `len([])` is 0 and
`reverse([])` is `[]` -- the list exists and the answer is about it -- while `max([])` is
null, because there is no maximum. This file records which side each method falls on rather
than imposing one, since imposing either would be inventing a semantics.

The boundary timestamps are chosen to break things: the epoch, a leap-day second before
midnight, a pre-1970 date, and 2262-04-11, which is where nanosecond timestamps run out.
"""

from __future__ import annotations

import datetime as dtm
import inspect

import pytest

import batcher as bt

pytestmark = pytest.mark.unit

#: Epoch, leap day at the last second, pre-1970 (negative epoch), the nanosecond ceiling,
#: null, and a plain date.
TIMESTAMPS = [
    dtm.datetime(1970, 1, 1),
    dtm.datetime(2024, 2, 29, 23, 59, 59),
    dtm.datetime(1900, 1, 1),
    dtm.datetime(2262, 4, 11),
    None,
    dtm.datetime(2000, 12, 31),
]

#: Ordinary, empty, null, single, all-zero, containing a null, and extreme magnitudes.
LISTS = [[1.0, 2.0, 3.0], [], None, [1.0], [0.0, 0.0], [None, 1.0], [-1.0, 1e308]]


def _zero_argument(namespace: str, column: str) -> list[str]:
    accessor = getattr(bt.col(column), namespace)
    names = []
    for name in sorted(dir(accessor)):
        if name.startswith("_"):
            continue
        function = getattr(accessor, name, None)
        if not callable(function):
            continue
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            continue
        if [p for p in signature.parameters if not p.startswith("_")]:
            continue
        names.append(name)
    return names


TEMPORAL = _zero_argument("dt", "t")
#: `flatten` wants a list *of lists*; handing it `List<Float64>` is a type error rather than
#: an edge case, and it refuses with one naming both types. Excluded because the fixture is
#: the wrong shape for it, not because it misbehaves.
LIST_METRICS = [n for n in _zero_argument("list", "v") if n != "flatten"]


def _temporal(name: str) -> list:
    return (
        bt.from_pydict({"t": TIMESTAMPS}).select(r=getattr(bt.col("t").dt, name)()).to_pydict()["r"]
    )


def _listwise(name: str) -> list:
    return bt.from_pydict({"v": LISTS}).select(r=getattr(bt.col("v").list, name)()).to_pydict()["r"]


def test_the_sweep_found_both_namespaces():
    assert len(TEMPORAL) >= 40, f"only {len(TEMPORAL)} zero-argument .dt accessors"
    assert len(LIST_METRICS) >= 25, f"only {len(LIST_METRICS)} zero-argument .list accessors"


@pytest.mark.parametrize("name", TEMPORAL)
def test_a_null_timestamp_yields_null(name):
    """Not 0. `year` returning 0 for a null claims the year is zero."""
    assert _temporal(name)[TIMESTAMPS.index(None)] is None


@pytest.mark.parametrize("name", TEMPORAL)
def test_no_boundary_timestamp_raises(name):
    """The epoch, a leap second before midnight, pre-1970, and the nanosecond ceiling."""
    assert len(_temporal(name)) == len(TIMESTAMPS)


@pytest.mark.parametrize("name", LIST_METRICS)
def test_a_null_list_yields_null(name):
    """Not 0 and not `[]`. Both would claim an empty list where there is no list."""
    assert _listwise(name)[LISTS.index(None)] is None


@pytest.mark.parametrize("name", LIST_METRICS)
def test_no_list_shape_raises(name):
    """Empty, single, all-zero, containing a null, and 1e308."""
    assert len(_listwise(name)) == len(LISTS)


class TestEmptyIsNotNull:
    """The distinction this file refuses to flatten. Both answers are right, for different
    questions, and pinning a few of each stops a future change quietly unifying them."""

    @pytest.mark.parametrize(("name", "expected"), [("len", 0), ("n_unique", 0)])
    def test_a_size_of_the_empty_list_is_zero(self, name, expected):
        assert _listwise(name)[LISTS.index([])] == expected

    @pytest.mark.parametrize("name", ["reverse", "drop_nulls", "cum_sum"])
    def test_a_transform_of_the_empty_list_is_the_empty_list(self, name):
        assert _listwise(name)[LISTS.index([])] == []

    def test_the_null_list_is_still_null_for_those_same_methods(self):
        """The control: if null and empty behaved identically the split above would be
        describing nothing."""
        for name in ("len", "reverse", "n_unique"):
            assert _listwise(name)[LISTS.index(None)] is None
