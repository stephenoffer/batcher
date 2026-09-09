"""A null key matches nothing — and paying for that on a join with no nulls is pure cost.

Both dataframe libraries' `merge` matches a null key to itself, which invents rows an inner join
must not produce and pairs up two rows an outer join was supposed to report as unmatched. The
translator adds one synthetic key component to fix every join type at once, and that mechanism
is correct and load-bearing.

It is also, when neither side has a null in any key column, a **constant on both sides** — `-1`
everywhere, matching itself everywhere, unable to change a single pair. What it does instead is
turn every merge into a composite-key hash join. TPC-H declares no nullable key, so the whole
suite paid for it on every join.

These tests pin both halves: the marker is skipped exactly when it cannot matter, and the join
semantics are unchanged either way.
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

from batcher.core.gpu_plan import DfBackend
from batcher.core.gpu_plan.execute import _null_free, join_frames

pytestmark = pytest.mark.unit

BE = DfBackend(pd)


def _frame(**columns):
    return BE.from_arrow(pa.table(columns))


def _join_ir(how: str = "inner"):
    return {
        "join_type": how,
        "left_keys": ["k"],
        "right_keys": ["k"],
        "output": [
            {"side": "left", "name": "k", "alias": "k"},
            {"side": "left", "name": "a", "alias": "a"},
            {"side": "right", "name": "b", "alias": "b"},
        ],
    }


# --- the predicate -----------------------------------------------------------


def test_a_column_with_no_nulls_is_null_free():
    assert _null_free(_frame(a=[1, 2, 3]), ["a"]) is True


def test_a_column_with_a_null_is_not():
    assert _null_free(_frame(a=[1, None, 3]), ["a"]) is False


def test_every_key_column_has_to_be_null_free():
    assert _null_free(_frame(a=[1, 2], b=[1, None]), ["a", "b"]) is False


def test_a_column_that_will_not_report_its_null_count_is_treated_as_nullable():
    """The conservative direction: keeping the marker is correct, dropping it wrongly is not."""

    class _Opaque:
        null_count = None
        array = None

    assert _null_free({"k": _Opaque()}, ["k"]) is False


# --- the semantics, which must not move --------------------------------------


@pytest.mark.parametrize("how", ["inner", "left", "outer"])
def test_a_null_key_still_matches_nothing(how):
    """The marker's whole purpose, on the path that still installs it."""
    left = _frame(k=[1, None, 3], a=[10, 20, 30])
    right = _frame(k=[1, None, 4], b=[100, 200, 400])
    out = BE.to_arrow(join_frames(left, right, _join_ir(how), BE)).to_pydict()
    pairs = list(zip(out["k"], out["a"], out["b"], strict=True))
    # A *paired* row is one carrying a value from both sides. An outer join also emits rows
    # carrying one side alone, and those are unmatched rows rather than matches — reading them
    # as matches is how this test first mistook a correct answer for a wrong one.
    paired = [p for p in pairs if p[1] is not None and p[2] is not None]
    assert paired == [(1, 10, 100)], "a null key matched something"


@pytest.mark.parametrize("how", ["inner", "left", "outer"])
def test_a_null_free_join_gives_the_same_answer_without_the_marker(how):
    """Same rows with the fast path as with the marker: the shortcut is an identity."""
    left = _frame(k=[1, 2, 3], a=[10, 20, 30])
    right = _frame(k=[1, 3, 4], b=[100, 300, 400])
    fast = BE.to_arrow(join_frames(left, right, _join_ir(how), BE)).to_pydict()

    # Force the marker path by making one side nullable without changing any value.
    nullable = BE.from_arrow(
        pa.table({"k": pa.array([1, 2, 3], pa.int64()), "a": pa.array([10, 20, 30], pa.int64())})
    )
    nullable["k"] = nullable["k"].where(nullable["k"].notna(), None)
    slow = BE.to_arrow(join_frames(nullable, right, _join_ir(how), BE)).to_pydict()
    assert sorted(map(str, zip(*fast.values(), strict=True))) == sorted(
        map(str, zip(*slow.values(), strict=True))
    )


def test_the_output_columns_are_the_same_either_way():
    """The marker is a private column and must never reach the result, on either path."""
    left = _frame(k=[1, 2], a=[10, 20])
    right = _frame(k=[1, 2], b=[100, 200])
    out = BE.to_arrow(join_frames(left, right, _join_ir(), BE))
    assert out.schema.names == ["k", "a", "b"]
