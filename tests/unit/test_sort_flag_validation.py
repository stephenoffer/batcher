"""`sort`'s `descending`/`nulls_first` flags are checked at the API edge.

Every wrong shape used to be reported somewhere unhelpful, and a string was the worst of
them: `str` is iterable, so `sort("a", descending="maybe")` measured `len("maybe")` and
answered "descending list has 5 entries but there are 1 keys" -- a sentence about a list
the caller never wrote. A non-sequence fell through to a bare
`TypeError: object of type 'int' has no len()`. A list of non-bools reached the engine,
which rejected it as "malformed plan IR: invalid type: string, expected a boolean" -- an
IR diagnostic for a typo in a keyword argument.

No shape ever produced a *wrong sort*, because the IR deserializer refused the bad values;
the defect was that three ordinary mistakes produced three unrecognizable reports, one of
them only after the scan had run.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit


def _ds():
    return bt.from_pydict({"a": [1, 2, 3], "b": [3, 2, 1]})


@pytest.mark.parametrize("flag", ["maybe", "ab", "", 2, 0, None, 1.5, object()])
def test_a_non_bool_non_sequence_flag_is_refused_by_shape(flag):
    with pytest.raises(PlanError, match="must be a bool, or a list of bools"):
        _ds().sort(["a", "b"], descending=flag)


def test_a_string_is_not_read_as_a_sequence_of_flags():
    """The specific misreading: `len("ab") == 2` matched the key count exactly."""
    with pytest.raises(PlanError, match="got str 'ab'"):
        _ds().sort(["a", "b"], descending="ab")


@pytest.mark.parametrize("flag", [[1, 0], [True, "x"], [None, None]])
def test_a_list_of_non_bools_is_refused_before_the_engine_sees_it(flag):
    with pytest.raises(PlanError, match="must contain only bools"):
        _ds().sort(["a", "b"], descending=flag)


def test_a_length_mismatch_still_reports_the_count():
    with pytest.raises(PlanError, match="entries but there are 2 keys"):
        _ds().sort(["a", "b"], descending=[True])


def test_nulls_first_is_validated_the_same_way():
    with pytest.raises(PlanError, match="nulls_first must be a bool"):
        _ds().sort("a", nulls_first="yes")


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        (True, [3, 2, 1]),
        (False, [1, 2, 3]),
        ([True, True], [3, 2, 1]),
        ((True, True), [3, 2, 1]),
        ([False, False], [1, 2, 3]),
    ],
)
def test_every_valid_shape_still_sorts(flag, expected):
    """A bool broadcasts, and a list or tuple of bools is taken one per key."""
    assert _ds().sort(["a", "b"], descending=flag).to_pydict()["a"] == expected


def test_nulls_first_still_works():
    ds = bt.from_pydict({"a": [2, None, 1]})
    assert ds.sort("a", nulls_first=True).to_pydict()["a"] == [None, 1, 2]
    assert ds.sort("a", nulls_first=False).to_pydict()["a"] == [1, 2, None]
