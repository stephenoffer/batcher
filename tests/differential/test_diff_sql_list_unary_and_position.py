"""`list_first`/`list_last`/`list_median`/`list_position` from SQL — vs DuckDB.

Five DuckDB names whose `.list` method already existed and answered correctly, but which
no SQL spelling reached. They were missed by the earlier sweep for the same reason each
time: that sweep paired every `list_X` with its `array_X` twin, and these four have no
twin in DuckDB's catalogue, so they were never proposed.

Two argument shapes, which is why they are two tables rather than one. `list_first(l)` is
a bare unary call; `list_position(l, v)` takes a *literal* value and — unlike
`list_extract` — returns a 1-based index that `.list.position` already produces, so there
is no origin to shift. Getting that wrong is a silent off-by-one, so the fixture includes
a value that is absent (answer NULL) and one at the first position (answer 1, not 0).

The `array_*` vector aliases added alongside are deliberately **not** asserted against
DuckDB here. DuckDB's `array_cosine_similarity` and friends accept only its fixed-size
`ARRAY` type and reject a variable-size `LIST`; Batcher's kernel is the other way round.
The names now reach the right kernel for a Batcher list, which is a strict improvement,
but the DuckDB-typed call is a separate type-support question.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered

QUERIES = [
    "SELECT k, list_first(l) r FROM t ORDER BY k",
    "SELECT k, list_last(l) r FROM t ORDER BY k",
    "SELECT k, list_median(l) r FROM t ORDER BY k",
    "SELECT k, list_position(l, 1) r FROM t ORDER BY k",
    "SELECT k, array_position(l, 1) r FROM t ORDER BY k",
    "SELECT k, list_position(l, 99) r FROM t ORDER BY k",
]


@pytest.fixture
def lists(duck):
    t = pa.table(
        {
            "k": [0, 1, 2, 3],
            # row 0 has the probe value first (position 1, the off-by-one tell), row 1
            # has it last, row 2 lacks it entirely, row 3 is a single element.
            "l": [[1, 3, 2], [3, 2, 1], [5, 4, 6], [1]],
        }
    )
    duck.register("t", t)
    return t


@pytest.mark.differential
@pytest.mark.parametrize("query", QUERIES)
def test_matches_duckdb(duck, lists, query):
    assert_same_ordered(bt.sql(query, t=lists).collect(), duck.sql(query))


@pytest.mark.differential
def test_position_is_one_based_and_null_when_absent(lists):
    """The two answers a 0-based or not-found-as-zero implementation gets wrong."""
    rows = bt.sql("SELECT k, list_position(l, 1) r FROM t ORDER BY k", t=lists).to_pydict()
    assert rows["r"] == [1, 3, None, 1]


@pytest.mark.differential
def test_first_and_last_are_not_min_and_max(lists):
    """`list_first` is positional, not the smallest element — row 0 is `[1, 3, 2]`, so a
    min/max confusion answers 1/3 where the right answers are 1/2."""
    first = bt.sql("SELECT k, list_first(l) r FROM t ORDER BY k", t=lists).to_pydict()["r"]
    last = bt.sql("SELECT k, list_last(l) r FROM t ORDER BY k", t=lists).to_pydict()["r"]
    assert first == [1, 3, 5, 1]
    assert last == [2, 1, 6, 1]
