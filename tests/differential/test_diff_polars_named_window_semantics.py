"""The Polars-named window helpers follow SQL, and this is where that is held.

`cum_sum`, `rolling_mean`, and `rank` carry Polars spellings and DuckDB semantics, and the
two disagree on exactly the inputs that make the difference invisible: a partial leading
frame, a null inside the frame, and a null being ranked. A port from Polars gets different
numbers with no error, which is why the divergence is documented in
`migrate-from-polars-or-pandas` -- and why "fixing" these toward Polars would be a
regression against the oracle rather than a bug fix. Pinned against DuckDB so that cannot
happen quietly.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential

# NaN and null in one float column, which is what separates the two semantics.
_ROWS = {"i": [0, 1, 2, 3, 4], "x": [1.0, float("nan"), 3.0, None, 5.0]}


def _norm(values: list) -> list:
    """NaN compares unequal to itself, so name it before comparing."""
    return ["nan" if isinstance(v, float) and math.isnan(v) else v for v in values]


@pytest.fixture
def registered(duck):
    duck.register("t", pa.table(_ROWS))
    return duck


def _oracle(duck, query: str) -> list:
    return _norm([row[0] for row in duck.execute(query).fetchall()])


def _batcher(expression) -> list:
    ds = bt.from_pydict(_ROWS).sort("i")
    return _norm(ds.select(r=expression).to_pydict()["r"])


def test_cum_sum_is_an_unbounded_preceding_sum(registered):
    want = _oracle(
        registered,
        "SELECT sum(x) OVER (ORDER BY i ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) "
        "FROM t ORDER BY i",
    )
    assert _batcher(bt.col("x").cum_sum()) == want


def test_rolling_mean_gives_the_partial_leading_frame_a_value(registered):
    want = _oracle(
        registered,
        "SELECT avg(x) OVER (ORDER BY i ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) "
        "FROM t ORDER BY i",
    )
    assert _batcher(bt.col("x").rolling_mean(2)) == want
    # The distinguishing values, spelled out: Polars returns null at 0, 3 and 4.
    assert want[0] is not None


def test_rank_ranks_the_null_row_and_returns_an_integer(registered):
    want = _oracle(registered, "SELECT rank() OVER (ORDER BY x) FROM t ORDER BY i")
    got = _batcher(bt.col("x").rank())
    assert got == want
    assert None not in got, "SQL ranks a null row; Polars returns null for it"
    assert all(isinstance(v, int) for v in got), "RANK() is integral"
