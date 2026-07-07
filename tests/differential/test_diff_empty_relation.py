"""`empty_relation` rules keep results identical to DuckDB.

A contradictory filter (`a > 100 AND a < 0`) is reduced to a constant-FALSE predicate by
`predicate_infer` and then to the empty marker by `filter_false_to_empty`; the marker
propagates through Project/grouped-Aggregate/Window. Each shape must still match DuckDB —
including the global-aggregate case, which stays one row (never folds to empty).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.extra.empty_relation
from batcher import col

_DATA = pa.table({"a": [1, 2, 3, 4, 5], "b": [10, 10, 20, 20, 30]})


@pytest.fixture
def t(duck):
    duck.register("t", _DATA)
    return _DATA


_CONTRADICTION = (col("a") > 100) & (col("a") < 0)
_WHERE = "WHERE a > 100 AND a < 0"


def test_filter_contradiction_empty(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).filter(_CONTRADICTION).collect()
    assert_same(out, duck.sql(f"SELECT * FROM t {_WHERE}"))


def test_project_over_empty(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).filter(_CONTRADICTION).select(c=col("a") + 1).collect()
    assert_same(out, duck.sql(f"SELECT a + 1 AS c FROM t {_WHERE}"))


def test_grouped_aggregate_over_empty(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).filter(_CONTRADICTION).group_by("b").agg(s=col("a").sum()).collect()
    assert_same(out, duck.sql(f"SELECT b, sum(a) AS s FROM t {_WHERE} GROUP BY b"))


def test_global_aggregate_over_empty_is_one_row(duck, t):
    # The guard: a keyless aggregate over empty input is ONE row (sum NULL), not empty.
    from conftest import assert_same

    out = bt.from_arrow(t).filter(_CONTRADICTION).group_by().agg(s=col("a").sum()).collect()
    expected = duck.sql(f"SELECT sum(a) AS s FROM t {_WHERE}")
    assert out.num_rows == 1
    assert_same(out, expected)


def test_window_over_empty(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .filter(_CONTRADICTION)
        .window(partition_by=["b"], functions={"w": ("sum", "a")})
        .collect()
    )
    assert_same(out, duck.sql(f"SELECT *, sum(a) OVER (PARTITION BY b) AS w FROM t {_WHERE}"))
