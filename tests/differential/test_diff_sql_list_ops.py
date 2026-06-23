"""SQL list/array operations (length, index, reductions, contains) vs DuckDB."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture
def t(duck):
    tbl = pa.table({"id": [1, 2, 3], "x": [10, 20, 30], "y": [3, 1, 2]})
    duck.register("t", tbl)
    return tbl


@pytest.mark.parametrize(
    "q",
    [
        "SELECT array_length([x, y]) n FROM t",
        "SELECT [x, y, id][1] e FROM t",
        "SELECT [x, y, id][2] e FROM t",
        "SELECT list_contains([x, y], 10) c FROM t",
        "SELECT list_contains([x, y], 99) c FROM t",
        "SELECT array_length(list_reverse([x, y, id])) n FROM t",
    ],
)
def test_list_ops(duck, t, q):
    from conftest import assert_same

    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        "SELECT list_sum([x, y]) s FROM t",
        "SELECT list_min([x, y, id]) m FROM t",
        "SELECT list_max([x, y, id]) m FROM t",
        "SELECT list_sum(array_agg(x)) s FROM t",
    ],
)
def test_list_reductions(duck, t, q):
    from conftest import assert_same

    # assert_same tolerates int/Decimal vs float (list reductions cast to float).
    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))
