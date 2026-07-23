"""The metadata provably-empty short-circuit returns the exact empty result, scan-free.

`_collect` answers a query metadata proves empty (contradiction filter, `limit(0)`,
project/aggregate over an empty subtree) by returning an empty table of the output
schema — no engine execution. The result (schema + zero rows) MUST equal DuckDB's.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

_DATA = pa.table({"a": [1, 2, 3, 4, 5], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
_CONTRA = (col("a") > 100) & (col("a") < 0)


@pytest.fixture
def t(duck):
    duck.register("t", _DATA)
    return _DATA


def test_contradiction_collect_matches_duckdb(duck, t):
    out = bt.from_arrow(t).filter(_CONTRA).collect()
    assert out.num_rows == 0
    assert out.schema.names == ["a", "b"]
    assert_same(out, duck.sql("SELECT * FROM t WHERE a > 100 AND a < 0"))


def test_limit_zero_collect(duck, t):
    out = bt.from_arrow(t).limit(0).collect()
    assert out.num_rows == 0 and out.schema.names == ["a", "b"]


def test_project_over_empty_schema(duck, t):
    out = bt.from_arrow(t).filter(_CONTRA).select(c=col("a") + 1, d=col("b") * 2).collect()
    assert out.num_rows == 0 and out.schema.names == ["c", "d"]
    assert_same(out, duck.sql("SELECT a + 1 AS c, b * 2 AS d FROM t WHERE a > 100 AND a < 0"))


def test_nonempty_query_still_executes(duck, t):
    out = bt.from_arrow(t).filter(col("a") > 2).collect()
    assert out.num_rows == 3
    assert_same(out, duck.sql("SELECT * FROM t WHERE a > 2"))
