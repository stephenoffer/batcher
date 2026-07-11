"""DuckDB COLUMNS(*) / COLUMNS('regex') dynamic-column expressions vs DuckDB.

Batcher maps ``COLUMNS(...)`` onto its DataFrame column selectors, so a projection
over a dynamic set of columns — including a scalar function applied to each — must
match DuckDB, keeping each source column's name.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture
def wide(duck):
    t = pa.table(
        {
            "sales_q1": [10, 20, 30],
            "sales_q2": [40, 50, 60],
            "region": ["us", "eu", "us"],
            "cost": [1.5, 2.5, 3.5],
        }
    )
    duck.register("t", t)
    return t


@pytest.mark.differential
@pytest.mark.parametrize(
    "query",
    [
        "SELECT COLUMNS(*) FROM t",
        "SELECT COLUMNS('sales_.*') FROM t",
        "SELECT region, COLUMNS('sales_.*') FROM t",
        "SELECT COLUMNS('sales_.*') * 2 FROM t",
        "SELECT COLUMNS('^s') FROM t",
    ],
)
def test_columns_matches_duckdb(duck, wide, query):
    from conftest import assert_same

    out = bt.sql(query, t=wide).collect()
    duck_table = duck.sql(query).to_arrow_table()
    # COLUMNS keeps each matched source column's name, in schema order.
    assert out.column_names == duck_table.column_names
    assert_same(out, duck.sql(query))


@pytest.mark.differential
def test_columns_with_no_match_raises_not_silent(wide):
    """A COLUMNS regex matching nothing raises rather than producing a phantom result."""
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError, match="at least one column"):
        bt.sql("SELECT COLUMNS('^nope') FROM t", t=wide).collect()
