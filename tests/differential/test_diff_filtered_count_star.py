"""A filtered ``COUNT(*) WHERE col = v`` / ``col <> v`` matches DuckDB exactly.

Batcher answers this shape from metadata — a learned per-value count over an immutable
in-memory source — instead of scanning. These cases pin that the metadata answer equals
the executed (DuckDB) answer across the SQL-null edges: an equality that matches, one
that matches nothing, the complementary inequality (which excludes nulls, per SQL
``NULL <> v`` being unknown), an all-null column, and a value outside the column range.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


def _hits(duck):
    t = pa.table({"AdvEngineID": pa.array([0, 0, 2, 3, None], type=pa.int64())})
    duck.register("hits", t)
    return bt.from_arrow(t)


def _duck_count(duck, where: str) -> int:
    return duck.sql(f"SELECT COUNT(*) FROM hits WHERE {where}").fetchone()[0]


def test_ne_excludes_nulls(duck):
    assert _hits(duck).filter(col("AdvEngineID") != 0).count() == _duck_count(
        duck, "AdvEngineID <> 0"
    )


def test_eq_interior(duck):
    assert _hits(duck).filter(col("AdvEngineID") == 2).count() == _duck_count(
        duck, "AdvEngineID = 2"
    )


def test_eq_out_of_range(duck):
    assert _hits(duck).filter(col("AdvEngineID") == 99).count() == _duck_count(
        duck, "AdvEngineID = 99"
    )


def test_ne_out_of_range_still_excludes_nulls(duck):
    assert _hits(duck).filter(col("AdvEngineID") != 99).count() == _duck_count(
        duck, "AdvEngineID <> 99"
    )


@pytest.mark.parametrize(
    ("predicate", "where"),
    [
        (col("AdvEngineID") < 2, "AdvEngineID < 2"),
        (col("AdvEngineID") <= 2, "AdvEngineID <= 2"),
        (col("AdvEngineID") > 0, "AdvEngineID > 0"),
        (col("AdvEngineID") >= 2, "AdvEngineID >= 2"),
        (col("AdvEngineID") < 0, "AdvEngineID < 0"),  # boundary: empty from EXACT min
        (col("AdvEngineID") >= 100, "AdvEngineID >= 100"),  # boundary: empty from EXACT max
    ],
)
def test_range_comparisons_match_duckdb(duck, predicate, where):
    # A range filter-count over the interior (partial overlap) is answered from the learned
    # per-predicate count; at a boundary (provably empty) from the EXACT min/max bounds.
    ds = _hits(duck)
    assert ds.filter(predicate).count() == _duck_count(duck, where)


def test_all_null_column(duck):
    t = pa.table({"x": pa.array([None, None, None], type=pa.int64())})
    duck.register("nulls", t)
    got = bt.from_arrow(t).filter(col("x") != 5).count()
    assert got == duck.sql("SELECT COUNT(*) FROM nulls WHERE x <> 5").fetchone()[0]


def test_sql_count_star_matches(duck):
    _hits(duck)
    sess = bt.Session()
    sess.register(
        "hits",
        bt.from_arrow(pa.table({"AdvEngineID": pa.array([0, 0, 2, 3, None], type=pa.int64())})),
    )
    got = sess.sql("SELECT COUNT(*) AS n FROM hits WHERE AdvEngineID <> 0").to_pylist()[0]["n"]
    assert got == _duck_count(duck, "AdvEngineID <> 0")
