"""Differential: metadata-answered terminals match DuckDB.

`count()`, `is_empty()`, and global `min`/`max`/`count(*)` are answered from
Parquet footer statistics without execution. They must equal DuckDB on the same
data — across empty, single-row, all-null, and multi-row-group inputs — exactly
as if they had executed.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher import col, count

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("batcher._native", reason="native engine not built")

from conftest import assert_same  # noqa: E402  (pytest puts the test dir on sys.path)


@pytest.fixture
def people(tmp_path):
    table = pa.table(
        {
            "id": list(range(1, 1001)),
            "age": [20 + (i % 50) for i in range(1000)],
            "dept": [["eng", "sales", "ops"][i % 3] for i in range(1000)],
        }
    )
    path = str(tmp_path / "people.parquet")
    # Multiple row groups exercise cross-row-group footer aggregation.
    pq.write_table(table, path, row_group_size=128)
    return path


def _both(duck, path):
    return bt.read.parquet(path), duck.read_parquet(path)


def test_count_matches_duckdb(duck, people):
    ds, rel = _both(duck, people)
    assert ds.count() == rel.aggregate("count(*)").fetchone()[0]


def test_limit_count_matches_duckdb(duck, people):
    ds, _ = _both(duck, people)
    for n in (0, 1, 10, 999, 5000):
        expected = duck.sql(
            f"SELECT count(*) FROM (SELECT * FROM '{people}' LIMIT {n})"
        ).fetchone()[0]
        assert ds.limit(n).count() == expected


def test_global_aggregate_matches_duckdb(duck, people):
    ds, _ = _both(duck, people)
    got = ds.agg(lo=col("age").min(), hi=col("age").max(), c=count()).collect()
    want = duck.sql(f"SELECT min(age) AS lo, max(age) AS hi, count(*) AS c FROM '{people}'")
    assert_same(got, want)


def test_is_empty_matches_duckdb(duck, people):
    ds, _ = _both(duck, people)
    assert ds.is_empty() is False
    assert ds.limit(0).is_empty() is True
    # A contradictory filter executes (no metadata answer) but is still correct.
    filtered = ds.filter(col("age") > 10_000)
    empty = duck.sql(f"SELECT count(*) FROM '{people}' WHERE age > 10000").fetchone()[0] == 0
    assert filtered.is_empty() == empty


def test_empty_source_count(duck, tmp_path):
    path = str(tmp_path / "empty.parquet")
    pq.write_table(pa.table({"x": pa.array([], type=pa.int64())}), path)
    ds = bt.read.parquet(path)
    assert ds.count() == 0
    assert ds.is_empty() is True


def test_single_row_count(duck, tmp_path):
    path = str(tmp_path / "one.parquet")
    pq.write_table(pa.table({"x": [42]}), path)
    ds = bt.read.parquet(path)
    assert ds.count() == 1
    assert ds.agg(mx=col("x").max()).to_pydict() == {"mx": [42]}


def test_zonemap_pruned_filters_match_duckdb(duck, people):
    ds, _ = _both(duck, people)
    cases = {
        "age < 0": col("age") < 0,  # always false (min age 20) → empty
        "age < 1000": col("age") < 1000,  # always true → drop filter
        "age >= 0": col("age") >= 0,  # always true
        "age = 999": col("age") == 999,  # outside range → empty
        "age > 30": col("age") > 30,  # satisfiable → unchanged, must still match
        "age < 0 OR age > 30": (col("age") < 0) | (col("age") > 30),
    }
    for where, predicate in cases.items():
        got = ds.filter(predicate).collect()
        want = duck.sql(f"SELECT * FROM '{people}' WHERE {where}")
        assert_same(got, want)
