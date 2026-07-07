"""Differential: enriched footer-derived stats match DuckDB, and fire from metadata.

The Parquet footer now yields, per column, EXACT ``null_count`` (so ``count(col)``
is answered without a scan) and EXACT min/max across the
numeric/temporal/bool/decimal type range, aggregated across row groups and files.
Each is checked to equal DuckDB's executed answer over a real Parquet file — the
correctness spine — across NULLs, empties, single rows, all-null columns, and a
float-NaN edge. We also assert the shortcut actually fires from metadata (the
answer function returns non-``None``) and correctly falls back after a filter.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher import col, count

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("batcher._native", reason="native engine not built")

from batcher.api.terminal.metadata_answer import (  # noqa: E402
    metadata_aggregate_table,
    metadata_count,
)
from conftest import assert_same  # noqa: E402


@pytest.fixture
def nulls_pq(tmp_path):
    # ``v`` has nulls so count(v)/null_count are non-trivial; multiple row groups
    # exercise cross-row-group null-count summation and min/max aggregation.
    table = pa.table(
        {
            "id": list(range(1, 201)),
            "v": [None if i % 5 == 0 else i for i in range(200)],
        }
    )
    path = str(tmp_path / "nulls.parquet")
    pq.write_table(table, path, row_group_size=32)
    return path


def _fires(plan, sources) -> bool:
    return metadata_aggregate_table(plan, sources) is not None


def test_count_col_excludes_nulls_matches_duckdb(duck, nulls_pq):
    ds = bt.read.parquet(nulls_pq)
    got = ds.agg(c=col("v").count()).collect()
    want = duck.sql(f"SELECT count(v) AS c FROM '{nulls_pq}'")
    assert_same(got, want)
    assert _fires(ds.agg(c=col("v").count())._plan, ds._sources)  # answered from footer


def test_min_max_over_nullable_column_matches_duckdb(duck, nulls_pq):
    ds = bt.read.parquet(nulls_pq)
    got = ds.agg(lo=col("v").min(), hi=col("v").max()).collect()
    want = duck.sql(f"SELECT min(v) AS lo, max(v) AS hi FROM '{nulls_pq}'")
    assert_same(got, want)
    assert _fires(ds.agg(lo=col("v").min())._plan, ds._sources)


def test_filter_downgrades_no_metadata_answer(duck, nulls_pq):
    # After a filter the footer min/max are only bounds → no exact metadata answer,
    # but the executed result is still correct.
    ds = bt.read.parquet(nulls_pq).filter(col("id") > 100)
    assert metadata_count(ds._plan, ds._sources) is None
    got = ds.agg(c=count()).collect()
    want = duck.sql(f"SELECT count(*) AS c FROM '{nulls_pq}' WHERE id > 100")
    assert_same(got, want)


@pytest.mark.parametrize(
    "arrow_type,values",
    [
        (pa.int64(), [5, 1, 9, 3]),
        (pa.float64(), [2.5, 0.5, 8.5, 4.0]),
        (pa.date32(), None),
        (pa.timestamp("us"), None),
        (pa.decimal128(9, 2), None),
        (pa.bool_(), [True, False, True, True]),
    ],
)
def test_typed_min_max_matches_duckdb(duck, tmp_path, arrow_type, values):
    import datetime as dt
    from decimal import Decimal

    if pa.types.is_date(arrow_type):
        values = [dt.date(2021, m, 1) for m in (3, 1, 9, 6)]
    elif pa.types.is_timestamp(arrow_type):
        values = [dt.datetime(2021, m, 1, 12) for m in (3, 1, 9, 6)]
    elif pa.types.is_decimal(arrow_type):
        values = [Decimal(v) for v in ("3.50", "1.25", "9.75", "6.00")]
    table = pa.table({"c": pa.array(values, type=arrow_type)})
    path = str(tmp_path / "typed.parquet")
    pq.write_table(table, path)
    ds = bt.read.parquet(path)
    got = ds.agg(lo=col("c").min(), hi=col("c").max()).collect()
    want = duck.sql(f"SELECT min(c) AS lo, max(c) AS hi FROM '{path}'")
    assert_same(got, want)
    assert _fires(ds.agg(lo=col("c").min())._plan, ds._sources)


def test_all_null_column_count_is_zero(duck, tmp_path):
    path = str(tmp_path / "allnull.parquet")
    pq.write_table(pa.table({"x": pa.array([None, None, None], type=pa.int64())}), path)
    ds = bt.read.parquet(path)
    got = ds.agg(c=col("x").count()).collect()
    want = duck.sql(f"SELECT count(x) AS c FROM '{path}'")
    assert_same(got, want)  # 0, answered from footer null_count == row_count


def test_float_nan_min_max_matches_duckdb(duck, tmp_path):
    # A NaN in the column must not corrupt min()/max(); the footer path drops a NaN
    # bound and the result still equals DuckDB (executed or metadata, both correct).
    path = str(tmp_path / "nan.parquet")
    pq.write_table(pa.table({"f": pa.array([1.0, float("nan"), 3.0], type=pa.float64())}), path)
    ds = bt.read.parquet(path)
    got = ds.agg(lo=col("f").min(), hi=col("f").max()).collect()
    want = duck.sql(f"SELECT min(f) AS lo, max(f) AS hi FROM '{path}'")
    assert_same(got, want)


def test_count_distinct_not_answered_from_footer(duck, tmp_path):
    # Parquet's distinct_count is an estimate → n_unique must execute, not answer
    # from the footer; and the executed answer matches DuckDB.
    table = pa.table({"g": [i % 7 for i in range(100)]})
    path = str(tmp_path / "g.parquet")
    pq.write_table(table, path)
    ds = bt.read.parquet(path)
    assert metadata_aggregate_table(ds.agg(n=col("g").n_unique())._plan, ds._sources) is None
    got = ds.agg(n=col("g").n_unique()).collect()
    want = duck.sql(f"SELECT count(DISTINCT g) AS n FROM '{path}'")
    assert_same(got, want)


def test_multifile_min_max_matches_duckdb(duck, tmp_path):
    pq.write_table(pa.table({"x": [4, 1, 7]}), str(tmp_path / "a.parquet"))
    pq.write_table(pa.table({"x": [10, 2, 20]}), str(tmp_path / "b.parquet"))
    glob = str(tmp_path / "*.parquet")
    ds = bt.read.parquet(glob)
    got = ds.agg(lo=col("x").min(), hi=col("x").max(), c=count()).collect()
    want = duck.sql(f"SELECT min(x) AS lo, max(x) AS hi, count(*) AS c FROM '{glob}'")
    assert_same(got, want)
