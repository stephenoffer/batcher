"""Per-column metadata summary equals DuckDB — over a real Parquet footer and in memory.

`answer_column_summary` builds a per-column snapshot (count / null_count / min / max /
n_unique) from EXACT footer statistics, filling only the facets it can prove. Every facet
it returns MUST equal DuckDB's executed aggregate for that column; a facet it omits is a
fallback the caller executes. Checked over a Parquet file (footer drives the snapshot) and
in memory (no footer → falls back to None). Covers NULLs, an all-null column, a constant
column, and int/float/str/date/bool edges. `approx_column_summary`'s SKETCH-backed
`approx_n_unique` is covered separately from the exact path.
"""

from __future__ import annotations

import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher import core
from batcher.api.orchestration import collect_source_stats
from batcher.kyber.metadata_summary import answer_column_summary, approx_column_summary

pytestmark = pytest.mark.differential

_TABLE = pa.table(
    {
        "i": pa.array([3, 1, 2, None, 5], type=pa.int64()),
        "f": pa.array([1.5, None, 2.5, 2.5, 4.0], type=pa.float64()),
        "s": pa.array(["b", "a", None, "a", "c"], type=pa.string()),
        "d": pa.array(
            [
                datetime.date(2024, 1, 3),
                datetime.date(2024, 1, 1),
                None,
                None,
                datetime.date(2024, 1, 5),
            ],
            type=pa.date32(),
        ),
        "b": pa.array([True, False, True, None, True], type=pa.bool_()),
        "allnull": pa.array([None] * 5, type=pa.int64()),
        "k": pa.array([7, 7, 7, 7, 7], type=pa.int64()),
    }
)
_ALL_COLS = ["i", "f", "s", "d", "b", "allnull", "k"]


@pytest.fixture
def pq_path(tmp_path):
    path = str(tmp_path / "t.parquet")
    pq.write_table(_TABLE, path)
    return path


def _duck(con):
    con.register("t", _TABLE)
    return con


def _stats(ds):
    return collect_source_stats(ds._sources, core.default_hub())


def test_summary_facets_match_duckdb(pq_path, duck):
    _duck(duck)
    ds = bt.read.parquet(pq_path)
    summary = answer_column_summary(
        ds._plan, _ALL_COLS, ds._sources, _stats(ds), core.default_hub()
    )
    assert summary is not None
    # At least the numeric columns' bounds/counts are answered from the footer.
    assert {"i", "f", "allnull", "k"} <= set(summary)
    for col, entry in summary.items():
        row = duck.execute(
            f"select count({col}), count(*) - count({col}), min({col}), max({col}), "
            f"count(distinct {col}) from t"
        ).fetchone()
        d_count, d_null, d_min, d_max, d_ndv = row
        # Every facet the snapshot provides must equal DuckDB's executed aggregate.
        if "count" in entry:
            assert entry["count"] == d_count, col
        if "null_count" in entry:
            assert entry["null_count"] == d_null, col
        if "min" in entry:
            assert entry["min"] == d_min, col
        if "max" in entry:
            assert entry["max"] == d_max, col
        if "n_unique" in entry:
            assert entry["n_unique"] == d_ndv, col


def test_allnull_and_constant_columns(pq_path, duck):
    _duck(duck)
    ds = bt.read.parquet(pq_path)
    summary = answer_column_summary(
        ds._plan, ["allnull", "k"], ds._sources, _stats(ds), core.default_hub()
    )
    # An all-null column: count 0, null_count = rows, and no min/max (SQL NULL, omitted).
    assert summary["allnull"]["count"] == 0
    assert summary["allnull"]["null_count"] == 5
    assert "min" not in summary["allnull"]
    # A constant column: min == max == the value, no nulls.
    assert summary["k"]["count"] == 5
    assert summary["k"]["null_count"] == 0
    assert summary["k"]["min"] == 7
    assert summary["k"]["max"] == 7


def test_in_memory_falls_back(pq_path):
    # No footer statistics → nothing EXACT-derivable → None (caller runs real describe).
    ds = bt.from_arrow(_TABLE)
    assert answer_column_summary(ds._plan, _ALL_COLS, ds._sources) is None


def test_filter_downgrades_to_fallback(pq_path):
    # A filter downgrades every column away from EXACT → the snapshot declines.
    ds = bt.read.parquet(pq_path).filter(bt.col("i") > 1)
    assert answer_column_summary(ds._plan, _ALL_COLS, ds._sources, _stats(ds)) is None


def test_approx_summary_adds_sketch_ndv():
    from batcher.kyber.learning import record_column_stats
    from batcher.metadata import MetadataHub
    from batcher.metadata.backends import InProcessBackend

    hub = MetadataHub(InProcessBackend())
    record_column_stats(hub, {"x": 4.0}, {})  # a learned (SKETCH) distinct count
    ds = bt.from_arrow(pa.table({"x": pa.array([1, 2, 2, None], type=pa.int64())}))
    exact = answer_column_summary(ds._plan, ["x"], ds._sources, None, hub)
    approx = approx_column_summary(ds._plan, ["x"], ds._sources, None, hub)
    # Exact `n_unique` is NOT claimed from a sketch; `approx_n_unique` is, and separately.
    assert exact is None
    assert approx == {"x": {"approx_n_unique": 4}}
