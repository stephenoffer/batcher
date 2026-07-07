"""The per-column summary fills only EXACT facets, and falls back otherwise.

Pins the provenance firewall of `metadata_summary`: a Parquet footer answers count /
null_count / min / max with no scan, a filter (which downgrades away from EXACT) makes the
snapshot decline (`None`), and the exact `n_unique` never comes from a sketch — only the
explicitly-approximate `approx_n_unique` does. Exercises the pure `_exact_entry` helper
directly plus a real footer end to end.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher import core
from batcher.api.orchestration import collect_source_stats
from batcher.kyber.metadata_summary import answer_column_summary, approx_column_summary
from batcher.kyber.metadata_summary.answers import _exact_entry
from batcher.plan.stats import ColumnStat, Provenance, RelStats

pytestmark = pytest.mark.unit


def _rel(rows, prov, **col):
    return RelStats(float(rows), prov, {"x": ColumnStat(**col)})


def test_exact_entry_full_bundle():
    rel = _rel(10, Provenance.EXACT, min=1, max=9, null_count=2, ndv=7, provenance=Provenance.EXACT)
    assert _exact_entry(rel, "x") == {
        "null_count": 2,
        "count": 8,  # rows - null_count
        "min": 1,
        "max": 9,
        "n_unique": 7,
    }


def test_exact_entry_omits_count_when_rows_inexact():
    # Column bundle EXACT but the relation's row count is not → no derived non-null count.
    rel = RelStats(
        10.0,
        Provenance.DEFAULT,
        {"x": ColumnStat(min=1, max=9, null_count=2, provenance=Provenance.EXACT)},
    )
    entry = _exact_entry(rel, "x")
    assert "count" not in entry
    assert entry["null_count"] == 2
    assert entry["min"] == 1


def test_exact_entry_omits_absent_facets():
    # An all-null column: null_count present, min/max/ndv absent (omitted, not guessed).
    rel = _rel(5, Provenance.EXACT, null_count=5, provenance=Provenance.EXACT)
    assert _exact_entry(rel, "x") == {"null_count": 5, "count": 0}


def test_downgraded_bundle_yields_nothing():
    rel = _rel(10, Provenance.DEFAULT, min=1, max=9, null_count=2, provenance=Provenance.DEFAULT)
    assert _exact_entry(rel, "x") == {}


@pytest.fixture
def pq_path(tmp_path):
    table = pa.table(
        {
            "x": pa.array([3, 1, 2, None, 5], type=pa.int64()),
            "allnull": pa.array([None] * 5, type=pa.int64()),
        }
    )
    path = str(tmp_path / "t.parquet")
    pq.write_table(table, path)
    return path


def _stats(ds):
    return collect_source_stats(ds._sources, core.default_hub())


def test_footer_summary_fires(pq_path):
    ds = bt.read.parquet(pq_path)
    summary = answer_column_summary(ds._plan, ["x"], ds._sources, _stats(ds), core.default_hub())
    assert summary == {"x": {"null_count": 1, "count": 4, "min": 1, "max": 5}}


def test_footer_summary_omits_exact_ndv(pq_path):
    # Parquet footers carry no EXACT distinct count → `n_unique` is not claimed.
    ds = bt.read.parquet(pq_path)
    summary = answer_column_summary(ds._plan, ["x"], ds._sources, _stats(ds), core.default_hub())
    assert "n_unique" not in summary["x"]


def test_filter_downgrades_summary(pq_path):
    ds = bt.read.parquet(pq_path).filter(bt.col("x") > 1)
    assert answer_column_summary(ds._plan, ["x"], ds._sources, _stats(ds)) is None


def test_empty_columns_returns_none(pq_path):
    ds = bt.read.parquet(pq_path)
    assert answer_column_summary(ds._plan, [], ds._sources, _stats(ds)) is None


def test_approx_variant_carries_exact_and_sketch(pq_path):
    ds = bt.read.parquet(pq_path)
    approx = approx_column_summary(ds._plan, ["x"], ds._sources, _stats(ds), core.default_hub())
    # Exact facets still present; no sketch ndv recorded here → no approx key.
    assert approx["x"]["min"] == 1
    assert "approx_n_unique" not in approx["x"]
