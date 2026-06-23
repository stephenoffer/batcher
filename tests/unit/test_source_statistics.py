"""Connector metadata extraction: footers and headers become `SourceStatistics`.

Covers the testable, highest-value paths — Parquet footer column stats and NumPy
``.npy`` header counts — end to end, including that they flow into the estimator
as EXACT row counts and column bounds. Lakehouse/SQL extractors require external
services and are exercised by their own (skipped-without-deps) suites.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher.io.formats.structured.parquet import ParquetSource
from batcher.io.source import source_statistics
from batcher.io.stats.free_counts import npy_header_rows
from batcher.kyber.stats import StatsEstimator
from batcher.plan.stats import Provenance


@pytest.fixture
def parquet_file(tmp_path):
    table = pa.table({"x": list(range(100)), "label": ["a", "b"] * 50})
    path = str(tmp_path / "data.parquet")
    pq.write_table(table, path)
    return path


def test_parquet_statistics_exact_count_and_numeric_bounds(parquet_file):
    src = ParquetSource(parquet_file)
    stats = src.statistics()
    assert stats is not None
    assert stats.row_count == 100 and stats.exact_rows
    x = stats.columns["x"]
    assert x.min == 0 and x.max == 99
    assert x.null_count == 0
    assert x.provenance is Provenance.EXACT  # numeric footer min/max is exact


def test_parquet_string_bounds_not_exact(parquet_file):
    # String min/max may be writer-truncated → usable as bounds, never as exact max().
    stats = ParquetSource(parquet_file).statistics()
    label = stats.columns["label"]
    assert label.provenance is not Provenance.EXACT


def test_parquet_multifile_aggregates(tmp_path):
    pq.write_table(pa.table({"x": [1, 2, 3]}), str(tmp_path / "a.parquet"))
    pq.write_table(pa.table({"x": [10, 20]}), str(tmp_path / "b.parquet"))
    stats = ParquetSource(str(tmp_path)).statistics()
    assert stats.row_count == 5
    assert stats.columns["x"].min == 1 and stats.columns["x"].max == 20


def test_parquet_stats_feed_estimator_exact(parquet_file):
    # A scan over the parquet file estimates EXACT rows and carries column bounds.
    ds = bt.read.parquet(parquet_file)
    src_stats = [source_statistics(s) for s in ds._sources]
    rs = StatsEstimator(ds._sources, source_stats=src_stats).estimate(ds._plan)
    assert rs.rows == 100 and rs.rows_exact
    assert rs.column("x").max == 99 and rs.column("x").provenance is Provenance.EXACT


def test_npy_header_row_count(tmp_path):
    path = tmp_path / "arr.npy"
    np.save(path, np.arange(42 * 3).reshape(42, 3))
    with open(path, "rb") as fh:
        assert npy_header_rows(fh) == 42


def test_numpy_source_statistics(tmp_path):
    path = tmp_path / "arr.npy"
    np.save(path, np.arange(17))
    from batcher.io.formats.ml.numpy import NumpySource

    stats = NumpySource(str(path)).statistics()
    assert stats is not None and stats.row_count == 17 and stats.exact_rows


def test_source_statistics_helper_falls_back_to_row_count():
    # An in-memory source has no statistics() but a known row_count → wrapped.
    ds = bt.from_pydict({"a": [1, 2, 3]})
    stats = source_statistics(ds._sources[0])
    assert stats is not None and stats.row_count == 3
