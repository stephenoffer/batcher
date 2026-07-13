"""Unit tests for the footer/manifest statistics enrichment.

Covers the pure extraction functions in `batcher.io.stats` directly (no engine):

  - the Parquet footer's provenance discipline (ndv never rides an EXACT column;
    a NaN float bound is dropped; null_count summed exactly),
  - the Delta manifest's partition-value EXACT stats and numeric-exact bounds,
  - the per-row-group pruning bounds and range-pruning row count,
  - the ORC footer's exact row count.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batcher.io.stats.columnar_footer import (
    _accumulate,
    _ColAcc,
    _finalize_columns,
    orc_statistics,
    parquet_statistics,
)
from batcher.io.stats.lakehouse_manifest import manifest_statistics
from batcher.io.stats.pruning import parquet_row_group_bounds, surviving_rows_for_range
from batcher.plan.stats import Provenance


class _FakeStats:
    """A duck-typed stand-in for a pyarrow column-chunk `statistics` object."""

    def __init__(self, cmin, cmax, null_count, *, has_min_max=True, distinct_count=None):
        self.min = cmin
        self.max = cmax
        self.null_count = null_count
        self.has_null_count = null_count is not None
        self.has_min_max = has_min_max
        self.distinct_count = distinct_count


class _FakeCol:
    def __init__(self, stats, num_values=10):
        self.statistics = stats
        self.num_values = num_values


class _FakeFS:
    """Minimal filesystem exposing `open(path)` over local files."""

    def open(self, path):
        return open(path, "rb")


def _finalize(acc, schema, *, single_row_group=True):
    return _finalize_columns({"c": acc}, schema, single_row_group=single_row_group)


# --- Parquet footer provenance discipline -----------------------------------


def test_ndv_never_on_exact_column():
    # A numeric column with a footer distinct_count stays EXACT for min/max/null,
    # but its (estimate) ndv is dropped so it can never answer an exact distinct.
    acc = _ColAcc()
    _accumulate(acc, _FakeCol(_FakeStats(0, 99, 0, distinct_count=50)))
    col = _finalize(acc, pa.schema([pa.field("c", pa.int64())]))["c"]
    assert col.provenance is Provenance.EXACT
    assert col.min == 0 and col.max == 99 and col.null_count == 0
    assert col.ndv is None  # the estimate is not exposed on an EXACT bundle


def test_ndv_kept_on_inexact_string_column():
    # A string column is already inexact (truncatable), so its ndv may ride along
    # for cost / approx_count_distinct without risking an exact answer.
    acc = _ColAcc()
    _accumulate(acc, _FakeCol(_FakeStats("a", "z", 0, distinct_count=12)))
    col = _finalize(acc, pa.schema([pa.field("c", pa.string())]))["c"]
    assert col.provenance is Provenance.DEFAULT
    assert col.ndv == 12.0


def test_float_nan_bound_is_dropped():
    # A NaN min/max is unordered → bounds dropped so an exact min()/max() falls
    # back rather than returning a wrong extreme; null_count stays exact.
    acc = _ColAcc()
    _accumulate(acc, _FakeCol(_FakeStats(math.nan, 5.0, 2)))
    col = _finalize(acc, pa.schema([pa.field("c", pa.float64())]))["c"]
    assert col.min is None and col.max is None
    assert col.null_count == 2.0


def test_null_count_summed_across_chunks_exact():
    acc = _ColAcc()
    _accumulate(acc, _FakeCol(_FakeStats(0, 5, 3)))
    _accumulate(acc, _FakeCol(_FakeStats(6, 9, 4)))
    col = _finalize(acc, pa.schema([pa.field("c", pa.int64())]), single_row_group=False)["c"]
    assert col.null_count == 7.0
    assert col.min == 0 and col.max == 9
    assert col.provenance is Provenance.EXACT


def test_unknown_null_count_downgrades():
    # A chunk with no null_count makes the summed count unknown → not EXACT.
    acc = _ColAcc()
    _accumulate(acc, _FakeCol(_FakeStats(0, 5, None)))
    col = _finalize(acc, pa.schema([pa.field("c", pa.int64())]))["c"]
    assert col.null_count is None
    assert col.provenance is Provenance.DEFAULT


def test_parquet_statistics_temporal_and_bool_exact(tmp_path):
    import datetime as dt

    table = pa.table(
        {
            "d": pa.array([dt.date(2020, 1, 1), dt.date(2020, 6, 1)], type=pa.date32()),
            "b": pa.array([True, False]),
        }
    )
    path = str(tmp_path / "t.parquet")
    pq.write_table(table, path)
    stats = parquet_statistics(_FakeFS(), [path], table.schema)
    assert stats.columns["d"].provenance is Provenance.EXACT
    assert stats.columns["b"].provenance is Provenance.EXACT
    assert stats.columns["b"].min is False and stats.columns["b"].max is True


# --- Delta manifest partition + numeric-exact enrichment --------------------


def _add_actions(**cols):
    return pa.table(cols)


def test_delta_partition_column_is_exact():
    add = _add_actions(
        num_records=pa.array([10, 20], type=pa.int64()),
        **{"partition.region": pa.array(["us", "eu"])},
    )
    stats = manifest_statistics(add)
    assert stats.row_count == 30
    region = stats.columns["region"]
    assert region.provenance is Provenance.EXACT  # literal partition value, untruncated
    assert region.min == "eu" and region.max == "us"


def test_delta_single_partition_value_min_equals_max():
    add = _add_actions(
        num_records=pa.array([5, 5], type=pa.int64()),
        **{"partition.day": pa.array(["2020-01-01", "2020-01-01"])},
    )
    day = manifest_statistics(add).columns["day"]
    assert day.min == day.max == "2020-01-01"
    assert day.provenance is Provenance.EXACT


def test_delta_numeric_column_exact_when_all_files_recorded():
    add = _add_actions(
        num_records=pa.array([2, 3], type=pa.int64()),
        **{
            "min.x": pa.array([1, 10], type=pa.int64()),
            "max.x": pa.array([9, 40], type=pa.int64()),
            "null_count.x": pa.array([0, 1], type=pa.int64()),
        },
    )
    x = manifest_statistics(add).columns["x"]
    assert x.min == 1 and x.max == 40
    assert x.null_count == 1.0
    assert x.provenance is Provenance.EXACT


def test_delta_numeric_column_downgraded_when_a_file_lacks_stats():
    add = _add_actions(
        num_records=pa.array([2, 3], type=pa.int64()),
        **{
            "min.x": pa.array([1, None], type=pa.int64()),  # second file has no min
            "max.x": pa.array([9, 40], type=pa.int64()),
        },
    )
    x = manifest_statistics(add).columns["x"]
    assert x.provenance is Provenance.DEFAULT  # aggregate min may be wrong → bound only


def test_delta_string_column_is_bound_only():
    add = _add_actions(
        num_records=pa.array([2], type=pa.int64()),
        **{"min.name": pa.array(["aaa"]), "max.name": pa.array(["zzz"])},
    )
    name = manifest_statistics(add).columns["name"]
    assert name.provenance is Provenance.DEFAULT  # writer-truncated → never exact


# --- Row-group pruning surface ----------------------------------------------


@pytest.fixture
def multi_rg(tmp_path):
    table = pa.table({"x": list(range(100))})
    path = str(tmp_path / "rg.parquet")
    pq.write_table(table, path, row_group_size=25)  # 4 row groups
    return path


def test_row_group_bounds_one_entry_per_group(multi_rg):
    bounds = parquet_row_group_bounds(_FakeFS(), [multi_rg], ["x"])
    assert len(bounds) == 4
    assert sum(b.num_rows for b in bounds) == 100
    assert bounds[0].mins["x"] == 0 and bounds[0].maxs["x"] == 24


def test_range_pruning_counts_survivors(multi_rg):
    bounds = parquet_row_group_bounds(_FakeFS(), [multi_rg], ["x"])
    # x in [30, 60] keeps only the groups [25,49] and [50,74] → 50 rows upper bound.
    assert surviving_rows_for_range(bounds, "x", lower=30, upper=60) == 50
    # Entirely out of range → provably zero (basis for an exact is_empty()).
    assert surviving_rows_for_range(bounds, "x", lower=1000) == 0
    # Unbounded → nothing pruned.
    assert surviving_rows_for_range(bounds, "x") == 100


# --- ORC exact row count -----------------------------------------------------


def test_orc_statistics_exact_row_count(tmp_path):
    orc = pytest.importorskip("pyarrow.orc")
    path = str(tmp_path / "t.orc")
    orc.write_table(pa.table({"x": list(range(37))}), path)
    stats = orc_statistics(_FakeFS(), [path])
    assert stats is not None and stats.row_count == 37 and stats.exact_rows
