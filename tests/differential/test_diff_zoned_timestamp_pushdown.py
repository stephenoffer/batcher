"""A filter on a timezone-aware timestamp column pushes into Parquet and ORC reads.

The predicate IR carries a timestamp literal as UTC micros with no zone, and pyarrow refuses to
compare a zoned column against a naive scalar. `to_pyarrow_expression` already knows how to type
the literal to its column, but only when it is handed the schema, and neither file source
passed one. The two failed differently, and both are pinned here:

* **Parquet** caught the error and fell back to an unfiltered read, so the query was right and
  read every row group the predicate should have pruned.
* **ORC** had no fallback, so the query *raised*
  ``Function 'greater_equal' has no kernel matching input types (timestamp[ns, tz=UTC], ...)``.
"""

from __future__ import annotations

import datetime

import pyarrow as pa
import pyarrow.orc as po
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_same
from batcher.io.formats.structured.parquet.source import ParquetSource

pytestmark = pytest.mark.differential

_ROWS = 20_000
_BOUND = datetime.datetime(1970, 1, 1, 5, 0, tzinfo=datetime.UTC)


def _table():
    micros = [i * 1_000_000 for i in range(_ROWS)]
    return pa.table(
        {"ts": pa.array(micros, pa.timestamp("us", tz="UTC")), "v": pa.array(range(_ROWS))}
    )


def test_parquet_pushes_and_matches_duckdb(duck, tmp_path, monkeypatch):
    path = str(tmp_path / "t.parquet")
    pq.write_table(_table(), path, row_group_size=1_000)
    read = {"rows": 0}
    original = ParquetSource.read

    def _read(self, projection=None, predicate=None):
        out = original(self, projection, predicate)
        read["rows"] += sum(b.num_rows for b in out)
        return out

    monkeypatch.setattr(ParquetSource, "read", _read)
    got = bt.read.parquet(path).filter(bt.col("ts") >= _BOUND).select("v").collect()
    sql = f"SELECT v FROM read_parquet('{path}') WHERE ts >= TIMESTAMPTZ '1970-01-01 05:00:00+00'"
    assert_same(got, duck.sql(sql))
    assert read["rows"] < _ROWS, "the zoned predicate did not reach the reader"


def test_orc_filter_answers_instead_of_raising(tmp_path):
    path = str(tmp_path / "t.orc")
    table = _table()
    po.write_table(table, path)
    got = bt.read.orc(path).filter(bt.col("ts") >= _BOUND).collect()
    assert got.num_rows == _ROWS - 5 * 3600
    assert min(got.column("v").to_pylist()) == 5 * 3600
