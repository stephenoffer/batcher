"""A predicated Parquet read that prunes row groups and decodes the rest whole matches DuckDB.

`ParquetSource.read` now skips the row groups whose footer bounds rule a predicate out and hands
the survivors to the engine undecoded-for-rows, leaving the `Filter` above the scan to remove the
non-matching rows (`io/formats/structured/parquet/routing.py`). That moves the whole of row
selection out of the reader, so every predicate shape the reader used to filter exactly is checked
here against DuckDB: dates, zoned timestamps, integers, strings, `IN`, `NOT`, and floats holding
NaN, which the footer bounds leave out.

Two controls keep the test able to fail. On a table clustered on its key the reader must return
far fewer rows than the table holds, or pruning stopped happening. And with the memory envelope
shrunk until the survivors cannot fit, the read must fall back to the filtered path and still
return the same answer.
"""

from __future__ import annotations

import datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_same
from batcher.io.formats.structured.parquet import routing
from batcher.io.formats.structured.parquet.source import ParquetSource

pytestmark = pytest.mark.differential

_ROWS = 40_000
_EPOCH = datetime.date(1970, 1, 1)


@pytest.fixture(scope="module")
def path(tmp_path_factory):
    rng = np.random.default_rng(9)
    ids = np.arange(_ROWS, dtype="int64")
    floats = rng.random(_ROWS)
    floats[rng.choice(_ROWS, 400, replace=False)] = np.nan
    table = pa.table(
        {
            "id": ids,  # clustered: row groups prune
            "day": pa.array((ids // 40 + 9000).astype("int32"), pa.int32()).cast(pa.date32()),
            "ts": pa.array(ids * 1_000_000, pa.timestamp("us", tz="UTC")),
            "bucket": pa.array(rng.integers(0, 50, _ROWS)),  # scattered: nothing prunes
            "name": pa.array([f"k{i % 211:03d}" for i in range(_ROWS)]),
            "f": pa.array(floats),
        }
    )
    out = tmp_path_factory.mktemp("pruned_whole") / "t.parquet"
    pq.write_table(table, out, row_group_size=2_000)
    return str(out)


def _day(n: int) -> datetime.date:
    return _EPOCH + datetime.timedelta(days=n)


_PREDICATES = [
    ("id >= 38000", lambda: bt.col("id") >= 38_000),
    ("id < 100 OR id > 39900", lambda: (bt.col("id") < 100) | (bt.col("id") > 39_900)),
    (
        f"day BETWEEN DATE '{_day(9100)}' AND DATE '{_day(9110)}'",
        lambda: (bt.col("day") >= _day(9100)) & (bt.col("day") <= _day(9110)),
    ),
    (
        "ts >= TIMESTAMPTZ '1970-01-01 10:00:00+00'",
        lambda: bt.col("ts") >= datetime.datetime(1970, 1, 1, 10, tzinfo=datetime.UTC),
    ),
    ("bucket = 7", lambda: bt.col("bucket") == 7),
    ("bucket IN (1, 2, 3)", lambda: bt.col("bucket").is_in([1, 2, 3])),
    ("name >= 'k200'", lambda: bt.col("name") >= "k200"),
    ("NOT (id < 39000)", lambda: ~(bt.col("id") < 39_000)),
    ("f > 0.95", lambda: bt.col("f") > 0.95),
    ("f <= 0.01", lambda: bt.col("f") <= 0.01),
]
_IDS = [sql for sql, _ in _PREDICATES]


@pytest.mark.parametrize("sql_pred, expr", _PREDICATES, ids=_IDS)
def test_rows_match_duckdb(duck, path, sql_pred, expr):
    got = bt.read.parquet(path).filter(expr()).select("id", "name").collect()
    sql = f"SELECT id, name FROM read_parquet('{path}', can_have_nan=true) WHERE {sql_pred}"
    assert_same(got, duck.sql(sql))


@pytest.mark.parametrize("sql_pred, expr", _PREDICATES, ids=_IDS)
def test_the_filtered_fallback_answers_the_same(duck, path, monkeypatch, sql_pred, expr):
    """With no memory to decode survivors whole, the old filtered read must give the same rows."""
    monkeypatch.setattr(routing, "MEMORY_FRACTION", 0.0)
    pruned = []
    original = ParquetSource._native_read_pruned

    def _spy(self, projection, predicate):
        out = original(self, projection, predicate)
        pruned.append(out)
        return out

    monkeypatch.setattr(ParquetSource, "_native_read_pruned", _spy)
    got = bt.read.parquet(path).filter(expr()).select("id", "name").collect()
    sql = f"SELECT id, name FROM read_parquet('{path}', can_have_nan=true) WHERE {sql_pred}"
    assert_same(got, duck.sql(sql))
    assert pruned and all(out is None for out in pruned), "the guard did not decline"


def test_a_clustered_predicate_reads_only_its_row_groups(path, monkeypatch):
    read = {"rows": 0}
    original = ParquetSource.read

    def _read(self, projection=None, predicate=None):
        out = original(self, projection, predicate)
        read["rows"] += sum(b.num_rows for b in out)
        return out

    monkeypatch.setattr(ParquetSource, "read", _read)
    got = bt.read.parquet(path).filter(bt.col("id") >= 38_000).select("id").collect()
    assert got.num_rows == 2_000
    assert read["rows"] == 2_000, "pruning should leave exactly the one matching row group"
