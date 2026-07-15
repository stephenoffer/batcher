"""Regression tests for IO defects found in the bug hunt.

Each test fails on the pre-fix behavior:

- Hive NULL partition key silently dropped every row of the NULL group (data loss).
- `splits(predicate=...)` raised `TypeError` for CSV/JSON/ORC/Arrow-IPC sources
  (the base's `_file_splits` contract grew a `predicate` arg; the overrides didn't).
- A schema-evolving (`union`/`latest`) Parquet read with a pushed predicate returned
  differently-typed batches that could not concatenate (crash / corruption).
"""

from __future__ import annotations

import glob
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batcher.io.formats.semistructured.json import JSONSource
from batcher.io.formats.structured.arrow_ipc import ArrowIPCSource
from batcher.io.formats.structured.csv import CSVSource
from batcher.io.formats.structured.parquet.sink import ParquetSink
from batcher.io.formats.structured.parquet.source import ParquetSource
from batcher.io.source import plan_splits

pytestmark = pytest.mark.unit


def _cmp(col: str, op: str, value: int) -> dict:
    return {
        "e": "binary",
        "op": op,
        "left": {"e": "col", "name": col},
        "right": {"e": "lit", "value": {"int": value}},
    }


_GT_ZERO = _cmp("x", "gt", 0)


def _read_part(path: str) -> pa.Table:
    """Read one part file directly, bypassing pyarrow's Hive-partition inference."""
    with open(path, "rb") as fh:
        return pq.read_table(fh)


def test_hive_null_partition_keeps_its_rows(tmp_path):
    table = pa.table({"c": ["a", None, "b", None], "x": [1, 2, 3, 4]})
    written = ParquetSink().write_partitioned(table, str(tmp_path), partition_by=["c"])

    # Every row must survive: the NULL group used to write an empty file.
    assert sum(w.rows for w in written) == table.num_rows

    all_x: list[int] = []
    for f in glob.glob(os.path.join(str(tmp_path), "**", "*.parquet"), recursive=True):
        all_x.extend(_read_part(f).column("x").to_pylist())
    assert sorted(all_x) == [1, 2, 3, 4]

    # The NULL rows land under __HIVE_DEFAULT_PARTITION__ and are readable.
    null_dir = os.path.join(str(tmp_path), "c=__HIVE_DEFAULT_PARTITION__")
    null_rows = [
        x for f in glob.glob(null_dir + "/*.parquet") for x in _read_part(f).column("x").to_pylist()
    ]
    assert sorted(null_rows) == [2, 4]


@pytest.mark.parametrize("suffix,writer", [(".csv", "csv"), (".json", "json"), (".arrow", "arrow")])
def test_splits_accepts_pushed_predicate(tmp_path, suffix, writer):
    path = str(tmp_path / f"data{suffix}")
    table = pa.table({"x": [1, 2, 3], "y": [4, 5, 6]})
    if writer == "csv":
        import pyarrow.csv as pacsv

        pacsv.write_csv(table, path)
        src = CSVSource(path)
    elif writer == "json":
        import pandas as pd

        pd.DataFrame(table.to_pydict()).to_json(path, orient="records", lines=True)
        src = JSONSource(path)
    else:
        import pyarrow.feather as feather

        feather.write_feather(table, path)
        src = ArrowIPCSource(path)

    # Must not raise TypeError; a predicate that cannot prune keeps every split.
    assert src.splits(target_size=None, predicate=_GT_ZERO)
    assert plan_splits(src, predicate=_GT_ZERO)


def test_hive_partition_values_round_trip_through_url_encoding(tmp_path):
    from batcher.io.formats.structured.parquet.dataset import ParquetDatasetSource

    # Values with '/', '=', space, '%', ':' must survive the write→read round trip.
    values = ["x/y", "a=b", "hello world", "p%q", "c:d", "normal"]
    table = pa.table({"c": values, "v": list(range(len(values)))})
    ParquetSink().write_partitioned(table, str(tmp_path), partition_by=["c"])

    back = pa.Table.from_batches(ParquetDatasetSource(str(tmp_path)).read())
    got = sorted(zip(back.column("c").to_pylist(), back.column("v").to_pylist(), strict=True))
    assert got == sorted(zip(values, range(len(values)), strict=True))


def test_schema_evolving_parquet_read_with_predicate_concatenates(tmp_path):
    # f1: a:int32, no c. f2: a:int64, c present. Widened + added column.
    pq.write_table(pa.table({"a": pa.array([1, 2, 3], pa.int32())}), str(tmp_path / "f1.parquet"))
    pq.write_table(
        pa.table({"a": pa.array([4, 5], pa.int64()), "c": [10, 11]}), str(tmp_path / "f2.parquet")
    )
    src = ParquetSource(str(tmp_path), schema_mode="union")
    pred = _cmp("a", "ge", 2)

    batches = src.read(None, pred)
    # All batches share the unified schema, so concatenation succeeds.
    assert len({b.schema for b in batches}) == 1
    table = pa.Table.from_batches(batches)  # would raise pre-fix
    assert table.column("a").to_pylist() == [1, 2, 3, 4, 5]
    assert table.column("c").to_pylist() == [None, None, None, 10, 11]

    # Streaming path likewise reconciles.
    stream = pa.Table.from_batches(list(src.iter_batches(None, pred)))
    assert stream.num_rows == 5
