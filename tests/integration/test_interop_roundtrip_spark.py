"""Spark <-> Batcher, both directions, holding column types as well as values.

Most of this suite needs no JVM, and that is deliberate: there is none on the machines this
repository is developed on, and a suite that skipped whenever Java was absent would never run.
`Dataset.to_spark` only calls ``spark.createDataFrame`` or ``spark.read.parquet``, and
`bt.from_spark` only reads ``__arrow_c_stream__``, ``toArrow`` or ``toPandas``, so a fake
session and fake frames exercise every branch of both: which path a result takes, the exact
Arrow table or Parquet file Spark would receive, and the batches Batcher builds from what
Spark would send. What the fakes cannot show is Spark's own reading of those inputs, and
`test_a_real_spark_session_round_trips_the_fixture` covers that where a JVM exists.

The shared fixture and pinned schemas are in `tests/_interop_cases.py`.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _interop_cases import ENGINE_SCHEMA, ROWS, TABLE, assert_types, dataset, empty
from batcher.io import interop
from batcher.io.source import InMemorySource, IteratorSource

pytest.importorskip("pyspark")

pytestmark = pytest.mark.integration

_HAS_JVM = bool(os.environ.get("JAVA_HOME") or shutil.which("java"))


class _FakeReader:
    def __init__(self) -> None:
        self.paths: list[str] = []

    def parquet(self, path: str) -> tuple[str, str]:
        self.paths.append(path)
        return ("parquet", path)


class _FakeSession:
    """Records what `to_spark` hands a `SparkSession`, and returns a tag naming the path."""

    def __init__(self) -> None:
        self.read = _FakeReader()
        self.created: list[pa.Table] = []

    def createDataFrame(self, data: pa.Table) -> tuple[str, pa.Table]:
        self.created.append(data)
        return ("arrow", data)


class _StreamingFrame:
    """A PySpark >= 4.1 frame: exports an Arrow stream, counting each time it is opened."""

    def __init__(self, table: pa.Table, chunk: int = 1) -> None:
        self._table = table
        self._chunk = chunk
        self.opened = 0

    def __arrow_c_stream__(self, requested_schema=None):
        self.opened += 1
        batches = self._table.to_batches(max_chunksize=self._chunk)
        reader = pa.RecordBatchReader.from_batches(self._table.schema, batches)
        return reader.__arrow_c_stream__(requested_schema)


class _EagerFrame:
    """A PySpark 4.0 frame: ``toArrow`` only."""

    def __init__(self, table: pa.Table) -> None:
        self._table = table

    def toArrow(self) -> pa.Table:
        return self._table


def test_a_small_result_goes_to_create_dataframe_as_one_arrow_table():
    spark = _FakeSession()
    kind, table = dataset().to_spark(spark)
    assert kind == "arrow"
    assert spark.read.paths == []
    assert isinstance(table, pa.Table)
    assert_types(table.schema, ENGINE_SCHEMA)
    assert table.to_pylist() == ROWS


def test_a_large_result_is_staged_as_parquet_under_the_staging_path(tmp_path: Path):
    spark = _FakeSession()
    kind, path = dataset().to_spark(spark, max_arrow_bytes=0, staging_path=str(tmp_path))
    assert kind == "parquet"
    assert spark.created == []
    assert Path(path).parent == tmp_path
    staged = pq.read_table(path)
    assert_types(staged.schema, ENGINE_SCHEMA)
    assert staged.to_pylist() == ROWS
    # A second hand-off must not land beside the first, or Spark would read both.
    _, second = dataset().to_spark(spark, max_arrow_bytes=0, staging_path=str(tmp_path))
    assert second != path
    assert len(list(Path(second).iterdir())) == 1


def test_the_threshold_is_what_decides_the_path():
    # Positive control for the two tests above: the same result, with the limit on either
    # side of its size, takes each path.
    spark = _FakeSession()
    assert dataset().to_spark(spark, max_arrow_bytes=1 << 30)[0] == "arrow"
    assert dataset().to_spark(spark, max_arrow_bytes=1)[0] == "parquet"


def test_an_empty_result_reaches_spark_with_its_schema():
    spark = _FakeSession()
    kind, table = empty().to_spark(spark, max_arrow_bytes=0)
    assert kind == "arrow", "an empty result has nothing to stage"
    assert table.num_rows == 0
    assert_types(table.schema, ENGINE_SCHEMA)


def test_staging_refuses_to_truncate_a_nanosecond_timestamp(tmp_path: Path):
    ns = pa.table({"t": pa.array([1_000_000_001], pa.timestamp("ns", tz="UTC"))})
    ds = bt.from_arrow(ns)
    assert ds.schema.field("t").type == pa.timestamp("ns", tz="UTC"), "fixture keeps nanoseconds"
    with pytest.raises(pa.ArrowInvalid, match=r"(?i)truncat|lose data"):
        ds.to_spark(_FakeSession(), max_arrow_bytes=0, staging_path=str(tmp_path))


def test_from_spark_streams_the_arrow_export_and_keeps_types():
    frame = _StreamingFrame(TABLE)
    source = interop.from_spark(frame)
    assert isinstance(source, IteratorSource)
    ds = bt.from_spark(frame)
    opened_to_plan = frame.opened
    out = ds.to_arrow()
    assert frame.opened == opened_to_plan + 1, "one execution should open one stream"
    assert_types(out.schema, ENGINE_SCHEMA)
    assert out.to_pylist() == ROWS


def test_from_spark_of_an_empty_stream_keeps_its_schema():
    out = bt.from_spark(_StreamingFrame(TABLE.slice(0, 0))).to_arrow()
    assert out.num_rows == 0
    assert_types(out.schema, ENGINE_SCHEMA)


def test_from_spark_collects_when_there_is_no_stream():
    frame = _EagerFrame(TABLE)
    assert isinstance(interop.from_spark(frame), InMemorySource)
    out = bt.from_spark(frame).to_arrow()
    assert_types(out.schema, ENGINE_SCHEMA)
    assert out.to_pylist() == ROWS


@pytest.mark.skipif(not _HAS_JVM, reason="PySpark needs a JVM, and this environment has none")
def test_a_real_spark_session_round_trips_the_fixture(tmp_path: Path):
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[1]")
        .appName("batcher-interop-roundtrip")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    try:
        ids = [row["id"] for row in dataset().to_spark(spark).orderBy("id").collect()]
        assert ids == [0, 1, 2]
        staged = dataset().to_spark(spark, max_arrow_bytes=0, staging_path=str(tmp_path))
        back = bt.from_spark(staged).sort("id")
        got = back.to_pylist()
        assert [row["id"] for row in got] == [0, 1, 2]
        assert [row["s"] for row in got] == [row["s"] for row in ROWS]
        assert [row["d"] for row in got] == [row["d"] for row in ROWS]
    finally:
        spark.stop()
