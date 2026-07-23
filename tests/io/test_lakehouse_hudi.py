"""Hudi reads in parallel, prunes partitions, and knows its own row count.

Three things were wrong, and none of them announced itself:

* **`splits()` returned one `WholeSourceSplit`**, so a Hudi table was read entirely
  through the driver and re-scattered. No distributed read, no file skipping, and the
  driver's memory as the ceiling on table size — for a format whose timeline lists its
  file slices outright.
* **Predicate pushdown never ran.** hudi-rs takes a filter's value as *text* and parses it
  against the column's type; handing it a Python `int` raises, and the connector caught
  that and quietly re-read unfiltered. Every filter silently degraded, and the fallback is
  what hid it.
* **`row_count()` returned `None`**, so the estimator guessed at a table whose size the
  timeline states exactly.

The tables here are built by hand rather than by Spark: hudi-rs is a *reader* (its
`HudiTableBuilder` builds a reader config, not a writer), so a Hudi write needs the
Spark/Flink stack. Constructing the timeline and base files directly is what lets this run
anywhere — and it is why the connector can be tested at all.
"""

from __future__ import annotations

import json
import uuid

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt

pytest.importorskip("hudi", reason="hudi-rs not installed")

from batcher.io.formats.lakehouse.hudi import HudiSource, _hudi_filters

pytestmark = pytest.mark.integration

INSTANT = "20240101000000000"

_PROPERTIES = """hoodie.table.name={name}
hoodie.table.type=COPY_ON_WRITE
hoodie.table.version=6
hoodie.timeline.layout.version=1
hoodie.table.base.file.format=PARQUET
hoodie.table.recordkey.fields=id
hoodie.table.partition.fields={partition_fields}
hoodie.table.keygenerator.class=org.apache.hudi.keygen.{keygen}
hoodie.datasource.write.hive_style_partitioning=true
hoodie.datasource.write.partitionpath.urlencode=false
hoodie.datasource.write.drop.partition.columns=false
hoodie.populate.meta.fields=true
hoodie.table.checksum=0
"""


def _write_slice(root, partition: str, days: list[int], ids: list[int]) -> dict:
    """One Hudi base file (a file slice) plus the write-stat the timeline records for it."""
    directory = root / partition if partition else root
    directory.mkdir(parents=True, exist_ok=True)
    file_id = str(uuid.uuid4())
    name = f"{file_id}_0-1-0_{INSTANT}.parquet"
    rows = len(ids)
    table = pa.table(
        {
            "_hoodie_commit_time": pa.array([INSTANT] * rows),
            "_hoodie_commit_seqno": pa.array([f"{INSTANT}_{i}" for i in range(rows)]),
            "_hoodie_record_key": pa.array([str(i) for i in ids]),
            "_hoodie_partition_path": pa.array([partition] * rows),
            "_hoodie_file_name": pa.array([name] * rows),
            "id": pa.array(ids, pa.int64()),
            "day": pa.array(days, pa.int64()),
        }
    )
    pq.write_table(table, directory / name)
    return {
        "fileId": file_id,
        "path": f"{partition}/{name}" if partition else name,
        "numWrites": rows,
        "numInserts": rows,
        "numDeletes": 0,
        "numUpdateWrites": 0,
        "totalWriteBytes": 1000,
        "partitionPath": partition,
        "prevCommit": "null",
        "fileSizeInBytes": 1000,
    }


def _build_table(root, *, partitioned: bool) -> str:
    """A copy-on-write Hudi table of 3 file slices, one distinct ``day`` each."""
    (root / ".hoodie").mkdir(parents=True)
    (root / ".hoodie" / "hoodie.properties").write_text(
        _PROPERTIES.format(
            name=root.name,
            partition_fields="day" if partitioned else "",
            keygen="SimpleKeyGenerator" if partitioned else "NonpartitionedKeyGenerator",
        )
    )
    write_stats: dict[str, list[dict]] = {}
    for day in range(3):
        partition = f"day={day}" if partitioned else ""
        ids = [day * 10 + i for i in range(5)]
        stat = _write_slice(root, partition, [day] * 5, ids)
        write_stats.setdefault(partition, []).append(stat)

    commit = {
        "partitionToWriteStats": write_stats,
        "compacted": False,
        "extraMetadata": {},
        "operationType": "INSERT",
        "totalRecordsDeleted": 0,
        "totalLogRecordsCompacted": 0,
    }
    for suffix in (".commit.requested", ".inflight", ".commit"):
        payload = json.dumps(commit) if suffix == ".commit" else ""
        (root / ".hoodie" / f"{INSTANT}{suffix}").write_text(payload)
    return str(root)


@pytest.fixture
def partitioned(tmp_path) -> str:
    return _build_table(tmp_path / "hudi_p", partitioned=True)


@pytest.fixture
def flat(tmp_path) -> str:
    return _build_table(tmp_path / "hudi_f", partitioned=False)


def _predicate(day: int) -> dict:
    return {
        "e": "binary",
        "op": "eq",
        "left": {"e": "col", "name": "day"},
        "right": {"e": "lit", "value": {"int": day}},
    }


# --- the filter-type bug ---------------------------------------------------


def test_filter_values_are_stringified() -> None:
    """hudi-rs parses the value from text; an int raises and the caller swallowed it.

    That swallow is what made the bug invisible — every filter fell back to an unfiltered
    read and the query still returned the right rows, just after reading the whole table.
    """
    assert _hudi_filters(_predicate(1)) == [("day", "=", "1")]


# --- splits ----------------------------------------------------------------


def test_each_file_slice_becomes_its_own_split(flat: str) -> None:
    """It used to be one WholeSourceSplit — the entire table through the driver."""
    splits = HudiSource(flat).splits()

    assert len(splits) == 3
    assert [s.row_count() for s in splits] == [5, 5, 5]


def test_a_split_reads_its_slice_independently(flat: str) -> None:
    """The property a distributed read needs: a worker reads one file, not the table."""
    splits = HudiSource(flat).splits()

    rows = [sum(b.num_rows for b in s.read()) for s in splits]
    assert rows == [5, 5, 5]
    assert sum(rows) == 15


def test_a_predicate_prunes_partitions_at_plan_time(partitioned: str) -> None:
    source = HudiSource(partitioned)

    assert len(source.splits()) == 3
    assert len(source.splits(predicate=_predicate(1))) == 1


def test_pruning_needs_a_partition_column(flat: str) -> None:
    """An unpartitioned table has nothing to prune; the engine's Filter does the work."""
    assert len(HudiSource(flat).splits(predicate=_predicate(1))) == 3


# --- statistics ------------------------------------------------------------


def test_row_count_comes_from_the_timeline(flat: str) -> None:
    """The timeline records each slice's record count — it is metadata, not a scan."""
    source = HudiSource(flat)

    assert source.row_count() == 15
    stats = source.statistics()
    assert stats is not None
    assert stats.row_count == 15
    assert stats.exact_rows is True


# --- end to end ------------------------------------------------------------


def test_the_engine_reads_and_filters_a_hudi_table(partitioned: str) -> None:
    ds = bt.read.table("hudi", partitioned)

    assert ds.collect().num_rows == 15
    filtered = ds.filter(bt.col("day") == 1).collect()
    assert filtered.num_rows == 5
    assert set(filtered.column("day").to_pylist()) == {1}


def test_a_pruned_read_returns_the_same_rows_as_an_unpruned_one(partitioned: str) -> None:
    """Pruning is an I/O optimization; it must not change a single row."""
    source = HudiSource(partitioned)

    pruned = source.splits(predicate=_predicate(2))
    from_splits = sorted(
        i
        for s in pruned
        for b in s.read(predicate=_predicate(2))
        for i in b.column("id").to_pylist()
    )
    whole = bt.read.table("hudi", partitioned).filter(bt.col("day") == 2).collect()
    assert from_splits == sorted(whole.column("id").to_pylist())


def test_hudi_writes_are_refused_with_a_reason() -> None:
    """hudi-rs is a reader; a write needs Spark/Flink. Saying so beats a confusing failure."""
    from batcher._internal.errors import BackendError
    from batcher.io.formats.lakehouse import HudiSink

    with pytest.raises(BackendError, match="Spark/Flink"):
        HudiSink()
