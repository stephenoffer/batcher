"""Lakehouse metadata: file skipping on read, metadata-only commits, exact transaction counts.

Three contracts, each of which was broken and is now pinned here:

* **Read.** A predicate the transaction log can decide must eliminate whole data files at
  *plan time* — they are never opened and never become worker tasks. Pruning must also be
  *sound*: it may only ever drop a file the log proves cannot match, so every case that
  aggregates an unknown (a missing statistic, an undecidable predicate) is asserted to
  keep the file. Results are checked against DuckDB, the differential oracle.

* **Write.** A distributed write must move its bytes exactly once. The driver commits
  `AddAction`s, not data — and the statistics it records are what makes the *next* read
  skippable, so a write that loses them silently produces an unskippable table.

* **Transactions.** N micro-batches must leave exactly N transactions in the log, and a
  micro-batch replayed after a crash must add neither a transaction nor a duplicate row.
  A distributed write must be ONE transaction, not one per worker.
"""

from __future__ import annotations

import json
import os

import pyarrow as pa
import pytest

import batcher as bt
from batcher.io.stats.file_skipping import surviving_files

deltalake = pytest.importorskip("deltalake", reason="deltalake not installed")
duckdb = pytest.importorskip("duckdb", reason="duckdb not installed")

pytestmark = pytest.mark.integration


# --- helpers ---------------------------------------------------------------


def _lit(value: int) -> dict:
    return {"e": "lit", "value": {"int": value}}


def _cmp(op: str, column: str, value: int) -> dict:
    return {"e": "binary", "op": op, "left": {"e": "col", "name": column}, "right": _lit(value)}


def _manifest(rows: list[dict]) -> pa.Table:
    """A per-file manifest in the add-action layout `file_skipping` prunes against."""
    return pa.Table.from_pylist(rows)


def _log_transactions(table_uri: str) -> list[str]:
    """The commit files in a Delta table's `_delta_log` — one per transaction."""
    log = os.path.join(table_uri, "_delta_log")
    return sorted(f for f in os.listdir(log) if f.endswith(".json"))


def _clustered_table(files: int, rows_per_file: int) -> str:
    """A Delta table of `files` data files, each holding exactly one distinct ``day``.

    Naturally clustered, so a ``day == k`` predicate can be answered from the log: exactly
    one file can match and the other `files - 1` must never be opened.
    """
    import tempfile

    uri = tempfile.mkdtemp(prefix="delta_clustered_")
    for day in range(files):
        batch = pa.table(
            {
                "day": pa.array([day] * rows_per_file, pa.int64()),
                "id": pa.array(range(rows_per_file), pa.int64()),
            }
        )
        deltalake.write_deltalake(uri, batch, mode="overwrite" if day == 0 else "append")
    return uri


@pytest.fixture(scope="module")
def clustered() -> str:
    return _clustered_table(files=8, rows_per_file=100)


# --- read: file skipping is sound ------------------------------------------


def test_skipping_keeps_a_file_whose_bounds_straddle_the_literal() -> None:
    manifest = _manifest(
        [
            {"path": "a", "num_records": 10, "min.x": 0, "max.x": 5, "null_count.x": 0},
            {"path": "b", "num_records": 10, "min.x": 6, "max.x": 9, "null_count.x": 0},
        ]
    )
    assert surviving_files(_cmp("eq", "x", 3), manifest) == ["a"]
    assert surviving_files(_cmp("eq", "x", 7), manifest) == ["b"]
    assert surviving_files(_cmp("eq", "x", 99), manifest) == []
    assert surviving_files(_cmp("ge", "x", 6), manifest) == ["b"]
    assert surviving_files(_cmp("lt", "x", 6), manifest) == ["a"]


def test_skipping_keeps_a_file_with_no_recorded_statistic() -> None:
    """A missing bound means "unknown", never "no match" — the file must survive.

    This is the soundness invariant. A null statistic that pruned would silently drop
    rows, which is the one failure mode file skipping must never have.
    """
    manifest = _manifest(
        [
            {"path": "a", "num_records": 10, "min.x": 0, "max.x": 5, "null_count.x": 0},
            {
                "path": "unrecorded",
                "num_records": 10,
                "min.x": None,
                "max.x": None,
                "null_count.x": None,
            },
        ]
    )
    assert surviving_files(_cmp("eq", "x", 99), manifest) == ["unrecorded"]


def test_skipping_declines_on_an_undecidable_predicate() -> None:
    """No statistic for the column → no pruning at all (None = "read everything")."""
    manifest = _manifest([{"path": "a", "num_records": 10, "min.x": 0, "max.x": 5}])
    assert surviving_files(_cmp("eq", "other", 1), manifest) is None


def test_skipping_conjunction_prunes_on_the_decidable_half() -> None:
    """``decidable AND unknown`` still prunes — otherwise a compound filter never would."""
    manifest = _manifest(
        [
            {"path": "a", "num_records": 10, "min.x": 0, "max.x": 5},
            {"path": "b", "num_records": 10, "min.x": 6, "max.x": 9},
        ]
    )
    predicate = {
        "e": "binary",
        "op": "and",
        "left": _cmp("eq", "x", 3),
        "right": _cmp("eq", "unknown_column", 1),
    }
    assert surviving_files(predicate, manifest) == ["a"]


def test_skipping_disjunction_declines_when_one_side_is_unknown() -> None:
    """``decidable OR unknown`` cannot prune: the unknown side might match anywhere."""
    manifest = _manifest([{"path": "a", "num_records": 10, "min.x": 0, "max.x": 5}])
    predicate = {
        "e": "binary",
        "op": "or",
        "left": _cmp("eq", "x", 99),
        "right": _cmp("eq", "unknown_column", 1),
    }
    assert surviving_files(predicate, manifest) is None


# --- read: pruning reaches split planning, and the answer still matches DuckDB ---


def test_predicate_prunes_splits_at_plan_time(clustered: str) -> None:
    """The whole point: a selective predicate must cut the *task* count, not just filter rows."""
    from batcher.io.formats.lakehouse import DeltaSource

    source = DeltaSource(clustered)
    assert len(source.splits()) == 8  # unpruned: one split per data file
    assert len(source.splits(predicate=_cmp("eq", "day", 3))) == 1
    assert len(source.splits(predicate=_cmp("ge", "day", 6))) == 2
    assert source.splits(predicate=_cmp("eq", "day", 999)) != []  # empty → still typed


def test_split_carries_its_row_count_from_the_log(clustered: str) -> None:
    """The manifest already knows each file's size; a split must not re-read the footer."""
    from batcher.io.formats.lakehouse import DeltaSource

    split = DeltaSource(clustered).splits(predicate=_cmp("eq", "day", 3))[0]
    assert split.row_count() == 100


@pytest.mark.parametrize("predicate", ["day = 3", "day >= 6", "day = 999", "day < 2"])
def test_pruned_read_matches_duckdb(clustered: str, predicate: str) -> None:
    """Pruning is an I/O optimization; the rows must be exactly DuckDB's."""
    column, op, value = predicate.split()
    ops = {"=": "eq", ">=": "ge", "<": "lt"}
    got = bt.read.delta(clustered).filter(_expr(column, op, int(value))).collect()
    expected = duckdb.sql(
        f"select day, id from delta_scan('{clustered}') where {predicate}"
    ).to_arrow_table()
    assert got.num_rows == expected.num_rows
    assert sorted(got.column("id").to_pylist()) == sorted(expected.column("id").to_pylist())
    assert ops[op]  # the op is one the log can decide


def _expr(column: str, op: str, value: int):
    c = bt.col(column)
    return {"=": c == value, ">=": c >= value, "<": c < value}[op]


# --- write: the commit records the stats the next read prunes against ---------


def test_write_records_file_statistics_in_the_log(tmp_path) -> None:
    """A write that loses its statistics produces a table nothing can skip in."""
    uri = str(tmp_path / "t")
    bt.from_arrow(
        pa.table({"day": pa.array([1, 1, 2], pa.int64()), "v": pa.array([10, 20, 30], pa.int64())})
    ).write.delta(uri)

    entry = json.loads(
        (tmp_path / "t" / "_delta_log" / "00000000000000000000.json").read_text().splitlines()[-1]
    )
    stats = json.loads(entry["add"]["stats"])
    assert stats["numRecords"] == 3
    assert stats["minValues"]["day"] == 1
    assert stats["maxValues"]["day"] == 2
    assert stats["nullCount"]["v"] == 0


def test_second_write_does_not_overwrite_the_first_write_s_files(tmp_path) -> None:
    """Data files are referenced by the log forever, so their names must be unique per write.

    A deterministic ``part-00000.parquet`` would let the second append clobber the file
    version 0 still points at — silently rewriting history and breaking time travel.
    """
    uri = str(tmp_path / "t")
    first = pa.table({"x": pa.array([1, 2, 3], pa.int64())})
    second = pa.table({"x": pa.array([4], pa.int64())})
    bt.from_arrow(first).write.delta(uri)
    bt.from_arrow(second).write.delta(uri)

    assert sorted(bt.read.delta(uri).collect().column("x").to_pylist()) == [1, 2, 3, 4]
    # time travel back to version 0 must still find its own (un-clobbered) data
    assert bt.read.delta(uri, version=0).collect().num_rows == 3


def test_partitioned_write_round_trips_and_prunes(tmp_path) -> None:
    uri = str(tmp_path / "t")
    table = pa.table({"c": pa.array(["a", "a", "b"]), "x": pa.array([1, 2, 3], pa.int64())})
    bt.from_arrow(table).write.delta(uri, partition_by=["c"])

    back = bt.read.delta(uri).collect()
    assert back.num_rows == 3
    assert set(back.column_names) == {"c", "x"}  # the partition column is reconstructed
    expected = duckdb.sql(f"select count(*) from delta_scan('{uri}') where c = 'a'").fetchone()[0]
    assert bt.read.delta(uri).filter(bt.col("c") == "a").count() == expected


def test_overwrite_replaces_the_table(tmp_path) -> None:
    uri = str(tmp_path / "t")
    bt.from_arrow(pa.table({"x": pa.array([1, 2, 3], pa.int64())})).write.delta(uri)
    bt.from_arrow(pa.table({"x": pa.array([9], pa.int64())})).write.delta(uri, mode="overwrite")
    assert sorted(bt.read.delta(uri).collect().column("x").to_pylist()) == [9]


# --- transactions: exactly one per micro-batch, and replay is a no-op ----------


def test_streaming_writes_exactly_one_transaction_per_micro_batch(tmp_path) -> None:
    from batcher.io.formats.streaming.sinks import DeltaStreamSink

    uri = str(tmp_path / "t")
    sink = DeltaStreamSink(uri, query_name="q")
    sink.open()
    for batch_id in range(4):
        sink.write_batch(batch_id, pa.table({"id": pa.array([batch_id], pa.int64())}))
    sink.close()

    assert len(_log_transactions(uri)) == 4
    assert sorted(bt.read.delta(uri).collect().column("id").to_pylist()) == [0, 1, 2, 3]


def test_replayed_micro_batch_adds_no_transaction_and_no_duplicate_row(tmp_path) -> None:
    """The exactly-once contract.

    The engine write-ahead-logs a micro-batch's source offset *before* processing it, so a
    crash between processing and committing leaves a batch the next run replays. Without a
    Delta ``txn`` action the replay appends the rows a second time — which is precisely the
    bug this pins: the rows must not double, and the log must not grow.
    """
    from batcher.io.formats.streaming.sinks import DeltaStreamSink

    uri = str(tmp_path / "t")
    sink = DeltaStreamSink(uri, query_name="q")
    sink.open()
    for batch_id in range(3):
        sink.write_batch(batch_id, pa.table({"id": pa.array([batch_id], pa.int64())}))

    # crash + restart: the checkpoint replays micro-batch 2
    sink.write_batch(2, pa.table({"id": pa.array([2], pa.int64())}))

    assert len(_log_transactions(uri)) == 3, "a replayed batch must not commit again"
    assert sorted(bt.read.delta(uri).collect().column("id").to_pylist()) == [0, 1, 2]


def test_many_worker_shards_commit_as_one_transaction(tmp_path) -> None:
    """The distributed-write contract, exercised at the sink boundary (no Ray needed).

    This is what the distributed path does: every worker writes its own shard's data
    files and returns only their locators; the driver merges those into one manifest and
    commits *once*. The tempting wrong design — each worker committing its own files —
    would leave one transaction per worker in the log, make the write non-atomic (a
    reader could see half of it), and grow the log with the cluster. So the assertion is
    on the transaction count, not just the rows.
    """
    from batcher.io.formats.base import SINKS
    from batcher.io.manifest import WriteManifest

    uri = str(tmp_path / "t")
    workers = 8
    schema = pa.schema([pa.field("day", pa.int64()), pa.field("id", pa.int64())])
    sink = SINKS.get("delta")()

    written = []
    for worker in range(workers):  # each of these runs on a different machine
        shard = pa.table(
            {
                "day": pa.array([worker] * 10, pa.int64()),
                "id": pa.array(range(worker * 10, worker * 10 + 10), pa.int64()),
            },
            schema=schema,
        )
        written.extend(sink.write_partitioned(shard, uri, file_index=worker))

    assert len(written) == workers  # every worker produced its own data file
    sink.commit(WriteManifest(tuple(written), schema=schema), uri)  # ...one commit

    assert len(_log_transactions(uri)) == 1, "one write must be one transaction"
    assert bt.read.delta(uri).count() == workers * 10
    # and every worker's statistics landed, so the table it wrote is still skippable
    from batcher.io.formats.lakehouse import DeltaSource

    source = DeltaSource(uri)
    assert len(source.splits()) == workers
    assert len(source.splits(predicate=_cmp("eq", "day", 3))) == 1


def test_a_restarted_query_resumes_without_duplicating(tmp_path) -> None:
    """A brand-new sink object (a process restart) must still see the prior transactions."""
    from batcher.io.formats.streaming.sinks import DeltaStreamSink

    uri = str(tmp_path / "t")
    first = DeltaStreamSink(uri, query_name="q")
    first.open()
    for batch_id in range(3):
        first.write_batch(batch_id, pa.table({"id": pa.array([batch_id], pa.int64())}))

    restarted = DeltaStreamSink(uri, query_name="q")  # fresh process, same query name
    restarted.open()
    for batch_id in (1, 2):  # replays the last committed batches
        restarted.write_batch(batch_id, pa.table({"id": pa.array([batch_id], pa.int64())}))
    restarted.write_batch(3, pa.table({"id": pa.array([3], pa.int64())}))

    assert len(_log_transactions(uri)) == 4
    assert sorted(bt.read.delta(uri).collect().column("id").to_pylist()) == [0, 1, 2, 3]


# --- the snapshot cache must never serve an old version ---------------------


def test_count_and_collect_agree_after_a_write(tmp_path) -> None:
    """They did not. `count()` kept answering 3 while `collect()` returned 5.

    Some terminals are answered from the session's cached `SourceStatistics` without
    executing, and that cache is keyed by the source's `identity()`. A Delta source's
    identity said ``delta:/t@latest`` — the *same string* for every version — so a cached row
    count outlived the table it described. Two spellings of the same query, two different
    answers, and the metadata path is the one that lies.

    The identity now names the resolved version, so a new version is simply a new key.
    """
    uri = str(tmp_path / "t")
    bt.from_pydict({"x": [1, 2, 3]}).write.delta(uri, mode="overwrite")
    assert bt.read.delta(uri).count() == bt.read.delta(uri).collect().num_rows == 3

    bt.from_pydict({"x": [4]}).write.delta(uri, mode="append")
    assert bt.read.delta(uri).count() == bt.read.delta(uri).collect().num_rows == 4


def test_a_table_written_by_someone_else_is_not_served_stale(tmp_path) -> None:
    """The case invalidation could never cover.

    Dropping the cache on *our* commits only ever helped writes Batcher made. A table
    appended to by Spark, by a streaming job, or by another process went stale with nothing
    to notice. Keying on the version means there is no stale entry to serve, whoever wrote.
    """
    uri = str(tmp_path / "t")
    bt.from_pydict({"x": [1, 2, 3]}).write.delta(uri, mode="overwrite")
    assert bt.read.delta(uri).count() == 3

    # a writer that is not us
    deltalake.write_deltalake(uri, pa.table({"x": pa.array([4, 5], pa.int64())}), mode="append")

    assert bt.read.delta(uri).count() == 5
    assert bt.read.delta(uri).collect().num_rows == 5


def test_time_travel_still_pins_its_own_version(tmp_path) -> None:
    """The shared handle rolls forward; a pinned read must not roll with it."""
    uri = str(tmp_path / "t")
    bt.from_pydict({"x": [1, 2, 3]}).write.delta(uri, mode="overwrite")
    bt.from_pydict({"x": [4]}).write.delta(uri, mode="append")

    assert bt.read.delta(uri, version=0).count() == 3
    assert bt.read.delta(uri, version=1).count() == 4
    assert bt.read.delta(uri).count() == 4  # and latest is still latest


def test_the_identity_names_the_version_it_reads(tmp_path) -> None:
    from batcher.io.formats.lakehouse import DeltaSource

    uri = str(tmp_path / "t")
    bt.from_pydict({"x": [1]}).write.delta(uri, mode="overwrite")
    first = DeltaSource(uri).identity()

    bt.from_pydict({"x": [2]}).write.delta(uri, mode="append")
    second = DeltaSource(uri).identity()

    assert first != second, "a new version must be a new cache key"
    assert DeltaSource(uri, version=0).identity() == first
