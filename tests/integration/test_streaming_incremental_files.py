"""End-to-end: a directory-watching stream keeps ingesting for the life of the query.

`files_incremental` is Batcher's Auto Loader analog, and the shape a file-drop pipeline is
written in: point a stream at a directory, keep writing files into it. It used to run its
first discovery pass and then stop — `is_active` went False within a trigger or two, every
later file was ignored, and nothing raised or logged. These tests pin the whole path
(source, runner, engine, sink) rather than the runner alone, because the runner is where
the defect was and the query handle is where it was visible.
"""

from __future__ import annotations

import time

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher.config import Config, StreamingConfig, config_context


def _write(directory, name: str, values: list[int]) -> None:
    pq.write_table(pa.table({"a": pa.array(values, type=pa.int64())}), str(directory / name))


def _wait_until(predicate, timeout: float = 10.0) -> bool:
    """Poll `predicate` up to `timeout` seconds; True as soon as it holds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


@pytest.fixture
def watched(tmp_path):
    """A directory to drop files into, plus the source's state directory."""
    data = tmp_path / "landing"
    data.mkdir()
    return data, str(tmp_path / "state")


@pytest.mark.integration
def test_files_arriving_after_the_first_pass_are_ingested(watched):
    data, state = watched
    _write(data, "f1.parquet", [1, 2, 3])
    ds = bt.read.files_incremental(str(data), "parquet", state_dir=state)

    with config_context(Config().replace(streaming=StreamingConfig(idle_poll_seconds=0.02))):
        q = ds.write.memory("inc_watch", trigger=bt.Trigger.processing_time("0.05 seconds"))
        try:
            assert _wait_until(lambda: bt.read_memory("inc_watch").count() == 3)
            assert q.is_active, "the query stopped after its first discovery pass"

            _write(data, "f2.parquet", [4, 5])
            assert _wait_until(lambda: bt.read_memory("inc_watch").count() == 5), (
                "a file that arrived after the first pass was never ingested"
            )
            assert q.is_active
        finally:
            q.stop()

    assert sorted(bt.read_memory("inc_watch").to_pydict()["a"]) == [1, 2, 3, 4, 5]


@pytest.mark.integration
def test_each_file_is_ingested_exactly_once_across_passes(watched):
    data, state = watched
    _write(data, "f1.parquet", [1])

    with config_context(Config().replace(streaming=StreamingConfig(idle_poll_seconds=0.02))):
        ds = bt.read.files_incremental(str(data), "parquet", state_dir=state)
        q = ds.write.memory("inc_once", trigger=bt.Trigger.processing_time("0.05 seconds"))
        try:
            assert _wait_until(lambda: bt.read_memory("inc_once").count() == 1)
            # Several idle passes over an unchanged directory must add nothing.
            time.sleep(0.3)
            assert bt.read_memory("inc_once").count() == 1
            _write(data, "f2.parquet", [2])
            assert _wait_until(lambda: bt.read_memory("inc_once").count() == 2)
            time.sleep(0.3)
            assert bt.read_memory("inc_once").count() == 2
        finally:
            q.stop()


@pytest.mark.integration
def test_available_now_drains_the_backlog_and_stops(watched):
    data, state = watched
    _write(data, "f1.parquet", [1, 2])
    _write(data, "f2.parquet", [3])
    ds = bt.read.files_incremental(str(data), "parquet", state_dir=state)

    q = ds.write.memory("inc_drain", trigger=bt.Trigger.available_now())
    assert q.await_termination(timeout=30) is True
    assert q.is_active is False
    assert sorted(bt.read_memory("inc_drain").to_pydict()["a"]) == [1, 2, 3]


@pytest.mark.integration
def test_max_files_per_trigger_splits_a_backlog_across_micro_batches(watched):
    data, state = watched
    for i in range(4):
        _write(data, f"f{i}.parquet", [i])
    ds = bt.read.files_incremental(
        str(data), "parquet", state_dir=state, max_files_per_trigger=1
    )

    q = ds.write.memory("inc_rate", trigger=bt.Trigger.available_now())
    assert q.await_termination(timeout=30) is True
    assert sorted(bt.read_memory("inc_rate").to_pydict()["a"]) == [0, 1, 2, 3]
    assert len(q.recent_progress) >= 4, "the backlog was not split one file per micro-batch"
