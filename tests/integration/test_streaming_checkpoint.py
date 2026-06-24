"""Workstream D — exactly-once streaming recovery across a restart.

A streaming write with a `checkpoint=` directory records source offsets and sink
commits per micro-batch. Restarting the same query against the same checkpoint +
output resumes from the last committed offset (the replayable `rate` source seeks to
its cursor), so the combined output is exactly-once: no row lost, none duplicated.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytestmark = pytest.mark.integration


def test_restart_resumes_exactly_once(tmp_path):
    out = str(tmp_path / "out")
    ckpt = str(tmp_path / "ckpt")

    # Run A: process the first 12 of an eventual 20 rows, then "stop" (available_now
    # over a 12-row bounded rate stream), committing each micro-batch to the checkpoint.
    q1 = bt.read.rate(5, num_rows=12, pace=False).write(
        out, format="parquet", trigger=bt.Trigger.available_now(), checkpoint=ckpt
    )
    q1.await_termination()
    assert sorted(bt.read(out, format="parquet").to_pydict()["value"]) == list(range(12))

    # Run B (restart): the same query over the full 20-row stream resumes from the
    # checkpoint — the rate source seeks to value=12, so only 12..19 are produced and
    # appended. No row is reprocessed or skipped.
    q2 = bt.read.rate(5, num_rows=20, pace=False).write(
        out, format="parquet", trigger=bt.Trigger.available_now(), checkpoint=ckpt
    )
    q2.await_termination()
    final = sorted(bt.read(out, format="parquet").to_pydict()["value"])
    assert final == list(range(20))  # exactly-once: 0..19, no loss, no duplicates


def test_restart_after_complete_is_idempotent(tmp_path):
    out = str(tmp_path / "out")
    ckpt = str(tmp_path / "ckpt")
    for _ in range(2):  # run twice with the same checkpoint + output
        q = bt.read.rate(4, num_rows=8, pace=False).write(
            out, format="parquet", trigger=bt.Trigger.available_now(), checkpoint=ckpt
        )
        q.await_termination()
    # The second run recovers fully-committed state and produces nothing new.
    assert sorted(bt.read(out, format="parquet").to_pydict()["value"]) == list(range(8))


def test_crash_midstream_resumes_exactly_once(tmp_path):
    import time

    out = str(tmp_path / "out")
    ckpt = str(tmp_path / "ckpt")

    # Run 1: a *paced* (slow) unbounded rate stream — stop it mid-flight to simulate a
    # crash after some micro-batches have committed their offsets to the checkpoint.
    q1 = bt.read.rate(10, pace=True).write(
        out, format="parquet", trigger=bt.Trigger.processing_time(0), checkpoint=ckpt
    )
    time.sleep(0.3)
    q1.stop()
    committed = sorted(bt.read(out, format="parquet").to_pydict()["value"])
    assert committed == list(range(len(committed))) and 0 < len(committed) < 50

    # Run 2 (restart): a bounded run over the same checkpoint resumes from the last
    # committed rate cursor — no value reprocessed or skipped.
    q2 = bt.read.rate(10, num_rows=50, pace=False).write(
        out, format="parquet", trigger=bt.Trigger.available_now(), checkpoint=ckpt
    )
    q2.await_termination()
    assert sorted(bt.read(out, format="parquet").to_pydict()["value"]) == list(range(50))


def test_rate_source_is_checkpointable():
    from batcher.io.formats.streaming.dev import RateSource

    src = RateSource(5, num_rows=20)
    batch = next(src.iter_batches())
    assert src.snapshot_position() == {"value": batch.num_rows}  # advanced past the first batch
    src.seek({"value": 12})
    assert src.snapshot_position() == {"value": 12}
