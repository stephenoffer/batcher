"""Exactly-once checkpoint/resume for broker sources (Kafka/Kinesis/… base).

The broker base (`BrokerSource`) tracks the latest offset per partition and exposes
`snapshot_position`/`seek`, so a streaming write with a `checkpoint=` directory
resumes strictly after the last *committed* micro-batch on restart — no message lost
or duplicated. This drives a finite, replayable in-repo broker (no client needed)
through the real `StreamingQueryEngine` + `CheckpointStore`, the same path Kafka and
Kinesis take.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.io.formats.base import SOURCES
from batcher.io.formats.streaming.broker import BrokerMessage, BrokerSource

pytestmark = pytest.mark.integration


@SOURCES.register("ckpt_test_broker")
class _CheckpointTestBroker(BrokerSource):
    """A finite, replayable broker that honors `_resume_from` (offsets 0..total-1)."""

    format_name = "ckpt_test_broker"
    __slots__ = ("_cursor", "_started", "_total")

    def __init__(self, topic: str = "t", *, total: int = 20, poll_size: int = 4, **opts):
        super().__init__(topic, poll_size=poll_size, **opts)
        self._total = total
        self._cursor = 0
        self._started = False

    def _discover_partitions(self):
        return [0]

    def _poll(self):
        if not self._started:
            self._started = True
            resume = self._resume_from.get(0)
            self._cursor = 0 if resume is None else int(resume) + 1
        if self._cursor >= self._total:
            return None
        end = min(self._cursor + self.poll_size, self._total)
        msgs = [
            BrokerMessage(
                value=str(o).encode(), partition=0, offset=o, timestamp=o, topic=self.topic
            )
            for o in range(self._cursor, end)
        ]
        self._cursor = end
        return msgs


def _offsets(path: str) -> list[int]:
    return sorted(bt.read(path, format="parquet").to_pydict()["offset"])


def test_broker_restart_resumes_exactly_once(tmp_path):
    out = str(tmp_path / "out")
    ckpt = str(tmp_path / "ckpt")

    # Run A: drain a 12-message stream, committing each micro-batch's offset.
    q1 = bt.read.table("ckpt_test_broker", total=12).write(
        out, format="parquet", trigger=bt.Trigger.available_now(), checkpoint=ckpt
    )
    q1.await_termination()
    assert _offsets(out) == list(range(12))

    # Run B (restart): the same checkpoint over a longer 20-message stream resumes
    # strictly after offset 11 — only 12..19 are appended, none replayed.
    q2 = bt.read.table("ckpt_test_broker", total=20).write(
        out, format="parquet", trigger=bt.Trigger.available_now(), checkpoint=ckpt
    )
    q2.await_termination()
    assert _offsets(out) == list(range(20))  # exactly-once: no loss, no duplicates


def test_broker_restart_after_complete_is_idempotent(tmp_path):
    out = str(tmp_path / "out")
    ckpt = str(tmp_path / "ckpt")
    for _ in range(2):  # same checkpoint + output, run twice
        q = bt.read.table("ckpt_test_broker", total=8).write(
            out, format="parquet", trigger=bt.Trigger.available_now(), checkpoint=ckpt
        )
        q.await_termination()
    assert _offsets(out) == list(range(8))  # second run recovers fully; nothing new
