"""A broker must advance its offsets only after an epoch is *published* — never at poll time.

The micro-batch contract (`core/streaming_query.py::_process_next`) is: **stage** the epoch,
**write-ahead** the source position it consumed, then **publish** it. A crash can therefore only
lose an epoch that was staged and not published, which the next run replays into an idempotent
sink.

Committing the broker's own offsets inside `_poll` breaks that. The offsets advance the moment
the messages are *read*, so a crash between the poll and the publish leaves the broker believing
those messages were handled: on restart it resumes past them and they are never processed. That
is at-most-once — silent data loss — and it is the opposite of what the Kafka source's docstring
promised. It shipped that way.

`BrokerSource.iter_batches` is a generator, so control returns past its `yield` only when the
consumer asks for the *next* batch — i.e. after the previous epoch was published. That is the
one safe commit point, and these tests pin it.
"""

from __future__ import annotations

import pytest

from batcher.io.formats.streaming.broker import BrokerMessage, BrokerSource

pytestmark = pytest.mark.unit


class _RecordingBroker(BrokerSource):
    """A finite broker that records the order of its polls and commits."""

    format_name = "commit_order_test_broker"
    __slots__ = ("_cursor", "_total", "events")

    def __init__(self, topic: str = "t", *, total: int = 3, **opts):
        super().__init__(topic, poll_size=1, **opts)
        self._total = total
        self._cursor = 0
        self.events: list[str] = []

    def _discover_partitions(self):
        return [0]

    def _poll(self):
        if self._cursor >= self._total:
            return None
        offset = self._cursor
        self._cursor += 1
        self.events.append(f"poll{offset}")
        return [
            BrokerMessage(
                value=str(offset).encode(),
                partition=0,
                offset=offset,
                timestamp=offset,
                topic=self.topic,
            )
        ]

    def _commit_delivered(self) -> None:
        self.events.append(f"commit{self._cursor - 1}")


def test_offsets_are_not_committed_while_a_batch_is_still_unpublished():
    src = _RecordingBroker(total=3)
    stream = src.iter_batches()

    next(stream)  # the engine now *holds* batch 0 — it has not staged or published it
    assert src.events == ["poll0"], (
        "the broker committed offset 0 while the engine was still holding the batch: a crash "
        "here would skip those messages forever"
    )

    next(stream)  # asking for batch 1 means batch 0 was published
    assert src.events == ["poll0", "commit0", "poll1"]

    next(stream)
    assert src.events == ["poll0", "commit0", "poll1", "commit1", "poll2"]


def test_every_delivered_batch_is_eventually_committed_when_the_stream_is_drained():
    """A fully-drained stream commits every batch it handed out — no offset left behind."""
    src = _RecordingBroker(total=3)
    assert len(list(src.iter_batches())) == 3

    commits = [e for e in src.events if e.startswith("commit")]
    # Draining the generator resumes past the final `yield`, so the last batch commits too.
    assert commits == ["commit0", "commit1", "commit2"]


def test_the_base_commit_hook_is_a_no_op_for_a_broker_with_no_server_side_offsets():
    """A broker whose only offset store is Batcher's checkpoint log has nothing to advance."""

    class _Plain(BrokerSource):
        format_name = "plain_test_broker"
        __slots__ = ()

        def _discover_partitions(self):
            return [0]

        def _poll(self):
            return None

    _Plain("t")._commit_delivered()  # must not raise
