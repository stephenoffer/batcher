"""An idle unbounded source is not a finished one.

A source that discovers its work per pass — the Auto Loader's directory listing is the
one that ships — hands back an `iter_batches` generator that *ends* when the pass finds
nothing new. The single-node runner read that as end-of-stream and stopped the query, so
a `files_incremental` stream ran one discovery pass and silently terminated: no error, no
log line, and every file that arrived afterwards ignored forever. The distributed runner
never had the bug (it asks the source for work each epoch), so the two disagreed about
when a stream is over.

These pin the contract both runners now share: only a *drain* trigger, a stop signal, or
a genuinely bounded source ends the stream.
"""

from __future__ import annotations

import threading

import pyarrow as pa
import pytest

from batcher.config import Config, StreamingConfig, config_context
from batcher.core.streaming_runner import LocalRunner
from batcher.plan.streaming import Trigger


class _PerPassSource:
    """An unbounded source whose `iter_batches` ends after each pass, like a directory scan."""

    bounded = False
    continues_across_passes = True

    def __init__(self) -> None:
        self.available: list[pa.RecordBatch] = []
        self.passes = 0

    def schema(self) -> pa.Schema:
        return pa.schema([("a", pa.int64())])

    def iter_batches(self, projection=None):
        self.passes += 1
        ready, self.available = self.available, []
        yield from ready


class _Processor:
    def process(self, batch: pa.RecordBatch) -> list[pa.RecordBatch]:
        return [batch]


class _Sink:
    def __init__(self) -> None:
        self.written: list[pa.Table] = []

    def open(self) -> None:
        pass

    def write_batch(self, batch_id: int, table: pa.Table) -> str:
        self.written.append(table)
        return f"t{batch_id}"

    def close(self) -> None:
        pass


def _batch(*values: int) -> pa.RecordBatch:
    return pa.record_batch({"a": pa.array(list(values), type=pa.int64())})


@pytest.fixture
def fast_idle():
    """Shrink the idle wait so a test that must observe two passes stays quick."""
    cfg = Config().replace(streaming=StreamingConfig(idle_poll_seconds=0.01))
    with config_context(cfg):
        yield


def test_an_exhausted_pass_of_an_unbounded_source_is_re_opened(fast_idle):
    source = _PerPassSource()
    source.available = [_batch(1, 2, 3)]
    stop = threading.Event()
    runner = LocalRunner(source, _Processor(), _Sink(), should_stop=stop.is_set)

    first = runner.stage(0)
    assert first is not None and first.num_rows == 3
    runner.publish(0, first)

    # Nothing available yet: the runner must keep looking, not report end-of-stream.
    def arrive_later() -> None:
        source.available = [_batch(4, 5)]

    threading.Timer(0.05, arrive_later).start()
    second = runner.stage(1)
    assert second is not None, "an idle unbounded source was read as a finished one"
    assert second.column("a").to_pylist() == [4, 5]
    assert source.passes > 1, "the spent iterator was never re-opened"


def test_a_drain_trigger_still_ends_on_an_empty_pass():
    source = _PerPassSource()
    source.available = [_batch(1)]
    runner = LocalRunner(source, _Processor(), _Sink(), drain=True)

    assert runner.stage(0) is not None
    # `once` / `available_now` process what is available and stop — that is the drain.
    assert runner.stage(1) is None


def test_a_stop_signal_ends_an_idle_stream(fast_idle):
    source = _PerPassSource()
    stop = threading.Event()
    runner = LocalRunner(source, _Processor(), _Sink(), should_stop=stop.is_set)

    threading.Timer(0.05, stop.set).start()
    assert runner.stage(0) is None


def test_a_bounded_source_still_ends_when_its_batches_run_out():
    class _Bounded(_PerPassSource):
        bounded = True

    source = _Bounded()
    source.available = [_batch(1, 2)]
    runner = LocalRunner(source, _Processor(), _Sink())

    assert runner.stage(0) is not None
    assert runner.stage(1) is None


@pytest.mark.parametrize(
    ("trigger", "drains"),
    [
        (Trigger.once(), True),
        (Trigger.available_now(), True),
        (Trigger.processing_time("1 second"), False),
        (Trigger.continuous("1 second"), False),
    ],
)
def test_trigger_is_drain_names_the_two_that_stop(trigger, drains):
    assert trigger.is_drain is drains


def test_idle_poll_seconds_must_be_positive():
    with pytest.raises(ValueError, match="idle_poll_seconds"):
        StreamingConfig(idle_poll_seconds=0.0)


def test_progress_history_must_hold_at_least_one_batch():
    with pytest.raises(ValueError, match="progress_history"):
        StreamingConfig(progress_history=0)


def test_an_unbounded_source_that_replays_is_never_re_opened():
    """A source whose next pass hands back the *same* rows must be read exactly once.

    `IteratorSource` calls its batch factory again from the beginning, so re-opening it
    would write the whole stream to the sink a second time — and then a third, forever.
    Declining to re-open is the safe answer, and it is why `continues_across_passes`
    defaults to False rather than being inferred from `bounded`.
    """

    class _Replaying(_PerPassSource):
        continues_across_passes = False

    source = _Replaying()
    source.available = [_batch(1, 2)]
    runner = LocalRunner(source, _Processor(), _Sink())

    assert runner.stage(0) is not None
    assert runner.stage(1) is None
    assert source.passes == 1
