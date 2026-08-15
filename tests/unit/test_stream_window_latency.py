"""A `map_batches` streaming window has to close on age, not only on size.

`stream_windowed` batches source batches into a window before applying the UDF, because a
`map_batches` pipeline only fans across the worker pool when it is handed several batches at
once. Both of its bounds were *size* bounds -- 4,000,000 rows (`target_rows_per_task`) or
128 MiB -- which is the right question for a bounded input and the wrong one for a stream: a
stream is bounded in rate, so a size bound is a duration. At 2,000 rows/s the first window
closed after 33 minutes; on a 10 rows/s device topic, after about 4.6 days. The pipeline was
not hung and not leaking, it was buffering, and nothing said so.

These pin the age bound itself, and the rule that only an unbounded source pays for it.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa
import pytest

import batcher as bt
from batcher.api.terminal.map_stream import stream_windowed
from batcher.api.terminal.stream.pipeline import _window_latency
from batcher.config import Config, config_context
from batcher.io.source import InMemorySource

_SCHEMA = pa.schema([("v", pa.int64())])


def _batches(n: int) -> list[pa.RecordBatch]:
    return [pa.record_batch({"v": [i]}, schema=_SCHEMA) for i in range(n)]


def _windows(latency: float | None, *, target_rows: int = 1_000_000) -> list[int]:
    """The batch-count of every window `run_window` was handed, for `_batches(6)`."""
    seen: list[int] = []

    def run_window(buf: list[pa.RecordBatch]) -> list[pa.RecordBatch]:
        seen.append(len(buf))
        return buf

    list(
        stream_windowed(
            InMemorySource(_batches(6)),
            run_window,
            target_rows,
            None,
            latency_seconds=latency,
        )
    )
    return seen


def test_without_a_latency_bound_the_window_fills_to_its_size_budget():
    """The pre-existing behavior, unchanged: six one-row batches are one window."""
    assert _windows(None) == [6]


def test_an_aged_window_is_flushed_before_it_reaches_its_size_budget():
    """A bound small enough that every batch ages out cuts one window per batch.

    The size budget is a million rows and the input is six, so nothing but the age bound
    can be closing these windows.
    """
    assert _windows(1e-9) == [1, 1, 1, 1, 1, 1]


def test_a_latency_bound_far_in_the_future_never_fires():
    """An hour-long bound leaves the size budget in charge, so the window is whole again.

    Without this, `_windows(1e-9)` above would pass for a `stream_windowed` that flushed
    every batch unconditionally.
    """
    assert _windows(3600.0) == [6]


def test_the_age_is_measured_from_the_window_not_from_the_query():
    """Each window restarts the clock, so a long stream is cut into many windows.

    Measuring from the query's start instead would age out every window after the first,
    which `_windows(1e-9)` cannot tell apart from the correct behavior -- both give one
    batch per window. A bound that fires only *once* is the failure this catches.
    """
    seen: list[int] = []

    def run_window(buf: list[pa.RecordBatch]) -> list[pa.RecordBatch]:
        seen.append(len(buf))
        return buf

    # Two rows per window: the first batch opens it, the second finds it already aged.
    list(stream_windowed(InMemorySource(_batches(6)), run_window, 2, None, latency_seconds=3600.0))
    assert seen == [2, 2, 2]


def test_the_window_yields_every_row_whatever_the_bound_cuts():
    """Cutting a window early may not lose or duplicate a row -- the result is unchanged."""

    def run_window(buf: list[pa.RecordBatch]) -> list[pa.RecordBatch]:
        return buf

    for latency in (None, 1e-9, 3600.0):
        out = list(
            stream_windowed(InMemorySource(_batches(6)), run_window, 1_000_000, None, None, latency)
        )
        rows = [v for b in out for v in b.column("v").to_pylist()]
        assert rows == [0, 1, 2, 3, 4, 5], latency


def test_a_bounded_source_gets_no_latency_bound():
    """Batch keeps the pure size-based window, so its throughput is untouched."""
    assert _window_latency(InMemorySource(_batches(1))) is None


def test_an_unbounded_source_gets_the_configured_bound():
    ds = bt.from_batches(lambda: iter(_batches(3)), _SCHEMA, bounded=False)
    (source,) = ds._sources
    assert _window_latency(source) == pytest.approx(Config().streaming.max_window_latency_seconds)


def test_a_non_positive_setting_restores_the_size_only_window():
    """`0` is how an operator asks for the old behavior back, not for a flush per batch."""
    base = Config()
    cfg = base.replace(
        streaming=dataclasses.replace(base.streaming, max_window_latency_seconds=0.0)
    )
    ds = bt.from_batches(lambda: iter(_batches(3)), _SCHEMA, bounded=False)
    (source,) = ds._sources
    with config_context(cfg):
        assert _window_latency(source) is None
