"""Transient object-store failures must be retried; permanent ones must not.

The IO layer had no retry anywhere, so a 503 slow-down or a dropped connection — the kind
of blip every cloud SDK absorbs — either aborted the query or, under `on_error="skip"`,
was recorded as a *corrupt file* and its rows silently dropped. That second outcome is the
worse one: healthy data turned into missing data by a network hiccup, reported only as a
warning.

The classification is the load-bearing part, so it is tested in both directions: a
throttle retries, a 404 does not (retrying it only delays a clear failure), and an
unrecognized error is treated as permanent so a real bug never burns the retry budget.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batcher.io.base import _transient
from batcher.io.base._transient import is_transient, with_retry
from batcher.io.formats.structured.parquet.source import ParquetSource

pytestmark = pytest.mark.unit


# ---- classification -------------------------------------------------------
@pytest.mark.parametrize(
    "message",
    [
        "SlowDown: Please reduce your request rate",
        "An error occurred (Throttling) when calling GetObject",
        "503 Server Error: Service Unavailable",
        "504 Server Error: Gateway Timeout",
        "429 Client Error: Too Many Requests",
        "Connection reset by peer",
        "The read operation timed out",
    ],
)
def test_blips_are_transient(message: str) -> None:
    assert is_transient(RuntimeError(message))


@pytest.mark.parametrize(
    "message",
    [
        "404 Client Error: Not Found",
        "NoSuchKey: The specified key does not exist",
        "NoSuchBucket: The specified bucket does not exist",
        "An error occurred (AccessDenied) when calling GetObject",
        "403 Client Error: Forbidden",
        "InvalidAccessKeyId: signature mismatch",
    ],
)
def test_facts_are_not_transient(message: str) -> None:
    """Retrying these only delays the same failure behind three backoffs."""
    assert not is_transient(RuntimeError(message))


def test_an_unrecognized_error_is_not_retried() -> None:
    """The default must be 'permanent' so a real bug fails fast instead of retrying."""
    assert not is_transient(ValueError("Parquet magic bytes not found in footer"))


def test_a_socket_timeout_is_transient_whatever_it_says() -> None:
    assert is_transient(TimeoutError("nope"))
    assert is_transient(ConnectionError("nope"))


def test_a_permanent_marker_beats_a_transient_one() -> None:
    """'404 Not Found' mentioning 'error' must not read as retryable."""
    assert not is_transient(RuntimeError("500 Server Error: object not found"))


# ---- the retry loop -------------------------------------------------------
def test_a_transient_failure_is_retried_until_it_succeeds() -> None:
    calls = []

    def _op() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("SlowDown: reduce your request rate")
        return "ok"

    assert with_retry(_op, attempts=3, backoff_base_s=0, sleep=lambda _s: None) == "ok"
    assert len(calls) == 3


def test_a_permanent_failure_is_raised_on_the_first_attempt() -> None:
    calls = []

    def _op() -> str:
        calls.append(1)
        raise RuntimeError("404 Client Error: Not Found")

    with pytest.raises(RuntimeError, match="404"):
        with_retry(_op, attempts=5, backoff_base_s=0, sleep=lambda _s: None)
    assert len(calls) == 1, "a permanent error must not consume the retry budget"


def test_exhausted_attempts_reraise_the_last_failure() -> None:
    def _op() -> str:
        raise RuntimeError("SlowDown: reduce your request rate")

    with pytest.raises(RuntimeError, match="SlowDown"):
        with_retry(_op, attempts=2, backoff_base_s=0, sleep=lambda _s: None)


def test_backoff_grows_and_is_jittered_within_bounds() -> None:
    slept: list[float] = []

    def _op() -> str:
        raise RuntimeError("timed out")

    with pytest.raises(RuntimeError):
        with_retry(_op, attempts=4, backoff_base_s=1.0, sleep=slept.append)

    assert len(slept) == 3
    # Equal jitter: each wait is within [ceiling/2, ceiling] for a doubling ceiling.
    for i, waited in enumerate(slept):
        ceiling = 1.0 * (2**i)
        assert ceiling / 2 <= waited <= ceiling


# ---- wired into the read spine -------------------------------------------
@pytest.fixture
def one_file(tmp_path):
    pq.write_table(pa.table({"id": list(range(10))}), tmp_path / "p0.parquet")
    return str(tmp_path)


def test_read_retries_a_transient_backend_failure(one_file, monkeypatch) -> None:
    """The end-to-end property: a blip on the first GET must not fail the read."""
    no_sleep = type("_NoSleep", (), {"sleep": staticmethod(lambda _s: None)})
    monkeypatch.setattr(_transient, "time", no_sleep)
    real = ParquetSource._read_by_path
    calls = []

    def _flaky(self, path, projection):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("SlowDown: please reduce your request rate")
        return real(self, path, projection)

    monkeypatch.setattr(ParquetSource, "_read_by_path", _flaky)
    src = ParquetSource(one_file)

    rows = sum(b.num_rows for b in src.read())

    assert rows == 10
    assert len(calls) == 2, "expected exactly one retry"


def test_read_does_not_retry_a_corrupt_file(tmp_path, monkeypatch) -> None:
    """A malformed file is not a blip — it must fail (or skip) on the first attempt.

    Guards the budget: before the classifier existed, treating every failure as
    retryable would have re-read every corrupt file `attempts` times for nothing.
    """
    (tmp_path / "bad.parquet").write_bytes(b"not a parquet file")
    calls = []
    real = ParquetSource._read_by_path

    def _counted(self, path, projection):
        calls.append(1)
        return real(self, path, projection)

    monkeypatch.setattr(ParquetSource, "_read_by_path", _counted)
    src = ParquetSource(str(tmp_path), on_error="skip")

    assert sum(b.num_rows for b in src.read()) == 0
    assert src.corrupt_files(), "the corrupt file is still reported"
    assert len(calls) == 1, "a corrupt file must not be retried"


# ---- the write side -------------------------------------------------------
# Retry covered the read path only. An object store answers a burst of concurrent PUTs at
# one key prefix with `SlowDown`/503 — which is exactly the shape of a directory write from
# a fast pipeline — and a throttled PUT killed a job that had already written 99% of its
# output. Retrying is safe here specifically because a failed attempt publishes nothing: the
# atomic writer discards the partial and the table is still in memory.


def _flaky_sink(failures: int, message: str):
    """A ParquetSink whose first `failures` writes raise `message`."""
    from batcher.io.formats.structured.parquet.sink import ParquetSink

    class _Flaky(ParquetSink):
        attempts = 0

        def _write_file(self, table, fh):
            type(self).attempts += 1
            if type(self).attempts <= failures:
                raise OSError(message)
            return super()._write_file(table, fh)

    return _Flaky


def test_a_throttled_write_is_retried(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(_transient.time, "sleep", lambda _s: None)
    sink = _flaky_sink(2, "SlowDown: please reduce your request rate")()
    out = str(tmp_path / "part.parquet")
    written = sink.write(pa.table({"x": [1, 2, 3]}), out)
    assert written.rows == 3
    assert type(sink).attempts == 3, "expected two failures then a success"
    # The retry must have produced real, readable output — not merely stopped raising.
    assert pq.read_table(out).num_rows == 3


def test_a_permanent_write_failure_is_not_retried(tmp_path, monkeypatch) -> None:
    """Retrying an access-denied write only delays a clear auth failure behind backoffs."""
    monkeypatch.setattr(_transient.time, "sleep", lambda _s: None)
    sink = _flaky_sink(99, "AccessDenied: not authorized")()
    with pytest.raises(OSError, match="AccessDenied"):
        sink.write(pa.table({"x": [1]}), str(tmp_path / "p.parquet"))
    assert type(sink).attempts == 1


def test_a_write_that_never_recovers_reraises(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(_transient.time, "sleep", lambda _s: None)
    sink = _flaky_sink(99, "503 server error")()
    with pytest.raises(OSError, match="503"):
        sink.write(pa.table({"x": [1]}), str(tmp_path / "p.parquet"))
    assert type(sink).attempts > 1, "a transient failure must have been retried"


def test_a_retry_after_a_partial_write_does_not_duplicate_rows(tmp_path, monkeypatch) -> None:
    """The guides record "write operations and task retries can produce duplicates" as a
    live failure. Retrying a write is only safe because a failed attempt publishes nothing —
    so the dangerous case is a write that succeeds in *writing bytes* and then fails, which
    is exactly what a throttle mid-upload looks like. The atomic writer must discard it.
    """
    from batcher.io.formats.structured.parquet.sink import ParquetSink

    monkeypatch.setattr(_transient.time, "sleep", lambda _s: None)
    attempts = {"n": 0}

    class _PartialThenThrottled(ParquetSink):
        def _write_file(self, table, fh):
            attempts["n"] += 1
            super()._write_file(table, fh)  # bytes really are written...
            if attempts["n"] == 1:
                raise OSError("SlowDown: throttled after writing")  # ...then it fails

    out = str(tmp_path / "part.parquet")
    _PartialThenThrottled().write(pa.table({"x": [1, 2, 3]}), out)
    assert attempts["n"] == 2, "expected one partial failure then a retry"
    assert pq.read_table(out).num_rows == 3, "the partial write must not have been published"
