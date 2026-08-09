"""A streaming write to a table format writes *that* format.

`TransactionalStreamSink` replaced a Delta-pinned sink that the conductor routed every
mode-aware format to, so `write(path, format="iceberg", trigger=...)` produced a Delta
table at the Iceberg path — right rows, right location, wrong format, no error anywhere.
These pin the routing, the per-batch transaction where a format has one, and the honest
warning where it does not.
"""

from __future__ import annotations

from typing import ClassVar

import pyarrow as pa
import pytest

from batcher._internal.errors import CommitError
from batcher.io.formats import SINKS
from batcher.io.formats.streaming.sinks import (
    DeltaStreamSink,
    FileStreamSink,
    TransactionalStreamSink,
)

_TABLE = pa.table({"a": pa.array([1, 2], type=pa.int64())})


class _Recording:
    """A transactional sink that records what it was asked to do.

    Class-level, because the sink under test constructs a *fresh* instance per
    micro-batch: an instance attribute would be discarded before the test could read it.
    The `registered` fixture resets them.
    """

    calls: ClassVar[list[tuple[str, tuple]]] = []
    committed: ClassVar[set[tuple[str, int]]] = set()
    fail_commit_once = False

    def __init__(self, *, app_id=None, txn_version=None, **opts):
        self.app_id = app_id
        self.txn_version = txn_version
        self.opts = opts

    def is_committed(self, uri) -> bool:
        return (self.app_id, self.txn_version) in _Recording.committed

    def write(self, table, uri):
        _Recording.calls.append(("write", (uri, table.num_rows)))
        from batcher.io.manifest import WrittenFile

        return WrittenFile(path=f"{uri}/f", rows=table.num_rows, bytes=1)

    def commit(self, manifest, uri):
        _Recording.calls.append(("commit", (uri, self.app_id, self.txn_version)))
        if _Recording.fail_commit_once:
            _Recording.fail_commit_once = False
            _Recording.committed.add((self.app_id, self.txn_version))
            raise CommitError("lost the optimistic race")
        _Recording.committed.add((self.app_id, self.txn_version))


class _Appending:
    """A table sink with no per-batch transaction marker, like Iceberg's."""

    calls: ClassVar[list[tuple[str, tuple]]] = []

    def __init__(self, **opts):
        self.opts = opts

    def write(self, table, uri):
        _Appending.calls.append(("write", (uri, table.num_rows)))
        from batcher.io.manifest import WrittenFile

        return WrittenFile(path=f"{uri}/f", rows=table.num_rows, bytes=1)

    def commit(self, manifest, uri):
        _Appending.calls.append(("commit", (uri,)))


@pytest.fixture
def registered():
    """Register the two stand-in formats, and take them back out again."""
    _Recording.calls, _Recording.committed = [], set()
    _Recording.fail_commit_once = False
    _Appending.calls = []
    SINKS.register("_txn_test")(_Recording)
    SINKS.register("_append_test")(_Appending)
    yield
    SINKS._items.pop("_txn_test", None)
    SINKS._items.pop("_append_test", None)


def test_the_destination_format_is_the_one_written(registered):
    sink = TransactionalStreamSink("t://x", "_txn_test", query_name="q")
    sink.open()
    sink.write_batch(0, _TABLE)
    assert [c[0] for c in _Recording.calls] == ["write", "commit"]
    assert _Appending.calls == []


def test_each_micro_batch_carries_its_own_transaction_id(registered):
    sink = TransactionalStreamSink("t://x", "_txn_test", query_name="q")
    sink.open()
    sink.write_batch(0, _TABLE)
    sink.write_batch(1, _TABLE)
    commits = [c[1] for c in _Recording.calls if c[0] == "commit"]
    assert commits == [("t://x", "q", 0), ("t://x", "q", 1)]


def test_a_replayed_micro_batch_writes_nothing_and_commits_nothing(registered):
    sink = TransactionalStreamSink("t://x", "_txn_test", query_name="q")
    sink.open()
    sink.write_batch(0, _TABLE)
    _Recording.calls.clear()
    token = sink.write_batch(0, _TABLE)
    assert _Recording.calls == [], "a replayed batch wrote a second copy of its rows"
    assert token.endswith("already-committed")


def test_a_lost_commit_race_whose_transaction_landed_is_benign(registered):
    """The pre-check is not atomic with the commit, so a concurrent writer under the same
    app id can win. Losing is only a failure if the transaction still is not recorded."""
    _Recording.fail_commit_once = True
    sink = TransactionalStreamSink("t://x", "_txn_test", query_name="q")
    sink.open()
    assert sink.write_batch(0, _TABLE).endswith("already-committed")


def test_the_query_name_is_the_stable_application_id(registered):
    sink = TransactionalStreamSink("t://x", "_txn_test", query_name="nightly")
    sink.open()
    sink.write_batch(0, _TABLE)
    assert ("commit", ("t://x", "nightly", 0)) in _Recording.calls


def test_without_a_query_name_the_app_id_is_derived_from_the_table(registered):
    sink = TransactionalStreamSink("t://x/", "_txn_test")
    sink.open()
    sink.write_batch(0, _TABLE)
    assert ("commit", ("t://x/", "batcher-stream:t://x", 0)) in _Recording.calls


def test_a_format_without_a_transaction_marker_warns_once_and_appends(registered):
    sink = TransactionalStreamSink("t://x", "_append_test")
    with pytest.warns(UserWarning, match="at-least-once"):
        sink.open()
    sink.write_batch(0, _TABLE)
    sink.write_batch(0, _TABLE)  # a replay: it appends again, which is the point
    assert [c[0] for c in _Appending.calls] == ["write", "commit", "write", "commit"]


def test_the_delta_sink_is_the_transactional_sink_pinned_to_delta():
    sink = DeltaStreamSink("t://x", query_name="q")
    assert isinstance(sink, TransactionalStreamSink)
    assert sink._fmt == "delta"


def test_the_conductor_routes_a_table_format_to_the_table_sink():
    from batcher.api.io_namespace.writer import Writer

    build = Writer._stream_sink_for
    for fmt in ("delta", "iceberg", "hudi"):
        sink = build(None, "t://x", fmt, {}, "q")
        assert isinstance(sink, TransactionalStreamSink)
        assert sink._fmt == fmt, f"{fmt} was routed to a {sink._fmt} sink"


def test_the_conductor_routes_a_plain_file_format_to_the_file_sink():
    from batcher.api.io_namespace.writer import Writer

    sink = Writer._stream_sink_for(None, "/tmp/out", "parquet", {}, "q")
    assert isinstance(sink, FileStreamSink)


def test_iceberg_gets_the_constructor_kwargs_its_sink_needs():
    """The batch write path knew Iceberg needs its identifier at construction; the
    streaming sink did not, and failed on the first micro-batch with a bare TypeError."""
    from batcher.io.sink import table_sink_kwargs

    kwargs = table_sink_kwargs("iceberg", "ns.tbl")
    assert kwargs["identifier"] == "ns.tbl"
    assert kwargs["write_token"]
    assert table_sink_kwargs("delta", "t://x") == {}
