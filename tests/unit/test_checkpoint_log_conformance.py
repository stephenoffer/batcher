"""The SQLite and file-per-batch checkpoint logs must answer identically.

There are two implementations because a local location wants a lockable seekable file and
an object store has none, and two implementations of one contract drift. This drives both
through the same operation sequence and compares every answer, so a change to one that the
other does not get is a failing test rather than a checkpoint that recovers differently
depending on where it was written.

The file log is driven over a plain local directory here, which is not a stand-in for
object-store semantics. It does not need to be: the file log's whole job is to go through
the filesystem façade instead of SQLite, and the façade *is* the S3/GCS/HDFS backend. What
this file pins is that it answers the same questions the same way. Which backend a location
picks, and that a URI never becomes a local directory, are the last three tests.
"""

from __future__ import annotations

import pytest

from batcher.io.formats.streaming.checkpoint.fs_logs import FileCommitLog, FileOffsetLog
from batcher.io.formats.streaming.checkpoint.logs import CommitLog, OffsetLog

pytestmark = pytest.mark.unit


@pytest.fixture
def offset_logs(tmp_path):
    """One of each offset log, on disjoint directories."""
    sqlite = OffsetLog(str(tmp_path / "offsets.sqlite"))
    files = FileOffsetLog(str(tmp_path / "offsets"))
    yield sqlite, files
    sqlite.close()
    files.close()


@pytest.fixture
def commit_logs(tmp_path):
    """One of each commit log, on disjoint directories."""
    sqlite = CommitLog(str(tmp_path / "commits.sqlite"))
    files = FileCommitLog(str(tmp_path / "commits"))
    yield sqlite, files
    sqlite.close()
    files.close()


def _both(logs, call):
    """Apply `call` to each implementation and return the pair of answers."""
    return tuple(call(log) for log in logs)


def _agree(logs, call):
    """Assert both implementations answered the same thing, and return it."""
    first, second = _both(logs, call)
    assert first == second, f"the two logs disagree: {first!r} vs {second!r}"
    return first


class TestOffsetLog:
    def test_an_empty_log_has_no_latest_batch(self, offset_logs) -> None:
        assert _agree(offset_logs, lambda log: log.latest_batch()) is None
        assert _agree(offset_logs, lambda log: log.position_at(0)) == {}

    def test_positions_round_trip_and_report_the_latest(self, offset_logs) -> None:
        for batch_id in (0, 1, 2):
            _both(offset_logs, lambda log, b=batch_id: log.record(b, 0, {"offsets": {"0": b}}))
        assert _agree(offset_logs, lambda log: log.latest_batch()) == 2
        assert _agree(offset_logs, lambda log: log.position_at(1)) == {0: {"offsets": {"0": 1}}}

    def test_re_recording_a_batch_overwrites_rather_than_duplicates(self, offset_logs) -> None:
        """A replayed epoch records its position again; the log must hold one answer."""
        _both(offset_logs, lambda log: log.record(4, 0, {"offsets": {"0": 1}}))
        _both(offset_logs, lambda log: log.record(4, 0, {"offsets": {"0": 9}}))
        assert _agree(offset_logs, lambda log: log.position_at(4)) == {0: {"offsets": {"0": 9}}}

    def test_several_sources_share_one_batch(self, offset_logs) -> None:
        _both(offset_logs, lambda log: log.record(7, 0, {"o": 1}))
        _both(offset_logs, lambda log: log.record(7, 1, {"o": 2}))
        assert _agree(offset_logs, lambda log: log.position_at(7)) == {0: {"o": 1}, 1: {"o": 2}}

    def test_prune_keeps_the_recovery_row_and_drops_the_rest(self, offset_logs) -> None:
        for batch_id in range(5):
            _both(offset_logs, lambda log, b=batch_id: log.record(b, 0, {"o": b}))
        _both(offset_logs, lambda log: log.prune(3))
        assert _agree(offset_logs, lambda log: log.position_at(2)) == {}
        assert _agree(offset_logs, lambda log: log.position_at(3)) == {0: {"o": 3}}
        assert _agree(offset_logs, lambda log: log.latest_batch()) == 4


class TestCommitLog:
    def test_an_empty_log_has_committed_nothing(self, commit_logs) -> None:
        assert _agree(commit_logs, lambda log: log.last_committed()) is None
        assert _agree(commit_logs, lambda log: log.is_committed(0)) is False
        assert _agree(commit_logs, lambda log: log.sink_token(0)) is None

    def test_a_commit_carries_its_sink_token(self, commit_logs) -> None:
        _both(commit_logs, lambda log: log.commit(3, "delta:3:120"))
        assert _agree(commit_logs, lambda log: log.last_committed()) == 3
        assert _agree(commit_logs, lambda log: log.is_committed(3)) is True
        assert _agree(commit_logs, lambda log: log.sink_token(3)) == "delta:3:120"

    def test_a_commit_with_no_token_is_still_a_commit(self, commit_logs) -> None:
        """A batch that wrote no rows commits with a NULL token, not with no row."""
        _both(commit_logs, lambda log: log.commit(1))
        assert _agree(commit_logs, lambda log: log.is_committed(1)) is True
        assert _agree(commit_logs, lambda log: log.sink_token(1)) is None

    def test_recommitting_updates_the_token(self, commit_logs) -> None:
        _both(commit_logs, lambda log: log.commit(2, "first"))
        _both(commit_logs, lambda log: log.commit(2, "second"))
        assert _agree(commit_logs, lambda log: log.sink_token(2)) == "second"

    def test_prune_preserves_the_maximum(self, commit_logs) -> None:
        """Recovery reads only the maximum, so pruning must never lower it."""
        for batch_id in range(6):
            _both(commit_logs, lambda log, b=batch_id: log.commit(b, f"t{b}"))
        _both(commit_logs, lambda log: log.prune(4))
        assert _agree(commit_logs, lambda log: log.last_committed()) == 5
        assert _agree(commit_logs, lambda log: log.is_committed(3)) is False
        assert _agree(commit_logs, lambda log: log.is_committed(4)) is True


@pytest.mark.parametrize(
    ("location", "local"),
    [
        ("/var/lib/ckpt", True),
        ("relative/ckpt", True),
        ("file:///var/lib/ckpt", True),
        ("s3://bucket/ckpt", False),
        ("gs://bucket/ckpt", False),
        ("hdfs://namenode:8020/ckpt", False),
        ("abfss://c@a.dfs.core.windows.net/ckpt", False),
    ],
)
def test_which_locations_count_as_node_local(location: str, local: bool) -> None:
    """One answer for the durability warning and for the store's backend choice.

    They disagreed: the warning recommended ``s3://`` while the store resolved it with
    `os.makedirs`, producing a local directory named ``s3:`` that a reclaimed node took
    with it — the exact failure the warning existed to prevent.
    """
    from batcher.io.formats.streaming.checkpoint import is_local_location

    assert is_local_location(location) is local


def test_a_remote_location_routes_to_the_file_logs_and_never_to_os_makedirs(
    monkeypatch, tmp_path
) -> None:
    """An ``s3://`` checkpoint must build file logs under it, not a local directory.

    `CheckpointDir` is the only thing that touches the filesystem here, so replacing it
    records exactly which roots the store asked for without needing a bucket. `os.makedirs`
    is barred outright, because calling it on a URI is precisely the old bug.
    """
    import os as os_mod

    from batcher.io.formats.streaming.checkpoint import fs_logs, state_store
    from batcher.io.formats.streaming.checkpoint import store as store_mod

    roots: list[str] = []

    class _RecordingDir:
        def __init__(self, root: str) -> None:
            roots.append(root)

    monkeypatch.setattr(fs_logs, "CheckpointDir", _RecordingDir)
    monkeypatch.setattr(state_store, "CheckpointDir", _RecordingDir)
    monkeypatch.setattr(
        os_mod, "makedirs", lambda *a, **k: pytest.fail(f"os.makedirs called on {a[0]!r}")
    )

    built = store_mod.CheckpointStore("s3://bucket/ckpt/")
    assert type(built.offsets) is fs_logs.FileOffsetLog
    assert type(built.commits) is fs_logs.FileCommitLog
    assert roots == [
        "s3://bucket/ckpt/offsets",
        "s3://bucket/ckpt/commits",
        "s3://bucket/ckpt/state",
    ]


def test_a_whole_checkpoint_round_trips_over_a_non_local_scheme() -> None:
    """The end-to-end claim: offsets, state and commits on a URI, and recovery reads them.

    `memory://` is an fsspec filesystem, so this exercises the real
    `resolve_filesystem` fallback rather than a stand-in — the same code path an ``s3://``
    location takes, minus the network. What it proves is what was broken: the whole
    checkpoint lands *under the URI*, and no local directory named ``memory:`` appears.
    """
    import os

    import pyarrow as pa

    pytest.importorskip("fsspec")
    from batcher.io.formats.streaming.checkpoint import CheckpointStore, recover

    store = CheckpointStore("memory://bt-ckpt-roundtrip")
    store.record_offsets(0, {0: {"offsets": {"0": 7}}})
    store.snapshot_state(0, pa.record_batch({"k": [1], "v": [2]}))
    store.commit(0, "tok")
    store.record_offsets(1, {0: {"offsets": {"0": 9}}})
    store.commit(1, None)

    assert store.state.restore(0).to_pydict() == {"k": [1], "v": [2]}
    plan = recover(store)
    assert plan.start_batch == 2
    assert plan.seek == {0: {"offsets": {"0": 9}}}
    assert not os.path.exists("memory:"), "a URI was resolved as a local directory"


def test_a_local_location_keeps_the_sqlite_logs(tmp_path) -> None:
    """SQLite is the right tool where there is a lockable seekable file; keep using it."""
    from batcher.io.formats.streaming.checkpoint import CheckpointStore

    built = CheckpointStore(str(tmp_path / "ckpt"))
    try:
        assert type(built.offsets) is OffsetLog
        assert type(built.commits) is CommitLog
        assert (tmp_path / "ckpt" / "offsets.sqlite").exists()
    finally:
        built.close()
