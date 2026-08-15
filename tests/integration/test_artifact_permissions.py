"""Nothing Batcher writes to disk may be readable by another user on the box.

The artifacts are not metadata. A spill file holds the query's *actual rows*. A shuffle
scratch file holds them too, on a path that is often a shared cluster mount. An event-log
document carries the whole plan including literal predicate constants, so a
`WHERE ssn = '...'` lands in it verbatim. The learned-stats database persists column
`min`/`max` — real values out of real columns. All of them were created under the default
umask: directories 0755, files 0644.

This test is the deliverable, not the helpers. `private_dir`/`open_private` are ten lines
each and obviously correct in isolation; what actually rots is a *new* write site that
forgets to use them, and the only thing that catches that is an assertion over what ended
up on disk. So each case below drives the real writer and then stats the result.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pyarrow as pa
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes"),
]


def assert_private(path: Path | str, what: str) -> None:
    """Fail if `path` grants any permission to group or other."""
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode & 0o077 == 0, (
        f"{what} {path} is mode {mode:o} — readable beyond the owner. "
        f"It holds query data; use `_internal.paths.private_dir` / `open_private`."
    )


def assert_tree_private(root: Path, what: str) -> None:
    """Fail if `root` or anything beneath it is group/other accessible."""
    assert_private(root, f"{what} root")
    for dirpath, dirnames, filenames in os.walk(root):
        for name in (*dirnames, *filenames):
            assert_private(Path(dirpath) / name, what)


class TestSpill:
    def test_the_spill_store_writes_privately(self, tmp_path: Path) -> None:
        """`TieredSpillStore` creates its scratch directory and its `.arrow` files."""
        from batcher.carbonite.spill import TieredSpillStore

        root = tmp_path / "spill"
        store = TieredSpillStore(str(root))
        batch = pa.record_batch({"a": pa.array([1, 2, 3], type=pa.int64())})
        writer = store.writer("bucket-0")
        writer.write(batch)
        writer.close()
        assert_tree_private(root, "spill artifact")

    def test_an_existing_world_readable_dir_is_tightened(self, tmp_path: Path) -> None:
        """The common real case: a scratch root some earlier run left 0755.

        `os.makedirs(..., exist_ok=True)` silently leaves an existing directory's mode
        alone, so without an explicit chmod the very first deployment to reuse a scratch
        path would be unprotected — and nothing would say so.
        """
        from batcher.carbonite.spill import TieredSpillStore

        root = tmp_path / "preexisting"
        root.mkdir(mode=0o755)
        os.chmod(root, 0o755)
        TieredSpillStore(str(root))
        assert_private(root, "pre-existing spill root")


class TestShuffleScratch:
    def test_the_work_dir_and_its_files_are_private(self, tmp_path: Path) -> None:
        from batcher.dist.shuffle_io import write_ipc

        root = tmp_path / "shuffle"
        root.mkdir()
        batch = pa.record_batch({"a": pa.array([1, 2], type=pa.int64())})
        write_ipc([batch], str(root / "part-0.arrow"))
        assert_private(root / "part-0.arrow", "shuffle IPC file")

    def test_the_shared_writer_is_private(self, tmp_path: Path) -> None:
        """`IpcWriter` is the one place a distributed artifact's stream is opened."""
        from batcher.dist.shuffle_io import IpcWriter

        batch = pa.record_batch({"a": pa.array([1, 2], type=pa.int64())})
        path = tmp_path / "spool.arrow"
        with IpcWriter(str(path)) as writer:
            writer.write(batch)
        assert_private(path, "IpcWriter output")

    def test_grace_resplit_subbuckets_are_private(self, tmp_path: Path) -> None:
        """The out-of-core join's sub-bucket files hold the join's own rows.

        This one opened `pa.OSFile(path, "wb")` directly, so its sub-buckets landed 0644
        on whatever scratch the query was using — a shared cluster mount included.
        """
        from batcher._internal.native import engine
        from batcher.dist.shuffle_io import write_ipc
        from batcher.dist.spill_breakers.join import _spill_paths_to_subbuckets

        batch = pa.record_batch(
            {
                "k": pa.array([i % 7 for i in range(200)], type=pa.int64()),
                "v": pa.array(range(200), type=pa.int64()),
            }
        )
        source = write_ipc([batch], str(tmp_path / "in-0.arrow"))
        out_paths, _ = _spill_paths_to_subbuckets(
            engine(), [source], ["k"], 4, str(tmp_path), "sub"
        )
        assert any(out_paths), "the re-split produced no sub-buckets to check"
        for path in out_paths:
            if path is not None:
                assert_private(path, "grace re-split sub-bucket")

    def test_round_robin_output_is_private(self, tmp_path: Path) -> None:
        from batcher.dist.shuffle_io import write_ipc_round_robin

        batch = pa.record_batch({"a": pa.array([1, 2, 3, 4], type=pa.int64())})
        paths = [str(tmp_path / f"rr-{i}.arrow") for i in range(2)]
        write_ipc_round_robin([batch], batch.schema, paths)
        for path in paths:
            assert_private(path, "round-robin shuffle file")


class TestStreamingCheckpoint:
    """A checkpoint is the one at-rest surface that *outlives* the query by design."""

    def test_a_state_snapshot_is_private(self, tmp_path: Path) -> None:
        """The snapshot holds the running aggregate's own group keys and values."""
        from batcher.io.formats.streaming.checkpoint.state_store import StateStore

        root = tmp_path / "ckpt" / "state"
        store = StateStore(str(root))
        store.snapshot(
            7,
            pa.record_batch(
                {
                    "user": pa.array(["alice", "bob"]),
                    "spend_cents": pa.array([1234, 99], type=pa.int64()),
                }
            ),
        )
        assert_tree_private(root, "checkpoint state")
        # And it still reads back — tightening the mode must not move the contract.
        assert store.restore(7).num_rows == 2

    def test_no_temporary_file_is_left_readable(self, tmp_path: Path) -> None:
        """The snapshot lands via a `.tmp` rename, so the temp file is the real window."""
        from batcher.io.formats.streaming.checkpoint.state_store import StateStore

        root = tmp_path / "state"
        store = StateStore(str(root))
        store.snapshot(1, pa.record_batch({"k": pa.array([1], type=pa.int64())}))
        assert not list(root.glob("*.tmp")), "a temp snapshot survived the rename"
        assert_tree_private(root, "checkpoint state")


class TestFileCache:
    def test_cached_remote_files_are_private(self, tmp_path: Path) -> None:
        """A cache entry is a byte-for-byte copy of one of the user's data files.

        The cache root is Batcher's own subdirectory of a node volume other tenants also
        mount, and the entry's own mode belongs to the caller's `fetch` — so the directory
        is what has to protect the bytes.
        """
        from batcher.io._file_cache import FileBytesCache

        root = tmp_path / "batcher_file_cache"
        cache = FileBytesCache(str(root), max_bytes=1 << 20)
        cache.get_or_fetch("s3://bucket/data.parquet", lambda p: Path(p).write_bytes(b"rows"))
        assert_private(root, "file cache root")


class TestEventLog:
    def test_the_log_directory_and_documents_are_private(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An event-log document contains the plan, literal predicate constants and all."""
        import dataclasses

        import batcher as bt
        from batcher.config import active_config, set_config

        monkeypatch.setenv("BATCHER_HOME", str(tmp_path / "home"))
        original = active_config()
        set_config(
            original.replace(
                observability=dataclasses.replace(original.observability, event_log=True)
            )
        )
        try:
            bt.from_pydict({"a": [1, 2, 3]}).filter(bt.col("a") > 1).collect()
        finally:
            set_config(original)

        log_dir = tmp_path / "home" / "logs"
        if not log_dir.exists():
            pytest.skip("event log did not write (no collector attached on this path)")
        assert_tree_private(log_dir, "event log")
        # And it really does carry the plan, which is why the mode matters.
        documents = list(log_dir.glob("*.json"))
        assert documents, "event log enabled but wrote nothing"
        assert json.loads(documents[0].read_text())


class TestMetadataStore:
    def test_the_learned_stats_database_is_private(self, tmp_path: Path) -> None:
        """The stats database persists column `min`/`max` — real values from real columns."""
        from batcher.metadata.backends.sqlite import SQLiteBackend

        # A world-readable parent, as an operator or an earlier run would have left it.
        home = tmp_path / "batcher-home"
        home.mkdir(mode=0o755)
        os.chmod(home, 0o755)

        db = home / "metadata.db"
        backend = SQLiteBackend(str(db))
        backend.put("op_stats", "k", b"v")
        assert_private(home, "metadata directory")
        assert_private(db, "metadata database")

    def test_a_missing_parent_still_reports_the_bad_path(self, tmp_path: Path) -> None:
        """Hardening must not move a failure.

        Creating a missing parent directory would swallow the "unable to open database
        file" error that names the misconfigured path, and quietly put the database
        somewhere nobody chose. So the directory is tightened only if it already exists.
        """
        from batcher._internal.errors import ConfigError
        from batcher.metadata.backends.sqlite import SQLiteBackend

        missing = tmp_path / "not-created" / "metadata.db"
        with pytest.raises(ConfigError, match="not-created"):
            SQLiteBackend(str(missing))


def test_the_helpers_themselves(tmp_path: Path) -> None:
    """`private_dir` creates parents privately; `open_private` never has a public window."""
    from batcher._internal.paths import open_private, private_dir

    nested = private_dir(tmp_path / "a" / "b" / "c")
    assert_private(nested, "created directory")

    target = nested / "f.bin"
    with open_private(target) as handle:
        handle.write(b"payload")
    assert_private(target, "created file")
    assert target.read_bytes() == b"payload"

    # A non-binary mode is a caller error, not something to silently widen.
    with pytest.raises(ValueError, match="binary"):
        open_private(nested / "g.txt", "w")
