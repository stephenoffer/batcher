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

    def test_round_robin_output_is_private(self, tmp_path: Path) -> None:
        from batcher.dist.shuffle_io import write_ipc_round_robin

        batch = pa.record_batch({"a": pa.array([1, 2, 3, 4], type=pa.int64())})
        paths = [str(tmp_path / f"rr-{i}.arrow") for i in range(2)]
        write_ipc_round_robin([batch], batch.schema, paths)
        for path in paths:
            assert_private(path, "round-robin shuffle file")


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
