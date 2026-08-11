"""`FileSystem.remove_empty_dir` — the one delete that must never take data with it.

It exists so a re-partitioning rewrite can drop the directories it abandoned. On an
object store a "directory" is a key prefix, so deleting one deletes every object under
it; the emptiness check is therefore the whole safety argument and is what these pin.
"""

from __future__ import annotations

import pytest

from batcher.io.filesystem import resolve_filesystem

pytestmark = pytest.mark.integration


def test_an_empty_directory_is_removed(tmp_path):
    target = tmp_path / "gone"
    target.mkdir()
    fs = resolve_filesystem(str(tmp_path))
    assert fs.remove_empty_dir(str(target)) is True
    assert not target.exists()


def test_a_directory_holding_a_file_is_left_alone(tmp_path):
    target = tmp_path / "kept"
    target.mkdir()
    (target / "data.parquet").write_bytes(b"x")
    fs = resolve_filesystem(str(tmp_path))
    assert fs.remove_empty_dir(str(target)) is False
    assert (target / "data.parquet").exists()


def test_a_directory_holding_only_a_subdirectory_is_left_alone(tmp_path):
    target = tmp_path / "outer"
    (target / "inner").mkdir(parents=True)
    fs = resolve_filesystem(str(tmp_path))
    assert fs.remove_empty_dir(str(target)) is False
    assert (target / "inner").exists()


def test_a_nested_file_still_protects_the_ancestor(tmp_path):
    target = tmp_path / "outer"
    (target / "inner").mkdir(parents=True)
    (target / "inner" / "data.parquet").write_bytes(b"x")
    fs = resolve_filesystem(str(tmp_path))
    assert fs.remove_empty_dir(str(target)) is False
    assert (target / "inner" / "data.parquet").exists()


def test_a_file_path_is_not_a_directory_and_is_untouched(tmp_path):
    path = tmp_path / "a.parquet"
    path.write_bytes(b"x")
    fs = resolve_filesystem(str(tmp_path))
    assert fs.remove_empty_dir(str(path)) is False
    assert path.exists()


def test_an_absent_path_is_a_no_op(tmp_path):
    fs = resolve_filesystem(str(tmp_path))
    assert fs.remove_empty_dir(str(tmp_path / "never-existed")) is False
