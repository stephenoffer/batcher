"""Directory/glob reads skip metadata/marker files (`_SUCCESS`, `.crc`, …).

Spark/Hive/Hadoop write `_SUCCESS`, `_metadata`, and `.crc` files alongside the
data; reading them as data is a documented Ray Data failure (ray#57704 / ray#61373).
Batcher follows the Spark convention: basenames starting with ``_`` or ``.`` are not
data, for both a directory read and an explicit glob.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher.io.filesystem import LocalFileSystem, _is_data_file


def test_is_data_file_predicate():
    assert _is_data_file("part-0.parquet")
    assert _is_data_file("/dir/data.parquet")
    assert not _is_data_file("_SUCCESS")
    assert not _is_data_file("/dir/_SUCCESS")
    assert not _is_data_file("/dir/_committed_123")
    assert not _is_data_file(".crc")
    assert not _is_data_file("/dir/.DS_Store")
    assert not _is_data_file("/dir/_part-0.parquet")  # Spark temp file


def _spark_style_dir(tmp_path) -> str:
    tbl = pa.table({"id": [1, 2, 3], "v": [10, 20, 30]})
    pq.write_table(tbl, f"{tmp_path}/part-00000.parquet")
    # Marker/metadata files Spark & friends leave behind:
    (tmp_path / "_SUCCESS").write_text("")
    (tmp_path / "_metadata").write_text("x")
    (tmp_path / ".part-00000.parquet.crc").write_bytes(b"\x00")
    return str(tmp_path)


def test_expand_directory_skips_markers(tmp_path):
    path = _spark_style_dir(tmp_path)
    files = LocalFileSystem().expand(path, suffix=".parquet")
    assert [f.rsplit("/", 1)[-1] for f in files] == ["part-00000.parquet"]


def test_expand_glob_skips_markers(tmp_path):
    path = _spark_style_dir(tmp_path)
    # An explicit `*` glob would otherwise pick up `_SUCCESS`/`_metadata`.
    files = LocalFileSystem().expand(f"{path}/*", suffix=".parquet")
    assert [f.rsplit("/", 1)[-1] for f in files] == ["part-00000.parquet"]


def test_read_parquet_directory_ignores_markers(tmp_path):
    path = _spark_style_dir(tmp_path)
    out = bt.read.parquet(path).collect()
    assert out.num_rows == 3
    assert set(out.column_names) == {"id", "v"}


def test_empty_after_marker_filter_errors(tmp_path):
    (tmp_path / "_SUCCESS").write_text("")
    from batcher._internal.errors import IOError as BatcherIOError

    with pytest.raises(BatcherIOError):
        LocalFileSystem().expand(str(tmp_path), suffix=".parquet")
