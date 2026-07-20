"""The native per-file manifest must agree with the Python footer walk, exactly.

`parquet_file_manifest` gained a Rust fast path. Its output decides which files a
copy-on-write ``MERGE`` opens at all (`io.stats.key_pruning`), so a disagreement between the
two paths is not a slower merge — it is a merge that skips a file holding rows it was
supposed to update, on whichever path a given backend happens to take.

Every test asserts *equivalence* rather than expected values, and the null-bound cases get
particular attention: a NULL bound means "unknown, keep this file", and the one way to turn
this optimization into data loss is to let a null be read as "no match".
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batcher.io.filesystem import resolve_filesystem
from batcher.io.stats import parquet_manifest
from batcher.io.stats.parquet_manifest import parquet_file_manifest


def _python_manifest(fs, files, columns):
    """`parquet_file_manifest` with the native fast path disabled."""
    original = parquet_manifest._native_manifest
    parquet_manifest._native_manifest = lambda *a, **k: None
    try:
        return parquet_file_manifest(fs, files, columns)
    finally:
        parquet_manifest._native_manifest = original


def _write(tmp_path, tables, row_group_size=2):
    paths = []
    for i, table in enumerate(tables):
        p = tmp_path / f"part-{i}.parquet"
        pq.write_table(table, p, row_group_size=row_group_size)
        paths.append(str(p))
    return resolve_filesystem(paths[0]), paths


def _assert_equivalent(fs, paths, columns):
    native = parquet_file_manifest(fs, paths, columns)
    python = _python_manifest(fs, paths, columns)
    assert native is not None and python is not None

    assert native.num_rows == python.num_rows == len(paths)
    assert set(native.column_names) == set(python.column_names)
    for name in python.column_names:
        got, want = native.column(name).to_pylist(), python.column(name).to_pylist()
        assert got == want, f"column {name!r}: native {got} vs python {want}"


def test_per_file_bounds_are_not_collapsed_across_the_dataset(tmp_path):
    """Each row describes its own file — the distinction the whole module exists for."""
    a = pa.table({"k": pa.array([1, 2, 3, 4], pa.int64())})
    b = pa.table({"k": pa.array([900, 950], pa.int64())})
    fs, paths = _write(tmp_path, [a, b])
    _assert_equivalent(fs, paths, ["k"])

    m = parquet_file_manifest(fs, paths, ["k"])
    assert m.column("min.k").to_pylist() == [1, 900]
    assert m.column("max.k").to_pylist() == [4, 950]
    assert m.column("num_records").to_pylist() == [4, 2]


def test_row_order_matches_the_file_order_given(tmp_path):
    """A misaligned row would attribute one file's bounds to another's path."""
    tables = [pa.table({"k": pa.array([i * 100, i * 100 + 1], pa.int64())}) for i in range(5)]
    fs, paths = _write(tmp_path, tables)
    m = parquet_file_manifest(fs, paths, ["k"])
    assert m.column("path").to_pylist() == paths
    assert m.column("min.k").to_pylist() == [0, 100, 200, 300, 400]


def test_nulls_in_the_key_column(tmp_path):
    t = pa.table({"k": pa.array([1, None, 5, None], pa.int64())})
    fs, paths = _write(tmp_path, [t])
    _assert_equivalent(fs, paths, ["k"])


def test_all_null_key_column(tmp_path):
    t = pa.table({"k": pa.array([None, None], pa.int64())})
    fs, paths = _write(tmp_path, [t])
    _assert_equivalent(fs, paths, ["k"])


def test_string_and_multiple_key_columns(tmp_path):
    t = pa.table(
        {
            "k": pa.array(["pear", "apple", "quince", "fig"], pa.string()),
            "j": pa.array([4, 1, 9, 2], pa.int64()),
        }
    )
    fs, paths = _write(tmp_path, [t])
    _assert_equivalent(fs, paths, ["k", "j"])


def test_unreadable_file_keeps_its_row_with_null_bounds(tmp_path):
    """The failure mode that matters: a dropped row shifts every later file's bounds.

    A NULL bound must mean *unknown* — the consumer keeps the file — so a merge cannot
    silently skip a file whose footer it could not read.
    """
    fs, paths = _write(tmp_path, [pa.table({"k": pa.array([1, 2], pa.int64())})])
    with_missing = [str(tmp_path / "absent.parquet"), *paths]
    m = parquet_file_manifest(fs, with_missing, ["k"])
    assert m is not None
    assert m.num_rows == 2
    assert m.column("path").to_pylist() == with_missing
    assert m.column("min.k").to_pylist()[0] is None
    assert m.column("num_records").to_pylist()[0] is None
    # The readable file is still described exactly.
    assert m.column("min.k").to_pylist()[1] == 1


def test_column_absent_from_every_file(tmp_path):
    """An unknown column yields nulls, never a wrong bound and never an exception."""
    fs, paths = _write(tmp_path, [pa.table({"k": pa.array([1, 2], pa.int64())})])
    m = parquet_file_manifest(fs, paths, ["nope"])
    assert m is not None
    assert all(v is None for v in m.column("null_count.nope").to_pylist())


def test_column_present_in_only_some_files(tmp_path):
    a = pa.table({"k": pa.array([1, 2], pa.int64())})
    b = pa.table({"k": pa.array([8, 9], pa.int64()), "extra": pa.array([3, 4], pa.int64())})
    fs, paths = _write(tmp_path, [a, b])
    _assert_equivalent(fs, paths, ["k"])


@pytest.mark.parametrize("row_group_size", [1, 3, 1000])
def test_row_group_size_does_not_change_the_manifest(tmp_path, row_group_size):
    """Per-file bounds are per *file*, so the writer's chunking must not be observable."""
    t = pa.table({"k": pa.array([7, 2, 9, 4, 11, 1], pa.int64())})
    fs, paths = _write(tmp_path, [t], row_group_size=row_group_size)
    _assert_equivalent(fs, paths, ["k"])
    assert parquet_file_manifest(fs, paths, ["k"]).column("min.k").to_pylist() == [1]


def test_empty_file_list(tmp_path):
    fs = resolve_filesystem(str(tmp_path))
    assert parquet_file_manifest(fs, [], ["k"]) is None
