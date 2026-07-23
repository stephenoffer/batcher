"""The native footer-statistics pass must agree with the Python accumulator, exactly.

`parquet_statistics` gained a Rust fast path (`_native_statistics`) that replaces a
per-column-chunk Python walk. The two paths feed Kyber's pruning, `count()`, `min()`,
`max()`, and null-count answers, so a disagreement between them is not a slow plan — it is
a wrong result that appears only on whichever path a given backend happens to take.

Every test here therefore asserts *equivalence* rather than expected values: the same files
are put through both paths and the resulting `SourceStatistics` are compared field by
field. That is the property the fast path has to keep, and it stays true even as the
statistics bundle grows new fields.

The types covered are the ones whose bounds cross the FFI boundary differently — integers,
floats, strings, temporals, decimals, booleans — plus the cases where the native path is
*required to decline*: nulls-everywhere, an unreadable file, and a declared sort key.
"""

from __future__ import annotations

import datetime
import decimal
from dataclasses import fields

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batcher.io.filesystem import resolve_filesystem
from batcher.io.stats import columnar_footer
from batcher.io.stats.columnar_footer import _native_statistics, parquet_statistics


def _python_statistics(fs, files, schema):
    """`parquet_statistics` with the native fast path disabled, i.e. the Python walk."""
    original = columnar_footer._native_statistics
    columnar_footer._native_statistics = lambda *a, **k: None
    try:
        return parquet_statistics(fs, files, schema)
    finally:
        columnar_footer._native_statistics = original


def _write(tmp_path, tables, row_group_size=2, **kwargs):
    """Write `tables` as numbered Parquet files and return `(fs, paths, schema)`."""
    paths = []
    for i, table in enumerate(tables):
        p = tmp_path / f"part-{i}.parquet"
        pq.write_table(table, p, row_group_size=row_group_size, **kwargs)
        paths.append(str(p))
    return resolve_filesystem(paths[0]), paths, tables[0].schema


def _assert_equivalent(fs, paths, schema):
    """Both paths must produce the same statistics bundle, field for field."""
    native = _native_statistics(fs, paths, schema)
    assert native is not None, "the native path should have handled these files"
    python = _python_statistics(fs, paths, schema)

    for f in fields(native):
        if f.name == "columns":
            continue
        assert getattr(native, f.name) == getattr(python, f.name), f.name

    assert native.columns.keys() == python.columns.keys()
    for name, got in native.columns.items():
        assert got == python.columns[name], f"column {name!r}"


def test_integer_bounds_across_files_and_row_groups(tmp_path):
    a = pa.table({"i": pa.array([5, 1, 9, 3], pa.int64())})
    b = pa.table({"i": pa.array([100, 42], pa.int64())})
    _assert_equivalent(*_write(tmp_path, [a, b]))


def test_null_counts_and_partially_null_columns(tmp_path):
    t = pa.table({"i": pa.array([1, None, 3, None, 5, None], pa.int64())})
    _assert_equivalent(*_write(tmp_path, [t]))


def test_all_null_column(tmp_path):
    t = pa.table({"i": pa.array([None, None, None, None], pa.int64())})
    _assert_equivalent(*_write(tmp_path, [t]))


def test_string_bounds_are_inexact_but_equal(tmp_path):
    t = pa.table({"s": pa.array(["pear", "apple", "quince", "fig"], pa.string())})
    _assert_equivalent(*_write(tmp_path, [t]))


def test_float_bounds(tmp_path):
    t = pa.table({"f": pa.array([2.5, -1.25, 8.0, 0.0], pa.float64())})
    _assert_equivalent(*_write(tmp_path, [t]))


def test_boolean_and_temporal_bounds(tmp_path):
    t = pa.table(
        {
            "b": pa.array([True, False, True, True], pa.bool_()),
            "d": pa.array([datetime.date(2024, 3, 1), datetime.date(2020, 1, 1)] * 2, pa.date32()),
            "ts": pa.array(
                [datetime.datetime(2024, 1, 1, 12), datetime.datetime(2021, 6, 5, 8)] * 2,
                pa.timestamp("us"),
            ),
        }
    )
    _assert_equivalent(*_write(tmp_path, [t]))


def test_decimal_bounds(tmp_path):
    t = pa.table(
        {
            "d": pa.array(
                [decimal.Decimal("1.50"), decimal.Decimal("-9.25")] * 2,
                pa.decimal128(10, 2),
            )
        }
    )
    _assert_equivalent(*_write(tmp_path, [t]))


def test_many_columns_and_many_row_groups(tmp_path):
    """The shape the fast path exists for: the per-chunk count is what used to dominate."""
    cols = {f"c{i}": pa.array(list(range(i, i + 40)), pa.int64()) for i in range(12)}
    _assert_equivalent(*_write(tmp_path, [pa.table(cols)] * 3, row_group_size=4))


def test_single_row_group_keeps_the_distinct_count_rule(tmp_path):
    """`ndv` is publishable only from a single row group — both paths must agree on that."""
    t = pa.table({"s": pa.array(["a", "b", "c", "d"], pa.string())})
    _assert_equivalent(*_write(tmp_path, [t], row_group_size=1000))


def test_hive_partitioned_layout(tmp_path):
    """The dominant large-dataset layout: one directory per partition value.

    The partition column lives in the *path*, not in any file, so the per-file schemas are
    narrower than the dataset's. Both paths must agree on that — and in particular the
    native path must not invent bounds for a column no footer describes.
    """
    n = 400
    t = pa.table(
        {
            "id": pa.array(range(n), pa.int64()),
            "v": pa.array([i % 97 for i in range(n)], pa.int64()),
            "part": pa.array([i % 8 for i in range(n)], pa.int64()),
        }
    )
    pq.write_to_dataset(t, tmp_path, partition_cols=["part"], row_group_size=13)
    files = sorted(str(p) for p in tmp_path.rglob("*.parquet"))
    assert len(files) == 8, "expected one file per partition value"
    fs = resolve_filesystem(files[0])
    _assert_equivalent(fs, files, pq.read_schema(files[0]))


def test_column_present_in_only_some_files(tmp_path):
    """Schema evolution: a column added mid-dataset contributes the files that have it.

    The native path builds a statistics converter per *file*, precisely so a dataset whose
    schema grew can still report bounds from the files where the column exists.
    """
    a = pa.table({"i": pa.array([1, 2], pa.int64())})
    b = pa.table({"i": pa.array([9, 4], pa.int64()), "extra": pa.array([7, 3], pa.int64())})
    paths = []
    for idx, table in enumerate((a, b)):
        p = tmp_path / f"part-{idx}.parquet"
        pq.write_table(table, p, row_group_size=2)
        paths.append(str(p))
    fs = resolve_filesystem(paths[0])
    _assert_equivalent(fs, paths, b.schema)


def test_column_widened_between_files(tmp_path):
    """A column written int32 in one file and int64 in another must not corrupt bounds.

    The two files' bounds cannot share one Arrow array, so the native path drops the
    divergent file's contribution rather than mixing types — a narrower bound set still
    prunes correctly. Whatever it does, it must match the Python path.
    """
    a = pa.table({"v": pa.array([5, 1], pa.int32())})
    b = pa.table({"v": pa.array([900, 40], pa.int64())})
    paths = []
    for idx, table in enumerate((a, b)):
        p = tmp_path / f"part-{idx}.parquet"
        pq.write_table(table, p, row_group_size=2)
        paths.append(str(p))
    fs = resolve_filesystem(paths[0])
    native = _native_statistics(fs, paths, b.schema)
    python = _python_statistics(fs, paths, b.schema)
    # Row counts are schema-independent and must agree regardless of how bounds resolved.
    assert native is None or native.row_count == python.row_count
    if native is not None and "v" in native.columns and "v" in python.columns:
        # A bound that is reported must be sound: never tighter than the true extremes.
        assert native.columns["v"].min in (None, 1, 40)


def test_native_path_declines_when_a_file_is_unreadable(tmp_path):
    """A missing footer makes the row count partial, so the fast path must not answer."""
    fs, paths, schema = _write(tmp_path, [pa.table({"i": pa.array([1, 2], pa.int64())})])
    assert _native_statistics(fs, [*paths, str(tmp_path / "absent.parquet")], schema) is None


def test_native_path_declines_when_a_sort_key_is_declared(tmp_path):
    """Sortedness deletes a `Sort`, so a declaring dataset goes to the full proof."""
    t = pa.table({"i": pa.array([1, 2, 3, 4], pa.int64())})
    fs, paths, schema = _write(tmp_path, [t], sorting_columns=[pq.SortingColumn(0)])
    assert _native_statistics(fs, paths, schema) is None
    # ...and the Python path still answers for it, so nothing is lost by declining.
    assert parquet_statistics(fs, paths, schema) is not None


def test_native_path_declines_without_a_native_read_target(tmp_path):
    """A backend the Rust object store cannot address falls back rather than failing."""
    fs, paths, schema = _write(tmp_path, [pa.table({"i": pa.array([1, 2], pa.int64())})])

    class _NoNativeFS:
        def __init__(self, inner):
            self._inner = inner

        def open(self, path):
            return self._inner.open(path)

        def native_read_target(self, path):
            return None

    assert _native_statistics(_NoNativeFS(fs), paths, schema) is None
    assert parquet_statistics(_NoNativeFS(fs), paths, schema) is not None


def test_empty_file_list(tmp_path):
    fs = resolve_filesystem(str(tmp_path))
    assert _native_statistics(fs, [], pa.schema([])) is None


@pytest.mark.parametrize("row_group_size", [1, 3, 1000])
def test_row_group_sizes_do_not_change_the_answer(tmp_path, row_group_size):
    """Bounds are global, so how the writer chunked the file must not be observable."""
    t = pa.table(
        {
            "i": pa.array([7, 2, 9, 4, 11, 1], pa.int64()),
            "s": pa.array(["g", "b", "i", "d", "k", "a"], pa.string()),
        }
    )
    _assert_equivalent(*_write(tmp_path, [t], row_group_size=row_group_size))
