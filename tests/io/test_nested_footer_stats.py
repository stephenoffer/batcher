"""A nested Parquet column's footer bounds must not be filed under a bare field name.

Parquet stores one column chunk per *leaf*. `ParquetSchema.names` reports each leaf's bare
field name; `ColumnChunkMetaData.path_in_schema` reports its full dotted path. For a flat
table the two agree, which is why this held for so long — and for a nested one they do not,
so a struct field's bounds merged into whatever top-level column happened to share its name.

Those bounds carry `Provenance.EXACT` for a numeric column, and that provenance is what lets
Kyber answer `max(a)` from metadata without reading the data. So the collision is a wrong
answer, not a loose estimate.
"""

from __future__ import annotations

from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher.io.stats.columnar_footer as cf

pytestmark = pytest.mark.io


class _LocalFS:
    """The minimal filesystem `parquet_statistics` needs: an `open` returning a handle."""

    def open(self, path):
        return open(path, "rb")


def _colliding_table() -> pa.Table:
    """A top-level `a` of 1..3 beside a struct `s` whose field is also named `a`, 1000..3000."""
    return pa.table(
        {
            "a": pa.array([1, 2, 3]),
            "s": pa.array([{"a": 1000}, {"a": 2000}, {"a": 3000}]),
        }
    )


def _python_path_stats(path: str, table: pa.Table):
    """Footer statistics via the Python accumulator, with the native walk declined.

    The native path is not a substitute for testing this one: it is the *fallback* that runs
    whenever the Rust reader declines — an fsspec backend, a read-through byte cache, a
    declared `sorting_columns`, or any native failure — so it must be correct on its own.
    """
    with mock.patch.object(cf, "_native_statistics", lambda *a, **k: None):
        return cf.parquet_statistics(_LocalFS(), [path], table.schema)


def test_a_struct_field_does_not_pollute_a_top_level_column(tmp_path):
    table = _colliding_table()
    path = str(tmp_path / "c.parquet")
    pq.write_table(table, path)
    stats = _python_path_stats(path, table)
    assert stats is not None
    top = stats.columns["a"]
    # The true range of the top-level column, not the union with the struct field's.
    assert (top.min, top.max) == (1, 3)


def test_the_struct_field_is_filed_under_its_path(tmp_path):
    table = _colliding_table()
    path = str(tmp_path / "c.parquet")
    pq.write_table(table, path)
    stats = _python_path_stats(path, table)
    assert "s.a" in stats.columns
    assert (stats.columns["s.a"].min, stats.columns["s.a"].max) == (1000, 3000)


def test_the_two_footer_paths_agree(tmp_path):
    # One statistic with two implementations is one chance for them to disagree, and they
    # did: the native walk keys by the path and reported [1, 3] for the file the Python walk
    # reported [1, 3000] for.
    table = _colliding_table()
    path = str(tmp_path / "c.parquet")
    pq.write_table(table, path)
    native = cf.parquet_statistics(_LocalFS(), [path], table.schema)
    python = _python_path_stats(path, table)
    assert native is not None and python is not None
    assert (native.columns["a"].min, native.columns["a"].max) == (
        python.columns["a"].min,
        python.columns["a"].max,
    )


def test_a_flat_table_is_completely_unaffected(tmp_path):
    # The safety property: for a flat schema `path_in_schema` and the bare name are the same
    # string, so every existing statistic is byte-for-byte what it was.
    table = pa.table({"a": pa.array([1, 2, 3]), "b": pa.array([10.0, 20.0, 30.0])})
    path = str(tmp_path / "flat.parquet")
    pq.write_table(table, path)
    stats = _python_path_stats(path, table)
    assert sorted(stats.columns) == ["a", "b"]
    assert (stats.columns["a"].min, stats.columns["a"].max) == (1, 3)
    assert (stats.columns["b"].min, stats.columns["b"].max) == (10.0, 30.0)


def test_a_list_column_does_not_claim_a_top_level_name(tmp_path):
    # The same hazard through the other nesting: a `list<int64>`'s leaf is `l.list.element`,
    # whose bare name would be `element` -- harmless here only by luck of naming.
    table = pa.table({"element": pa.array([1, 2, 3]), "l": pa.array([[100], [200], [300]])})
    path = str(tmp_path / "list.parquet")
    pq.write_table(table, path)
    stats = _python_path_stats(path, table)
    assert (stats.columns["element"].min, stats.columns["element"].max) == (1, 3)
