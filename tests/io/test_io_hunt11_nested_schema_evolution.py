"""Nested-type multi-file schema-evolution reconciliation defects.

The reconciliation lattice (`io.schema.evolution`) widened *flat* columns across
files (``int32`` in older files, ``int64`` in newer) but treated any **nested** type
difference as an irreconcilable conflict — raising ``SchemaError`` even when a clean,
lossless common type existed:

- a list column whose element width grew (``list<int32>`` → ``list<int64>``),
- a struct column that gained a field (``struct<a>`` → ``struct<a, b>``),
- a struct whose inner field widened (``struct<a: int32>`` → ``struct<a: int64>``).

These are the routine "a nested field was added / its element width grew across
files" evolution shapes, exactly analogous to the flat widening the lattice already
performed. Each now unifies to the lossless supertype and matches DuckDB's
``union_by_name`` read; a genuine conflict (int vs string) still raises.
"""

from __future__ import annotations

import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher._internal.errors import SchemaError
from batcher.io.schema import unify_schemas

pytestmark = pytest.mark.differential


def _write(tables: list[pa.Table]) -> str:
    d = tempfile.mkdtemp()
    for i, t in enumerate(tables):
        pq.write_table(t, os.path.join(d, f"f{i}.parquet"))
    return d


def test_unify_list_element_width_widens() -> None:
    """``list<int32>`` unioned with ``list<int64>`` must yield ``list<int64>``.

    Fails before the fix: `unify_schemas` raised ``SchemaError`` on the nested
    difference instead of recursing into the element type.
    """
    unified = unify_schemas(
        [
            pa.schema([pa.field("emb", pa.list_(pa.int32()))]),
            pa.schema([pa.field("emb", pa.list_(pa.int64()))]),
        ],
        mode="union",
    )
    assert pa.types.is_list(unified.field("emb").type)
    assert unified.field("emb").type.value_type.equals(pa.int64())


def test_unify_struct_gains_field() -> None:
    """``struct<a>`` unioned with ``struct<a, b>`` must yield ``struct<a, b>``."""
    unified = unify_schemas(
        [
            pa.schema([pa.field("m", pa.struct([("a", pa.int64())]))]),
            pa.schema([pa.field("m", pa.struct([("a", pa.int64()), ("b", pa.int64())]))]),
        ],
        mode="union",
    )
    names = [f.name for f in unified.field("m").type]
    assert names == ["a", "b"]


def test_read_list_widen_matches_duckdb() -> None:
    """A directory of parquet with ``list<int32>`` then ``list<int64>`` reads losslessly."""
    d = _write(
        [
            pa.table({"emb": pa.array([[1, 2]], pa.list_(pa.int32()))}),
            pa.table({"emb": pa.array([[2**40, 4]], pa.list_(pa.int64()))}),
        ]
    )
    got = bt.read(d, format="parquet", schema_mode="union").to_pydict()["emb"]

    duckdb = pytest.importorskip("duckdb")
    rows = duckdb.sql(
        f"SELECT emb FROM read_parquet('{d}/*.parquet', union_by_name=true)"
    ).fetchall()
    assert sorted(got) == sorted(r[0] for r in rows)
    assert [2**40, 4] in got  # the widened element survives exactly


def test_read_struct_evolution_matches_duckdb() -> None:
    """A struct that gains a field reads with the absent field as null, like DuckDB."""
    d = _write(
        [
            pa.table({"m": pa.array([{"a": 1}], pa.struct([("a", pa.int32())]))}),
            pa.table(
                {
                    "m": pa.array(
                        [{"a": 2, "b": 3}], pa.struct([("a", pa.int64()), ("b", pa.int64())])
                    )
                }
            ),
        ]
    )
    got = bt.read(d, format="parquet", schema_mode="union").to_pydict()["m"]

    duckdb = pytest.importorskip("duckdb")
    rows = duckdb.sql(f"SELECT m FROM read_parquet('{d}/*.parquet', union_by_name=true)").fetchall()
    assert sorted(got, key=lambda r: r["a"]) == sorted((r[0] for r in rows), key=lambda r: r["a"])
    assert {"a": 1, "b": None} in got  # absent field read as null


def test_genuine_type_conflict_still_raises() -> None:
    """A real int/string collision has no non-lossy common type and must still raise."""
    with pytest.raises(SchemaError):
        unify_schemas(
            [
                pa.schema([pa.field("a", pa.int64())]),
                pa.schema([pa.field("a", pa.string())]),
            ],
            mode="union",
        )


def test_read_string_and_large_string_unify() -> None:
    """``string`` and ``large_string`` are one logical type — a union read must not raise.

    Fails before the fix: `unify_schemas` treated the offset-width difference as a conflict
    and raised, though DuckDB reads both as VARCHAR.
    """
    d = _write(
        [
            pa.table({"s": pa.array(["a", "b"], pa.string())}),
            pa.table({"s": pa.array(["c"], pa.large_string())}),
        ]
    )
    got = bt.read(d, format="parquet", schema_mode="union").to_pydict()["s"]
    assert sorted(got) == ["a", "b", "c"]


def test_read_dictionary_and_plain_string_unify() -> None:
    """A dict-encoded ``string`` file and a plain ``string`` file read as one column.

    A `dictionary<string>` is an encoding of `string`, not a distinct type; mixing dict and
    plain parquet pages across files is routine and must decode to `string`, not raise.
    """
    d = _write(
        [
            pa.table({"c": pa.array(["x", "y"]).dictionary_encode()}),
            pa.table({"c": pa.array(["z"], pa.string())}),
        ]
    )
    got = bt.read(d, format="parquet", schema_mode="union").to_pydict()["c"]
    assert sorted(got) == ["x", "y", "z"]


def test_read_timestamp_unit_widens_to_finer() -> None:
    """``timestamp[ms]`` and ``timestamp[us]`` unify to the finer unit (matching DuckDB)."""
    import datetime as _dt

    d = _write(
        [
            pa.table({"t": pa.array([_dt.datetime(2021, 1, 1)], pa.timestamp("ms"))}),
            pa.table({"t": pa.array([_dt.datetime(2021, 1, 2)], pa.timestamp("us"))}),
        ]
    )
    got = bt.read(d, format="parquet", schema_mode="union").to_pydict()["t"]
    assert sorted(got) == [_dt.datetime(2021, 1, 1), _dt.datetime(2021, 1, 2)]
