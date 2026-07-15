"""Wave-15 multi-file schema-evolution reconciliation defects.

The reconciliation path (`io.schema.evolution`) unifies the schemas of files with
differing shapes into one and normalizes each file's batches to it. The bug pinned
here is a crash on a reachable read:

- A column stored as ``int64`` in older files and ``float64`` in newer ones is
  promoted by the lattice to ``float64`` (its one deliberately-lossy widening). But
  `normalize_batch` cast the int column with a *safe* cast, which rejects any int64
  above 2^53 that float64 cannot hold exactly — so a ``schema_mode="union"`` read of
  such a dataset raised ``ArrowInvalid`` instead of coercing. DuckDB coerces the same
  union to ``double`` (large ints rounded to the nearest float); Batcher now matches.
"""

from __future__ import annotations

import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher.io.schema import normalize_batch, unify_schemas

pytestmark = pytest.mark.differential


def _write(tables: list[pa.Table]) -> str:
    d = tempfile.mkdtemp()
    for i, t in enumerate(tables):
        pq.write_table(t, os.path.join(d, f"f{i}.parquet"))
    return d


def test_union_int64_float64_large_value_coerces_not_crashes() -> None:
    """int64(>2^53) file + float64 file, unioned, must coerce to double, not raise.

    Fails before the fix with ``ArrowInvalid: Integer value ... not in range`` from the
    safe cast inside `normalize_batch`.
    """
    big = 2**60 + 1  # > 2^53, not exactly representable in float64
    d = _write(
        [
            pa.table({"a": pa.array([big, 5], pa.int64())}),
            pa.table({"a": pa.array([1.5], pa.float64())}),
        ]
    )
    got = bt.read(d, format="parquet", schema_mode="union").to_pydict()["a"]

    # Oracle: DuckDB coerces the union to DOUBLE and rounds the big int to the
    # nearest float64 — the same value pyarrow's unsafe cast produces.
    duckdb = pytest.importorskip("duckdb")
    rows = duckdb.sql(
        f"SELECT a FROM read_parquet('{d}/*.parquet', union_by_name=true) ORDER BY a"
    ).fetchall()
    assert sorted(got) == sorted(r[0] for r in rows)
    assert float(big) in got  # rounded, not dropped or nulled


def test_normalize_batch_int_to_float_is_unsafe_lossy() -> None:
    """`normalize_batch` promotes a large int column to float64 by coercion (unit)."""
    big = 2**60 + 1
    batch = pa.RecordBatch.from_arrays([pa.array([big], pa.int64())], names=["a"])
    target = pa.schema([pa.field("a", pa.float64())])
    out = normalize_batch(batch, target)
    assert out.column("a").type == pa.float64()
    assert out.column("a")[0].as_py() == float(big)


def test_union_reorder_merges_by_name_not_position() -> None:
    """[a,b] + [b,a] must merge BY NAME — a position merge silently swaps values."""
    d = _write(
        [
            pa.table({"a": pa.array([1, 2], pa.int64()), "b": pa.array([101, 102], pa.int64())}),
            pa.table({"b": pa.array([201], pa.int64()), "a": pa.array([3], pa.int64())}),
        ]
    )
    got = bt.read(d, format="parquet", schema_mode="union").to_pydict()
    assert sorted(got["a"]) == [1, 2, 3]
    assert sorted(got["b"]) == [101, 102, 201]


def test_unify_and_normalize_added_dropped_column() -> None:
    """Added/dropped column across files: missing cells become typed NULLs (unit)."""
    s1 = pa.schema([("a", pa.int64()), ("b", pa.int64())])
    s2 = pa.schema([("a", pa.int64()), ("b", pa.int64()), ("c", pa.int64())])
    unified = unify_schemas([s1, s2], "union")
    assert unified.names == ["a", "b", "c"]
    b1 = pa.RecordBatch.from_arrays(
        [pa.array([1], pa.int64()), pa.array([2], pa.int64())], names=["a", "b"]
    )
    out = normalize_batch(b1, unified)
    assert out.column("c").to_pylist() == [None]
