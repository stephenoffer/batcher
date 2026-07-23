"""Wave-3 IO format round-trip / projection regressions (arrow-ipc, ORC, msgpack).

Each test fails on the pre-fix behavior:

- An ORC scan honored the *set* of projected columns but not their *order*:
  ``read(projection=["c", "a"])`` returned ``a, c`` (the file's schema order),
  unlike Parquet/CSV/Arrow which preserve the requested order. A projection pushed
  to an ORC scan is an ordered list, so the columns came back transposed.
- A MessagePack read inferred a schema *per morsel* (16,384 rows). A file whose
  column is integer-looking in the first morsel and null (or a different type) in
  the tail produced batches with disagreeing schemas that failed to concatenate
  (``Table.from_batches`` raised "Schema ... different") — a crash on a reachable
  file, not a wrong-schema.
"""

from __future__ import annotations

import os
import struct
import tempfile

import pyarrow as pa
import pytest

from batcher.io.formats.base import SINKS, SOURCES

pytestmark = pytest.mark.unit


def _rt(fmt: str, table: pa.Table, suffix: str) -> pa.Table:
    path = tempfile.mktemp(suffix=suffix)
    try:
        SINKS.get(fmt)().write(table, path)
        src = SOURCES.get(fmt)(path)
        return pa.Table.from_batches(src.read(), schema=src.schema())
    finally:
        if os.path.exists(path):
            os.remove(path)


@pytest.mark.parametrize("fmt,suffix", [("arrow", ".arrow"), ("orc", ".orc")])
def test_roundtrip_preserves_nulls_unicode_nested(fmt: str, suffix: str) -> None:
    # Null, empty-string-vs-null, unicode, nested list/struct, all-null, bool, and
    # an int64 beyond 2**53 must survive a write→read round trip byte-for-byte.
    table = pa.table(
        {
            "i": pa.array([1, None, 2**53 + 1], pa.int64()),
            "s": pa.array(["", None, "日本語😀"], pa.string()),
            "b": pa.array([True, None, False], pa.bool_()),
            "n": pa.array([None, None, None], pa.int64()),
            "lst": pa.array([[1, 2], None, []], pa.list_(pa.int64())),
            "st": pa.array([{"x": 1}, None, {"x": None}], pa.struct([("x", pa.int64())])),
        }
    )
    out = _rt(fmt, table, suffix)
    assert out.schema.equals(table.schema), f"{out.schema}\nvs\n{table.schema}"
    assert out.equals(table)


def test_orc_projection_preserves_requested_column_order() -> None:
    # `read(projection=["c", "a"])` must yield columns in THAT order (c, a) — as
    # Parquet/CSV/Arrow do — not the ORC file's own schema order (a, c).
    path = tempfile.mktemp(suffix=".orc")
    try:
        SINKS.get("orc")().write(pa.table({"a": [1, 2], "b": [3, 4], "c": [5, 6]}), path)
        src = SOURCES.get("orc")(path)
        out = pa.Table.from_batches(src.read(projection=["c", "a"]))
        assert out.schema.names == ["c", "a"], out.schema.names
        assert out.column("c").to_pylist() == [5, 6]
        assert out.column("a").to_pylist() == [1, 2]
        # Every stripe split must honor the order too (the distributed read path).
        for split in src.splits(target_size=None):
            sub = pa.Table.from_batches(split.read(projection=["c", "a"]))
            assert sub.schema.names == ["c", "a"], sub.schema.names
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_msgpack_multibatch_null_tail_reads_without_schema_conflict() -> None:
    pytest.importorskip("ormsgpack")
    import ormsgpack

    length = struct.Struct(">I")
    path = tempfile.mktemp(suffix=".msgpack")
    n_head = 16_384  # exactly one morsel of integers ...
    try:
        with open(path, "wb") as fh:
            for i in range(n_head):
                payload = ormsgpack.packb({"a": i})
                fh.write(length.pack(len(payload)))
                fh.write(payload)
            for _ in range(3):  # ... then an all-null tail in the next morsel.
                payload = ormsgpack.packb({"a": None})
                fh.write(length.pack(len(payload)))
                fh.write(payload)
        src = SOURCES.get("msgpack")(path)
        table = pa.Table.from_batches(src.read())  # must NOT raise a schema conflict
        assert table.num_rows == n_head + 3
        assert pa.types.is_integer(table.schema.field("a").type)
        assert table.column("a").to_pylist()[-3:] == [None, None, None]
    finally:
        if os.path.exists(path):
            os.remove(path)
