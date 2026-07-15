"""Bug-hunt (wave 4): nested variant-type normalization for lakehouse reads.

`normalize_engine_types` exists because pyiceberg maps Iceberg `StringType` to Arrow
`large_string` and `ListType` to `large_list`, which the engine's kernels reject. It used
to normalize only *top-level* columns, so a struct's string field, a list's element, or a
map's value came back still in its 64-bit form and the same crash reappeared the moment a
query touched it:

    s.field("name") == "x"   -> Invalid comparison operation: LargeUtf8 == Utf8
    tags.list.contains("a")  -> expected a Utf8 argument, got LargeList(... LargeUtf8)

These pin the recursive normalization: a query over a nested string/list column of an
ordinary Iceberg table must run, not crash.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg", reason="pyiceberg not installed")

import batcher as bt
from batcher.io.formats.lakehouse._arrow import engine_schema


def _catalog_spec(tmp_path):
    from pyiceberg.catalog.sql import SqlCatalog

    warehouse = tmp_path / "wh"
    warehouse.mkdir()
    spec = {
        "type": "sql",
        "uri": f"sqlite:///{warehouse}/catalog.db",
        "warehouse": f"file://{warehouse}",
    }
    catalog = SqlCatalog("default", uri=spec["uri"], warehouse=spec["warehouse"])
    catalog.create_namespace("db")
    return catalog, spec


def test_engine_schema_normalizes_nested_variant_types() -> None:
    """`large_string`/`large_list` nested inside struct/list/map fold to the plain types."""
    schema = pa.schema(
        [
            ("a", pa.large_string()),
            ("l", pa.large_list(pa.field("element", pa.large_string(), nullable=False))),
            ("st", pa.struct([pa.field("name", pa.large_string())])),
            ("m", pa.map_(pa.large_string(), pa.large_string())),
        ]
    )
    out = engine_schema(schema)
    assert out.field("a").type == pa.string()
    assert out.field("l").type == pa.list_(pa.field("element", pa.string(), nullable=False))
    assert out.field("st").type == pa.struct([pa.field("name", pa.string())])
    assert out.field("m").type == pa.map_(pa.string(), pa.string())


def test_a_filter_on_a_nested_struct_string_field_does_not_crash(tmp_path) -> None:
    catalog, spec = _catalog_spec(tmp_path)
    schema = pa.schema(
        [pa.field("id", pa.int64()), pa.field("s", pa.struct([pa.field("name", pa.string())]))]
    )
    table = catalog.create_table("db.st", schema=schema)
    table.append(
        pa.table(
            {"id": [1, 2], "s": [{"name": "x"}, {"name": "y"}]}, schema=table.schema().as_arrow()
        )
    )

    got = (
        bt.read.table("iceberg", "db.st", catalog=spec)
        .filter(bt.col("s").struct.field("name") == "x")
        .collect()
    )
    assert got.to_pylist() == [{"id": 1, "s": {"name": "x"}}]


def test_list_contains_on_a_nested_string_list_does_not_crash(tmp_path) -> None:
    catalog, spec = _catalog_spec(tmp_path)
    schema = pa.schema([pa.field("id", pa.int64()), pa.field("tags", pa.list_(pa.string()))])
    table = catalog.create_table("db.lt", schema=schema)
    table.append(
        pa.table({"id": [1, 2], "tags": [["a", "b"], ["c"]]}, schema=table.schema().as_arrow())
    )

    got = (
        bt.read.table("iceberg", "db.lt", catalog=spec)
        .with_columns(bt.col("tags").list.contains("a").alias("has_a"))
        .collect()
    )
    rows = sorted(got.to_pylist(), key=lambda r: r["id"])
    assert [r["has_a"] for r in rows] == [True, False]
