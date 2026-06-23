"""`Dataset.explode` (SQL ``UNNEST``) vs DuckDB.

Explode a list/array column into one row per element; null and empty lists drop
the row (DuckDB ``UNNEST`` semantics). Covers ints, the RAG string-chunk fan-out,
in-place (default alias), explode-then-filter (predicate stays above the explode),
and the typed error for an unknown column. The exploded column is preserved across
the kyber projection-pushdown rewrite (its list column must not be pruned).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col


def test_explode_int_list(duck):
    from conftest import assert_same

    tbl = pa.table(
        {
            "id": [1, 2, 3, 4],
            "xs": pa.array([[10, 20], [], None, [30]], type=pa.list_(pa.int64())),
        }
    )
    duck.register("t", tbl)
    out = bt.from_arrow(tbl).explode("xs", alias="x").collect()
    # Empty list (id=2) and null list (id=3) produce no rows.
    assert_same(out, duck.sql("SELECT id, unnest(xs) AS x FROM t"))


def test_explode_string_chunks(duck):
    # The RAG chunk fan-out shape: explode a chunks list into one row per chunk.
    from conftest import assert_same

    tbl = pa.table(
        {
            "doc": [1, 2, 3],
            "chunks": pa.array([["a", "b", "c"], ["d"], []], type=pa.list_(pa.string())),
        }
    )
    duck.register("t", tbl)
    out = bt.from_arrow(tbl).explode("chunks", alias="chunk").select("doc", "chunk").collect()
    assert_same(out, duck.sql("SELECT doc, unnest(chunks) AS chunk FROM t"))


def test_explode_in_place_default_alias(duck):
    from conftest import assert_same

    tbl = pa.table({"id": [1, 2], "xs": pa.array([[1, 2, 3], [4]], type=pa.list_(pa.int64()))})
    duck.register("t", tbl)
    # No alias → the exploded column keeps its name in place.
    out = bt.from_arrow(tbl).explode("xs").collect()
    assert_same(out, duck.sql("SELECT id, unnest(xs) AS xs FROM t"))


def test_explode_then_filter(duck):
    # The predicate references the exploded column, so it must stay ABOVE the
    # explode (it cannot be pushed below). Exercises the pushdown safety branch.
    from conftest import assert_same

    tbl = pa.table({"id": [1, 2], "xs": pa.array([[1, 2, 3], [4, 5]], type=pa.list_(pa.int64()))})
    duck.register("t", tbl)
    out = bt.from_arrow(tbl).explode("xs", alias="x").filter(col("x") > 2).collect()
    assert_same(
        out,
        duck.sql("SELECT id, x FROM (SELECT id, unnest(xs) AS x FROM t) sub WHERE x > 2"),
    )


def test_explode_only_uses_list_column(duck):
    # Projecting just the exploded value still reads the list column (its length
    # drives the row count) — verifies projection pushdown keeps `xs`.
    from conftest import assert_same

    tbl = pa.table({"id": [1, 2], "xs": pa.array([[7, 8], [9]], type=pa.list_(pa.int64()))})
    duck.register("t", tbl)
    out = bt.from_arrow(tbl).explode("xs", alias="x").select("x").collect()
    assert_same(out, duck.sql("SELECT unnest(xs) AS x FROM t"))


def test_explode_unknown_column_raises():
    from batcher._internal.errors import PlanError

    tbl = pa.table({"id": [1], "xs": pa.array([[1]], type=pa.list_(pa.int64()))})
    with pytest.raises(PlanError, match="explode"):
        bt.from_arrow(tbl).explode("nope")
