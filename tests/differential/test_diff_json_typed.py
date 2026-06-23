"""Typed JSON extraction — ``col.json.extract_{int,float,bool}`` vs DuckDB.

Locks in parity with DuckDB's ``json_extract(...)::<type>``: a missing path or
invalid JSON yields NULL, and present typed values extract to the right Arrow type.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col


def _json_table():
    # No malformed-JSON row here: DuckDB's json_extract raises on malformed input
    # (Batcher leniently returns NULL — asserted separately below), so a malformed
    # row cannot be compared against the oracle.
    return pa.table(
        {
            "j": [
                '{"n": 42, "f": 3.5, "b": true, "s": "hi"}',
                '{"n": -7, "f": -0.25, "b": false, "s": "yo"}',
                '{"other": 1}',  # path missing → null
                None,  # null input → null
            ]
        }
    )


def test_json_extract_malformed_is_null():
    """Batcher returns NULL for malformed JSON (lenient, ETL-friendly) where DuckDB
    would raise — a deliberate, documented divergence for dirty-data ingestion."""
    t = pa.table({"j": ["not json", '{"n": 5}', None]})
    out = bt.from_arrow(t).select(n=col("j").json.extract_int("$.n")).to_pydict()
    assert out["n"] == [None, 5, None]


def test_json_extract_int(duck):
    from conftest import assert_same

    t = _json_table()
    duck.register("j", t)
    out = bt.from_arrow(t).select(n=col("j").json.extract_int("$.n")).collect()
    assert_same(out, duck.sql("SELECT CAST(json_extract(j, '$.n') AS BIGINT) n FROM j"))


def test_json_extract_float(duck):
    from conftest import assert_same

    t = _json_table()
    duck.register("j", t)
    out = bt.from_arrow(t).select(f=col("j").json.extract_float("$.f")).collect()
    assert_same(out, duck.sql("SELECT CAST(json_extract(j, '$.f') AS DOUBLE) f FROM j"))


def test_json_extract_bool(duck):
    from conftest import assert_same

    t = _json_table()
    duck.register("j", t)
    out = bt.from_arrow(t).select(b=col("j").json.extract_bool("$.b")).collect()
    assert_same(out, duck.sql("SELECT CAST(json_extract(j, '$.b') AS BOOLEAN) b FROM j"))
