"""Typed `.json` extraction coerces cross-type leaves like DuckDB's ``CAST``.

DuckDB's ``json_extract(...)::BIGINT/DOUBLE/BOOLEAN`` (the oracle the typed-JSON
differential tests already use) coerces across scalar JSON types: a JSON float
extracted as int rounds to nearest (ties to even), a JSON number extracted as bool
is ``!= 0``, and a JSON bool extracts as ``1``/``0``. Batcher previously returned
NULL for every one of these, silently dropping data. These lock in parity.

Malformed JSON and non-numeric string leaves stay a deliberate, documented lenient
divergence (Batcher → NULL; DuckDB raises), so they are excluded from the oracle
comparison here.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from _harness import assert_same
from batcher import col


def test_json_extract_int_rounds_float_leaf(duck):
    t = pa.table(
        {
            "j": [
                '{"x": 42.0}',  # integral float -> 42
                '{"x": 1e2}',  # exponent form -> 100
                '{"x": 3.5}',  # ties to even -> 4
                '{"x": 2.5}',  # ties to even -> 2
                '{"x": -2.5}',  # ties to even -> -2
                '{"x": 2.6}',  # -> 3
                '{"x": 7}',  # plain int unchanged
                '{"x": true}',  # bool -> 1
                '{"other": 1}',  # missing -> null
                None,  # null input -> null
            ]
        }
    )
    duck.register("j", t)
    out = bt.from_arrow(t).select(n=col("j").json.extract_int("$.x")).collect()
    assert_same(out, duck.sql("SELECT CAST(json_extract(j, '$.x') AS BIGINT) n FROM j"))


def test_json_extract_bool_coerces_numbers(duck):
    t = pa.table(
        {
            "j": [
                '{"x": 1}',  # nonzero -> true
                '{"x": 0}',  # zero -> false
                '{"x": 2}',  # nonzero -> true
                '{"x": 1.0}',  # nonzero float -> true
                '{"x": -0.0}',  # signed zero -> false
                '{"x": true}',  # bool unchanged
                '{"x": false}',
                '{"other": 1}',  # missing -> null
                None,
            ]
        }
    )
    duck.register("j", t)
    out = bt.from_arrow(t).select(b=col("j").json.extract_bool("$.x")).collect()
    assert_same(out, duck.sql("SELECT CAST(json_extract(j, '$.x') AS BOOLEAN) b FROM j"))


def test_json_extract_float_coerces_bool_leaf(duck):
    t = pa.table(
        {
            "j": [
                '{"x": true}',  # -> 1.0
                '{"x": false}',  # -> 0.0
                '{"x": 42}',  # int -> 42.0
                '{"x": 3.5}',  # float unchanged
                None,
            ]
        }
    )
    duck.register("j", t)
    out = bt.from_arrow(t).select(f=col("j").json.extract_float("$.x")).collect()
    assert_same(out, duck.sql("SELECT CAST(json_extract(j, '$.x') AS DOUBLE) f FROM j"))
