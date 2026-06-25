"""Coverage for the Phase 2 string functions.

`position`/`regexp_extract_all`/`levenshtein` are checked against DuckDB; the
Spark-only ones (`substring_index`, `overlay`, `regexp_count`) and `soundex`
(a DuckDB extension, not built in) are pinned to fixtures.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


def _strs():
    return pa.table({"s": pa.array(["a.b.c.d", "hello", "smith", "robert", None])})


def test_position_matches_duckdb(duck):
    from conftest import assert_same

    duck.register("t", _strs())
    out = bt.from_arrow(_strs()).select(r=col("s").str.position(".")).collect()
    assert_same(out, duck.sql("SELECT instr(s, '.') AS r FROM t"))


def test_levenshtein_matches_duckdb(duck):
    from conftest import assert_same

    duck.register("t", _strs())
    out = bt.from_arrow(_strs()).select(r=col("s").str.levenshtein("smith")).collect()
    assert_same(out, duck.sql("SELECT levenshtein(s, 'smith') AS r FROM t"))


def test_regexp_extract_all_matches_duckdb(duck):
    from conftest import assert_same

    duck.register("t", _strs())
    out = bt.from_arrow(_strs()).select(r=col("s").str.regexp_extract_all("[a-z]")).collect()
    assert_same(out, duck.sql("SELECT regexp_extract_all(s, '[a-z]') AS r FROM t"))


def test_spark_only_string_fns_fixtures():
    out = (
        bt.from_arrow(_strs())
        .select(
            si=col("s").str.substring_index(".", 2),
            ov=col("s").str.overlay("XY", 2, 1),
            rc=col("s").str.regexp_count("[a-z]"),
            sx=col("s").str.soundex(),
        )
        .collect()
        .to_pydict()
    )
    assert out["si"] == ["a.b", "hello", "smith", "robert", None]
    assert out["ov"] == ["aXYb.c.d", "hXYllo", "sXYith", "rXYbert", None]
    assert out["rc"] == [4, 5, 5, 6, None]
    # American Soundex: Smith→S530, Robert→R163 (the canonical textbook examples).
    assert out["sx"] == ["A123", "H400", "S530", "R163", None]
