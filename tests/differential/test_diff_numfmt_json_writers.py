"""Int → text formatting and the JSON writers, against DuckDB.

Two families that needed a kernel rather than a composition. The formatting one is the
reason: `chr`, `bin`/`to_base`, `format_bytes` and `hex(<int>)` all map Int → Utf8, and
the string dispatch downcast its argument to `StringArray` *before* reaching the kernel,
so no wiring could have made them work. They are now dispatched before that downcast,
alongside the `Binary` family, and every other string function is untouched.

The JSON writers complete the reader half: `json_value` (which, unlike
`json_extract_string`, answers only for a scalar and keeps its quotes — DuckDB draws the
same line), `json_contains`, `json_pretty` and `json_structure`.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

FORMAT_QUERIES = [
    "SELECT chr(65) AS r",
    "SELECT chr(233) AS r",
    "SELECT bin(13) AS r",
    "SELECT bin(0) AS r",
    "SELECT to_base(255, 16) AS r",
    "SELECT to_base(255, 2) AS r",
    "SELECT hex(255) AS r",
    "SELECT hex(16) AS r",
    "SELECT format_bytes(512) AS r",
    "SELECT format_bytes(1024) AS r",
    "SELECT format_bytes(1536) AS r",
    "SELECT format_bytes(1048576) AS r",
]


@pytest.mark.parametrize("q", FORMAT_QUERIES)
def test_number_formatting_matches_duckdb(duck, q):
    assert_same(bt.sql(q).collect(), duck.sql(q))


def test_number_formatting_over_a_column(duck):
    t = pa.table({"n": [65, 97, 8364, None]})
    duck.register("t", t)
    q = "SELECT chr(n::INTEGER) AS c, to_base(n, 16) AS h, format_bytes(n) AS b FROM t"
    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


def test_the_dataframe_spelling_agrees_with_the_sql_one():
    ds = bt.from_pydict({"n": [15, 255]})
    out = ds.select(
        c=bt.col("n").chr(),
        b=bt.col("n").to_base(2),
        h=bt.col("n").to_base(16),
        f=bt.col("n").format_bytes(),
        s=bt.col("n").format_bytes(si=True),
    ).to_pydict()
    assert out["b"] == ["1111", "11111111"]
    assert out["h"] == ["F", "FF"]  # uppercase, as DuckDB writes it
    assert out["f"] == ["15 bytes", "255 bytes"]
    assert out["s"] == ["15 bytes", "255 bytes"]
    assert out["c"] == ["\x0f", "ÿ"]


def test_a_negative_number_is_converted_where_duckdb_refuses_it(duck):
    # A pinned divergence, both sides asserted: DuckDB errors on a negative `to_base`
    # argument; the engine writes the magnitude with a `-`, which is what every other
    # base conversion in the language does.
    assert bt.sql("SELECT to_base(-42, 10) AS r").to_pydict()["r"] == ["-42"]
    with pytest.raises(Exception, match="number must be"):
        duck.sql("SELECT to_base(-42, 10) AS r").fetchall()


def test_an_out_of_range_radix_is_refused_at_plan_time():
    with pytest.raises(Exception, match="radix"):
        bt.col("n").to_base(37)


def test_a_code_point_that_is_not_a_character_is_null_not_a_crash():
    # A surrogate has no character. DuckDB errors; the engine's rule on a data path is
    # null for an unrepresentable conversion, and the point is that it does not panic.
    ds = bt.from_pydict({"n": [0xD800, 65]})
    assert ds.select(r=bt.col("n").chr()).to_pydict()["r"] == [None, "A"]


@pytest.fixture
def docs(duck):
    t = pa.table(
        {
            "j": [
                '{"a": 1, "b": "x", "c": [1, 2], "d": {"e": 1}}',
                "[1, 2, 3]",
                '{"a": null}',
            ]
        }
    )
    duck.register("docs", t)
    return t


JSON_QUERIES = [
    "SELECT json_value(j, '$.a') AS r FROM docs",
    "SELECT json_value(j, '$.b') AS r FROM docs",
    "SELECT json_value(j, '$.c') AS r FROM docs",
    "SELECT json_value(j, '$.d') AS r FROM docs",
    "SELECT json_pretty(j) AS r FROM docs",
    "SELECT json_structure(j) AS r FROM docs",
]


@pytest.mark.parametrize("q", JSON_QUERIES)
def test_json_writers_match_duckdb(duck, docs, q):
    assert_same(bt.sql(q, docs=docs).collect(), duck.sql(q))


def test_json_value_answers_only_for_a_scalar(duck, docs):
    # The whole distinction from `json_extract_string`, asserted side by side so neither
    # can drift into the other.
    out = bt.sql(
        "SELECT json_value(j, '$.c') AS v, json_extract_string(j, '$.c') AS e FROM docs",
        docs=docs,
    ).to_pydict()
    assert out["v"][0] is None  # a container has no scalar value
    assert out["e"][0] == "[1,2]"  # the extractor renders it


def test_json_contains_matches_duckdb(duck, docs):
    q = "SELECT json_contains(j, '1') AS r FROM docs"
    assert_same(bt.sql(q, docs=docs).collect(), duck.sql(q))


def test_json_contains_ignores_whitespace_and_key_order():
    ds = bt.from_pydict({"j": ['[{"a": 1, "b": 2}]']})
    got = ds.select(r=bt.col("j").json.contains('{"b":2,"a":1}')).to_pydict()
    assert got["r"] == [True]


def test_the_json_writers_are_null_for_text_that_is_not_json():
    ds = bt.from_pydict({"j": ["oops", None]})
    out = ds.select(
        p=bt.col("j").json.pretty(),
        s=bt.col("j").json.structure(),
        v=bt.col("j").json.value("$.a"),
    ).to_pydict()
    assert out["p"] == [None, None]
    assert out["s"] == [None, None]
    assert out["v"] == [None, None]


# --- the series functions ---------------------------------------------------------------

SERIES_QUERIES = [
    "SELECT range(3) AS r",
    "SELECT range(1, 5) AS r",
    "SELECT range(1, 5, 2) AS r",
    "SELECT range(0, 0) AS r",
    "SELECT generate_series(1, 3) AS r",
    "SELECT generate_series(1, 5, 2) AS r",
    "SELECT generate_series(5, 1, -2) AS r",
    "SELECT generate_series(3, 3) AS r",
]


@pytest.mark.parametrize("q", SERIES_QUERIES)
def test_series_functions_match_duckdb(duck, q):
    # `range` excludes its stop and `generate_series` includes it — the engine's node is
    # the inclusive one, so the exclusive form pulls the stop in by a step. That is the
    # only rewrite that stays right for a step other than 1, which the ±2 cases pin.
    assert_same(bt.sql(q).collect(), duck.sql(q))


def test_spark_sequence_is_the_inclusive_spelling():
    got = bt.sql("SELECT sequence(1, 5) AS r", dialect="spark").to_pydict()
    assert got["r"] == [[1, 2, 3, 4, 5]]


def test_a_series_over_columns(duck):
    t = pa.table({"lo": [1, 5, 2], "hi": [3, 5, 1]})
    duck.register("t", t)
    q = "SELECT generate_series(lo, hi) AS r FROM t"
    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))
