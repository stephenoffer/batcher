"""`json_valid` and `json_array_length` on NULL and on wrong-shaped documents.

Both functions answered null for several unrelated reasons and collapsed them.

`json_valid(NULL)` returned FALSE where DuckDB returns NULL. A predicate has to propagate
its input's nullness, and this one reported a NULL document as *invalid JSON* rather than
unknown -- so `WHERE NOT json_valid(j)`, which is how you isolate malformed documents,
returned every NULL row alongside the genuinely bad ones. That is a row-set difference from
a query that raises nothing, which is why the filter forms are asserted here and not just
the scalar values.

`json_array_length` returned null for any document that is not an array, where DuckDB
returns 0: `{"a":1}`, `"s"` and `5` all have zero elements. Null was reserved for the two
cases that really are unknown, a null input and text that does not parse.

Unparseable text is the one case that still differs: DuckDB raises, and Batcher answers
null. That predates these fixes and is narrowed by them rather than introduced.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

#: Every JSON root type, plus an empty array, plus a NULL document.
_DOCS = ['{"a":1}', "[1,2]", '"s"', "5", "true", "null", "[]", None]


def _table() -> pa.Table:
    return pa.table({"id": pa.array(range(len(_DOCS)), type=pa.int64()), "j": pa.array(_DOCS)})


def test_json_valid_propagates_null(duck):
    """NULL in, NULL out — not FALSE."""
    table = _table()
    sql = "SELECT id, json_valid(j) AS v FROM t"
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


@pytest.mark.parametrize(
    "predicate",
    ["json_valid(j)", "NOT json_valid(j)", "json_valid(j) IS NULL", "json_valid(j) IS NOT NULL"],
)
def test_the_filter_forms_select_the_same_rows_as_duckdb(duck, predicate):
    """The assertion that matters: a scalar comparison alone missed the row-set change."""
    table = pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int64()),
            "j": pa.array(['{"a":1}', "not json", None]),
        }
    )
    sql = f"SELECT id FROM t WHERE {predicate}"
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


def test_a_null_document_is_not_reported_as_invalid():
    """Stated directly, because both readings return a boolean and only one is right."""
    table = pa.table({"j": pa.array(['{"a":1}', "not json", None])})
    got = bt.sql("SELECT json_valid(j) AS v FROM t", t=table).to_pydict()["v"]
    assert got == [True, False, None]


def test_json_array_length_is_zero_for_a_non_array(duck):
    """An object, a string and a number all have zero elements, and none has null."""
    table = _table()
    sql = "SELECT id, json_array_length(j) AS n FROM t"
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


def test_json_array_length_keeps_null_only_for_a_null_document():
    """The distinction the fix turns on: not-an-array is 0, unknown stays null."""
    table = _table()
    got = bt.sql("SELECT json_array_length(j) AS n FROM t", t=table).to_pydict()["n"]
    assert got == [0, 2, 0, 0, 0, 0, 0, None]


def test_an_unparseable_document_stays_null_rather_than_zero():
    """The remaining divergence, pinned so a later change has to be deliberate.

    DuckDB raises here. Answering 0 would be worse than either, because it would report a
    corrupt document as a well-formed empty one.
    """
    table = pa.table({"j": pa.array(["not json"])})
    assert bt.sql("SELECT json_array_length(j) AS n FROM t", t=table).to_pydict()["n"] == [None]
    assert bt.sql("SELECT json_valid(j) AS v FROM t", t=table).to_pydict()["v"] == [False]


def test_both_survive_a_partitioned_collect(duck):
    """A per-row predicate must not depend on how the rows were split."""
    table = _table()
    sql = "SELECT id, json_valid(j) AS v, json_array_length(j) AS n FROM t"
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).repartition(3).collect(), duck.sql(sql))
