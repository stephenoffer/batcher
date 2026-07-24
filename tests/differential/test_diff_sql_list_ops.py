"""SQL list/array operations (length, index, reductions, contains) vs DuckDB."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_same_ordered


@pytest.fixture
def t(duck):
    tbl = pa.table({"id": [1, 2, 3], "x": [10, 20, 30], "y": [3, 1, 2]})
    duck.register("t", tbl)
    return tbl


@pytest.mark.parametrize(
    "q",
    [
        "SELECT array_length([x, y]) n FROM t",
        "SELECT [x, y, id][1] e FROM t",
        "SELECT [x, y, id][2] e FROM t",
        "SELECT list_contains([x, y], 10) c FROM t",
        "SELECT list_contains([x, y], 99) c FROM t",
        "SELECT array_length(list_reverse([x, y, id])) n FROM t",
    ],
)
def test_list_ops(duck, t, q):
    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        "SELECT list_sum([x, y]) s FROM t",
        "SELECT list_min([x, y, id]) m FROM t",
        "SELECT list_max([x, y, id]) m FROM t",
        "SELECT list_sum(array_agg(x)) s FROM t",
    ],
)
def test_list_reductions(duck, t, q):
    # assert_same tolerates int/Decimal vs float (list reductions cast to float).
    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


@pytest.fixture
def vt(duck):
    tbl = pa.table(
        {
            "id": [1, 2, 3],
            "a": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            "b": [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        }
    )
    duck.register("vt", tbl)
    return tbl


@pytest.mark.parametrize(
    "q",
    [
        # Two-column vector functions, using DuckDB's canonical `list_*` spellings so the
        # same SQL runs on both engines — this is the SQL-level vector-search surface.
        "SELECT id, list_cosine_similarity(a, b) s FROM vt",
        "SELECT id, list_distance(a, b) s FROM vt",
        "SELECT id, list_dot_product(a, b) s FROM vt",
        # Against a query-vector array literal, the RAG retrieval shape.
        "SELECT id, list_cosine_similarity(a, [1.0, 0.0]) s FROM vt",
        "SELECT id, list_distance(a, [1.0, 0.0]) s FROM vt ORDER BY s ASC",
    ],
)
def test_sql_vector_functions_match_duckdb(duck, vt, q):
    assert_same_ordered(bt.sql(q, vt=vt).collect(), duck.sql(q))


def test_sql_vector_search_top_k():
    # The end-to-end ML-in-SQL retrieval query: rank rows by similarity to a query vector.
    tbl = pa.table({"id": [1, 2, 3], "emb": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]})
    out = bt.sql(
        "SELECT id FROM docs ORDER BY list_cosine_similarity(emb, [1.0, 0.0]) DESC LIMIT 2",
        docs=tbl,
    ).to_pydict()
    assert out["id"] == [1, 3]
