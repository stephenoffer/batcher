"""`explode(outer=…, index=…)` — the two shapes a document pipeline needs.

Plain `explode` drops a row whose list is null or empty. That is DuckDB's `UNNEST`
semantics and correct as a default, but it is a trap for chunking pipelines: a document
that chunked to nothing disappears from the relation entirely, taking its id and metadata
with it. No error, no warning — the row count is simply smaller than the document count.

`outer=True` is the `LEFT JOIN LATERAL … ON true` form that keeps it, and `index=` is the
0-based element position (`posexplode`) that lets chunks be reordered after a shuffle.
Both are checked here against DuckDB expressing the same thing its own way.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

# Row 3's list is empty and row 4's is null — the two cases `outer` is about.
_ROWS = {"id": [1, 2, 3, 4], "xs": [[10, 20], [30], [], None]}


@pytest.fixture
def nested(duck):
    t = pa.table(_ROWS)
    duck.register("t", t)
    return t, duck


def test_default_drops_empty_and_null_lists(nested) -> None:
    """The existing behaviour must be unchanged by the new arguments."""
    t, duck = nested
    got = bt.from_arrow(t).explode("xs").collect()
    assert_same(got, duck.sql("SELECT id, unnest(xs) AS xs FROM t"))


def test_outer_keeps_them_with_a_null_element(nested) -> None:
    t, duck = nested
    got = bt.from_arrow(t).explode("xs", outer=True).collect()
    assert_same(
        got,
        duck.sql(
            "SELECT t.id AS id, u.xs AS xs FROM t "
            "LEFT JOIN LATERAL (SELECT unnest(t.xs) AS xs) u ON true"
        ),
    )


def test_index_is_the_zero_based_position(nested) -> None:
    """DuckDB's `generate_subscripts` is 1-based; Batcher is 0-based, matching its own
    `with_row_index`. One convention per engine beats matching each source dialect."""
    t, duck = nested
    got = bt.from_arrow(t).explode("xs", index="i").collect()
    assert_same(
        got,
        duck.sql("SELECT id, unnest(xs) AS xs, generate_subscripts(xs, 1) - 1 AS i FROM t"),
    )


def test_outer_and_index_together(nested) -> None:
    """A row kept only by `outer` has no element, so its position is NULL too."""
    t, duck = nested
    got = bt.from_arrow(t).explode("xs", outer=True, index="i").collect()
    assert_same(
        got,
        duck.sql(
            "SELECT t.id AS id, u.xs AS xs, u.i AS i FROM t LEFT JOIN LATERAL ("
            "  SELECT unnest(t.xs) AS xs, generate_subscripts(t.xs, 1) - 1 AS i"
            ") u ON true"
        ),
    )


def test_outer_with_an_alias(nested) -> None:
    t, duck = nested
    got = bt.from_arrow(t).explode("xs", alias="x", outer=True).collect()
    assert_same(
        got,
        duck.sql(
            "SELECT t.id AS id, u.x AS x FROM t "
            "LEFT JOIN LATERAL (SELECT unnest(t.xs) AS x) u ON true"
        ),
    )


def test_every_list_empty(duck) -> None:
    """Plain explode returns nothing at all here; `outer` returns one row each."""
    t = pa.table({"id": [1, 2], "xs": [[], []]})
    duck.register("t", t)

    assert bt.from_arrow(t).explode("xs").collect().num_rows == 0
    assert_same(
        bt.from_arrow(t).explode("xs", outer=True).collect(),
        duck.sql(
            "SELECT t.id AS id, u.xs AS xs FROM t "
            "LEFT JOIN LATERAL (SELECT unnest(t.xs) AS xs) u ON true"
        ),
    )


def test_strings_the_document_chunking_shape(duck) -> None:
    t = pa.table({"doc": ["a", "b", "c"], "chunks": [["p", "q", "r"], [], ["s"]]})
    duck.register("t", t)
    got = bt.from_arrow(t).explode("chunks", outer=True, index="i").collect()

    assert_same(
        got,
        duck.sql(
            "SELECT t.doc AS doc, u.chunks AS chunks, u.i AS i FROM t LEFT JOIN LATERAL ("
            "  SELECT unnest(t.chunks) AS chunks,"
            "         generate_subscripts(t.chunks, 1) - 1 AS i"
            ") u ON true"
        ),
    )
    # Every document survives, which is the whole point.
    assert sorted(set(got.column("doc").to_pylist())) == ["a", "b", "c"]


# ---- plan-level guarantees --------------------------------------------------


def test_the_index_column_is_in_the_advertised_schema() -> None:
    ds = bt.from_pydict(_ROWS).explode("xs", index="i")
    assert "i" in ds.columns
    assert ds.schema.field("i").type == pa.int64()


def test_an_index_name_that_collides_is_rejected() -> None:
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError, match="collides"):
        bt.from_pydict(_ROWS).explode("xs", index="id")


def test_position_survives_a_multi_batch_input() -> None:
    """The position is per-list, not a running counter, so batching cannot change it."""
    rows = {"id": list(range(1000)), "xs": [[i, i + 1] for i in range(1000)]}
    got = bt.from_pydict(rows).explode("xs", index="i").collect()

    assert got.column("i").to_pylist() == [0, 1] * 1000
