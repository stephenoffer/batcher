"""SQL ``UNNEST`` in the FROM clause, checked against DuckDB.

`Dataset.explode` has always existed; SQL could not reach it. That mattered more than an
ordinary gap because nested media *is* the multimodal shape — chunks of a document, frames
of a clip, segments of an audio file all arrive as a list column — so a SQL user could not
express the one operation those pipelines are built on.

The semantics that are easy to get wrong and are therefore pinned here: an empty or null
list contributes **no** rows (it does not null-extend), other columns repeat per element,
and the element column takes its name from the `AS u(x)` column list when given.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


@pytest.fixture
def nested(duck):
    t = pa.table({"id": [1, 2, 3, 4], "xs": [[10, 20], [30], [], None]})
    duck.register("t", t)
    return t, duck


def test_comma_unnest(nested) -> None:
    t, duck = nested
    query = "SELECT id, x FROM t, UNNEST(t.xs) AS u(x)"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


def test_cross_join_unnest(nested) -> None:
    t, duck = nested
    query = "SELECT id, u.x AS x FROM t CROSS JOIN UNNEST(t.xs) AS u(x)"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


def test_unqualified_column(nested) -> None:
    t, duck = nested
    query = "SELECT id, x FROM t, UNNEST(xs) AS u(x)"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


def test_aggregate_over_exploded_elements(nested) -> None:
    t, duck = nested
    query = "SELECT sum(x) AS s, count(*) AS n FROM t, UNNEST(t.xs) AS u(x)"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


def test_group_by_the_parent_row(nested) -> None:
    t, duck = nested
    query = "SELECT id, count(*) AS n FROM t, UNNEST(t.xs) AS u(x) GROUP BY id"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


def test_filter_on_the_exploded_element(nested) -> None:
    t, duck = nested
    query = "SELECT id, x FROM t, UNNEST(t.xs) AS u(x) WHERE x > 15"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


def test_strings_not_only_numbers(duck) -> None:
    """The multimodal shape: a document exploded into its chunks."""
    t = pa.table({"doc": ["a", "b"], "chunks": [["p1", "p2", "p3"], ["q1"]]})
    duck.register("t", t)
    query = "SELECT doc, c FROM t, UNNEST(t.chunks) AS u(c)"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


def test_every_list_empty_yields_no_rows(duck) -> None:
    t = pa.table({"id": [1, 2], "xs": [[], []]})
    duck.register("t", t)
    query = "SELECT id, x FROM t, UNNEST(t.xs) AS u(x)"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


# ---- rejections: mistranslating these silently would be worse than refusing ----


def test_unknown_column_is_rejected(nested) -> None:
    from batcher._internal.errors import PlanError

    t, _ = nested
    with pytest.raises(PlanError, match="UNNEST: unknown column"):
        bt.sql("SELECT id, x FROM t, UNNEST(t.nope) AS u(x)", t=t).collect()


def test_multi_expression_unnest_is_rejected(nested) -> None:
    """SQL zips several UNNEST arguments; `explode` does not, so refuse rather than differ."""
    t, _ = nested
    with pytest.raises(NotImplementedError, match="UNNEST of 2 expressions"):
        bt.sql("SELECT x, y FROM t, UNNEST(t.xs, t.xs) AS u(x, y)", t=t).collect()


# ---- the outer / ordinality forms ------------------------------------------


def test_left_join_unnest_on_true_is_an_outer_unnest(nested) -> None:
    """SQL's spelling of `explode(outer=True)`: the row survives an empty/null list."""
    t, duck = nested
    query = "SELECT id, x FROM t LEFT JOIN UNNEST(t.xs) AS u(x) ON TRUE"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


def test_with_ordinality_numbers_the_elements(nested) -> None:
    """SQL ordinality is 1-based — the SQL surface keeps SQL's convention even though
    `Dataset.explode(index=)` is 0-based like the rest of Batcher."""
    t, duck = nested
    query = "SELECT id, x, i FROM t, UNNEST(t.xs) WITH ORDINALITY AS u(x, i)"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


def test_outer_and_ordinality_together(nested) -> None:
    t, duck = nested
    query = "SELECT id, x, i FROM t LEFT JOIN UNNEST(t.xs) WITH ORDINALITY AS u(x, i) ON TRUE"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


def test_a_real_join_predicate_is_still_rejected(nested) -> None:
    """`ON TRUE` is meaningful; anything else has nothing to join against."""
    t, _ = nested
    with pytest.raises(NotImplementedError, match="no join predicate other than ON TRUE"):
        bt.sql("SELECT id, x FROM t LEFT JOIN UNNEST(t.xs) AS u(x) ON id = 1", t=t).collect()
