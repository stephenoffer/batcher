"""``FROM generate_series(...)`` / ``FROM range(...)`` — the generated integer spine.

Neither reached a handler. `_table` looked the name up in the *registered* table-function
registry, found nothing, and reported ``unknown table ''`` — an empty name, because
sqlglot parses both into a `GenerateSeries` node that carries no table name at all.

They are the same node; ``generate_series(a, b)`` includes `b` and ``range(a, b)``
excludes it, which is the one thing an implementation can get backwards. Both directions
of that off-by-one are covered below, along with the descending and empty series.

The relation is materialized in memory, so an unbounded series is refused with a stated
limit rather than truncated silently. That refusal is pinned too.
"""

from __future__ import annotations

import pytest

import batcher as bt
from _harness import assert_same
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.differential


@pytest.mark.parametrize(
    "query",
    [
        # Inclusive end.
        "SELECT * FROM generate_series(1, 5)",
        "SELECT * FROM generate_series(0, 0)",
        # Exclusive end.
        "SELECT * FROM range(3)",
        "SELECT * FROM range(1, 4)",
        "SELECT * FROM range(0, 0)",
        # An explicit step, ascending and descending.
        "SELECT * FROM range(1, 10, 2)",
        "SELECT * FROM generate_series(1, 10, 3)",
        "SELECT * FROM range(5, 0, -1)",
        # A start past the end yields no rows rather than running away.
        "SELECT * FROM generate_series(3, 1)",
        # Negative bounds.
        "SELECT * FROM generate_series(-3, 1)",
        # `AS t(v)` renames the single output column.
        "SELECT * FROM generate_series(1, 3) AS t(v)",
        # Used as an ordinary relation: aggregated, filtered, and joined.
        "SELECT sum(generate_series) AS s FROM generate_series(1, 10)",
        "SELECT * FROM range(10) WHERE range % 3 = 0",
    ],
)
def test_series_matches_duckdb(duck, query):
    assert_same(bt.sql(query).collect(), duck.sql(query))


def test_generate_series_includes_its_end_and_range_excludes_it(duck):
    """The one off-by-one that separates the two spellings."""
    inclusive = bt.sql("SELECT * FROM generate_series(1, 3)").collect().to_pydict()
    exclusive = bt.sql("SELECT * FROM range(1, 3)").collect().to_pydict()
    assert inclusive == {"generate_series": [1, 2, 3]}
    assert exclusive == {"range": [1, 2]}


def test_an_unbounded_series_is_refused_rather_than_truncated():
    with pytest.raises(PlanError, match="row limit"):
        bt.sql("SELECT * FROM range(1, 100000000000)").collect()


def test_a_zero_step_is_refused():
    with pytest.raises(PlanError, match="step of 0"):
        bt.sql("SELECT * FROM range(1, 10, 0)").collect()
