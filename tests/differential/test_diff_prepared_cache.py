"""A cached execution must agree with DuckDB, not merely with itself.

`api/orchestration/prepared.py` answers a re-issued query from a memo. `tests/unit/
test_prepared_cache.py` pins the identity invariants that decide *which* memo is served;
this file pins the other half, which no amount of self-consistency can establish: that the
rows the memo serves are still the right rows.

Every case runs the same query **twice** under `execution.fast_path` -- the first call
derives and stores, the second is served from the cache -- and holds *both* against DuckDB.
Checking only the second would miss a path that is consistently wrong, and checking only the
first would not exercise the cache at all.

The shapes are chosen to be the ones a memoized plan can get wrong: nulls (a different
branch in the empty/short-circuit logic), an empty result (its schema comes from a field the
entry precomputes rather than from the data), a sort (the one thing an order-independent
comparison cannot see), and a two-source join (where the per-source projection and predicate
maps the entry replays are indexed by `source_id`).
"""

from __future__ import annotations

import pytest

import batcher as bt
from _harness import assert_same, assert_same_ordered
from batcher.api.orchestration import prepared
from batcher.config import Config, ExecutionConfig, config_context

_FAST = Config(execution=ExecutionConfig(fast_path=True))

#: Rows deliberately few: this is a correctness file, and a big input only slows it down.
_N = 120

_LEFT = {
    "id": list(range(_N)),
    "grp": [i % 7 for i in range(_N)],
    "s": [None if i % 17 == 0 else f"k{i % 11}" for i in range(_N)],
    "v": [None if i % 13 == 0 else float(i) - 30.0 for i in range(_N)],
}
_RIGHT = {"id": list(range(0, _N, 3)), "w": [float(i % 5) for i in range(0, _N, 3)]}


@pytest.fixture
def registered(duck):
    """Both tables in DuckDB and in Batcher, and an empty prepared cache."""
    prepared.clear()
    left, right = bt.from_pydict(_LEFT), bt.from_pydict(_RIGHT)
    duck.register("left_t", left.collect())
    duck.register("right_t", right.collect())
    yield left, right, duck
    prepared.clear()


def _twice(build):
    """Run `build()` cold then cache-warm, returning both results.

    The assertion that the second call was actually *served* from the cache is the
    `len(_CACHE)` check in the unit suite; here the point is only that both agree with the
    oracle, so a miss would weaken the test but never make it pass wrongly.
    """
    with config_context(_FAST):
        return build(), build()


class TestACachedResultStillMatchesDuckDB:
    def test_filter_and_projection(self, registered):
        left, _right, duck = registered
        for got in _twice(lambda: left.filter(bt.col("id") > 40).select("id", "v").collect()):
            assert_same(got, duck.sql("SELECT id, v FROM left_t WHERE id > 40"))

    def test_grouped_aggregate_over_nulls(self, registered):
        left, _right, duck = registered
        for got in _twice(
            lambda: (
                left.group_by("grp")
                .agg(s=bt.col("v").sum(), c=bt.col("v").count(), m=bt.col("v").mean())
                .collect()
            )
        ):
            assert_same(
                got,
                duck.sql(
                    "SELECT grp, SUM(v) AS s, COUNT(v) AS c, AVG(v) AS m FROM left_t GROUP BY grp"
                ),
            )

    def test_a_sort_keeps_its_order(self, registered):
        """`assert_same_ordered` on purpose -- an order-independent compare cannot see a
        sort served from a stale memo."""
        left, _right, duck = registered
        for got in _twice(lambda: left.sort("id", descending=True).limit(25).collect()):
            assert_same_ordered(got, duck.sql("SELECT * FROM left_t ORDER BY id DESC LIMIT 25"))

    def test_a_join_across_two_sources(self, registered):
        """Two sources, so the entry's `source_id`-keyed projection and predicate maps are
        replayed in an order that matters."""
        left, right, duck = registered
        for got in _twice(lambda: left.join(right, on="id").select("id", "w", "grp").collect()):
            assert_same(
                got,
                duck.sql("SELECT l.id, r.w, l.grp FROM left_t l JOIN right_t r USING (id)"),
            )

    def test_an_empty_result_keeps_the_promised_columns(self, registered):
        left, _right, duck = registered
        for got in _twice(lambda: left.filter(bt.col("id") < 0).select("id", "s").collect()):
            assert got.num_rows == 0
            assert_same(got, duck.sql("SELECT id, s FROM left_t WHERE id < 0"))

    def test_distinct_over_a_column_with_nulls(self, registered):
        left, _right, duck = registered
        for got in _twice(lambda: left.select("s").distinct().collect()):
            assert_same(got, duck.sql("SELECT DISTINCT s FROM left_t"))
