"""Semi-join pushdown into a decorrelated aggregate must not change any answer.

The rule deletes *groups* from a decorrelated aggregate — the groups whose key the
other join side cannot produce. That is only invisible when the consumer discards
unmatched right rows, and the shapes below are the ones where getting it wrong shows
up as missing or extra rows rather than an error: the TPC-H Q21 shape (an `EXISTS` and
a `NOT EXISTS` correlated on the same key, which is what decorrelates into the
conditional min/max aggregate), a bare `EXISTS`, a bare `NOT EXISTS` (which *needs*
the absent groups), a correlated scalar subquery, an outer join over a decorrelated
aggregate (which must not be rewritten at all), and the degenerate inputs — empty,
all-null keys, and a key set where nothing survives.

The thresholds are lowered to zero throughout: at these table sizes the cost model
would decline the rewrite, and a test that silently exercises nothing is worse than no
test. `_forced` is the fixture that makes the rule actually fire.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher.kyber.rules.joins.agg_semijoin as _mod
from _harness import assert_same

pytestmark = pytest.mark.differential


@pytest.fixture
def forced():
    """Drop the cost gates so the rewrite fires on test-sized tables."""
    old = (_mod._MIN_GROUP_REDUCTION, _mod._MIN_AGG_INPUT_ROWS)
    _mod._MIN_GROUP_REDUCTION, _mod._MIN_AGG_INPUT_ROWS = 0.0, 0.0
    yield
    _mod._MIN_GROUP_REDUCTION, _mod._MIN_AGG_INPUT_ROWS = old


# `outer` holds a few keys; `inner` holds many, so most groups are droppable. Nulls and
# a key present in `inner` but not `outer` (and vice versa) are deliberate.
_OUTER = {
    "ok": [1, 2, 3, 4, None, 7],
    "sup": [10, 20, 30, 40, 50, 70],
    "tag": ["a", "b", "c", "d", "e", "g"],
}
_INNER = {
    "ik": [1, 1, 2, 2, 3, 5, 6, 6, None, 3],
    "isup": [10, 11, 20, 20, 31, 50, 60, 61, 99, 30],
    "late": [True, False, True, True, False, True, False, True, False, True],
}


# Explicit schemas: an empty column built from a bare `[]` infers as null-typed, and the
# join-key type check rejects that long before this rule is consulted — the empty-input
# cases would fail for a reason that has nothing to do with the rewrite.
_OUTER_SCHEMA = pa.schema(
    [pa.field("ok", pa.int64()), pa.field("sup", pa.int64()), pa.field("tag", pa.string())]
)
_INNER_SCHEMA = pa.schema(
    [pa.field("ik", pa.int64()), pa.field("isup", pa.int64()), pa.field("late", pa.bool_())]
)


def _setup(duck, outer=None, inner=None):
    """Register the same two tables in Batcher and DuckDB."""
    import batcher as bt

    outer_t = pa.table(_OUTER if outer is None else outer, schema=_OUTER_SCHEMA)
    inner_t = pa.table(_INNER if inner is None else inner, schema=_INNER_SCHEMA)
    session = bt.Session()
    session.register("outer_t", bt.from_arrow(outer_t))
    session.register("inner_t", bt.from_arrow(inner_t))
    duck.register("outer_t", outer_t)
    duck.register("inner_t", inner_t)
    return session


def _check(duck, sql, outer=None, inner=None):
    session = _setup(duck, outer, inner)
    assert_same(session.sql(sql).to_arrow(), duck.execute(sql))


# --- the Q21 shape: EXISTS and NOT EXISTS correlated on the same key ---------

Q21_SHAPE = """
SELECT o.ok, o.tag
FROM outer_t o
WHERE EXISTS (SELECT * FROM inner_t i WHERE i.ik = o.ok AND i.isup <> o.sup)
  AND NOT EXISTS (SELECT * FROM inner_t i2
                  WHERE i2.ik = o.ok AND i2.isup <> o.sup AND i2.late)
"""


def test_q21_shape(duck, forced):
    _check(duck, Q21_SHAPE)


def test_exists_only(duck, forced):
    _check(
        duck,
        "SELECT o.ok, o.tag FROM outer_t o "
        "WHERE EXISTS (SELECT * FROM inner_t i WHERE i.ik = o.ok AND i.isup <> o.sup)",
    )


def test_not_exists_only(duck, forced):
    # NOT EXISTS is the case that *needs* the groups a careless restriction would drop:
    # a row qualifies precisely because the subquery found nothing.
    _check(
        duck,
        "SELECT o.ok, o.tag FROM outer_t o "
        "WHERE NOT EXISTS (SELECT * FROM inner_t i WHERE i.ik = o.ok AND i.isup <> o.sup)",
    )


def test_correlated_scalar_subquery(duck, forced):
    _check(
        duck,
        "SELECT o.ok, (SELECT max(i.isup) FROM inner_t i WHERE i.ik = o.ok) AS mx FROM outer_t o",
    )


def test_correlated_scalar_subquery_in_predicate(duck, forced):
    _check(
        duck,
        "SELECT o.ok FROM outer_t o "
        "WHERE o.sup < (SELECT max(i.isup) FROM inner_t i WHERE i.ik = o.ok)",
    )


# --- the shape that must NOT be rewritten -----------------------------------

# These two were `strict` xfails: Batcher used to fill an unmatched RIGHT/FULL join row's
# preserved side with the *other* side's key instead of NULL, so the 6-row fixtures below
# returned `(5, 5, 50)` where DuckDB returns `(None, 5, 50)`. That was a SQL-translation
# bug independent of this rule (which refuses right/full joins outright, as
# `tests/unit/test_agg_semijoin_pushdown.py` pins) and is now fixed — SQL's `ON` form no
# longer merges the two sides' key columns. They are plain passing assertions again, and
# remain the only differential coverage of this shape.


def test_right_join_over_a_decorrelated_aggregate_is_not_rewritten(duck, forced):
    # A RIGHT JOIN preserves unmatched *right* rows, so the groups this rule would drop
    # are exactly the rows that must survive. If the guard regresses, this test sees
    # missing rows — nothing else in the suite would.
    _check(
        duck,
        "SELECT o.ok, g.gk, g.mx FROM outer_t o RIGHT JOIN "
        "(SELECT ik, min(ik) AS gk, max(isup) AS mx FROM inner_t GROUP BY ik) g ON o.ok = g.ik",
    )


def test_full_join_over_a_decorrelated_aggregate_is_not_rewritten(duck, forced):
    _check(
        duck,
        "SELECT o.ok, g.gk, g.mx FROM outer_t o FULL JOIN "
        "(SELECT ik, min(ik) AS gk, max(isup) AS mx FROM inner_t GROUP BY ik) g ON o.ok = g.ik",
    )


def test_left_join_over_a_decorrelated_aggregate_is_rewritten_safely(duck, forced):
    # The mirror image: a LEFT JOIN *is* rewritten, and null-extends the left rows whose
    # group was dropped — which is what it would have done anyway.
    _check(
        duck,
        "SELECT o.ok, g.mx FROM outer_t o "
        "LEFT JOIN (SELECT ik, max(isup) AS mx FROM inner_t GROUP BY ik) g ON o.ok = g.ik",
    )


# --- degenerate inputs -------------------------------------------------------


def test_empty_outer(duck, forced):
    _check(duck, Q21_SHAPE, outer={"ok": [], "sup": [], "tag": []})


def test_empty_inner(duck, forced):
    _check(duck, Q21_SHAPE, inner={"ik": [], "isup": [], "late": []})


def test_all_null_keys(duck, forced):
    # NULL never equals NULL, so every group is unmatched: the rewrite must drop them
    # all and still agree with DuckDB's three-valued logic.
    _check(
        duck,
        Q21_SHAPE,
        outer={"ok": [None, None], "sup": [1, 2], "tag": ["a", "b"]},
        inner={"ik": [None, None, None], "isup": [1, 2, 3], "late": [True, False, True]},
    )


def test_no_key_survives(duck, forced):
    # Disjoint key spaces: the semi-join deletes every group. The aggregate runs over an
    # empty input, which is the path most likely to produce a spurious row.
    _check(
        duck,
        Q21_SHAPE,
        outer={"ok": [100, 200], "sup": [1, 2], "tag": ["a", "b"]},
        inner={"ik": [1, 2, 3], "isup": [9, 8, 7], "late": [True, False, True]},
    )


def test_single_row_each_side(duck, forced):
    _check(
        duck,
        Q21_SHAPE,
        outer={"ok": [1], "sup": [5], "tag": ["a"]},
        inner={"ik": [1], "isup": [6], "late": [False]},
    )


def test_duplicate_keys_on_the_restricting_side(duck, forced):
    # The semi-join build side is not deduplicated (a semi-join ignores build-side
    # duplicates); prove that stays true rather than fanning the aggregate's input out.
    _check(
        duck,
        Q21_SHAPE,
        outer={"ok": [1, 1, 1, 2], "sup": [10, 11, 12, 20], "tag": ["a", "b", "c", "d"]},
    )
