"""Bug-hunt regression coverage for `Session` catalog + prepared-statement-cache correctness.

The prepared-statement cache reuses a built `Dataset` for a repeated query text against an
unchanged catalog. A stale entry surviving a catalog mutation would be a silent wrong answer
(an S1), so these pin the generation-bump invalidation and per-call-override paths, plus
session isolation and re-registration semantics.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError


@pytest.mark.unit
def test_reregister_invalidates_cached_plan() -> None:
    s = bt.Session()
    s.register("t", bt.from_pydict({"x": [1, 2, 3]}))
    assert s.sql("SELECT SUM(x) AS s FROM t").to_pydict() == {"s": [6]}
    # Re-registering the same name must invalidate the cached plan, not serve the old sum.
    s.register("t", bt.from_pydict({"x": [10, 20, 30]}))
    assert s.sql("SELECT SUM(x) AS s FROM t").to_pydict() == {"s": [60]}


@pytest.mark.unit
def test_per_call_override_does_not_poison_catalog_cache() -> None:
    s = bt.Session()
    s.register("t", bt.from_pydict({"x": [1, 2, 3]}))
    assert s.sql("SELECT SUM(x) AS s FROM t").to_pydict() == {"s": [6]}
    # A one-off per-call binding must not be cached under the catalog key...
    assert s.sql("SELECT SUM(x) AS s FROM t", t=bt.from_pydict({"x": [100]})).to_pydict() == {
        "s": [100]
    }
    # ...so the next catalog-only run still resolves the registered table.
    assert s.sql("SELECT SUM(x) AS s FROM t").to_pydict() == {"s": [6]}


@pytest.mark.unit
def test_drop_invalidates_and_sessions_are_isolated() -> None:
    s1, s2 = bt.Session(), bt.Session()
    s1.register("t", bt.from_pydict({"x": [1]}))
    assert "t" not in s2  # two sessions share no catalog
    s1.drop("t")
    with pytest.raises(PlanError):
        s1.table("t")


@pytest.mark.unit
def test_unregistered_table_raises_typed_error() -> None:
    s = bt.Session()
    with pytest.raises(PlanError):
        s.table("missing")


@pytest.mark.unit
def test_per_call_binding_is_cached_but_keyed_by_object_identity() -> None:
    """A repeated per-call binding may reuse its plan; a *different* object may not.

    `Dataset.sql()` always passes a per-call binding, so caching those is what makes the
    primary SQL entry point skip the sqlglot parse + AST translation. The danger is that
    two distinct datasets can be structurally identical — same schema, same row count, same
    plan shape — so anything short of an identity check would serve one query's plan for
    the other's data. That is a silent wrong answer, which is why this pins values, not
    just plan reuse.
    """
    s = bt.Session()
    a = bt.from_pydict({"x": [1, 2, 3]})
    b = bt.from_pydict({"x": [10, 20, 30]})  # same schema and shape as `a`
    q = "SELECT SUM(x) AS s FROM t"

    assert s.sql(q, t=a).to_pydict() == {"s": [6]}
    assert s.sql(q, t=a).to_pydict() == {"s": [6]}  # repeat: cache hit, same answer
    assert s.sql(q, t=b).to_pydict() == {"s": [60]}  # different object: must NOT reuse `a`
    assert s.sql(q, t=a).to_pydict() == {"s": [6]}  # and back again


@pytest.mark.unit
def test_dataset_sql_is_cached_per_dataset() -> None:
    """`ds.sql()` on two same-shaped datasets must not share a plan."""
    a = bt.from_pydict({"x": [1, 2, 3]})
    b = bt.from_pydict({"x": [10, 20, 30]})
    q = "SELECT SUM(x) AS s FROM self"
    assert a.sql(q).to_pydict() == {"s": [6]}
    assert b.sql(q).to_pydict() == {"s": [60]}
    assert a.sql(q).to_pydict() == {"s": [6]}


@pytest.mark.unit
def test_prepared_cache_is_bounded() -> None:
    """The cache pins its bound datasets alive, so it must evict rather than grow."""
    from batcher.api.sql_session import Session

    s = bt.Session()
    q = "SELECT SUM(x) AS s FROM t"
    for i in range(Session._PLAN_CACHE_MAX + 50):
        assert s.sql(q, t=bt.from_pydict({"x": [i]})).to_pydict() == {"s": [i]}
    assert len(s._plan_cache) <= Session._PLAN_CACHE_MAX
