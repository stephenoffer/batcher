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
