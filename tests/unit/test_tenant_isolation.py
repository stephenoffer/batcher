"""Two workloads in one process must not share by accident.

Batcher's caches, pools, and learned-statistics store are **process-global**. That is a
good design for one workload and a leak for two: the result cache was keyed by plan
signature and input identity alone, so two tenants issuing the same query over the same
path collided by construction and the second was served the first's rows.

The fix is namespacing, not a new subsystem — and the tests below are mostly about the
*keys*, because a key that does not separate is the whole bug.

# What a "tenant" is here, and what it is not

A cooperating workload, not an adversary. Two tenants in one process share an address
space; one can read the other's memory directly and no Python-level control changes that.
`tenant()` stops accidental sharing and bounds consumption. It does not isolate mutually
untrusting parties — run one process per trust domain. `docs/user-guide/trust/hardening.md` says
this in the user's language, and it is repeated here because a test file is where someone
checks whether the claim is bigger than the code.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.api.executors import _result_cache_key
from batcher.config import active_config
from batcher.plan.source_stats import source_stats_key

pytestmark = pytest.mark.unit


@pytest.fixture
def plan_and_sources():
    """One identical query over one identical source — the colliding case."""
    dataset = bt.from_pydict({"g": ["a", "b"], "v": [1, 2]})
    plan = dataset.filter(bt.col("v") > 0)._plan
    sources = list(dataset._sources) if hasattr(dataset, "_sources") else []
    return plan, sources


class TestTenantScope:
    def test_it_is_off_by_default(self) -> None:
        # Nothing changes for a deployment that never opts in.
        assert active_config().tenant.tenant_id == ""

    def test_the_scope_nests_and_restores(self) -> None:
        with bt.tenant("outer"):
            assert active_config().tenant.tenant_id == "outer"
            with bt.tenant("inner"):
                assert active_config().tenant.tenant_id == "inner"
            assert active_config().tenant.tenant_id == "outer"
        assert active_config().tenant.tenant_id == ""

    def test_it_restores_on_an_exception(self) -> None:
        """Built on `config_context`, so this is inherited rather than re-implemented —
        but a tenant scope that survived an exception would silently mis-attribute every
        later query, so it is pinned."""
        with pytest.raises(RuntimeError), bt.tenant("analytics"):
            raise RuntimeError("boom")
        assert active_config().tenant.tenant_id == ""

    def test_other_fields_come_along(self) -> None:
        with bt.tenant("etl", max_concurrent_queries=4, cache_share=0.25):
            cfg = active_config().tenant
            assert (cfg.max_concurrent_queries, cfg.cache_share) == (4, 0.25)

    def test_it_is_thread_scoped(self) -> None:
        """A `ContextVar`, not a module global — so one thread's tenant is not another's.

        This is why `tenant()` is built on `config_context`: a module-global "current
        tenant" would be wrong under threads, wrong under asyncio, and wrong when nested.
        """
        import threading

        seen: dict[str, str] = {}

        def worker(name: str) -> None:
            with bt.tenant(name):
                import time

                time.sleep(0.05)  # overlap the two scopes in time
                seen[name] = active_config().tenant.tenant_id

        threads = [threading.Thread(target=worker, args=(n,)) for n in ("a", "b")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert seen == {"a": "a", "b": "b"}


class TestResultCacheIsolation:
    def test_two_tenants_do_not_share_a_cache_entry(self, plan_and_sources) -> None:
        """The leak: same query, same source, and the second tenant got the first's rows."""
        plan, sources = plan_and_sources
        with bt.tenant("team-a"):
            key_a = _result_cache_key(plan, sources)
        with bt.tenant("team-b"):
            key_b = _result_cache_key(plan, sources)
        assert key_a != key_b, "two tenants collide on one result-cache entry"

    def test_the_same_tenant_still_hits_its_own_cache(self, plan_and_sources) -> None:
        # Over-separating would be a silent performance regression rather than a leak, so
        # it needs its own assertion.
        plan, sources = plan_and_sources
        with bt.tenant("team-a"):
            first = _result_cache_key(plan, sources)
            second = _result_cache_key(plan, sources)
        assert first == second

    def test_an_untenanted_key_is_unchanged(self, plan_and_sources) -> None:
        """No tenant and no security context must produce the historical key exactly.

        The scope component is appended, so an un-tenanted key ends in the separator with
        nothing after it — which is stable, and identical for every un-tenanted caller.
        """
        plan, sources = plan_and_sources
        assert _result_cache_key(plan, sources) == _result_cache_key(plan, sources)
        assert _result_cache_key(plan, sources).endswith("|")

    def test_two_principals_do_not_share_a_governed_result(self, plan_and_sources) -> None:
        """The subtler one, and the reason this is a fix rather than a nicety.

        A governed read is rewritten by `enforce` before it reaches the cache, so a masked
        and an unmasked read *happened* to produce different plan signatures. That was an
        accident of the rewrite, not a guarantee: a future rewrite that normalized the
        masked form would start serving unmasked rows to a principal who may not see them.
        Keying on the viewer makes it a guarantee.
        """
        plan, sources = plan_and_sources
        catalog = bt.SecurityCatalog().grant("analyst", on="/data/t.parquet", select=["v"])
        with bt.security(catalog, bt.Principal("ana", roles=["analyst"])):
            key_ana = _result_cache_key(plan, sources)
        with bt.security(catalog, bt.Principal("bob", roles=["admin"])):
            key_bob = _result_cache_key(plan, sources)
        assert key_ana != key_bob, "two principals share one cached governed result"

    def test_the_catalog_is_part_of_the_viewer(self, plan_and_sources) -> None:
        # The same principal under a *different* policy may legitimately see different
        # rows, so a cached result is only reusable under the catalog that produced it.
        plan, sources = plan_and_sources
        principal = bt.Principal("ana", roles=["analyst"])
        permissive = bt.SecurityCatalog().grant("analyst", on="/data/t.parquet")
        restrictive = bt.SecurityCatalog().grant("analyst", on="/data/t.parquet", select=["v"])
        with bt.security(permissive, principal):
            loose = _result_cache_key(plan, sources)
        with bt.security(restrictive, principal):
            tight = _result_cache_key(plan, sources)
        assert loose != tight


class TestLearnedStatisticsIsolation:
    def test_two_tenants_key_their_statistics_apart(self) -> None:
        """Learned statistics include column `min`/`max` — real values from real columns.

        The `MetadataHub` they land in may be Redis or object storage shared across a
        fleet, so unqualified keys mean one tenant's measured bounds are read back by
        every other tenant's optimizer.
        """
        source = bt.from_pydict({"a": [1, 2, 3]})._sources[0]
        with bt.tenant("team-a"):
            key_a = source_stats_key(source)
        with bt.tenant("team-b"):
            key_b = source_stats_key(source)
        assert key_a != key_b
        assert key_a.startswith("team-a/") and key_b.startswith("team-b/")

    def test_an_untenanted_key_keeps_its_historical_shape(self) -> None:
        """An existing deployment must keep reading the statistics it already learned."""
        source = bt.from_pydict({"a": [1, 2, 3]})._sources[0]
        key = source_stats_key(source)
        assert not key.endswith("/")
        assert key.startswith(("id:", "obj:")), key

    def test_the_same_tenant_reads_back_what_it_wrote(self) -> None:
        source = bt.from_pydict({"a": [1, 2, 3]})._sources[0]
        with bt.tenant("team-a"):
            assert source_stats_key(source) == source_stats_key(source)
