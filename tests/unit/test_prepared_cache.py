"""The prepared-execution cache: a hit must be the same derivation, never a neighbour's.

`api/orchestration/prepared.py` answers a re-issued small query from a memo instead of
re-deriving its plan. Every test here pins an *identity* invariant, because that is the only
way this cache can fail: a stale or aliased entry returns the wrong table at full speed,
with nothing raising and every other suite still green.

`test_fast_path.py` already pins the gate and result-invariance of the path that fills this
cache. What is tested here is the part that is new -- that two queries which merely *look*
alike never share an entry.
"""

from __future__ import annotations

import gc
import weakref

import pyarrow as pa
import pytest

import batcher as bt
from batcher.api.orchestration import prepared
from batcher.config import Config, ExecutionConfig, active_config, config_context

_FAST = Config(execution=ExecutionConfig(fast_path=True))


@pytest.fixture
def clean_cache():
    """An empty cache before and after, so ordering between tests cannot mask a bug."""
    prepared.clear()
    yield
    prepared.clear()


def _table(values: list[int]) -> pa.Table:
    return pa.table({"a": pa.array(values, type=pa.int64())})


def _select_a(table: pa.Table) -> list[int]:
    return bt.from_arrow(table).select("a").collect().to_pydict()["a"]


class TestItOnlyRunsWhenTheFastPathDoes:
    """The cache inherits the fast path's gate and its documented trade wholesale. It must
    not become a second, wider way into the same skipped orchestration."""

    def test_nothing_is_cached_while_the_flag_is_off(self, clean_cache):
        assert active_config().execution.fast_path is False
        _select_a(_table([1, 2, 3]))
        assert len(prepared._CACHE) == 0

    def test_an_entry_appears_once_the_flag_is_on(self, clean_cache):
        with config_context(_FAST):
            _select_a(_table([1, 2, 3]))
            assert len(prepared._CACHE) == 1


class TestAHitIsTheSameQuery:
    """Every way two queries can share a plan fingerprint without sharing an answer."""

    def test_distinct_tables_of_the_same_schema_do_not_share_an_entry(self, clean_cache):
        """A `Scan`'s IR is only its `source_id`, so these two plans fingerprint alike.
        Only the weak-reference identity check separates them."""
        t1, t2 = _table([1, 2, 3]), _table([10, 20, 30])
        with config_context(_FAST):
            assert _select_a(t1) == [1, 2, 3]
            assert _select_a(t2) == [10, 20, 30]
            assert _select_a(t1) == [1, 2, 3]

    def test_a_differing_literal_does_not_share_an_entry(self, clean_cache):
        ds = bt.from_arrow(_table([10, 20, 30]))
        with config_context(_FAST):
            assert ds.filter(bt.col("a") > 5).collect().num_rows == 3
            assert ds.filter(bt.col("a") > 15).collect().num_rows == 2
            assert ds.filter(bt.col("a") > 5).collect().num_rows == 3

    def test_same_column_names_and_different_types_do_not_share_an_entry(self, clean_cache):
        """`content_key` carries `Scan.identity_suffix()` (the schema) for exactly this."""
        ints = pa.table({"a": pa.array([1, 2], type=pa.int64())})
        floats = pa.table({"a": pa.array([1.5, 2.5], type=pa.float64())})
        with config_context(_FAST):
            assert bt.from_arrow(ints).select("a").collect().schema.field(0).type == pa.int64()
            assert bt.from_arrow(floats).select("a").collect().schema.field(0).type == pa.float64()

    def test_a_recycled_source_id_cannot_alias_one_table_onto_another(self, clean_cache):
        """CPython reuses `id()` freely once an object is freed, and source ids are part of
        the key. The weak reference is what makes reuse a miss instead of a wrong answer.

        Loops rather than asserting a single collision because reuse is not schedulable:
        the point is that no iteration may ever return another table's rows, whether or not
        an address happened to come back around.
        """
        with config_context(_FAST):
            for i in range(300):
                assert _select_a(_table([i, i + 1])) == [i, i + 1]
                gc.collect()

    def test_a_rebuilt_dataset_over_the_same_table_still_answers_correctly(self, clean_cache):
        """The hit case: a fresh `Dataset` each call, same table, same plan content."""
        table = _table([1, 2, 3])
        with config_context(_FAST):
            for _ in range(5):
                got = bt.from_arrow(table).filter(bt.col("a") > 1).select("a").collect()
                assert got.to_pydict()["a"] == [2, 3]


class TestTheEntryIsBounded:
    """A parameterized workload mints a fresh key per call; the cache must not grow with it."""

    def test_the_cap_is_honored(self, clean_cache):
        with config_context(_FAST):
            for i in range(prepared.MAX_ENTRIES + 40):
                bt.from_arrow(_table([1, 2, 3])).filter(bt.col("a") > i).collect()
        assert len(prepared._CACHE) <= prepared.MAX_ENTRIES

    def test_its_own_references_to_a_source_are_weak(self, clean_cache):
        """The entry must not be what keeps a table resident.

        Asserted against the entry rather than end to end, because end to end cannot see
        it: `kyber.plan_cache` stores its sources *strongly* (`plan_cache.py`, the
        `(result, tuple(sources))` value), so a source stays reachable after any query,
        cached or not. That retention is bounded by that cache's own LRU and predates this
        module -- but it does mean a liveness assertion on the source itself would pass
        here while proving nothing about these references.
        """
        with config_context(_FAST):
            bt.from_arrow(_table([1, 2, 3])).select("a").collect()
        entries = list(prepared._CACHE.values())
        assert entries, "expected the query to leave an entry"
        for entry in entries:
            assert entry.source_refs, "an entry with no source reference cannot be verified"
            for ref in entry.source_refs:
                assert isinstance(ref, weakref.ReferenceType)

    def test_a_dead_source_misses_instead_of_answering(self, clean_cache):
        """The failure mode the weak reference exists to prevent, forced directly: an entry
        whose source has been collected must decline, not serve a table that is gone."""
        with config_context(_FAST):
            ds = bt.from_arrow(_table([1, 2, 3]))
            ds.select("a").collect()
        key, entry = next(iter(prepared._CACHE.items()))
        live = ds._sources
        # `entry.config`, not `active_config()`: `with_auto_config` runs every terminal op
        # under a *resolved* config, so that -- not the ambient one -- is the object the
        # entry captured and the object a real hit is compared against.
        assert prepared.lookup(key, live, entry.config) is entry
        gone = _table([9])
        dead = weakref.ref(gone)
        del gone
        gc.collect()
        object.__setattr__(entry, "source_refs", (dead,))
        assert prepared.lookup(key, live, entry.config) is None


class TestTheResultMatchesTheOrdinaryPath:
    """A cached run and an uncached one are the same query; a divergence is a defect."""

    @staticmethod
    def _ordinary_then_cached(build):
        """`build()` with the cache cold, then the ordinary path, then cache-warm."""
        prepared.clear()
        with config_context(_FAST):
            cold = build()
            warm = build()
        ordinary = build()
        return ordinary, cold, warm

    def test_a_grouped_aggregate_over_nulls_agrees(self, clean_cache):
        frame = bt.from_pydict(
            {
                "grp": [i % 5 for i in range(120)],
                "v": [None if i % 11 == 0 else float(i) for i in range(120)],
            }
        )
        ordinary, cold, warm = self._ordinary_then_cached(
            lambda: (
                frame.group_by("grp").agg(s=bt.col("v").sum(), c=bt.col("v").count()).to_pydict()
            )
        )
        assert ordinary == cold == warm

    def test_a_descending_sort_agrees_in_order(self, clean_cache):
        """Ordered comparison on purpose: an order-independent one could not see a sort bug."""
        frame = bt.from_pydict({"id": list(range(80))})
        ordinary, cold, warm = self._ordinary_then_cached(
            lambda: frame.sort("id", descending=True).limit(20).to_pydict()["id"]
        )
        assert ordinary == cold == warm
        assert warm == sorted(warm, reverse=True)

    def test_an_empty_result_keeps_its_schema_on_a_hit(self, clean_cache):
        """The empty branch builds its schema from a field the entry precomputes, so a
        cached empty result is exactly where a lost column list would show up."""
        frame = bt.from_pydict({"id": list(range(30)), "v": [1.0] * 30})
        ordinary, cold, warm = self._ordinary_then_cached(
            lambda: frame.filter(bt.col("id") < 0).select("id", "v").collect()
        )
        assert warm.num_rows == 0
        assert warm.schema == cold.schema == ordinary.schema


class TestConcurrentPipelinesShareItSafely:
    """The cache is a process singleton, so several pipelines record into it at once. Each
    `OrderedDict` operation is atomic under the GIL but the *sequences* are not: a
    `move_to_end` on a key another thread just evicted raises into the query."""

    def test_many_threads_over_an_overflowing_cache_neither_raise_nor_lie(self):
        """Deliberately more distinct queries than the cache holds, so eviction races the
        promotion on every call. Each thread checks its own answer, so a mis-served entry
        fails as a wrong value rather than only as an exception."""
        import sys
        import threading

        prepared.clear()
        errors: list[BaseException] = []
        wrong: list[tuple[int, list[int]]] = []
        original = sys.getswitchinterval()
        sys.setswitchinterval(1e-9)  # force preemption inside the LRU bookkeeping
        try:

            def worker(base: int) -> None:
                try:
                    with config_context(_FAST):
                        for i in range(base, base + 120):
                            got = _select_a(_table([i, i + 1]))
                            if got != [i, i + 1]:
                                wrong.append((i, got))
                except BaseException as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(t * 1000,)) for t in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=180)
        finally:
            sys.setswitchinterval(original)

        assert not errors, f"the cache raised into a query: {errors[0]!r}"
        assert not wrong, f"the cache served another query's rows: {wrong[:3]}"
        assert len(prepared._CACHE) <= prepared.MAX_ENTRIES
