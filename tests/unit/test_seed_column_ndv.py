"""`seed_column_ndv` measures a resident source's distinct counts before the optimizer runs.

`ndv` is the one statistic no file footer carries, so without seeding a query's *first*
execution orders its joins blind. These pin the contract: the sketch is parallel-merged
but must agree with the sequential HLL, it lands in the `SKETCH`-provenance learned
channel (never `EXACT`, so it can never answer a `count_distinct`), it runs at most once
per column, it skips non-resident sources, and it respects the cell budget.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa
import pytest

import batcher as bt
from batcher import core, kyber
from batcher.api.terminal._metadata import ndv_columns, seed_column_ndv
from batcher.config import active_config, config_context
from batcher.io.source import InMemorySource
from batcher.metadata.backends import InProcessBackend
from batcher.metadata.hub import MetadataHub
from batcher.plan.source_stats import source_stats_key

pytestmark = pytest.mark.unit

pytest.importorskip("batcher._native", reason="native engine not built")


def _source(n: int = 50_000) -> InMemorySource:
    """A relation with a unique key, a 10-way key, and a constant column."""
    table = pa.table(
        {
            "uniq": pa.array(range(n), type=pa.int64()),
            "ten": pa.array([i % 10 for i in range(n)], type=pa.int64()),
            "const": pa.array(["x"] * n),
        }
    )
    return InMemorySource(table.to_batches())


def _hub() -> MetadataHub:
    return MetadataHub(InProcessBackend())


def _ndv(hub: MetadataHub, src: InMemorySource | None = None) -> dict[str, float]:
    """The distinct counts learned **for `src`** (every source's, when `src` is None).

    Column statistics are filed per source, so reading them back means asking about a
    source — a bare column name identifies nothing (two tables both have an `id`).
    """
    learned = kyber.load_learned_stats(hub)
    if src is None:
        return learned.get(kyber.NDV_KEY, {})
    return kyber.columns_for(learned, kyber.NDV_KEY, source_stats_key(src))


def test_seeds_distinct_counts_for_a_resident_source():
    hub = _hub()
    assert _ndv(hub) == {}
    src = _source()
    seed_column_ndv(hub, [src])
    ndv = _ndv(hub, src)
    assert ndv["uniq"] == pytest.approx(50_000, rel=0.03)  # HLL error budget
    assert ndv["ten"] == pytest.approx(10, rel=0.03)
    assert ndv["const"] == pytest.approx(1, rel=0.03)


def test_parallel_sketch_agrees_with_the_sequential_hll():
    """The rayon fold is over a `Mergeable` sketch, so it must be bit-identical."""
    import batcher._native as native

    batches = _source().read()
    parallel = core.column_ndv(batches, ["uniq", "ten", "const"])
    for name, value in parallel.items():
        assert value == native.estimate_distinct(name, batches)


def test_a_measured_column_is_never_re_sketched():
    """The second call is a hub read: `known` already covers every column.

    Sketching is O(rows); re-running it per query would put a scan back on the hot path.
    The source's `read` is made to raise, so a second sketch could not go unnoticed.
    """

    class NeverRead(InMemorySource):
        def read(self, projection=None):
            raise AssertionError("re-sketched an already-measured column")

    hub = _hub()
    seed_column_ndv(hub, [_source()])
    first = dict(_ndv(hub))
    assert first

    table = pa.table({name: pa.array([1], type=pa.int64()) for name in first})
    seed_column_ndv(hub, [NeverRead(table.to_batches())])
    assert _ndv(hub) == first


def test_non_resident_sources_are_skipped():
    """Re-reading a file-backed source just to sketch would double the query's I/O."""

    class FileBacked(InMemorySource):
        resident = False

    hub = _hub()
    table = pa.table({"a": pa.array([1, 2, 3], type=pa.int64())})
    seed_column_ndv(hub, [FileBacked(table.to_batches())])
    assert _ndv(hub) == {}


def test_cell_budget_refuses_an_oversized_source():
    hub = _hub()
    cfg = active_config()
    tiny = cfg.replace(optimizer=dataclasses.replace(cfg.optimizer, ndv_sketch_max_cells=1))
    with config_context(tiny):
        seed_column_ndv(hub, [_source()])
    assert _ndv(hub) == {}


def test_seeding_never_raises_on_a_broken_source():
    """Learning is best-effort: a measurement failure must not break a query."""

    class Broken:
        resident = True

        def schema(self):
            raise RuntimeError("boom")

    seed_column_ndv(_hub(), [Broken()])  # must not raise


# --- which columns are worth sketching -----------------------------------------


def _tables() -> bt.Session:
    session = bt.Session()
    session.register(
        "a", pa.table({"ak": pa.array([1]), "av": pa.array([1]), "tag": pa.array(["x"])})
    )
    session.register("b", pa.table({"bk": pa.array([1]), "bv": pa.array([1])}))
    return session


def test_join_keys_and_group_keys_are_sketched():
    session = _tables()
    plan = session.sql(
        "SELECT a.tag, sum(b.bv) s FROM a JOIN b ON a.ak = b.bk GROUP BY a.tag"
    )._plan
    assert {"ak", "bk", "tag"} <= ndv_columns(plan)


def test_equality_and_in_predicates_are_sketched():
    """`col = v` and `col IN (...)` use `1/ndv`, so their columns matter."""
    session = _tables()
    plan = session.sql("SELECT av FROM a WHERE ak = 3 AND tag IN ('x', 'y')")._plan
    assert {"ak", "tag"} <= ndv_columns(plan)


def test_range_only_columns_are_not_sketched():
    """A range predicate interpolates min/max; its `ndv` steers nothing."""
    session = _tables()
    plan = session.sql("SELECT av FROM a WHERE ak > 3")._plan
    assert "ak" not in ndv_columns(plan)


def test_seeding_restricted_to_relevant_columns():
    """Passing a plan sketches only its keys — the reason a 60M-row source fits the budget."""
    hub = _hub()
    src = _source()
    table = pa.table({"uniq": pa.array([1]), "ten": pa.array([1]), "const": pa.array(["x"])})
    session = bt.Session()
    session.register("t", table)
    plan = session.sql("SELECT const, sum(ten) s FROM t GROUP BY const")._plan
    seed_column_ndv(hub, [src], plan)
    measured = set(_ndv(hub, src))
    assert "const" in measured
    assert "uniq" not in measured  # never referenced by the plan
