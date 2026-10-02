"""An adaptive stage's intermediate must not write into the cross-query learned store.

A stage boundary hands the next stage its intermediate wrapped as an `InMemorySource`, and
an in-memory source is keyed by **object identity** (its `identity()` is only shape-based,
so two different relations would collide on it). That object dies with the execution, so
every statistic filed under its key is filed under a name no later query can utter.

Recording them anyway cost three separate things, and the third is the expensive one:

1. the sketch was recomputed on every run — TPC-H Q8 at sf10 re-sketched 807k rows per
   `collect`, and 280M rows on the first;
2. the learned store grew by one dead ``obj:<id>`` entry per execution, without bound;
3. because a column absent from the store is by definition "measured for the first time",
   `record_column_stats` advanced the learned **generation** every single execution — and
   the generation is part of the plan cache's key, so the cache never once hit and every
   run re-planned from scratch. Measured: Q8 spent 130 ms and Q2 50 ms in Kyber per
   execution, against DuckDB's 84 ms and 46 ms for the *entire query*.

The same argument disqualifies the plan cache from storing a plan built over such a source:
the entry could never be read again, it evicts one that would have hit, and `store` pins
the source tuple alive — so the stage's whole materialized intermediate stays resident.
The exception is a stage source named by its *derivation* (`plan.source_stats.derivation_key`
of the stage's subplan over its inputs' keys): the next run of the query derives the same
relation and asks for the same key, so the plan over it is cached, and `store` does not pin it.
It is still skipped by every statistics writer, on `ephemeral` alone.

These tests pin the marker, the two writers that must honor it, and — because a learner that
learns nothing is a different bug — that an ordinary registered relation still gets measured.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import kyber
from batcher.api.adaptive.staging import _stage_source
from batcher.api.terminal._metadata import learn_column_stats, seed_column_ndv
from batcher.io import InMemorySource
from batcher.kyber import plan_cache
from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend
from batcher.plan.expr_ir import col


def _table() -> pa.Table:
    return pa.table({"k": [1, 2, 3, 4] * 25, "v": [10, 20, 30, 40] * 25})


def _ndv_entries(hub: MetadataHub) -> dict:
    return kyber.load_learned_stats(hub).get(kyber.NDV_KEY) or {}


def _avg_bytes_entries(hub: MetadataHub) -> dict:
    return kyber.load_learned_stats(hub).get(kyber.AVG_BYTES_KEY) or {}


# --------------------------------------------------------------------------------------
# The marker
# --------------------------------------------------------------------------------------


def test_an_ordinary_in_memory_source_is_not_ephemeral() -> None:
    assert InMemorySource(_table().to_batches()).ephemeral is False


def test_a_stage_boundary_marks_its_intermediate_ephemeral() -> None:
    """`_stage_source` is the one place a per-execution relation becomes a source."""
    source, _schema = _stage_source(_table())
    assert source.ephemeral is True


def test_a_stage_boundary_still_reports_its_exact_row_count() -> None:
    """The marker suppresses *learning*, not the measurement re-optimization reads."""
    source, _schema = _stage_source(_table())
    assert source.row_count() == 100


# --------------------------------------------------------------------------------------
# The two writers
# --------------------------------------------------------------------------------------


def test_seeding_skips_an_ephemeral_source() -> None:
    hub = MetadataHub(InProcessBackend())
    stage, _schema = _stage_source(_table())
    plan = bt.from_arrow(_table()).filter(col("k") == 1)._plan

    seed_column_ndv(hub, [stage], plan)

    assert _ndv_entries(hub) == {}, "a stage intermediate was sketched into the learned store"


def test_seeding_still_measures_an_ordinary_resident_source() -> None:
    """The counterpart: suppressing the ephemeral case must not suppress the real one."""
    hub = MetadataHub(InProcessBackend())
    ds = bt.from_arrow(_table()).filter(col("k") == 1)

    seed_column_ndv(hub, ds._sources, ds._plan)

    measured = {str(q).rsplit("\x1f", 1)[-1] for q in _ndv_entries(hub)}
    assert "k" in measured


def test_the_post_run_learner_skips_an_ephemeral_source() -> None:
    hub = MetadataHub(InProcessBackend())
    stage, _schema = _stage_source(_table())
    plan = bt.from_arrow(_table()).filter(col("k") == 1)._plan

    learn_column_stats(hub, [_table().to_batches()], [stage], plan)

    assert _avg_bytes_entries(hub) == {}, "a stage intermediate was sketched after the run"


def test_the_post_run_learner_still_measures_an_ordinary_source() -> None:
    hub = MetadataHub(InProcessBackend())
    ds = bt.from_arrow(_table()).filter(col("k") == 1)

    learn_column_stats(hub, [_table().to_batches()], ds._sources, ds._plan)

    measured = {str(q).rsplit("\x1f", 1)[-1] for q in _avg_bytes_entries(hub)}
    assert "k" in measured


def test_seeding_an_ephemeral_source_does_not_advance_the_generation() -> None:
    """The generation is the plan cache's key; advancing it every run empties the cache."""
    hub = MetadataHub(InProcessBackend())
    stage, _schema = _stage_source(_table())
    plan = bt.from_arrow(_table()).filter(col("k") == 1)._plan

    before = kyber.learning.generation()
    seed_column_ndv(hub, [stage], plan)
    seed_column_ndv(hub, [stage], plan)

    assert kyber.learning.generation() == before


# --------------------------------------------------------------------------------------
# The plan cache
# --------------------------------------------------------------------------------------


def test_no_plan_is_cached_over_an_ephemeral_source() -> None:
    hub = MetadataHub(InProcessBackend())
    stage, _schema = _stage_source(_table())
    from batcher.config import active_config

    key = plan_cache.cache_key("plan-fingerprint", [stage], active_config(), hub)

    assert key is None, "a plan keyed by a source that dies with the execution was cached"


def test_a_plan_over_an_ordinary_source_is_still_cached() -> None:
    hub = MetadataHub(InProcessBackend())
    from batcher.config import active_config

    source = InMemorySource(_table().to_batches())
    key = plan_cache.cache_key("plan-fingerprint", [source], active_config(), hub)

    assert key is not None


def test_one_ephemeral_source_disqualifies_a_mixed_plan() -> None:
    """A plan is only reusable if *every* source it was chosen over can recur."""
    hub = MetadataHub(InProcessBackend())
    from batcher.config import active_config

    stage, _schema = _stage_source(_table())
    ordinary = InMemorySource(_table().to_batches())
    key = plan_cache.cache_key("plan-fingerprint", [ordinary, stage], active_config(), hub)

    assert key is None


def test_a_stage_named_by_its_derivation_is_cached_and_not_pinned() -> None:
    """A derived stage source recurs by name, so the plan over it can be read back."""
    hub = MetadataHub(InProcessBackend())
    from batcher.config import active_config
    from batcher.kyber.plan_cache import memo

    keys = []
    for _ in range(2):
        stage, _schema = _stage_source(_table(), "the-same-derivation")
        keys.append(plan_cache.cache_key("plan-fingerprint", [stage], active_config(), hub))
    assert keys[0] is not None
    assert keys[0] == keys[1], "two runs deriving the same stage must ask for the same plan"

    memo.store(keys[0], "a-plan", [stage], max_entries=8)
    try:
        exact, _ = memo._split(keys[0])
        assert memo._CACHE[exact][1] == (), "a derived intermediate was pinned by the cache"
        assert plan_cache.lookup(keys[1]) == "a-plan"
    finally:
        plan_cache.clear()


def test_a_staged_query_replans_no_stage_once_warm(monkeypatch) -> None:
    """End to end: every stage's optimize hits the plan cache on a repeated staged query.

    The control is the first run, which must miss for every stage; without it a plan cache
    disabled by configuration would pass the warm half by never being asked.
    """
    from batcher.api.adaptive import staging
    from batcher.kyber.optimizer import facade

    plan_cache.clear()
    outcomes: list[bool] = []
    real_lookup = facade.plan_cache.lookup

    def spy(key, holds=None):
        hit = real_lookup(key, holds)
        outcomes.append(hit is not None)
        return hit

    monkeypatch.setattr(facade.plan_cache, "lookup", spy)
    monkeypatch.setattr(staging, "_worth_staging", lambda *_: lambda _node: True)
    fact = bt.from_pydict({"k": [i % 50 for i in range(5_000)], "v": list(range(5_000))})
    dim = bt.from_pydict({"k": list(range(50)), "w": [i * 3 for i in range(50)]})
    query = fact.group_by("k").agg(s=col("v").sum()).join(dim, on="k")
    runs = []
    for _ in range(6):
        outcomes.clear()
        rows = query.collect(adaptive=True).sort_by("k").to_pydict()
        runs.append(list(outcomes))
    assert rows["s"][:2] == [sum(range(0, 5_000, 50)), sum(range(1, 5_000, 50))]
    assert len(runs[0]) >= 2 and not any(runs[0]), f"the first run must plan every stage: {runs}"
    assert runs[-1] and all(runs[-1]), f"a warm staged query re-planned a stage: {runs}"
