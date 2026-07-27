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
