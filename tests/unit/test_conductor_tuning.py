"""Unit tests for the conductor's adaptive-tuning decisions (`batcher.api` wiring).

Each decision is checked twice: **cold** (an empty hub → the conductor keeps its static
default, so a first run is unchanged) and **warm** (seeded measured signals → the decision
moves to the learned value). These pin the *decision*; result-invariance (tuning on/off →
identical results) is proven in `tests/differential/test_diff_conductor_tuning.py`.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.kyber.signature import plan_signature
from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend

pytestmark = pytest.mark.unit


def _hub() -> MetadataHub:
    return MetadataHub(InProcessBackend())


# --- 1 + 2. resolve_adaptive learned router ⇄ record_adaptive_route ---------------------------
def test_resolve_adaptive_learned_router_cold_and_warm(monkeypatch):
    from batcher.api.adaptive import gating, resolve_adaptive
    from batcher.kyber.learned_tuning import record_adaptive_route

    left = bt.from_arrow(pa.table({"k": list(range(20)), "x": list(range(20))}))
    right = bt.from_arrow(pa.table({"k": [1, 2], "w": [9, 8]}))
    ds = left.join(right, on="k")
    hub = _hub()
    # Small inputs: the size floor refuses staging outright, before anything learned.
    assert resolve_adaptive("auto", ds._plan, ds._sources, hub) is False

    # Above the floor, a history where staging measured far cheaper turns it on.
    monkeypatch.setattr(gating, "_ADAPTIVE_MIN_INPUT_ROWS", 1)
    sig = plan_signature(ds._plan)
    for _ in range(4):
        record_adaptive_route(hub, sig, "staged", 10.0)
        record_adaptive_route(hub, sig, "one_shot", 1000.0)
    assert resolve_adaptive("auto", ds._plan, ds._sources, hub) is True


def test_the_size_floor_binds_the_learned_router_too(monkeypatch):
    """A plan signature is scale-blind, so a route learned at scale must not replay below it.

    `plan_signature` normalizes literals so statistics generalize across runs; the same query
    over sf1 and sf10 therefore shares a signature. Consulting the router before the size floor
    replayed sf10's routes at interactive scale and took TPC-H q8 from 18.8 ms to 181.9 ms.
    """
    from batcher.api.adaptive import gating, resolve_adaptive
    from batcher.kyber.learned_tuning import record_adaptive_route

    left = bt.from_arrow(pa.table({"k": list(range(20)), "x": list(range(20))}))
    right = bt.from_arrow(pa.table({"k": [1, 2], "w": [9, 8]}))
    ds = left.join(right, on="k")
    hub = _hub()
    sig = plan_signature(ds._plan)
    for _ in range(4):
        record_adaptive_route(hub, sig, "staged", 10.0)
        record_adaptive_route(hub, sig, "one_shot", 1000.0)

    # The very same history that turns staging on above the floor is ignored below it.
    monkeypatch.setattr(gating, "_ADAPTIVE_MIN_INPUT_ROWS", 1)
    assert resolve_adaptive("auto", ds._plan, ds._sources, hub) is True
    monkeypatch.setattr(gating, "_ADAPTIVE_MIN_INPUT_ROWS", 20_000_000)
    assert resolve_adaptive("auto", ds._plan, ds._sources, hub) is False


def test_resolve_adaptive_explicit_flag_ignores_learning():
    from batcher.api.adaptive import resolve_adaptive
    from batcher.kyber.learned_tuning import record_adaptive_route

    ds = bt.from_arrow(pa.table({"x": [1, 2, 3]})).filter(col("x") > 0)
    hub = _hub()
    for _ in range(4):
        record_adaptive_route(hub, plan_signature(ds._plan), "staged", 10.0)
        record_adaptive_route(hub, plan_signature(ds._plan), "one_shot", 1000.0)
    assert resolve_adaptive(False, ds._plan, ds._sources, hub) is False
    assert resolve_adaptive(True, ds._plan, ds._sources, hub) is True


# --- 3. auto_num_partitions seeded from measured shuffle rows --------------------------------
def test_auto_num_partitions_prefers_learned_count():
    from batcher.api.orchestration import auto_num_partitions
    from batcher.config import active_config
    from batcher.kyber.learned_tuning import record_partition_rows

    ds = (
        bt.from_arrow(pa.table({"k": [1, 2, 1, 2], "v": [1, 2, 3, 4]}))
        .group_by("k")
        .agg(s=col("v").sum())
    )
    hub = _hub()
    cold = auto_num_partitions(ds._plan, ds._sources, hub)

    target = active_config().optimizer.target_rows_per_task
    record_partition_rows(hub, plan_signature(ds._plan), float(target * 10))
    warm = auto_num_partitions(ds._plan, ds._sources, hub)
    assert warm == 10  # ceil(10*target / target), clamped into [4, 4096]
    assert cold != warm


# --- 8. resolve_distributed uses the learned per-signature size ------------------------------
@pytest.fixture
def _multinode(monkeypatch):
    class _Ray:
        @staticmethod
        def is_initialized():
            return True

    monkeypatch.setitem(__import__("sys").modules, "ray", _Ray)
    monkeypatch.setattr("batcher.dist.cluster_topology", lambda: {"nodes": 4}, raising=False)


class _UnknownSizeSource:
    """A source that cannot cheaply report a row count (→ distribute, absent other signal)."""

    def row_count(self) -> int | None:
        return None


def test_resolve_distributed_learned_small_stays_single_node(_multinode):
    from batcher import core, kyber
    from batcher.api.terminal.routing import resolve_distributed

    ds = bt.from_arrow(pa.table({"x": [1, 2, 3]})).filter(col("x") > 0)
    # Unknown source size → cold "auto" distributes (safe for large/unknown data).
    assert resolve_distributed("auto", ds._plan, [_UnknownSizeSource()]) is True
    # But once we've MEASURED this shape as tiny, it stays single-node (dodge the fan-out tax).
    kyber.record_execution(core.default_hub(), ds._plan, 100)
    assert resolve_distributed("auto", ds._plan, [_UnknownSizeSource()]) is False


def test_resolve_distributed_learned_large_distributes(_multinode):
    from batcher import core, kyber
    from batcher.api.terminal.routing import resolve_distributed
    from batcher.config import active_config

    ds = bt.from_arrow(pa.table({"x": [1, 2, 3]})).filter(col("x") > 0)
    big = active_config().distributed.distribute_min_rows * 5
    kyber.record_execution(core.default_hub(), ds._plan, big)
    assert resolve_distributed("auto", ds._plan, [_UnknownSizeSource()]) is True


# --- 9. learned_num_workers ------------------------------------------------------------------
def test_learned_num_workers_cold_and_warm():
    from batcher.api.tuning import learned_num_workers
    from batcher.config import active_config

    ds = bt.from_arrow(pa.table({"k": [1, 2], "v": [1, 2]})).group_by("k").agg(s=col("v").sum())
    hub = _hub()
    # No hub → the scheduler keeps its own default.
    assert learned_num_workers(None, ds._plan, ds._sources, 8) is None

    target = active_config().optimizer.target_rows_per_task
    from batcher import kyber

    kyber.record_execution(hub, ds._plan, target * 10)  # measured 10 workers' worth of data
    assert learned_num_workers(hub, ds._plan, ds._sources, 20) == 10
    assert learned_num_workers(hub, ds._plan, ds._sources, 4) == 4  # clamped to the cluster


# --- 5 + 6 + 7. join-outcome recording closes the bandit / crossover loops -------------------
def test_record_join_outcomes_feeds_the_bandit():
    from batcher.api.tuning import record_join_outcomes
    from batcher.kyber.learned_tuning import learned_build_sides, learned_join_strategy
    from batcher.kyber.rules.selection import BuildSideDecision

    left = bt.from_arrow(pa.table({"k": [1, 2, 3], "v": [1, 2, 3]}))
    right = bt.from_arrow(pa.table({"k": [1, 2], "w": [9, 8]}))
    joined = left.join(right, on="k")
    hub = _hub()
    dec = BuildSideDecision(left_rows=3.0, right_rows=2.0, swapped=False, provenance="exact")

    # Find the Join signature the reader keys on.
    from batcher.api.tuning.decisions import _all_joins

    join = _all_joins(joined._plan)[0]
    sig = plan_signature(join)

    assert learned_join_strategy(hub, sig) is None  # cold
    for _ in range(4):
        record_join_outcomes(hub, joined._plan, [dec], wall_ms=12.0)
    assert learned_join_strategy(hub, sig) is not None  # the bandit now has an arm
    assert learned_build_sides(hub, sig) == (3.0, 2.0)  # measured sides recorded


def test_record_join_outcomes_skips_multi_join_ambiguity():
    from batcher.api.tuning import record_join_outcomes
    from batcher.kyber.rules.selection import BuildSideDecision

    hub = _hub()
    a = bt.from_arrow(pa.table({"k": [1], "v": [1]}))
    b = bt.from_arrow(pa.table({"k": [1], "w": [2]}))
    c = bt.from_arrow(pa.table({"k": [1], "z": [3]}))
    multi = a.join(b, on="k").join(c, on="k")
    dec = BuildSideDecision(1.0, 1.0, False, "exact")
    # Two joins but a single (mismatched) decision list → recorded nothing (no mis-attribution).
    record_join_outcomes(hub, multi._plan, [dec], wall_ms=5.0)
    assert hub.load_keyed_params("tuning.join_arm") == {}


# --- 10 + 11. shuffle-window record ⇄ warm-start credit grant --------------------------------
def test_record_shuffle_outcome_warm_starts_the_credit_window():
    from batcher.api.tuning import record_shuffle_outcome
    from batcher.carbonite.policies import load_shuffle_window

    ds = bt.from_arrow(pa.table({"k": [1, 2], "v": [1, 2]})).group_by("k").agg(s=col("v").sum())
    hub = _hub()
    sig = plan_signature(ds._plan)
    assert load_shuffle_window(hub, sig) is None  # cold
    record_shuffle_outcome(hub, ds._plan, 48)
    assert load_shuffle_window(hub, sig) == 48


# --- 13. group-reduction recording feeds learned pre-aggregation -----------------------------
def test_record_run_feedback_records_group_reduction():
    from batcher.api.tuning import record_run_feedback
    from batcher.kyber.learned_tuning import learned_partial_agg

    tbl = pa.table({"k": [1, 1, 2], "v": [1, 2, 3]})
    ds = bt.from_arrow(tbl).group_by("k").agg(s=col("v").sum())
    hub = _hub()
    sig = plan_signature(ds._plan)
    assert learned_partial_agg(hub, sig) is None  # cold
    # 5 groups out of 1000 input rows → a strong collapse → pre-aggregation pays.
    record_run_feedback(hub, ds._plan, ds._plan, [], out_rows=5, input_rows=1000, wall_ms=3.0)
    assert learned_partial_agg(hub, sig) is True


# --- 4. spill-compression scope ---------------------------------------------------------------
class _StubRM:
    def __init__(self, verdict: bool | None):
        self._v = verdict

    def recommend_spill_compression(self, _plan):
        return self._v


def test_spill_compression_scope_applies_learned_codec():
    from batcher.api.tuning import spill_compression_scope
    from batcher.config import active_config

    base = active_config().memory.spill_compression
    with spill_compression_scope(_StubRM(True), None):
        assert active_config().memory.spill_compression == "zstd"
    with spill_compression_scope(_StubRM(False), None):
        assert active_config().memory.spill_compression is None
    with spill_compression_scope(_StubRM(None), None):
        assert active_config().memory.spill_compression == base  # un-sized → default kept
