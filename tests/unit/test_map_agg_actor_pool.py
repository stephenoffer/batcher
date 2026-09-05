"""The warm pool a `map_batches -> aggregate` stage uses holds ONE pipeline at a time.

`_distributed_map_aggregate` ran stateless tasks, so the partition a worker read was almost
never the one it had cached; addressing persistent actors by index instead is worth 1.35-2.3x
warm on TPC-H sf100 (`benchmarks/BENCHMARK_RESULTS.md`). The residency policy is the part
that needs pinning, and it is not the inference registry's policy.

Two resident models are a feature: each holds the devices it needs and neither can use the
other's. Two resident CPU aggregate pools are a hazard: they hold general-purpose cores that
every other stage also wants, and a pool only earns them from the scan cache of the pipeline
that filled it. Left to accumulate, three pipelines in one session put 960 of 1024 cores under
reservation and the third stalled at the barrier with `0/256 tasks finished`. So `_AGG_POOLS`
is its own registry and a second pipeline evicts the first.

These run with no cluster: `_new_map_actor` is the single actor-creation point, so stubbing it
accounts for every actor the pool would have built.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.dist.executors import map as M

pytestmark = pytest.mark.unit

pytest.importorskip("ray", reason="the pool registry's teardown calls ray.kill")


class _FakeActor:
    """Stands in for a `_MapActor` handle; only its identity is under test."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.cpu_workers: int | None = None


@pytest.fixture
def pooling(monkeypatch):
    """A clean `_AGG_POOLS`, actor creation stubbed, and every kill recorded.

    `_AGG_POOL_WIDTH` is reset with it. It is module state that outlives a query by design
    (an actor's width is fixed in `__init__`, so the pool must notice a fleet that resized),
    and leaving it set leaks one test's fleet into the next one's eviction decision.
    """
    import ray

    monkeypatch.setattr(M, "_AGG_POOLS", {})
    monkeypatch.setattr(M, "_AGG_POOL_WIDTH", [])
    monkeypatch.setattr(M, "_agg_actor_width", lambda size: 4)
    built: list[_FakeActor] = []
    killed: list[_FakeActor] = []

    def new_actor(plan0, opts, cpu_workers=None):
        actor = _FakeActor(f"a{len(built)}")
        actor.cpu_workers = cpu_workers
        built.append(actor)
        return actor

    monkeypatch.setattr(M, "_new_map_actor", new_actor)
    monkeypatch.setattr(M, "_healthy_actors", lambda pool: list(pool))
    monkeypatch.setattr(ray, "kill", killed.append)
    return built, killed


def _prefix(fn):
    """The map prefix `_agg_actor_pool` is called with, built through the public API."""
    return bt.from_pydict({"a": [1.0, 2.0]}).map_batches(fn, output_columns=["a"])._plan


def test_a_cpu_stage_gets_one_actor_per_worker(pooling):
    """The positive control. Every assertion below is vacuous if nothing is ever pooled."""
    built, _ = pooling
    pool = M._agg_actor_pool(_prefix(lambda b: b), 4)

    assert pool is not None and len(pool) == 4
    assert len(built) == 4, "one actor per worker, built through the one creation point"


def test_the_same_pipeline_reuses_its_actors(pooling):
    """Residency is the whole point: a second run must not rebuild the workers."""
    built, killed = pooling
    fn = lambda b: b  # noqa: E731 - one identity, so both calls key the same pool
    prefix = _prefix(fn)

    first = M._agg_actor_pool(prefix, 3)
    second = M._agg_actor_pool(_prefix(fn), 3)

    assert first == second, "the same pipeline must meet the same actors"
    assert len(built) == 3 and killed == [], "no rebuild, no teardown"


def test_a_second_pipeline_evicts_the_first(pooling):
    """The invariant. Accumulating pools is what reserved 960 of 1024 cores and hung."""
    built, killed = pooling

    first = M._agg_actor_pool(_prefix(lambda b: b), 2)
    second = M._agg_actor_pool(_prefix(lambda b: b.slice(0, 1)), 2)

    assert len(M._AGG_POOLS) == 1, "at most one CPU aggregate pool is resident"
    assert killed == first, "the displaced pipeline's actors are killed, not orphaned"
    assert second is not None and set(second).isdisjoint(first)
    assert len(built) == 4


def test_a_gpu_stage_keeps_the_stateless_task_path(pooling, monkeypatch):
    """A GPU pool is sized and placed from device measurements; none of that applies here."""
    built, _ = pooling
    monkeypatch.setattr(M, "_map_resources", lambda plan: (1.0, True, None, None, {}))

    assert M._agg_actor_pool(_prefix(lambda b: b), 4) is None
    assert built == []


def test_warm_pools_off_keeps_the_stateless_task_path(pooling):
    """`warm_inference_pools` is the switch that says actors must not outlive a query."""
    import dataclasses

    from batcher.config import Config

    built, _ = pooling
    cfg = Config()
    cfg = dataclasses.replace(
        cfg, distributed=dataclasses.replace(cfg.distributed, warm_inference_pools=False)
    )
    with bt.config_context(cfg):
        assert M._agg_actor_pool(_prefix(lambda b: b), 4) is None
    assert built == []


def test_a_failure_acquiring_the_pool_falls_back_rather_than_failing_the_stage(
    pooling, monkeypatch
):
    """A pool is an optimisation. Losing it must cost speed, never the query."""
    monkeypatch.setattr(M, "_map_resources", lambda plan: (_ for _ in ()).throw(RuntimeError("no")))

    assert M._agg_actor_pool(_prefix(lambda b: b), 4) is None


def test_releasing_the_warm_pools_frees_the_aggregate_pool_too(pooling):
    """`release_inference_pools` is the documented way to get the cores back."""
    _, killed = pooling
    pool = M._agg_actor_pool(_prefix(lambda b: b), 2)

    M.release_inference_pools()

    assert killed == pool and M._AGG_POOLS == {}


def test_no_pool_goes_straight_to_the_stateless_tasks():
    """The route the change did not touch: no actors, no recovery, no eviction."""
    used: list[str] = []

    state = M._gather_with_pool_recovery(
        lambda launch: (used.append(launch), "state")[1], "actor", "task", object(), None
    )

    assert state == "state" and used == ["task"]


def test_a_healthy_pool_never_reaches_the_task_launcher():
    """The positive control for the test below: without it, "fell back" is unfalsifiable."""
    used: list[str] = []

    state = M._gather_with_pool_recovery(
        lambda launch: (used.append(launch), "actors")[1], "actor", "task", object(), ["a"]
    )

    assert state == "actors" and used == ["actor"]


def test_a_pool_that_dies_mid_stage_redoes_the_work_on_tasks(pooling):
    """A warm pool must never turn a preemption into a failed query (`_run_warm_pool`)."""
    from ray.exceptions import RayActorError

    prefix = _prefix(lambda b: b)
    pool = M._agg_actor_pool(prefix, 2)
    _, killed = pooling
    used: list[str] = []

    def gather(launch):
        used.append(launch)
        if launch == "actor":
            raise RayActorError  # what a preempted node's actor call actually raises
        return "recovered"

    state = M._gather_with_pool_recovery(gather, "actor", "task", prefix, pool)

    assert state == "recovered", "the stage completed on the fallback"
    assert used == ["actor", "task"], "it tried the pool first, then the tasks"
    assert killed == pool and M._AGG_POOLS == {}, "and the dead pool was evicted, not reused"


def test_each_actor_is_built_with_the_width_the_fleet_implies(pooling, monkeypatch):
    """The width is the lever: 4 threads against 16 measured 1,068 ms against 746 ms."""
    built, _ = pooling
    monkeypatch.setattr(M, "_agg_actor_width", lambda size: 16)

    M._agg_actor_pool(_prefix(lambda b: b), 3)

    assert [a.cpu_workers for a in built] == [16, 16, 16]


def test_a_resized_fleet_rebuilds_the_pool_rather_than_reusing_the_old_width(pooling, monkeypatch):
    """An actor fixes its width in `__init__`, so a pool built for the old fleet is stale."""
    built, killed = pooling
    fn = lambda b: b  # noqa: E731 - one identity, so only the width differs between calls
    first = M._agg_actor_pool(_prefix(fn), 2)

    monkeypatch.setattr(M, "_agg_actor_width", lambda size: 16)
    second = M._agg_actor_pool(_prefix(fn), 2)

    assert killed == first, "the old-width actors are killed, not reused"
    assert [a.cpu_workers for a in second] == [16, 16]
    assert len(built) == 4


def test_an_unchanged_fleet_does_not_rebuild(pooling):
    """The positive control: without it, the test above passes on a pool that always rebuilds."""
    built, killed = pooling
    fn = lambda b: b  # noqa: E731 - same identity, same width, so the pool must be reused

    first = M._agg_actor_pool(_prefix(fn), 2)
    second = M._agg_actor_pool(_prefix(fn), 2)

    assert first == second and killed == [] and len(built) == 2


def test_the_warm_pool_gives_its_cores_back_when_the_session_goes_idle(pooling, monkeypatch):
    """Single-tenancy stopped pools accumulating; it did not stop one holding the cluster.

    Measured on the 65-node fleet: after a `udf` query returned, `ray status` reported 960 of
    1,024 CPU in use for the life of the driver, and a Ray Data run started next in the same
    process could not schedule its final aggregate. The pool now takes the shuffle fleet's
    discipline and its knob (`distributed.session_fleet_idle_s`).
    """
    _, killed = pooling
    pool = M._agg_actor_pool(_prefix(lambda b: b), 2)

    M._arm_agg_idle_release()
    assert M._AGG_IDLE_TIMER, "a warm pool must have a pending release"
    M._release_agg_pool_if_idle()  # what the timer calls when it fires

    assert killed == pool and M._AGG_POOLS == {}


def test_a_stage_still_running_is_not_released_underneath_itself(pooling):
    """The lease. A query longer than the idle window must not lose its own actors."""
    _, killed = pooling
    M._agg_actor_pool(_prefix(lambda b: b), 2)

    with M._agg_pool_in_use():
        M._release_agg_pool_if_idle()  # the timer fires mid-stage
        assert killed == [], "the pool is in use, so it is not idle"

    assert M._AGG_IDLE_TIMER, "leaving the stage re-arms the clock"
    M._release_agg_pool_if_idle()
    assert M._AGG_POOLS == {}


def test_entering_a_stage_cancels_a_pending_release(pooling):
    """Back-to-back queries never see the timer at all."""
    M._agg_actor_pool(_prefix(lambda b: b), 2)
    M._arm_agg_idle_release()

    with M._agg_pool_in_use():
        assert not M._AGG_IDLE_TIMER, "the pending release is cancelled on entry"


def test_the_idle_release_is_switchable_off(pooling):
    """`session_fleet_idle_s <= 0` is the existing spelling of 'hold it'."""
    from batcher.config import Config, DistributedConfig, config_context

    M._agg_actor_pool(_prefix(lambda b: b), 2)
    cfg = Config().replace(distributed=DistributedConfig(session_fleet_idle_s=0.0))
    with config_context(cfg):
        M._arm_agg_idle_release()

    assert not M._AGG_IDLE_TIMER


def test_releasing_the_pools_drops_a_pending_timer(pooling):
    """Nothing should be left ticking against an empty registry."""
    M._agg_actor_pool(_prefix(lambda b: b), 2)
    M._arm_agg_idle_release()

    M.release_inference_pools()

    assert not M._AGG_IDLE_TIMER and M._AGG_POOLS == {}
