"""Task granularity: a shuffle's map stage cuts finer than one partition per node.

One map partition per worker makes the *node* the unit of scheduling and of recovery, and
both costs are then charged whole: a worker running at half speed still holds a full
partition and the barrier waits on it, and a worker that dies loses a full partition that
one survivor replays end to end. `map_partitions` sizes a finer unit and `map_barrier`
deals it out as actors go idle.

The properties pinned here are the ones a wrong answer would hide behind:

* every source runs exactly once and every result lands at its own index — an
  index-addressed assembly means a mis-dealt source is a silently wrong shuffle, not an
  error;
* the in-flight count is exactly the actor pool and no wider — a wider window queues
  sources behind a busy actor and hands the assignment back to arrival order, which is the
  static dealing this exists to avoid. The pool is `workers x map_slots_per_worker()`: a
  map task reads its partition from object storage and then folds it, so one slot per actor
  left a node doing neither while doing the other, and the barrier now deals each actor
  several. What is pinned is the *equality* — the window tracks the pool — not the number;
* a slow actor takes fewer sources (the whole point) and a dead one's sources are re-dealt
  across survivors rather than replayed onto one;
* `SourcePlacement` knows where each source actually landed. Recovery is driven by *worker*
  death and has to answer "what did that lose", which stops being "the source with its id"
  the moment there is more than one source per worker.
"""

from __future__ import annotations

import collections
import contextlib

import pytest

from _fake_ray import install_fake_ray
from batcher.carbonite.resilience import RecoveryPolicy, SourcePlacement
from batcher.config import Config, DistributedConfig, config_context
from batcher.plan.resource import SchedulingEnvelope


def _raise(exc: BaseException):
    raise exc


def _multiplier(m: int, cap: int = 2048):
    return config_context(
        Config().replace(
            distributed=DistributedConfig(map_partition_multiplier=m, max_shuffle_partitions=cap)
        )
    )


#: The three envelope shapes `_idle_pool` distinguishes, by the `worker_cpus` each carries.
#: A homogeneous cluster produces the empty tuple; a per-node fleet produces one grant per
#: worker, which `_idle_pool` still deals evenly when they are all the same.
_UNIFORM: tuple[float, ...] = ()
_UNIFORM_SPELLED_OUT = (4.0,) * 8
_UNEQUAL = (4.0, 4.0, 4.0, 4.0, 96.0, 96.0, 96.0, 96.0)


@contextlib.contextmanager
def _fleet(cpus: tuple[float, ...]):
    """Install an ambient envelope whose fleet holds `cpus` as its per-worker core grants."""
    from batcher.dist.executors.ray_runtime.scheduling import (
        reset_scheduling_envelope,
        set_scheduling_envelope,
    )

    env = SchedulingEnvelope(n_tasks=8, worker_cpus=cpus)
    token = set_scheduling_envelope(env)
    try:
        yield env
    finally:
        reset_scheduling_envelope(token)


# --- the policy ------------------------------------------------------------------


def test_map_partitions_scales_with_the_multiplier():
    from batcher.dist.executors.ray_runtime import map_partitions

    with _multiplier(4):
        assert map_partitions(8) == 32


def test_map_partitions_of_one_pins_the_old_per_worker_unit():
    from batcher.dist.executors.ray_runtime import map_partitions

    with _multiplier(1):
        assert map_partitions(8) == 8


def test_map_partitions_never_drops_below_the_worker_count():
    # The count is also the parallelism floor: fewer partitions than workers would idle
    # workers for the whole map phase, which no multiplier may cause.
    from batcher.dist.executors.ray_runtime import map_partitions

    with _multiplier(4, cap=2):
        assert map_partitions(8) == 8


def test_map_partitions_respects_the_shuffle_cap():
    # The exchange opens `mappers x reducers` streams and this is the first factor, so it
    # is bounded for the same O(nodes²) reason the reduce side is.
    from batcher.dist.executors.ray_runtime import map_partitions

    with _multiplier(4, cap=100):
        assert map_partitions(64) == 100


def test_a_multiplier_that_can_still_buy_a_redeal_survives_a_uniform_fleet():
    """Raising the multiplier past the slot count keeps it, even on a uniform fleet.

    The collapse exists because at the shipped defaults the multiplier buys nothing on a
    uniform fleet: `workers x 4` partitions against a `workers x 4`-deep pool leaves the
    barrier nothing to re-deal. That is a fact about *two* numbers, and the first version of
    this rule read only one of them — it asked whether the fleet was uniform and collapsed
    regardless of how the multiplier compared to the slot count.

    So a cluster configured with `map_partition_multiplier=8` against 4 slots per worker would
    have had a reachable re-deal taken away from it, on exactly the fleets where the barrier is
    widest. This pins the boundary rather than the default: at or below the slot count the
    multiplier is spent, above it the multiplier is kept.
    """
    from batcher.dist.executors.ray_runtime.reducers import _map_slots, map_partitions

    slots = _map_slots()
    assert slots >= 1, "the slot count must be readable, or the rule below is vacuous"
    # At the slot count: nothing to re-deal, so the multiplier collapses.
    with _multiplier(slots), _fleet(_UNIFORM):
        assert map_partitions(8) == 8
    # One above it: a re-deal becomes reachable, so the multiplier is kept.
    with _multiplier(slots + 1), _fleet(_UNIFORM):
        assert map_partitions(8) == 8 * (slots + 1)


def test_a_uniform_fleet_spends_no_multiplier():
    # The multiplier buys the barrier's dynamic re-deal, and a uniform fleet's initial deal
    # is already `workers x map_slots_per_worker()` deep, so there is nothing left to
    # re-deal. What it does still cost is the exchange's `mappers x reducers` stream count.
    from batcher.dist.executors.ray_runtime import map_partitions

    with _multiplier(4), _fleet(_UNIFORM):
        assert map_partitions(8) == 8


def test_an_unequal_fleet_keeps_its_multiplier():
    # The positive control for the test above: on a fleet whose workers hold different core
    # grants, `_idle_pool` weights the deal by cores and the extra partitions are what it
    # weights. Nothing here may change for those.
    from batcher.dist.executors.ray_runtime import map_partitions

    with _multiplier(4), _fleet(_UNEQUAL):
        assert map_partitions(8) == 32


def test_a_fleet_whose_equal_grants_are_spelled_out_is_still_uniform():
    # `_idle_pool` deals evenly on an empty `worker_cpus` AND on a flat one (`max == min`), so
    # both are fleets the multiplier buys nothing on. Reading only the empty case would leave a
    # per-node fleet of identical machines paying for a re-deal that cannot happen there either.
    from batcher.dist.executors.ray_runtime import map_partitions

    with _multiplier(4), _fleet(_UNIFORM_SPELLED_OUT):
        assert map_partitions(8) == 8


def test_no_fleet_at_all_keeps_the_multiplier():
    # The second control. "No ambient envelope" is not "a uniform fleet" — it is a caller
    # outside a distributed execution, which knows nothing about the deal and must not be
    # narrowed on a guess.
    from batcher.dist.executors.ray_runtime import map_partitions
    from batcher.dist.executors.ray_runtime.scheduling import current_envelope

    assert current_envelope() is None
    with _multiplier(4):
        assert map_partitions(8) == 32


# --- the barrier -----------------------------------------------------------------


def test_over_partitioned_barrier_runs_every_source_exactly_once(monkeypatch):
    from batcher.dist.executors.ray_runtime import map_barrier

    install_fake_ray(monkeypatch)
    runs: collections.Counter = collections.Counter()

    def launch(host: int, src: int):
        runs[src] += 1
        return lambda h=host, s=src: f"addr{h}/{s}"

    addrs, dead = map_barrier(12, launch, RecoveryPolicy(max_attempts=3), workers=4)

    assert dead == set()
    assert len(addrs) == 12
    assert all(runs[s] == 1 for s in range(12))
    # Index-addressed: `addrs[src]` is src's own result, never a neighbour's.
    assert all(addrs[s].endswith(f"/{s}") for s in range(12))


def test_over_partitioned_barrier_keeps_the_window_equal_to_the_actor_pool(monkeypatch):
    # The actor pool is the window. If MORE than the pool were submitted at once, sources
    # would queue behind a busy actor and the dealing would be static again; if fewer, the
    # slots would be unusable. The pool used to be `workers` and is now
    # `workers x map_slots_per_worker()` -- this asserts the relationship, not the constant,
    # so it keeps discriminating if the slot count is ever retuned.
    from batcher.dist.executors.ray_runtime import map_barrier
    from batcher.dist.executors.ray_runtime.scheduling import map_slots_per_worker

    install_fake_ray(monkeypatch)
    inflight, peak = [], []

    def launch(host: int, src: int):
        inflight.append(src)
        peak.append(len(inflight))

        def _run():
            inflight.remove(src)
            return f"addr{host}"

        return _run

    workers, sources = 4, 64
    pool = workers * map_slots_per_worker()
    map_barrier(sources, launch, RecoveryPolicy(max_attempts=3), workers=workers)

    assert max(peak) <= pool, "the window outgrew the pool: sources queue behind a busy actor"
    assert max(peak) == pool, "the window never filled the pool: slots left unusable"


def test_over_partitioned_barrier_books_an_actor_exactly_its_slot_count(monkeypatch):
    """An actor takes several sources at once, and never more than it was spawned to hold.

    This test used to assert an actor is *never* double-booked, which was the contract when
    the barrier dealt one source per actor. It deals `map_slots_per_worker()` now, on a
    measurement -- the map phase of a 64-worker TPC-H sf100 scan held the cluster at 7-14% of
    its cores while every actor waited on a single S3 read, and overlapping them took the map
    barrier from 1,247 ms to 157 ms. So the bound moved; it did not go away, and the ceiling
    is what the actors' own `max_concurrency` was set to. Anything above it is not overlap,
    it is queueing inside Ray.
    """
    from batcher.dist.executors.ray_runtime import map_barrier
    from batcher.dist.executors.ray_runtime.scheduling import map_slots_per_worker

    install_fake_ray(monkeypatch)
    slots = map_slots_per_worker()
    live: collections.Counter[int] = collections.Counter()
    peak_per_host: collections.Counter[int] = collections.Counter()

    def launch(host: int, src: int):
        live[host] += 1
        peak_per_host[host] = max(peak_per_host[host], live[host])

        def _run():
            live[host] -= 1
            return f"addr{host}"

        return _run

    map_barrier(80, launch, RecoveryPolicy(max_attempts=3), workers=5)

    assert max(peak_per_host.values()) <= slots, "an actor was booked past its concurrency"
    if slots > 1:
        assert max(peak_per_host.values()) > 1, "no actor ever overlapped two sources"


def test_a_dead_worker_s_sources_spread_across_survivors(monkeypatch):
    # The granularity payoff under failure: four partitions belonged to the dead worker and
    # they are re-dealt, not replayed as one lump onto whichever survivor is picked first.
    from batcher.dist.executors.ray_runtime import map_barrier

    RayError, _ = install_fake_ray(monkeypatch)

    def launch(host: int, src: int):
        if host == 1:
            return lambda: _raise(RayError("preempted"))
        return lambda: f"addr{host}"

    addrs, dead = map_barrier(16, launch, RecoveryPolicy(max_attempts=4), workers=4)

    assert dead == {1}
    assert len(addrs) == 16 and all(a is not None for a in addrs)
    assert all(a != "addr1" for a in addrs)  # nothing was left on the dead worker


def test_the_barrier_records_where_each_source_landed(monkeypatch):
    # What recovery reads. `sources_on(host)` must name every source the host holds,
    # because that — not the source id — is what the host's death loses.
    from batcher.dist.executors.ray_runtime import map_barrier

    install_fake_ray(monkeypatch)
    placement = SourcePlacement(3)

    def launch(host: int, src: int):
        return lambda: f"addr{host}"

    map_barrier(9, launch, RecoveryPolicy(), workers=3, placement=placement)

    held = [placement.sources_on(h) for h in range(3)]
    assert sorted(s for group in held for s in group) == list(range(9))
    assert all(placement.host_of(src) in range(3) for src in range(9))


def test_one_source_per_worker_still_pins_host_to_src(monkeypatch):
    # The unchanged path: with as many sources as workers the barrier deals `host == src`,
    # which is what every existing caller and its recovery arithmetic assume.
    from batcher.dist.executors.ray_runtime import map_barrier

    install_fake_ray(monkeypatch)
    seen: list[tuple[int, int]] = []

    def launch(host: int, src: int):
        seen.append((host, src))
        return lambda: f"addr{host}"

    map_barrier(4, launch, RecoveryPolicy())

    assert seen == [(i, i) for i in range(4)]


# --- the placement record ---------------------------------------------------------


def test_placement_seeded_with_an_initial_assignment():
    placement = SourcePlacement(2, hosts=[0, 1, 0, 1])

    assert placement.host_of(2) == 0
    assert placement.sources_on(0) == {0, 2}
    assert placement.sources_on(1) == {1, 3}


def test_placement_relocation_moves_a_seeded_source():
    placement = SourcePlacement(2, hosts=[0, 1, 0, 1])
    placement.relocate(2, 1)

    assert placement.host_of(2) == 1
    assert placement.sources_on(0) == {0}
    assert placement.sources_on(1) == {1, 2, 3}


def test_unseeded_placement_is_unchanged():
    # No initial assignment ⇒ the sparse "source s lives on worker s" form, which is what
    # every one-partition-per-worker caller relies on.
    placement = SourcePlacement(3)

    assert placement.host_of(2) == 2
    assert placement.sources_on(2) == {2}
    placement.relocate(2, 0)
    assert placement.sources_on(2) == set()
    assert placement.sources_on(0) == {0, 2}


# --- the descriptors --------------------------------------------------------------


def test_max_partitions_is_a_ceiling_not_a_target():
    # A source with fewer splits than the ceiling yields fewer partitions rather than
    # empty tasks — each of which would still cost a task and a full set of empty bucket
    # publishes, i.e. the cost of fine granularity with none of the benefit.
    import pyarrow as pa

    import batcher as bt
    from batcher.dist.executors.partition_io import partition_descriptors

    ds = bt.from_arrow(pa.table({"a": list(range(64))}))
    parts = partition_descriptors(ds._sources[0], 2, max_partitions=32)

    assert 2 <= len(parts) <= 32


@pytest.mark.parametrize("workers", [1, 3, 8])
def test_descriptors_default_to_one_per_worker(workers):
    import pyarrow as pa

    import batcher as bt
    from batcher.dist.executors.partition_io import partition_descriptors

    ds = bt.from_arrow(pa.table({"a": list(range(64))}))

    assert len(partition_descriptors(ds._sources[0], workers)) == workers
