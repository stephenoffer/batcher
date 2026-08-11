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
* at most `workers` tasks are in flight, because the actor pool *is* the window — a wider
  one queues sources behind a busy actor and hands the assignment back to arrival order,
  which is the static dealing this exists to avoid;
* a slow actor takes fewer sources (the whole point) and a dead one's sources are re-dealt
  across survivors rather than replayed onto one;
* `SourcePlacement` knows where each source actually landed. Recovery is driven by *worker*
  death and has to answer "what did that lose", which stops being "the source with its id"
  the moment there is more than one source per worker.
"""

from __future__ import annotations

import collections

import pytest

from _fake_ray import install_fake_ray
from batcher.carbonite.resilience import RecoveryPolicy, SourcePlacement
from batcher.config import Config, DistributedConfig, config_context


def _raise(exc: BaseException):
    raise exc


def _multiplier(m: int, cap: int = 2048):
    return config_context(
        Config().replace(
            distributed=DistributedConfig(map_partition_multiplier=m, max_shuffle_partitions=cap)
        )
    )


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


def test_over_partitioned_barrier_keeps_one_task_per_worker_in_flight(monkeypatch):
    # The actor pool is the window. If more than `workers` were submitted at once, two
    # sources would queue on one actor and the dealing would be static again.
    from batcher.dist.executors.ray_runtime import map_barrier

    install_fake_ray(monkeypatch)
    inflight, peak = [], []

    def launch(host: int, src: int):
        inflight.append(src)
        peak.append(len(inflight))

        def _run():
            inflight.remove(src)
            return f"addr{host}"

        return _run

    map_barrier(16, launch, RecoveryPolicy(max_attempts=3), workers=4)

    assert max(peak) <= 4


def test_over_partitioned_barrier_never_double_books_an_actor(monkeypatch):
    from batcher.dist.executors.ray_runtime import map_barrier

    install_fake_ray(monkeypatch)
    busy: set[int] = set()
    conflicts: list[int] = []

    def launch(host: int, src: int):
        if host in busy:
            conflicts.append(host)
        busy.add(host)

        def _run():
            busy.discard(host)
            return f"addr{host}"

        return _run

    map_barrier(20, launch, RecoveryPolicy(max_attempts=3), workers=5)

    assert conflicts == []


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
