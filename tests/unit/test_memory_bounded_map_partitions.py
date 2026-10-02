"""A join's map stage is cut fine enough that one task's input fits its share of a node.

On a uniform fleet `map_partitions` settles on one partition per worker, and a join mapper
holds its partition's bucketed output until it publishes. On 3 x m5.4xlarge at TPC-H SF1000
that was a third of `lineitem` per task, and q9's workers were OOM-killed.
`memory_bounded_map_partitions` raises the count by projected input; it must never lower it.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.ray_runtime import reducers, scaling

pytestmark = pytest.mark.unit

_GIB = 1 << 30


@pytest.fixture
def three_64g_workers(monkeypatch):
    monkeypatch.setattr(scaling, "worker_node_memory_bytes", lambda: 64 * _GIB)
    monkeypatch.setattr(reducers, "map_partitions", lambda workers: workers)


def test_a_side_too_large_for_one_task_per_worker_is_cut_finer(three_64g_workers):
    # q9's lineitem side at SF1000: ~288 GB projected, against 4 GiB per map task.
    n = reducers.memory_bounded_map_partitions(3, 288 * _GIB)
    assert n == 72  # 288 / 4, already a multiple of the three workers
    assert n % 3 == 0


def test_a_side_that_fits_keeps_the_scheduling_count(three_64g_workers):
    assert reducers.memory_bounded_map_partitions(3, 1 * _GIB) == 3
    assert reducers.memory_bounded_map_partitions(8, 29 * _GIB) == 8


def test_an_unknown_size_or_memory_keeps_the_scheduling_count(three_64g_workers, monkeypatch):
    assert reducers.memory_bounded_map_partitions(3, 0) == 3
    monkeypatch.setattr(scaling, "worker_node_memory_bytes", lambda: 0)
    assert reducers.memory_bounded_map_partitions(3, 288 * _GIB) == 3
