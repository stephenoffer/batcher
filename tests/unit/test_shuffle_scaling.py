"""How many reducers an all-to-all exchange fans out to.

Two forces pull against each other here. Too few buckets and workers sit idle: a bucket is
reduced by exactly one worker, so fewer buckets than workers leaves the rest out of the
reduce phase entirely — which is what the reduce paths' `actors[bucket]` indexing used to
force. Too many and the exchange, which opens `mappers x reducers` streams, drowns in
per-fetch overhead; at one reducer per worker it is already O(nodes^2), ~100M streams at
10k nodes.

Note that buckets above the floor do *not* fix skew — a dominant key is indivisible by
hashing, and finer buckets only shrink the mean around it. That is salting's job.

So `shuffle_partitions` multiplies the worker count by a small factor and caps the result.
The mergeable algebra makes any reducer count result-correct (that equivalence is covered
on a live cluster in the distributed integration suite), so both knobs are purely about
scaling.
"""

from __future__ import annotations

import dataclasses

import pytest

from batcher.config import active_config, config_context
from batcher.dist.executors.ray_runtime import shuffle_partitions

pytestmark = pytest.mark.unit


def _with(**kw):
    base = active_config()
    return config_context(base.replace(distributed=dataclasses.replace(base.distributed, **kw)))


def test_a_cold_cluster_gets_one_bucket_per_worker():
    """No measured history: full worker parallelism at the smallest stream count.

    This is a floor as much as a default. A bucket is reduced by exactly one worker, so
    fewer buckets than workers idles the rest for the whole reduce phase.
    """
    assert shuffle_partitions(8) == 8
    assert shuffle_partitions(2048) == 2048


def test_the_multiplier_is_a_ceiling_not_a_target():
    """It bounds how far a measured volume may raise the count, and never lowers it."""
    with _with(shuffle_partition_multiplier=1):
        assert shuffle_partitions(8) == 8
    with _with(shuffle_partition_multiplier=8):
        assert shuffle_partitions(8) == 8  # cold: still one per worker, not 64


def test_huge_cluster_is_capped():
    # 10k nodes -> capped, so the exchange is 10k * cap, not 10k * 10k.
    cap = active_config().distributed.max_shuffle_partitions
    assert shuffle_partitions(10_000) == cap
    assert cap < 10_000


def test_cap_is_configurable_and_disablable():
    with _with(max_shuffle_partitions=64):
        assert shuffle_partitions(1000) == 64
    with _with(max_shuffle_partitions=0):
        assert shuffle_partitions(10_000) == 10_000  # 0 disables the cap


def test_at_least_one_partition():
    assert shuffle_partitions(0) == 1
