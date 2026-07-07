"""The shuffle partition count is capped so an all-to-all exchange stays bounded at scale.

Leaving the reducer count equal to the worker fan-out (one per node) makes the exchange
O(nodes^2) — ~100M mapper->reducer streams at 10k nodes. `shuffle_partitions` caps it at
`distributed.max_shuffle_partitions`, so regular clusters are unchanged and huge clusters
stay bounded. The mergeable algebra makes any reducer count result-correct (that equivalence
is covered on a live cluster in the distributed integration suite).
"""

from __future__ import annotations

import dataclasses

import pytest

from batcher.config import active_config, config_context
from batcher.dist.executors.ray_runtime import shuffle_partitions

pytestmark = pytest.mark.unit


def test_regular_cluster_is_unchanged():
    # A cluster at or below the cap keeps one reducer per worker (no behavior change).
    assert shuffle_partitions(8) == 8
    assert shuffle_partitions(2048) == 2048


def test_huge_cluster_is_capped():
    # 10k nodes -> capped, so the exchange is 10k * cap, not 10k * 10k.
    cap = active_config().distributed.max_shuffle_partitions
    assert shuffle_partitions(10_000) == cap
    assert cap < 10_000


def test_cap_is_configurable_and_disablable():
    base = active_config()
    with config_context(
        base.replace(distributed=dataclasses.replace(base.distributed, max_shuffle_partitions=64))
    ):
        assert shuffle_partitions(1000) == 64
    with config_context(
        base.replace(distributed=dataclasses.replace(base.distributed, max_shuffle_partitions=0))
    ):
        assert shuffle_partitions(10_000) == 10_000  # 0 disables the cap


def test_at_least_one_partition():
    assert shuffle_partitions(0) == 1
