"""Unit tests for `dist.executor._even_cpu_share` — the distributed worker CPU grant.

The grant must be placeable on every node (capped at the *smallest* node, since workers
are SPREAD across nodes), never over-subscribe the cluster (`workers x grant <= total`),
and never drop below 1 (the historical default). It is deliberately uniform — data/compute
skew is handled by split LPT-balancing and join-key salting, not by uneven grants.
"""

from __future__ import annotations

import pytest

from batcher.dist import executor

pytestmark = pytest.mark.unit


def _nodes(*cpus):
    return [{"Alive": True, "Resources": {"CPU": float(c)}} for c in cpus]


@pytest.fixture
def fake_ray(monkeypatch):
    import ray

    def set_nodes(*cpus):
        monkeypatch.setattr(ray, "nodes", lambda: _nodes(*cpus))

    return set_nodes


def test_one_full_node_per_worker_on_uniform_cluster(fake_ray):
    fake_ray(0, 16, 16, 16, 16, 16, 16, 16, 16)  # head=0 + 8x16-core workers
    assert executor._even_cpu_share(8) == 16.0  # a full node each


def test_two_workers_per_node_halves_the_grant(fake_ray):
    fake_ray(0, 16, 16, 16, 16, 16, 16, 16, 16)
    assert executor._even_cpu_share(16) == 8.0


def test_grant_capped_at_smallest_node_so_every_bundle_places(fake_ray):
    # A heterogeneous cluster: a worker requesting 16 couldn't place on the 8-core node.
    fake_ray(8, 16, 16)
    assert executor._even_cpu_share(2) == 8.0  # min node, not 40//2=20 and not max 16


def test_never_oversubscribes(fake_ray):
    fake_ray(0, 16, 16)  # 32 schedulable cpus
    assert executor._even_cpu_share(8) == 4.0  # 32//8, well under the 16-core node cap


def test_falls_back_to_one_on_tiny_cluster(fake_ray):
    fake_ray(8)  # single 8-core node
    assert executor._even_cpu_share(16) == 1.0  # 8//16 -> 0 -> floored to 1, no change


def test_falls_back_to_one_when_no_cpu_nodes(fake_ray):
    fake_ray(0, 0)  # only the (0-CPU) head visible
    assert executor._even_cpu_share(4) == 1.0
