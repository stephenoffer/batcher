"""The per-worker core grant is chosen to strand the fewest cores, not to fit the smallest node.

`min(node_cpus)` is right when nodes differ by a factor of two and pathological when one is
much smaller than the rest: a single 2-core utility node in a fleet of 64-core machines
pinned every worker to 2 cores, so each ran its scan and fold on a thirty-second of the node
it landed on. The smallest node set the shape of the whole cluster.

Nothing requires the grant to fit the smallest node. A grant a node cannot host means that
node hosts no workers, which costs its cores — obviously better than crippling every large
node's.
"""

from __future__ import annotations

import pytest

from batcher.dist.executor import _fill_grant

pytestmark = pytest.mark.unit


def test_homogeneous_cluster_grants_the_node_size():
    assert _fill_grant([16.0] * 8) == 16.0


def test_a_tiny_node_no_longer_pins_the_whole_fleet():
    """The pathology: one 2-core node among 64-core machines."""
    assert _fill_grant([2.0, 64.0, 64.0, 64.0]) == 64.0


def test_the_small_node_is_simply_unused_at_that_grant():
    """Losing one small node's cores beats crippling three large nodes."""
    grant = _fill_grant([2.0, 64.0, 64.0, 64.0])
    used = sum(int(c // grant) * grant for c in [2.0, 64.0, 64.0, 64.0])
    assert used == 192.0  # all three big nodes fully used, the 2-core node idle


def test_min_still_wins_when_it_is_genuinely_best():
    """A cluster that was already sized well keeps its exact previous behavior."""
    # 32 and 64: a 32-core grant uses 32 + 64 = 96 (everything); 64 uses only 64.
    assert _fill_grant([32.0, 64.0]) == 32.0


def test_a_factor_of_two_cluster_is_unchanged():
    assert _fill_grant([16.0, 32.0, 32.0]) == 16.0


def test_ties_prefer_the_larger_grant():
    """Fewer, fatter workers mean less shuffle fan-out for the same cores used."""
    # 8 and 16 both fully use [16, 16]: prefer 16.
    assert _fill_grant([16.0, 16.0, 8.0]) in (8.0, 16.0)
    assert _fill_grant([16.0, 16.0]) == 16.0


def test_non_integer_core_counts_are_floored():
    assert _fill_grant([15.5, 15.5]) == 15.0


def test_zero_and_negative_nodes_are_ignored():
    assert _fill_grant([0.0, 16.0, 16.0]) == 16.0


def test_no_usable_nodes_falls_back_to_one():
    assert _fill_grant([0.0, 0.0]) == 1.0


def test_the_grant_never_drops_below_one():
    assert _fill_grant([0.4, 0.4]) >= 1.0


class TestEvenShareAgreesWithTheFillGrant:
    """Both answer "how wide may a uniform worker be", so they must agree.

    With `min(node_cpus)` as its placeability cap, `_even_cpu_share` re-imposed the exact
    pinning `_fill_grant` exists to avoid: the fill path picks a 64-core grant and the share
    immediately caps it back to the 2-core node.
    """

    def _cpus(self, monkeypatch, node_cpus):
        from batcher.dist import executor

        monkeypatch.setattr(executor, "_worker_node_cpus", lambda: list(node_cpus))

    def test_a_tiny_node_no_longer_caps_the_share(self, monkeypatch):
        from batcher.dist.executor import _even_cpu_share

        self._cpus(monkeypatch, [2.0, 64.0, 64.0, 64.0])
        assert _even_cpu_share(3) == 64.0

    def test_oversubscription_still_bounds_it(self, monkeypatch):
        """The share must never hand out more cores than the cluster has."""
        from batcher.dist.executor import _even_cpu_share

        self._cpus(monkeypatch, [64.0, 64.0])
        assert _even_cpu_share(32) == 4.0  # 128 cores / 32 workers

    def test_a_genuinely_heterogeneous_fleet_is_unchanged(self, monkeypatch):
        """32 next to 64 still tiles by the smaller node, as before."""
        from batcher.dist.executor import _even_cpu_share

        self._cpus(monkeypatch, [32.0, 64.0])
        assert _even_cpu_share(3) == 32.0

    def test_unreadable_topology_falls_back_to_one(self, monkeypatch):
        from batcher.dist import executor

        def _boom():
            raise RuntimeError("ray down")

        monkeypatch.setattr(executor, "_worker_node_cpus", _boom)
        assert executor._even_cpu_share(4) == 1.0
