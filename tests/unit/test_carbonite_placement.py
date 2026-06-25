"""Locality-aware reducer placement policy (pure, Ray-free).

The benefit — turning cross-node fetches into same-node shared-memory hits — only
shows on a real multi-node cluster, but the *decision* is a pure function tested here:
a concentrated bucket pulls its reducer onto the data's node, an even bucket keeps the
default round-robin, and placement never collides actors onto a node it can't host.
"""

from __future__ import annotations

from batcher.carbonite.transfer.placement import assign_reducer_hosts, reducer_affinity


def test_affinity_picks_a_concentrated_bucket_node():
    # Bucket 0 is concentrated on nodeB (80%); bucket 1 is uniform across two nodes.
    bytes_ = {
        0: {"nodeA": 100, "nodeB": 400},
        1: {"nodeA": 250, "nodeB": 250},
    }
    aff = reducer_affinity(bytes_)
    assert aff == {0: "nodeB"}  # bucket 1 (uniform) has no affinity


def test_affinity_ignores_empty_and_evenly_split_buckets():
    assert reducer_affinity({0: {}, 1: {"n": 0}}) == {}
    # A 50/50 split across two nodes is uniform (each holds the uniform share), so no
    # node is "concentrated" — placement would just unbalance the fleet for no gain.
    assert reducer_affinity({0: {"a": 50, "b": 50}}) == {}
    # A whole bucket on one node is fully concentrated.
    assert reducer_affinity({0: {"a": 90}}) == {0: "a"}
    # A clear majority above the uniform share is flagged (ties break on node id).
    assert reducer_affinity({0: {"a": 70, "b": 30}}) == {0: "a"}


def test_assign_hosts_places_concentrated_reducer_on_its_node():
    # 4 actors on 2 nodes; bucket 2 is concentrated on nodeB → its reducer must land on
    # an actor that runs on nodeB (index 2 or 3).
    actor_nodes = ["nodeA", "nodeA", "nodeB", "nodeB"]
    affinity = {2: "nodeB"}
    hosts = assign_reducer_hosts(4, actor_nodes, affinity)
    assert hosts[2] in (2, 3)  # placed on a nodeB actor
    # Unconcentrated buckets keep the default round-robin (reducer r → actor r).
    assert hosts[0] == 0 and hosts[1] == 1 and hosts[3] == 3


def test_assign_hosts_spreads_multiple_hot_buckets_over_a_nodes_actors():
    # Two buckets both concentrated on nodeB spread across nodeB's two actors, not pile
    # onto one — so a hot node's reducers still parallelize.
    actor_nodes = ["nodeA", "nodeB", "nodeB", "nodeA"]
    affinity = {0: "nodeB", 3: "nodeB"}
    hosts = assign_reducer_hosts(4, actor_nodes, affinity)
    assert {hosts[0], hosts[3]} == {1, 2}  # the two nodeB actors, one each


def test_assign_hosts_no_affinity_is_plain_round_robin():
    # With no affinity (the unskewed case), the assignment is exactly the default —
    # reducer r on actor (r % n_actors) — so behavior is unchanged.
    actor_nodes = ["nodeA", "nodeB", "nodeC"]
    assert assign_reducer_hosts(3, actor_nodes, {}) == [0, 1, 2]
    assert assign_reducer_hosts(5, actor_nodes, {}) == [0, 1, 2, 0, 1]
