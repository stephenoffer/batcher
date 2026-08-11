"""Replica placement has to balance the memory it is actually adding.

A primary copy and a replica copy are the same bucket and cost the same bytes. Ranking
candidates by replica count alone made a worker already holding many primaries look like
the emptiest host on the cluster, so it attracted replicas too — the placement decision
concentrating memory on exactly the worker least able to take it.

That matters twice over. It is a memory-balance defect, so the busiest worker is the one
pushed toward its envelope. And it is a *durability* defect in slower motion: the more of
the shuffle one worker holds, the more a single loss costs, which is the thing replication
was paid for to prevent.

The failure-domain rule outranks balance and must stay that way. A replica on a different
node from its primary is the whole point — a node loss is the unit a spot reclamation
takes — so these tests pin that first and balance second.
"""

from __future__ import annotations

from collections import Counter

import pytest

from batcher.carbonite.resilience.replication import assign_replica_hosts

pytestmark = pytest.mark.unit


def _copies_per_worker(primaries: dict[int, int], replicas: dict[int, list[int]]) -> Counter:
    """Total buckets each worker holds, primaries and replicas together."""
    held: Counter = Counter(primaries.values())
    for reps in replicas.values():
        held.update(reps)
    return held


# --- the defect -----------------------------------------------------------------


def test_a_worker_full_of_primaries_is_not_treated_as_empty() -> None:
    """The measured case: worker 1 holds three of four primaries.

    Counting replicas alone, worker 1 scored 0 and won the first placement, ending with 4
    of the 8 copies while two idle workers held 1 each.
    """
    primaries = {0: 0, 1: 1, 2: 1, 3: 1}
    nodes = ["n1", "n2", "n3", "n4"]
    held = _copies_per_worker(primaries, assign_replica_hosts(primaries, nodes, factor=2))

    assert held[1] == 3, "the busiest worker was handed a replica on top of its primaries"
    assert max(held.values()) <= 3
    assert min(held.values()) >= 1, "a worker was left idle while another carried four"


def test_no_worker_holds_more_than_its_share_plus_one() -> None:
    """Balance stated as a bound rather than as a specific layout.

    Eight sources over four workers at factor 2 is 16 copies, so a perfectly even spread is
    4 each. Primaries are fixed and lopsided, so exact evenness is not reachable — but no
    worker should exceed the even share by more than the primary skew forces.
    """
    primaries = {i: (0 if i < 5 else i - 4) for i in range(8)}
    nodes = ["n1", "n2", "n3", "n4"]
    held = _copies_per_worker(primaries, assign_replica_hosts(primaries, nodes, factor=2))

    assert sum(held.values()) == 16
    # Worker 0 is forced to 5 by its primaries; nothing may exceed that.
    assert max(held.values()) <= 5


def test_replicas_go_to_the_workers_carrying_least() -> None:
    """With primaries piled on one worker, every replica should land elsewhere."""
    primaries = {0: 0, 1: 0, 2: 0}
    nodes = ["n1", "n2", "n3"]
    replicas = assign_replica_hosts(primaries, nodes, factor=2)
    assert all(0 not in reps for reps in replicas.values()), (
        "a replica was placed on the worker holding every primary"
    )


# --- what balance must not cost -------------------------------------------------


def test_a_replica_still_lands_off_the_primarys_node() -> None:
    """The failure-domain rule outranks balance, and a node loss is the real unit."""
    primaries = {0: 0, 1: 1, 2: 2, 3: 3}
    nodes = ["n1", "n1", "n2", "n2"]
    replicas = assign_replica_hosts(primaries, nodes, factor=2)
    for src, reps in replicas.items():
        for w in reps:
            assert nodes[w] != nodes[primaries[src]], (
                f"source {src}'s replica shares a node with its primary — one node loss "
                "takes both copies"
            )


def test_two_replicas_land_in_two_different_domains() -> None:
    """`factor=3` must buy two independent copies, not two on one node."""
    primaries = {0: 0}
    nodes = ["n1", "n2", "n2", "n3"]
    reps = assign_replica_hosts(primaries, nodes, factor=3)[0]
    assert len(reps) == 2
    assert len({nodes[w] for w in reps}) == 2, "both replicas landed in one failure domain"


def test_a_dead_worker_never_receives_a_copy() -> None:
    primaries = {0: 0, 1: 1}
    nodes = ["n1", "n2", "n3"]
    replicas = assign_replica_hosts(primaries, nodes, factor=2, dead={2})
    assert all(2 not in reps for reps in replicas.values())


def test_placement_is_deterministic() -> None:
    """A replay must assign the same hosts, or recovery looks in the wrong place."""
    primaries = {0: 0, 1: 1, 2: 1, 3: 2}
    nodes = ["n1", "n2", "n3", "n4"]
    first = assign_replica_hosts(primaries, nodes, factor=2)
    assert all(first == assign_replica_hosts(primaries, nodes, factor=2) for _ in range(5))


# --- degenerate inputs ----------------------------------------------------------


def test_no_replication_is_still_no_replication() -> None:
    """`factor=1` is the default and must remain byte-for-byte the old behaviour."""
    assert assign_replica_hosts({0: 0, 1: 1}, ["n1", "n2"], factor=1) == {0: [], 1: []}


def test_a_cluster_too_small_degrades_instead_of_failing() -> None:
    """Replication is an optimization: one worker means the recompute path, not an error."""
    assert assign_replica_hosts({0: 0}, ["n1"], factor=3) == {0: []}


def test_a_primary_on_an_unknown_worker_does_not_break_placement() -> None:
    """A primary index past the fleet is stale metadata, not a reason to raise."""
    out = assign_replica_hosts({0: 99}, ["n1", "n2"], factor=2)
    assert set(out[0]) <= {0, 1}


# --- spot capacity as a failure domain ---------------------------------------------------


def test_a_replica_prefers_durable_capacity_over_a_second_spot_node():
    """A spot reclamation takes an instance group, not a machine.

    Placing the only spare copy on a second spot node satisfies the off-node rule and buys
    nothing: both go away in the same wave. That is the fleet the `spot` resilience profile
    turns replication on for, so it is the fleet the ranking has to get right.
    """
    from batcher.carbonite.resilience.replication import assign_replica_hosts

    nodes = ["n0", "n1", "n2"]  # worker i on node i
    out = assign_replica_hosts({0: 0}, nodes, factor=2, preemptible=frozenset({0, 1}))
    assert out[0] == [2], "the on-demand worker must hold the copy"


def test_the_off_node_rule_still_outranks_the_market_preference():
    """A same-node copy is useless outright; a same-market copy is merely correlated.

    So an on-demand worker on the primary's own node must not beat a spot worker elsewhere.
    """
    from batcher.carbonite.resilience.replication import assign_replica_hosts

    nodes = ["n0", "n0", "n1"]  # workers 0 and 1 share a node
    out = assign_replica_hosts({0: 0}, nodes, factor=2, preemptible=frozenset({2}))
    assert out[0] == [2]


def test_an_all_spot_fleet_places_its_copies_as_before():
    """Preferring durable capacity must never mean placing no copy at all.

    On a fleet that is entirely spot every candidate ranks the same, so the placement falls
    back to the off-node and least-loaded rules unchanged.
    """
    from batcher.carbonite.resilience.replication import assign_replica_hosts

    nodes = ["n0", "n1", "n2", "n3"]
    spot = frozenset(range(4))
    assert assign_replica_hosts({0: 0, 1: 1}, nodes, factor=2, preemptible=spot) == (
        assign_replica_hosts({0: 0, 1: 1}, nodes, factor=2)
    )


def test_an_unlabelled_fleet_is_placed_exactly_as_it_was():
    """No market labels means no evidence, and no evidence must change nothing."""
    from batcher.carbonite.resilience.replication import assign_replica_hosts

    nodes = ["n0", "n1", "n2", "n3"]
    assert assign_replica_hosts({0: 0, 1: 1, 2: 2}, nodes, factor=2, preemptible=frozenset()) == (
        assign_replica_hosts({0: 0, 1: 1, 2: 2}, nodes, factor=2)
    )


def test_spot_ranks_below_the_off_node_rule_and_above_load():
    """The three-way order, pinned so a later tweak cannot quietly reorder it.

    Worker 1 is idle but spot; worker 2 carries a primary but is on-demand. On-demand wins,
    because an evenly spread set of copies that all vanish together is not a spread set.
    """
    from batcher.carbonite.resilience.replication import assign_replica_hosts

    nodes = ["n0", "n1", "n2"]
    out = assign_replica_hosts({0: 0, 1: 2}, nodes, factor=2, preemptible=frozenset({1}))
    assert out[0] == [2]
