"""A shard has to come back to the node that already has it, or nothing a worker keeps pays.

Ray places a stateless GPU task wherever a device is free. That is the right default, and it is
the wrong one for the two things these workers keep between tasks: the operating system's page
cache for the files a shard reads, and the decoded **device frame cache** that holds the shard
itself on the board. With six shards over six nodes, a free placement returns each shard to its
own node about one time in six — so a cache sized to hold the whole fan-out would miss five
times out of six and read everything again.

The mapping is a digest of what the shard *reads*, not of where it sits in the fan-out. Position
renumbers whenever the shard count changes, so the same query at eight shards would warm nothing
for itself at six. It also has to agree across driver processes, which rules out `hash()`:
Python salts string hashing per process, so a reconnected driver would map every shard somewhere
new.
"""

from __future__ import annotations

import pytest

from batcher.dist.gpu.resources import _shard_identity, shard_node_affinity

pytestmark = pytest.mark.unit


class _Split:
    def __init__(self, name: str):
        self.name = name

    def identity(self):
        return self.name


def _descriptor(*names: str) -> dict:
    return {"splits": [_Split(n) for n in names]}


def _node_id(n: int) -> str:
    """A syntactically valid Ray node id. Ray validates the hex length in the strategy's own
    constructor, so a readable stand-in like `"n0"` raises before this code is reached."""
    return f"{n:056x}"


class _Node:
    def __init__(self, node_id: str):
        self.node_id = node_id


@pytest.fixture
def fleet(monkeypatch):
    from batcher.dist.gpu.resources import reset_shard_affinity

    reads = []

    def _use(ids):
        reset_shard_affinity()
        monkeypatch.setattr(
            "batcher.dist.executors.ray_runtime.fabric.gpu_node_topology",
            lambda: reads.append(1) or tuple(_Node(i) for i in ids),
        )

    _use.reads = reads
    yield _use
    reset_shard_affinity()


# --- the identity ------------------------------------------------------------


def test_identity_is_stable_across_processes():
    """Not `hash()`: Python salts string hashing per process, so a reconnected driver would map
    the same shard somewhere new and warm nothing."""
    assert _shard_identity(_descriptor("a", "b")) == _shard_identity(_descriptor("a", "b"))


def test_identity_ignores_split_order():
    assert _shard_identity(_descriptor("a", "b")) == _shard_identity(_descriptor("b", "a"))


def test_different_shards_have_different_identities():
    assert _shard_identity(_descriptor("a")) != _shard_identity(_descriptor("b"))


def test_a_shard_that_cannot_identify_itself_has_no_identity():
    class _Anonymous:
        def identity(self):
            raise RuntimeError("no")

    assert _shard_identity({"splits": [_Anonymous()]}) is None
    assert _shard_identity({"batches": []}) is None


# --- the placement -----------------------------------------------------------


def test_the_same_shard_maps_to_the_same_node(fleet):
    fleet([_node_id(i) for i in range(4)])
    first = shard_node_affinity([_descriptor("a"), _descriptor("b")])
    second = shard_node_affinity([_descriptor("b"), _descriptor("a")])
    assert [s.node_id for s in first] == list(reversed([s.node_id for s in second]))


def test_the_mapping_survives_a_change_in_shard_count(fleet):
    """The reason it is keyed on content: a query at eight shards must warm the same query at
    six for the shards whose boundaries coincide."""
    fleet([_node_id(i) for i in range(3)])
    wide = shard_node_affinity([_descriptor(c) for c in "abcdef"])
    narrow = shard_node_affinity([_descriptor(c) for c in "abc"])
    assert [s.node_id for s in wide[:3]] == [s.node_id for s in narrow]


def test_the_mapping_survives_the_topology_being_reported_in_another_order(fleet):
    fleet([_node_id(i) for i in (2, 0, 1)])
    one = shard_node_affinity([_descriptor("a")])
    fleet([_node_id(i) for i in range(3)])
    two = shard_node_affinity([_descriptor("a")])
    assert one[0].node_id == two[0].node_id


def test_the_placement_is_a_hint_and_never_a_requirement(fleet):
    """A node that is busy, drained or gone must cost a cache miss, never an unplaceable task."""
    fleet([_node_id(i) for i in range(2)])
    assert all(s.soft for s in shard_node_affinity([_descriptor("a"), _descriptor("b")]))


def test_shards_spread_over_the_fleet(fleet):
    """A digest that sent everything to one node would be worse than no affinity at all."""
    fleet([_node_id(i) for i in range(6)])
    placed = {s.node_id for s in shard_node_affinity([_descriptor(f"s{i}") for i in range(60)])}
    assert len(placed) == 6


# --- when it declines --------------------------------------------------------


def test_a_single_node_fleet_gets_no_strategies(fleet):
    """One node places everything in one place already; a strategy object per shard buys none."""
    fleet([_node_id(0)])
    assert shard_node_affinity([_descriptor("a")]) == []


def test_an_unreadable_topology_leaves_rays_placement_alone(fleet):
    fleet([])
    assert shard_node_affinity([_descriptor("a")]) == []


def test_one_unidentifiable_shard_declines_the_whole_fan_out(fleet):
    """Placing some shards and not others would concentrate the unplaced ones wherever Ray
    happens to put them, which is the imbalance this exists to avoid."""

    class _Anonymous:
        def identity(self):
            raise RuntimeError("no")

    fleet([_node_id(i) for i in range(2)])
    assert shard_node_affinity([_descriptor("a"), {"splits": [_Anonymous()]}]) == []


def test_the_node_list_is_not_re_read_on_every_fan_out(fleet):
    """`gpu_node_topology` is a `ray.nodes()` round trip, and this is a placement *hint* — the
    whole driver-side dispatch of a warm six-shard query is about 0.15 s, so an RPC per query is
    a material share of it."""
    fleet([_node_id(i) for i in range(3)])
    for _ in range(20):
        shard_node_affinity([_descriptor("a")])
    assert len(fleet.reads) == 1
