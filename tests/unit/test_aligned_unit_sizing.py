"""How many key-range units an aligned cut is cut into, for a small input and a large one."""

from __future__ import annotations

import pytest

from batcher.dist.executors.aligned.units import SMALL_UNIT_BYTES, plan_units
from batcher.io.splits.parquet import FileKeyBounds

pytestmark = pytest.mark.unit

_STREAMS = 16


def _files(n: int, nbytes: int) -> dict[int, list]:
    return {0: [FileKeyBounds(f"f{i}", i * 10, i * 10 + 9, 1_000, nbytes) for i in range(n)]}


def test_a_small_input_is_one_round_of_one_unit_per_stream():
    """0.5 GB in 400 files: six units a stream would be 5 MB each, all fixed cost."""
    units = plan_units(_files(400, (512 << 20) // 400), {}, 6 << 30, 6 * _STREAMS, _STREAMS)
    assert units is not None and _STREAMS // 2 <= len(units) <= _STREAMS


def test_a_mid_size_input_is_whole_rounds_of_units():
    """Three rounds' worth of `SMALL_UNIT_BYTES` per stream: three units a stream."""
    total = 3 * _STREAMS * SMALL_UNIT_BYTES + 960 * (1 << 20)
    units = plan_units(_files(960, total // 960), {}, 6 << 30, 6 * _STREAMS, _STREAMS)
    assert units is not None and 2 * _STREAMS < len(units) <= 3 * _STREAMS + 2


def test_a_large_input_keeps_several_units_per_stream():
    units = plan_units(
        _files(960, 64 * SMALL_UNIT_BYTES // 96), {}, 6 << 30, 6 * _STREAMS, _STREAMS
    )
    assert units is not None and len(units) >= 4 * _STREAMS


def test_without_streams_the_cut_is_unchanged():
    units = plan_units(_files(400, (512 << 20) // 400), {}, 6 << 30, 6 * _STREAMS)
    assert units is not None and len(units) >= 4 * _STREAMS


def test_unit_tasks_are_placed_on_the_same_nodes_every_run(monkeypatch):
    """Task `t` goes to slot `t`'s node, slots interleaved across nodes, in node-id order.

    The order of `ray.nodes()` is not the order of the slots: two calls listing the nodes
    differently must still place every task on the same node, which is what lets a node's
    warm caches serve the units it read last time.
    """
    ray = pytest.importorskip("ray")

    from batcher.dist.executors.aligned import run

    a, b = "aa" * 28, "bb" * 28  # node ids are hex
    listed = [
        {"NodeID": b, "Alive": True, "Resources": {"CPU": 16.0}},
        {"NodeID": "head", "Alive": True, "Resources": {"CPU": 0.0}},
        {"NodeID": a, "Alive": True, "Resources": {"CPU": 16.0}},
        {"NodeID": "dead", "Alive": False, "Resources": {"CPU": 16.0}},
    ]
    monkeypatch.setattr(ray, "nodes", lambda: listed)
    first = run._slot_nodes(8)
    assert first == [a, b, a, b]
    monkeypatch.setattr(ray, "nodes", lambda: list(reversed(listed)))
    assert run._slot_nodes(8) == first
    strategy = run._placement(first, 5)["scheduling_strategy"]
    assert strategy.node_id == b and strategy.soft
    assert run._placement([], 5) == {}
