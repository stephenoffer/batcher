"""`distributed="auto"` routes by DATA SIZE, not just cluster topology.

On a multi-node cluster the Ray fan-out is a ~2 s fixed cost, so a small query must stay
single-node (the sub-second-small-query mandate). An unknown/large input distributes; an
explicit bool always wins. Result is identical either way — this only chooses where to run.

A GPU stage distributes regardless of size, because the work has to reach the accelerators
— but only if the cluster *has* any. That qualifier was missing, and on a GPU-less cluster
the forced route asked Ray for a `num_gpus=1` resource no node could ever offer, so the
task never scheduled: `TaskUnschedulableError`, or a hang, from a query that runs fine
in-process. The same plan already runs locally whenever Ray is not up, so falling through
to the size decision is the consistent answer, not a special case.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.api.terminal.routing import resolve_distributed

pytestmark = pytest.mark.unit


class _Src:
    def __init__(self, rows: int | None):
        self._rows = rows

    def row_count(self) -> int | None:
        return self._rows


def _fake_cluster(monkeypatch, *, nodes: int, gpus: float) -> None:
    """Pretend we're on an initialized Ray cluster of a given shape."""

    class _Ray:
        @staticmethod
        def is_initialized():
            return True

    monkeypatch.setitem(__import__("sys").modules, "ray", _Ray)
    monkeypatch.setattr(
        "batcher.dist.cluster_topology",
        lambda: {"nodes": nodes, "cpus": 32.0, "gpus": gpus},
        raising=False,
    )


@pytest.fixture
def multinode(monkeypatch):
    """A 4-node cluster with GPUs — the shape most of these tests assume."""
    _fake_cluster(monkeypatch, nodes=4, gpus=4.0)


@pytest.fixture
def multinode_cpu_only(monkeypatch):
    """A 4-node cluster with no accelerators at all."""
    _fake_cluster(monkeypatch, nodes=4, gpus=0.0)


def test_explicit_bool_always_wins(multinode):
    assert resolve_distributed(True) is True
    assert resolve_distributed(False) is False


def test_small_input_stays_single_node(multinode):
    # 80k rows, well below the 1M default threshold -> single-node (avoid the fan-out tax).
    assert resolve_distributed("auto", None, [_Src(80_000)]) is False


def test_large_input_distributes(multinode):
    assert resolve_distributed("auto", None, [_Src(50_000_000)]) is True


def test_unknown_size_distributes(multinode):
    # A source that can't cheaply report a row count -> distribute (safe for large data).
    assert resolve_distributed("auto", None, [_Src(None)]) is True
    assert resolve_distributed("auto", None, None) is True


def _gpu_stage():
    """A tiny input carrying a GPU map stage. The `fn` is CPU work; `num_gpus` is the tag."""
    return bt.from_pydict({"x": [1, 2, 3]}).map_batches(lambda b: b, num_gpus=1.0)


def test_gpu_stage_distributes_when_the_cluster_has_gpus(multinode):
    # A tiny input, but the work has to reach the cluster's accelerators.
    ds = _gpu_stage()
    assert resolve_distributed("auto", ds._plan, [_Src(10)]) is True


def test_gpu_stage_stays_local_on_a_gpuless_cluster(multinode_cpu_only):
    # Routing it to a cluster with no GPUs asks Ray for a resource no node can offer, and
    # the task never schedules. Running it here is both correct and the only thing that
    # finishes.
    ds = _gpu_stage()
    assert resolve_distributed("auto", ds._plan, [_Src(10)]) is False


def test_explicit_true_beats_the_gpuless_gate(multinode_cpu_only):
    # The gate is for "auto". An explicit choice is the caller's to make, and to own.
    ds = _gpu_stage()
    assert resolve_distributed(True, ds._plan, [_Src(10)]) is True


def test_threshold_respects_config(multinode, monkeypatch):
    import dataclasses

    from batcher.config import active_config

    base = active_config()
    dist_cfg = dataclasses.replace(base.distributed, distribute_min_rows=100)
    lowered = base.replace(distributed=dist_cfg)
    monkeypatch.setattr("batcher.config.active_config", lambda: lowered)
    # 80k rows now exceeds a tiny threshold -> distribute.
    assert resolve_distributed("auto", None, [_Src(80_000)]) is True
