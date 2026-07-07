"""`distributed="auto"` routes by DATA SIZE, not just cluster topology.

On a multi-node cluster the Ray fan-out is a ~2 s fixed cost, so a small query must stay
single-node (the sub-second-small-query mandate). A GPU stage always distributes; an
unknown/large input distributes; an explicit bool always wins. Result is identical either
way — this only chooses where to run.
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


@pytest.fixture
def multinode(monkeypatch):
    """Pretend we're on an initialized 4-node Ray cluster."""

    class _Ray:
        @staticmethod
        def is_initialized():
            return True

    monkeypatch.setitem(__import__("sys").modules, "ray", _Ray)
    monkeypatch.setattr("batcher.dist.cluster_topology", lambda: {"nodes": 4}, raising=False)


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


def test_gpu_stage_always_distributes(multinode):
    # A tiny input but a GPU map stage -> must distribute to reach the cluster's GPUs.
    ds = bt.from_pydict({"x": [1, 2, 3]}).map_batches(lambda b: b, num_gpus=1.0)
    assert resolve_distributed("auto", ds._plan, [_Src(10)]) is True


def test_threshold_respects_config(multinode, monkeypatch):
    import dataclasses

    from batcher.config import active_config

    base = active_config()
    dist_cfg = dataclasses.replace(base.distributed, distribute_min_rows=100)
    lowered = base.replace(distributed=dist_cfg)
    monkeypatch.setattr("batcher.config.active_config", lambda: lowered)
    # 80k rows now exceeds a tiny threshold -> distribute.
    assert resolve_distributed("auto", None, [_Src(80_000)]) is True
