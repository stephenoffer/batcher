"""Kyber's cost-based GPU-backend policy covers the regimes a single `backend="gpu"` flag missed.

`decide_gpu_backend` is a pure decision over the plan's estimated size and the cluster's GPU
count — head-runnable, no GPU. It must: never use a GPU when there is none; keep tiny inputs on
the CPU (the fixed GPU overhead isn't amortized); run single-device when the working set fits one
GPU; shard across GPUs when it exceeds one but fits the cluster; keep sharding past the cluster's
aggregate VRAM as long as one *shard* fits a device; and fall back to the CPU engine when the plan
exceeds every GPU with no mergeable reducer to divide it by. `force=True` (explicit
`backend="gpu"`) bypasses only the small-input threshold, never the memory routing.
"""

from __future__ import annotations

import dataclasses

import pytest

import batcher as bt
from batcher.config import active_config, set_config
from batcher.kyber.gpu.policy import decide_gpu_backend

pytestmark = pytest.mark.unit


@pytest.fixture
def restore_config():
    saved = active_config()
    yield
    set_config(saved)


def _set_gpu(**overrides):
    cfg = active_config()
    set_config(cfg.replace(distributed=dataclasses.replace(cfg.distributed, **overrides)))


def _plan(n_rows: int):
    ds = bt.from_pydict({"k": list(range(n_rows)), "v": [float(i) for i in range(n_rows)]})
    q = ds.group_by("k").agg(s=bt.col("v").sum())
    return q._plan, q._sources


def _unshardable_plan(n_rows: int):
    """A plan with no mergeable reducer: a Python `map_batches` does not lower to the engine IR,
    so there is no algebra to divide it by and every row has to be resident at once."""
    ds = bt.from_pydict({"k": list(range(n_rows)), "v": [float(i) for i in range(n_rows)]})
    q = ds.map_batches(lambda batch: batch)
    return q._plan, q._sources


def _working_set_gb(plan, sources) -> float:
    """The exact working-set the policy uses (reducing-op-aware), so memory-regime tests set
    budgets around the real number the decision sees."""
    from batcher.kyber.gpu.policy import _estimate

    _rows, ws = _estimate(plan, sources, None)
    return ws


def test_no_gpu_is_always_cpu():
    plan, sources = _plan(10)
    d = decide_gpu_backend(plan, sources, gpu_count=0, force=True)
    assert d.use_gpu is False


def test_tiny_input_stays_on_cpu_but_force_overrides(restore_config):
    _set_gpu(gpu_min_rows=200_000, gpu_memory_gb=1000.0)
    plan, sources = _plan(1000)
    # auto: below the threshold -> CPU (overhead not amortized)
    assert decide_gpu_backend(plan, sources, gpu_count=1, force=False).use_gpu is False
    # forced: honor the user, single-dispatch (fits the huge memory budget)
    d = decide_gpu_backend(plan, sources, gpu_count=1, force=True)
    assert d.use_gpu is True and d.distributed is False


def test_fits_one_gpu_runs_single_dispatch(restore_config):
    _set_gpu(gpu_min_rows=10, gpu_memory_gb=1000.0)
    plan, sources = _plan(5000)
    d = decide_gpu_backend(plan, sources, gpu_count=4, force=False)
    assert d.use_gpu is True and d.distributed is False


def test_exceeds_one_gpu_but_fits_cluster_shards(restore_config):
    # Per-GPU budget below the working set but 8 GPUs together above it -> shard across GPUs.
    plan, sources = _plan(5000)
    ws = _working_set_gb(plan, sources)
    _set_gpu(gpu_min_rows=10, gpu_memory_gb=ws / 4)  # one GPU holds a quarter; 8 hold 2x
    d = decide_gpu_backend(plan, sources, gpu_count=8, force=False)
    assert d.use_gpu is True and d.distributed is True


def test_exceeds_all_gpus_but_shards_small_enough_still_uses_them(restore_config):
    """Past the cluster's aggregate VRAM the question becomes how small a shard can be made.

    A mergeable plan oversubscribes shards and pipelines them, so what must fit a device is one
    shard rather than the working set. Refusing it outright would make the fan-out built for
    exactly this case unreachable.
    """
    plan, sources = _plan(5000)
    ws = _working_set_gb(plan, sources)
    # 2 GPUs hold under half the set, and 8 oversubscribed shards of it each fit one device's
    # *usable* budget with room to spare. Sized off `ws / 6` rather than `ws / 8` so the shard
    # clears the VRAM headroom instead of landing exactly on the un-derated device size, which
    # is a boundary no real device has.
    _set_gpu(gpu_min_rows=10, gpu_memory_gb=ws / 6)
    d = decide_gpu_backend(plan, sources, gpu_count=2, force=True)
    assert d.use_gpu is True and d.distributed is True


def test_exceeds_all_gpus_with_nothing_to_shard_on_falls_back_to_cpu(restore_config):
    """A plan with no mergeable reducer needs the whole set resident, so it goes to the CPU.

    The counter-case to the test above, and the reason the routing turns on shardability rather
    than on size alone: a single dispatch is the only accelerated form available for this shape,
    it would not fit, and the spillable CPU engine is the honest destination. `force=True` does
    not override it — an explicit `backend="gpu"` bypasses the small-input threshold only.
    """
    plan, sources = _unshardable_plan(5000)
    ws = _working_set_gb(plan, sources)
    _set_gpu(gpu_min_rows=10, gpu_memory_gb=ws / 8)
    d = decide_gpu_backend(plan, sources, gpu_count=2, force=True)
    assert d.use_gpu is False
    assert "nothing to shard on" in d.reason


def test_gpu_memory_budget_is_detected_not_assumed(monkeypatch):
    """Kyber must size against the GPU it actually has, not a 2018 T4.

    `gpu_memory_gb` was hardcoded to 12.0 ("targets a T4"). Three decisions ride on it — single
    GPU vs shard vs CPU routing, the `num_gpus` packing fraction, and the inference batch-size
    seed — so on an 80 GB A100 every one of them was wrong by ~6x in the direction that leaves
    the device idle: sharding a working set one card would have held, and seeding tiny batches.
    """
    import dataclasses

    import batcher._internal.hardware as hw
    from batcher.config import active_config

    dc = active_config().distributed
    assert dc.gpu_memory_gb == 0.0, "the default must be 'detect', not a device guess"

    # A *capacity*, in decimal GB, matching the unit Kyber measures a working set in
    # (`rows x width / 1e9`). Dividing by `1 << 30` here reported an 80 GiB board as "80"
    # against a GB-denominated working set, over-stating the device by 7.4%.
    monkeypatch.setattr(
        hw, "gpu_inventory", lambda: [{"index": 0, "name": "A100", "memory_bytes": 80 << 30}]
    )
    assert dc.resolved_gpu_memory_gb() == (80 << 30) / 1e9

    # A heterogeneous cluster plans for the SMALLEST device — sizing to the largest would
    # dispatch a working set that OOMs every other card.
    monkeypatch.setattr(
        hw,
        "gpu_inventory",
        lambda: [
            {"index": 0, "name": "A100", "memory_bytes": 80 << 30},
            {"index": 1, "name": "T4", "memory_bytes": 16 << 30},
        ],
    )
    assert dc.resolved_gpu_memory_gb() == (16 << 30) / 1e9

    # No visible device → a T4-shaped nameplate, so a CPU-only driver planning for a remote
    # GPU worker keeps a budget within a gigabyte of the one it always had.
    monkeypatch.setattr(hw, "gpu_inventory", lambda: [])
    assert dc.resolved_gpu_memory_gb() == 16.0

    # An explicit setting always wins over detection.
    pinned = dataclasses.replace(dc, gpu_memory_gb=40.0)
    assert pinned.resolved_gpu_memory_gb() == 40.0


def test_the_vram_headroom_knob_moves_the_routing_budget():
    """`accelerator.vram_headroom` must reach Kyber's routing, not only Carbonite's pool.

    It was one knob with five private copies: `0.15` in the packing math, `0.15` in the VRAM
    pool, `0.85` usable in the device recommender, `0.75` usable in the distributed config, and
    no headroom at all in `cluster_gpu_memory_gb` — which is the figure Kyber routes against on
    a live cluster. So a working set was dispatched to a single device at 100% of its VRAM,
    leaving nothing for the CUDA context or the hash table the kernel builds, and raising the
    knob for a fleet with a resident co-tenant moved neither.
    """
    from batcher.config import active_config, config_context

    plan, sources = _plan(5000)
    ws = _working_set_gb(plan, sources)

    def single_device(headroom: float) -> bool:
        base = active_config()
        cfg = base.replace(
            accelerator=dataclasses.replace(base.accelerator, vram_headroom=headroom)
        )
        with config_context(cfg):
            decision = decide_gpu_backend(
                plan, sources, gpu_count=4, force=True, gpu_memory_gb=ws * 1.25
            )
        return decision.use_gpu and not decision.distributed

    # A device 1.25x the working set holds it with 10% of the board reserved, and does not
    # once half the board is. Before, no headroom reached this decision at all and both
    # answered "fits one GPU" — including the one that leaves the kernel no room to build in.
    assert single_device(0.1)
    assert not single_device(0.5)
