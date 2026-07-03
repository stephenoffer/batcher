"""Kyber's cost-based GPU-backend policy covers the regimes a single `backend="gpu"` flag missed.

`decide_gpu_backend` is a pure decision over the plan's estimated size and the cluster's GPU
count — head-runnable, no GPU. It must: never use a GPU when there is none; keep tiny inputs on
the CPU (the fixed GPU overhead isn't amortized); run single-device when the working set fits one
GPU; shard across GPUs when it exceeds one but fits the cluster; and fall back to the CPU engine
when it exceeds every GPU. `force=True` (explicit `backend="gpu"`) bypasses only the small-input
threshold, never the memory routing.
"""

from __future__ import annotations

import dataclasses

import pytest

import batcher as bt
from batcher.config import active_config, set_config
from batcher.kyber.gpu_policy import decide_gpu_backend

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


def _working_set_gb(plan, sources) -> float:
    """The same working-set estimate the policy uses, so memory-regime tests set budgets around
    the real number instead of a guessed one."""
    from batcher.kyber.cardinality import CardinalityEstimator

    est = CardinalityEstimator(sources=sources)
    rows = int(est.estimate(plan).rows)
    width = est.row_width(plan, active_config().optimizer.row_bytes)
    return rows * max(width, 1) / 1e9


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


def test_exceeds_all_gpus_falls_back_to_cpu(restore_config):
    # Working set larger than every GPU combined -> the spillable CPU engine, even when forced.
    plan, sources = _plan(5000)
    ws = _working_set_gb(plan, sources)
    _set_gpu(gpu_min_rows=10, gpu_memory_gb=ws / 8)  # 2 GPUs hold only a quarter of the set
    d = decide_gpu_backend(plan, sources, gpu_count=2, force=True)
    assert d.use_gpu is False
