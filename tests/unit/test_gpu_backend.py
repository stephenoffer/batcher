"""`collect(backend="gpu")` routing: detect the supported shape, else fall back to CPU.

The GPU device execution is verified on the cluster (and `core.gpu_transform` is unit-tested);
here we cover the head-runnable control plane: which plans the GPU backend accepts, and that an
unsupported shape or a GPU-less host transparently uses the CPU engine (same result).
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import col
from batcher.api.terminal.gpu_backend import _gpu_agg_spec

pytestmark = pytest.mark.unit


def _plan(ds):
    return ds._plan


def test_detects_supported_single_key_aggregate():
    ds = bt.from_pydict({"k": [1, 2], "v": [1.0, 2.0]})
    spec = _gpu_agg_spec(_plan(ds.group_by("k").agg(s=col("v").sum(), m=col("v").mean())))
    assert spec is not None
    key_out, key_src, aggs, _scan = spec
    assert key_out == "k" and key_src == "k"
    assert aggs == {"s": ("v", "sum"), "m": ("v", "mean")}


def test_rejects_multi_key_and_non_scan_and_bad_agg():
    ds = bt.from_pydict({"k": [1, 2], "j": [1, 2], "v": [1.0, 2.0]})
    # multi-key group-by -> unsupported (single key only for now)
    assert _gpu_agg_spec(_plan(ds.group_by("k", "j").agg(s=col("v").sum()))) is None
    # a filter between the scan and the aggregate -> not a direct scan, unsupported
    assert _gpu_agg_spec(_plan(ds.filter(col("v") > 0).group_by("k").agg(s=col("v").sum()))) is None
    # a non-plain-column key expression -> unsupported
    assert _gpu_agg_spec(_plan(ds.group_by("k").agg(s=col("v").median()))) is None
    # a plain scan/filter (no aggregate) -> unsupported
    assert _gpu_agg_spec(_plan(ds.filter(col("v") > 0))) is None


def test_gpu_backend_falls_back_to_cpu_on_gpuless_host():
    # No GPU on the test host -> backend="gpu" must transparently equal backend="cpu".
    ds = bt.from_pydict({"k": [1, 1, 2, 3, 2], "v": [10.0, 20.0, 5.0, 7.0, 9.0]})
    q = ds.group_by("k").agg(s=col("v").sum(), c=col("v").count())
    assert q.collect(backend="gpu").to_pydict() == q.collect(backend="cpu").to_pydict()


def test_gpu_task_opts_carry_spot_preemption_retry_budget():
    # Every GPU dispatch task must retry a lost (spot-preempted) worker on a survivor rather than
    # collapsing the distributed GPU query to the single-node CPU fallback. The budget is the same
    # config knob the flight shuffle tasks use.
    from batcher.api.terminal.gpu_backend import _gpu_task_opts
    from batcher.config import active_config

    opts = _gpu_task_opts()
    assert opts["num_gpus"] == 1
    assert opts["max_retries"] == active_config().distributed.task_max_retries
    # A deterministic app error (OOM / unsupported expr) must fall back to CPU immediately, not
    # burn N retries -> retry_exceptions stays off for the GPU path.
    assert "retry_exceptions" not in opts
