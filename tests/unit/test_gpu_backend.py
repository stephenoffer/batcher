"""`collect(backend="gpu")` routing: detect the supported shape, else fall back to CPU.

The GPU device execution is verified on the cluster (and `core.gpu_transform` is unit-tested);
here we cover the head-runnable control plane: which plans the GPU backend accepts, and that an
unsupported shape or a GPU-less host transparently uses the CPU engine (same result).
"""

from __future__ import annotations

import numpy as np
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

    def _rows(tbl):
        d = tbl.to_pydict()
        return sorted(zip(*d.values(), strict=True))  # group-by output order is unspecified

    assert _rows(q.collect(backend="gpu")) == _rows(q.collect(backend="cpu"))


def test_gpu_task_opts_carry_spot_preemption_retry_budget():
    # Every GPU dispatch task must retry a lost (spot-preempted) worker on a survivor rather than
    # collapsing the distributed GPU query to the single-node CPU fallback. The budget is the same
    # config knob the flight shuffle tasks use.
    from batcher.config import active_config
    from batcher.dist.gpu import gpu_task_options

    opts = gpu_task_options()
    assert opts["num_gpus"] == 1
    assert opts["max_retries"] == active_config().distributed.task_max_retries
    # A deterministic app error (OOM / unsupported expr) must fall back to CPU immediately, not
    # burn N retries -> retry_exceptions stays off for the GPU path.
    assert "retry_exceptions" not in opts


def test_oversubscribed_shards_combine_to_the_same_aggregate():
    # Oversubscribing shards past the GPU count (to bound per-GPU memory + cheapen spot retries)
    # must not change the answer: the mergeable partial->combine algebra is exact for ANY shard
    # count. Simulate the per-shard GPU partials with pandas and fold them via the real combine.
    import pandas as pd
    import pyarrow as pa

    from batcher.dist.gpu.groupby import _combine_partials, partial_aggs

    rng = np.random.default_rng(0)
    n = 5000
    full = pd.DataFrame({"k": rng.integers(0, 7, n), "v": rng.random(n)})
    aggs = {"s": ("v", "sum"), "c": ("v", "count"), "m": ("v", "mean"), "mx": ("v", "max")}
    reductions = partial_aggs(aggs)

    def shard_partial(df: pd.DataFrame) -> pa.Table:
        cols: dict[str, list] = {"k": []}
        agg_map: dict[str, list] = {a: [] for a in reductions}
        for kval, grp in df.groupby("k"):
            cols["k"].append(kval)
            for alias, (colname, func) in reductions.items():
                agg_map[alias].append(getattr(grp[colname], func)())
        return pa.table({**cols, **agg_map})

    for n_shards in (1, 4, 32):
        shards = np.array_split(full.sample(frac=1.0, random_state=1), n_shards)
        partials = [shard_partial(s) for s in shards if len(s)]
        got = _combine_partials(partials, "k", aggs).to_pandas().sort_values("k").round(6)
        exp = (
            full.groupby("k")
            .agg(s=("v", "sum"), c=("v", "count"), m=("v", "mean"), mx=("v", "max"))
            .reset_index()
            .round(6)
        )
        assert got["s"].tolist() == exp["s"].round(6).tolist()
        assert got["c"].tolist() == exp["c"].tolist()
        assert got["m"].tolist() == exp["m"].round(6).tolist()
        assert got["mx"].tolist() == exp["mx"].round(6).tolist()
