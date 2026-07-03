"""The GPU-transform kernel gates on device availability and never silently mis-computes.

The GPU compute correctness is verified on a live GPU worker by
`benchmarks/gpu_backend/tpch_gpu_agg.py` (correctness-gated vs the CPU engine). Here we cover
what runs on a GPU-less host: availability gating and the explicit-error contract, so a caller
falls back to the CPU engine instead of getting a wrong answer.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher._internal.errors import BackendError
from batcher.core.gpu_transform import _torch_groupby_agg, gpu_available, gpu_groupby_agg

pytestmark = pytest.mark.unit


def test_torch_kernel_matches_cpu_engine():
    """The GPU kernel's algorithm (torch.unique densify + scatter reductions), run on
    ``device="cpu"`` so it's verifiable without a GPU, must match Batcher's own group-by for
    every reduction. The GPU path is the identical code with ``device="cuda"``."""
    pytest.importorskip("torch")
    import batcher as bt
    from batcher import col

    # Arbitrary (non-dense) integer keys to exercise the densify path.
    keys = [7, 3, 7, 100, 3, 7, 100, 3]
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    t = pa.table({"k": keys, "v": vals})

    aggs = {
        "s": ("v", "sum"),
        "c": ("v", "count"),
        "m": ("v", "mean"),
        "mn": ("v", "min"),
        "mx": ("v", "max"),
    }
    got = _torch_groupby_agg(t, "k", aggs, device="cpu").to_pydict()
    exp = (
        bt.from_arrow(t)
        .group_by("k")
        .agg(
            s=col("v").sum(),
            c=col("v").count(),
            m=col("v").mean(),
            mn=col("v").min(),
            mx=col("v").max(),
        )
        .collect()
        .to_pydict()
    )
    def _rows(d):
        cols = (d["k"], d["s"], d["c"], d["m"], d["mn"], d["mx"])
        return {r[0]: r[1:] for r in zip(*cols, strict=True)}

    gmap, emap = _rows(got), _rows(exp)
    assert set(gmap) == set(emap)
    for k in emap:
        es, ec, em, emn, emx = emap[k]
        gs, gc, gm, gmn, gmx = gmap[k]
        assert gc == ec
        assert abs(gs - es) < 1e-9 and abs(gm - em) < 1e-9
        assert gmn == emn and gmx == emx


def test_gpu_available_returns_bool_without_raising():
    assert isinstance(gpu_available(), bool)


def test_groupby_raises_when_no_gpu(monkeypatch):
    # Force the no-GPU path (the driver/head case) — must raise, never return a wrong table.
    monkeypatch.setattr("batcher.core.gpu_transform.gpu_available", lambda: True, raising=True)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False, raising=False)
    # gpu_available is monkeypatched True but the kernel also needs a real device; if torch is
    # absent or has no device the import/compute raises BackendError or a torch error we wrap.
    t = pa.table({"k": [1, 1, 2], "v": [10.0, 20.0, 5.0]})
    with pytest.raises(Exception):  # noqa: B017 - BackendError or a torch RuntimeError (no device)
        gpu_groupby_agg(t, "k", {"s": ("v", "sum")})


def test_unsupported_reduction_is_rejected(monkeypatch):
    monkeypatch.setattr("batcher.core.gpu_transform.gpu_available", lambda: True, raising=True)
    t = pa.table({"k": [1, 2], "v": [1.0, 2.0]})
    with pytest.raises(BackendError, match="unsupported GPU reduction"):
        gpu_groupby_agg(t, "k", {"bad": ("v", "median")})
