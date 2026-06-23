"""Vendor-agnostic accelerator detection + VRAM budgeting (no real GPU needed).

Detection degrades to CPU; per-vendor VRAM overhead and the packing math are pure;
utilization sampling returns None on a host without the vendor's SMI.
"""

from __future__ import annotations

import pytest

from batcher.ml.gpu import (
    detect_backend,
    max_actors_per_gpu,
    recommend_gpu_fraction,
    sample_gpu_utilization,
    torch_device,
    vram_context_overhead,
)


def test_detect_backend_is_known_value():
    assert detect_backend() in {"cuda", "rocm", "xpu", "mps", "cpu"}


def test_torch_device_maps_backend():
    assert torch_device("cuda") == "cuda"
    assert torch_device("rocm") == "cuda"  # HIP shims the CUDA device string
    assert torch_device("xpu") == "xpu"
    assert torch_device("mps") == "mps"
    assert torch_device("cpu") == "cpu"


def test_vram_overhead_per_vendor():
    assert vram_context_overhead("cuda") == 0.4
    assert vram_context_overhead("rocm") == 0.5
    assert vram_context_overhead("xpu") == 0.3
    assert vram_context_overhead("mps") == 0.0
    assert vram_context_overhead("cpu") == 0.0


def test_max_actors_uses_explicit_overhead():
    # 24GB GPU, 0.8 usable = 19.2; per actor = 1*1.5 + 0.4 = 1.9 -> 10 actors
    assert max_actors_per_gpu(1.0, 24.0, context_overhead_gb=0.4) == 10
    # A bigger context overhead packs fewer actors
    assert max_actors_per_gpu(1.0, 24.0, context_overhead_gb=2.0) < 10
    # A model too big for the GPU still gets a whole device (never 0)
    assert max_actors_per_gpu(40.0, 24.0) == 1


def test_recommend_gpu_fraction_floor():
    # A tiny model packs many actors but the fraction is floored at 0.25 (<= 4/GPU).
    assert recommend_gpu_fraction(0.1, 80.0) == 0.25
    # A model that fills the GPU gets a whole device.
    assert recommend_gpu_fraction(40.0, 48.0) == 1.0


def test_sample_utilization_unsupported_backend_is_none():
    assert sample_gpu_utilization("mps") is None
    assert sample_gpu_utilization("cpu") is None
    assert sample_gpu_utilization("xpu") is None


def test_triton_dtype_covers_modern_dtypes():
    import numpy as np

    from batcher.ml.serving.triton import _triton_dtype

    assert _triton_dtype(np.zeros(2, dtype="float32")) == "FP32"
    assert _triton_dtype(np.zeros(2, dtype="float16")) == "FP16"
    assert _triton_dtype(np.zeros(2, dtype="uint16")) == "UINT16"
    assert _triton_dtype(np.zeros(2, dtype="int16")) == "INT16"
    ml_dtypes = pytest.importorskip("ml_dtypes")
    assert _triton_dtype(np.zeros(2, dtype=ml_dtypes.bfloat16)) == "BF16"


def test_prefetch_propagates_errors_not_truncates():
    torch = pytest.importorskip("torch")  # noqa: F841
    import batcher as bt

    ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0]})

    def boom(_arrays):
        raise RuntimeError("collate failed")

    with pytest.raises(RuntimeError, match="collate failed"):
        list(ds.ml.iter_torch_batches(batch_size=2, collate_fn=boom, prefetch_batches=2))
