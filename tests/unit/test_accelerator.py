"""Vendor-agnostic accelerator detection + VRAM budgeting (no real GPU needed).

Detection degrades to CPU; per-vendor VRAM overhead and the packing math are pure;
utilization sampling returns None on a host without the vendor's SMI.
"""

from __future__ import annotations

import pytest

from batcher.ml.gpu import (
    _UTILIZATION,
    detect_backend,
    gpu_feedback_key,
    max_actors_per_gpu,
    recommend_gpu_fraction,
    sample_gpu_utilization,
    torch_device,
    vram_context_overhead,
)


def test_detect_backend_is_known_value():
    assert detect_backend() in {"cuda", "rocm", "xpu", "mps", "tpu", "cpu"}


def test_torch_device_maps_backend():
    assert torch_device("cuda") == "cuda"
    assert torch_device("rocm") == "cuda"  # HIP shims the CUDA device string
    assert torch_device("xpu") == "xpu"
    assert torch_device("mps") == "mps"
    assert torch_device("tpu") == "xla"  # torch_xla device string
    assert torch_device("cpu") == "cpu"


def test_vram_overhead_per_vendor():
    assert vram_context_overhead("cuda") == 0.4
    assert vram_context_overhead("rocm") == 0.5
    assert vram_context_overhead("xpu") == 0.3
    assert vram_context_overhead("mps") == 0.0
    assert vram_context_overhead("tpu") == 0.0
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


def test_sample_utilization_no_counter_backend_is_none():
    # Apple MPS, Cloud TPU, and CPU expose no per-process utilization counter, so
    # they have no registry probe and sampling is always None (loop is a no-op).
    assert sample_gpu_utilization("mps") is None
    assert sample_gpu_utilization("tpu") is None
    assert sample_gpu_utilization("cpu") is None


def test_utilization_registry_covers_counter_backends():
    # NVIDIA/AMD/Intel have a counter (a registry probe); MPS/TPU/CPU do not.
    assert set(_UTILIZATION) == {"cuda", "rocm", "xpu"}


def test_xpu_utilization_degrades_to_none_without_intel_gpu():
    # The Intel probe must never raise on a host without an Intel GPU — it returns
    # None (or, on a real Intel GPU, a fraction in [0, 1]).
    util = sample_gpu_utilization("xpu")
    assert util is None or 0.0 <= util <= 1.0


def test_gpu_feedback_key_is_accelerator_type_aware():
    import batcher as bt

    def model(batch):
        return batch

    a100 = bt.from_pydict({"x": [1, 2, 3]}).ml.map_batches(model, accelerator_type="A100")
    t4 = bt.from_pydict({"x": [1, 2, 3]}).ml.map_batches(model, accelerator_type="T4")
    plain = bt.from_pydict({"x": [1, 2, 3]}).ml.map_batches(model)

    assert "@A100" in gpu_feedback_key(a100._plan)
    # The same UDF on a different device class gets a distinct key (no cross-class
    # replay of learned utilization), while an unpinned stage keeps its bare key.
    assert gpu_feedback_key(a100._plan) != gpu_feedback_key(t4._plan)
    assert "@" not in gpu_feedback_key(plain._plan)


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
