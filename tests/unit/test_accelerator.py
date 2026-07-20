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


@pytest.mark.parametrize("backend", ["hpu", "neuron", "something_new"])
def test_unknown_backend_degrades_to_cpu_rather_than_raising(backend):
    """This was a bare dict lookup, so an unrecognized name raised `KeyError`.

    The names come from places this mapping does not control — a caller naming an
    accelerator it has not been taught, or a newer `detect_backend`. An accelerator layer
    is an optimization, so an unknown device must fall back to CPU rather than abort a job
    that would have run correctly."""
    assert torch_device(backend) == "cpu"


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


def test_visible_device_indices_honors_cuda_visible_devices(monkeypatch):
    from batcher.ml.gpu import _visible_device_indices

    # Unset/empty → all physical devices (an unpinned driver or monitor).
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert _visible_device_indices(4) == (0, 1, 2, 3)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert _visible_device_indices(4) == (0, 1, 2, 3)
    # A single-device pin (how Ray pins one GPU per actor) → just that device.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert _visible_device_indices(4) == (2,)
    # A multi-device pin → each visible device, in order.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    assert _visible_device_indices(4) == (1, 3)
    # Out-of-range / UUID entries are dropped; if nothing valid remains, fall back to all.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "9")
    assert _visible_device_indices(4) == (0, 1, 2, 3)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc123")
    assert _visible_device_indices(4) == (0, 1, 2, 3)
    # AMD ROCm pins through HIP_/ROCR_VISIBLE_DEVICES — honored when CUDA's is unset.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "3")
    assert _visible_device_indices(4) == (3,)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1")
    assert _visible_device_indices(4) == (0, 1)


def test_rocm_utilization_attributes_to_the_pinned_device(monkeypatch):
    import sys
    import types

    from batcher.ml import gpu

    utils = {0: 0, 1: 0, 2: 80, 3: 0}
    fake = types.ModuleType("amdsmi")
    fake.amdsmi_init = lambda: None
    fake.amdsmi_shut_down = lambda: None
    fake.amdsmi_get_processor_handles = lambda: [0, 1, 2, 3]
    fake.amdsmi_get_gpu_activity = lambda h: {"gfx_activity": utils[h]}
    monkeypatch.setitem(sys.modules, "amdsmi", fake)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2")
    # Device 2's 80% — not the 4-device mean (20%).
    assert gpu._rocm_utilization() == pytest.approx(0.80)


def test_gpu_samples_attribute_to_the_pinned_device(monkeypatch):
    # On a multi-GPU node a pinned actor must sample ITS device, not physical 0 or a
    # node-wide mean that a co-located idle/busy GPU would distort.
    import sys
    import types

    from batcher.ml import gpu

    # 4 devices; device 2 is the busy one this actor is pinned to. used/total differ per device.
    utils = {0: 0, 1: 0, 2: 90, 3: 0}
    mems = {i: types.SimpleNamespace(used=i + 1, total=10) for i in range(4)}
    fake = types.ModuleType("pynvml")
    fake.nvmlInit = lambda: None
    fake.nvmlShutdown = lambda: None
    fake.nvmlDeviceGetCount = lambda: 4
    fake.nvmlDeviceGetHandleByIndex = lambda i: i  # handle IS the index, for the mock
    fake.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=utils[h])
    fake.nvmlDeviceGetMemoryInfo = lambda h: mems[h]
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    gpu._nvml.cache_clear()
    gpu._nvml_handles.cache_clear()
    try:
        # Utilization is device 2's 90% — NOT the 4-device mean (22.5%).
        assert sample_gpu_utilization("cuda") == pytest.approx(0.90)
        # VRAM fraction is device 2's used/total = 3/10 — not physical 0's 1/10.
        assert gpu.sample_gpu_vram_fraction() == pytest.approx(0.30)
    finally:
        gpu._nvml.cache_clear()
        gpu._nvml_handles.cache_clear()


def test_nvml_session_initialized_once_across_many_samples(monkeypatch):
    # Per-batch GPU sampling (throughput autobatcher VRAM cap, adaptive num_gpus loop)
    # must share one NVML session, not init/shutdown the driver on every call.
    import sys
    import types

    from batcher.ml import gpu

    calls = {"init": 0, "shutdown": 0}
    fake = types.ModuleType("pynvml")
    fake.nvmlInit = lambda: calls.__setitem__("init", calls["init"] + 1)
    fake.nvmlShutdown = lambda: calls.__setitem__("shutdown", calls["shutdown"] + 1)
    fake.nvmlDeviceGetCount = lambda: 1
    fake.nvmlDeviceGetHandleByIndex = lambda i: f"handle-{i}"
    fake.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=42)
    fake.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(used=2, total=8)
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    gpu._nvml.cache_clear()
    gpu._nvml_handles.cache_clear()
    try:
        for _ in range(5):
            assert sample_gpu_utilization("cuda") == pytest.approx(0.42)
            assert gpu.sample_gpu_vram_fraction() == pytest.approx(0.25)
        assert calls["init"] == 1  # one handshake for the whole process
        assert calls["shutdown"] == 0  # session held open, not torn down per sample
    finally:
        gpu._nvml.cache_clear()
        gpu._nvml_handles.cache_clear()


def test_gpu_peak_vram_record_load_and_pack():
    # The memory twin of the utilization loop: a measured peak-VRAM fraction is persisted,
    # smoothed, and turned into an actors-per-GPU packing count.
    from batcher.metadata import MetadataHub
    from batcher.metadata.backends import InProcessBackend
    from batcher.ml.gpu import (
        actors_per_gpu_from_learned_vram,
        load_gpu_peak_vram,
        record_gpu_peak_vram,
    )

    hub = MetadataHub(InProcessBackend())
    record_gpu_peak_vram(hub, "pipe", 0.3)  # an actor peaked at 30% VRAM
    assert load_gpu_peak_vram(hub, "pipe") == pytest.approx(0.3)
    record_gpu_peak_vram(hub, "pipe", 0.5)  # exp-smoothed toward the new sample
    assert 0.3 < load_gpu_peak_vram(hub, "pipe") < 0.5
    # 30% peak → ~2 actors fit within an 0.8 usable budget; None measurement → None.
    assert actors_per_gpu_from_learned_vram(0.3) == 2
    assert actors_per_gpu_from_learned_vram(0.9) == 1
    assert actors_per_gpu_from_learned_vram(None) is None
    # Cold store / None inputs never raise.
    record_gpu_peak_vram(hub, "pipe", None)
    assert load_gpu_peak_vram(None, "pipe") is None


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
