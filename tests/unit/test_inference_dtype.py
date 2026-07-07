"""Zero-config half-precision-by-GPU for the managed ``ds.ml.infer`` transformers path.

`recommend_inference_dtype` picks BF16 on Ampere+, FP16 on Turing/Volta/MPS, and FP32
(None) below or on CPU. `_pipeline_accel_kwargs` turns that into the pipeline's
`device`/`torch_dtype`, never trading correctness for the fast path on a probe failure.
"""

from __future__ import annotations

import pytest

from batcher.ml.gpu import recommend_inference_dtype
from batcher.ml.inference import _pipeline_accel_kwargs

pytestmark = pytest.mark.unit


def test_cpu_and_unknown_backends_keep_fp32(monkeypatch):
    monkeypatch.setattr("batcher.ml.gpu.detect_backend", lambda: "cpu")
    assert recommend_inference_dtype() is None
    assert recommend_inference_dtype("tpu") is None


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((9, 0), "bfloat16"),  # Hopper
        ((8, 6), "bfloat16"),  # Ampere A10G
        ((8, 0), "bfloat16"),  # Ampere A100
        ((7, 5), "float16"),  # Turing T4
        ((7, 0), "float16"),  # Volta V100
        ((6, 1), None),  # Pascal — no fast half tensor cores
    ],
)
def test_dtype_by_capability(monkeypatch, capability, expected):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr("batcher.ml.gpu.detect_backend", lambda: "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    assert recommend_inference_dtype() == expected


def test_mps_uses_fp16(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr("batcher.ml.gpu.detect_backend", lambda: "mps")
    assert recommend_inference_dtype() == "float16"


def test_probe_failure_keeps_fp32(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr("batcher.ml.gpu.detect_backend", lambda: "cuda")

    def boom():
        raise RuntimeError("no driver")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    assert recommend_inference_dtype() is None


def test_accel_kwargs_places_and_casts_on_gpu(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr("batcher.ml.gpu.detect_backend", lambda: "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))
    kwargs = _pipeline_accel_kwargs()
    assert kwargs == {"device": 0, "torch_dtype": torch.bfloat16}


def test_accel_kwargs_empty_on_cpu(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr("batcher.ml.gpu.detect_backend", lambda: "cpu")
    assert _pipeline_accel_kwargs() == {}


@pytest.mark.parametrize(
    ("backend", "expected_device"),
    [("cuda", 0), ("rocm", 0), ("xpu", "xpu"), ("mps", "mps"), ("tpu", "xla")],
)
def test_accel_kwargs_places_on_any_accelerator(monkeypatch, backend, expected_device):
    """Vendor-agnostic device placement: NVIDIA/AMD -> int 0, Intel/Apple/TPU -> device
    string. Not only CUDA — the managed path works on any GPU type."""
    pytest.importorskip("torch")
    monkeypatch.setattr("batcher.ml.gpu.detect_backend", lambda: backend)
    # No dtype needed for this assertion; force it None so we isolate device placement.
    monkeypatch.setattr("batcher.ml.gpu.recommend_inference_dtype", lambda b=None: None)
    assert _pipeline_accel_kwargs() == {"device": expected_device}


def test_compile_helper_is_noop_off_or_on_cpu(monkeypatch):
    """`_maybe_compile_pipeline` never touches the model when compile is off or there is no
    GPU — a perf optimization must never break inference on the CPU/test host."""
    import dataclasses

    from batcher.config import active_config, config_context
    from batcher.ml.inference import _maybe_compile_pipeline

    class _Pipe:
        model = "orig"

    # off -> untouched
    base = active_config()
    off = base.replace(distributed=dataclasses.replace(base.distributed, torch_compile=False))
    with config_context(off):
        p = _Pipe()
        _maybe_compile_pipeline(p)
        assert p.model == "orig"

    # on but GPU-less host -> untouched (torch.cuda unavailable)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False, raising=False)
    p2 = _Pipe()
    _maybe_compile_pipeline(p2)
    assert p2.model == "orig"


def test_compile_skips_non_cnn_on_gpu(monkeypatch):
    """Only CNN (Conv2d) models are compiled — a text transformer (no conv) is left eager,
    because torch.compile regresses dynamic-shape text models."""
    torch = pytest.importorskip("torch")
    from batcher.ml.inference import _maybe_compile_pipeline

    class _Text(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

    class _Pipe:
        pass

    monkeypatch.setattr("torch.cuda.is_available", lambda: True, raising=False)
    p = _Pipe()
    p.model = _Text()
    _maybe_compile_pipeline(p)
    assert type(p.model).__name__ == "_Text"  # untouched — not a CNN
