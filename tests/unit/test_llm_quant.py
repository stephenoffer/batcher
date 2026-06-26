"""Zero-config FP8-by-GPU quantization for vLLM batch inference.

`recommend_quantization` picks FP8 only on native-FP8 GPUs (Ada/Hopper); `vllm_engine`'s
`quantization="auto"` resolves through it, with an explicit value always winning.
"""

from __future__ import annotations

import pytest

from batcher.ml.gpu import recommend_quantization
from batcher.ml.llm import _with_auto_quant

pytestmark = pytest.mark.unit


def test_non_cuda_backend_keeps_native_precision(monkeypatch):
    monkeypatch.setattr("batcher.ml.gpu.detect_backend", lambda: "mps")
    assert recommend_quantization() is None
    monkeypatch.setattr("batcher.ml.gpu.detect_backend", lambda: "cpu")
    assert recommend_quantization() is None


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((9, 0), "fp8"), ((8, 9), "fp8"), ((8, 6), None), ((8, 0), None), ((7, 5), None)],
)
def test_fp8_only_on_native_fp8_gpus(monkeypatch, capability, expected):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr("batcher.ml.gpu.detect_backend", lambda: "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    assert recommend_quantization() == expected


def test_probe_failure_keeps_native_precision(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr("batcher.ml.gpu.detect_backend", lambda: "cuda")

    def boom():
        raise RuntimeError("no driver")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    assert recommend_quantization() is None


def test_auto_quant_resolves_through_probe(monkeypatch):
    monkeypatch.setattr("batcher.ml.gpu.recommend_quantization", lambda: "fp8")
    assert _with_auto_quant("auto", {}) == {"quantization": "fp8"}
    monkeypatch.setattr("batcher.ml.gpu.recommend_quantization", lambda: None)
    assert _with_auto_quant("auto", {}) == {}


def test_explicit_quantization_wins_over_auto(monkeypatch):
    monkeypatch.setattr("batcher.ml.gpu.recommend_quantization", lambda: "fp8")
    # An explicit engine_kwargs value is never overridden by the auto pick.
    assert _with_auto_quant("auto", {"quantization": "awq"}) == {"quantization": "awq"}
    # A literal (non-"auto") value is used verbatim, no probe.
    assert _with_auto_quant("awq", {}) == {"quantization": "awq"}
    assert _with_auto_quant(None, {}) == {}
