"""Intel XPU VRAM probing — the OOM-guard branch that was missing for Intel GPUs.

`gpu_vram_gb`/`sample_gpu_vram_fraction` claimed XPU coverage but had no XPU path, so the
throughput autobatcher's predictive VRAM cap was inert on Intel GPUs and the hill-climb
grew until a hard OOM. These fake a `torch.xpu` and assert the branch is taken. Needs torch
importable (CPU-only is fine); no real accelerator.
"""

from __future__ import annotations

import types

import pytest


def _fake_xpu(total: int, reserved: int) -> types.SimpleNamespace:
    props = types.SimpleNamespace(total_memory=total)
    return types.SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda i: props,
        memory_reserved=lambda i: reserved,
    )


def test_gpu_vram_gb_reads_xpu_total(monkeypatch):
    torch = pytest.importorskip("torch")
    import batcher.ml.gpu as gpu

    monkeypatch.setattr(gpu, "_vram_handle", lambda: None)  # skip NVML
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "xpu", _fake_xpu(total=8 * (1 << 30), reserved=0), raising=False)
    assert gpu.gpu_vram_gb() == pytest.approx(8.0)


def test_sample_gpu_vram_fraction_reads_xpu_reserved(monkeypatch):
    torch = pytest.importorskip("torch")
    import batcher.ml.gpu as gpu

    monkeypatch.setattr(gpu, "_vram_handle", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch, "xpu", _fake_xpu(total=10 * (1 << 30), reserved=3 * (1 << 30)), raising=False
    )
    assert gpu.sample_gpu_vram_fraction() == pytest.approx(0.3)
