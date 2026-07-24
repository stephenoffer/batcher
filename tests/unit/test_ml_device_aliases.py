"""Accelerator and dtype name resolution — the explicit-selection surface.

Detection (`torch_device`) already knows Trainium/Inferentia/Gaudi and FP8; these assert a
user can *name* them at `ds.ml.infer(device=...)` / `dtype=...`, which was blocked before.
Pure string mapping — no torch, no accelerator.
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import PlanError
from batcher.ml.devices import resolve_device, resolve_dtype


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("neuron", "xla"),
        ("trainium", "xla"),
        ("inferentia", "xla"),
        ("hpu", "hpu"),
        ("gaudi", "hpu"),
    ],
)
def test_new_accelerator_names_resolve(name, expected):
    assert resolve_device(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("fp8", "float8_e4m3fn"),
        ("float8", "float8_e4m3fn"),
        ("float8_e4m3fn", "float8_e4m3fn"),
        ("e4m3", "float8_e4m3fn"),
        ("float8_e5m2", "float8_e5m2"),
        ("e5m2", "float8_e5m2"),
    ],
)
def test_fp8_dtype_aliases_resolve(name, expected):
    assert resolve_dtype(name) == expected


def test_existing_names_still_resolve():
    assert resolve_device("cuda") == "cuda"
    assert resolve_dtype("bf16") == "bfloat16"


def test_an_unknown_device_still_raises():
    with pytest.raises(PlanError):
        resolve_device("quantum")


def test_default_dtype_uses_bfloat16_on_ampere(monkeypatch):
    """The GPU default follows recommend_inference_dtype, not a blanket float16."""
    import batcher.ml.devices as devices
    import batcher.ml.gpu as gpu

    monkeypatch.setattr(devices, "resolve_device", lambda d=None: "cuda")
    monkeypatch.setattr(gpu, "detect_backend", lambda: "cuda")
    monkeypatch.setattr(gpu, "recommend_inference_dtype", lambda b=None: "bfloat16")
    assert devices.default_dtype("cuda") == "bfloat16"
    monkeypatch.setattr(gpu, "recommend_inference_dtype", lambda b=None: None)
    assert devices.default_dtype("cuda") == "float16"  # falls back when half gives no gain


def test_default_dtype_is_float32_on_cpu():
    from batcher.ml.devices import default_dtype

    assert default_dtype("cpu") == "float32"
