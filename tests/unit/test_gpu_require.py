"""`distributed.gpu_require` turns an explicit `backend="gpu"` fallback into an error.

`backend="gpu"` is always safe by contract: a declined shape, a device out of memory or a
GPU-less cluster answers on the CPU engine with the same rows. For a benchmark or a capacity
test that safety is the hazard, because the only trace a silent CPU run leaves is a timing
reported for a device that never ran. The flag makes the fallback refusable without changing
what any device run computes.

These run on a machine with no visible device, which is the one decline every CI host can
reproduce: the fallback is real here, so the default half of each pair is a positive control
that the flag, not the environment, is what raises.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import BackendError
from batcher.api.terminal.gpu_backend.audit import gpu_ledger, reset_gpu_ledger
from batcher.config import option_context

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean():
    reset_gpu_ledger()
    yield
    reset_gpu_ledger()


def _ds() -> bt.Dataset:
    return (
        bt.from_pydict({"k": [1, 1, 2], "v": [1.0, 2.0, 3.0]})
        .group_by("k")
        .agg(s=bt.col("v").sum())
    )


def test_the_default_falls_back_to_the_cpu_engine():
    """The control: without the flag this host's missing device is a silent CPU answer."""
    out = _ds().sort("k").collect(backend="gpu").to_pydict()
    assert out == {"k": [1, 2], "s": [3.0, 3.0]}
    assert gpu_ledger().declined.get("no visible device", 0) >= 1


def test_gpu_require_refuses_the_fallback_and_names_the_reason():
    with option_context("distributed.gpu_require", True), pytest.raises(BackendError) as info:
        _ds().collect(backend="gpu")
    assert "no visible device" in str(info.value)
    # The ledger records the decline either way, so it agrees with what was raised.
    assert gpu_ledger().declined.get("no visible device", 0) >= 1


def test_gpu_require_leaves_auto_to_kyber():
    """`backend="auto"` asks for Kyber's choice, not for the device, so it still falls back."""
    with option_context("distributed.gpu_require", True):
        out = _ds().sort("k").collect(backend="auto").to_pydict()
    assert out == {"k": [1, 2], "s": [3.0, 3.0]}


def test_gpu_require_does_not_touch_the_cpu_backend():
    with option_context("distributed.gpu_require", True):
        out = _ds().sort("k").collect(backend="cpu").to_pydict()
    assert out == {"k": [1, 2], "s": [3.0, 3.0]}
