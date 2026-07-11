"""First-class CPU inference threading — container-aware, not host-sized."""

from __future__ import annotations

import sys
import types

import pytest

from batcher.ml import inference

pytestmark = pytest.mark.unit


def test_cpu_inference_thread_target_honors_omp_env(monkeypatch):
    # Ray sets OMP_NUM_THREADS to the actor's num_cpus; an explicit value wins so the
    # per-actor allocation is respected rather than every actor grabbing the whole container.
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert inference._cpu_inference_thread_target() == 3
    # A non-numeric / empty value falls back to the container's usable cores.
    monkeypatch.setenv("OMP_NUM_THREADS", "")
    from batcher._internal import hardware

    monkeypatch.setattr(hardware, "available_cpu_count", lambda: 7)
    assert inference._cpu_inference_thread_target() == 7


def test_configure_cpu_inference_threads_only_lowers(monkeypatch):
    state = {"n": 32}
    fake_torch = types.ModuleType("torch")
    fake_torch.get_num_threads = lambda: state["n"]
    fake_torch.set_num_threads = lambda k: state.__setitem__("n", k)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("OMP_NUM_THREADS", "4")

    # Host-sized default (32) is lowered to the actor's 4 usable cores.
    assert inference._configure_cpu_inference_threads() == 4
    assert state["n"] == 4
    # Never *raises* above an already-lower setting (2 < 4 → left alone).
    state["n"] = 2
    assert inference._configure_cpu_inference_threads() == 4
    assert state["n"] == 2


def test_configure_is_safe_without_torch(monkeypatch):
    # No torch installed → a best-effort no-op, never an exception into the inference path.
    monkeypatch.setitem(sys.modules, "torch", None)
    assert inference._configure_cpu_inference_threads() is None
