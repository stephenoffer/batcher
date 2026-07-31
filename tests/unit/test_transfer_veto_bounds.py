"""The transfer veto's two guard rails, which matter more than the verdict itself.

A veto is a *removal*: it takes away a path the engine would otherwise have used. That makes
its failure mode asymmetric — a wrong verdict does not produce a wrong answer, it silently
disables an optimization a fleet had been winning with, and nothing in the result says so.

Two things bound it, and both are tested here rather than assumed. It needs a device model, so
an unlabelled fleet is never vetoed. And it steps aside the moment this fleet has *measured* a
GPU/CPU crossover, because a timing from this hardware outranks a model carrying a
CPU-bandwidth constant that may not describe it.
"""

from __future__ import annotations

import importlib

import pytest

from batcher.kyber.gpu import record_backend_timing
from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend

pytestmark = pytest.mark.unit

policy = importlib.import_module("batcher.kyber.gpu.policy")


def _hub_with_crossover(device: str) -> MetadataHub:
    """A hub carrying enough GPU and CPU timings for a crossover to be identifiable."""
    hub = MetadataHub(InProcessBackend())
    # GPU: high fixed cost, low per-row cost. CPU: the opposite. That shape is what
    # `learned_gpu_min_rows` needs to solve for a crossover at all, and the fit has its own
    # sample-count and spread gates — hence a spread of sizes rather than a token few.
    for i in range(1, 25):
        rows = i * 100_000
        record_backend_timing(hub, "gpu", rows, 500.0 + rows * 0.0001, device)
        record_backend_timing(hub, "cpu", rows, 5.0 + rows * 0.001)
    return hub


def test_the_veto_fires_on_a_link_too_slow_to_be_worth_it() -> None:
    reason = policy._transfer_veto("NVIDIA_TESLA_T4", working_set_gb=10.0, rows=100_000_000)
    assert reason is not None
    assert "CPU wins" in reason


def test_an_unlabelled_fleet_is_never_vetoed() -> None:
    # `decide_gpu_backend` gates on a truthy model name; without one there is no link to
    # price and the veto must not be formed at all.
    assert policy._transfer_veto("", working_set_gb=10.0, rows=100_000_000) is None
    assert policy._transfer_veto("MADE_UP", working_set_gb=10.0, rows=100_000_000) is None


def test_a_measured_crossover_disables_the_model(monkeypatch) -> None:
    from batcher.kyber.gpu.adaptive import learned_gpu_min_rows

    hub = _hub_with_crossover("NVIDIA_TESLA_T4")
    assert learned_gpu_min_rows(hub, "NVIDIA_TESLA_T4") is not None, (
        "the fixture must actually produce a crossover, or this test proves nothing"
    )

    calls: list[str] = []

    def _veto(accelerator_type, working_set_gb, rows):  # pragma: no cover - must not be called
        calls.append(accelerator_type)
        return "vetoed"

    monkeypatch.setattr(policy, "_transfer_veto", _veto)

    # A fleet that has timed both backends decides on those timings; the model steps aside.
    decision = policy.decide_gpu_backend(
        _aggregate_plan(),
        [],
        hub,
        gpu_count=4,
        gpu_memory_gb=16.0,
        accelerator_type="NVIDIA_TESLA_T4",
    )
    assert calls == [], "a measured fleet must not be second-guessed by a constant"
    assert decision.reason  # the decision still explains itself, whichever way it went


def _aggregate_plan():
    """A single-key aggregate over an in-memory source: the shape the GPU backend routes."""
    import batcher as bt

    return (
        bt.from_pydict({"g": [1, 1, 2], "v": [10, 20, 30]})
        .group_by("g")
        .agg(total=bt.col("v").sum())
        ._plan
    )


def test_a_forced_request_is_never_vetoed(monkeypatch) -> None:
    def _veto(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("force=True honors the user past every model")

    monkeypatch.setattr(policy, "_transfer_veto", _veto)
    # The routing may still send this to the CPU for *memory* reasons, which is a different
    # and legitimate decision. What must not happen is the veto being consulted at all.
    policy.decide_gpu_backend(
        _aggregate_plan(),
        [],
        None,
        gpu_count=4,
        force=True,
        gpu_memory_gb=16.0,
        accelerator_type="NVIDIA_TESLA_T4",
    )
