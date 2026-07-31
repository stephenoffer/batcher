"""A tensor-parallel group wider than the devices the worker holds.

The failure this pins is the expensive one: a stage scheduled with one GPU and told to build a
four-way group does not raise. The engine waits for peers that were never scheduled, holding
its slot, and the job looks hung rather than misconfigured. It is also the easiest mistake to
make on a multi-GPU node, where the node has the eight cards the degree implies and the *task*
does not.
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import PerformanceWarning
from batcher.ml.llm.engines import parallelism

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _unwarn(monkeypatch):
    """The warning fires once per process; reset it so each test sees its own."""
    monkeypatch.setattr(parallelism, "_TP_WARNED", False)


def _visible(monkeypatch, count: int) -> None:
    monkeypatch.setattr(parallelism, "local_device_count", lambda: count)


def test_a_group_wider_than_the_worker_is_called_out(monkeypatch):
    _visible(monkeypatch, 1)
    with pytest.warns(PerformanceWarning, match="this worker can see 1 device"):
        parallelism.warn_about_tensor_parallelism(4, 10.0, 80.0, "NVIDIA H100 80GB HBM3")


def test_the_message_names_both_fixes(monkeypatch):
    _visible(monkeypatch, 2)
    with pytest.warns(PerformanceWarning) as caught:
        parallelism.warn_about_tensor_parallelism(8, 10.0, 80.0, "NVIDIA H100 80GB HBM3")
    message = str(caught[0].message)
    assert "num_gpus" in message
    assert "tensor_parallel_size=2" in message
    assert "2 devices" in message


def test_a_group_that_fits_the_worker_says_nothing_about_the_count(monkeypatch, recwarn):
    _visible(monkeypatch, 8)
    parallelism.warn_about_tensor_parallelism(4, 10.0, 80.0, "NVIDIA H100 80GB HBM3")
    assert [w for w in recwarn if "worker can see" in str(w.message)] == []


def test_degree_one_is_never_a_count_problem(monkeypatch, recwarn):
    # A single-device group is built from the device the worker has by definition.
    _visible(monkeypatch, 0)
    parallelism.warn_about_tensor_parallelism(1, 10.0, 80.0, "NVIDIA H100 80GB HBM3")
    assert [w for w in recwarn if "worker can see" in str(w.message)] == []


def test_an_unreadable_device_count_does_not_invent_a_violation(monkeypatch, recwarn):
    # A driverless host reports zero devices; warning that a group of four exceeds zero would
    # fire on every CPU-only driver process that ever builds an engine factory.
    _visible(monkeypatch, 0)
    parallelism.warn_about_tensor_parallelism(4, 10.0, 80.0, "NVIDIA H100 80GB HBM3")
    assert [w for w in recwarn if "worker can see" in str(w.message)] == []


def test_the_count_check_outranks_the_interconnect_advice(monkeypatch):
    # Both apply on a one-GPU L4 worker asked for TP=2. The count is the certain failure and
    # the throughput note would be advice about a group that never starts.
    _visible(monkeypatch, 1)
    with pytest.warns(PerformanceWarning) as caught:
        parallelism.warn_about_tensor_parallelism(2, 1.0, 24.0, "NVIDIA L4")
    assert "worker can see" in str(caught[0].message)


def test_the_live_probe_is_zero_without_a_device() -> None:
    # The real probe, not the monkeypatched one: it must degrade to 0 rather than raise on a
    # host with no torch or no driver, which is every CI runner.
    assert parallelism.local_device_count() >= 0


def test_a_group_wider_than_the_model_needs_is_called_out(monkeypatch):
    """The quiet mistake: nothing fails, the group just pays an all-reduce on every layer for
    memory nobody uses, while the devices it consumed would each have served their own
    sequences at full rate."""
    _visible(monkeypatch, 8)
    with pytest.warns(PerformanceWarning) as caught:
        parallelism.warn_about_tensor_parallelism(8, 20.0, 80.0, "NVIDIA H100 80GB HBM3", needed=2)
    message = str(caught[0].message)
    assert "fit a group of 2" in message
    assert "4 replicas" in message


def test_the_footprint_bound_never_advises_shrinking(monkeypatch, recwarn):
    # Without a measured `needed` the bound ignores the KV cache, so it under-estimates: a
    # group trimmed to it would hold the weights and serve nothing.
    _visible(monkeypatch, 8)
    parallelism.warn_about_tensor_parallelism(8, 1.0, 80.0, "NVIDIA H100 80GB HBM3")
    assert [w for w in recwarn if "fit a group of" in str(w.message)] == []


def test_a_group_the_model_actually_needs_is_left_alone(monkeypatch, recwarn):
    _visible(monkeypatch, 8)
    parallelism.warn_about_tensor_parallelism(2, 100.0, 80.0, "NVIDIA H100 80GB HBM3", needed=2)
    assert [w for w in recwarn if "fit a group of" in str(w.message)] == []
