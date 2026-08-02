"""One accelerator predicate, and the single-node fallback that has to consult it.

Two layers ask "does this plan need a device?" for different reasons: the `api` router, to
decide whether `distributed="auto"` must fan out at all, and the `dist` dispatcher, before it
falls back to single-node. Both were answering it themselves, and the router's copy followed
only the `input` chain — so a GPU stage on the build side of a join was invisible to it and
the query got routed by input size alone, onto a CPU-only driver.

The fallback is the sharper failure. It is correct for CPU work (no splittable source means
no distributed data, so one node is the right plan), and silently wrong for a stage holding a
device, because the driver of a GPU cluster is routinely the one node without one.
"""

from __future__ import annotations

import warnings

import pytest

import batcher as bt
from batcher._internal.errors import PerformanceWarning
from batcher.plan.accelerator import plan_requests_accelerator

pytestmark = pytest.mark.unit


def _ds():
    return bt.from_pydict({"x": [1, 2, 3], "k": [1, 2, 3]})


def test_a_plain_plan_wants_no_accelerator():
    assert plan_requests_accelerator(_ds()._plan) is False
    assert plan_requests_accelerator(_ds().ml.map_batches(lambda b: b)._plan) is False
    assert plan_requests_accelerator(None) is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_gpus": 1},
        {"num_gpus": 0.25},
        # Ray reports only NVIDIA/AMD/Intel/MetaX as `GPU`; every other accelerator is a
        # custom resource and leaves `num_gpus == 0`, which is how the non-CUDA devices
        # became invisible to a `num_gpus`-only check.
        {"resources": {"TPU": 4}},
        {"resources": {"neuron_cores": 2}},
        {"resources": {"HPU": 8}},
    ],
)
def test_every_request_form_counts(kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        plan = _ds().ml.map_batches(lambda b: b, **kwargs)._plan
    assert plan_requests_accelerator(plan) is True


def test_a_device_stage_under_a_join_is_found():
    """The blind spot the router had: it walked the single `input` chain, so a device stage
    on the *build side* of a join did not exist as far as routing was concerned — and a
    pipeline that embeds its inference under a join is exactly one that must reach the
    cluster's devices."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        right = _ds().ml.map_batches(lambda b: b, num_gpus=1, output_columns=["x", "k"])
    joined = _ds().join(right, on="k")

    assert plan_requests_accelerator(joined._plan) is True
    # And the router now agrees, because it is the same function.
    from batcher.api.terminal.routing import _plan_has_gpu_stage

    assert _plan_has_gpu_stage(joined._plan) is True


def test_the_single_node_fallback_warns_for_a_device_stage(monkeypatch, recwarn):
    """The measured case: a shuffle beneath `map_batches` is a shape the dispatcher has no
    one-shot path for, and its intermediate is in-memory, so it lands in the fallback. On the
    4xT4 cluster that ran every batch on the driver's CPU with all four devices idle."""
    from batcher.dist import executor

    monkeypatch.setattr(
        "batcher._internal.hardware.devices.presence.local_accelerator_present", lambda: False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        plan = _ds().ml.map_batches(lambda b: b, num_gpus=1)._plan

    executor._warn_accelerator_stage_falls_back(plan, "an unsupported operator combination")

    messages = [str(w.message) for w in recwarn if w.category is PerformanceWarning]
    assert any("requested an accelerator" in m for m in messages), messages
    assert any("run on CPU" in m for m in messages), messages


def test_the_fallback_stays_quiet_without_a_device_stage(monkeypatch, recwarn):
    from batcher.dist import executor

    monkeypatch.setattr(
        "batcher._internal.hardware.devices.presence.local_accelerator_present", lambda: False
    )
    executor._warn_accelerator_stage_falls_back(_ds()._plan, "whatever")
    assert [w for w in recwarn if w.category is PerformanceWarning] == []


def test_the_fallback_stays_quiet_when_a_device_is_present_or_unreadable(monkeypatch, recwarn):
    """Only a positively established absence warns: a host whose devices cannot be read is
    not a host without devices, and a false warning on every GPU query would be worse than
    the silence this replaces."""
    from batcher.dist import executor

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        plan = _ds().ml.map_batches(lambda b: b, num_gpus=1)._plan

    for verdict in (True, None):
        monkeypatch.setattr(
            "batcher._internal.hardware.devices.presence.local_accelerator_present",
            lambda v=verdict: v,
        )
        executor._warn_accelerator_stage_falls_back(plan, "whatever")

    assert [
        str(w.message)
        for w in recwarn
        if w.category is PerformanceWarning and "requested an accelerator" in str(w.message)
    ] == []


def test_the_presence_probe_is_three_valued():
    """`None` is a real answer, not a failure: it is what keeps the warning off a host whose
    devices merely could not be read."""
    from batcher._internal.hardware.devices.presence import local_accelerator_present

    assert local_accelerator_present() in (True, False, None)
