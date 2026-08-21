"""Every symbol and call signature the GPU backend depends on actually resolves.

The GPU backend is written to fall back: an unsupported shape, a device out of memory, a lost
worker, a cluster with no GPU — all of them return `None` and the CPU engine answers the query.
That is right, and it is also the reason two whole-path outages shipped here unnoticed.

* `_with_gpu_capacity` imported `await_autoscale` from a module that has never defined it, so
  it raised `ImportError` on its first line. Every multi-device fan-out — aggregate, join and
  union — was dead, and every GPU query quietly ran on one device or on the host.
* `_try_sharded_join` called `sharded_gpu_join` without its required `broadcast` argument, so
  that fan-out raised `TypeError` into the same handler with the same result.

Neither is visible in a result, because the result is correct either way; both cost most of the
cluster. Nothing else in the suite can catch them, because every other GPU test either runs the
translator directly (never touching the dispatch wiring) or asserts on rows (which the fallback
produces identically). These assertions are deliberately about *wiring* rather than behavior:
the import resolves, the signature matches, the defect is logged as a defect.
"""

from __future__ import annotations

import inspect
import logging

import pytest

pytestmark = pytest.mark.unit


def test_autoscale_helpers_resolve_where_the_backend_imports_them():
    """The three names `_with_gpu_capacity` imports must exist where it imports them from."""
    from batcher.dist.executors.ray_runtime import (
        await_autoscale,
        release_autoscale,
        request_autoscale,
    )

    assert callable(await_autoscale)
    assert callable(release_autoscale)
    assert callable(request_autoscale)


def test_with_gpu_capacity_reaches_its_run_callback(monkeypatch):
    """A stand-in for the whole outage: the function must get past its own imports.

    Asserted by giving it a `run` that records it was called. Before the import fix this never
    ran, on any cluster, for any query.

    The three capacity helpers are stubbed rather than left to the ambient cluster. Off Ray
    they *are* no-ops and the callback runs, which is what this used to rely on — but "off
    Ray" is a property of the **process**, not of this test: a neighbour that connected to a
    cluster makes the request real, and on a GPU-less one it waits, finds nothing, and returns
    `None` instead of reaching `run`. The test then fails for a reason it is not about, and
    only when something else ran first. What it asserts is the wiring, so the wiring is what
    it should control.
    """
    from batcher.api.terminal.gpu_backend import fanout
    from batcher.api.terminal.gpu_backend.fanout import _with_gpu_capacity
    from batcher.dist.executors import ray_runtime
    from batcher.dist.gpu import dispatch
    from batcher.kyber.gpu.policy import GpuDecision

    monkeypatch.setattr(ray_runtime, "request_autoscale", lambda *_a, **_k: None)
    monkeypatch.setattr(ray_runtime, "release_autoscale", lambda *_a, **_k: None)
    monkeypatch.setattr(ray_runtime, "await_autoscale", lambda *_a, **_k: None)
    monkeypatch.setattr(dispatch, "await_gpu_admission", lambda *_a, **_k: 1)
    assert fanout is not None  # the module under test imported at all

    seen = []
    decision = GpuDecision(True, False, "test", 1, 1)
    result = _with_gpu_capacity(1, decision, lambda live: seen.append(live) or "ran")
    assert result == "ran"
    assert seen


@pytest.mark.parametrize(
    ("caller", "callee"),
    [
        ("_try_sharded_aggregate", "batcher.dist.gpu.aggregate:sharded_gpu_aggregate"),
        ("_try_sharded_join", "batcher.dist.gpu.join:sharded_gpu_join"),
        ("_try_sharded_union", "batcher.dist.gpu.union:sharded_gpu_union"),
        ("_try_tree", "batcher.dist.gpu.tree:sharded_gpu_tree"),
    ],
)
def test_every_fanout_is_called_with_all_its_required_arguments(caller, callee):
    """Each fan-out entry point supplies every keyword its target has no default for.

    Read off the source rather than by running it, because running it needs a cluster — and the
    version of this that needed a cluster is precisely why the missing `broadcast` argument sat
    undetected: the only environment that would have raised was the one nothing tested in.
    """
    import importlib

    from batcher.api.terminal.gpu_backend import fanout as gpu_backend

    module_name, attr = callee.split(":")
    target = getattr(importlib.import_module(module_name), attr)
    required = {
        name
        for name, p in inspect.signature(target).parameters.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is inspect.Parameter.empty
    }
    source = inspect.getsource(getattr(gpu_backend, caller))
    missing = {name for name in required if f"{name}=" not in source}
    assert not missing, f"{caller} never passes {sorted(missing)} to {attr}"


class TestFailureClassification:
    """A broken backend must read as broken, not as a device that declined."""

    @pytest.mark.parametrize("exc", [ImportError("x"), AttributeError("x"), TypeError("x")])
    def test_a_defect_is_logged_as_a_warning(self, exc, caplog):
        from batcher.api.terminal.gpu_backend.failure import note_gpu_failure

        with caplog.at_level(logging.WARNING, logger="batcher.api"):
            note_gpu_failure("test", exc)
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    @pytest.mark.parametrize("exc", [MemoryError("out of memory"), RuntimeError("no device")])
    def test_an_ordinary_decline_stays_quiet(self, exc, caplog):
        """A device that ran out of memory is the case the fallback exists for. It must not
        warn, or the signal above becomes noise nobody reads."""
        from batcher.api.terminal.gpu_backend.failure import note_gpu_failure

        with caplog.at_level(logging.DEBUG, logger="batcher.api"):
            note_gpu_failure("test", exc)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
