"""Suite-wide fixtures.

The one cross-cutting concern every test shares: the process-global `MetadataHub`.
It accumulates learned statistics (cardinalities, selectivities, GPU utilization)
across executions so plans improve with use — but in a test process that makes
outcomes *order-dependent*: a test asserting on cardinality- or cost-driven plan
shape (join build-side choice, adaptive cardinalities, approximate quantiles) can be
perturbed by stats an earlier test recorded. Resetting the hub before each test makes
the suite deterministic regardless of collection order, without changing production
behavior (the reset only drops the cached in-process handle).

Learning *within* a single test (multiple `collect()`s in one function) is preserved
— the reset happens only between tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_DOCS_TESTS = Path(__file__).parent / "docs"


def pytest_configure(config):
    """Drop the platform's injected Ray runtime-env hook before any test starts Ray.

    A managed workspace (Anyscale) exports ``RAY_RUNTIME_ENV_HOOK``, and that hook merges
    the workspace's cluster-wide pip list into *every* runtime env Ray builds. If any entry
    in that list is unresolvable, every Ray worker dies in `RuntimeEnvSetupError` before it
    runs a line — and one entry is reliably unresolvable, because installing this project
    the way its own docs say to (``pip install 'batcher-engine[delta]'``) registers
    ``batcher-engine[delta]`` as a cluster dependency that no index can serve.

    The engine already pins ``pip: None`` in the runtime env it builds itself
    (`dist/executors/ray_runtime/lifecycle.py::_self_ship_runtime_env`), but the hook runs
    on *any* ``ray.init``, including the implicit one Ray Data does inside
    `bt.from_ray_dataset`. So the interop and distributed tests inherited the broken list
    from a code path the engine never sees.

    Dropping the hook for the test process only. Workers then use the node's base image,
    which is where the workspace's packages already live, so nothing is lost. This is the
    same defence the engine applies to a hook whose module is missing.
    """
    for var in ("RAY_RUNTIME_ENV_HOOK", "RAY_RUNTIME_ENV_PLUGINS"):
        os.environ.pop(var, None)


@pytest.fixture(autouse=True)
def _docs_run_like_a_reader(request, monkeypatch):
    """Doc examples execute the way a reader runs them: one process, no attached cluster.

    `resolve_distributed("auto", ...)` consults the *live* Ray session, which makes the docs
    suite order-dependent. Run it alone and every example executes locally, exactly as a
    reader sees it; run it after a suite that happened to start Ray (the io and distributed
    tests do) and the same examples suddenly route to a multi-node cluster.

    That is not hypothetical. A doc example that defines its own `Source` — the custom
    connector guide does, and it is the whole point of that page — cannot report a row
    count, so "auto" takes the *unknown size means assume large* branch and distributes it.
    The class is defined in the doc block, so it exists on no worker, and the page fails for
    a reason that has nothing to do with the page.

    Scoped by path here rather than in a `tests/docs/conftest.py`, because a second
    top-level module named `conftest` shadows `tests/differential/conftest.py` — and 174
    differential tests import `assert_same` from it by bare name.
    """
    if _DOCS_TESTS not in request.path.parents:
        return
    try:
        import ray
    except ImportError:
        return
    monkeypatch.setattr(ray, "is_initialized", lambda: False)


@pytest.fixture(autouse=True)
def _disable_event_log():
    """Turn the per-query event log off for the suite (env is import-time, so use config).

    The event log is on by default in production, but writing a JSON document per query
    to ``~/.batcher/logs`` would pollute the developer's home directory and add file I/O
    to every test. Disabling it keeps the suite fast and hermetic; a test that exercises
    the event log explicitly re-enables it against ``tmp_path``. No-op where the engine
    config can't be imported.
    """
    try:
        import dataclasses

        from batcher.config import active_config, set_config
    except Exception:
        yield
        return
    prev = active_config()
    set_config(prev.replace(observability=dataclasses.replace(prev.observability, event_log=False)))
    yield
    set_config(prev)


@pytest.fixture(autouse=True)
def _isolate_metadata_hub():
    """Reset the process-wide MetadataHub around every test for deterministic order.

    No-op (yields cleanly) in a pure-Python environment where Core can't be imported,
    so tests that don't touch the engine still run.
    """
    try:
        from batcher.core import reset_default_hub
    except Exception:
        yield
        return
    reset_default_hub()
    yield
    reset_default_hub()
