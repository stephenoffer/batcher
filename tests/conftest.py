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

import importlib.util
import os
import re
import shutil
import sys
from pathlib import Path

import pytest

_DOCS_TESTS = Path(__file__).parent / "docs"

# The heavy optional backends `ci.yml` deliberately does NOT install (it runs `.[dev]`):
# its own comment says the tests that need them "skip via importorskip", so the
# deterministic core gate must stay green with them absent. Any test that reaches one of
# these only at runtime — `collect(distributed=True)`, a `torch` autocast helper, a bare
# `import ray` in the body — fails instead of skipping unless guarded, and many are not.
_OPTIONAL_BACKENDS = ("ray", "torch", "tensorflow", "vllm", "cuda")


def _backend_available(name: str) -> bool:
    # `find_spec` rather than import (these are heavy) and tolerate a blocking meta-path
    # finder raising instead of returning None.
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


_MISSING_BACKENDS = frozenset(b for b in _OPTIONAL_BACKENDS if not _backend_available(b))


def _is_absent_backend_error(exc: BaseException | None) -> bool:
    """True if `exc`, or an error it wraps, is "an absent optional backend is not installed".

    Three shapes reach here: a bare ``import torch``/``import ray`` raises
    ``ModuleNotFoundError`` (``.name`` set by the import system); ``collect(distributed=True)``
    and the ML autocast/loader paths raise batcher's typed ``BackendError`` /
    ``MissingDependencyError`` naming the extra; and the same wrapped as the cause of a
    later error. Matched by class name + module name so this file imports nothing from
    ``batcher``. Only backends confirmed *absent* count, so a test failing for a real reason
    while the backend is installed is never silently skipped.
    """
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, ModuleNotFoundError):
            root = (exc.name or "").split(".")[0]
            if root in _MISSING_BACKENDS:
                return True
        if type(exc).__name__ in {"BackendError", "MissingDependencyError"}:
            msg = str(exc).lower()
            if any(b in msg for b in _MISSING_BACKENDS):
                return True
        exc = exc.__cause__ or exc.__context__
    return False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Turn an absent-optional-backend failure into a skip, per ci.yml's stated intent.

    The distributed / GPU / autoscale / torch-loader tests need a heavy backend `ci.yml`
    does not install. Most guard with ``importorskip``, but the ones that reach the backend
    only at runtime (``collect(distributed=True)``, a torch autocast helper, a bare
    ``import ray``) do not, so without it they *fail* rather than skip — reddening the exact
    core gate ci.yml runs. Converting that one error class to a skip here covers every such
    test (including modules that mix single-node and distributed cases, where a module-level
    ``importorskip`` would wrongly skip the single-node ones) from one place. It keys off
    backends *proven absent* at startup, so it is inert wherever the backend is installed —
    the dev env, the bench lineup, CI's own backend-bearing jobs — and a real regression
    there still fails.
    """
    outcome = yield
    if not _MISSING_BACKENDS or call.excinfo is None:
        return
    if not _is_absent_backend_error(call.excinfo.value):
        return
    report = outcome.get_result()
    if report.outcome == "failed":
        report.outcome = "skipped"
        absent = ", ".join(sorted(_MISSING_BACKENDS))
        reason = f"Skipped: optional backend not installed ({absent})"
        report.longrepr = (str(item.fspath), item.location[1] or 0, reason)


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


@pytest.fixture(autouse=True)
def _isolate_topology_cache():
    """Drop the windowed cluster-shape reads around every test.

    `scaling._LIVE_TTL_S` reuses `ray.nodes()` and the per-node free-CPU figures for 50 ms,
    which is what keeps a distributed query from making nineteen O(nodes) GCS round trips.
    A unit test stubs the topology and asserts on what the scheduler decides, and it runs in
    well under 50 ms — so without this the second test to stub a *different* cluster would be
    answered from the first one's snapshot. That is a test reading process state rather than
    the code under test, and it fails in whichever order happens to run second.

    Looked up in `sys.modules` rather than imported, for two reasons. Importing it would pull
    the whole package into a pure-Python test that never asked for it (half a second, once per
    session). And it cannot be needed: only code that imported the module can have populated
    the window, so clearing it after each test that did is enough to keep it out of the next.
    """
    yield
    scaling = sys.modules.get("batcher.dist.executors.ray_runtime.scaling")
    if scaling is not None:
        scaling._reset_topology_cache()


#: Env var naming a directory mounted on **every** node of the Ray cluster under test.
#: Takes precedence over the conventional mounts below.
SHARED_DIR_ENV = "BATCHER_TEST_SHARED_DIR"

#: Conventional cluster-wide mounts to fall back on when the env var is unset. These are the
#: shared volumes a managed Ray cluster provides; on a laptop and in CI neither exists, so the
#: fixtures below get `tmp_path` and behave exactly as they always did.
_SHARED_DIR_CANDIDATES = ("/mnt/cluster_storage", "/mnt/shared_storage")


def _shared_base() -> Path | None:
    """A directory every worker node can read, or `None` if there is no reason to think so.

    Existence plus writability is the whole test. It is a heuristic — a path that exists on
    the driver is not *proof* the workers mount it — but the failure mode is bounded: the
    worst case is the `FileNotFoundError` these tests already produced, so a wrong guess is
    never worse than not guessing.
    """
    for candidate in (os.environ.get(SHARED_DIR_ENV), *_SHARED_DIR_CANDIDATES):
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_dir() and os.access(path, os.W_OK):
            return path
    return None


def _safe_dirname(name: str) -> str:
    """A test id turned into a directory component that is safe to *glob*.

    A parametrized test is named `test_x[flight]`, and the readers under test open their
    input through a glob — where `[...]` is a character class, not two literal brackets. A
    directory named after the raw id therefore reads back as "matched no files", which is a
    confusing way to say "your path had metacharacters in it".
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)[:120]


@pytest.fixture
def cluster_tmp_path(tmp_path, request):
    """A scratch directory every Ray worker can open — `tmp_path` when they all share one.

    A distributed test that writes its fixture to pytest's `tmp_path` is writing to
    **driver-local** disk. On a single-node Ray that is fine, because the workers are the
    same machine. On a genuine multi-node cluster it is not: the worker opens a path that
    does not exist on its node and the test dies with `FileNotFoundError` from inside a Ray
    task, which reads like an engine defect and is not one.

    The cost of that is not hypothetical. `tests/integration/test_distributed.py` had never
    been run against a multi-node cluster — CI installs no Ray, and a local single-node Ray
    is the most it had ever seen. The first real-cluster run surfaced **two genuine defects**
    (a silently zero-row `iter_batches(distributed=True)` over Flight, and five monkeypatches
    dead since `_broadcast_max_bytes` gained a parameter) underneath **21** failures of
    exactly this fixture kind, which is a good way to have real findings dismissed as noise.
    With a cluster-visible directory the same file goes from 40 failed / 56 passed to 96
    passed.

    Set `BATCHER_TEST_SHARED_DIR` to override the mount chosen.
    """
    base = _shared_base()
    if base is None:
        yield tmp_path
        return
    scratch = base / "batcher-tests" / _safe_dirname(request.node.name)
    scratch.mkdir(parents=True, exist_ok=True)
    yield scratch
    shutil.rmtree(scratch, ignore_errors=True)


@pytest.fixture(scope="session")
def cluster_tmp_dir(tmp_path_factory):
    """Session-scoped [`cluster_tmp_path`], for a corpus a whole module builds once.

    Same contract and same reason; separate only because `tmp_path_factory` is
    session-scoped and a function-scoped fixture cannot be consumed from one.

    The directory is named per **xdist worker**, which the sibling above gets for free by
    naming itself after the test. Under `-n N` there are N worker *processes*, each with its
    own pytest session, so a single fixed name gave all of them one directory — and the first
    one to finish ran this teardown's `rmtree` on a corpus the others were still reading. It
    surfaced as `FileNotFoundError` on someone else's parquet, from inside a distributed sort,
    which reads exactly like an engine defect and is not one. Isolated per worker, the
    teardown can only ever remove that worker's own files.
    """
    base = _shared_base()
    if base is None:
        yield tmp_path_factory.mktemp("cluster")
        return
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    scratch = base / f"batcher-tests-session-{_safe_dirname(worker)}"
    scratch.mkdir(parents=True, exist_ok=True)
    yield scratch
    shutil.rmtree(scratch, ignore_errors=True)
