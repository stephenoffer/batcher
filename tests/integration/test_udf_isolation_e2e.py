"""The UDF credential exposure, exercised through the real process pool.

`tests/unit/test_udf_isolation.py` tests the mechanism by calling `child_initializer`
directly. That is necessary but not sufficient: it would pass just as happily if the
initializer were never wired into the pool, which is exactly the "green gate, wrong
behaviour" failure this project keeps hitting. These tests run a real UDF in a real
forkserver child and read back what it could see.

They also pin the **limit** of the fix, deliberately. Isolation covers the process path
only; a UDF on the thread path runs inside the engine process and can read its
environment, because it is that process. Writing that down as a test is what stops the
next reader assuming a containment property the engine does not have.
"""

from __future__ import annotations

import dataclasses
import os
import sys

import pyarrow as pa
import pytest

from batcher.config import active_config, set_config
from batcher.core.udf.processes import run_map_processes, shutdown_pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _udf_probe import report_environment

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.platform == "win32", reason="forkserver/POSIX environment"),
]

SECRET = "SUPER-SECRET-VALUE"
HELPER = "/usr/local/bin/fetch-any-secret"


@pytest.fixture
def engine_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put a credential and the secret-fetching helper in the engine's environment."""
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", SECRET)
    monkeypatch.setenv("BATCHER_SECRET_COMMAND", HELPER)


@pytest.fixture
def isolation_mode(monkeypatch: pytest.MonkeyPatch):
    """Set `execution.udf_isolation` and give the test a fresh pool under it.

    The pool must be torn down on both sides: children apply isolation once at startup, so
    a pool left warm from another test would answer under the wrong regime.
    """
    original = active_config()

    def apply(mode: str) -> None:
        set_config(
            original.replace(execution=dataclasses.replace(original.execution, udf_isolation=mode))
        )
        shutdown_pool()

    try:
        yield apply
    finally:
        set_config(original)
        shutdown_pool()


def _run_in_workers() -> dict:
    """Run the probe UDF across the process pool and return its first row."""
    batches = [pa.record_batch({"a": pa.array([1, 2], type=pa.int64())})]
    results = run_map_processes(report_environment, batches, 2, "pyarrow")
    return results[0].to_pydict()


def test_a_worker_cannot_read_the_engines_credentials(
    engine_credentials: None, isolation_mode
) -> None:
    """The exploit, asserted to fail on the path it used to work on."""
    isolation_mode("env")
    row = _run_in_workers()
    assert row["secret"][0] == "<gone>", "a UDF worker still sees AWS_SECRET_ACCESS_KEY"
    assert row["helper"][0] == "<gone>", "a UDF worker still sees BATCHER_SECRET_COMMAND"
    assert row["pid"][0] != os.getpid(), "this did not actually run in a child process"


def test_opting_out_restores_the_old_behaviour(engine_credentials: None, isolation_mode) -> None:
    """`udf_isolation="none"` must be a genuine escape hatch, not a no-op flag.

    An embedder whose UDFs are as trusted as its own code, and who needs a variable the
    engine cannot know about, has to be able to turn this off. This test doubles as proof
    that the previous test measures the isolation rather than an unrelated environment
    quirk: same UDF, same pool, opposite result.
    """
    isolation_mode("none")
    row = _run_in_workers()
    assert row["secret"][0] == SECRET
    assert row["helper"][0] == HELPER


def test_the_thread_path_is_knowingly_not_covered(engine_credentials: None) -> None:
    """A UDF running on a thread reads the engine's environment, and always will.

    This is not a bug to be fixed later — it is what "in-process" means. Threads share the
    engine's address space, so there is no separate environment to scrub, and any claim to
    the contrary would be false. Pinned here so the limitation is discovered by reading the
    tests rather than by a security review.
    """
    batch = pa.record_batch({"a": pa.array([1], type=pa.int64())})
    row = report_environment(batch).to_pydict()
    assert row["secret"][0] == SECRET
    assert row["pid"][0] == os.getpid()
