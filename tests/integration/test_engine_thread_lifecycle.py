"""The engine must survive a Python thread that used it and then exited.

# Why this is a subprocess test

The failure this pins is not an exception, it is a **`SIGSEGV` inside the allocator**:
after a thread that allocated through the engine exits, the next thread the process
creates faulted in `mi_thread_init`. A crash takes the whole interpreter down, so an
in-process assertion would report the run as an error with no attribution — and, worse,
would take every unrelated test in the same process with it. Each case therefore runs in a
fresh interpreter and asserts on its exit status.

# The shape that broke

Two threads, in sequence, each running a query and then exiting. The first always worked;
the second faulted before it reached any Batcher code, because the fault is in the
allocator's per-thread initialization rather than in anything the query does. Any embedding
that runs Batcher off the main thread produces this shape: a request handler, a data
loader's worker, `concurrent.futures`, or `iter_torch_batches(prefetch_batches>0)`, which
is where it was found.

The second case adds the other half: a *native* pool built after such a thread exits.
`column_ndv` is the smallest caller that builds rayon's global pool, so it spawns fresh
worker threads at exactly the moment the allocator's thread state has to be right.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.integration


def _run(body: str) -> subprocess.CompletedProcess:
    """Run `body` in a fresh interpreter and return the completed process."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=300,
    )


def _assert_clean(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, (
        f"exit {result.returncode} (negative means a fatal signal)\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr[-2000:]}"
    )
    assert result.stdout.strip().endswith("OK"), result.stdout


def test_queries_on_successive_threads_do_not_crash():
    _assert_clean(
        _run(
            """
            import threading
            import batcher as bt

            ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0]})
            for i in range(3):
                t = threading.Thread(target=lambda: ds.collect())
                t.start()
                t.join()
                print("thread", i, "ok", flush=True)
            print("OK")
            """
        )
    )


def test_native_pool_starts_after_a_worker_thread_exits():
    _assert_clean(
        _run(
            """
            import threading
            import pyarrow as pa
            import batcher as bt
            from batcher.core.stats import column_ndv

            ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0]})
            t = threading.Thread(target=lambda: ds.collect())
            t.start()
            t.join()

            batches = pa.table({"k": [1, 2, 3, 1, 2, 3]}).to_batches()
            ndv = column_ndv(batches, ["k"])
            assert 2.0 <= ndv["k"] <= 4.0, ndv
            print("OK")
            """
        )
    )
