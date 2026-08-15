"""A timed-out call is abandoned without costing the process its exit or its capacity.

Python cannot cancel a running call, so every timeout in the engine abandons one. Both
regressions pinned here come from *where* the abandoned call was left running, and neither
is visible to an assertion about a query's result:

* A `ThreadPoolExecutor` joins every worker it ever started at interpreter exit, so an
  abandoned call wedged the process after the query had already returned.
* A hung call parked in the module-level shared media pool held its worker forever, so once
  `workers` fetches had hung, every later download in the process timed out on a full queue
  however healthy its URL was — nulls and timeout errors for good data, permanently.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time

import pytest

from batcher._internal.concurrency.timeout import call_with_timeout, start_call
from batcher.ml.decode.stage import _bounded_map


def test_returns_the_calls_value():
    assert call_with_timeout(lambda: 2 + 2, timeout=5.0) == 4


def test_reraises_the_calls_error_with_its_type_intact():
    """`retry_on` tuples and `except` clauses must classify a guarded call as an unguarded one."""

    def boom() -> None:
        raise ConnectionResetError("upstream went away")

    with pytest.raises(ConnectionResetError, match="upstream went away"):
        call_with_timeout(boom, timeout=5.0)


def test_raises_timeout_error_with_the_callers_message():
    with pytest.raises(TimeoutError, match="fetching thing"):
        call_with_timeout(
            lambda: time.sleep(30), timeout=0.05, on_timeout=lambda: "fetching thing gave up"
        )


def test_runs_under_the_callers_context():
    """The guarded call must see the caller's `Config`, not the process default.

    A bare `threading.Thread` reads every `contextvars.ContextVar` at its default, which is
    how a pinned memory bound or spill directory silently reverts across a thread boundary.
    """
    import contextvars

    var: contextvars.ContextVar[str] = contextvars.ContextVar("probe", default="default")
    var.set("caller-set")
    assert call_with_timeout(var.get, timeout=5.0) == "caller-set"


def test_abandoned_call_does_not_hold_the_interpreter_open():
    """The regression a result-only assertion cannot see: the query finishes, the process hangs."""
    script = textwrap.dedent(
        """
        import time
        from batcher._internal.concurrency.timeout import call_with_timeout
        try:
            call_with_timeout(lambda: time.sleep(600), timeout=0.05)
        except TimeoutError:
            pass
        print("DONE", flush=True)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=90,  # a TimeoutExpired here IS the regression
    )
    assert "DONE" in proc.stdout
    assert proc.returncode == 0


def test_a_hung_call_does_not_starve_later_calls():
    """The shared media pool must not be permanently consumed by abandoned fetches."""
    workers = 2
    release = threading.Event()

    def hangs(_: int) -> str:
        release.wait(600)
        return "never"

    timed_out = list(
        _bounded_map(hangs, range(workers), workers, timeout=0.3, on_timeout=lambda: "TIMEDOUT")
    )
    assert timed_out == ["TIMEDOUT"] * workers

    # A healthy call, on the same shared pool, immediately afterwards.
    started = time.monotonic()
    healthy = list(
        _bounded_map(str.upper, ["ok"], workers, timeout=5.0, on_timeout=lambda: "TIMEDOUT")
    )
    assert healthy == ["OK"]
    assert time.monotonic() - started < 1.0
    release.set()


@pytest.mark.parametrize("timeout", [None, 10.0])
def test_bounded_map_keeps_workers_in_flight(timeout):
    """Both paths stay concurrent and order-preserving; the guarded one must not serialize."""
    workers, n, dwell = 4, 8, 0.4

    def slow(i: int) -> int:
        time.sleep(dwell)
        return i

    kwargs = {} if timeout is None else {"timeout": timeout, "on_timeout": lambda: None}
    started = time.monotonic()
    out = list(_bounded_map(slow, range(n), workers, **kwargs))
    elapsed = time.monotonic() - started

    assert out == list(range(n)), "results must stay in input order"
    assert elapsed < (n / workers) * dwell * 2.5, f"ran serially ({elapsed:.2f}s)"


def test_start_call_awaits_an_already_running_call():
    handle = start_call(lambda: 6 * 7)
    assert handle.result(5.0) == 42
