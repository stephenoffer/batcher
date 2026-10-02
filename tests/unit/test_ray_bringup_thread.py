"""A cold Ray bring-up runs on a thread that outlives the pipeline thread that asked for it.

`ray.init` starts a local cluster's GCS and raylet with `PR_SET_PDEATHSIG = SIGKILL`, which
Linux delivers when the spawning *thread* exits. A distributed pipeline run from a worker
thread therefore killed the cluster it had just started as soon as it finished, and Ray then
terminated the whole driver. `bring_up_outliving_caller` moves a cold bring-up onto one
process-lifetime thread. Ray-free: a stand-in reports whether Ray is up, and the assertions
are about which thread ran the bring-up and whether that thread is still alive.
"""

from __future__ import annotations

import contextvars
import threading
from types import SimpleNamespace

import pytest

from batcher.dist.executors.ray_runtime.readiness import bring_up_outliving_caller

pytestmark = pytest.mark.unit

_COLD = SimpleNamespace(is_initialized=lambda: False)
_WARM = SimpleNamespace(is_initialized=lambda: True)


def _from_a_short_lived_thread(ray, fn, lock=None):
    """Call the bring-up from a thread that exits right after, as a pipeline thread does."""
    out: dict = {}

    def caller() -> None:
        try:
            out["value"] = bring_up_outliving_caller(ray, lock or threading.Lock(), fn)
        except BaseException as exc:
            out["error"] = exc

    t = threading.Thread(target=caller)
    t.start()
    t.join(timeout=30)
    assert not t.is_alive()
    return out


def test_a_cold_bring_up_runs_on_a_thread_that_outlives_its_caller():
    ran_on: list[threading.Thread] = []
    out = _from_a_short_lived_thread(_COLD, lambda: ran_on.append(threading.current_thread()))
    assert "error" not in out
    (thread,) = ran_on
    assert thread is not threading.main_thread()
    # The caller has exited; the thread the cluster's processes would be parented to has not.
    assert thread.is_alive()
    assert thread.daemon, "a non-daemon thread would block interpreter exit"


def test_every_cold_bring_up_shares_the_one_lasting_thread():
    ran_on: list[threading.Thread] = []
    for _ in range(3):
        _from_a_short_lived_thread(_COLD, lambda: ran_on.append(threading.current_thread()))
    assert len(set(ran_on)) == 1


def test_a_warm_bring_up_stays_on_the_calling_thread():
    callers: list[threading.Thread] = []
    ran_on: list[threading.Thread] = []

    def fn() -> None:
        ran_on.append(threading.current_thread())

    def caller() -> None:
        callers.append(threading.current_thread())
        bring_up_outliving_caller(_WARM, threading.Lock(), fn)

    t = threading.Thread(target=caller)
    t.start()
    t.join(timeout=30)
    assert ran_on == callers


def test_the_main_thread_brings_ray_up_itself():
    ran_on: list[threading.Thread] = []
    bring_up_outliving_caller(
        _COLD, threading.Lock(), lambda: ran_on.append(threading.current_thread())
    )
    assert ran_on == [threading.main_thread()]


def test_the_callers_context_value_and_error_cross_over():
    marker: contextvars.ContextVar[str] = contextvars.ContextVar("marker", default="unset")

    def read_marker() -> str:
        return marker.get()

    def caller_with_marker(out: dict) -> None:
        marker.set("caller")
        out["value"] = bring_up_outliving_caller(_COLD, threading.Lock(), read_marker)

    out: dict = {}
    t = threading.Thread(target=caller_with_marker, args=(out,))
    t.start()
    t.join(timeout=30)
    assert out["value"] == "caller"

    def boom() -> None:
        raise ConnectionError("head did not answer")

    failed = _from_a_short_lived_thread(_COLD, boom)
    assert isinstance(failed.get("error"), ConnectionError)


def test_the_lock_is_held_while_the_bring_up_runs():
    lock = threading.Lock()
    held: list[bool] = []
    _from_a_short_lived_thread(_COLD, lambda: held.append(lock.locked()), lock)
    bring_up_outliving_caller(_WARM, lock, lambda: held.append(lock.locked()))
    assert held == [True, True]
    assert not lock.locked()
