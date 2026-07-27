"""Bounded admission: the slot pool, the fair-share width, and the deadlock it avoids.

# What these tests do and do not establish

They establish that the **mechanism** is correct: slots are never exceeded, waiters are
released in order, a full queue raises instead of growing, a failed query returns its slot,
and a nested `collect()` inside a UDF does not deadlock against its own outer query.

They do **not** establish that this fixes the measured throughput inversion
(`BENCHMARK_RESULTS.md`: 1 client 124 QPS, 16 clients 88 QPS, p50 7.6 ms -> 178 ms). That
is a performance claim and needs `benchmarks/concurrency/` against a real workload on a
quiet box. It was not measurable here: the Python control plane is GIL-bound, so concurrent
`collect()` calls on small in-memory data serialize before they ever contend for cores —
every grant in a live 8-thread run reported `concurrent == 1`.

Saying so matters more than the tests below. An unverified perf fix that everyone assumes
is verified is worse than an open ticket.
"""

from __future__ import annotations

import threading
import time

import pytest

from batcher.carbonite.policies.concurrency import (
    AdmissionTimeout,
    ConcurrencyLimiter,
    process_limiter,
    reset_process_limiter,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_limiter():
    reset_process_limiter()
    yield
    reset_process_limiter()


class TestSlotBounding:
    def test_slots_are_never_exceeded(self) -> None:
        """The property the whole thing exists for, measured by peak occupancy."""
        limiter = ConcurrencyLimiter(3, cores=12)
        peak = 0
        live = 0
        lock = threading.Lock()

        def worker() -> None:
            nonlocal peak, live
            limiter.acquire()
            try:
                with lock:
                    live += 1
                    peak = max(peak, live)
                time.sleep(0.02)  # hold the slot long enough to genuinely overlap
            finally:
                with lock:
                    live -= 1
                limiter.release()

        threads = [threading.Thread(target=worker) for _ in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert peak <= 3, f"{peak} queries ran at once against a 3-slot limiter"
        assert peak > 1, "the test did not actually overlap; it proves nothing"
        assert limiter.active == 0, "a slot was leaked"

    def test_unbounded_is_the_default_and_costs_nothing(self) -> None:
        # `max_concurrent_queries=0` must be a true bypass, not a limiter with a big
        # number — an unconfigured deployment should not acquire a lock per query.
        limiter = ConcurrencyLimiter(0)
        grant = limiter.acquire()
        assert grant.workers == 0
        assert grant.concurrent == 1
        limiter.release()

    def test_a_raising_query_returns_its_slot(self) -> None:
        """A slot leaked on the error path wedges the process after N failures."""
        limiter = ConcurrencyLimiter(1, cores=8)
        for _ in range(5):
            limiter.acquire()
            try:
                raise RuntimeError("query failed")
            except RuntimeError:
                pass
            finally:
                limiter.release()
        assert limiter.active == 0
        limiter.acquire()  # still obtainable
        limiter.release()


class TestFairShareWidth:
    def test_one_query_gets_the_whole_machine(self) -> None:
        # Unchanged behavior for the single-query case, which is most deployments.
        assert ConcurrencyLimiter(4, cores=16).width_for(1) == 0  # 0 == unbounded

    def test_cores_are_divided_among_running_queries(self) -> None:
        limiter = ConcurrencyLimiter(16, cores=16)
        assert limiter.width_for(4) == 4
        assert limiter.width_for(16) == 1

    def test_width_never_reaches_zero(self) -> None:
        """More queries than cores must still request one worker each, not zero.

        A pool width of 0 means *unbounded* elsewhere in this module, so an integer
        division that reached 0 would silently mean the opposite of what it computed —
        every query asking for every core at maximum contention.
        """
        limiter = ConcurrencyLimiter(64, cores=4)
        assert limiter.width_for(64) == 1

    def test_an_unbounded_limiter_never_narrows(self) -> None:
        assert ConcurrencyLimiter(0, cores=16).width_for(8) == 0


class TestQueueing:
    """Every contending acquire here happens on a *separate thread*, and that is required.

    An `acquire()` from the thread that already holds a slot takes the re-entrancy path
    and returns immediately — correctly, since that is the nested-`collect()` case. An
    earlier draft of these tests contended from the main thread and so tested nothing;
    they failed with "DID NOT RAISE", which is the re-entrancy feature working.
    """

    @staticmethod
    def _acquire_on_another_thread(limiter: ConcurrencyLimiter, timeout: float):
        """Run `acquire` on a fresh thread; return (grant_or_None, exception_or_None)."""
        result: list = [None, None]

        def attempt() -> None:
            try:
                result[0] = limiter.acquire(timeout=timeout)
                limiter.release()
            except BaseException as exc:
                result[1] = exc

        thread = threading.Thread(target=attempt)
        thread.start()
        thread.join(timeout=10)
        return result[0], result[1]

    def test_a_full_queue_raises_instead_of_growing(self) -> None:
        """An unbounded queue is an outage that presents as slowness."""
        limiter = ConcurrencyLimiter(1, queue_depth=1, cores=4)
        limiter.acquire()  # main thread occupies the only slot
        waiting = threading.Event()

        def waiter() -> None:
            waiting.set()
            try:
                limiter.acquire(timeout=5.0)
                limiter.release()
            except BaseException:
                pass

        queued = threading.Thread(target=waiter)
        queued.start()
        waiting.wait(timeout=2)
        time.sleep(0.1)  # let it reach the condition wait, filling the queue

        # The queue holds its one permitted waiter; a further arrival must be refused.
        _grant, error = self._acquire_on_another_thread(limiter, timeout=0.5)
        assert isinstance(error, AdmissionTimeout), f"expected refusal, got {error!r}"
        assert "already queued" in str(error)

        limiter.release()
        queued.join(timeout=5)

    def test_the_wait_deadline_is_honoured(self) -> None:
        limiter = ConcurrencyLimiter(1, cores=4)
        limiter.acquire()
        try:
            started = time.monotonic()
            _grant, error = self._acquire_on_another_thread(limiter, timeout=0.2)
            assert isinstance(error, AdmissionTimeout), f"expected a timeout, got {error!r}"
            assert "waited" in str(error)
            assert time.monotonic() - started >= 0.2
        finally:
            limiter.release()

    def test_the_error_names_the_knob(self) -> None:
        # An admission failure with no remedy in it just moves the confusion.
        limiter = ConcurrencyLimiter(1, cores=4)
        limiter.acquire()
        try:
            _grant, error = self._acquire_on_another_thread(limiter, timeout=0.05)
            assert isinstance(error, AdmissionTimeout)
            assert "admission_timeout_s" in error.hint
        finally:
            limiter.release()

    def test_a_waiter_is_admitted_once_the_slot_frees(self) -> None:
        """The queue must drain, not just refuse — otherwise it is a broken gate."""
        limiter = ConcurrencyLimiter(1, cores=4)
        limiter.acquire()
        admitted = threading.Event()

        def waiter() -> None:
            limiter.acquire(timeout=5.0)
            admitted.set()
            limiter.release()

        thread = threading.Thread(target=waiter)
        thread.start()
        time.sleep(0.05)
        assert not admitted.is_set(), "a waiter was admitted while the slot was held"
        limiter.release()
        assert admitted.wait(timeout=5), "the waiter was never admitted after the release"
        thread.join(timeout=5)


class TestReentrancy:
    def test_a_nested_acquire_does_not_deadlock(self) -> None:
        """The trap: a `collect()` inside a `map_batches` UDF.

        The inner query would ask for a slot while the outer one holds it. With one slot
        configured, the process would wait on itself forever — a hang, in the middle of a
        pipeline, that looks like a slow UDF. The outer query already paid for the
        machine; the inner one is part of its work, not a competitor for it.
        """
        limiter = ConcurrencyLimiter(1, cores=8)
        outer = limiter.acquire()
        try:
            inner = limiter.acquire(timeout=1.0)  # would hang without re-entrancy
            limiter.release()
            assert inner.concurrent == outer.concurrent
        finally:
            limiter.release()
        assert limiter.active == 0

    def test_nesting_releases_in_the_right_order(self) -> None:
        """Only the outermost release frees the slot; the inner ones must not."""
        limiter = ConcurrencyLimiter(1, cores=8)
        limiter.acquire()
        limiter.acquire()
        limiter.acquire()
        limiter.release()
        assert limiter.active == 1, "an inner release freed the outer query's slot"
        limiter.release()
        assert limiter.active == 1
        limiter.release()
        assert limiter.active == 0

    def test_re_entrancy_is_per_thread(self) -> None:
        """One thread's nesting depth must not let another thread skip the queue."""
        limiter = ConcurrencyLimiter(1, cores=8)
        limiter.acquire()  # this thread is at depth 1
        refused: list[bool] = []

        def other() -> None:
            try:
                limiter.acquire(timeout=0.2)
                limiter.release()
                refused.append(False)
            except AdmissionTimeout:
                refused.append(True)

        thread = threading.Thread(target=other)
        thread.start()
        thread.join(timeout=5)
        limiter.release()
        assert refused == [True], "another thread bypassed a full limiter"


class TestProcessLimiter:
    def test_it_is_shared_within_a_configuration(self) -> None:
        from batcher.config import active_config

        config = active_config()
        assert process_limiter(config) is process_limiter(config)

    def test_it_rebuilds_when_the_slot_count_changes(self) -> None:
        import dataclasses

        from batcher.config import active_config

        base = active_config()
        two = base.replace(execution=dataclasses.replace(base.execution, max_concurrent_queries=2))
        four = base.replace(execution=dataclasses.replace(base.execution, max_concurrent_queries=4))
        assert process_limiter(two) is not process_limiter(four)
