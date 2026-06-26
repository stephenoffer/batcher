"""Spot-preemption detection so the engine drains proactively, not reactively.

A spot/preemptible node is given a short termination notice before it is reclaimed
(AWS ~2 min via the metadata ``instance-action`` endpoint, GCP a ``preempted`` flag,
or a ``SIGTERM`` from the scheduler / Kubernetes). Without watching for it, the
engine learns of the loss only *after* in-flight work is gone — a failed fetch or a
dead actor — and pays a full recompute. This monitor turns the notice into an early
``is_draining()`` signal plus a one-shot drain hook, so the orchestrator can stop
scheduling new work onto the node and flush in-flight intermediates to durable
storage before it dies.

Carbonite owns it (a resource "protect" concern); Core and the distributed workers
consult it. It is a process-wide singleton — one background poller per worker
process — and is started only under the ``spot`` resilience profile, so a stable
on-demand cluster pays nothing.
"""

from __future__ import annotations

import contextlib
import signal
import threading
from collections.abc import Callable

__all__ = ["PreemptionMonitor", "cloud_preemption_probe", "preemption_monitor"]

# Link-local metadata endpoints answer in microseconds; a tight timeout keeps a
# probe from ever stalling the poll loop (and reads a partition as "not draining").
_PROBE_TIMEOUT_S = 0.3


def cloud_preemption_probe() -> bool:
    """Return True when the cloud metadata endpoint reports imminent reclamation.

    Checks the AWS spot ``instance-action`` endpoint (200 only when an action is
    scheduled) and the GCP ``preempted`` flag. Any error or non-preempt response
    reads as "not draining", so a transient probe failure never false-positives a
    drain. Cheap link-local HTTP with a tight timeout, called from the poll thread.
    """
    import urllib.request

    probes: tuple[tuple[str, dict[str, str], Callable[[str], bool]], ...] = (
        (
            "http://169.254.169.254/latest/meta-data/spot/instance-action",
            {},
            bool,
        ),
        (
            "http://metadata.google.internal/computeMetadata/v1/instance/preempted",
            {"Metadata-Flavor": "Google"},
            lambda body: body.strip().upper() == "TRUE",
        ),
    )
    for url, headers, is_drain in probes:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=_PROBE_TIMEOUT_S) as resp:
                if resp.status == 200 and is_drain(resp.read().decode("utf-8", "replace")):
                    return True
        except Exception:
            continue
    return False


class PreemptionMonitor:
    """Process-wide watcher that flips ``is_draining()`` on a termination notice.

    Polls a `probe` (default: cloud metadata) on a daemon thread and also traps
    ``SIGTERM`` (what a scheduler / Kubernetes sends on eviction). On the first signal
    it sets a sticky draining flag and runs each registered drain callback once — the
    hook the orchestrator uses to stop scheduling onto this node and flush in-flight
    intermediates. Sticky by design: a drain is never un-seen. Idempotent — starting
    or triggering twice is a no-op.

    Examples:
        .. doctest::

            >>> from batcher.carbonite.resilience.preemption import PreemptionMonitor
            >>> seen = []
            >>> mon = PreemptionMonitor(probe=lambda: False)
            >>> mon.on_drain(lambda: seen.append("flushed"))
            >>> mon.is_draining()
            False
            >>> mon.trigger()  # what the SIGTERM handler / poll loop calls
            >>> mon.is_draining(), seen
            (True, ['flushed'])
    """

    def __init__(
        self, probe: Callable[[], bool] | None = None, poll_interval_s: float = 5.0
    ) -> None:
        self._probe = probe or cloud_preemption_probe
        self._poll_interval_s = poll_interval_s
        self._draining = threading.Event()
        self._callbacks: list[Callable[[], None]] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._prev_sigterm: object = None

    def is_draining(self) -> bool:
        """Whether a termination notice has been observed for this node."""
        return self._draining.is_set()

    def on_drain(self, callback: Callable[[], None]) -> None:
        """Register a callback run once when draining begins (or now, if already)."""
        with self._lock:
            self._callbacks.append(callback)
            already = self._draining.is_set()
        if already:
            self._safe_call(callback)

    def start(self) -> None:
        """Begin polling and trap SIGTERM. Idempotent."""
        with self._lock:
            if self._thread is not None:
                return
            self._stop.clear()
            thread = threading.Thread(
                target=self._poll_loop, name="batcher-preemption", daemon=True
            )
            self._thread = thread
        self._install_sigterm()
        thread.start()

    def stop(self) -> None:
        """Stop polling and restore the prior SIGTERM handler. Idempotent."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        with self._lock:
            self._thread = None
        self._restore_sigterm()

    def trigger(self) -> None:
        """Mark draining now and fire each callback once (SIGTERM / test entry point)."""
        if self._draining.is_set():
            return
        self._draining.set()
        with self._lock:
            callbacks = list(self._callbacks)
        for callback in callbacks:
            self._safe_call(callback)

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            draining = False
            with contextlib.suppress(Exception):
                draining = self._probe()
            if draining:
                self.trigger()
                return  # sticky — nothing more to watch
            self._stop.wait(self._poll_interval_s)

    def _install_sigterm(self) -> None:
        try:
            self._prev_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, self._on_sigterm)
        except (ValueError, OSError):
            # Not the main thread (e.g. a Ray worker) — rely on the metadata poll.
            self._prev_sigterm = None

    def _on_sigterm(self, signum: int, frame: object) -> None:
        self.trigger()
        prev = self._prev_sigterm
        if callable(prev):
            prev(signum, frame)

    def _restore_sigterm(self) -> None:
        if self._prev_sigterm is not None:
            with contextlib.suppress(ValueError, OSError):
                signal.signal(signal.SIGTERM, self._prev_sigterm)  # type: ignore[arg-type]
            self._prev_sigterm = None

    @staticmethod
    def _safe_call(callback: Callable[[], None]) -> None:
        # A drain hook must never raise into the poll thread or the signal handler.
        with contextlib.suppress(Exception):
            callback()


_MONITOR: PreemptionMonitor | None = None
_MONITOR_LOCK = threading.Lock()


def preemption_monitor() -> PreemptionMonitor:
    """The process-wide `PreemptionMonitor` (created on first use, not yet started)."""
    global _MONITOR
    with _MONITOR_LOCK:
        if _MONITOR is None:
            _MONITOR = PreemptionMonitor()
        return _MONITOR
