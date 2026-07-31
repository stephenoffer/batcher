"""Preemption detection so the engine drains proactively, not reactively.

A node under a scheduler is taken away in one of three ways, and the engine has to
watch for all of them because a given cluster only ever offers one:

* A cloud spot instance announces its own reclamation (AWS ~2 min via the metadata
  ``instance-action`` endpoint, GCP a ``preempted`` flag, Azure Scheduled Events).
* An orchestrator sends a signal — ``SIGTERM`` from Kubernetes on eviction, and
  ``SIGUSR1`` from Slurm when the job was submitted with ``--signal=B:USR1@<lead>``.
* A batch scheduler gives no notice at all and simply kills the allocation when its
  wall clock runs out. That case is read from the deadline itself
  (`batcher.config.deadline`, which sits in layer 0 so the profile resolution below
  Carbonite and this monitor inside it can share one answer).

Without watching, the engine learns of the loss only *after* in-flight work is gone —
a failed fetch or a dead actor — and pays a full recompute. This monitor turns any of
the three into an early ``is_draining()`` signal plus a one-shot drain hook, so the
orchestrator can stop scheduling new work onto the node and flush in-flight
intermediates to durable storage before it dies.

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

from batcher._internal.logging import note_suppressed

__all__ = [
    "PreemptionMonitor",
    "cloud_preemption_probe",
    "preemption_monitor",
    "termination_probe",
]

# Signals that mean "this process is going away shortly". `SIGTERM` is what Kubernetes
# sends on eviction and what Slurm sends at the time limit (with `KillWait` seconds before
# the `SIGKILL`). `SIGUSR1` is Slurm's *early warning*: a job submitted with
# `--signal=B:USR1@120` gets it two minutes before the limit, which is the whole point —
# it is the only advance notice an HPC allocation ever gets, and by default its disposition
# is to terminate the process outright, so trapping it is strictly safer than ignoring it.
# Absent on a platform that lacks one (Windows has no `SIGUSR1`), hence the `getattr`.
_DRAIN_SIGNAL_NAMES = ("SIGTERM", "SIGUSR1")

# Link-local metadata endpoints answer in microseconds; a tight timeout keeps a
# probe from ever stalling the poll loop (and reads a partition as "not draining").
_PROBE_TIMEOUT_S = 0.3

# EC2 IMDSv2: a session token is minted by PUT and then presented on the metadata GET.
#
# **Without it the AWS spot probe cannot fire at all on a modern instance.** IMDSv2 is
# enforced whenever the instance is launched with `HttpTokens=required`, which is the
# default for recent launch templates and is commonly mandated org-wide by policy. An
# unauthenticated GET there returns 401, the probe treats any error as "not draining"
# (correctly — being off EC2 must not false-positive), and the two cases are
# indistinguishable from inside. The result is a spot fleet that never sees a termination
# notice and never proactively drains, failing exactly as an on-prem cluster would, with
# nothing to say the feature is off.
#
# The TTL is short because the token is used once, immediately.
_IMDS_TOKEN_URL = "http://169.254.169.254/latest/api/token"
_IMDS_TOKEN_TTL_S = 60


def _imds_v2_headers() -> dict[str, str]:
    """An IMDSv2 session-token header, or `{}` when one cannot be minted.

    Empty is the right fallback rather than an error: it is what an IMDSv1-only instance
    and a non-EC2 host both produce, and the GET that follows still works on IMDSv1. So
    this only ever adds reach, and costs one link-local round trip on a spot worker's poll.

    Returns:
        `{"X-aws-ec2-metadata-token": ...}`, or `{}`.
    """
    import urllib.request

    try:
        req = urllib.request.Request(
            _IMDS_TOKEN_URL,
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": str(_IMDS_TOKEN_TTL_S)},
        )
        with urllib.request.urlopen(req, timeout=_PROBE_TIMEOUT_S) as resp:
            if resp.status == 200:
                token = resp.read().decode("utf-8", "replace").strip()
                if token:
                    return {"X-aws-ec2-metadata-token": token}
    except Exception as exc:
        # Off EC2, or IMDSv1-only, or IMDS hop-limited. All normal, none fatal.
        note_suppressed("carbonite", "mint an IMDSv2 session token", exc)
    return {}


def _azure_is_draining(body: str) -> bool:
    """Whether Azure's Scheduled Events payload announces reclamation of *this* node.

    The endpoint always answers 200 with a (usually empty) ``Events`` list, so unlike the
    AWS and GCP probes the status code says nothing — the payload has to be read. Only
    ``Preempt`` (spot eviction) and ``Terminate`` (announced shutdown) mean the node is
    going away; ``Reboot``/``Freeze``/``Redeploy`` are disruptions the job rides out, and
    treating them as a drain would migrate shuffle output on every host maintenance blip.
    """
    import json

    try:
        events = json.loads(body).get("Events") or []
    except (ValueError, AttributeError):
        return False
    return any(e.get("EventType") in ("Preempt", "Terminate") for e in events)


def cloud_preemption_probe() -> bool:
    """Return True when the cloud metadata endpoint reports imminent reclamation.

    Checks the AWS spot ``instance-action`` endpoint (200 only when an action is
    scheduled), the GCP ``preempted`` flag, and Azure Scheduled Events. Any error or
    non-preempt response reads as "not draining", so a transient probe failure never
    false-positives a drain. Cheap link-local HTTP with a tight timeout, called from the
    poll thread — and only ever from a spot-profile worker, so a fixed on-prem cluster
    never pays for probes its infrastructure would not answer.

    The AWS probe presents an IMDSv2 session token when one can be minted. Without it the
    probe is silently dead on any instance launched with `HttpTokens=required`, which is
    both the modern default and a common org-wide policy — the GET returns 401, that reads
    as "not draining" like every other error, and the fleet simply never drains.
    """
    import urllib.request

    probes: tuple[tuple[str, dict[str, str], Callable[[str], bool]], ...] = (
        (
            "http://169.254.169.254/latest/meta-data/spot/instance-action",
            _imds_v2_headers(),
            bool,
        ),
        (
            "http://metadata.google.internal/computeMetadata/v1/instance/preempted",
            {"Metadata-Flavor": "Google"},
            lambda body: body.strip().upper() == "TRUE",
        ),
        (
            "http://169.254.169.254/metadata/scheduledevents?api-version=2020-07-01",
            {"Metadata": "true"},
            _azure_is_draining,
        ),
    )
    for url, headers, is_drain in probes:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=_PROBE_TIMEOUT_S) as resp:
                if resp.status == 200 and is_drain(resp.read().decode("utf-8", "replace")):
                    return True
        except Exception as exc:
            # A probe that cannot be reached is the normal case off the matching cloud, so
            # it must not raise. Recording it is what separates "not on EC2" from "the
            # metadata endpoint has been unreachable since the VPC change", which otherwise
            # looks identical: a fleet that silently never sees a preemption notice.
            note_suppressed("carbonite", f"probe preemption endpoint {url}", exc)
            continue
    return False


def termination_probe() -> bool:
    """Return True when this node is going away, from any signal source available.

    The wall-clock deadline is checked *first* and deliberately so. It is a local clock
    comparison, so it costs nothing and cannot fail; the cloud probes are three link-local
    HTTP round trips that, on the clusters where a deadline exists (Slurm, a leased VM, a
    CI runner), all time out and return False every poll. Checking the free and decisive
    signal before the expensive and usually-absent one keeps the poll loop cheap where it
    matters most.

    The lead time comes from `DistributedConfig.drain_lead_s`. Reading it per poll rather
    than capturing it at construction is what lets a worker honor a lead the driver set,
    since a worker process resolves its own config from the environment.

    Returns:
        Whether a termination notice or an imminent deadline has been observed.
    """
    from batcher.config import active_config
    from batcher.config.deadline import deadline_probe

    lead = 0.0
    try:
        lead = float(active_config().distributed.drain_lead_s)
    except Exception as exc:  # pragma: no cover - config is resolvable in practice
        # A worker whose config cannot be resolved must still drain on the cloud/signal
        # path rather than raise out of the poll thread and stop watching entirely.
        note_suppressed("carbonite", "read the drain lead time", exc)
    if deadline_probe(lead)():
        return True
    return cloud_preemption_probe()


class PreemptionMonitor:
    """Process-wide watcher that flips ``is_draining()`` on a termination notice.

    Polls a `probe` (default: the wall-clock deadline, then cloud metadata) on a daemon
    thread and also traps ``SIGTERM`` (Kubernetes eviction, Slurm's time limit) and
    ``SIGUSR1`` (Slurm's ``--signal=B:USR1@<lead>`` early warning). On the first of any
    of them it sets a sticky draining flag and runs each registered drain callback once — the
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
        self._probe = probe or termination_probe
        self._poll_interval_s = poll_interval_s
        self._draining = threading.Event()
        self._callbacks: list[Callable[[], None]] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._prev_handlers: dict[int, object] = {}

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
        """Begin polling and trap the drain signals. Idempotent."""
        with self._lock:
            if self._thread is not None:
                return
            self._stop.clear()
            thread = threading.Thread(
                target=self._poll_loop, name="batcher-preemption", daemon=True
            )
            self._thread = thread
        self._install_signals()
        thread.start()

    def stop(self) -> None:
        """Stop polling and restore the prior signal handlers. Idempotent."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            # Never join from inside the poll thread itself (a drain callback that calls
            # `stop()` would otherwise deadlock waiting for its own thread to finish).
            thread.join(timeout=1.0)
        with self._lock:
            self._thread = None
        self._restore_signals()

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
        try:
            while not self._stop.is_set():
                draining = False
                with contextlib.suppress(Exception):
                    draining = self._probe()
                if draining:
                    self.trigger()
                    return  # sticky — nothing more to watch
                self._stop.wait(self._poll_interval_s)
        finally:
            # Release the thread slot on the way out, whichever way we left. `start()` is
            # a no-op while `_thread` is set, so a loop that exited on its own (a drain
            # was observed, or the probe raised out) left the monitor permanently
            # un-startable — with a dead thread standing in for a live one.
            with self._lock:
                if self._thread is threading.current_thread():
                    self._thread = None

    def _install_signals(self) -> None:
        """Trap each drain signal this platform has, chaining to the prior handler.

        Installed per signal rather than all-or-nothing: `SIGTERM` must still be trapped on
        a platform with no `SIGUSR1`, and off the main thread (a Ray worker) none of them
        can be installed at all — there the metadata/deadline poll is the whole mechanism.
        """
        for name in _DRAIN_SIGNAL_NAMES:
            signum = getattr(signal, name, None)
            if signum is None:  # not on this platform (e.g. SIGUSR1 on Windows)
                continue
            try:
                previous = signal.getsignal(signum)
                signal.signal(signum, self._on_signal)
            except (ValueError, OSError):
                # Not the main thread (e.g. a Ray worker) — rely on the poll instead.
                continue
            self._prev_handlers[int(signum)] = previous

    def _on_signal(self, signum: int, frame: object) -> None:
        self.trigger()
        prev = self._prev_handlers.get(int(signum))
        if callable(prev):
            prev(signum, frame)

    def _restore_signals(self) -> None:
        for signum, previous in list(self._prev_handlers.items()):
            if previous is None:
                # `getsignal` reports None for a handler installed from C, which
                # `signal.signal` cannot take back — leave that one trapped rather than raise.
                continue
            with contextlib.suppress(ValueError, OSError, TypeError):
                signal.signal(signum, previous)  # type: ignore[arg-type]
        self._prev_handlers.clear()

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
