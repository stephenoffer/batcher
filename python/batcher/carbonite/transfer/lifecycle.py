"""Process-level shuffle lifecycle — the shared consumer, and the exit-time drain.

Two things a shuffle needs exactly one of per process, both of which outlive any single
`ShuffleSession` and neither of which belongs to one:

- **The pooled consumer.** `ShuffleClient` holds a tokio runtime and a gRPC channel pool
  keyed by peer address, so sharing it is not merely an optimization: a per-session client
  would accumulate a runtime and a set of background threads per session, which on a
  many-actor worker is what destabilizes the process.
- **The exit-time drain.** A published partition is a zero-copy view of a pyarrow array
  whose release callback needs the GIL. If a tokio serve thread drops one *after* Python
  has begun finalizing, the GIL acquire turns into a thread-exit that unwinds through Rust
  and aborts the process (`std::terminate`). Clearing on the main thread, with the GIL
  held, before finalization, means no background thread touches the GIL at shutdown.

Split out of `session.py` because neither is per-session state, and keeping them there
made that module read as though a session owned the process.
"""

from __future__ import annotations

import atexit
import contextlib
import threading
import weakref
from typing import TYPE_CHECKING

from batcher.carbonite.transfer.server import ShuffleClient

if TYPE_CHECKING:
    from batcher.carbonite.transfer.session import ShuffleSession

__all__ = ["host_of", "local_session", "process_client", "register_session"]

_shared_client: ShuffleClient | None = None
# Guards the lazy build. A join reducer gathers its two sides on two threads, and both
# reach this on the first fetch of a fresh worker: without the lock they each construct a
# `ShuffleClient`, which means two tokio runtimes and two channel pools, one of which is
# then dropped on the floor with its background threads still running — the exact
# accumulation the shared client exists to prevent.
_client_lock = threading.Lock()

# Every live session, tracked weakly so an already-collected one drops out on its own.
_live_sessions: weakref.WeakSet = weakref.WeakSet()
_atexit_registered = False


def process_client() -> ShuffleClient:
    """The one pooled shuffle consumer for this process, built on first use.

    Returns:
        The shared `ShuffleClient`.
    """
    global _shared_client
    client = _shared_client
    if client is None:
        with _client_lock:
            if _shared_client is None:
                _shared_client = ShuffleClient()
            client = _shared_client
    return client


def _drain_live_sessions() -> None:
    """Evict every live session's published partitions at interpreter exit."""
    for session in list(_live_sessions):
        with contextlib.suppress(Exception):  # best-effort teardown must never raise
            session.clear()


def register_session(session: ShuffleSession) -> None:
    """Track `session` for exit-time draining, registering the hook once.

    Args:
        session: The session to drain at interpreter exit.
    """
    global _atexit_registered
    _live_sessions.add(session)
    if not _atexit_registered:
        atexit.register(_drain_live_sessions)
        _atexit_registered = True


def local_session(prefer_addr: str | None = None) -> ShuffleSession | None:
    """A live `ShuffleSession` in this process, preferring the one serving `prefer_addr`.

    A worker process holds exactly one session, but a *reader* of an already-published
    partition (a next-stage scan of an intermediate left on the fleet) reaches the
    transport through a `Split`, not through the session it happens to share a process
    with — so without this it dials its own Flight server over loopback gRPC for bytes
    already sitting in its heap. Handing it the session restores the mode selection:
    `DIRECT_MEMORY` when this process published the bucket, `SHARED_MEMORY` when a peer
    on this node did, `NETWORK` otherwise.

    Any live session fetches *correctly* — `select_mode` compares the source address to
    the session's own, so a non-matching one simply resolves to the remote path — but the
    match is what makes the local fast path reachable and attributes the locality counters
    to the session that actually served the bytes.

    Args:
        prefer_addr: The advertised address the caller is fetching from; a session serving
            it is returned ahead of any other.

    Returns:
        A live session, or `None` in a process that has none (the driver).
    """
    fallback: ShuffleSession | None = None
    for session in _live_sessions:
        if prefer_addr and session.addr == prefer_addr:
            return session
        if fallback is None:
            fallback = session
    return fallback


def host_of(addr: str) -> str:
    """The node identity of a shuffle address.

    Args:
        addr: An advertised shuffle address, which is always ``{node_ip}:{port}``.

    Returns:
        The host part, so equal hosts mean the same node.
    """
    return addr.rsplit(":", 1)[0]
