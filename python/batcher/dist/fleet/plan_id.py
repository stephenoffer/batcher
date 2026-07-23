"""The per-query shuffle plan id — the fence that keeps concurrent pipelines apart.

A shuffle ticket is `(plan, stage, src, dst, epoch)`, and `plan` is what makes one query's
published partitions unreadable by another. It cannot be the fleet's spawn-time id: several
pipelines deliberately share one warm fleet rather than each reserving the cluster's whole
CPU capacity, so two concurrent queries would inherit the *same* id and their tickets at the
same coordinate would be byte-identical — one silently reading the other's bucket.

So the id is minted per **query** and travels as an explicit argument on every worker
call, since an actor shared by both queries cannot hold either one's id in its own state.

`query_shuffle_scope` is where it is minted, and it is **reentrant** so the two entry
points compose: the adaptive loop opens it once per query via `session_fleet_lease` (all
its stages then share one fence, keeping a stage's intermediate readable by the next),
while `execute_distributed` opens it for a one-shot query that never took that lease.
Fencing only the adaptive path is not enough — `adaptive="auto"` resolves to False for the
commonest distributed shapes, so those queries are the ones most likely to collide.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import threading

__all__ = [
    "adopt_plan_id",
    "mint_query_plan_id",
    "query_plan_id",
    "query_shuffle_scope",
    "reset_query_plan_id",
    "with_query_shuffle_scope",
]

_QUERY_PLAN_ID: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "batcher_query_plan_id", default=None
)

# How many queries hold a shuffle scope in this process right now. The `ContextVar` above
# answers "which query am I?"; this answers "is anyone else running?", which is what the
# fleet needs before it retunes shared workers or reserves a second cluster-wide gang.
# Deliberately ONE counter for both entry points: an earlier version counted only the
# adaptive path's leases, so the fleet happily re-granted out from under a concurrent
# *non-adaptive* query — the commonest kind. A lock, not a bare `+=`, because concurrent
# pipelines open their scopes from different driver threads.
_SCOPE_LOCK = threading.Lock()
_ACTIVE_SCOPES = 0


def query_plan_id() -> int | None:
    """The plan id fencing this query's shuffle, or None outside a query scope."""
    return _QUERY_PLAN_ID.get()


def active_query_scopes() -> int:
    """How many queries currently hold a shuffle scope in this process."""
    return _ACTIVE_SCOPES


def mint_query_plan_id() -> tuple[int, contextvars.Token]:
    """Start a query's shuffle scope: a fresh id, installed here and on the driver.

    Returns `(plan_id, token)`; the caller MUST pass the token to `reset_query_plan_id`.
    """
    import logging

    from batcher._internal.logging import get_logger, log_kv
    from batcher.dist.flight_worker import new_plan_id, set_current_plan_id

    plan_id = new_plan_id()
    token = _QUERY_PLAN_ID.set(plan_id)
    set_current_plan_id(plan_id)
    # Emit the fence so a shuffle can be attributed to a pipeline. Every shuffle log,
    # ticket, and spilled bucket is keyed by `plan_id`, but nothing else prints it — so on
    # a cluster running several pipelines at once there was no way to tell whose shuffle
    # failed. Pair it with the thread so concurrent drivers in one process are separable.
    log_kv(
        get_logger("dist"),
        logging.DEBUG,
        "shuffle_scope_opened",
        plan_id=plan_id,
        thread=threading.current_thread().name,
    )
    return plan_id, token


def reset_query_plan_id(token: contextvars.Token) -> None:
    """End the query scope opened by `mint_query_plan_id`."""
    _QUERY_PLAN_ID.reset(token)


@contextlib.contextmanager
def query_shuffle_scope():
    """Fence this query's shuffle, unless an enclosing scope already did.

    **Reentrant, and that is the point.** The adaptive loop opens one scope per query
    (`session_fleet_lease`) and each stage runs inside it, so every stage shares one id and
    a stage's published intermediate stays readable by the next. A one-shot query has no
    such enclosing scope and opens its own here.

    Without this, only the *adaptive* path was fenced. `adaptive="auto"` resolves to False
    for the commonest shapes (a plain distributed `group_by`), so those queries fell back
    to the **fleet's** spawn-time id — which the warm session fleet, on by default, shares
    across every concurrent pipeline. Two of them then published identical tickets.
    """
    global _ACTIVE_SCOPES

    if _QUERY_PLAN_ID.get() is not None:
        yield  # an enclosing query scope owns the fence; nesting must not re-mint or recount
        return
    _plan_id, token = mint_query_plan_id()
    with _SCOPE_LOCK:
        _ACTIVE_SCOPES += 1
    try:
        yield
    finally:
        with _SCOPE_LOCK:
            _ACTIVE_SCOPES = max(0, _ACTIVE_SCOPES - 1)
        reset_query_plan_id(token)


def with_query_shuffle_scope(fn):
    """Run `fn` inside `query_shuffle_scope` — for the distributed entry point itself.

    Applied at the definition rather than the call sites so a new caller cannot forget it
    and silently reintroduce the shared-fence bug.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with query_shuffle_scope():
            return fn(*args, **kwargs)

    return wrapper


def adopt_plan_id(fleet_plan_id: int) -> None:
    """Install this operator's ticket fence: the query's own id, else the fleet's own."""
    from batcher.dist.flight_worker import set_current_plan_id

    query_id = _QUERY_PLAN_ID.get()
    set_current_plan_id(fleet_plan_id if query_id is None else query_id)
