"""The security context: which catalog and principal are in effect for this scope.

Split from `_binding` (which applies the policy) because the two answer different
questions — *what governs* versus *how a scan is governed* — and only this half needs
to reason about scoping across threads and async tasks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from batcher.governance import GovernanceEvent, Principal, SecurityCatalog

__all__ = ["SecurityContext", "current_security", "security"]

#: Receives one `GovernanceEvent` per governed table, as each table is read.
AuditSink = Callable[[GovernanceEvent], None]


@dataclass(frozen=True, slots=True)
class SecurityContext:
    """The catalog, principal, and audit sink in effect for the current scope."""

    catalog: SecurityCatalog
    principal: Principal
    audit: AuditSink | None = None


# A `ContextVar`, not a module global: it is correct under asyncio and threads, and a
# nested `security()` block restores the outer policy exactly on exit.
_CURRENT: ContextVar[SecurityContext | None] = ContextVar("batcher_security", default=None)


def current_security() -> SecurityContext | None:
    """Return the security context governing this scope, or None if ungoverned.

    Returns:
        The active `SecurityContext`, or None outside a `security()` block.
    """
    return _CURRENT.get()


@contextmanager
def security(
    catalog: SecurityCatalog, principal: Principal, *, audit: AuditSink | None = None
) -> Iterator[None]:
    """Read tables inside this block as `principal`, governed by `catalog`.

    A table read inside the block carries the block's policy for the whole life of the
    resulting `Dataset`, including terminal operations performed after the block exits.
    A table read outside any `security()` block is ungoverned.

    This is a context manager rather than a setter because governance is applied when a
    table is *read* (see `batcher.api.security._binding`). Inside a block, a read cannot
    precede the policy that governs it.

    Examples:
        .. doctest::

            >>> import os
            >>> import tempfile

            >>> import batcher as bt
            >>> path = os.path.join(tempfile.mkdtemp(), "customers.parquet")
            >>> _ = bt.from_pydict({"id": [1, 2], "email": ["a@x.com", "b@x.com"]}).write(
            ...     path, format="parquet"
            ... )
            >>> catalog = bt.SecurityCatalog().mask_column(
            ...     path, "email", lambda c: bt.mask(c, show_last=5)
            ... )
            >>> with bt.security(catalog, bt.Principal("ana", roles=["analyst"])):
            ...     ds = bt.read.parquet(path)
            >>> ds.sort("id").to_pydict()["email"]
            ['XXx.com', 'XXx.com']

            Record every decision for a compliance log:

            >>> seen = []
            >>> with bt.security(catalog, bt.Principal("ana"), audit=seen.append):
            ...     _ = bt.read.parquet(path)
            >>> seen[0].masked
            ('email',)

    Args:
        catalog: The policies to enforce.
        principal: The identity that reads inside the block run as.
        audit: Called with one `GovernanceEvent` per governed table as it is read —
            the compliance record of what was allowed, withheld, and masked. Every
            decision is also logged at INFO regardless of this sink.

    Yields:
        None. The policy is in effect for the duration of the block.
    """
    token = _CURRENT.set(SecurityContext(catalog, principal, audit))
    try:
        yield
    finally:
        _CURRENT.reset(token)
