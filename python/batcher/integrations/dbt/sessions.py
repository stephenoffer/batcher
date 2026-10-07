"""The one `Session` per dbt target that every dbt thread's connection shares.

dbt opens a connection per worker thread. Batcher's state lives in a `Session`, not behind
a server, so a model one thread builds is visible to another only when both connections
wrap the same session. This module keeps one per ``(path, database)`` for the life of the
dbt process, with the target's catalog attached under the dbt ``database`` name, so a
relation rendered ``database.schema.identifier`` resolves to a catalog table.

A ``path`` attaches `Catalog.from_directory`, which persists tables on disk across dbt
invocations. Without one, the catalog is in memory and lives only as long as the process.
Imports no dbt, so it is testable without the ``dbt`` extra.

This is the `integrations` layer.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from batcher.api.sql_session import Session

__all__ = ["session_for"]

_SESSIONS: dict[tuple[str | None, str], Session] = {}
_LOCK = threading.Lock()


def session_for(path: str | None, database: str) -> Session:
    """The shared session for one dbt target, created on first use.

    Args:
        path: A directory to keep the catalog's tables in, or None for an in-memory one.
        database: The dbt ``database``; the catalog is attached under this name.

    Returns:
        The session.
    """
    from batcher.api.catalog import Catalog
    from batcher.api.sql_session import Session

    key = (path, database)
    with _LOCK:
        session = _SESSIONS.get(key)
        if session is None:
            session = Session()
            catalog = (
                Catalog.from_directory(path, name=database)
                if path
                else Catalog.from_pydict({}, name=database)
            )
            session.catalog.attach(catalog)
            _SESSIONS[key] = session
        return session
