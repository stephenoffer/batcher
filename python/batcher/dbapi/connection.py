"""The PEP 249 connection: a handle on one Batcher `Session`, and the module's globals.

A connection does not own a database. It wraps a `Session`, which is where tables, views,
functions and attached catalogs live, and every cursor it creates runs against that session.
`connect()` with no argument uses `bt.current_session()`, the one `bt.sql` uses, so tables
registered through the Python API are visible to a DB-API client in the same process.

**There are no transactions.** Each statement takes effect when it runs, the way an
autocommit connection behaves elsewhere. `commit` therefore has nothing to do and returns.
`rollback` cannot undo anything: it raises `NotSupportedError` when a statement that writes
has run since the last `commit`, and returns quietly when none has, since then there is
nothing to undo. That second case is what lets a client that always rolls back a read-only
connection on release, as SQLAlchemy's pool does, work unchanged, while a client that writes
and then expects a rollback to discard it is told plainly that it did not.

This is the `dbapi` layer.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from batcher.dbapi.cursor import Cursor
from batcher.dbapi.errors import InterfaceError, NotSupportedError

if TYPE_CHECKING:
    from batcher.api.sql_session import Session

__all__ = ["Connection", "apilevel", "connect", "paramstyle", "threadsafety"]

#: The PEP 249 version this module implements.
apilevel = "2.0"
#: Threads may share the module but not a connection: a `Session` is not locked.
threadsafety = 1
#: ``?`` placeholders bound from a sequence. A mapping binds ``$name`` placeholders as well.
paramstyle = "qmark"


class Connection:
    """A PEP 249 connection over a Batcher `Session`, created by `connect`.

    Examples:
        .. doctest::

            >>> from batcher import dbapi
            >>> conn = dbapi.connect()
            >>> conn.cursor().execute("SELECT 41 + 1 AS answer").fetchone()
            (42,)
            >>> conn.close()
    """

    def __init__(self, session: Session) -> None:
        """Wrap `session`; use `connect` instead.

        Args:
            session: The session every cursor of this connection runs against.
        """
        self._session = session
        self._closed = False
        self._cursors: list[Cursor] = []
        self._writes = 0
        self._lock = threading.Lock()

    @property
    def session(self) -> Session:
        """The `Session` this connection runs statements against.

        Not part of PEP 249. It is how code holding a connection reaches the rest of the
        Python API, such as registering a table for the next query.

        Returns:
            The session.

        Raises:
            InterfaceError: The connection is closed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher import dbapi
                >>> s = bt.Session()
                >>> dbapi.connect(s).session is s
                True
        """
        self._check_open()
        return self._session

    @property
    def closed(self) -> bool:
        """Whether `close` has run.

        Returns:
            True once closed.

        Examples:
            .. doctest::

                >>> from batcher import dbapi
                >>> dbapi.connect().closed
                False
        """
        return self._closed

    def cursor(self) -> Cursor:
        """A new cursor on this connection.

        Returns:
            The cursor.

        Raises:
            InterfaceError: The connection is closed.

        Examples:
            .. doctest::

                >>> from batcher import dbapi
                >>> type(dbapi.connect().cursor()).__name__
                'Cursor'
        """
        self._check_open()
        cursor = Cursor(self)
        with self._lock:
            self._cursors.append(cursor)
        return cursor

    def commit(self) -> None:
        """Acknowledge the statements run so far; they already took effect, so nothing runs.

        Raises:
            InterfaceError: The connection is closed.

        Examples:
            .. doctest::

                >>> from batcher import dbapi
                >>> dbapi.connect().commit()
        """
        self._check_open()
        self._writes = 0

    def rollback(self) -> None:
        """Refuse to pretend a write was undone; a no-op when nothing was written.

        Batcher has no transactions, so a statement that wrote has already taken effect and
        nothing can reverse it. When one has run since the last `commit`, this raises rather
        than return as though it had been undone. The raise also acknowledges those writes,
        so the next `rollback` reports only writes made after it.

        Raises:
            NotSupportedError: A statement that writes ran since the last `commit`.
            InterfaceError: The connection is closed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher import dbapi
                >>> conn = dbapi.connect(bt.Session())
                >>> conn.rollback()  # nothing written: nothing to undo
                >>> _ = conn.cursor().execute("CREATE TABLE t AS SELECT 1 AS x")
                >>> try:
                ...     conn.rollback()
                ... except dbapi.NotSupportedError:
                ...     print("the write stays applied")
                the write stays applied
        """
        self._check_open()
        writes, self._writes = self._writes, 0
        if writes:
            raise NotSupportedError(
                f"rollback() cannot undo the {writes} statement(s) that wrote since the last "
                "commit(): Batcher has no transactions, and each statement took effect when it "
                "ran"
            )

    def close(self) -> None:
        """Close this connection and every cursor it created; the session stays as it was.

        Examples:
            .. doctest::

                >>> from batcher import dbapi
                >>> conn = dbapi.connect()
                >>> cur = conn.cursor()
                >>> conn.close()
                >>> conn.closed, cur.closed
                (True, True)
        """
        with self._lock:
            cursors, self._cursors = self._cursors, []
        for cursor in cursors:
            cursor.close()
        self._closed = True

    def __enter__(self) -> Connection:
        """Return this connection; the ``with`` block closes it on exit."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the connection."""
        self.close()

    def _check_open(self) -> None:
        if self._closed:
            raise InterfaceError("the connection is closed")

    def _note_write(self) -> None:
        self._writes += 1


def connect(session: Session | None = None) -> Connection:
    """Open a PEP 249 connection over `session`, or over `bt.current_session()`.

    Closing the connection does not clear or close the session: the session belongs to the
    caller, and other code may be using it.

    Args:
        session: The session to run statements against. None uses the session `bt.sql`
            uses here, so tables registered through the Python API are visible.

    Returns:
        The connection.

    Raises:
        InterfaceError: `session` is not a `bt.Session`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher import dbapi
            >>> s = bt.Session()
            >>> _ = s.register("orders", bt.from_pydict({"id": [1, 2], "amount": [5.0, 7.5]}))
            >>> conn = dbapi.connect(s)
            >>> cur = conn.cursor()
            >>> cur.execute("SELECT id FROM orders WHERE amount > ?", [6.0]).fetchall()
            [(2,)]
            >>> conn.close()
    """
    from batcher.api.session.sql import current_session
    from batcher.api.sql_session import Session

    if session is None:
        session = current_session()
    if not isinstance(session, Session):
        raise InterfaceError(f"connect() expects a bt.Session, got {type(session).__name__}")
    return Connection(session)
