"""The PEP 249 exception hierarchy, and the one mapping from Batcher's typed errors onto it.

PEP 249 fixes the class names and their tree: `Warning` and `Error` under `Exception`,
`InterfaceError` and `DatabaseError` under `Error`, and five more under `DatabaseError`. A
client written against any DB-API driver catches these, so every Batcher failure a cursor
reports is re-raised as one of them, with the original chained as ``__cause__`` so its
message, hint and type are still there for anyone who looks.

The mapping is by *who can fix it*, which is how PEP 249 draws its lines: a statement that
does not parse, names a missing column or breaks a session rule is the programmer's
(`ProgrammingError`); a construct Batcher does not translate is `NotSupportedError`; bad
values in the data are `DataError`; and a run that failed while executing, ran out of
memory or was cancelled is `OperationalError`.

This is the `dbapi` layer: it sits above `api` and imports only its error types.
"""

from __future__ import annotations

import builtins

from batcher._internal.errors import (
    BatcherError,
    ConfigError,
    DataQualityError,
    ExecutionError,
    FormatError,
    PlanError,
    ResourceError,
    SchemaError,
    SQLUnsupportedError,
    TransportError,
)
from batcher._internal.errors import IOError as BatcherIOError

__all__ = [
    "DataError",
    "DatabaseError",
    "Error",
    "IntegrityError",
    "InterfaceError",
    "InternalError",
    "NotSupportedError",
    "OperationalError",
    "ProgrammingError",
    "Warning",
    "translate",
]


class Warning(builtins.Warning):
    """The PEP 249 warning class; Batcher's adapter defines it and never raises it.

    Examples:
        .. doctest::

            >>> from batcher import dbapi
            >>> issubclass(dbapi.Warning, Warning)
            True
    """


class Error(Exception):
    """The base of every error a Batcher DB-API connection or cursor raises.

    Examples:
        .. doctest::

            >>> from batcher import dbapi
            >>> issubclass(dbapi.ProgrammingError, dbapi.Error)
            True
    """


class InterfaceError(Error):
    """Misuse of the adapter itself, such as a cursor used after it was closed.

    Examples:
        .. doctest::

            >>> from batcher import dbapi
            >>> cur = dbapi.connect().cursor()
            >>> cur.close()
            >>> try:
            ...     cur.execute("SELECT 1")
            ... except dbapi.InterfaceError as exc:
            ...     print(exc)
            the cursor is closed
    """


class DatabaseError(Error):
    """A failure in the engine; the parent of every other database error.

    Examples:
        .. doctest::

            >>> from batcher import dbapi
            >>> issubclass(dbapi.OperationalError, dbapi.DatabaseError)
            True
    """


class DataError(DatabaseError):
    """A problem with the data being processed, such as a failed quality check or bad file.

    Examples:
        .. doctest::

            >>> from batcher import dbapi
            >>> issubclass(dbapi.DataError, dbapi.DatabaseError)
            True
    """


class OperationalError(DatabaseError):
    """A failure while running a statement: execution, memory, transport or cancellation.

    Examples:
        .. doctest::

            >>> from batcher import dbapi
            >>> issubclass(dbapi.OperationalError, dbapi.DatabaseError)
            True
    """


class IntegrityError(DatabaseError):
    """The PEP 249 constraint-violation class; never raised, as Batcher enforces no keys.

    Examples:
        .. doctest::

            >>> from batcher import dbapi
            >>> issubclass(dbapi.IntegrityError, dbapi.DatabaseError)
            True
    """


class InternalError(DatabaseError):
    """The PEP 249 class for an internal inconsistency in the database.

    Examples:
        .. doctest::

            >>> from batcher import dbapi
            >>> issubclass(dbapi.InternalError, dbapi.DatabaseError)
            True
    """


class ProgrammingError(DatabaseError):
    """A statement Batcher refused: bad syntax, a missing table or column, a wrong binding.

    Examples:
        .. doctest::

            >>> from batcher import dbapi
            >>> cur = dbapi.connect().cursor()
            >>> try:
            ...     cur.execute("SELEC 1")
            ... except dbapi.ProgrammingError:
            ...     print("refused")
            refused
    """


class NotSupportedError(DatabaseError):
    """A construct or call Batcher does not support, such as rolling back a write.

    Examples:
        .. doctest::

            >>> from batcher import dbapi
            >>> issubclass(dbapi.NotSupportedError, dbapi.DatabaseError)
            True
    """


#: Most specific first: the first Batcher class an error is an instance of picks its type.
_MAPPING: tuple[tuple[type[BaseException], type[DatabaseError]], ...] = (
    (SQLUnsupportedError, NotSupportedError),
    (DataQualityError, DataError),
    (SchemaError, DataError),
    (FormatError, DataError),
    (PlanError, ProgrammingError),
    (ConfigError, ProgrammingError),
    (ExecutionError, OperationalError),
    (ResourceError, OperationalError),
    (TransportError, OperationalError),
    (BatcherIOError, OperationalError),
    (BatcherError, DatabaseError),
)


def translate(exc: Exception) -> Error:
    """The PEP 249 error to raise for `exc`, carrying its message.

    Args:
        exc: What the engine raised.

    Returns:
        An `Error` instance; raise it ``from exc`` so the original stays reachable.
    """
    if isinstance(exc, Error):
        return exc
    for source, target in _MAPPING:
        if isinstance(exc, source):
            return target(str(exc))
    return DatabaseError(f"{type(exc).__name__}: {exc}")
