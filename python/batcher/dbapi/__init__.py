"""A PEP 249 (DB-API 2.0) adapter over a Batcher `Session`.

``from batcher import dbapi`` gives a standard Python SQL client what it expects to find:
`connect`, a `Connection` with `cursor`/`commit`/`rollback`/`close`, a `Cursor` with
`execute`/`executemany`/`fetchone`/`fetchmany`/`fetchall`/`description`/`rowcount`, the
module globals `apilevel`, `threadsafety` and `paramstyle`, the exception hierarchy, and
the type objects and constructors. It is a bounded subset, documented in
``docs/integrations/sql-clients/dbapi.md``: no transactions, and a `rowcount` of -1.

This is the `dbapi` layer, one above `api`: it uses the public `Session` and `Dataset`, and
nothing below `api` imports it.
"""

from __future__ import annotations

from batcher.dbapi.connection import Connection, apilevel, connect, paramstyle, threadsafety
from batcher.dbapi.cursor import Cursor
from batcher.dbapi.errors import (
    DatabaseError,
    DataError,
    Error,
    IntegrityError,
    InterfaceError,
    InternalError,
    NotSupportedError,
    OperationalError,
    ProgrammingError,
    Warning,
)
from batcher.dbapi.typeobjects import (
    BINARY,
    DATETIME,
    NUMBER,
    ROWID,
    STRING,
    Binary,
    Date,
    DateFromTicks,
    Time,
    TimeFromTicks,
    Timestamp,
    TimestampFromTicks,
)

__all__ = [
    "BINARY",
    "DATETIME",
    "NUMBER",
    "ROWID",
    "STRING",
    "Binary",
    "Connection",
    "Cursor",
    "DataError",
    "DatabaseError",
    "Date",
    "DateFromTicks",
    "Error",
    "IntegrityError",
    "InterfaceError",
    "InternalError",
    "NotSupportedError",
    "OperationalError",
    "ProgrammingError",
    "Time",
    "TimeFromTicks",
    "Timestamp",
    "TimestampFromTicks",
    "Warning",
    "apilevel",
    "connect",
    "paramstyle",
    "threadsafety",
]
