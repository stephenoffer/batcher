"""Which SQL backend serves this call — the one router the read and the write share.

Batcher reaches a SQL database three ways, and the user picks none of them. ADBC is
Arrow-native and covers PostgreSQL, SQLite, DuckDB, Snowflake, BigQuery and FlightSQL.
ConnectorX is Arrow-native too and covers most of the rest, read-only. PEP 249 covers
everything, more slowly, and is the only one that can execute a statement — which is what
an ``UPDATE`` or an upsert is.

Routing used to live in two places that disagreed. `bt.read.sql` routed by scheme;
``ds.write.sql`` did not route at all, sending every write to ADBC. Three defects followed
from the same root, and each of them made a working database look unsupported:

* ``ds.write.sql(table, uri="mysql://…")`` raised "no ADBC driver" on a database
  ``bt.read.sql`` reads happily, because MySQL has no ADBC driver and the write never
  considered a second backend.
* ``bt.read.sql(query, module="psycopg", …)`` raised ``ADBCSource.__init__() got an
  unexpected keyword argument 'module'`` — a `connection=` routed to the DB-API source but
  the ``module=``/``connect_kwargs=`` spelling of the same thing did not.
* Either call failed for a *missing dependency* while a driver that could serve it was
  already installed: ``sqlite3`` ships with Python and ``adbc_driver_sqlite`` does not, and
  routing on "this scheme has an ADBC driver" cannot see the difference.

## The rule

Prefer the Arrow-native backend when it is **installed**, because it never materializes a
Python object. Fall back to PEP 249 when it is not, or when the operation needs a statement
rather than an ingest. Availability is checked with `importlib.util.find_spec`, which
resolves the module without importing it — so choosing a backend costs no import, and
`import batcher` stays free of the driver stack.
"""

from __future__ import annotations

from typing import Any

__all__ = ["read_backend", "write_backend"]

#: Keywords that name a PEP 249 connection directly, whichever backend the URI implies.
_DBAPI_KEYWORDS = ("module", "connection", "connect_kwargs", "paramstyle")

#: Keywords that name an ADBC connection directly.
_ADBC_KEYWORDS = ("driver", "db_kwargs", "conn_kwargs")

#: The write modes an Arrow bulk ingest can express. Everything else needs a statement.
_INGEST_MODES = frozenset({"append", "overwrite", "create", "replace", "create_append"})


def _installed(module: str) -> bool:
    """Whether `module` can be imported, resolved without importing it."""
    from batcher.io.formats.sql.dbapi._dsn import module_available

    return module_available(module)


def _dbapi_available(scheme: str) -> bool:
    """Whether a PEP 249 driver for `scheme` is installed on this machine."""
    from batcher.io.formats.sql.dbapi._dsn import installed_driver

    return installed_driver(scheme) is not None


def read_backend(uri: str | None, opts: dict[str, Any]) -> str:
    """The `SOURCES` name that should serve this read.

    Args:
        uri: The connection URI, or None when the caller named a connection directly.
        opts: The read's keyword options.

    Returns:
        ``"adbc"``, ``"connectorx"`` or ``"dbapi"``.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.routing import read_backend
            >>> read_backend(None, {"module": "psycopg"})
            'dbapi'
    """
    if any(key in opts for key in _DBAPI_KEYWORDS):
        return "dbapi"
    if uri is None:
        return "adbc"
    from batcher.io.formats.sql.uri import parse_uri

    parsed = parse_uri(uri)
    if parsed.backend == "adbc":
        if _installed("adbc_driver_manager") and _installed(str(parsed.driver)):
            return "adbc"
    elif _installed("connectorx"):
        return "connectorx"
    # The Arrow-native backend for this scheme is not installed. Fall back to PEP 249 only
    # if one of *its* drivers is, so a genuinely uninstallable read still names the
    # dependency the user most likely meant rather than a second one they have never heard
    # of.
    if _dbapi_available(parsed.scheme):
        return "dbapi"
    return parsed.backend


def write_backend(mode: str, opts: dict[str, Any]) -> str:
    """The `SINKS` name that should serve this write.

    Args:
        mode: The write mode the caller asked for.
        opts: The write's keyword options.

    Returns:
        ``"adbc"`` or ``"dbapi"``.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.routing import write_backend
            >>> write_backend("upsert", {"uri": "postgresql://h/db"})
            'dbapi'
    """
    if mode not in _INGEST_MODES:
        return "dbapi"
    if any(key in opts for key in _DBAPI_KEYWORDS):
        return "dbapi"
    if any(key in opts for key in _ADBC_KEYWORDS):
        return "adbc"
    uri = opts.get("uri")
    if not isinstance(uri, str):
        return "adbc"
    from batcher.io.formats.sql.uri import parse_uri

    parsed = parse_uri(uri)
    if parsed.driver is None:
        # ConnectorX is read-only, so a scheme it serves has no Arrow-native *write* path
        # at all and PEP 249 is the only route. Returning "adbc" here would fail with
        # "unsupported scheme", which is both wrong and unactionable; the DB-API error
        # names the drivers that would work.
        return "dbapi"
    if _installed("adbc_driver_manager") and _installed(parsed.driver):
        return "adbc"
    return "dbapi" if _dbapi_available(parsed.scheme) else "adbc"
