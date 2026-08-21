"""Connection URI → the PEP 249 driver and the ``connect()`` kwargs it wants.

`uri.parse_uri` resolves a URI as far as *which backend serves it*, which is all an
Arrow-native reader needs: ADBC and ConnectorX both take the whole URI. A PEP 249 driver
does not. It takes discrete keyword arguments, and no two driver families spell them the
same way — ``dbname`` for psycopg, ``database`` for pg8000 and PyMySQL, a ``dsn`` string
for Oracle, a bare file path for SQLite. This module is that last mile, so the same
``postgresql://…`` URI a user reads with also writes with.

## Why a candidate list rather than one driver per scheme

There is no single Python driver for PostgreSQL or for MySQL — there are three of each,
and which one is installed is the user's decision, usually made years earlier by some
other part of their stack. Naming one would make a working environment look unsupported.
So each scheme carries the drivers that speak it in preference order, newest first, and
the first one that imports is used. `driver_for` reports what it chose, and the error when
none is present lists every alternative rather than only the first.

## What it does not do

Snowflake, BigQuery, Databricks and Trino authenticate with account identifiers, tokens or
service-account keys that a host/port/database URI cannot express. Each already has a
dedicated connector, and inventing a URI mapping for them here would be a worse spelling of
a thing that already works. They raise, naming the connector that does.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from batcher._internal.errors import BackendError, MissingDependencyError
from batcher.io.formats.sql.uri import ParsedURI

__all__ = [
    "DRIVER_CANDIDATES",
    "connect_target",
    "driver_for",
    "installed_driver",
    "module_available",
    "resolve_connection",
]

#: Scheme → the PEP 249 drivers that speak it, in preference order.
#:
#: Preference is "the actively maintained one first": psycopg 3 over psycopg2 over pg8000,
#: PyMySQL over mysqlclient over the vendor connector, python-oracledb over cx_Oracle.
DRIVER_CANDIDATES: dict[str, tuple[str, ...]] = {
    "postgresql": ("psycopg", "psycopg2", "pg8000"),
    "postgres": ("psycopg", "psycopg2", "pg8000"),
    "cockroachdb": ("psycopg", "psycopg2", "pg8000"),
    "cockroach": ("psycopg", "psycopg2", "pg8000"),
    "timescaledb": ("psycopg", "psycopg2", "pg8000"),
    "alloydb": ("psycopg", "psycopg2", "pg8000"),
    "greenplum": ("psycopg", "psycopg2", "pg8000"),
    "yugabytedb": ("psycopg", "psycopg2", "pg8000"),
    "yugabyte": ("psycopg", "psycopg2", "pg8000"),
    "risingwave": ("psycopg", "psycopg2", "pg8000"),
    "materialize": ("psycopg", "psycopg2", "pg8000"),
    "questdb": ("psycopg", "psycopg2", "pg8000"),
    "crate": ("psycopg", "psycopg2", "pg8000"),
    "cratedb": ("psycopg", "psycopg2", "pg8000"),
    "redshift": ("psycopg", "psycopg2", "pg8000"),
    "mysql": ("pymysql", "MySQLdb", "mysql.connector"),
    "mariadb": ("pymysql", "MySQLdb", "mysql.connector"),
    "tidb": ("pymysql", "MySQLdb", "mysql.connector"),
    "singlestore": ("pymysql", "MySQLdb", "mysql.connector"),
    "memsql": ("pymysql", "MySQLdb", "mysql.connector"),
    "percona": ("pymysql", "MySQLdb", "mysql.connector"),
    "starrocks": ("pymysql", "MySQLdb", "mysql.connector"),
    "doris": ("pymysql", "MySQLdb", "mysql.connector"),
    "sqlite": ("sqlite3",),
    "duckdb": ("duckdb",),
    "oracle": ("oracledb", "cx_Oracle"),
    "clickhouse": ("clickhouse_driver",),
}

#: Schemes whose "host" is a local file path rather than a network address.
_FILE_BACKED = frozenset({"sqlite", "duckdb"})

#: Drivers spelling the database name ``dbname`` rather than ``database``.
_DBNAME_DRIVERS = frozenset({"psycopg", "psycopg2"})

#: Schemes reachable only through their own connector, mapped to the call that works.
#:
#: Each authenticates with something a host/port/database URI cannot carry, so a mapping
#: invented here would be a second, worse spelling of a working connector.
_DEDICATED: dict[str, str] = {
    "snowflake": "ds.write.snowflake(table, connection_kwargs=...)",
    "bigquery": "bt.read.table('bigquery', ...)",
    "databricks": "bt.read.databricks(...)",
    "trino": "ds.write(table, 'dbapi', module='trino.dbapi', connect_kwargs=...)",
    "mssql": "ds.write(table, 'dbapi', module='pymssql', connect_kwargs=...)",
    "sqlserver": "ds.write(table, 'dbapi', module='pymssql', connect_kwargs=...)",
}


def module_available(module: str) -> bool:
    """Whether `module` can be imported, resolved without importing it.

    `importlib.util.find_spec` is the cheap answer, and it does not merely return None for
    a module that is absent: asked about a submodule it *imports the parent package first*,
    so ``find_spec("mysql.connector")`` raises `ModuleNotFoundError` on a machine with no
    ``mysql`` package at all rather than answering False. Every caller here is asking a
    yes/no question about an optional driver, so the exception is the answer "no".

    Args:
        module: An importable module name, possibly dotted.

    Returns:
        Whether the module can be imported.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.dbapi._dsn import module_available
            >>> module_available("sqlite3")
            True
            >>> module_available("mysql.connector.definitely_not_here")
            False
    """
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
        # ImportError: the parent package is absent or fails to import. AttributeError /
        # ValueError: a namespace package with no loader, or a stale entry in sys.modules.
        return False


def installed_driver(scheme: str) -> str | None:
    """The first installed PEP 249 driver for `scheme`, or None when there is none.

    The question `driver_for` answers, asked where the answer "none" is ordinary rather
    than an error: a caller choosing *between* backends needs to know whether this one can
    serve the write, not to be handed an exception about it.

    Args:
        scheme: A normalized connection-URI scheme.

    Returns:
        The importable module name, or None.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.dbapi._dsn import installed_driver
            >>> installed_driver("sqlite")
            'sqlite3'
            >>> installed_driver("nonesuch") is None
            True
    """
    for candidate in DRIVER_CANDIDATES.get(scheme, ()):
        if module_available(candidate):
            return candidate
    return None


def driver_for(scheme: str) -> str:
    """The first installed PEP 249 driver for `scheme`.

    Args:
        scheme: A normalized connection-URI scheme, e.g. ``"postgresql"``.

    Returns:
        The importable module name of the driver to use.

    Raises:
        BackendError: If Batcher maps no PEP 249 driver to `scheme`.
        MissingDependencyError: If it maps some, and none of them is installed.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.dbapi._dsn import driver_for
            >>> driver_for("sqlite")
            'sqlite3'
    """
    candidates = DRIVER_CANDIDATES.get(scheme)
    if not candidates:
        if scheme in _DEDICATED:
            raise BackendError(
                f"a {scheme!r} URI has no PEP 249 spelling Batcher can derive — it "
                f"authenticates with credentials a host/port/database URI cannot carry. "
                f"Use {_DEDICATED[scheme]} instead."
            )
        raise BackendError(
            f"no PEP 249 driver is mapped to scheme {scheme!r}. Pass module= and "
            "connect_kwargs= naming your driver directly, e.g. "
            "ds.write(table, 'dbapi', module='pyodbc', connect_kwargs={...})."
        )
    found = installed_driver(scheme)
    if found is not None:
        return found
    raise MissingDependencyError(
        f"writing to {scheme!r} over PEP 249 needs one of {', '.join(candidates)} "
        f"installed; none of them is. Install any one of them (pip install "
        f"{candidates[0]}), or pass module= naming a different driver."
    )


def _file_path(parsed: ParsedURI) -> str:
    """The local path a ``sqlite://`` / ``duckdb://`` URI addresses.

    SQLAlchemy's convention is that the *fourth* slash begins an absolute path:
    ``sqlite:///relative.db`` against ``sqlite:////var/db/app.db``. Stripping exactly one
    leading separator is what preserves that distinction; stripping all of them would turn
    every absolute path into a relative one and write the database into the working
    directory instead.
    """
    path = urlsplit(parsed.uri).path
    return path[1:] if path.startswith("/") else path


def connect_target(parsed: ParsedURI, *, module: str | None = None) -> tuple[str, dict[str, Any]]:
    """The driver module and ``connect()`` kwargs `parsed` resolves to.

    The password is carried through **unresolved** — as the ``env:``/``file:`` reference the
    user wrote, if that is what they wrote — because these kwargs are pickled onto every
    worker and `_connect` resolves them on the machine that dials the connection. A resolved
    secret here would be a secret in the pickle.

    Args:
        parsed: The URI, already resolved to a scheme by `uri.parse_uri`.
        module: An explicit driver module, overriding the per-scheme candidate list.

    Returns:
        The driver module name and the keyword arguments its ``connect()`` takes.

    Raises:
        BackendError: If the scheme has no derivable PEP 249 connection.
        MissingDependencyError: If no candidate driver for the scheme is installed.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.uri import parse_uri
            >>> from batcher.io.formats.sql.dbapi._dsn import connect_target
            >>> connect_target(parse_uri("sqlite:////tmp/app.db"))
            ('sqlite3', {'database': '/tmp/app.db'})
    """
    driver = module or driver_for(parsed.scheme)
    if parsed.scheme in _FILE_BACKED:
        return driver, {"database": _file_path(parsed)}
    kwargs: dict[str, Any] = {}
    if parsed.host:
        kwargs["host"] = parsed.host
    if parsed.port is not None:
        kwargs["port"] = parsed.port
    if parsed.username:
        kwargs["user"] = parsed.username
    if parsed.password is not None:
        kwargs["password"] = parsed.password
    if parsed.database:
        kwargs["dbname" if driver in _DBNAME_DRIVERS else "database"] = parsed.database
    if parsed.scheme == "oracle":
        # Oracle addresses a *service*, not a database: python-oracledb takes one `dsn`
        # string in the Easy Connect form rather than host/port/database kwargs.
        service = parsed.database or ""
        host = parsed.host or "localhost"
        port = f":{parsed.port}" if parsed.port is not None else ""
        kwargs = {k: v for k, v in kwargs.items() if k in ("user", "password")}
        kwargs["dsn"] = f"{host}{port}/{service}" if service else f"{host}{port}"
    # Query-string parameters are the driver's own options (`sslmode`, `connect_timeout`,
    # `charset`), passed through verbatim: they are already spelled the way the driver
    # expects, because that is where the user copied them from.
    kwargs.update(parsed.options)
    return driver, kwargs


def resolve_connection(
    uri: str,
    *,
    password: str | None = None,
    module: str | None = None,
    connect_kwargs: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], str, str]:
    """Everything a PEP 249 source or sink needs from a connection URI.

    The read path and the write path resolve a URI identically, and both need the same four
    answers, so they ask once here rather than each growing its own copy. The `connect_kwargs`
    the caller supplied win over the derived ones, which is how a driver-specific option the
    URI cannot express (``cursorclass``, ``autocommit``, a TLS context object) is added
    without giving up URI resolution.

    Args:
        uri: The connection URI.
        password: The password, as a literal or an ``env:``/``file:`` reference.
        module: An explicit driver module, overriding the per-scheme candidate list.
        connect_kwargs: Caller-supplied kwargs, merged over the derived ones.

    Returns:
        The driver module, its ``connect()`` kwargs, the URI with any inline password
        removed, and the dialect scheme.

    Raises:
        BackendError: If the scheme has no derivable PEP 249 connection.
        MissingDependencyError: If no candidate driver for the scheme is installed.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.dbapi._dsn import resolve_connection
            >>> driver, kwargs, safe_uri, scheme = resolve_connection("sqlite:////tmp/a.db")
            >>> driver, scheme
            ('sqlite3', 'sqlite')
    """
    from batcher.io.formats.sql.uri import parse_uri

    parsed = parse_uri(uri, password=password)
    driver, derived = connect_target(parsed, module=module)
    return driver, {**derived, **dict(connect_kwargs or {})}, parsed.uri, parsed.scheme
