"""Connection-URI parsing — one industry-standard URI, routed to the right backend.

Every SQL connector here takes its own native connection spelling: ADBC wants a
``driver`` plus a ``db_kwargs`` dict, ConnectorX wants a SQLAlchemy-style URI string,
ODBC wants a DSN or a semicolon-delimited connection string, ClickHouse wants discrete
host/port/user/password fields. That is six different ways to say "this database", and
it is the single largest source of friction when moving an existing warehouse workload
onto Batcher: the query ports unchanged, the connection does not.

This module makes the **RFC 3986 / SQLAlchemy-style URI** the one spelling users need —
``postgresql://user@host:5432/mydb``, ``snowflake://account/db/schema``,
``sqlite:///local.db`` — and resolves it to whichever backend can actually serve it. The
scheme vocabulary deliberately matches SQLAlchemy's, because that is what users already
have in a config file, an ``$DATABASE_URL``, or a dbt profile; a URI that works in
SQLAlchemy, pandas' ``read_sql``, or Polars' ``read_database_uri`` works here unchanged.

**The password never travels in the URI.** A connection URI is the string that reaches
log lines, error messages, and `identity()`, so a password embedded in its userinfo
leaks everywhere the URI is merely *mentioned*. `parse_uri` therefore lifts any inline
password out into a separate `ParsedURI.password` field (``repr=False``) and returns a
`uri` with the userinfo password removed.

That separation is not by itself encryption: the password is still carried, and still
pickled onto the split that ships to every worker. To keep the secret out of the process
image entirely, pass an ``env:``/``file:`` reference (see `batcher.io.credentials`) as
``password=`` — the reference is what gets pickled, and it is resolved to a secret only
on the worker, at connect time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl, quote, unquote, urlsplit

from batcher._internal.errors import BackendError, ConfigError

__all__ = [
    "ParsedURI",
    "adbc_connection",
    "known_schemes",
    "parse_uri",
    "redact_uri",
]

#: Schemes served by an ADBC driver, mapped to the driver module ADBC loads.
#:
#: These are the backends with a first-class Arrow path: the driver hands back Arrow
#: directly, so there is no row-wise conversion between the server and the engine.
_ADBC_DRIVERS: dict[str, str] = {
    "postgresql": "adbc_driver_postgresql",
    "postgres": "adbc_driver_postgresql",
    "sqlite": "adbc_driver_sqlite",
    "duckdb": "adbc_driver_duckdb",
    "snowflake": "adbc_driver_snowflake",
    "bigquery": "adbc_driver_bigquery",
    "flightsql": "adbc_driver_flightsql",
    "grpc": "adbc_driver_flightsql",
    "grpc+tcp": "adbc_driver_flightsql",
    "grpc+tls": "adbc_driver_flightsql",
    # --- PostgreSQL wire-protocol databases -------------------------------------
    #
    # These are not PostgreSQL, but they speak its wire protocol, which is the only
    # thing the driver needs: `adbc_driver_postgresql` connects, executes, and returns
    # Arrow against every one of them. Without a scheme entry a user holding a perfectly
    # workable `cockroachdb://` URI is told the database is unsupported, when the only
    # missing piece is this line.
    #
    # The caveat is real and worth stating: wire compatibility is not SQL compatibility.
    # The connection works; a query using a PostgreSQL-specific function these engines
    # do not implement still fails, on the server, with the server's own error. That is
    # the right place for it to fail, and it is the same trade a psycopg user makes.
    "cockroachdb": "adbc_driver_postgresql",
    "cockroach": "adbc_driver_postgresql",
    "timescaledb": "adbc_driver_postgresql",
    "alloydb": "adbc_driver_postgresql",
    "greenplum": "adbc_driver_postgresql",
    "yugabytedb": "adbc_driver_postgresql",
    "yugabyte": "adbc_driver_postgresql",
    "risingwave": "adbc_driver_postgresql",
    "materialize": "adbc_driver_postgresql",
    "questdb": "adbc_driver_postgresql",
    "crate": "adbc_driver_postgresql",
    "cratedb": "adbc_driver_postgresql",
}

#: Schemes with no ADBC driver, served by ConnectorX's Arrow reader instead.
#:
#: ConnectorX is still Arrow-native end to end; it simply owns its own connection
#: vocabulary, which is the same SQLAlchemy URI we already parsed.
_CONNECTORX_SCHEMES: frozenset[str] = frozenset(
    {
        "mysql",
        "mariadb",
        "mssql",
        "sqlserver",
        "oracle",
        "redshift",
        "trino",
        "clickhouse",
        # MySQL wire-protocol databases — same reasoning as the PostgreSQL block above:
        # ConnectorX's MySQL reader connects to each of these unchanged, and the only
        # thing standing between a user and a working read was a missing scheme name.
        "singlestore",
        "memsql",
        "tidb",
        "starrocks",
        "doris",
        "percona",
        # Deliberately NOT here: `presto`. Presto and Trino diverged after the fork, and
        # ConnectorX ships a *Trino* reader — routing Presto to it would be a guess
        # dressed up as support. It stays in `_ALTERNATIVE_ROUTES`, pointing at ODBC.
    }
)

#: SQLAlchemy dialect+driver suffixes to strip: the DBAPI driver is SQLAlchemy's
#: concern, not ours, and ``postgresql+psycopg2://`` must resolve exactly like
#: ``postgresql://``. Users paste these verbatim out of existing config.
_DIALECT_SEPARATOR = "+"

#: Schemes whose "host" component is really a locator, not a network address.
#: ``sqlite:///path/to.db`` and ``duckdb:///path/to.db`` address a local file.
_FILE_BACKED = frozenset({"sqlite", "duckdb"})

#: Stores Batcher *can* read, but not from a URI — mapped to the call that does work.
#:
#: These schemes are ones a user plausibly types. Answering them with the generic
#: "supported schemes are …" list is technically true and useless: it lists SQL
#: databases to someone holding a MongoDB URI, implying Batcher cannot read MongoDB at
#: all, when in fact it can and the only issue is the spelling. Naming the working call
#: turns a dead end into a one-line fix.
_ALTERNATIVE_ROUTES: dict[str, str] = {
    "databricks": "bt.read.databricks(table=..., workspace=..., token=...)",
    "mongodb": "bt.read.table('mongo', ...)",
    "mongodb+srv": "bt.read.table('mongo', ...)",
    "elasticsearch": "bt.read.table('elasticsearch', ...)",
    "dynamodb": "bt.read.table('dynamodb', ...)",
    "cassandra": "bt.read.table('cassandra', ...)",
    "redis": "bt.read.table('redis', ...)",
    "neo4j": "bt.read.table('neo4j', ...)",
    "couchbase": "bt.read.table('couchbase', ...)",
    # Reached through ODBC, which takes a DSN or a driver connection string rather
    # than a URI — there is no URI spelling for these at all.
    "db2": "bt.read.table('odbc', connection_string=...)",
    "teradata": "bt.read.table('odbc', connection_string=...)",
    "hana": "bt.read.table('odbc', connection_string=...)",
    "vertica": "bt.read.table('odbc', connection_string=...)",
    "hive": "bt.read.table('odbc', connection_string=...)",
    "athena": "bt.read.table('odbc', connection_string=...)",
    "presto": "bt.read.table('odbc', connection_string=...)",
}


@dataclass(frozen=True, slots=True)
class ParsedURI:
    """A database connection URI resolved to a concrete Batcher backend.

    Attributes:
        backend: The `SOURCES` registry name that can serve this URI — ``"adbc"``
            for a driver-backed database, ``"connectorx"`` otherwise.
        scheme: The normalized scheme, with any ``+driver`` suffix stripped.
        driver: The ADBC driver module to load, or None for a non-ADBC backend.
        uri: The connection URI **with any inline password removed**, safe to log
            and to carry on a pickled split.
        username: The username, or None when the URI carried none.
        password: The password as given — a literal, or an ``env:``/``file:``
            reference resolved on the worker. Never included in `uri`.
        database: The database/catalog name from the URI path, if any.
        options: Query-string parameters (``?sslmode=require``) as a dict.
    """

    backend: str
    scheme: str
    driver: str | None
    uri: str
    username: str | None = None
    password: str | None = field(default=None, repr=False)
    database: str | None = None
    # `repr=False`: a query string is ordinary connection options most of the time,
    # but nothing stops a URI carrying `?password=` or `?token=` there.
    options: dict[str, str] = field(default_factory=dict, repr=False)

    def db_kwargs(self) -> dict[str, Any]:
        """The driver connection kwargs for this URI, minus the password.

        The password is deliberately absent: it is resolved and merged in at connect
        time on the worker, so it is never part of the pickled split. Callers pass it
        separately via the source's own credential field.

        Returns:
            Driver kwargs suitable for `ADBCSource.db_kwargs`.
        """
        kwargs: dict[str, Any] = {"uri": self.uri}
        if self.username:
            kwargs["username"] = self.username
        kwargs.update(self.options)
        return kwargs


#: Schemes whose server accepts a trailing ``LIMIT n``.
#:
#: Deliberately an allow-list rather than a deny-list, because the two ways of being
#: wrong here are not symmetric. Failing to cap a read costs the rows it would have
#: skipped; emitting a cap the server cannot parse turns a working query into a syntax
#: error. So a scheme absent from this set — including one added later, and including
#: `mssql` and `oracle`, which spell the same thing as ``TOP`` and ``FETCH FIRST`` — is
#: read uncapped, exactly as every scheme was before row-cap pushdown existed.
#:
#: ODBC is not routed by scheme at all (its DSN names a driver, not a dialect), which is
#: why `ODBCSource` never caps.
_LIMIT_CLAUSE_SCHEMES: frozenset[str] = frozenset(
    {
        # PostgreSQL and its wire-protocol family.
        "postgres",
        "postgresql",
        "cockroach",
        "cockroachdb",
        "timescaledb",
        "alloydb",
        "greenplum",
        "yugabyte",
        "yugabytedb",
        "risingwave",
        "materialize",
        "questdb",
        "crate",
        "cratedb",
        "redshift",
        # MySQL and its wire-protocol family.
        "mysql",
        "mariadb",
        "singlestore",
        "memsql",
        "tidb",
        "starrocks",
        "doris",
        "percona",
        # Everything else that spells a row cap `LIMIT n`.
        "sqlite",
        "duckdb",
        "clickhouse",
        "snowflake",
        "bigquery",
        "trino",
    }
)


#: How each dialect family delimits an identifier: ``(open, close)``.
#:
#: MySQL is the reason this cannot be one constant. ANSI double quotes only delimit an
#: identifier there under ``ANSI_QUOTES``, which is off by default — so ``"user"`` is a
#: *string literal*, and quoting a column that way would silently select the constant
#: ``'user'`` for every row instead of failing. Backticks are unambiguous on MySQL and its
#: wire-protocol family, brackets on SQL Server, double quotes everywhere else here.
_QUOTE_STYLES: dict[str, tuple[str, str]] = {
    "backtick": ("`", "`"),
    "bracket": ("[", "]"),
    "ansi": ('"', '"'),
}

#: Schemes that delimit with backticks rather than ANSI double quotes.
_BACKTICK_SCHEMES: frozenset[str] = frozenset(
    {
        "mysql",
        "mariadb",
        "singlestore",
        "memsql",
        "tidb",
        "starrocks",
        "doris",
        "percona",
        "bigquery",
    }
)

#: Schemes that delimit with square brackets.
_BRACKET_SCHEMES: frozenset[str] = frozenset({"mssql", "sqlserver"})

#: Schemes that delimit with ANSI double quotes.
_ANSI_QUOTE_SCHEMES: frozenset[str] = frozenset(
    {
        "postgres",
        "postgresql",
        "cockroach",
        "cockroachdb",
        "timescaledb",
        "alloydb",
        "greenplum",
        "yugabyte",
        "yugabytedb",
        "risingwave",
        "materialize",
        "questdb",
        "crate",
        "cratedb",
        "redshift",
        "sqlite",
        "duckdb",
        "clickhouse",
        "snowflake",
        "trino",
        "oracle",
    }
)


#: Schemes whose ``ORDER BY`` accepts an explicit ``NULLS FIRST`` / ``NULLS LAST``.
#:
#: Required, not cosmetic. Servers disagree about where a null sorts by default — SQLite
#: puts nulls *first* on an ascending order where PostgreSQL and DuckDB put them last — so
#: a top-N pushed without saying which would ask the server for a different "first n" than
#: the engine would compute, and return the wrong rows rather than merely reading extra
#: ones. Measured on sqlite 3.52: ``ORDER BY k LIMIT 2`` over ``[3, NULL, 1, NULL, 2]``
#: yields ``[NULL, NULL]`` there and ``[1, 2]`` in DuckDB; with ``NULLS LAST`` both give
#: ``[1, 2]``, which is what Batcher's own sort gives.
#:
#: MySQL and SQL Server have no such clause at all, so a top-N simply does not push to
#: them and the read is unordered and uncapped, exactly as it was.
_NULLS_ORDER_SCHEMES: frozenset[str] = frozenset(
    {
        "postgres",
        "postgresql",
        "cockroach",
        "cockroachdb",
        "timescaledb",
        "alloydb",
        "greenplum",
        "yugabyte",
        "yugabytedb",
        "risingwave",
        "materialize",
        "questdb",
        "crate",
        "cratedb",
        "redshift",
        "sqlite",
        "duckdb",
        "clickhouse",
        "snowflake",
        "bigquery",
        "trino",
        "oracle",
    }
)


def supports_nulls_ordering(scheme: str) -> bool:
    """Whether `scheme` accepts ``ORDER BY x NULLS FIRST|LAST``.

    Args:
        scheme: A connection-URI scheme, with or without a ``+driver`` suffix.

    Returns:
        True when a top-N may be pushed to this backend.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.uri import supports_nulls_ordering
            >>> supports_nulls_ordering("postgresql")
            True
            >>> supports_nulls_ordering("mysql")
            False
    """
    return _normalize_scheme(scheme) in _NULLS_ORDER_SCHEMES


def quote_identifier(name: str, scheme: str) -> str:
    """`name` delimited for `scheme`, or unchanged when the dialect is unknown.

    Column names reach the server verbatim otherwise, and three ordinary names break that
    way. A reserved word (``order``, ``user``, ``key``, ``value``, ``date``) is a syntax
    error. A name holding a space is worse than an error: ``SELECT my col`` parses as the
    column ``my`` *aliased* to ``col``, so the query succeeds and returns the wrong column
    under the right name. And an unaliased aggregate in a user's own query yields a result
    column literally called ``count(*)``, which unquoted is re-parsed as a function call.

    Quoting is safe here precisely because the names are not the user's free text: a
    projection comes from the plan, which was validated against the schema the *server*
    reported, so it already carries the server's own spelling and case. That is what makes
    the case-folding question moot — an unquoted identifier folds, but there is nothing to
    fold it to that differs from what is already written.

    An unrecognized scheme is returned unchanged rather than guessed at, for the same
    reason `supports_limit_clause` is an allow-list: quoting with the wrong delimiter is a
    new failure, while not quoting is the behavior every read already had.

    Args:
        name: The identifier to delimit.
        scheme: A connection-URI scheme, with or without a ``+driver`` suffix.

    Returns:
        The delimited identifier, or `name` unchanged for an unknown dialect.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.uri import quote_identifier
            >>> quote_identifier("order", "postgresql")
            '"order"'
            >>> quote_identifier("order", "mysql")
            '`order`'
            >>> quote_identifier("order", "some-unknown-database")
            'order'
    """
    style = _quote_style(scheme)
    if style is None:
        return name
    open_char, close_char = _QUOTE_STYLES[style]
    # Doubling the closing delimiter is how all three dialect families escape it.
    return f"{open_char}{name.replace(close_char, close_char * 2)}{close_char}"


def _quote_style(scheme: str) -> str | None:
    """The delimiter family for `scheme`, or None when it is not recognized."""
    normalized = _normalize_scheme(scheme)
    if normalized in _BACKTICK_SCHEMES:
        return "backtick"
    if normalized in _BRACKET_SCHEMES:
        return "bracket"
    if normalized in _ANSI_QUOTE_SCHEMES:
        return "ansi"
    return None


def supports_limit_clause(scheme: str) -> bool:
    """Whether `scheme`'s server accepts a trailing ``LIMIT n``.

    Args:
        scheme: A connection-URI scheme, with or without a ``+driver`` suffix.

    Returns:
        True when a row cap can be appended to a query for this backend.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.uri import supports_limit_clause
            >>> supports_limit_clause("postgresql+psycopg2")
            True
            >>> supports_limit_clause("mssql")
            False
    """
    return _normalize_scheme(scheme) in _LIMIT_CLAUSE_SCHEMES


def known_schemes() -> tuple[str, ...]:
    """Every connection-URI scheme Batcher can route, sorted.

    Returns:
        The supported scheme names, e.g. ``("bigquery", "clickhouse", ...)``.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.uri import known_schemes
            >>> "postgresql" in known_schemes()
            True
    """
    return tuple(sorted(set(_ADBC_DRIVERS) | _CONNECTORX_SCHEMES))


def _normalize_scheme(raw: str) -> str:
    """Strip a SQLAlchemy ``+driver`` suffix and lowercase.

    ``postgresql+psycopg2`` and ``POSTGRESQL`` both resolve to ``postgresql``. The
    grpc FlightSQL schemes are the exception — their suffix names a transport, not a
    DBAPI driver, so it is load-bearing and kept.
    """
    scheme = raw.lower()
    if scheme in _ADBC_DRIVERS:
        return scheme
    head, sep, _tail = scheme.partition(_DIALECT_SEPARATOR)
    return head if sep else scheme


def _rebuild_without_password(parts: Any, scheme: str) -> str:
    """Reassemble the URI with the userinfo password stripped.

    `urlsplit` gives us the components; putting them back by hand (rather than with
    `urlunsplit` on a mutated netloc) is what lets us drop exactly the password and
    keep everything else — including a username containing an escaped ``@`` — byte
    for byte as the user wrote it.
    """
    netloc = ""
    if parts.username:
        netloc += quote(unquote(parts.username), safe="")
        netloc += "@"
    if parts.hostname:
        netloc += parts.hostname
    if parts.port:
        netloc += f":{parts.port}"
    rebuilt = f"{scheme}://{netloc}{parts.path}"
    if parts.query:
        rebuilt += f"?{parts.query}"
    return rebuilt


def redact_uri(uri: str) -> str:
    """Return `uri` with any inline password replaced by ``***``.

    Use this anywhere a connection URI reaches a log line, an error message, or a
    ``repr`` — an embedded password would otherwise leak wherever that string lands.

    Args:
        uri: A connection URI, possibly carrying ``user:password@`` userinfo.

    Returns:
        The URI with the password replaced, or unchanged when it carried none.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.uri import redact_uri
            >>> redact_uri("postgresql://alice:hunter2@db:5432/app")
            'postgresql://alice:***@db:5432/app'
            >>> redact_uri("postgresql://db:5432/app")
            'postgresql://db:5432/app'
    """
    parts = urlsplit(uri)
    if not parts.password:
        return uri
    user = parts.username or ""
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    rest = parts.path + (f"?{parts.query}" if parts.query else "")
    return f"{parts.scheme}://{user}:***@{host}{port}{rest}"


def adbc_connection(
    uri: str,
    *,
    password: str | None = None,
    driver: str | None = None,
    db_kwargs: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], str]:
    """Resolve a connection URI into ADBC's ``(driver, db_kwargs, sanitized_uri)``.

    Shared by `ADBCSource` and `ADBCSink` so reading and writing accept exactly the same
    connection spelling. An explicit `driver`/`db_kwargs` always wins over anything
    derived from `uri`, so the URI is a convenience and never a constraint.

    Args:
        uri: The connection URI.
        password: The password, as a literal or an ``env:``/``file:`` reference. It is
            placed in the returned kwargs **unresolved**, for the worker to resolve.
        driver: An explicit ADBC driver that overrides the one `uri` implies.
        db_kwargs: Explicit driver kwargs, merged over the derived ones.

    Returns:
        The driver module name, the connection kwargs, and the URI with any inline
        password removed (safe to keep on a field that has a ``repr``).

    Raises:
        BackendError: If `uri` names a scheme with no ADBC driver.
    """
    parsed = parse_uri(uri, password=password)
    if parsed.backend != "adbc":
        raise BackendError(
            f"{parsed.scheme!r} has no ADBC driver; read it with "
            f"bt.read.sql(query, uri=...), which routes to the {parsed.backend!r} backend, "
            "or call bt.read.table('connectorx', ...) directly."
        )
    merged: dict[str, Any] = dict(parsed.db_kwargs())
    if parsed.password is not None:
        merged["password"] = parsed.password
    merged.update(db_kwargs or {})
    resolved = driver or parsed.driver
    if resolved is None:
        raise ConfigError(
            f"no ADBC driver for {parsed.uri!r}; pass driver=... or use a scheme with a "
            "registered driver"
        )
    return resolved, merged, parsed.uri


def _database_from_path(path: str, scheme: str) -> str | None:
    """The database/catalog name a URI path addresses, or None.

    For a file-backed scheme the path *is* the locator (``sqlite:///data.db``), so it
    is not a database name and is left on the URI untouched.
    """
    if scheme in _FILE_BACKED:
        return None
    stripped = path.lstrip("/")
    return stripped or None


def parse_uri(uri: str, *, password: str | None = None) -> ParsedURI:
    """Resolve a SQLAlchemy-style connection URI to a Batcher backend.

    The scheme vocabulary matches SQLAlchemy, pandas' ``read_sql``, and Polars'
    ``read_database_uri``, so an existing ``$DATABASE_URL`` works unchanged. A
    ``+driver`` suffix (``postgresql+psycopg2``) is accepted and ignored — which DBAPI
    driver SQLAlchemy would have used is not meaningful here, since the read goes
    through ADBC or ConnectorX in Arrow.

    Args:
        uri: The connection URI, e.g. ``"postgresql://user@host:5432/mydb"``.
        password: The password, as a literal or an ``env:``/``file:`` reference. Takes
            precedence over any password embedded in `uri`.

    Returns:
        The parsed URI with its resolved `backend` and driver.

    Raises:
        BackendError: If `uri` has no scheme, or names a scheme Batcher cannot route.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.uri import parse_uri
            >>> parsed = parse_uri("postgresql://alice@db:5432/app")
            >>> parsed.backend, parsed.driver, parsed.database
            ('adbc', 'adbc_driver_postgresql', 'app')

            >>> parse_uri("mysql+pymysql://svc@db/shop").backend
            'connectorx'
    """
    if not isinstance(uri, str) or "://" not in uri:
        raise BackendError(
            f"invalid connection URI {redact_uri(str(uri))!r}: expected a scheme, e.g. "
            "'postgresql://user@host:5432/dbname'"
        )
    parts = urlsplit(uri)
    scheme = _normalize_scheme(parts.scheme)
    if scheme not in _ADBC_DRIVERS and scheme not in _CONNECTORX_SCHEMES:
        if scheme in _ALTERNATIVE_ROUTES:
            raise BackendError(
                f"Batcher reads {scheme!r}, but not from a connection URI — use "
                f"{_ALTERNATIVE_ROUTES[scheme]} instead."
            )
        raise BackendError(
            f"unsupported database scheme {scheme!r}; Batcher can route "
            f"{', '.join(known_schemes())}. For a driver with no URI scheme, construct the "
            "source directly, e.g. bt.read.table('odbc', connection_string=...)."
        )
    # An inline password is lifted out rather than trusted in place: the rebuilt URI is
    # what gets pickled onto every split, and a password there would ship with it.
    secret = password if password is not None else parts.password
    return ParsedURI(
        backend="adbc" if scheme in _ADBC_DRIVERS else "connectorx",
        scheme=scheme,
        driver=_ADBC_DRIVERS.get(scheme),
        uri=_rebuild_without_password(parts, scheme),
        username=unquote(parts.username) if parts.username else None,
        password=secret,
        database=_database_from_path(parts.path, scheme),
        options=dict(parse_qsl(parts.query)),
    )
