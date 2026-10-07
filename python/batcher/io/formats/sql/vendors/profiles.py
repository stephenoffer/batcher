"""Per-vendor profiles — which route serves a database, what it can write, what its types do.

A generic route is not a certification. ``bt.read.sql(uri="mysql://...")`` reaches MySQL
through ConnectorX or any PEP 249 driver, and nothing about that sentence says what happens
to a ``BIGINT UNSIGNED`` above 2^63, a ``'0000-00-00'`` date, or an upsert. A user who
assumes "generic DB-API" means "everything works" finds out on a production table.

A `VendorProfile` is that missing statement, for the databases people most often point
Batcher at: the routes that serve it (and the package that installs each), the write modes
each route can carry, and the type rules — what Batcher converts, what it refuses and why,
and what it merely passes through from the driver. `capability_matrix` flattens the
profiles into the rows the documentation publishes, and a test holds the published table to
these profiles so the two cannot drift.

Every profile's `status` is ``"untested"``: none of this has run against a live server.
The rules Batcher *enforces* are pinned by unit tests against fake drivers; the rules it
only *declares* describe documented driver behaviour and are marked as such.

Nothing here opens a connection or imports a driver.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "PROFILES",
    "Route",
    "TypeRule",
    "VendorProfile",
    "capability_matrix",
    "install_hint",
    "profile_for",
]

#: The DB-API sink's row-level modes. Restated as a literal here only to avoid importing the
#: sink at module import; `test_vendor_profiles` holds it equal to `dbapi.sink.WRITE_MODES`.
_DBAPI_MODES = ("append", "overwrite", "upsert", "update", "delete", "delete_insert")

#: What an ADBC bulk ingest can express, as Batcher save modes.
_INGEST_MODES = ("append", "overwrite")


@dataclass(frozen=True, slots=True)
class Route:
    """One way Batcher reaches a vendor: a backend, the driver it loads, and what it can do.

    Attributes:
        backend: The `SOURCES`/`SINKS` registry name (``"adbc"``, ``"connectorx"``,
            ``"dbapi"``, ...).
        driver: The importable module the backend loads.
        package: The ``pip install`` name that provides `driver`.
        reads: Whether the route serves reads.
        write_modes: The write modes Batcher builds statements or ingests for on this route.
            Empty for a read-only route.
    """

    backend: str
    driver: str
    package: str
    reads: bool = True
    write_modes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TypeRule:
    """What happens to one problematic vendor type on its way to Arrow.

    Attributes:
        vendor_type: The server-side type, as the vendor spells it.
        rule: What Batcher does with it, or what the driver hands over.
        enforced: True when Batcher's own code applies the rule (and a unit test pins it);
            False when it describes documented driver behaviour Batcher passes through.
    """

    vendor_type: str
    rule: str
    enforced: bool


@dataclass(frozen=True, slots=True)
class VendorProfile:
    """Everything Batcher states about one database vendor.

    Attributes:
        name: The vendor's name, as the documentation prints it.
        schemes: The connection-URI schemes that reach it.
        routes: The routes that serve it, preferred first.
        type_rules: The problematic types and what happens to each.
        live_env: The environment variable naming a live server for the smoke test.
        status: ``"untested"`` until a live run is recorded.
    """

    name: str
    schemes: tuple[str, ...]
    routes: tuple[Route, ...]
    type_rules: tuple[TypeRule, ...]
    live_env: str
    status: str = "untested"


_UNSIGNED = TypeRule(
    "BIGINT UNSIGNED",
    "Arrives as uint64. A value above 2^63-1 is refused naming the column; pass "
    "unsigned='decimal' to read the column as decimal128(20, 0) instead.",
    enforced=True,
)

PROFILES: dict[str, VendorProfile] = {
    "postgresql": VendorProfile(
        name="PostgreSQL",
        schemes=("postgresql", "postgres"),
        routes=(
            Route("adbc", "adbc_driver_postgresql", "adbc-driver-postgresql", True, _INGEST_MODES),
            Route("dbapi", "psycopg", "psycopg[binary]", True, _DBAPI_MODES),
        ),
        type_rules=(
            TypeRule(
                "NUMERIC 'NaN' / 'Infinity' (DB-API)",
                "Refused naming the column: an Arrow decimal cannot hold them. CAST the "
                "column to double precision in the query to keep them.",
                enforced=True,
            ),
            TypeRule(
                "NUMERIC (DB-API)",
                "Python Decimal values become decimal128, or decimal256 past 38 digits, "
                "sized to the values in the first batch.",
                enforced=False,
            ),
            TypeRule(
                "ARRAY (DB-API)",
                "Python lists become list<T>; a multi-dimensional array nests as list<list<T>>.",
                enforced=False,
            ),
            TypeRule(
                "TIMESTAMPTZ (DB-API)",
                "Timezone-aware values become timestamp[us, tz] with every instant preserved.",
                enforced=False,
            ),
        ),
        live_env="BATCHER_LIVE_POSTGRES_URI",
    ),
    "mysql": VendorProfile(
        name="MySQL / MariaDB",
        schemes=("mysql", "mariadb"),
        routes=(
            Route("connectorx", "connectorx", "connectorx"),
            Route("dbapi", "pymysql", "pymysql", True, _DBAPI_MODES),
        ),
        type_rules=(
            _UNSIGNED,
            TypeRule(
                "DATE / DATETIME '0000-00-00' (DB-API)",
                "PyMySQL returns a zero date as a string. Refused naming the column by "
                "default; pass zero_dates='null' to read it as NULL. mysqlclient already "
                "returns None for it.",
                enforced=True,
            ),
            TypeRule(
                "JSON",
                "Arrives as a string column; extract fields with the .json accessor.",
                enforced=False,
            ),
        ),
        live_env="BATCHER_LIVE_MYSQL_URI",
    ),
    "mssql": VendorProfile(
        name="SQL Server",
        schemes=("mssql", "sqlserver"),
        routes=(
            Route("connectorx", "connectorx", "connectorx"),
            Route("dbapi", "pymssql", "pymssql", True, _DBAPI_MODES),
        ),
        type_rules=(
            TypeRule(
                "DECIMAL / NUMERIC (DB-API)",
                "pymssql returns Python Decimal, which becomes an exact Arrow decimal.",
                enforced=False,
            ),
            TypeRule(
                "NVARCHAR / NCHAR (DB-API)",
                "Decoded to string by pymssql, whose connection charset defaults to UTF-8.",
                enforced=False,
            ),
            TypeRule(
                "DATETIME2(7)",
                "Python datetime holds microseconds, so the seventh fractional digit is "
                "truncated by the driver.",
                enforced=False,
            ),
            TypeRule(
                "Bulk writes",
                "Rows are bound through executemany; there is no BCP path. staged=True makes "
                "a distributed append or overwrite atomic.",
                enforced=False,
            ),
        ),
        live_env="BATCHER_LIVE_MSSQL_URI",
    ),
    "oracle": VendorProfile(
        name="Oracle",
        schemes=("oracle",),
        routes=(
            Route("connectorx", "connectorx", "connectorx"),
            Route("dbapi", "oracledb", "oracledb", True, _DBAPI_MODES),
        ),
        type_rules=(
            TypeRule(
                "NUMBER with a fractional scale (DB-API)",
                "python-oracledb returns a float by default. Pass oracle_numbers='decimal' "
                "to fetch every NUMBER as an exact Decimal.",
                enforced=True,
            ),
            TypeRule(
                "TIMESTAMP WITH TIME ZONE (DB-API)",
                "python-oracledb returns a naive datetime, dropping the offset; select "
                "SYS_EXTRACT_UTC(column) to read the instant.",
                enforced=False,
            ),
            TypeRule(
                "CLOB / BLOB (DB-API)",
                "Returned as LOB handles unless the driver is configured to fetch them as "
                "str/bytes; an unconvertible value is refused naming the column and type.",
                enforced=True,
            ),
        ),
        live_env="BATCHER_LIVE_ORACLE_URI",
    ),
    "trino": VendorProfile(
        name="Trino",
        schemes=("trino",),
        routes=(
            Route("connectorx", "connectorx", "connectorx"),
            Route("dbapi", "trino.dbapi", "trino", True, ("append", "overwrite")),
        ),
        type_rules=(
            TypeRule(
                "Authentication",
                "A password in the URI or password= becomes the client's "
                "BasicAuthentication on the worker, over https unless http_scheme is set.",
                enforced=True,
            ),
            TypeRule(
                "Catalog and schema",
                "trino://user@host:443/catalog/schema sets the session catalog and schema.",
                enforced=True,
            ),
            TypeRule(
                "Row-level writes",
                "UPDATE, DELETE and MERGE depend on the catalog connector, so only append "
                "and overwrite are listed; upsert has no Trino spelling in Batcher.",
                enforced=False,
            ),
        ),
        live_env="BATCHER_LIVE_TRINO_URI",
    ),
    "redshift": VendorProfile(
        name="Redshift",
        schemes=("redshift",),
        routes=(
            Route("connectorx", "connectorx", "connectorx"),
            Route("dbapi", "redshift_connector", "redshift-connector", True, _DBAPI_MODES),
        ),
        type_rules=(
            TypeRule(
                "Upsert",
                "Spelled as MERGE, a statement shape not yet run against Redshift. "
                "mode='delete_insert' uses only DELETE and INSERT.",
                enforced=False,
            ),
        ),
        live_env="BATCHER_LIVE_REDSHIFT_URI",
    ),
}

#: Scheme → the profile key, for every scheme a profile names.
_BY_SCHEME: dict[str, str] = {
    scheme: key for key, profile in PROFILES.items() for scheme in profile.schemes
}


def profile_for(scheme: str) -> VendorProfile | None:
    """The vendor profile for a connection-URI scheme, or None when there is none.

    Args:
        scheme: A connection-URI scheme, with or without a ``+driver`` suffix.

    Returns:
        The profile, or None for a scheme no profile covers.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.vendors import profile_for
            >>> profile_for("mysql+pymysql").name
            'MySQL / MariaDB'
            >>> profile_for("informix") is None
            True
    """
    key = _BY_SCHEME.get(scheme.split("+", maxsplit=1)[0].strip().lower())
    return PROFILES[key] if key is not None else None


def install_hint(scheme: str) -> str | None:
    """What to install to reach `scheme`, route by route, or None when no profile covers it.

    A missing-driver error that names only the first candidate tells a MySQL user to
    install a reader when they were trying to upsert. This names every route and what each
    one buys, so the user picks the package for the job they are doing.

    Args:
        scheme: A connection-URI scheme.

    Returns:
        A sentence naming each route's ``pip install`` command, or None.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.vendors import install_hint
            >>> print(install_hint("oracle"))
            For Oracle: pip install connectorx (reads); pip install oracledb (reads, writes).
    """
    profile = profile_for(scheme)
    if profile is None:
        return None
    parts = [
        f"pip install {route.package} ({_capability(route)})"
        for route in profile.routes
        if route.reads or route.write_modes
    ]
    return f"For {profile.name}: {'; '.join(parts)}."


def _capability(route: Route) -> str:
    """``"reads"``, ``"writes"`` or ``"reads, writes"`` for a route."""
    return ", ".join(
        word for word, has in (("reads", route.reads), ("writes", route.write_modes)) if has
    )


def capability_matrix() -> list[dict[str, str]]:
    """The published capability matrix: one row per vendor and route.

    The upsert spelling is not restated here: it is read from the statement builder that
    produces the SQL (`dbapi._statements.upsert_style`), so the matrix reports what the
    sink would actually send.

    Returns:
        Rows with ``vendor``, ``route``, ``package``, ``read``, ``write_modes``,
        ``upsert`` and ``status`` keys, every value a string.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.vendors import capability_matrix
            >>> row = capability_matrix()[0]
            >>> row["vendor"], row["route"], row["status"]
            ('PostgreSQL', 'adbc', 'untested')
    """
    from batcher.io.formats.sql.dbapi._statements import upsert_style

    rows: list[dict[str, str]] = []
    for profile in PROFILES.values():
        for route in profile.routes:
            upsert = ""
            if "upsert" in route.write_modes:
                upsert = upsert_style(profile.schemes[0]) or "none"
            rows.append(
                {
                    "vendor": profile.name,
                    "route": route.backend,
                    "package": route.package,
                    "read": "yes" if route.reads else "no",
                    "write_modes": ", ".join(route.write_modes) or "read-only",
                    "upsert": upsert or "n/a",
                    "status": profile.status,
                }
            )
    return rows
