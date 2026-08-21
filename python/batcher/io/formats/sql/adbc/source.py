"""ADBC / FlightSQL source — Arrow-native database connectivity.

ADBC (Arrow Database Connectivity) is the zero-copy, Arrow-first driver layer:
results arrive as Arrow streams without a row-by-row Python materialization.
This module is the generic entry point for any ADBC driver (SQLite, PostgreSQL,
Snowflake-ADBC, DuckDB, …) and is the *only* backend with true, shippable
distributed partitions — via FlightSQL's ``adbc_execute_partitions``.

Single-submission contract:

- A FlightSQL driver returns ``(partition_descriptors, schema)`` from a single
  ``adbc_execute_partitions(sql)`` call. We build one `_ADBCPartitionSplit` per
  opaque descriptor; each split carries the descriptor bytes plus the driver +
  ``db_kwargs`` needed to rebuild a *fresh* connection on the worker, then reads
  its slice by streaming ``adbc_read_partition(desc).fetch_record_batch()``.
- A non-partitioning driver yields a single `_ADBCQuerySplit` that streams the
  whole query once with ``fetch_record_batch``.

Credentials live only in ``db_kwargs``/``conn_kwargs`` carried on the split, which
are excluded from every ``repr`` so a worker traceback cannot render them. They are
still *pickled* with the split, so prefer an ``env:``/``file:`` reference (resolved on
the worker by `_connect`) over a literal. Connections are never pickled.

Connections may be given either natively (``driver=`` plus ``db_kwargs=``) or as a
standard ``uri=`` — see `batcher.io.formats.sql.uri`, which maps a SQLAlchemy-style
URI onto the right driver.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa

from batcher.io.credentials import resolve_secret
from batcher.io.formats.base import SOURCES
from batcher.io.formats.sql._common import (
    connection_fingerprint,
    identifier_quoter,
    probe_is_typed,
    push_down,
    pushed_sql,
    require_module,
    schema_probe,
)
from batcher.io.formats.sql.partition import range_predicates

if TYPE_CHECKING:
    from batcher.io.splits import Split

__all__ = ["ADBCSource"]

#: `adbc_ingest` dispositions that discard whatever the table already held. Safe for a
#: single writer; ruinous when every shard of a distributed write applies one.
_DESTRUCTIVE_MODES = frozenset({"replace", "create"})

_EXTRA = "sql"
_MODULE = "adbc_driver_manager.dbapi"


def _connect(driver: str, db_kwargs: dict[str, Any], conn_kwargs: dict[str, Any] | None) -> Any:
    """Open a fresh DBAPI connection for `driver` (rebuilt per worker).

    `db_kwargs` carries the DSN/URI and any credentials, so each string value is resolved
    here — on the worker — and the split ships only the reference."""
    dbapi = require_module(_MODULE, extra=_EXTRA)
    db_kwargs = {
        k: (resolve_secret(v, what=f"ADBC {k}") if isinstance(v, str) else v)
        for k, v in db_kwargs.items()
    }
    return dbapi.connect(driver=driver, db_kwargs=db_kwargs, **(conn_kwargs or {}))


@dataclass(frozen=True, slots=True)
class _ADBCQuerySplit:
    """A single streaming read of one logical query over a fresh connection."""

    driver: str
    db_kwargs: dict[str, Any] = field(repr=False)
    conn_kwargs: dict[str, Any] | None = field(repr=False)
    sql: str

    def _table(self) -> pa.Table:
        conn = _connect(self.driver, self.db_kwargs, self.conn_kwargs)
        try:
            cur = conn.cursor()
            cur.execute(self.sql)
            return cur.fetch_arrow_table()
        finally:
            conn.close()

    def schema(self) -> pa.Schema:
        """The query's column types, taken off the stream without draining it.

        `fetch_arrow_table` would pull every row into memory to read a schema the reader
        already carries in its header. That is normally hidden because `ADBCSource.schema`
        asks a ``WHERE 1 = 0`` probe, which returns nothing — but the fallback for a driver
        whose probe comes back untyped runs the *real* query, and there materializing a
        whole relation to learn its column names is the difference between an expensive
        schema lookup and an OOM.
        """
        conn = _connect(self.driver, self.db_kwargs, self.conn_kwargs)
        try:
            cur = conn.cursor()
            cur.execute(self.sql)
            return cur.fetch_record_batch().schema
        finally:
            conn.close()

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        table = self._table()
        if projection is not None:
            table = table.select(projection)
        return table.to_batches()

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        conn = _connect(self.driver, self.db_kwargs, self.conn_kwargs)
        try:
            cur = conn.cursor()
            cur.execute(self.sql)
            reader = cur.fetch_record_batch()
            for batch in reader:
                yield batch.select(projection) if projection is not None else batch
        finally:
            conn.close()

    @contextmanager
    def catalog_session(self) -> Iterator[tuple[Any, Any]]:
        """Yield ``(run_scalar, run_rows)`` sharing **one** connection for the catalog probes.

        A `statistics()` call issues three catalog queries; opening a connection per query
        tripled the connect round trips against the database. They share one connection here
        — one connect, three cheap `execute`s, one close — each on its own short-lived cursor,
        rolling back on failure so a Postgres-style transaction abort does not poison the
        queries that follow. A catalog answer is one row, so `execute`/`fetch` is used rather
        than `fetch_arrow_table`, whose Arrow materialization would be pure overhead here.
        """
        conn = _connect(self.driver, self.db_kwargs, self.conn_kwargs)

        def _run(sql: str, *, many: bool) -> Any:
            cur = conn.cursor()
            try:
                cur.execute(sql)
                return list(cur.fetchall()) if many else (cur.fetchone() or (None,))[0]
            except Exception:
                with suppress(Exception):
                    conn.rollback()
                raise
            finally:
                cur.close()

        try:
            yield (lambda sql: _run(sql, many=False), lambda sql: _run(sql, many=True))
        finally:
            conn.close()

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"adbc:{self.driver}:{self.sql}"


@dataclass(frozen=True, slots=True)
class _ADBCPartitionSplit:
    """One FlightSQL partition descriptor, read via a fresh worker connection."""

    driver: str
    db_kwargs: dict[str, Any] = field(repr=False)
    conn_kwargs: dict[str, Any] | None = field(repr=False)
    descriptor: bytes
    index: int

    def _table(self) -> pa.Table:
        conn = _connect(self.driver, self.db_kwargs, self.conn_kwargs)
        try:
            cur = conn.cursor()
            reader = cur.adbc_read_partition(self.descriptor)
            return reader.fetch_arrow_table()
        finally:
            conn.close()

    def schema(self) -> pa.Schema:
        """This partition's column types, without downloading the partition."""
        conn = _connect(self.driver, self.db_kwargs, self.conn_kwargs)
        try:
            cur = conn.cursor()
            reader = cur.adbc_read_partition(self.descriptor)
            return reader.fetch_record_batch().schema
        finally:
            conn.close()

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        table = self._table()
        if projection is not None:
            table = table.select(projection)
        return table.to_batches()

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Stream this partition, rather than materializing it and then chunking it.

        This used to be ``yield from self.read(...)``, which calls `fetch_arrow_table` —
        so the "streaming" path pulled the entire partition into memory before yielding
        its first batch. On a FlightSQL partition sized for a whole worker that is the
        difference between bounded and unbounded memory, and it silently defeated every
        caller that chose `iter_batches` precisely to avoid materializing.
        """
        conn = _connect(self.driver, self.db_kwargs, self.conn_kwargs)
        try:
            cur = conn.cursor()
            reader = cur.adbc_read_partition(self.descriptor)
            for batch in reader.fetch_record_batch():
                yield batch.select(projection) if projection is not None else batch
        finally:
            conn.close()

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"adbc-part:{self.driver}:{self.index}"


@SOURCES.register("adbc")
@dataclass(frozen=True, slots=True)
class ADBCSource:
    """A relation read through an ADBC driver, optionally FlightSQL-partitioned.

    Args:
        query: The SQL to run. Mutually exclusive with `table`.
        table: A table name to read in full (``SELECT * FROM table``).
        driver: The ADBC driver to load (e.g. ``"adbc_driver_sqlite"`` or a
            FlightSQL driver path).
        db_kwargs: Driver/database connection kwargs (DSN, uri, credentials).
            Carried on splits verbatim and excluded from `repr`.
        conn_kwargs: Extra ``connect()`` kwargs (autocommit, etc).
        partition: If True, attempt FlightSQL ``adbc_execute_partitions`` to
            produce one split per server-side partition (one query submission).
        uri: A standard connection URI (``postgresql://host:5432/db``) supplying
            `driver` and `db_kwargs` in place of naming them individually. An
            explicit `driver`/`db_kwargs` still wins over anything derived here.
        password: The password, as a literal or an ``env:``/``file:`` reference
            resolved on the worker. Preferred over embedding one in `uri`.
        partition_on: A numeric, indexed column to split the read across
            `num_partitions` parallel queries, for drivers with no server-side
            partitioning. See `batcher.io.formats.sql.partition`.
        lower_bound: Approximate minimum of `partition_on` — a cut point, not a filter.
        upper_bound: Approximate maximum of `partition_on` — a cut point, not a filter.
        num_partitions: How many parallel queries to issue.

    Raises:
        BackendError: If `adbc_driver_manager` is not installed, neither `query`
            nor `table` is given, no connection is specified, or `uri` names a
            scheme with no ADBC driver.
    """

    # Predicate pushdown: Kyber's pushed predicate → an appended SQL WHERE (the
    # server filters before returning Arrow). Class var, not a dataclass field.
    supports_predicate: ClassVar[bool] = True

    driver: str | None = None
    db_kwargs: dict[str, Any] | None = field(default=None, repr=False)
    query: str | None = None
    table: str | None = None
    conn_kwargs: dict[str, Any] | None = field(default=None, repr=False)
    partition: bool = False
    uri: str | None = None
    password: str | None = field(default=None, repr=False)
    dialect: str | None = None
    #: Memoized `schema()` / `statistics()`. `init=False` keeps them out of the public
    #: constructor and `compare=False` keeps two sources describing the same relation equal
    #: whether or not either has been asked yet — `identity()` and the plan cache key on
    #: what a source *is*, never on what it has cached.
    _schema_cache: pa.Schema | None = field(default=None, init=False, repr=False, compare=False)
    _stats_cache: Any = field(default=None, init=False, repr=False, compare=False)
    partition_on: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    num_partitions: int = 1

    def __post_init__(self) -> None:
        from batcher._internal.errors import BackendError

        if self.query is None and self.table is None:
            raise BackendError("ADBCSource requires either query= or table=")
        if self.partition_on is not None and (self.lower_bound is None or self.upper_bound is None):
            raise BackendError(
                f"partition_on={self.partition_on!r} requires lower_bound= and upper_bound=. "
                "They are cut points rather than filters, so approximate values are fine "
                "and rows outside them are still read."
            )
        if self.uri is not None:
            self._apply_uri()
        if self.dialect is None:
            from batcher.io.formats.sql.uri import scheme_for_adbc_driver

            object.__setattr__(self, "dialect", scheme_for_adbc_driver(self.driver))
        if self.db_kwargs is None:
            object.__setattr__(self, "db_kwargs", {})
        if self.driver is None:
            raise BackendError(
                "ADBCSource requires either uri= (e.g. 'postgresql://host/db') or an "
                "explicit driver= and db_kwargs="
            )

    def _apply_uri(self) -> None:
        """Fill `driver`/`db_kwargs` from `uri`, keeping the password out of the URI.

        The password is placed in `db_kwargs` **as the reference the user gave** (an
        ``env:``/``file:`` string, or a literal). `_connect` resolves every string kwarg
        on the worker, so a reference stays a reference for the whole pickled journey and
        only becomes a secret inside the process that opens the connection. Assigning
        through `object.__setattr__` is how a frozen dataclass normalizes in `__post_init__`.
        """
        from batcher.io.formats.sql.uri import adbc_connection, parse_uri

        if self.dialect is None:
            # Read before `adbc_connection` replaces `uri` with the sanitized form: the
            # scheme is what names the dialect, and it decides identifier quoting and
            # whether a row cap may be pushed.
            object.__setattr__(self, "dialect", parse_uri(str(self.uri)).scheme)
        driver, merged, sanitized = adbc_connection(
            self.uri, password=self.password, driver=self.driver, db_kwargs=self.db_kwargs
        )
        object.__setattr__(self, "driver", driver)
        object.__setattr__(self, "db_kwargs", merged)
        # Keep the *sanitized* URI on the field too. `db_kwargs` is `repr=False`, but this
        # one is not — it is useful in a traceback — so leaving the user's original string
        # here would put an inline password straight back into every repr we just removed
        # it from.
        object.__setattr__(self, "uri", sanitized)

    @property
    def supports_limit(self) -> bool:
        """Whether a row cap may be appended to this backend's SQL.

        An allow-list per `uri.supports_limit_clause`: a missing cap costs the rows the
        server would have skipped, while a cap it cannot parse turns a working query into a
        syntax error. Nothing was pushed at all before this source could name its dialect,
        so ``bt.read.sql(query, uri=...).head(10)`` streamed the whole result over the wire.
        """
        from batcher.io.formats.sql.uri import supports_limit_clause

        return bool(self.dialect) and supports_limit_clause(self.dialect)

    @property
    def supports_ordering(self) -> bool:
        """Whether a top-N may be pushed: the dialect must accept an explicit ``NULLS`` clause.

        Without it the server's "first n" and the engine's differ wherever they place a
        null, so the read returns the wrong rows rather than merely extra ones.
        """
        from batcher.io.formats.sql.uri import supports_nulls_ordering

        return bool(self.dialect) and supports_nulls_ordering(self.dialect)

    @property
    def _quote(self) -> Callable[[str], str]:
        """How to delimit an identifier for this dialect; verbatim when it is unknown."""
        return identifier_quoter(self.dialect)

    def _sql(
        self,
        projection: list[str] | None = None,
        predicate: dict | None = None,
        limit: int | None = None,
        ordering: tuple[tuple[str, bool, bool], ...] | None = None,
        extra_where: str | None = None,
    ) -> str:
        """The query with every pushable part of the plan folded in (see `push_down`).

        An *ordered* cap is only sound if the ordering goes with it: a dialect that takes
        ``LIMIT`` but cannot spell ``NULLS FIRST|LAST`` must drop the cap too, or it returns
        its own idea of the first n — the wrong rows, silently.
        """
        return pushed_sql(
            self.query,
            predicate=predicate,
            projection=projection,
            limit=limit,
            ordering=ordering,
            extra_where=extra_where,
            table=self.table,
            quote=self._quote,
            supports_ordering=self.supports_ordering,
            supports_limit=self.supports_limit,
        )

    def schema(self) -> pa.Schema:
        """The relation's columns, from a zero-row probe rather than the whole query.

        See `schema_probe`: this used to execute the full query and discard every row.

        Memoized on the instance, the way `nosql.ScanSource` and `DBAPISource` memoize
        theirs. Planning asks for the schema several times per terminal op, and each ask
        opened a **connection** — against a warehouse that is a TLS handshake and an
        authentication round trip, which dwarfs the probe it was opened for.
        """
        if self._schema_cache is not None:
            return self._schema_cache
        probed = _ADBCQuerySplit(
            self.driver,
            self.db_kwargs,
            self.conn_kwargs,
            schema_probe(self.query, table=self.table),
        ).schema()
        resolved = probed if probe_is_typed(probed) else self.splits()[0].schema()
        object.__setattr__(self, "_schema_cache", resolved)
        return resolved

    def _direct_splits(
        self,
        predicate: dict | None,
        projection: list[str] | None,
        limit: int | None = None,
        ordering: tuple[tuple[str, bool, bool], ...] | None = None,
    ) -> list[Split]:
        """Splits for a read happening *here*, rather than fanned out across workers.

        Range partitioning exists to give N machines a query each. Executed on one
        machine it is pure loss: the same rows arrive, having cost N round trips and N
        server-side planner invocations instead of one. So a direct `read`/`iter_batches`
        skips it, while `splits` — the distribution path — applies it.

        Server-side FlightSQL partitioning is different and is kept here: it splits one
        *already-submitted* result set, so it costs no extra query and can only help.
        """
        sql = self._sql(projection, predicate, limit, ordering)
        if self.partition:
            parts = self._execute_partitions(sql)
            if parts is not None:
                return parts
        return [_ADBCQuerySplit(self.driver, self.db_kwargs, self.conn_kwargs, sql)]

    def read(
        self,
        projection: list[str] | None = None,
        predicate: dict | None = None,
        limit: int | None = None,
        ordering: tuple[tuple[str, bool, bool], ...] | None = None,
    ) -> list[pa.RecordBatch]:
        out: list[pa.RecordBatch] = []
        for split in self._direct_splits(predicate, projection, limit, ordering):
            out.extend(split.read(projection))
        return out

    def iter_batches(
        self,
        projection: list[str] | None = None,
        predicate: dict | None = None,
        limit: int | None = None,
        ordering: tuple[tuple[str, bool, bool], ...] | None = None,
    ) -> Iterator[pa.RecordBatch]:
        for split in self._direct_splits(predicate, projection, limit, ordering):
            yield from split.iter_batches(projection)

    def row_count(self) -> int | None:
        return None

    def statistics(self) -> Any:
        """Catalog row count, byte size, and per-column stats for a `table=` read, else None.

        A base ``table=`` has a system-catalog entry the driver's own dialect can answer
        without scanning it; a ``query=`` read is an arbitrary expression with none. The
        dialect is read from the ADBC driver name (``adbc_driver_postgresql`` -> Postgres),
        so a Postgres-over-ADBC table reaches Kyber with the same null/ndv/mcv/quantile
        facets a Parquet footer would supply. Best-effort: any catalog failure yields None.
        """
        if self.table is None or self.driver is None:
            return None
        from batcher.io.stats import dialect_for_driver, sql_statistics

        if self._stats_cache is not None:
            cached = self._stats_cache
            return dataclasses.replace(cached, exact_rows=False) if cached.exact_rows else cached
        dialect = dialect_for_driver(self.driver)
        if dialect is None:
            return None
        split = _ADBCQuerySplit(self.driver, self.db_kwargs or {}, self.conn_kwargs, "")
        try:
            with split.catalog_session() as (run_scalar, run_rows):
                probed = sql_statistics(
                    dialect, self.table, run_scalar=run_scalar, run_rows=run_rows
                )
            if probed is not None:
                object.__setattr__(self, "_stats_cache", probed)
            return probed
        except Exception:
            return None

    def identity(self) -> str:
        # The connection is part of the key: the same query against production and
        # staging is two different relations, and conflating their learned statistics
        # gives the optimizer one table's cardinalities for the other's data.
        return (
            f"adbc:{self.driver}:{connection_fingerprint(self.db_kwargs or {})}:"
            f"{self.query or self.table}"
        )

    def splits(
        self,
        target_size: int | None = None,  # noqa: ARG002 (protocol signature)
        predicate: dict | None = None,
        projection: list[str] | None = None,
    ) -> list[Split]:
        """One split per server-side partition, each running the *pushed-down* query.

        The pushdown is folded into the SQL the split carries, so the *worker's* query is the
        filtered one. A predicate left outside the split never reaches the server: the worker
        rebuilds an unfiltered read and the engine's `Filter` discards the rows after they have
        already crossed the wire.
        """
        if self.partition_on is None:
            return self._direct_splits(predicate, projection)
        if self.partition:
            # Server-side FlightSQL partitioning is strictly better when available — it
            # splits one already-submitted result set, so it costs no extra query. Range
            # partitioning is the fallback for drivers with none: N independent queries,
            # and so is only reached once `_execute_partitions` has declined.
            parts = self._execute_partitions(self._sql(projection, predicate))
            if parts is not None:
                return parts
        assert self.lower_bound is not None and self.upper_bound is not None  # __post_init__
        return [
            _ADBCQuerySplit(
                self.driver,
                self.db_kwargs,
                self.conn_kwargs,
                push_down(
                    self.query, predicate, projection, table=self.table, extra_where=fragment
                ),
            )
            for fragment in range_predicates(
                self.partition_on,
                self.lower_bound,
                self.upper_bound,
                self.num_partitions,
                quote=self._quote,
            )
        ]

    def _execute_partitions(self, sql: str) -> list[Split] | None:
        """ONE submission → opaque descriptors. None if the driver can't partition."""
        conn = _connect(self.driver, self.db_kwargs, self.conn_kwargs)
        try:
            cur = conn.cursor()
            descriptors, _schema, _rows = cur.adbc_execute_partitions(sql)
        except (AttributeError, NotImplementedError):
            return None
        finally:
            conn.close()
        return [
            _ADBCPartitionSplit(self.driver, self.db_kwargs, self.conn_kwargs, bytes(desc), i)
            for i, desc in enumerate(descriptors)
        ]
