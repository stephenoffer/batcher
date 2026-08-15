"""Template-Method base for a query-backed source the server returns as one result.

`io.base.FileSource` is the spine every *file* format subclasses. Nothing played that role
for the query backends, so each of them restated the same five methods, and they had to:
the shape is forced by the two protocols they satisfy at once. A `Source` is asked for a
schema, a read, a stream and a split list; a `Split` is what a distributed worker rebuilds
its reader from. When the server vends no externally-addressable result handles — ClickHouse
streams one result, ODBC hands back one cursor, ConnectorX does its own range-partitioning
*inside* the driver — the source is exactly one split, and all five methods reduce to "build
the split, ask it".

Restating that five times was not merely repetitive, it was where the backends diverged on
the thing that costs the most. **The pushdown has to be folded into the SQL the split
carries**, because a worker rebuilds its reader from the split alone: a predicate held on the
source and not in the split's own query never reaches the server, so the worker issues an
unfiltered read, the whole relation crosses the wire, and the engine's `Filter` discards it
afterwards — correct, and arbitrarily expensive. Each backend had to remember that
independently, and the comments in each one say so in slightly different words.

What a subclass still owns is everything genuinely per-backend: how to dial the server, how
to turn one SQL string into a picklable split, and its `identity()` fingerprint. Nothing here
touches a row.

Subclasses are `@dataclass(frozen=True, slots=True)`, so this base carries `__slots__ = ()`
and declares no fields of its own; `query` is an annotation the subclass's dataclass supplies.

**One trap that costs an hour if you meet it cold: a subclass here cannot call zero-argument
`super()`.** `@dataclass(slots=True)` cannot add slots to a class that already exists, so it
builds a *replacement* class and returns that. The `__class__` cell the compiler wires into
`super()` still points at the original, discarded one, and the call fails at runtime with
``TypeError: super(type, obj): obj must be an instance or subtype of type`` — not at import,
and only on the line that runs it. Name the base explicitly
(``SingleResultQuerySource.splits(self, ...)``) if a subclass genuinely needs to extend rather
than replace a method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, ClassVar

import pyarrow as pa

from batcher.io.formats.sql._common import (
    count_query,
    probe_is_typed,
    push_down,
    schema_probe,
)

if TYPE_CHECKING:
    from batcher.io.splits import Split

__all__ = ["SingleResultQuerySource"]


class SingleResultQuerySource(ABC):
    """A SQL relation whose server returns one result, so the source is one split.

    Subclass this and implement `_split_for` and `identity`. The five protocol methods
    (`schema`, `read`, `iter_batches`, `row_count`, `splits`) come from here, and each of
    them routes through `_split`, so the pushed predicate and projection are in the split's
    own SQL on every path — the single-node one and the distributed one alike.
    """

    __slots__ = ()

    #: Kyber's pushed predicate becomes an appended SQL `WHERE`, so the server filters before
    #: returning Arrow. True for every backend here; a subclass that cannot push overrides it.
    supports_predicate: ClassVar[bool] = True

    #: Whether a `LIMIT n` may be appended to this backend's SQL. **Off by default, and the
    #: default is the point.** A missing cap costs the rows the server would have skipped; a
    #: cap the server cannot parse turns a working query into a syntax error, and this base
    #: is shared by ODBC, whose DSN names a driver rather than a dialect. A subclass opts in
    #: only where it knows the dialect (`uri.supports_limit_clause`).
    supports_limit: ClassVar[bool] = False

    #: The connection-URI scheme naming this backend's SQL dialect, or ``""`` when it
    #: cannot be known. It decides how a column name is delimited — `uri.quote_identifier`
    #: — and an empty value means "emit identifiers verbatim", the behavior every backend
    #: had before. ODBC leaves it empty on purpose: a DSN names a driver, not a dialect.
    sql_dialect: ClassVar[str] = ""

    @property
    def supports_ordering(self) -> bool:
        """Whether a top-N may be pushed: the dialect must accept an explicit `NULLS`.

        Without that clause the server's "first n" and the engine's differ wherever they
        place a null, so the read returns the *wrong rows* rather than extra ones. MySQL
        and SQL Server have no such clause, so a top-N does not push to them at all.
        """
        from batcher.io.formats.sql.uri import supports_nulls_ordering

        return bool(self.sql_dialect) and supports_nulls_ordering(self.sql_dialect)

    @property
    def _quote(self):
        """The identifier delimiter for this backend's dialect."""
        from batcher.io.formats.sql.uri import quote_identifier

        dialect = self.sql_dialect
        return (lambda name: quote_identifier(name, dialect)) if dialect else (lambda n: n)

    #: The one logical query, supplied as a field by the concrete dataclass.
    query: str

    @abstractmethod
    def _split_for(self, sql: str) -> Split:
        """Build the picklable split that reads `sql` from this backend.

        The one thing every backend does differently: which connection material rides on the
        split, and which driver call it makes. `sql` already carries any pushdown, so an
        implementation must use it verbatim rather than reaching for `self.query`.

        Args:
            sql: The SQL the split should execute.

        Returns:
            A split satisfying the `Split` protocol.
        """

    def _probe_split_for(self, sql: str) -> Split:
        """Build the split used for the zero-row schema probe.

        The same as `_split_for` unless the backend's split carries *driver-internal*
        parallelism, in which case fanning a `WHERE 1 = 0` probe into several sub-queries
        costs several round trips to learn one column list. ConnectorX overrides this to
        probe unpartitioned.

        Args:
            sql: The probe SQL.

        Returns:
            A split satisfying the `Split` protocol.
        """
        return self._split_for(sql)

    def _split(
        self,
        predicate: dict | None = None,
        projection: list[str] | None = None,
        limit: int | None = None,
        order_by: tuple[tuple[str, bool, bool], ...] | None = None,
    ) -> Split:
        """The split for a real read, with the pushdown already folded into its SQL.

        Args:
            predicate: Kyber's pushed predicate, or `None`.
            projection: The columns the plan needs, or `None` for all of them.
            limit: The most rows the plan needs, or `None`. Folded into the split's own
                SQL for the same reason the predicate is — a worker rebuilds its reader
                from the split alone, so a cap held anywhere else never reaches the server.
                Ignored unless the subclass declares `supports_limit`.
            order_by: The ordering `limit` is taken in, when the cap is a top-N rather
                than a prefix.

        Returns:
            One split whose own query carries `predicate`, `projection`, `limit` and
            `order_by`.
        """
        # An *ordered* cap is only sound if the ordering goes with it. A backend that can
        # take `LIMIT` but cannot spell `NULLS FIRST|LAST` must therefore drop the cap as
        # well, or it would return its own idea of the first n — the wrong rows, silently,
        # which is the one outcome pushdown may never produce.
        ordered = order_by if self.supports_ordering else None
        capped = limit if self.supports_limit else None
        if order_by and ordered is None:
            capped = None
        return self._split_for(
            push_down(
                self.query,
                predicate,
                projection,
                limit=capped,
                order_by=ordered,
                quote=self._quote,
            )
        )

    def schema(self) -> pa.Schema:
        """The relation's columns, from a zero-row probe rather than the whole query.

        See `schema_probe`. The probe is best-effort: a driver that answers a `WHERE 1 = 0`
        with an all-null or empty schema teaches nothing, and `probe_is_typed` catches that,
        so this falls back to asking the real query. Worst case is the full read every one of
        these backends used to do unconditionally; usually it is a metadata round trip.

        Returns:
            The Arrow schema the relation reads as.
        """
        probed = self._probe_split_for(schema_probe(self.query)).schema()
        return probed if probe_is_typed(probed) else self._split().schema()

    def read(
        self,
        projection: list[str] | None = None,
        predicate: dict | None = None,
        limit: int | None = None,
        ordering: tuple[tuple[str, bool, bool], ...] | None = None,
    ) -> list[pa.RecordBatch]:
        """Read the whole relation, with the pushdown applied by the server.

        Args:
            projection: The columns to return, or `None` for all of them.
            predicate: Kyber's pushed predicate, or `None`.
            limit: The most rows the plan needs, or `None`.
            ordering: The ordering `limit` is taken in, or `None`.

        Returns:
            The result as Arrow record batches.
        """
        return self._split(predicate, projection, limit, ordering).read(projection)

    def iter_batches(
        self,
        projection: list[str] | None = None,
        predicate: dict | None = None,
        limit: int | None = None,
        ordering: tuple[tuple[str, bool, bool], ...] | None = None,
    ) -> Iterator[pa.RecordBatch]:
        """Stream the relation, with the pushdown applied by the server.

        Args:
            projection: The columns to return, or `None` for all of them.
            predicate: Kyber's pushed predicate, or `None`.
            limit: The most rows the plan needs, or `None`.
            ordering: The ordering `limit` is taken in, or `None`.

        Yields:
            Arrow record batches, in the order the driver produces them.
        """
        yield from self._split(predicate, projection, limit, ordering).iter_batches(projection)

    #: Whether `exact_row_count` may ask the server for a ``COUNT(*)``. True for every
    #: backend here — the query is ANSI and needs no dialect gate — but a subclass whose
    #: `query` is not safely re-runnable would turn it off.
    supports_count: ClassVar[bool] = True

    def exact_row_count(self) -> int | None:
        """The relation's row count, from one ``COUNT(*)`` round trip.

        Deliberately *not* `row_count`, which stays `None`. The two are asked at different
        moments and only one of them is worth a query: `row_count` is consulted while
        *planning*, where Kyber wants a free estimate and has a better one in the learned
        statistics, so charging the user a round trip per plan would be a poor trade. This
        is consulted only when the caller asked for the count itself (`ds.count()`), where
        the alternative is not an estimate but reading the whole relation.

        Returns:
            The exact row count, or None if the server returned nothing recognizable.
        """
        # Read the one column positionally rather than by name: the query aliases it, but
        # an unquoted alias is case-folded by the server (Oracle and Snowflake upper-case
        # it), so asking for it back under the name written here would raise on the very
        # backends that ran the query.
        for batch in self._split_for(count_query(self.query)).read():
            if batch.num_rows:
                value = batch.column(0)[0].as_py()
                return None if value is None else int(value)
        return None

    def row_count(self) -> int | None:
        """The relation's row count, always `None` here.

        Returns:
            `None`: counting means a second `SELECT COUNT(*)` round trip against a server
            whose cost the engine cannot see, and Kyber estimates better from the learned
            statistics keyed on `identity()` than from a count it charged the user for.
        """
        return None

    def splits(
        self,
        target_size: int | None = None,  # noqa: ARG002 (protocol signature)
        predicate: dict | None = None,
        projection: list[str] | None = None,
        limit: int | None = None,
    ) -> list[Split]:
        """One split, whose SQL already carries the pushdown.

        `target_size` is ignored: the server decides how it returns one result, so there is
        no byte budget the control plane can spend here. Backends that *do* vend addressable
        result handles (BigQuery streams, Snowflake result chunks, ADBC partitions) return
        several splits and do not subclass this.

        Args:
            target_size: Ignored, present for the `Source` protocol.
            predicate: Kyber's pushed predicate, or `None`.
            projection: The columns the plan needs, or `None` for all of them.
            limit: The most rows the plan needs, or `None`.

        Returns:
            A one-element list holding the filtered, projected, capped split.
        """
        return [self._split(predicate, projection, limit)]

    @abstractmethod
    def identity(self) -> str:
        """The learned-statistics key: the connection *and* the query, never the query alone.

        Keyed on the query alone, ``SELECT * FROM orders`` against production and against
        staging is one relation, so Kyber plans the thousand-row table with the billion-row
        table's cardinalities. Nothing errors — it is simply the wrong plan, from good code.
        Fingerprint the connection with `connection_fingerprint`, which excludes credentials
        so a password rotation does not orphan the statistics.

        Returns:
            A stable, non-secret string naming this relation.
        """
