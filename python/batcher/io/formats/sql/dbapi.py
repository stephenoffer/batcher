"""DB-API 2.0 (PEP 249) source — the universal fallback for any Python driver.

Every other connector in this package is Arrow-native: the server hands back Arrow and
nothing is ever materialized as Python objects. That is how it should be, and ADBC or
ConnectorX is the right answer whenever one of them covers your database.

They do not cover every database. PEP 249 is the one interface essentially every Python
database driver implements — `psycopg`, `pymysql`, `cx_Oracle`, `pyodbc`, `sqlite3`,
`ibm_db_dbi`, `pyhive`, `teradatasql`, and the long tail of vendor drivers. This source
takes any of them and turns it into a Batcher relation, so "my warehouse has a Python
driver" is sufficient to read it. The cost is real and stated plainly below.

**Rows are converted at batch granularity, never one at a time.** The engine's contract
is that Python never touches a tuple in a hot loop. A DB-API cursor is inherently
row-shaped, so the boundary is drawn at `fetchmany(batch_size)`: the driver returns a
block of rows, that block is transposed column-wise and handed to Arrow in one call, and
from there everything is columnar. This is the same concession the Avro reader makes —
row-wise decode confined to one batch-sized step at the IO edge — and it is why
`batch_size` is the parameter that matters most for throughput here.

**This is the slow path, on purpose.** A DB-API read pays Python-object materialization
for every value; ADBC does not. Expect it to be several times slower than
`bt.read.sql(query, uri=...)` against the same database, and prefer this only when no
Arrow-native driver exists. `schema()` still costs nothing — it uses the same zero-row
``WHERE 1 = 0`` probe every connector here uses, so planning never runs your query.

Distribution: a DB-API connection is not shippable and PEP 249 defines no way to
partition a *result set*, so by default this source yields a single split and reads on
one worker. Set `partition_on` with bounds to fan the read out instead — that issues N
independent range queries, one per worker, which is the only parallelism this protocol
allows. See `batcher.io.formats.sql.partition`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.credentials import resolve_secret
from batcher.io.formats.base import SOURCES
from batcher.io.formats.sql._common import (
    connection_fingerprint,
    probe_is_typed,
    push_down,
    schema_probe,
)
from batcher.io.formats.sql.partition import range_predicates

if TYPE_CHECKING:
    from batcher.io.splits import Split

__all__ = ["DBAPISource"]

#: Rows pulled from the cursor per `fetchmany` call, and per Arrow batch produced.
#: Matches `bc_arrow::Morsel`'s 16,384 so a batch that crosses FFI needs no rechunking.
DEFAULT_BATCH_SIZE = 16_384


def _import_driver(module_name: str) -> Any:
    """Import a user-named DB-API driver, or raise naming the driver itself.

    Unlike every other connector here the driver is not a Batcher extra — the user
    chose it — so the install hint has to name their package, not one of ours.
    """
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise BackendError(
            f"DB-API driver {module_name!r} is not installed; install it with "
            f"pip install {module_name}"
        ) from exc


def _connect(module_name: str, connect_kwargs: dict[str, Any]) -> Any:
    """Import the driver and open a fresh connection (rebuilt per worker).

    Credentials arrive as ``env:``/``file:`` references and are resolved *here* — on the
    machine that opens the connection — so the pickled split carries only the reference.
    """
    module = _import_driver(module_name)
    if not hasattr(module, "connect"):
        raise BackendError(
            f"{module_name!r} is not a DB-API 2.0 driver: it has no module-level "
            "connect(). See PEP 249."
        )
    resolved = {
        k: (resolve_secret(v, what=f"{module_name} {k}") if isinstance(v, str) else v)
        for k, v in connect_kwargs.items()
    }
    return module.connect(**resolved)


def _arrow_type(module: Any, type_code: Any) -> pa.DataType | None:
    """Map a `cursor.description` type code to an Arrow type via PEP 249's type objects.

    PEP 249 requires a driver to expose `STRING`, `BINARY`, `NUMBER`, and `DATETIME`
    singletons that compare equal to the type codes it reports. That comparison is the
    only *portable* type information a DB-API driver offers, and it is coarse: `NUMBER`
    covers integers, floats, and decimals alike, so it cannot be resolved to one Arrow
    type here.

    Returning None is therefore the common, expected outcome, and it is not a failure —
    it tells `DBAPISource.schema` to fall back to inferring from real data rather than
    to guess. Guessing `int64` for a `NUMBER` column that holds prices would corrupt
    every value silently, which is exactly the class of bug `probe_is_typed` exists to
    prevent elsewhere in this package.

    Args:
        module: The driver module, which carries the PEP 249 type singletons.
        type_code: The second element of a `cursor.description` entry.

    Returns:
        The Arrow type, or None when it cannot be determined portably.
    """
    if type_code is None:
        return None
    try:
        if type_code == getattr(module, "STRING", object()):
            return pa.string()
        if type_code == getattr(module, "BINARY", object()):
            return pa.binary()
        if type_code == getattr(module, "DATETIME", object()):
            return pa.timestamp("us")
    except Exception:  # pragma: no cover - a driver whose type codes reject ==
        return None
    return None


def _rows_to_batch(rows: list[tuple], names: list[str], schema: pa.Schema | None) -> pa.RecordBatch:
    """Transpose a block of DB-API rows into one Arrow `RecordBatch`.

    `zip(*rows)` is the whole row-to-column step, and it runs once per *batch*, not once
    per row — this is the single place row-shaped data becomes columnar.

    When `schema` is known the batch is built against it so every batch of a multi-batch
    read has identical types; without it Arrow infers, which is only safe for the first
    batch (the caller then reuses that inferred schema for the rest).
    """
    if not rows:
        if schema is not None:
            return pa.RecordBatch.from_pylist([], schema=schema)
        return pa.RecordBatch.from_pylist([])
    columns = [list(col) for col in zip(*rows, strict=False)]
    if schema is not None:
        arrays = [pa.array(col, type=schema.field(i).type) for i, col in enumerate(columns)]
        return pa.RecordBatch.from_arrays(arrays, schema=schema)
    return pa.RecordBatch.from_arrays([pa.array(col) for col in columns], names=names)


def _reconcile(
    running: pa.Schema | None, batch: pa.RecordBatch
) -> tuple[pa.Schema, pa.RecordBatch]:
    """Fit `batch` to the relation's running schema, widening only where that is lossless.

    A DB-API driver types nothing: the relation's types come from whatever Python objects
    the first rows happened to contain. Batch 2 can therefore disagree with batch 1, and
    the two ways of handling that are both traps.

    Forcing every batch into batch 1's types — the obvious approach, and what this did
    first — **silently corrupts data**: a column whose first batch held ``1`` is typed
    `int64`, and a ``2.5`` in batch 2 is written as ``2``. No error, no warning, a wrong
    number in the output. Arrow's `safe=True` does not catch it because the value arrives
    as a Python float, never as an Arrow cast.

    The mirror failure is a column that is NULL in batch 1: the relation pins to
    `pa.null()` and the first real value dies with ``Invalid null value``. Nulls-first is
    an ordinary shape — any nullable column whose NULLs sort to the front — so that made
    perfectly good relations unreadable.

    So: promote `null` to a real type freely (nothing is lost, and the batches already
    emitted held only nulls), and refuse anything else. A genuine int-to-float widening
    cannot be applied retroactively to batches already handed downstream, and guessing is
    what produced the truncation. `schema_override=` states the types up front and skips
    this path entirely.

    Args:
        running: The relation's schema so far, or None before the first batch.
        batch: The freshly-inferred batch.

    Returns:
        The updated running schema and the batch cast to it.

    Raises:
        BackendError: If the batch's types conflict with the running schema in a way that
            cannot be resolved without either losing data or retyping what was already
            emitted.
    """
    if running is None:
        return batch.schema, batch
    if batch.schema == running:
        return running, batch

    fields: list[pa.Field] = []
    arrays: list[pa.Array] = []
    for index, field_ in enumerate(running):
        incoming = batch.column(index)
        if pa.types.is_null(field_.type) and not pa.types.is_null(incoming.type):
            # Safe promotion: every value emitted for this column so far was NULL, so
            # adopting the incoming type retypes nothing that has already been handed out.
            fields.append(batch.schema.field(index))
            arrays.append(incoming)
            continue
        try:
            # `cast` is safe by default, so it raises rather than truncating. That is the
            # whole guard: int64 under a double relation converts exactly and is allowed,
            # while a double under an int64 relation raises instead of dropping the
            # fraction. Arrow decides what is lossless; this code only decides what to do
            # about it.
            arrays.append(incoming.cast(field_.type))
        except pa.ArrowInvalid as exc:
            raise BackendError(
                f"column {field_.name!r} changed type mid-read: batches so far were "
                f"{field_.type}, this batch is {incoming.type}, and converting would lose "
                "data. A DB-API driver reports no types, so the relation's types are "
                "inferred from the first rows and cannot be widened once earlier batches "
                "have been emitted. Pass schema_override= with the intended Arrow schema, "
                "or CAST the column in your query."
            ) from exc
        fields.append(field_)
    unified = pa.schema(fields)
    return unified, pa.RecordBatch.from_arrays(arrays, schema=unified)


@dataclass(frozen=True, slots=True)
class _DBAPISplit:
    """One query executed over a freshly-opened DB-API connection."""

    module_name: str
    connect_kwargs: dict[str, Any] = field(repr=False)
    sql: str
    batch_size: int = DEFAULT_BATCH_SIZE
    declared_schema: pa.Schema | None = None
    #: A caller-owned connection to borrow instead of opening one. Never closed here.
    borrowed: Any = field(default=None, repr=False, compare=False)

    @contextmanager
    def _cursor(self) -> Iterator[Any]:
        """A cursor over either a borrowed connection or one opened just for this read.

        The distinction that matters is the teardown. A connection Batcher opened must be
        closed or it leaks; a connection the *caller* handed us must not be, because they
        still hold it and will keep using it. Closing a borrowed handle is the kind of bug
        that shows up far away — the user's next query fails on a connection Batcher shut
        while they were not looking.
        """
        conn = self.borrowed if self.borrowed is not None else _connect(
            self.module_name, self.connect_kwargs
        )
        try:
            yield conn.cursor()
        finally:
            if self.borrowed is None:
                conn.close()

    def schema(self) -> pa.Schema:
        """The relation's columns, over a single connection.

        Everything this needs — the column names, the driver's type codes, and (only when
        those do not resolve) one batch of real values — comes off **one cursor from one
        `execute`**. Asking the driver to describe the query and then opening a second
        connection to sample it would submit the same query twice, which on a warehouse
        that bills per query is a second invoice for information the first cursor was
        already holding.
        """
        with self._cursor() as cur:
            cur.execute(self.sql)
            description = cur.description or []
            names = [d[0] for d in description]
            module = _import_driver(self.module_name)
            types = [_arrow_type(module, d[1] if len(d) > 1 else None) for d in description]
            if names and all(t is not None for t in types):
                return pa.schema(list(zip(names, types, strict=True)))
            # Types were not portably determinable, so infer them from real values — and
            # from one batch only. The cursor is abandoned immediately after, which stops
            # the server streaming a result set nobody is going to read.
            rows = cur.fetchmany(self.batch_size)
            if rows:
                return _rows_to_batch(list(rows), names, self.declared_schema).schema
            # No rows anywhere: the driver told us the column *names* but nothing can
            # tell us their types. Typing them `null` says exactly that — dropping the
            # columns instead would hand the planner an empty relation for a table that
            # has a real shape, and guessing `string` would be a lie the first non-empty
            # read exposes. `schema_override=` states the types when this matters.
            return pa.schema([(name, pa.null()) for name in names])

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return list(self.iter_batches(projection))

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        with self._cursor() as cur:
            cur.execute(self.sql)
            names = [d[0] for d in (cur.description or [])]
            running = self.declared_schema
            for rows in iter(lambda: cur.fetchmany(self.batch_size), []):
                if self.declared_schema is not None:
                    # The user stated the types; they are authoritative and every batch is
                    # built against them.
                    batch = _rows_to_batch(list(rows), names, self.declared_schema)
                else:
                    batch = _rows_to_batch(list(rows), names, None)
                    running, batch = _reconcile(running, batch)
                yield batch.select(projection) if projection is not None else batch

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"dbapi:{self.module_name}:{self.sql}"


@SOURCES.register("dbapi")
@dataclass(frozen=True, slots=True)
class DBAPISource:
    """A relation read through any PEP 249 (DB-API 2.0) driver.

    Args:
        module: The importable driver module name, e.g. ``"psycopg"``, ``"pymysql"``,
            ``"sqlite3"``. It must expose a module-level ``connect()``. Omit when
            passing `connection`.
        connect_kwargs: Keyword arguments passed to the driver's ``connect()``.
            String values may be ``env:``/``file:`` references, resolved on the
            worker at connect time so no secret is pickled.
        connection: An already-open PEP 249 connection (or a SQLAlchemy
            ``Connection``/``Engine``\'s underlying DBAPI connection), the way
            ``pandas.read_sql(query, con)`` accepts one. Single-node only — see
            `splits` — and **never closed by Batcher**, since the caller owns it.
        query: The SQL to run. Mutually exclusive with `table`.
        table: A table to read in full (``SELECT * FROM table``).
        batch_size: Rows per ``fetchmany`` call and per Arrow batch produced. This is
            the main throughput knob on this path.
        schema_override: An explicit Arrow schema, bypassing type inference. Supply
            this when a driver reports types too coarsely to resolve (PEP 249's
            ``NUMBER`` covers int, float, and decimal alike).
        partition_on: A numeric, indexed column to split the read across
            `num_partitions` parallel queries. See
            `batcher.io.formats.sql.partition` — the bounds are cut points, not
            filters, so rows outside them are still read.
        lower_bound: Approximate minimum of `partition_on`.
        upper_bound: Approximate maximum of `partition_on`.
        num_partitions: How many parallel queries to issue.

    Raises:
        BackendError: If the driver is not installed or is not a DB-API 2.0 module,
            if neither `query` nor `table` is given, or if `partition_on` is set
            without both bounds.
    """

    # Kyber's pushed predicate becomes a SQL WHERE, so the server filters before any
    # row is materialized as a Python object — which is the expensive part here.
    supports_predicate: ClassVar[bool] = True

    module: str = ""
    connect_kwargs: dict[str, Any] = field(default_factory=dict, repr=False)
    connection: Any = field(default=None, repr=False, compare=False)
    query: str | None = None
    table: str | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    schema_override: pa.Schema | None = None
    partition_on: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    num_partitions: int = 1

    def __post_init__(self) -> None:
        if self.query is None and self.table is None:
            raise BackendError("DBAPISource requires either query= or table=")
        if not self.module and self.connection is None:
            raise BackendError(
                "DBAPISource requires either module= (with connect_kwargs=) or an "
                "already-open connection="
            )
        if self.connection is not None and self.partition_on is not None:
            raise BackendError(
                "connection= cannot be combined with partition_on=: range partitioning "
                "runs one query per worker, and a live connection belongs to this process "
                "alone. Pass module= and connect_kwargs= so each worker can open its own."
            )
        if self.batch_size < 1:
            raise BackendError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.partition_on is not None and (self.lower_bound is None or self.upper_bound is None):
            raise BackendError(
                f"partition_on={self.partition_on!r} requires lower_bound= and upper_bound=. "
                "They are cut points rather than filters, so approximate values are fine "
                "and rows outside them are still read."
            )

    def _split(self, sql: str) -> _DBAPISplit:
        return _DBAPISplit(
            self.module,
            self.connect_kwargs,
            sql,
            self.batch_size,
            self.schema_override,
            self.connection,
        )

    def schema(self) -> pa.Schema:
        """The relation's columns, from a zero-row probe rather than the whole query."""
        if self.schema_override is not None:
            return self.schema_override
        probed = self._split(schema_probe(self.query, table=self.table)).schema()
        if probe_is_typed(probed):
            return probed
        return self._split(push_down(self.query, table=self.table)).schema()

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        return list(self.iter_batches(projection, predicate))

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        sql = push_down(self.query, predicate, projection, table=self.table)
        yield from self._split(sql).iter_batches(projection)

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        # The connection is part of the key: the same query against production and
        # staging is two different relations, and conflating their learned statistics
        # gives the optimizer one table's cardinalities for the other's data.
        #
        # A *borrowed* connection has no stable identity across runs — it is a live
        # object, and its address says nothing about which database it points at. So the
        # key falls back to the driver that produced it, which cannot distinguish two
        # databases reached through the same driver. That is a real limitation of passing
        # a connection rather than a description of one, and `module=`/`connect_kwargs=`
        # is the spelling that gets a precise key.
        if self.connection is not None:
            driver = type(self.connection).__module__.split(".")[0]
            return f"dbapi:{driver}:conn:{self.query or self.table}"
        return (
            f"dbapi:{self.module}:{connection_fingerprint(self.connect_kwargs)}:"
            f"{self.query or self.table}"
        )

    def splits(
        self,
        target_size: int | None = None,  # noqa: ARG002 (protocol signature)
        predicate: dict | None = None,
        projection: list[str] | None = None,
    ) -> list[Split]:
        """One split per key range, or a single split when not partitioned.

        PEP 249 defines no way to partition a *result set*, so parallelism here comes
        from issuing several independent queries instead — one per slice of
        `partition_on`. Each split opens its own connection on its own worker, which is
        the only form of parallel read this protocol allows.

        Without `partition_on` there is exactly one split and the read happens on one
        worker. Either way the pushdown is folded into the SQL each split carries, so
        the *server* filters and the discarded rows never become Python objects at all.
        """
        fragments: list[str | None] = [None]
        if self.partition_on is not None:
            assert self.lower_bound is not None and self.upper_bound is not None  # __post_init__
            fragments = range_predicates(
                self.partition_on, self.lower_bound, self.upper_bound, self.num_partitions
            )
        return [
            self._split(
                push_down(self.query, predicate, projection, table=self.table, extra_where=fragment)
            )
            for fragment in fragments
        ]
