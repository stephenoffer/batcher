"""Shared shape for NoSQL / operational-store scan sources.

These stores (MongoDB, Cassandra, DynamoDB, Redis, …) are row-based key/value or
document engines with no Arrow-native file layout. Reading them is the same
recipe every time: open a per-worker connection from serialized connection
kwargs, enumerate the store's natural parallel unit (a token range, a scan
segment, a query offset window — never a live connection), fetch each unit's rows,
and assemble Arrow at *batch* granularity (one `pa.RecordBatch` per chunk of
rows). `ScanSource` captures that recipe so each concrete connector overrides
only two things: how to enumerate partitions and how to fetch one partition.

The connection kwargs are stored verbatim and **never logged** — they carry
credentials. A `_ScanSplit` is a frozen, picklable value object that holds only
the connector class, the (never-logged) connection kwargs, and the opaque
partition locator; the worker reconstructs the connector from those and fetches
just its partition. Missing optional drivers raise `BackendError` with an
actionable ``pip install 'batcher-engine[<extra>]'`` hint.

`BulkSink` is the mirror for the write path, and it exists for the same reason:
writing one of these stores is also the same recipe every time. Where a warehouse is
*loaded*, an operational store is **maintained** — a batch of rows is upserted onto
the keys it already holds, a set of expired ones is deleted — so the vocabulary is
`upsert`/`append`/`overwrite`/`delete` rather than a save mode, and every store that
cannot express one of those declines it by name instead of approximating it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from batcher.config import active_config
from batcher.io.manifest import WrittenFile

__all__ = [
    "STORE_WRITE_MODES",
    "BulkSink",
    "ScanSource",
    "offset_windows",
    "rows_to_batches",
    "schema_from_rows",
]

# An opaque, picklable partition locator (token range, segment id, offset, …).
# It is connector-defined; the base treats it as a black box it round-trips to
# ``_read_partition`` on the worker.
PartitionLocator = Any


def require_driver(module: str, extra: str) -> Any:
    """Import `module` or raise `BackendError` pointing at the right extra.

    Args:
        module: The dotted import path of the optional driver (e.g. ``"pymongo"``).
        extra: The Batcher extras key that installs it (e.g. ``"mongo"``).

    Returns:
        The imported module object.

    Raises:
        BackendError: If the driver is not installed.
    """
    from batcher._internal.optional import require

    # Through the one guard rather than a third copy of it: `require` raises
    # `MissingDependencyError`, which is an `ImportError` as well as a `BackendError` and
    # carries the install command in a field a caller can surface its own way.
    return require(module, feature=f"The {extra} source", provides=module, extra=extra)


def rows_to_batches(
    rows: Iterator[dict[str, Any]],
    schema: pa.Schema | None = None,
    batch_rows: int | None = None,
) -> Iterator[pa.RecordBatch]:
    """Assemble an iterator of row dicts into Arrow batches of `batch_rows`.

    This is the row→Arrow bridge for drivers with no Arrow-native reader: rows
    accumulate into a buffer and are converted in bulk (`pa.RecordBatch.from_pylist`)
    once per batch — never per row in the hot path of an operator, only at the
    source boundary where the data is intrinsically row-shaped.

    Args:
        rows: An iterator of row dictionaries (column name → scalar value).
        schema: Optional Arrow schema to coerce each batch to; inferred if None.
        batch_rows: Target row count per emitted batch; defaults to the engine's
            configured morsel size (`ExecutionConfig.morsel_rows`) so source batches
            match downstream operator granularity.

    Yields:
        `pa.RecordBatch`, each holding up to `batch_rows` rows.
    """
    if batch_rows is None:
        batch_rows = active_config().execution.morsel_rows
    buffer: list[dict[str, Any]] = []
    for row in rows:
        buffer.append(row)
        if len(buffer) >= batch_rows:
            yield pa.RecordBatch.from_pylist(buffer, schema=schema)
            buffer = []
    if buffer:
        yield pa.RecordBatch.from_pylist(buffer, schema=schema)


def schema_from_rows(rows: list[dict[str, Any]]) -> pa.Schema:
    """Infer an Arrow schema from a list of sampled row dicts.

    This is the schema half of the row->Arrow bridge: `_infer_schema` samples a
    handful of rows from the store (usually a single ``LIMIT 1`` row) and needs
    the Arrow schema those rows imply. Arrow derives it from the sample the same
    way `rows_to_batches` builds data batches, so both halves agree by
    construction.

    An empty sample carries no type information, so the result is the empty
    schema (`pa.schema([])`). A connector that has a meaningful schema for the
    empty case (a fixed key column, say) must keep its own guard rather than
    call this.

    Args:
        rows: The sampled row dictionaries (column name -> scalar value). Pass a
            single-element list to infer from one sampled row.

    Returns:
        The Arrow schema the sample implies; the empty schema when `rows` is empty.
    """
    return pa.schema([]) if not rows else pa.RecordBatch.from_pylist(rows).schema


@dataclass(frozen=True, slots=True)
class _ScanSplit:
    """A picklable, independently-readable slice of a `ScanSource`.

    Carries only locators: the connector class, the (never-logged) connection
    kwargs needed to rebuild a connection on the worker, and the opaque partition
    locator. It holds **no live connection** — the worker reconstructs the
    connector and calls `_read_partition` for just this partition.
    """

    source_cls: type[ScanSource]
    conn_kwargs: dict[str, Any] = field(repr=False)
    partition: PartitionLocator
    identity_prefix: str
    predicate: dict | None = None

    def _source(self) -> ScanSource:
        return self.source_cls(**self.conn_kwargs)

    def schema(self) -> pa.Schema:
        return self._source().schema()

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        return list(self.iter_batches(projection, predicate))

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        # The predicate Kyber pushed at *plan* time is baked into the split (`self.predicate`);
        # `predicate` is the one the distributed reader hands us at *read* time. They are the
        # same filter arriving by two routes, so either will do — but one of them must arrive,
        # or the worker fetches the whole store and the engine's `Filter` throws it away.
        pushed = predicate if predicate is not None else self.predicate
        yield from self._source()._read_partition(self.partition, projection, pushed)

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"{self.identity_prefix}:part={self.partition!r}"


def offset_windows(total: int | None, segments: int) -> list[tuple[int, int]]:
    """``(offset, limit)`` windows that are a disjoint **and exhaustive** cover of a result.

    An offset window is the only way to split a store that has no native shard/token/segment
    primitive — Couchbase's SQL++ and Neo4j's Cypher both fall here. It is easy to get subtly,
    silently wrong, and it was:

        return [(i * _WINDOW_ROWS, _WINDOW_ROWS) for i in range(segments)]

    With a fixed window size, `segments` windows cover only ``segments * _WINDOW_ROWS`` rows.
    Every row past that was **dropped, with no error**: eight segments over a billion-row
    collection returned 800,000 rows and reported success. Turning on parallelism — the thing
    you do *because* the data is large — was what silently truncated it.

    Two properties make a set of windows a cover, and both are enforced here:

    * **Exhaustive** — the final window is *unbounded* (``limit == 0``), so the cover runs to
      the end of the result no matter how far off `total` is, including rows written after it
      was measured. This is what makes the function safe even when the count is a lie.
    * **Disjoint** — offsets are strictly increasing and each window's limit exactly reaches
      the next offset, so no row is read twice.

    Without a `total` there is no way to size the windows, and a guess would either truncate
    (fatal) or overlap. So an unknown `total` yields a **single unbounded window**: one serial
    reader, correct, slow — which is the right trade, because a slow right answer is a result
    and a fast wrong one is a bug.

    Args:
        total: The result's row count, if the store can be asked cheaply. None if it cannot.
        segments: The requested parallelism.

    Returns:
        The windows, in offset order. Always non-empty; the last is always unbounded.
    """
    if segments <= 1 or total is None or total <= 0:
        return [(0, 0)]  # 0 limit = unbounded: read to the end
    # More segments than rows only makes empty windows; one row per window is the useful floor.
    segments = min(segments, total)
    step = -(-total // segments)  # ceil, so the windows reach `total` before the last one
    windows = [(i * step, step) for i in range(segments - 1)]
    windows.append(((segments - 1) * step, 0))  # the tail, unbounded — this is the cover
    return windows


@dataclass(frozen=True, slots=True)
class PartitionSpec:
    """How a `ScanSource` is divided into parallel read units.

    Concrete connectors interpret this against their store: `segments` is the
    requested parallelism (DynamoDB ``TotalSegments``, slice count, offset
    windows), and `extra` carries connector-specific knobs (page size, ring
    token count, …). It is a picklable value object so it travels inside a split.
    """

    segments: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


class ScanSource(ABC):
    """Base for a row-based NoSQL/operational store read as Arrow.

    Construction stores the connection kwargs (verbatim, never logged) plus an
    optional `PartitionSpec`. The base implements the `Source` surface
    (`schema`/`read`/`iter_batches`/`row_count`/`identity`/`splits`) in terms of
    two overrides:

    * `_enumerate_partitions()` — the store's natural parallel units as opaque,
      picklable locators (token ranges, scan segments, offset windows).
    * `_read_partition(partition, projection)` — fetch one partition's rows and
      yield Arrow batches (typically via `rows_to_batches`).

    Subclasses also set `format_name` (the registry key) and implement
    `_infer_schema()`.
    """

    # The registry name and the picklable connection kwargs. Subclasses set
    # `format_name`; the base keeps `_conn_kwargs` opaque and never logs it.
    format_name: str = ""

    __slots__ = ("_conn_kwargs", "_partition_spec", "_schema_cache")

    def __init__(
        self,
        *,
        partition_spec: PartitionSpec | None = None,
        **conn_kwargs: Any,
    ) -> None:
        self._conn_kwargs = conn_kwargs
        self._partition_spec = partition_spec or PartitionSpec()
        self._schema_cache: pa.Schema | None = None

    # ---- shared, do-not-override ------------------------------------------
    def schema(self) -> pa.Schema:
        if self._schema_cache is None:
            self._schema_cache = self._infer_schema()
        return self._schema_cache

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        return list(self.iter_batches(projection, predicate))

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        for partition in self._enumerate_partitions():
            yield from self._read_partition(partition, projection, predicate)

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        """The learned-statistics key: the store, the human locator, and the connection.

        `identity()` is the key a source's learned statistics are stored under, so two
        sources sharing an identity are treated by Kyber as the *same relation*. Every
        connector here built one from `_identity_suffix()` alone, and every suffix is a
        partial view of the connection: Elasticsearch used the bare index name, so
        ``orders`` on production and ``orders`` on staging were one relation; Redis used
        ``host:port/db`` and dropped the ``match`` glob, so a scan of ``user:*`` and a scan
        of ``session:*`` shared a key. Kyber then applies the billion-row relation's
        cardinalities to the thousand-row one and picks a plan for the wrong data. Nothing
        errors — the query is correct and the plan is wrong.

        Folding the fingerprint in *here* rather than in each `_identity_suffix()` is what
        makes it hold for every connector, including ones added later: a subclass cannot
        forget the part that keeps its statistics its own.

        `connection_fingerprint` is `sha256` (not `hash()`, which Python salts per process,
        so the key would differ every run and no statistic would ever be reused) and it
        excludes credential-ish keys, so rotating a password neither leaks into the
        persisted key nor orphans everything already learned.
        """
        # Deferred: importing `batcher.io.formats.sql._common` executes the `sql` package
        # __init__, which imports every SQL connector. That is a needless import-time cost
        # for a NoSQL read, and it only matters once, when an identity is actually asked for.
        from batcher.io.formats.sql._common import connection_fingerprint

        return (
            f"{self.format_name}:{self._identity_suffix()}:"
            f"{connection_fingerprint(self._fingerprint_material())}"
        )

    def _fingerprint_material(self) -> dict[str, Any]:
        """The connection kwargs as they should be fingerprinted; `_conn_kwargs` by default.

        `connection_fingerprint` drops credentials it can recognize *by key name*, so a
        rotated ``password=`` does not change the key and orphan everything learned. That
        misses a credential embedded inside a larger value — a ``bolt://user:pass@host``
        URI is one field, and rotating the password inside it silently re-keys the
        relation. A connector whose connection carries a credential that way overrides
        this to mask it first, exactly as `odbc._connection_key` does.
        """
        return self._conn_kwargs

    def splits(
        self,
        target_size: int | None = None,  # noqa: ARG002 (protocol signature; a store splits by partition)
        predicate: dict | None = None,
    ) -> list[_ScanSplit]:
        """The store's partitions, each carrying the predicate Kyber pushed to this scan.

        Declaring `predicate` here is what connects a NoSQL store to the optimizer on the
        **distributed** path. `io.source.plan_splits` only passes a predicate to a `splits()`
        that asks for one, and `dist...scan_read._split_read` only passes it to a `Split.read`
        that asks for one — so a source that omits it silently reads its *entire* table on
        every worker and lets the engine's `Filter` discard the rows. Correct, and ruinous: on
        DynamoDB you pay read-capacity for every item; on Mongo you drag the collection across
        the network.

        Baking it into the split (rather than into per-read instance state, which is what the
        Cassandra/DynamoDB sources used to do) is also what makes pushdown *survive the trip to
        the worker*: a split is picklable and self-contained, and `_source()` rebuilds a clean
        connector from `conn_kwargs` that knows nothing about any earlier call.
        """
        prefix = self.identity()
        return [
            _ScanSplit(
                source_cls=type(self),
                conn_kwargs=dict(self._conn_kwargs),
                partition=partition,
                identity_prefix=prefix,
                predicate=predicate,
            )
            for partition in self._enumerate_partitions()
        ]

    # ---- override points --------------------------------------------------
    @abstractmethod
    def _infer_schema(self) -> pa.Schema:
        """Determine the Arrow schema (sampling a row or reading store metadata)."""

    @abstractmethod
    def _enumerate_partitions(self) -> list[PartitionLocator]:
        """Return the store's natural parallel units as opaque, picklable locators."""

    @abstractmethod
    def _read_partition(
        self,
        partition: PartitionLocator,
        projection: list[str] | None,
        predicate: dict | None = None,
    ) -> Iterator[pa.RecordBatch]:
        """Fetch one partition's rows and yield Arrow batches.

        `predicate` is the filter to push **into the store's own query** — a DynamoDB
        `FilterExpression`, a CQL `WHERE`, a Mongo query document, an ES DSL clause. It is
        passed as an argument rather than held on the instance so that pushdown is a pure
        function of `(partition, projection, predicate)`: reentrant under concurrent reads,
        and identical whether it runs on the driver or on a worker that rebuilt this source
        from a pickled split. A connector that cannot push it simply ignores it — the engine's
        `Filter` re-checks every row regardless, so ignoring a predicate is always correct and
        merely slower.
        """

    def _secret(self, key: str) -> Any:
        """A credential from `_conn_kwargs`, resolving an ``env:``/``file:`` reference.

        Call this from `_client`/`_driver`/`_cluster` — i.e. where the connection is
        actually opened, which is the worker — never from `__init__` or anything the
        driver evaluates. The source object and its pickled split then carry only the
        reference, so the secret never crosses the wire and cannot surface in a traceback
        or log line. A plain literal passes through unchanged, so a user who supplies a raw
        password keeps working exactly as before.
        """
        from batcher.io.credentials import resolve_secret

        return resolve_secret(
            self._conn_kwargs.get(key), what=f"{self.format_name or 'connector'} {key}"
        )

    def _identity_suffix(self) -> str:
        """A non-secret identity suffix; defaults to the connection target.

        Subclasses override to surface a stable, credential-free locator (host +
        keyspace, cluster + collection, …). The base falls back to a generic tag
        so credentials in `_conn_kwargs` never leak into an identity string.
        """
        return "store"


#: The write modes an operational store can be maintained with.
#:
#: This is the same vocabulary the SQL sink uses (`io.formats.sql.dbapi.sink.WRITE_MODES`),
#: minus the two forms that only mean something with a statement engine underneath. Keeping
#: the spelling identical is deliberate: "maintain this table by key" is one concept, and a
#: user moving a pipeline from Postgres to DynamoDB should not have to relearn it.
STORE_WRITE_MODES = ("upsert", "append", "overwrite", "delete")


class BulkSink(ABC):
    """Base for writing Arrow rows into a row-based operational store.

    Concrete sinks override `_apply`, which receives one batch of already-converted
    Python rows and applies `mode` to them in as few round trips as the driver allows.
    Everything around that is the same for every store and lives here: validating the
    mode against what this sink `supported_modes`, skipping an empty write, refusing a
    destructive mode past the first shard of a distributed write, and returning the
    `WrittenFile` the manifest is built from.

    **Declining is a first-class answer.** These stores genuinely differ in what they can
    express: a DynamoDB ``PutItem`` replaces by key, so it cannot be a true insert-only
    append; emptying a Redis keyspace is a ``FLUSHDB`` nobody should reach by passing a
    string. A sink lists what it implements in `supported_modes` and the base raises for
    the rest, naming the mode that does work. An approximation would be a wrong answer
    dressed as a feature.

    Args:
        key_field: The field identifying a row, for the modes that match on one.
        mode: One of `supported_modes`.
        conn_kwargs: Connection keywords, stored verbatim and never logged.

    Raises:
        BackendError: If `mode` is not one this sink implements.
    """

    #: The registry name, set by each subclass.
    format_name: str = ""

    #: Row-level writes, so `mode` is this sink's vocabulary rather than a save mode.
    #: Derived from `supported_modes` below, never set by hand — see `__init_subclass__`.
    dml_modes: tuple[str, ...] = STORE_WRITE_MODES

    #: The subset of `STORE_WRITE_MODES` this store can express. Subclasses narrow it.
    supported_modes: tuple[str, ...] = STORE_WRITE_MODES

    #: Modes that discard rows the write itself did not supply.
    destructive_modes: frozenset[str] = frozenset({"overwrite"})

    __slots__ = ("_conn_kwargs", "key_field", "mode")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Keep the writer's view of a sink's modes equal to what the sink implements.

        `Writer.__call__` asks a sink class for `dml_modes` to decide whether a `mode` is a
        row-level verb it should forward verbatim or a save mode it should normalize. Two
        attributes saying nearly the same thing is how they drift: a sink that narrowed
        `supported_modes` but left `dml_modes` alone would let a mode past the writer's gate
        only to raise at construction. Deriving one from the other removes the question.
        """
        super().__init_subclass__(**kwargs)
        cls.dml_modes = cls.supported_modes

    def __init__(self, *, key_field: str = "id", mode: str = "upsert", **conn_kwargs: Any) -> None:
        from batcher._internal.errors import BackendError

        if mode not in self.supported_modes:
            declined = set(STORE_WRITE_MODES) - set(self.supported_modes)
            extra = (
                f" {self.format_name} cannot express {sorted(declined)}."
                if mode in declined
                else ""
            )
            raise BackendError(
                f"unknown {self.format_name or 'store'} write mode {mode!r}; this sink "
                f"implements {list(self.supported_modes)}.{extra}"
            )
        self.key_field = key_field
        self.mode = mode
        self._conn_kwargs = conn_kwargs

    def write(self, table: pa.Table, path: str) -> WrittenFile:
        """Apply every row of `table` to `path` per `mode`, in as few round trips as possible.

        Args:
            table: The rows to apply.
            path: The target's logical name (table, collection, index, key prefix).

        Returns:
            A `WrittenFile` recording the rows applied.
        """
        from batcher.plan.types import logical_bytes

        rows = table.to_pylist()
        if not rows and self.mode != "overwrite":
            return WrittenFile(path=path, rows=0, bytes=0)
        self._apply(rows, path)
        return WrittenFile(path=path, rows=len(rows), bytes=logical_bytes(table))

    def write_partitioned(
        self,
        table: pa.Table,
        path: str,
        *,
        partition_by: list[str] | None = None,  # noqa: ARG002 - no Hive layout in these stores
        file_index: int = 0,
    ) -> list[WrittenFile]:
        """Write one shard; every shard targets the same store.

        A file sink gives each shard its own ``part-N`` file, so shards cannot collide. An
        operational store has no such luxury: a mode that discards rows the shard did not
        write is applied by *every* shard independently, so each one discards the shards
        before it. It is invisible single-node, where there is only ever one shard, and
        appears at cluster scale as missing rows rather than an error.

        Raises:
            BackendError: If a destructive `mode` meets a multi-shard write.
        """
        from batcher._internal.errors import BackendError

        if file_index > 0 and self.mode in self.destructive_modes:
            raise BackendError(
                f"mode={self.mode!r} cannot be used for a distributed write to {path!r}: "
                "every shard would apply it to the same target, so each one would discard "
                "the shards before it. Use mode='upsert', or empty the target beforehand "
                "and append."
            )
        return [self.write(table, path)]

    def commit(self, manifest: Any, path: str) -> None:  # noqa: B027 - see below
        """No-op: these stores have no commit phase — a write is visible when it lands.

        Deliberately concrete and deliberately empty. The `Sink` protocol has a two-phase
        shape because a transactional sink needs one: workers write data files and the
        driver publishes them in a single commit. None of these stores works that way —
        a document is live the moment ``bulk_write`` returns — so there is nothing for a
        subclass to override, and marking it abstract would force every one of them to
        write the same empty method.
        """

    def _secret(self, key: str) -> Any:
        """A credential from the connection kwargs, resolving an ``env:``/``file:`` reference.

        Call this where the connection is opened — on the worker — never in `__init__`, so
        the pickled sink carries only the reference. This is `ScanSource._secret`'s twin,
        and it exists because the failure it prevents already happened on the write side:
        `MongoSink` dialed `MongoClient(self.uri)` on the raw attribute, so an ``env:``
        reference that read fine was handed to the driver verbatim and failed to connect.
        """
        from batcher.io.credentials import resolve_secret

        return resolve_secret(
            self._conn_kwargs.get(key), what=f"{self.format_name or 'connector'} {key}"
        )

    @abstractmethod
    def _apply(self, rows: list[dict[str, Any]], path: str) -> None:
        """Apply `rows` to `path` under `self.mode`, in as few round trips as the driver allows.

        Called with a whole batch, never one row at a time: every store here has a bulk
        primitive (``bulk_write``, ``batch_writer``, a prepared-statement batch, a pipeline,
        ``_bulk``) and using it is the difference between one round trip and thousands.
        """
