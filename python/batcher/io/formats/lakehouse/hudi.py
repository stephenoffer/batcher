"""Apache Hudi format — read-only via `hudi` (hudi-rs).

`HudiSource` reads a Hudi table as Arrow: a snapshot read, time travel to an instant, an
incremental read between two instants, and — the part that was missing — a **file-slice
split** per data file, so a Hudi table is read across the cluster instead of streamed
through the driver.

Hudi writes require the Spark/Flink write stack (hudi-rs is a reader; `HudiTableBuilder`
only builds a reader config), so `HudiSink` exists to raise a clear `BackendError` rather
than pretend.

## Two things hudi-rs is particular about

**Filter values are strings.** `HudiReadOptions(filters=[("day", "=", 1)])` raises — hudi-rs
takes the value as text and parses it against the column's type. Passing the literal
through untouched therefore threw on every typed column, and the read fell back to
unfiltered, so Hudi's partition pruning never once ran. `_hudi_filters` stringifies.

**A filter prunes files, not rows.** The filters eliminate whole file slices -- by the
partition path, and (as of hudi-rs 0.5) also by a base file's column statistics, so even an
unpartitioned table skips files a predicate excludes. Nothing finer than a file is removed,
which is exactly the pruning worth having (it is I/O we never do), and the engine's `Filter`
re-checks the rows regardless -- so a filter the reader cannot use is a no-op rather than a
wrong answer.

## The row count does not come from `num_records`

hudi-rs 0.5 exposes `HudiFileSlice.num_records`, and on every slice of a table it returns
the **table's** total rather than that slice's. Summing it is therefore wrong by a factor of
the slice count -- measured on a 15-row, three-slice table, `row_count()` returned **45 and
declared it exact**, which is worse than returning nothing: it is fed to `count()` and to
Kyber's cardinality. The counts here come from the base files' Parquet footers instead
(`_parquet_native.footer_stats` / `file_manifest`, one native pass over metadata the reader
has usually already cached), which is metadata rather than a scan just as the timeline was,
and is exact per slice.

## Merge-on-read

A merge-on-read table's file slice is a base file *plus* log files carrying later
updates and deletes. The per-slice reader used by a split reads the **base file**, so on a
MoR table it would resurrect superseded rows. Rather than return stale data, a table with
log files falls back to a whole-source read, which hudi-rs merges correctly. Copy-on-write
tables — the ones a split read is for — keep the parallel path.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher._internal.logging import note_suppressed
from batcher._internal.optional import require
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.predicate import to_pyarrow_expression
from batcher.io.splits import Split, WholeSourceSplit
from batcher.plan.ir_tags import COMPARISON_FLIP
from batcher.plan.source_stats import SourceStatistics

__all__ = ["HudiFileSliceSplit", "HudiSink", "HudiSource"]


def _require_hudi() -> Any:
    """Import and return the hudi-rs `HudiTable` class or raise `BackendError`."""
    return require(
        "hudi", "HudiTable", feature="Hudi read support", provides="hudi-rs", extra="hudi"
    )


def _read_options(
    *,
    filters: list[Any] | None = None,
    as_of: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> Any:
    """A `HudiReadOptions` carrying whatever of these is set, or `None` when none is.

    hudi-rs takes every read modifier -- partition filters, time travel, the incremental
    window -- through one options object rather than through per-call keyword arguments.
    Returning `None` for the empty case keeps the plain `table.read()` / `get_file_slices()`
    call free of an object that would say nothing.
    """
    options_cls = require(
        "hudi", "HudiReadOptions", feature="Hudi read support", provides="hudi-rs", extra="hudi"
    )
    if not any((filters, as_of, start, end)):
        return None
    options = options_cls(filters=list(filters)) if filters else options_cls()
    if as_of is not None:
        options = options.with_as_of_timestamp(as_of)
    if start is not None:
        query_type = require(
            "hudi", "HudiQueryType", feature="Hudi read support", provides="hudi-rs", extra="hudi"
        )
        options = options.with_query_type(query_type.Incremental).with_start_timestamp(start)
        if end is not None:
            options = options.with_end_timestamp(end)
    return options


def _footer_rows(table_uri: str, relative_paths: list[str]) -> list[int] | None:
    """Exact rows per base file from its Parquet footer, in `relative_paths` order.

    The replacement for `HudiFileSlice.num_records`, which reports the table total on every
    slice (see the module docstring). `None` when the footers cannot be read, which leaves
    the caller to report an unknown count rather than a wrong one.
    """
    from batcher.io.formats.structured._parquet_native import file_manifest

    if not relative_paths:
        return []
    base = table_uri.rstrip("/")
    uris = [f"{base}/{rel.lstrip('/')}" for rel in relative_paths]
    manifest = file_manifest(uris, ["num_records"])
    if manifest is None or "num_records" not in manifest.column_names:
        return None
    counts = manifest.column("num_records").to_pylist()
    return None if len(counts) != len(uris) else [int(c) for c in counts]


_HUDI_OP = {"eq": "=", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}


def _hudi_filters(predicate: dict | None) -> list[tuple[str, str, str]]:
    """Translate an AND-of-comparisons predicate to hudi-rs ``(col, op, value)`` tuples.

    Returns ``[]`` when the predicate is not fully pushable (an OR, a computed term), which
    the caller reads as "scan unpruned" — the engine's `Filter` re-checks the rows either
    way, so an absent translation only ever costs I/O.

    **The value is stringified**, because hudi-rs takes it as text and parses it against the
    column's own type. Handing it a Python `int` raises ``'int' object cannot be converted
    to 'PyString'``, and the caller's fallback then swallowed that into an unfiltered read —
    which is why Hudi's partition pruning had never actually run.
    """
    if predicate is None:
        return []
    out: list[tuple[str, str, str]] = []

    def literal(node: dict) -> str:
        return str(next(iter(node["value"].values())))

    def walk(node: dict) -> bool:
        if node.get("e") != "binary":
            return False
        op = node["op"]
        if op == "and":
            return walk(node["left"]) and walk(node["right"])
        if op in _HUDI_OP:
            left, right = node["left"], node["right"]
            if left.get("e") == "col" and right.get("e") == "lit":
                out.append((left["name"], _HUDI_OP[op], literal(right)))
                return True
            if left.get("e") == "lit" and right.get("e") == "col":
                out.append((right["name"], _HUDI_OP[COMPARISON_FLIP[op]], literal(left)))
                return True
        return False

    return out if walk(predicate) else []


@dataclass(frozen=True, slots=True)
class HudiFileSliceSplit:
    """One Hudi file slice (a base data file), read independently on a worker.

    Carries only locators — the table root, the slice's base-file path, and the reader
    options — so it pickles cheaply and the worker reads just this file. `rows` is the
    slice's record count, which the Hudi timeline already knows, so the distributed planner
    can bin-pack by real size without opening a footer.
    """

    table_uri: str
    base_file_path: str
    # `repr=False`: the reader options carry the cloud storage credentials
    # (``aws.secret.key`` and friends). A split's `repr` is rendered into task logs and
    # into every traceback that crosses a worker, so a generated one publishes the secret
    # to wherever those are collected.
    options: dict[str, str] = field(repr=False)
    rows: int | None = None
    # Carried because a worker sees only the split: a slice enumerated as of an instant
    # must also be *read* as of it, or a merge-on-read slice resolves against the latest
    # log files and the historical read silently returns current data.
    as_of_instant: str | None = None

    def _slice_batches(self) -> Any:
        """This slice's batches, as the reader hands them back — not concatenated.

        Deliberately returns the reader's own iterable rather than a `pa.Table`. Wrapping
        it was what made `iter_batches` a stream in name only (see there).
        """
        hudi_table = _require_hudi()
        table = hudi_table(self.table_uri, options=dict(self.options))
        reader = table.create_file_group_reader_with_options()
        # No log files: `splits()` hands a slice back only for a copy-on-write table, and
        # falls back to a whole-source read the moment any slice carries one (see the
        # module docstring on merge-on-read).
        batch = reader.read_file_slice_from_paths(self.base_file_path, [])
        return [batch] if isinstance(batch, pa.RecordBatch) else batch

    @staticmethod
    def _shape(
        batch: pa.RecordBatch, expression: Any, projection: list[str] | None
    ) -> pa.RecordBatch | None:
        """One batch filtered and projected — the per-batch form of the old whole-table pass."""
        if expression is not None:
            batch = batch.filter(expression)
        if projection is not None:
            batch = batch.select(projection)
        return batch if batch.num_rows else None

    def schema(self) -> pa.Schema:
        return HudiSource(
            self.table_uri, options=dict(self.options), as_of_instant=self.as_of_instant
        ).schema()

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        return list(self.iter_batches(projection, predicate))

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        """Stream this file slice, holding one batch at a time.

        This used to build the whole slice into a `pa.Table`, filter and project *that*, and
        then `.to_batches()` the result — a stream in signature only. The reader's batches
        were drained into one table before a single row could be yielded, so peak memory was
        the decoded size of the entire data file, on the worker, per split. A slice larger
        than the worker's memory could not be read at all — which is the one thing a
        per-file split exists to make impossible.
        """
        expression = to_pyarrow_expression(predicate)
        for batch in self._slice_batches():
            shaped = self._shape(batch, expression, projection)
            if shaped is not None:
                yield shaped

    def row_count(self) -> int | None:
        return self.rows

    def identity(self) -> str:
        return f"hudi:{self.table_uri}:{self.base_file_path}"


@SOURCES.register("hudi")
class HudiSource:
    """A read-only Apache Hudi table read as Arrow.

    Args:
        table_uri: The table root (local path or cloud URI).
        as_of_instant: Optional Hudi instant timestamp for snapshot time travel.
        options: Optional hudi-rs reader options (incl. cloud storage options).
    """

    # Predicate pushdown: a pushed predicate becomes hudi-rs ``(col, op, value)`` filters,
    # which prune whole partitions at the timeline. The engine's `Filter` re-checks the
    # rows, so a predicate the backend cannot use only costs I/O.
    supports_predicate = True

    __slots__ = ("_as_of_instant", "_options", "_table_uri")

    def __init__(
        self,
        table_uri: str,
        *,
        as_of_instant: str | None = None,
        options: dict[str, str] | None = None,
    ) -> None:
        self._table_uri = table_uri
        self._as_of_instant = as_of_instant
        self._options = options or {}

    def _table(self) -> Any:
        hudi_table = _require_hudi()
        try:
            return hudi_table(self._table_uri, options=self._options)
        except Exception as exc:
            raise BackendError(f"failed to open Hudi table {self._table_uri!r}: {exc}") from exc

    def _snapshot(self, table: Any, filters: list[Any]) -> list[pa.RecordBatch]:
        options = _read_options(filters=filters, as_of=self._as_of_instant)
        return table.read(options) if options is not None else table.read()

    def _snapshot_batches(self, predicate: dict | None) -> Any:
        """The snapshot's batches as hudi-rs hands them back, partition-pruned where it can.

        Returns the reader's iterable rather than a materialized `pa.Table` — the
        concatenation is what `iter_batches` must not do (see there).
        """
        table = self._table()
        filters = _hudi_filters(predicate)
        try:
            try:
                return self._snapshot(table, filters)
            except Exception:
                # Backend rejected the pushed filters (version/format mismatch) →
                # read unpruned; the engine's Filter still produces the right rows.
                return self._snapshot(table, [])
        except Exception as exc:
            raise BackendError(f"Hudi snapshot read failed for {self._table_uri!r}: {exc}") from exc

    def _file_slices(self, predicate: dict | None = None) -> list[Any]:
        """The table's file slices, partition-pruned by `predicate` where it can be.

        Honours `as_of_instant`. Enumerating the *latest* slices for a time-travel read
        was silent: `read()` applied the instant, but `splits()` did not, so the same
        as-of query returned the historical snapshot single-node and the current table
        distributed — and a `count()`, which is answered from the slices, reported the
        current row count for a historical read with `exact_rows=True`.
        """
        table = self._table()
        filters = _hudi_filters(predicate)
        instant = self._as_of_instant
        for options in (
            _read_options(filters=filters, as_of=instant),
            _read_options(as_of=instant),
        ):
            try:
                return list(
                    table.get_file_slices(options)
                    if options is not None
                    else table.get_file_slices()
                )
            except Exception as exc:
                note_suppressed("io", "list file slices", exc)
                continue
        return list(table.get_file_slices())

    def schema(self) -> pa.Schema:
        return self._table().get_schema()

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        return list(self.iter_batches(projection, predicate))

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        """Stream the snapshot, projecting each batch as it arrives.

        This was ``self._read_table(...).to_batches()``: the reader's batches were
        concatenated into one `pa.Table`, projected as a whole, and re-chunked. Nothing
        could be yielded until every batch had been pulled, so the "streaming" entry point
        held the entire table — for a lakehouse table, the difference between bounded and
        unbounded memory.

        The bound this path can offer is honest but partial, and worth stating: hudi-rs
        resolves a snapshot read eagerly, so `_snapshot_batches` returns a sequence the
        backend has already built. What this removes is Batcher's own full copy on top of
        it — the concatenate-project-rechunk pass — and it makes the generator lazy with
        respect to whatever the backend does hand back incrementally. The per-slice split
        (`HudiFileSliceSplit`), which is the path a large table actually reads through,
        streams end to end.

        The predicate is deliberately *not* applied per batch here. A Hudi filter prunes
        partitions at the timeline, not rows, and the engine's `Filter` re-checks the rows
        regardless — re-applying it would be duplicated per-row work in the control plane.
        """
        for batch in self._snapshot_batches(predicate):
            yield batch.select(projection) if projection is not None else batch

    def read_incremental(self, start_instant: str, end_instant: str | None = None) -> pa.Table:
        """Read rows changed between two Hudi instants as an Arrow table."""
        table = self._table()
        try:
            batches = table.read(_read_options(start=start_instant, end=end_instant))
            return pa.Table.from_batches(batches)
        except Exception as exc:
            raise BackendError(
                f"Hudi incremental read failed for {self._table_uri!r}: {exc}"
            ) from exc

    def row_count(self) -> int | None:
        """Exact row count from the timeline's per-slice record counts — no scan.

        Hudi records how many rows each file slice holds, so the count is metadata. It used
        to return `None`, which made the estimator guess at a table whose size the timeline
        states outright.
        """
        try:
            slices = self._file_slices()
            counts = _footer_rows(self._table_uri, [s.base_file_relative_path() for s in slices])
        except Exception:
            return None
        return None if counts is None else sum(counts)

    def statistics(self) -> SourceStatistics | None:
        """Exact row count **and the table's partition keys** from the timeline; no scan.

        The partition keys were the missing half. Hudi states them in its table config and
        hands them back as a schema (`get_partition_schema`), and they are exactly what
        tells the planner that a filter on one of those columns eliminates whole file
        slices rather than merely filtering rows — the pruning `splits()` already performs
        but that nothing downstream was told about.

        Only what the timeline actually states is reported. A table whose partition schema
        cannot be read declares no keys rather than a guessed set: an invented partition
        key would have the planner expect a pruning that never happens.
        """
        rows = self.row_count()
        if rows is None:
            return None
        return SourceStatistics(
            row_count=rows, exact_rows=True, partition_keys=self._partition_keys()
        )

    def _partition_keys(self) -> tuple[str, ...]:
        """The table's partition columns as the Hudi timeline declares them, or ``()``."""
        try:
            return tuple(self._table().get_partition_schema().names)
        except Exception:
            return ()

    def identity(self) -> str:
        ref = self._as_of_instant or "latest"
        return f"hudi:{self._table_uri}@{ref}"

    def governed_name(self) -> str:
        """The table a governance policy is written about: the table URI.

        Distinct from `identity`, which names a *relation* and so carries the resolved instant — a
        relation's statistics must not be handed to a different relation. A policy is
        written about the **table**, before anyone has read it and without knowing which
        version they will land on. Reading the table name off the identity meant a policy
        on the table's own path never fired matched nothing, and an ungoverned read of a
        governed table raises
        nothing to say so.

        Returns:
            The table name a policy is keyed on.
        """
        return self._table_uri

    def splits(
        self,
        target_size: int | None = None,  # noqa: ARG002 - file slices are not coalescable
        predicate: dict | None = None,
    ) -> list[Split]:
        """One split per surviving file slice, so a Hudi table is read in parallel.

        `predicate` prunes whole partitions at the timeline before a single file is opened.
        This used to return one `WholeSourceSplit`, which meant the entire table was read
        through the driver and re-scattered — no distributed read, no pruning, and the
        driver's memory as the ceiling on table size.

        **Merge-on-read tables keep the whole-source path.** A MoR slice is a base file plus
        log files holding later updates and deletes, and the per-slice reader below reads the
        base file — so splitting one would resurrect superseded rows. Correctness first: a
        table with log files is read whole, which hudi-rs merges properly.
        """
        try:
            slices = self._file_slices(predicate)
        except Exception:
            return [WholeSourceSplit(self)]
        if not slices:
            return [WholeSourceSplit(self)]
        if any(_has_log_files(s) for s in slices):
            return [WholeSourceSplit(self)]
        paths = [s.base_file_relative_path() for s in slices]
        rows = _footer_rows(self._table_uri, paths) or [None] * len(paths)
        return [
            HudiFileSliceSplit(
                self._table_uri,
                path,
                dict(self._options),
                count,
                self._as_of_instant,
            )
            for path, count in zip(paths, rows, strict=True)
        ]


def _has_log_files(file_slice: Any) -> bool:
    """Whether a file slice carries merge-on-read log files (updates the base file lacks)."""
    try:
        return bool(file_slice.log_files_relative_paths())
    except Exception:
        return True  # cannot prove it is a plain base file → do not split it


@SINKS.register("hudi")
class HudiSink:
    """Placeholder Hudi sink — writes require the Spark/Flink write stack."""

    __slots__ = ()

    def __init__(self, *_: Any, **__: Any) -> None:
        raise BackendError("Hudi writes require Spark/Flink; Batcher supports Hudi reads only")
