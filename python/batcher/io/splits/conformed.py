"""The strict-mode contract, carried to the worker on the split itself.

A multi-file source in ``schema_mode="strict"`` (the default) declares that file 0's schema
is the schema of every file, and the single-node read enforces that per file
(`io.schema.conform_batch`): a missing column raises, an extra one is dropped, a differing
type is cast only when no value changes. A split, though, rebuilds a *single-file* reader on
the worker, and a one-file source's contract is that file's own schema, so the check never
ran on the distributed path. Each worker returned whatever its file held and the driver's
gather union-reconciled the lot: a column renamed from ``A`` to ``a`` came back as one row of
two, an ``int64`` file came back as ``float64`` under a plan that said ``int64``, and a
missing column became nulls where the single-node read raised.

`ConformedSplit` wraps a split with the declared schema so the worker applies the same
check to the same file and raises the same `SchemaError`. `FileSource.splits` wraps only the
splits whose file it cannot already prove matches the contract, so a homogeneous directory
keeps the split kinds the fast readers recognize.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from inspect import signature
from typing import TYPE_CHECKING

import pyarrow as pa

if TYPE_CHECKING:
    from batcher.io.splits.base import Split

__all__ = ["ConformedSplit", "fragments_conform"]


@dataclass(frozen=True, slots=True)
class ConformedSplit:
    """A split whose every batch is held to a strict-mode source's declared schema.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.io.splits import ConformedSplit, FileSplit
            >>> declared = pa.schema([("a", pa.int64())])
            >>> split = ConformedSplit(FileSplit("parquet", "f1.parquet"), declared)
            >>> split.schema().names
            ['a']

    Attributes:
        inner: The split as the format planned it.
        target: The source's declared schema, which every batch must match.
    """

    inner: Split
    target: pa.Schema

    def __getattr__(self, name: str) -> object:
        """The inner split's attribute, for code that inspects a split's locator.

        A split's `path`, `paths`, `format_name` and `kwargs` say *what* it reads, and the
        wrapper changes only how the batches are checked, so it answers with the inner
        split's. Reading goes through this class's own methods, which are found first.
        """
        if name.startswith("__") or name in ("inner", "target"):
            raise AttributeError(name)  # unpickling probes these before `inner` is set
        return getattr(object.__getattribute__(self, "inner"), name)

    def _paths(self) -> tuple[str, ...]:
        """The files this split reads, in order: one, or a grouped split's several."""
        paths = getattr(self.inner, "paths", None)
        return tuple(paths) if paths is not None else (self.inner.path,)  # type: ignore[attr-defined]

    def schema(self) -> pa.Schema:
        """The declared schema every batch of this split conforms to.

        Returns:
            The source's declared schema.
        """
        return self.target

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        """Read the inner split and conform each file's batches to the declared schema.

        Args:
            projection: Columns to read. All declared columns when omitted.
            predicate: A filter the format may apply during the read. The engine re-checks
                it regardless, so ignoring it is still correct.

        Returns:
            The batches, each matching the declared schema narrowed to `projection`.

        Raises:
            SchemaError: If a file lacks a declared column or holds one whose values do not
                convert to the declared type unchanged.
        """
        from batcher.io._concurrent import read_each_file
        from batcher.io.source import read_source

        readers = dict(self._readers())

        def _read(_fs: object, path: str) -> list[pa.RecordBatch]:
            reader = readers[path]
            wanted = self._file_projection(reader, projection)
            if isinstance(reader, _SplitReader):
                batches = reader.read(wanted, predicate)
            else:
                batches = read_source(reader, wanted, predicate)
            return list(self._conformed(batches, projection, path))

        # Concurrent on a remote store, serial locally, exactly as `MultiFileSplit.read`
        # decides for the same group of files.
        per_file = read_each_file(None, list(readers), _read)
        out = [batch for batches in per_file for batch in batches]
        # A reader returns at least one batch, because a batch is the only carrier of a
        # schema across the FFI boundary; keep that true of the conformed read.
        return out or [_empty(self._target(projection))]

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Stream the inner split, conforming each batch to the declared schema.

        Args:
            projection: Columns to read. All declared columns when omitted.

        Returns:
            An iterator over the conformed batches.
        """
        for path, reader in self._readers():
            wanted = self._file_projection(reader, projection)
            yield from self._conformed(reader.iter_batches(wanted), projection, path)

    def row_count(self) -> int | None:
        """The inner split's row count; conforming never adds or removes a row.

        Returns:
            The row count, or None when it is not known without reading.
        """
        return self.inner.row_count()

    def identity(self) -> str:
        """The inner split's identity, marked with the contract it is held to.

        Distinct from the inner identity on purpose: a worker's scan cache is keyed on it,
        and the conformed batches can carry different types from the file's own.

        Returns:
            A key that distinguishes this split from its siblings and from its inner split.
        """
        digest = hashlib.sha256(self.target.to_string().encode()).hexdigest()[:12]
        return f"{self.inner.identity()}#strict={digest}"

    def _readers(self) -> list[tuple[str, object]]:
        """One ``(path, reader)`` per file, so a mismatch names the file it came from."""
        paths = self._paths()
        if len(paths) == 1:
            return [(paths[0], _SplitReader(self.inner))]
        make = self.inner._reader  # type: ignore[attr-defined]
        return [(path, make(path)) for path in paths]

    def _file_projection(self, reader: object, projection: list[str] | None) -> list[str] | None:
        """`projection` narrowed to the columns the file holds.

        Asking a reader for a column its file lacks fails inside the format with a message
        about the format. Narrowing lets `conform_batch` see the gap and raise the same
        `SchemaError` the single-node read raises for it.
        """
        if projection is None:
            return None
        present = set(reader.schema().names)  # type: ignore[attr-defined]
        return [c for c in projection if c in present]

    def _conformed(
        self,
        batches: Iterator[pa.RecordBatch] | list[pa.RecordBatch],
        projection: list[str] | None,
        path: str,
    ) -> Iterator[pa.RecordBatch]:
        from batcher.io.schema import conform_batch

        target = self._target(projection)
        for batch in batches:
            # A zero-row batch carries no value to check. The single-node read yields none
            # for an empty file, so it never holds one to the contract, and neither may this.
            yield conform_batch(batch, target, path=path) if batch.num_rows else _empty(target)

    def _target(self, projection: list[str] | None) -> pa.Schema:
        if projection is None:
            return self.target
        return pa.schema([self.target.field(c) for c in projection])


def _empty(schema: pa.Schema) -> pa.RecordBatch:
    return pa.RecordBatch.from_arrays([pa.array([], f.type) for f in schema], schema=schema)


@dataclass(frozen=True, slots=True)
class _SplitReader:
    """A single-file split presented with a reader's surface, passing a predicate through."""

    split: Split

    def schema(self) -> pa.Schema:
        return self.split.schema()

    def read(self, projection: list[str] | None, predicate: dict | None) -> list[pa.RecordBatch]:
        if predicate is not None and "predicate" in signature(self.split.read).parameters:
            return self.split.read(projection, predicate=predicate)  # type: ignore[call-arg]
        return self.split.read(projection)

    def iter_batches(self, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        return self.split.iter_batches(projection)


def fragments_conform(splits: list, fragments: list, max_workers: int) -> bool:
    """Whether a coalesced Parquet scan may read `splits` as their own `fragments`.

    The scan reads each fragment with its own physical schema, so it may take a
    `ConformedSplit` only when there is nothing to conform: every footer equals the declared
    schema exactly. The footers are the ones the scan was about to read, and a fragment
    caches its physical schema, so they are not read twice. They are read concurrently,
    since a small-file partition is mostly latency.

    Args:
        splits: The partition's splits, some of which may be `ConformedSplit`s.
        fragments: The Parquet dataset fragments the scan built from them.
        max_workers: The most footers to read at once.

    Returns:
        True when no split carries a contract, or every fragment's footer matches it.
    """
    from concurrent.futures import ThreadPoolExecutor

    contract = next((s.target for s in splits if isinstance(s, ConformedSplit)), None)
    if contract is None:
        return True
    if len(fragments) == 1:
        return fragments[0].physical_schema.equals(contract)
    with ThreadPoolExecutor(max_workers=max(1, min(len(fragments), max_workers))) as pool:
        return all(pool.map(lambda frag: frag.physical_schema.equals(contract), fragments))
