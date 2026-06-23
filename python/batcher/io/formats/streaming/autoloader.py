"""Incremental file discovery — the Auto Loader analog (Databricks ``cloudFiles``).

:class:`IncrementalFileSource` watches a directory and, on each discovery pass,
yields the batches of files it has *not* seen before — exactly-once incremental
ingestion of a growing directory of files (logs, exports, CDC dumps, …).

How it works:

* it LISTs the directory with :func:`resolve_filesystem` (so local and cloud
  paths work unchanged);
* a **lexical fast-path** narrows the listing to files ordering after the maximum
  already-seen name — useful when files are written under monotonically
  increasing names (timestamps, sequence ids);
* a durable :class:`~batcher.io.formats.streaming.seen_store.SeenStore` (stdlib SQLite, no extra
  dependency) dedups across passes and process restarts;
* new files are read by delegating to the registered file reader for ``format``
  (looked up in the ``SOURCES`` registry), so any file format Batcher supports
  (Parquet, CSV, JSON, …) is ingestible incrementally with no extra dependency.

``iter_batches()`` performs **one** discovery pass and yields the new files'
batches; a streaming driver calls it repeatedly to keep ingesting. The schema is
the schema of the chosen file format (sampled from the first available file).
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pyarrow as pa

from batcher._internal.errors import IOError
from batcher.io.filesystem import resolve_filesystem
from batcher.io.formats.base import SOURCES
from batcher.io.formats.streaming.seen_store import SeenStore
from batcher.io.splits import FileSplit, Split

__all__ = ["IncrementalFileSource"]

_DEFAULT_SUFFIX = {"parquet": ".parquet", "csv": ".csv", "json": ".json"}


@SOURCES.register("files_incremental")
class IncrementalFileSource:
    """A directory watched for new files, ingested exactly once per file.

    Args:
        path: The directory (or glob) to watch. Local or any ``resolve_filesystem``
            scheme (``s3://``, ``gs://``, …).
        format: The registered file format of the files (``"parquet"``,
            ``"csv"``, ``"json"``, …) — its reader is used to read each new file.
        state_dir: Directory holding the durable seen-file store. Created if
            missing; the store file lives at ``<state_dir>/<format>_seen.sqlite``.
        suffix: File suffix to list (default derived from ``format``).

    Laziness: ``iter_batches`` runs one discovery pass per call (a streaming
    driver loops). ``row_count`` is ``None`` — the directory is unbounded over
    time. Splits are one :class:`FileSplit` per *new* file.
    """

    bounded = False  # the directory grows over time — an unbounded stream

    __slots__ = ("_format", "_fs", "_path", "_schema_cache", "_state_dir", "_suffix")

    def __init__(
        self,
        path: str,
        format: str,
        *,
        state_dir: str,
        suffix: str | None = None,
    ) -> None:
        self._path = path
        self._format = format
        self._state_dir = state_dir
        self._suffix = suffix if suffix is not None else _DEFAULT_SUFFIX.get(format, "")
        self._fs = resolve_filesystem(path)
        self._schema_cache: pa.Schema | None = None

    # ---- discovery --------------------------------------------------------
    def _store(self) -> SeenStore:
        self._fs.mkdirs(self._state_dir, exist_ok=True)
        return SeenStore(os.path.join(self._state_dir, f"{self._format}_seen.sqlite"))

    def _list_candidates(self, store: SeenStore) -> list[str]:
        """List directory files, applying the lexical fast-path against `store`.

        ``expand`` returns a sorted list; keeping only names strictly greater than
        the max-seen name skips re-listing the (lexically earlier) processed
        prefix when files are written under increasing names.
        """
        try:
            files = self._fs.expand(self._path, suffix=self._suffix)
        except IOError:
            return []  # empty / not-yet-populated directory is not an error here.
        max_seen = store.max_seen()
        if max_seen is None:
            return files
        return [f for f in files if f > max_seen]

    def discover(self) -> list[str]:
        """Return the list of new (unseen) files for the current pass.

        Pure of side effects on the store except marking — exposed so a driver
        can introspect what a pass found. Marks each returned file as seen.
        """
        with self._store() as store:
            candidates = self._list_candidates(store)
            new_files = store.unseen(candidates)
            for f in new_files:
                store.mark(f, size=_safe_size(self._fs, f), mtime=_safe_mtime(f))
            return new_files

    # ---- Source protocol --------------------------------------------------
    def schema(self) -> pa.Schema:
        """The file format's schema, sampled from the first available file."""
        if self._schema_cache is None:
            try:
                files = self._fs.expand(self._path, suffix=self._suffix)
            except IOError as exc:
                raise IOError(
                    f"cannot infer schema: no {self._suffix} files yet under {self._path!r}"
                ) from exc
            reader = SOURCES.get(self._format)(files[0])
            self._schema_cache = reader.schema()
        return self._schema_cache

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return list(self.iter_batches(projection))

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Run one discovery pass; yield the new files' batches via the file reader."""
        for path in self.discover():
            reader = SOURCES.get(self._format)(path)
            yield from reader.iter_batches(projection)

    def row_count(self) -> int | None:
        return None  # the watched directory grows over time.

    def identity(self) -> str:
        return f"files_incremental:{self._format}:{self._path}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        """One :class:`FileSplit` per new file (locator-only, picklable)."""
        return [FileSplit(self._format, path) for path in self.discover()]


def _safe_size(fs: object, path: str) -> int:
    try:
        return int(fs.size(path))  # type: ignore[attr-defined]
    except (OSError, ValueError, AttributeError):
        return 0


def _safe_mtime(path: str) -> float:
    """Best-effort modification time; 0.0 when the filesystem can't report one."""
    try:
        return float(os.path.getmtime(path))
    except (OSError, ValueError):
        return 0.0
