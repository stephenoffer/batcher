"""Incremental file discovery — the Auto Loader analog (Databricks ``cloudFiles``).

:class:`IncrementalFileSource` watches a directory and, on each discovery pass,
yields the batches of files it has *not* seen before — exactly-once incremental
ingestion of a growing directory of files (logs, exports, CDC dumps, …).

How it works:

* it LISTs the directory with :func:`resolve_filesystem` (so local and cloud
  paths work unchanged);
* a durable :class:`~batcher.io.formats.streaming.seen_store.SeenStore` (stdlib SQLite, no extra
  dependency) dedups across passes and process restarts;
* new files are read by delegating to the registered file reader for ``format``
  (looked up in the ``SOURCES`` registry), so any file format Batcher supports
  (Parquet, CSV, JSON, …) is ingestible incrementally with no extra dependency.

``iter_batches()`` performs **one** discovery pass and yields the new files'
batches; a streaming driver calls it repeatedly to keep ingesting. The schema is
the schema of the chosen file format (sampled from the first available file).

**A discovered file is not a processed file.** Discovery used to mark a file seen the
moment it was *listed* — before a single row of it had been read, let alone written to a
sink or committed. A query that died mid-epoch therefore came back to a store that already
claimed those files, skipped them forever, and silently dropped the data; and because the
source exposed no `seek`, no checkpoint could recover it. The durable store now records
only what has been **published**: discovery holds new files as *pending*, and `confirm()`
promotes them once the epoch that read them is committed. A crash leaves them unconfirmed,
so the next pass finds them again and the sink's per-epoch transaction makes the replay
idempotent — data loss traded for a replay that cannot double-write.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

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
        max_files_per_trigger: Cap on the new files one discovery pass admits, so a
            large backlog drains across many bounded micro-batches (Spark
            ``maxFilesPerTrigger``). ``None`` admits every new file.
        max_bytes_per_trigger: Cap on the total *size* one pass admits (Spark
            ``maxBytesPerTrigger``). Composes with `max_files_per_trigger`; whichever
            bound trips first wins. A single file larger than the budget is still
            admitted alone.
        reader_options: Everything else, forwarded verbatim to the file reader for each
            new file — a CSV ``delimiter``, a declared ``schema``, ``storage_options``
            for the store the files live in, and the two tolerance flags (``on_error``,
            ``on_bad_lines``). A continuous ingest is exactly where these matter most:
            the files arrive from a producer nobody is watching, so a stream that cannot
            be told to tolerate one malformed record stops on it and stays stopped.

    Laziness: ``iter_batches`` runs one discovery pass per call (a streaming
    driver loops). ``row_count`` is ``None`` — the directory is unbounded over
    time. Splits are one :class:`FileSplit` per *new* file.
    """

    bounded = False  # the directory grows over time — an unbounded stream
    #: A pass's new files are independent work units, so a micro-batch can be fanned
    #: across the cluster one file per worker (see `dist.streaming.microbatch`).
    partitionable = True
    #: A pass ends when the directory holds nothing new, and the durable seen-store means
    #: the next one returns only what arrived since. Asking again is how the stream keeps
    #: flowing — never a replay. See `io.source.continues_across_passes`.
    continues_across_passes = True

    __slots__ = (
        "_completed",
        "_completed_set",
        "_format",
        "_fs",
        "_max_bytes",
        "_max_files",
        "_path",
        "_pending",
        "_reader_options",
        "_schema_cache",
        "_state_dir",
        "_store_obj",
        "_suffix",
    )

    def __init__(
        self,
        path: str,
        format: str,
        *,
        state_dir: str,
        suffix: str | None = None,
        max_files_per_trigger: int | None = None,
        max_bytes_per_trigger: int | None = None,
        **reader_options: Any,
    ) -> None:
        self._path = path
        self._format = format
        self._state_dir = state_dir
        self._suffix = suffix if suffix is not None else _DEFAULT_SUFFIX.get(format, "")
        # Backpressure (Spark `maxFilesPerTrigger` / Auto Loader `cloudFiles.maxFilesPerTrigger`):
        # cap the new files a single discovery pass admits so a large backlog is drained across
        # many bounded micro-batches instead of one giant epoch that blows memory. The rest stay
        # undiscovered — neither pending nor seen — so the next pass finds them again.
        if max_files_per_trigger is not None and max_files_per_trigger < 1:
            from batcher._internal.errors import PlanError

            raise PlanError(f"max_files_per_trigger must be >= 1, got {max_files_per_trigger}")
        self._max_files = max_files_per_trigger
        # The same backpressure by *size* (Spark `maxBytesPerTrigger` / Auto Loader
        # `cloudFiles.maxBytesPerTrigger`). A file count is a poor proxy for the memory a
        # micro-batch will need: ten thousand 4 KiB JSON files and three 8 GiB Parquet files
        # are both "a backlog", and only one of them fits. Capping both means a pass admits
        # at most `max_files_per_trigger` files *and* at most `max_bytes_per_trigger` of
        # them, whichever bound trips first. A single file over the budget is still admitted
        # alone — refusing it would stall the stream permanently on one large arrival.
        if max_bytes_per_trigger is not None and max_bytes_per_trigger < 1:
            from batcher._internal.errors import PlanError

            raise PlanError(f"max_bytes_per_trigger must be >= 1, got {max_bytes_per_trigger}")
        self._max_bytes = max_bytes_per_trigger
        self._fs = resolve_filesystem(path)
        self._schema_cache: pa.Schema | None = None
        # Files handed out by `discover()` whose epoch has not been published yet. They
        # are withheld from later passes (so one run does not re-read them) but are *not*
        # in the durable store, so a crash before the commit re-offers them.
        self._pending: list[str] = []
        # The subset of `_pending` whose every row has been emitted — what `confirm()`
        # promotes to the durable store. The set mirrors the list purely for membership:
        # `complete()` is called once per file per epoch, and the linear `not in` scan it
        # used made draining a large backlog quadratic in the backlog.
        self._completed: list[str] = []
        self._completed_set: set[str] = set()
        self._store_obj: SeenStore | None = None
        # Validated once, here, by building a reader against the format's own option spec.
        # A streaming query is started and then left alone, so a typo that first raises on
        # the fifth file at three in the morning is a materially worse error than the same
        # typo raising at `start()`.
        self._reader_options = dict(reader_options)
        self._reject_unknown_options()

    def _reject_unknown_options(self) -> None:
        """Fail at construction on an option the file reader will not accept.

        The reader is built against a path that need not exist: every format resolves its
        keywords in `__init__` and touches the filesystem only when read, so this validates
        the vocabulary without listing the directory (which on a watched prefix may still be
        empty).
        """
        if not self._reader_options:
            return
        SOURCES.get(self._format)(self._path, **self._reader_options)

    def _reader(self, path: str):
        """The file reader for one discovered file, carrying the caller's options.

        One place, because the three call sites — schema inference, the read, and the
        distributed split — must agree. They did not: none of them forwarded anything, so a
        `delimiter=` or `storage_options=` accepted by the public entry point reached no
        reader at all.
        """
        return SOURCES.get(self._format)(path, **self._reader_options)

    # ---- discovery --------------------------------------------------------
    def _store(self) -> SeenStore:
        """The durable seen-store, opened once and kept for the life of the source.

        This used to open a fresh SQLite connection — preceded by a `mkdirs` — on every
        `discover()` *and* every `confirm()`. A streaming query calls both once per trigger,
        so a 200ms cadence paid four filesystem round-trips a second before listing a single
        file, and the driver's idle poll (which re-lists while waiting for new files) paid
        them again. The connection is cheap to hold and the store is single-writer by design.
        """
        if self._store_obj is None:
            self._fs.mkdirs(self._state_dir, exist_ok=True)
            path = os.path.join(self._state_dir, f"{self._format}_seen.sqlite")
            self._store_obj = SeenStore(path)
        return self._store_obj

    def close(self) -> None:
        """Release the seen-store connection. Idempotent; safe on a never-opened source."""
        if self._store_obj is not None:
            store, self._store_obj = self._store_obj, None
            store.close()

    def _list_candidates(self) -> list[str]:
        """List the directory. Every listed file is a candidate; the store decides.

        This used to keep only the names sorting **after** the greatest already-seen name,
        on the theory that files arrive under monotonically increasing names. When they do
        not, that filter is silent, permanent data loss: a file whose name sorts earlier
        than one already processed is never offered to `unseen()` and so is never ingested,
        with no error and nothing in the store to show it was skipped. The names real
        writers produce are exactly the wrong shape for it — Spark and Flink emit
        ``part-00000-<uuid>.parquet``, so after the first file roughly half of every later
        arrival sorted below the maximum and vanished.

        Dropping the filter costs almost nothing now that `unseen` is an index probe over
        the candidates rather than a scan of the whole table: the listing itself is
        unchanged, and the probe is O(candidates), not O(files ever seen).
        """
        try:
            return self._fs.expand(self._path, suffix=self._suffix)
        except IOError:
            return []  # empty / not-yet-populated directory is not an error here.

    def discover(self) -> list[str]:
        """Return the new (unseen, not-yet-pending) files for the current pass.

        Records them as *pending* — withheld from subsequent passes in this run, but not
        yet durable. `confirm()` makes them durable once their epoch is published; until
        then a restart re-offers them (see the module docstring).
        """
        store = self._store()
        candidates = self._list_candidates()
        held = set(self._pending)
        new_files = [f for f in store.unseen(candidates) if f not in held]
        if self._max_files is not None:
            # `candidates` comes back sorted, so capping the head drains the backlog in a
            # stable, oldest-name-first order; the tail is left undiscovered for a later pass.
            new_files = new_files[: self._max_files]
        if self._max_bytes is not None:
            new_files = self._within_byte_budget(new_files)
        self._pending.extend(new_files)
        return new_files

    def _within_byte_budget(self, files: list[str]) -> list[str]:
        """The head of `files` whose total size fits `max_bytes_per_trigger`.

        The first file is always admitted, even when it alone exceeds the budget: a stream
        that refuses to make progress on a large arrival is stalled rather than throttled,
        and the next pass would refuse it again forever.
        """
        admitted: list[str] = []
        used = 0
        for path in files:
            size = _safe_size(self._fs, path)
            if admitted and used + size > self._max_bytes:
                break
            admitted.append(path)
            used += size
        return admitted

    def complete(self, files: list[str]) -> None:
        """Mark `files` fully read — eligible to be confirmed once their epoch publishes.

        A file is only safe to remember when *every* row of it has been emitted. Confirming
        at discovery (or part-way through a large file) is what loses data: the store would
        claim rows the sink never saw.
        """
        for f in files:
            if f not in self._completed_set:
                self._completed_set.add(f)
                self._completed.append(f)

    def confirm(self) -> None:
        """Durably mark the fully-read files as seen — their epoch is now published."""
        if not self._completed:
            return
        store = self._store()
        store.mark_many([(f, _safe_size(self._fs, f), _safe_mtime(f)) for f in self._completed])
        done = self._completed_set
        self._pending = [f for f in self._pending if f not in done]
        self._completed = []
        self._completed_set = set()

    def snapshot_position(self) -> dict:
        """The files fully read but not yet durably confirmed (the write-ahead position)."""
        return {"pending": list(self._completed)}

    def seek(self, position: dict) -> None:
        """Resume from a checkpointed position: its files were published, so confirm them.

        Recovery restores the position of the last *committed* epoch. Those files reached
        the sink, so they are durably seen — anything the previous run had merely
        discovered stays unconfirmed and is picked up again.
        """
        self._completed = [str(f) for f in position.get("pending", [])]
        self._completed_set = set(self._completed)
        self.confirm()
        self._pending.clear()

    # ---- Source protocol --------------------------------------------------
    def schema(self) -> pa.Schema:
        """The file format's schema, sampled from the first available file.

        An empty directory is reported as the "no files yet" condition it is. Some
        filesystems raise from `expand` when nothing matches and some return an empty list;
        only the first was handled, so the other spelling reached `files[0]` and surfaced as
        a bare `IndexError` with no mention of the path or the suffix.
        """
        if self._schema_cache is None:
            try:
                files = self._fs.expand(self._path, suffix=self._suffix)
            except IOError as exc:
                raise IOError(
                    f"cannot infer schema: no {self._suffix} files yet under {self._path!r}"
                ) from exc
            if not files:
                raise IOError(
                    f"cannot infer schema: no {self._suffix} files yet under {self._path!r}"
                )
            self._schema_cache = self._reader(files[0]).schema()
        return self._schema_cache

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return list(self.iter_batches(projection))

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Run one discovery pass; yield the new files' batches via the file reader.

        A file is marked *complete* only after its final batch has been yielded, so a
        `confirm()` between micro-batches can never promote a file whose rows are still
        in flight.
        """
        for path in self.discover():
            yield from self._reader(path).iter_batches(projection)
            self.complete([path])
        # Reaching here means the consumer drained the pass: under the streaming engine
        # every batch yielded has been published (a publish precedes the next pull), and a
        # direct reader has all the rows in hand. Either way the pass is durable. A crash
        # part-way never gets here, which is exactly why the files stay replayable.
        self.confirm()

    def row_count(self) -> int | None:
        return None  # the watched directory grows over time.

    def identity(self) -> str:
        return f"files_incremental:{self._format}:{self._path}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        """One :class:`FileSplit` per new file (locator-only, picklable).

        The distributed streaming path fans these across the cluster, one worker per file,
        and publishes the whole pass as a single transaction — so the epoch's files are
        confirmed together or not at all.
        """
        # The options ride the split for the reason `FileSplit` carries them everywhere
        # else: a worker rebuilds the reader from the split alone, so anything left out
        # reverts to its default on the cluster and nowhere else.
        return [
            FileSplit(self._format, path, dict(self._reader_options)) for path in self.discover()
        ]


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
