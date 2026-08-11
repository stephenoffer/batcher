"""WebDataset format — `.tar` shard reader via stdlib `tarfile` (core, no extra).

WebDataset stores training samples as plain POSIX tar archives: files sharing a
basename form one sample, and the file extension names the field. ``a/b.jpg`` and
``a/b.json`` thus become one row ``{__key__: "a/b", jpg: <bytes>, json: <bytes>}``.
`WebDatasetSource` reads each shard with the stdlib `tarfile` (no dependency),
grouping members by key into Arrow rows whose value columns are ``binary``. One
tar shard is one `Split`, and each shard streams a morsel at a time rather than becoming
resident whole.
"""

from __future__ import annotations

import os
import tarfile
from collections.abc import Iterator
from typing import IO, Any

import pyarrow as pa

from batcher.config import active_config
from batcher.io.base import FileSource
from batcher.io.formats.base import SOURCES

__all__ = ["WebDatasetSource"]

# Payload bytes a batch may accumulate before it is flushed, *in addition* to the row-count
# morsel. Both bounds are needed and the byte one is the load-bearing half: a WebDataset row
# is an entire image, audio clip or video, so the 16,384-row morsel is 1 GiB of 64 KiB JPEGs
# and bounds nothing that matters. This is the same reasoning `base/source.py` applies to its
# read-ahead window ("one row can itself be a 200 MB video") and `binary.py` to its file
# batches — a count-only ceiling does not bound memory when rows are not narrow.
_BATCH_PAYLOAD_BYTES = max(
    1 << 20, int(os.environ.get("BATCHER_WEBDATASET_BATCH_BYTES", str(64 << 20)))
)


def _split_key_ext(member_name: str) -> tuple[str, str]:
    """Split ``dir/name.ext`` into ``(dir/name, ext)`` (WebDataset convention).

    The first dot in the basename separates the sample key from the extension, so
    ``a/b.tar.gz`` keys as ``a/b`` with extension ``tar.gz``.
    """
    directory, base = os.path.split(member_name)
    key, _, ext = base.partition(".")
    return (f"{directory}/{key}" if directory else key), ext


def _shard_extensions(fh: IO[bytes]) -> list[str]:
    """The distinct member extensions of a tar shard, in first-seen order.

    Reads only the tar's member *headers* — never a member's payload — so it is what
    `schema()` uses: the column set (``__key__`` plus one column per extension) is fully
    determined by the member names, and pulling every sample's bytes just to learn it
    is exactly the "materialize the file to read its schema" bug. For a shard of images
    that is the whole shard resident to answer a metadata question.

    Opened **seekably** (``"r"``) rather than in stream mode (``"r|*"``) when the handle
    allows it. Both read headers only, but a stream-mode reader cannot skip: to reach the
    next header it must read through the payload in between, so "headers only" still moved
    every byte of the shard. Seeking over them measured 482 ms against 1,116 ms on a 268 MB
    shard. A handle that cannot seek — a non-seekable object-store stream — falls back to
    the stream reader and behaves exactly as before.
    """
    try:
        with tarfile.open(fileobj=fh, mode="r") as tar:
            return _extensions_of(tar)
    except (tarfile.TarError, OSError, ValueError, AttributeError):
        fh.seek(0)  # not seekable, or not a plain tar — read it as a stream instead
    with tarfile.open(fileobj=fh, mode="r|*") as tar:
        return _extensions_of(tar)


def _extensions_of(tar: tarfile.TarFile) -> list[str]:
    """The distinct member extensions of an open tar, in first-seen order."""
    extensions: list[str] = []
    for member in tar:
        if not member.isfile():
            continue
        _key, ext = _split_key_ext(member.name)
        if ext not in extensions:
            extensions.append(ext)
    return extensions


def _shard_schema(extensions: list[str]) -> pa.Schema:
    """The Arrow schema of a WebDataset shard: ``__key__`` (string) + one binary per ext."""
    return pa.schema([("__key__", pa.string()), *((ext, pa.binary()) for ext in extensions)])


def _iter_shard_batches(
    fh: IO[bytes], schema: pa.Schema, batch_rows: int, path: str
) -> Iterator[pa.RecordBatch]:
    """Yield a shard's samples as batches of `batch_rows`, holding only one batch at a time.

    This used to accumulate the whole shard — every sample's payload as a Python `bytes`,
    in a dict keyed by sample, then transposed into per-extension lists, then built into a
    single `RecordBatch` — and hand back that one batch. A WebDataset shard is 100 MB to a
    few GB of images or audio by convention, and it is the input a training loader streams,
    so that was the worst possible shape: on a 268 MB shard of 4,000 samples the first row
    reached the caller after **1.84 s**, in **one** batch, having peaked at **587 MB** of
    resident memory — 2.2x the shard, because every payload existed as a Python object and
    again as Arrow.

    Streaming works because the format is designed for it: WebDataset requires a sample's
    members to be **consecutive** in the tar, which is what lets a reader emit a sample as
    soon as the next key appears. A key that reappears after its sample was emitted would be
    silently split across two rows, so it raises instead — see `_key_reappeared`.

    Every batch is built against `schema`, the source's own, rather than against whichever
    extensions this shard happened to contain. Per-shard schemas are how a two-shard read
    ends up with batches that cannot concatenate.
    """
    extensions = [name for name in schema.names if name != "__key__"]
    keys: list[str] = []
    columns: dict[str, list[bytes | None]] = {ext: [] for ext in extensions}
    seen: set[str] = set()
    current_key: str | None = None
    current: dict[str, bytes] = {}
    emitted = False
    held = 0

    def close_sample() -> int:
        keys.append(current_key)  # type: ignore[arg-type]
        for ext in extensions:
            columns[ext].append(current.get(ext))
        return sum(len(v) for v in current.values())

    def build() -> pa.RecordBatch:
        return pa.RecordBatch.from_pydict({"__key__": keys, **columns}, schema=schema)

    with tarfile.open(fileobj=fh, mode="r|*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            key, ext = _split_key_ext(member.name)
            if key != current_key:
                if current_key is not None:
                    held += close_sample()
                    if len(keys) >= batch_rows or held >= _BATCH_PAYLOAD_BYTES:
                        yield build()
                        emitted = True
                        keys = []
                        columns = {e: [] for e in extensions}
                        held = 0
                if key in seen:
                    raise _key_reappeared(path, key)
                seen.add(key)
                current_key, current = key, {}
            payload = tar.extractfile(member)
            current[ext] = payload.read() if payload is not None else b""
        if current_key is not None:
            close_sample()
    if keys or not emitted:
        # An empty shard still yields one empty batch, so the caller has a schema to build
        # an empty result from rather than nothing at all.
        yield build()


def _key_reappeared(path: str, key: str) -> Exception:
    """The error for a shard whose sample members are not consecutive."""
    from batcher._internal.errors import FormatError

    return FormatError(
        f"{path!r} is not a valid WebDataset shard: the members of sample {key!r} are not "
        f"consecutive in the tar, so the sample would be split across two rows. WebDataset "
        f"requires a sample's files to be stored adjacently; rewrite the shard with a "
        f"conforming writer (for example `webdataset.ShardWriter`)."
    )


@SOURCES.register("webdataset")
class WebDatasetSource(FileSource):
    """One or more WebDataset ``.tar`` shards (single file, directory, or glob).

    Each shard yields rows ``{__key__: str, <ext>: binary, ...}`` — files sharing
    a basename are one sample, the extension is the column name.
    """

    suffix = ".tar"
    format_name = "webdataset"

    __slots__ = ()

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:
        # Determined by member names alone — do not extract any payload (see `_shard_extensions`).
        return _shard_schema(_shard_extensions(fh))

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        """Every batch of one shard, materialized — the `read()` contract."""
        return list(self._shard_batches(fh, projection, self._path))

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        """Stream one shard a morsel at a time rather than decoding it whole.

        Without this the base falls back to `_read_file`, which returns a list — so
        `iter_batches` held the entire shard before its first batch reached the consumer.
        A WebDataset shard is the archetypal larger-than-memory training input; see
        `_iter_shard_batches`.

        Args:
            path: The shard to stream.
            projection: Columns the scan must produce. All columns when omitted.

        Yields:
            One `RecordBatch` per morsel of samples, in shard order.
        """
        with self._open(path) as fh:
            yield from self._shard_batches(fh, projection, path)

    def _shard_batches(
        self, fh: IO[Any], projection: list[str] | None, path: str
    ) -> Iterator[pa.RecordBatch]:
        """The one decoding path both `read()` and `iter_batches()` go through.

        Batches are built against the *source's* schema — read from the shard's member names
        alone — so an extension missing from one shard still yields a typed null column
        rather than a batch that cannot concatenate with its neighbours.
        """
        schema = self.schema()
        rows = active_config().execution.morsel_rows
        for batch in _iter_shard_batches(fh, schema, rows, path):
            yield batch.select(projection) if projection is not None else batch
