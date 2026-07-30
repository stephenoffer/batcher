"""Object-storage backend — durable, cluster-shared learned statistics.

Every `(table, key) -> value` is one object under a root URI (`file://`, `s3://`,
`gs://`, …) addressed through `fsspec`, so learned cardinalities and cost
calibration written by one driver are readable by every other driver on the cluster
— the moat compounds across jobs instead of resetting per process. Keys round-trip
through a URL-safe base64 of their JSON encoding, so `scan` can recover them from the
object names. One object per key keeps concurrent writers from clobbering each other
(the per-key write granularity the Hub's keyed-param model relies on).
"""

from __future__ import annotations

import base64
import contextlib
import json
from collections.abc import Iterator

from batcher._internal.errors import MissingDependencyError
from batcher.metadata.store import Key, decode_key, encode_key, require_uri

__all__ = ["ObjectStorageBackend"]

# Objects fetched per concurrent `cat`. Large enough that the round-trip latency of a scan is
# amortized across a batch, small enough that a scan of a large table does not hold the whole
# table in memory before yielding its first row.
_FETCH_CHUNK = 512


def _encode_name(key: Key) -> str:
    """A URL-safe object basename for `key` (reversible by `_decode_name`)."""
    raw = encode_key(key).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_name(name: str) -> Key:
    raw = base64.urlsafe_b64decode(name.encode("ascii"))
    return decode_key(raw.decode("utf-8"))


class ObjectStorageBackend:
    """A `MetadataBackend` storing each entry as one object under an fsspec root URI."""

    def __init__(self, uri: str | None) -> None:
        """Address a metadata store rooted at an fsspec URI.

        Args:
            uri: The root, e.g. ``s3://bucket/prefix`` or ``file:///var/batcher``.

        Raises:
            ConfigError: If `uri` is missing or empty.
            MissingDependencyError: If `fsspec` is not installed.
        """
        uri = require_uri("object_storage", uri, example="s3://bucket/prefix")
        try:
            import fsspec
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise MissingDependencyError.of(
                feature="The object_storage metadata backend", provides="fsspec", extra="cloud"
            ) from exc

        self._uri = uri
        self._fs, self._root = fsspec.core.url_to_fs(uri)
        self._root = self._root.rstrip("/")

    def __repr__(self) -> str:
        """Name the root URI — the one thing worth seeing when a shared store looks empty."""
        return f"ObjectStorageBackend(uri={self._uri!r})"

    def _dir(self, table: str) -> str:
        return f"{self._root}/{table}"

    def _path(self, table: str, key: Key) -> str:
        return f"{self._dir(table)}/{_encode_name(key)}"

    def get(self, table: str, key: Key) -> bytes | None:
        try:
            return self._fs.cat_file(self._path(table, key))
        except FileNotFoundError:
            return None

    def put(self, table: str, key: Key, value: bytes) -> None:
        path = self._path(table, key)
        self._fs.makedirs(self._dir(table), exist_ok=True)
        self._fs.pipe_file(path, value)

    def scan(self, table: str, prefix: Key = ()) -> Iterator[tuple[Key, bytes]]:
        """Every `(key, value)` under `prefix`, fetched in concurrent batches.

        One object per key is what gives this store its per-key write granularity, and it is
        also what makes a naive scan pathological: the feedback table holds tens of thousands
        of entries, so reading it one `cat_file` at a time is tens of thousands of *sequential*
        round-trips to object storage before the first plan of a process. Against S3 at a
        realistic 30 ms per GET that is a cold start measured in minutes.

        `fs.cat` issues a batch concurrently, and the batching is chunked rather than
        whole-table so memory stays bounded and the caller starts receiving rows before the
        last object has landed. Objects that vanish mid-scan — another driver pruning the
        feedback table at the same time — are omitted rather than raised: this is a read of
        learned statistics, and one missing row is not worth failing a query over.
        """
        directory = self._dir(table)
        try:
            names = self._fs.ls(directory, detail=False)
        except FileNotFoundError:
            return
        plen = len(prefix)
        wanted: list[tuple[str, Key]] = []
        for path in names:
            base = path.rstrip("/").rsplit("/", 1)[-1]
            try:
                key = _decode_name(base)
            except (ValueError, json.JSONDecodeError):
                continue  # a stray non-batcher object; skip rather than fail planning
            if key[:plen] == prefix:
                wanted.append((path, key))
        for start in range(0, len(wanted), _FETCH_CHUNK):
            chunk = wanted[start : start + _FETCH_CHUNK]
            blobs = self._cat([path for path, _key in chunk])
            for path, key in chunk:
                value = blobs.get(path)
                if isinstance(value, bytes):
                    yield key, value

    def _cat(self, paths: list[str]) -> dict[str, object]:
        """`{path: bytes}` for `paths`, concurrently, tolerating objects that have gone.

        `on_error="omit"` is the desired semantics but not universally implemented, so a
        filesystem that rejects it falls back to one request per object rather than failing.
        """
        try:
            return dict(self._fs.cat(paths, on_error="omit"))
        except TypeError:  # pragma: no cover - a filesystem with a narrower `cat`
            out: dict[str, object] = {}
            for path in paths:
                try:
                    out[path] = self._fs.cat_file(path)
                except FileNotFoundError:
                    continue
            return out

    def batch_put(self, table: str, items: list[tuple[Key, bytes]]) -> None:
        if not items:
            return
        self._fs.makedirs(self._dir(table), exist_ok=True)
        mapping = {self._path(table, key): value for key, value in items}
        try:
            self._fs.pipe(mapping)  # concurrent where the filesystem is async
        except (TypeError, NotImplementedError):  # pragma: no cover - a narrower filesystem
            for path, value in mapping.items():
                self._fs.pipe_file(path, value)

    def delete(self, table: str, keys: list[Key]) -> None:
        """Drop `keys` from `table`; absent objects are ignored.

        Optional beyond the four `MetadataBackend` methods. Offered because this is the store
        a whole cluster shares, so it accumulates feedback fastest — and the hub's prune, the
        one thing that keeps a scan of it from growing without bound, is a no-op against a
        backend that cannot delete.
        """
        if not keys:
            return
        paths = [self._path(table, key) for key in keys]
        # Already pruned by another driver is the expected concurrent outcome, not an error.
        with contextlib.suppress(FileNotFoundError):
            self._fs.rm(paths)
