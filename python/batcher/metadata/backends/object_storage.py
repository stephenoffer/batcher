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
import json
from collections.abc import Iterator

from batcher.metadata.store import Key

__all__ = ["ObjectStorageBackend"]


def _encode_name(key: Key) -> str:
    """A URL-safe object basename for `key` (reversible by `_decode_name`)."""
    raw = json.dumps(list(key), separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_name(name: str) -> Key:
    raw = base64.urlsafe_b64decode(name.encode("ascii"))
    return tuple(json.loads(raw))


class ObjectStorageBackend:
    """A `MetadataBackend` storing each entry as one object under an fsspec root URI."""

    def __init__(self, uri: str | None) -> None:
        if not uri:
            raise ValueError(
                "object_storage metadata backend requires a uri (e.g. s3://bucket/prefix)"
            )
        import fsspec

        self._fs, self._root = fsspec.core.url_to_fs(uri)
        self._root = self._root.rstrip("/")

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
        directory = self._dir(table)
        try:
            names = self._fs.ls(directory, detail=False)
        except FileNotFoundError:
            return
        plen = len(prefix)
        for path in names:
            base = path.rstrip("/").rsplit("/", 1)[-1]
            try:
                key = _decode_name(base)
            except (ValueError, json.JSONDecodeError):
                continue  # a stray non-batcher object; skip rather than fail planning
            if key[:plen] == prefix:
                yield key, self._fs.cat_file(path)

    def batch_put(self, table: str, items: list[tuple[Key, bytes]]) -> None:
        if not items:
            return
        self._fs.makedirs(self._dir(table), exist_ok=True)
        for key, value in items:
            self._fs.pipe_file(self._path(table, key), value)
