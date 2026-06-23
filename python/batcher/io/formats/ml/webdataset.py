"""WebDataset format — `.tar` shard reader via stdlib `tarfile` (core, no extra).

WebDataset stores training samples as plain POSIX tar archives: files sharing a
basename form one sample, and the file extension names the field. ``a/b.jpg`` and
``a/b.json`` thus become one row ``{__key__: "a/b", jpg: <bytes>, json: <bytes>}``.
`WebDatasetSource` reads each shard with the stdlib `tarfile` (no dependency),
grouping members by key into Arrow rows whose value columns are ``binary``. One
tar shard is one `Split`.
"""

from __future__ import annotations

import os
import tarfile
from typing import IO, Any

import pyarrow as pa

from batcher.io.base import FileSource
from batcher.io.formats.base import SOURCES

__all__ = ["WebDatasetSource"]


def _split_key_ext(member_name: str) -> tuple[str, str]:
    """Split ``dir/name.ext`` into ``(dir/name, ext)`` (WebDataset convention).

    The first dot in the basename separates the sample key from the extension, so
    ``a/b.tar.gz`` keys as ``a/b`` with extension ``tar.gz``.
    """
    directory, base = os.path.split(member_name)
    key, _, ext = base.partition(".")
    return (f"{directory}/{key}" if directory else key), ext


def _read_shard(fh: IO[bytes]) -> tuple[list[str], dict[str, list[bytes | None]]]:
    """Group a tar shard's members into ``__key__`` rows and ``<ext>`` columns."""
    samples: dict[str, dict[str, bytes]] = {}
    order: list[str] = []
    extensions: list[str] = []
    with tarfile.open(fileobj=fh, mode="r|*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            key, ext = _split_key_ext(member.name)
            if key not in samples:
                samples[key] = {}
                order.append(key)
            if ext not in extensions:
                extensions.append(ext)
            payload = tar.extractfile(member)
            samples[key][ext] = payload.read() if payload is not None else b""
    columns: dict[str, list[bytes | None]] = {ext: [] for ext in extensions}
    for key in order:
        for ext in extensions:
            columns[ext].append(samples[key].get(ext))
    return order, columns


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
        return self._read_file(fh, None)[0].schema

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        keys, columns = _read_shard(fh)
        data: dict[str, Any] = {"__key__": keys}
        data.update(columns)
        batch = pa.RecordBatch.from_pydict(data)
        return [batch.select(projection) if projection is not None else batch]
