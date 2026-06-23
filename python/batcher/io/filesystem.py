"""Filesystem resolution for IO sources and sinks — one cloud-agnostic backend.

Every file listing, glob, open, size, mkdir, and delete in Batcher goes through a
single `pyarrow.fs`-backed façade, so the *same* code path serves local disk, NFS /
on-prem mounts, S3 (incl. on-prem S3 like MinIO / Ceph via ``endpoint_override``),
GCS, Azure, and HDFS. The scheme is parsed from the path; `pyarrow.fs.FileSystem
.from_uri` constructs the right backend (reading credentials/region/endpoint from the
URI query string or the standard environment), and anything `pyarrow.fs` does not
support natively falls back to an fsspec backend wrapped behind the *same*
`pyarrow.fs` interface — so there is exactly one filesystem abstraction to reason
about. The façade exposes only what the IO bases need; the handles `open` returns are
accepted by every pyarrow reader.

On-prem / self-hosted object stores work without code changes — point at your
endpoint, e.g.
``read("s3://bucket/data/*.parquet?endpoint_override=https://minio.internal:9000")``
or set ``AWS_ENDPOINT_URL`` (and HDFS via ``hdfs://namenode:8020/path``).
"""

from __future__ import annotations

import contextlib
import fnmatch
import io
import os
import posixpath
import uuid
from collections.abc import Iterator
from typing import IO, Any, Protocol, runtime_checkable

import pyarrow as pa
import pyarrow.fs as pafs

from batcher._internal.errors import IOError

__all__ = ["FileSystem", "LocalFileSystem", "resolve_filesystem"]

# Object stores where a single PUT is already atomic (no partial-read visibility),
# so a write goes straight to the destination — a temp-then-rename would only add a
# full-object server-side copy with no atomicity gain. Everything else (local, NFS,
# HDFS) gets temp-write-then-rename so a crash never leaves a truncated file.
_OBJECT_STORE_SCHEMES = frozenset(
    {"s3", "s3a", "gs", "gcs", "abfs", "abfss", "az", "azure", "wasb", "wasbs"}
)
# Cloud scheme aliases → the canonical scheme `from_uri` / fsspec understand.
_SCHEME_ALIASES = {"s3a": "s3", "gcs": "gs", "abfss": "abfs", "wasbs": "wasb"}


def _scheme(path: str) -> str:
    """The URI scheme of `path` (``""`` for a bare local path)."""
    idx = path.find("://")
    return path[:idx].lower() if idx > 0 else ""


def _is_data_file(path: str) -> bool:
    """Whether `path`'s basename is a data file rather than a metadata/marker file.

    Files whose basename starts with ``_`` or ``.`` are skipped — ``_SUCCESS``,
    ``_metadata``, ``_committed_*``, ``.crc``, ``.DS_Store``, and Spark temp files.
    This is the Spark/Hive/Hadoop convention and fixes reading those marker files as
    data when a directory or glob is expanded (Ray Data's ray#57704 / ray#61373:
    ``read_parquet`` choking on ``_SUCCESS``/``.crc`` next to the real files)."""
    base = os.path.basename(path.rstrip("/"))
    return bool(base) and not base.startswith(("_", "."))


@runtime_checkable
class FileSystem(Protocol):
    """The minimal filesystem surface the IO bases depend on."""

    def expand(self, path: str, *, suffix: str) -> list[str]:
        """Resolve a file, directory, or glob into a sorted list of file paths."""
        ...

    def open(self, path: str, mode: str = "rb") -> IO[Any]:
        """Open a single file for reading; the handle is accepted by pyarrow."""
        ...

    def atomic_writer(self, path: str) -> contextlib.AbstractContextManager[IO[Any]]:
        """A context manager yielding a write handle that becomes visible at `path`
        only on clean exit. A crash/exception mid-write leaves any prior file at
        `path` intact (no truncated/half-written output) — closing Ray Data's
        ``write_parquet`` overwrite data-loss (ray#62019)."""
        ...

    def size(self, path: str) -> int:
        """The size of `path` in bytes."""
        ...

    def exists(self, path: str) -> bool:
        """Whether a file already exists at `path`. With atomic writes, a file is
        present only if a prior write fully committed it — so this is the
        skip-if-done test for resumable writes."""
        ...

    def mkdirs(self, path: str, *, exist_ok: bool = True) -> None:
        """Create a directory and any missing parents."""
        ...

    def list_dirs(self, path: str) -> list[str]:
        """The immediate subdirectories of `path` (one cheap, non-recursive list).

        Used to distribute directory-tree listing: the driver enumerates top-level
        partition dirs, and each worker lists only its own subtree.
        """
        ...

    def remove(self, path: str) -> None:
        """Delete a single file (no error if it is already absent). Used to clear the
        stale part-files left behind when compacting a multi-file output in place."""
        ...


class _ArrowFileSystem:
    """A `pyarrow.fs.FileSystem` behind the small façade the IO bases use.

    `prefix` is the ``scheme://authority`` portion that pyarrow strips from a URI to
    get an in-filesystem path (``""`` for local); it is removed on the way in and
    re-attached on the way out, so callers always see full paths/URIs while pyarrow
    sees its bucket-relative ones.
    """

    __slots__ = ("_atomic_rename", "_fs", "_prefix")

    def __init__(self, fs: pafs.FileSystem, prefix: str, *, atomic_rename: bool) -> None:
        self._fs = fs
        self._prefix = prefix
        self._atomic_rename = atomic_rename

    # ---- path <-> URI mapping ---------------------------------------------
    def _p(self, path: str) -> str:
        """A full path/URI → the in-filesystem path pyarrow expects.

        Any ``?query`` (e.g. ``endpoint_override``) is dropped — it is configuration
        already baked into `self._fs`, not part of the object path."""
        p = path.split("?", 1)[0]
        if self._prefix and p.startswith(self._prefix):
            return p[len(self._prefix) :]
        return p

    def _uri(self, in_path: str) -> str:
        """An in-filesystem path → the full path/URI callers see."""
        return f"{self._prefix}{in_path}" if self._prefix else in_path

    # ---- shared surface ----------------------------------------------------
    def expand(self, path: str, *, suffix: str) -> list[str]:
        if any(ch in path for ch in "*?["):
            return self._glob(path)
        in_path = self._p(path)
        info = self._fs.get_file_info(in_path)
        if info.type == pafs.FileType.Directory:
            sel = pafs.FileSelector(in_path, recursive=False)
            files = sorted(
                fi.path
                for fi in self._fs.get_file_info(sel)
                if fi.type == pafs.FileType.File
                and fi.path.endswith(suffix)
                and _is_data_file(fi.path)
            )
            if not files:
                raise IOError(f"no {suffix} files found in directory {path!r}")
            return [self._uri(f) for f in files]
        if info.type == pafs.FileType.NotFound:
            raise IOError(f"path {path!r} does not exist")
        return [path]

    def _glob(self, pattern: str) -> list[str]:
        in_pat = self._p(pattern)
        # The directory portion before the first wildcard is the listing root.
        base = in_pat
        for i, ch in enumerate(in_pat):
            if ch in "*?[":
                base = posixpath.dirname(in_pat[:i])
                break
        recursive = "**" in in_pat
        sel = pafs.FileSelector(base or ".", recursive=recursive, allow_not_found=True)
        matches = sorted(
            fi.path
            for fi in self._fs.get_file_info(sel)
            if fi.type == pafs.FileType.File
            and _is_data_file(fi.path)
            and fnmatch.fnmatch(fi.path, in_pat)
        )
        if not matches:
            raise IOError(f"glob {pattern!r} matched no files")
        return [self._uri(m) for m in matches]

    def open(self, path: str, mode: str = "rb") -> IO[Any]:  # noqa: ARG002 (read-only façade)
        # A buffered wrapper over the pyarrow input file gives the full Python file
        # protocol (read/readline/seek) the byte-range split readers rely on, while
        # staying acceptable to every pyarrow reader.
        return io.BufferedReader(self._fs.open_input_file(self._p(path)))  # type: ignore[arg-type]

    @contextlib.contextmanager
    def atomic_writer(self, path: str) -> Iterator[IO[Any]]:
        dest = self._p(path)
        # Ensure the parent directory exists (pyarrow's output stream does not create
        # it). Cheap and idempotent; a no-op marker on object stores.
        parent = posixpath.dirname(dest)
        if parent:
            self._fs.create_dir(parent, recursive=True)
        if not self._atomic_rename:
            # Object store: a single PUT is atomic — write straight to the destination.
            with self._fs.open_output_stream(dest) as fh:
                yield fh
            return
        # Local / HDFS: write a unique temp sibling, then atomically rename into place;
        # on any error drop the temp so the prior file at `path` is never touched.
        tmp = f"{dest}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        try:
            with self._fs.open_output_stream(tmp) as fh:
                yield fh
            self._fs.move(tmp, dest)
        except BaseException:
            with contextlib.suppress(Exception):
                self._fs.delete_file(tmp)
            raise

    def size(self, path: str) -> int:
        return self._fs.get_file_info(self._p(path)).size or 0

    def exists(self, path: str) -> bool:
        return self._fs.get_file_info(self._p(path)).type != pafs.FileType.NotFound

    def mkdirs(self, path: str, *, exist_ok: bool = True) -> None:  # noqa: ARG002 (parity)
        # pyarrow `create_dir(recursive=True)` is already exist-ok; `exist_ok` is kept
        # for interface parity.
        self._fs.create_dir(self._p(path), recursive=True)

    def list_dirs(self, path: str) -> list[str]:
        sel = pafs.FileSelector(self._p(path), recursive=False, allow_not_found=True)
        dirs = sorted(
            fi.path for fi in self._fs.get_file_info(sel) if fi.type == pafs.FileType.Directory
        )
        return [self._uri(d) for d in dirs]

    def remove(self, path: str) -> None:
        in_path = self._p(path)
        if self._fs.get_file_info(in_path).type != pafs.FileType.NotFound:
            with contextlib.suppress(FileNotFoundError):
                self._fs.delete_file(in_path)


class LocalFileSystem(_ArrowFileSystem):
    """The local filesystem (``pyarrow.fs.LocalFileSystem``), kept as a named type for
    callers/tests that construct it directly."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(pafs.LocalFileSystem(), "", atomic_rename=True)


def _local_prefix(path: str) -> str:
    """The ``file://`` prefix to strip for a local path (``""`` for a bare path)."""
    return "file://" if path.startswith("file://") else ""


def resolve_filesystem(path: str) -> FileSystem:
    """Return the `pyarrow.fs`-backed façade for `path`, dispatching on its scheme.

    Local and ``file://`` paths use the local filesystem; ``s3``/``gs``/``hdfs``/
    ``abfs``/… are constructed by `pyarrow.fs.FileSystem.from_uri` (credentials,
    region, and on-prem ``endpoint_override`` come from the URI query string or the
    environment); an unknown scheme falls back to an fsspec backend wrapped behind the
    same `pyarrow.fs` interface, so third-party backends work with no code change.
    """
    scheme = _scheme(path)
    if scheme in ("", "file"):
        prefix = _local_prefix(path)
        return _ArrowFileSystem(pafs.LocalFileSystem(), prefix, atomic_rename=True)
    try:
        fs, in_path = pafs.FileSystem.from_uri(path)
    except (ValueError, OSError, pa.ArrowInvalid, pa.ArrowNotImplementedError):
        # A scheme pyarrow.fs doesn't implement natively → fsspec fallback.
        return _fsspec_backed(scheme, path)
    # The prefix is `scheme://authority`; compute it from the path with any `?query`
    # (config like endpoint_override) removed, since pyarrow's in_path excludes both.
    base = path.split("?", 1)[0]
    prefix = base[: len(base) - len(in_path)]
    canonical = _SCHEME_ALIASES.get(scheme, scheme)
    return _ArrowFileSystem(fs, prefix, atomic_rename=canonical not in _OBJECT_STORE_SCHEMES)


def _fsspec_backed(scheme: str, path: str) -> FileSystem:
    """Wrap an fsspec backend behind the `pyarrow.fs` interface (the escape hatch for
    schemes pyarrow does not implement natively)."""
    try:
        import fsspec
        from pyarrow.fs import FSSpecHandler, PyFileSystem
    except ImportError as exc:
        raise IOError(
            f"reading {scheme}:// paths needs the cloud extra: pip install 'batcher[cloud]'"
        ) from exc
    protocol = _SCHEME_ALIASES.get(scheme, scheme)
    fs = PyFileSystem(FSSpecHandler(fsspec.filesystem(protocol)))
    in_path = path[len(scheme) + 3 :]  # strip "scheme://"
    prefix = path[: len(path) - len(in_path)]
    return _ArrowFileSystem(fs, prefix, atomic_rename=protocol not in _OBJECT_STORE_SCHEMES)
