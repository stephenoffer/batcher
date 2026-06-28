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
import hashlib
import io
import os
import posixpath
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterator
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

    __slots__ = ("_atomic_rename", "_cacheable", "_fs", "_prefix", "_strip_query")

    def __init__(
        self,
        fs: pafs.FileSystem,
        prefix: str,
        *,
        atomic_rename: bool,
        strip_query: bool = True,
        cacheable: bool = False,
    ) -> None:
        self._fs = fs
        self._prefix = prefix
        self._atomic_rename = atomic_rename
        # Remote object-store reads may be served from a local-SSD read-through cache
        # (`FileBytesCache`, below) when one is configured; local backends never cache
        # (the bytes are already local).
        self._cacheable = cacheable
        # Native backends carry config in the URI query (e.g. ``?endpoint_override=``),
        # which pyarrow has already consumed — so it is dropped from the object path.
        # fsspec-backed URLs (e.g. presigned ``https://…?signature=…``) keep it: the
        # query IS part of the addressable object there.
        self._strip_query = strip_query

    # ---- path <-> URI mapping ---------------------------------------------
    def _p(self, path: str) -> str:
        """A full path/URI → the in-filesystem path the backend expects."""
        p = path.split("?", 1)[0] if self._strip_query else path
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
        # A trailing slash on an object-store directory (``s3://bucket/dir/``) makes
        # pyarrow's `get_file_info` return `NotFound` — object stores have no real
        # directories, so the key ``dir/`` does not exist as an object. Strip it (but
        # never the lone root ``/``) so a directory URI written either way resolves to
        # the same listing. Harmless on local/`file://` paths (a dir resolves the same
        # with or without the slash).
        in_path = self._p(path)
        if len(in_path) > 1:
            in_path = in_path.rstrip("/")
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
        in_path = self._p(path)
        local = self._cached_local(in_path)
        if local is not None:
            return io.BufferedReader(open(local, "rb"))
        return io.BufferedReader(self._fs.open_input_file(in_path))  # type: ignore[arg-type]

    def _cached_local(self, in_path: str) -> str | None:
        """The local-cache copy of a remote file, fetching it on a miss; `None` when
        caching is off or unavailable. Best-effort — any failure falls back to a direct
        remote read, so the cache never breaks a read.

        The cache key folds in the file's size and mtime (one cheap HEAD/stat per open),
        so overwriting the same remote path with new content is a miss, not a stale hit
        — correctness over saving a metadata round-trip."""
        if not self._cacheable:
            return None
        try:
            cache = get_file_cache()
            if cache is None:
                return None
            info = self._fs.get_file_info(in_path)
            key = f"{in_path}\0{info.size}\0{info.mtime_ns}"
            return cache.get_or_fetch(key, lambda dst: self._download(in_path, dst))
        except Exception:  # pragma: no cover - a cache failure must not break reads
            return None

    def _download(self, in_path: str, dst: str) -> None:
        """Stream a remote file to local `dst` (chunked, so a large file never fully
        materializes in memory)."""
        with self._fs.open_input_file(in_path) as src, open(dst, "wb") as out:
            while chunk := src.read(1 << 20):
                out.write(chunk)

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
    # `from_uri` also strips a trailing slash from `in_path` (``…/dir/`` → ``…/dir``),
    # so subtracting `len(in_path)` from the *un*-trimmed base mis-slices the prefix by
    # the slash (``s3://`` → ``s3://r``) and every later `_p()` drops a real character.
    # Strip the trailing slash off `base` before the suffix math so the two align.
    base = path.split("?", 1)[0]
    trimmed = base.rstrip("/")
    prefix = (
        trimmed[: len(trimmed) - len(in_path)]
        if in_path and trimmed.endswith(in_path)
        else base[: len(base) - len(in_path)]
    )
    canonical = _SCHEME_ALIASES.get(scheme, scheme)
    is_object_store = canonical in _OBJECT_STORE_SCHEMES
    return _ArrowFileSystem(
        fs, prefix, atomic_rename=not is_object_store, cacheable=is_object_store
    )


def _fsspec_backed(scheme: str, path: str) -> FileSystem:
    """Wrap an fsspec backend behind the `pyarrow.fs` interface (the escape hatch for
    schemes pyarrow does not implement natively)."""
    try:
        import fsspec
        from pyarrow.fs import FSSpecHandler, PyFileSystem
    except ImportError as exc:
        raise IOError(
            f"reading {scheme}:// paths needs the cloud extra: pip install 'batcher-engine[cloud]'"
        ) from exc
    protocol = _SCHEME_ALIASES.get(scheme, scheme)
    fsspec_fs = fsspec.filesystem(protocol)
    fs = PyFileSystem(FSSpecHandler(fsspec_fs))
    # The in-filesystem path is whatever fsspec's `_strip_protocol` produces — object
    # stores strip the scheme ("bucket/key"), but HTTP(S) keep the whole URL. Derive
    # the prefix from that so listing/globbing line up with the paths fsspec returns,
    # and keep the query (presigned-URL signatures live there) by not stripping it.
    stripped = fsspec_fs._strip_protocol(path)
    prefix = path[: len(path) - len(stripped)] if path.endswith(stripped) else ""
    return _ArrowFileSystem(
        fs, prefix, atomic_rename=protocol not in _OBJECT_STORE_SCHEMES, strip_query=False
    )


# --- Local-SSD read-through file cache (the Disk-Cache analog) ----------------
# A remote object-store read may be served from a local-SSD copy: the first read of
# a remote file streams it here; later reads of the same file hit local disk, sparing
# the object-store round-trip. It lives with the filesystem that opens files (not in
# the `carbonite` subsystem) because `core`/`kyber` depend on `io`, so an io→carbonite
# edge would transitively break their independence; the budget comes from config. The
# cache is transparent and ephemeral — a miss just re-fetches, never a wrong result.


class FileBytesCache:
    """A byte-bounded, LRU local-disk cache of whole remote files.

    Keyed by the remote path; the cached copy lives at ``<cache_dir>/<sha256(path)>``.
    Thread-safe. Fetching happens outside the lock (it is slow I/O) into a unique temp
    file that is atomically renamed into place, so concurrent readers never observe a
    half-written file.
    """

    __slots__ = ("_dir", "_entries", "_lock", "_max_bytes", "_used")

    def __init__(self, cache_dir: str, max_bytes: int) -> None:
        """Create the cache rooted at `cache_dir`, bounded to `max_bytes` on disk."""
        self._dir = cache_dir
        self._max_bytes = max(0, int(max_bytes))
        self._lock = threading.Lock()
        # key → on-disk size; insertion/most-recent order drives LRU eviction.
        self._entries: OrderedDict[str, int] = OrderedDict()
        self._used = 0
        os.makedirs(cache_dir, exist_ok=True)

    def get_or_fetch(self, remote_path: str, fetch: Callable[[str], None]) -> str:
        """Return the local path of the cached copy of `remote_path`.

        On a miss, `fetch(local_tmp_path)` is called to materialize the bytes (it must
        write the full file to the given path); the result is then admitted under the
        byte budget, evicting the least-recently-used entries if needed.
        """
        key = hashlib.sha256(remote_path.encode("utf-8")).hexdigest()
        local = os.path.join(self._dir, key)
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)  # mark most-recently-used
                return local

        # Miss: fetch outside the lock (slow remote I/O) to a unique temp, then rename.
        tmp = f"{local}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        try:
            fetch(tmp)
            size = os.path.getsize(tmp)
            os.replace(tmp, local)
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(tmp)
            raise

        with self._lock:
            # A racing thread may have admitted the same key first; only one accounts
            # for the bytes (the file content is identical, so the rename is harmless).
            if key not in self._entries:
                self._entries[key] = size
                self._used += size
                self._evict_locked()
            else:
                self._entries.move_to_end(key)
        return local

    def _evict_locked(self) -> None:
        """Drop least-recently-used entries until within budget (caller holds lock)."""
        while self._used > self._max_bytes and self._entries:
            old_key, old_size = self._entries.popitem(last=False)
            self._used -= old_size
            with contextlib.suppress(OSError):
                os.remove(os.path.join(self._dir, old_key))

    @property
    def used_bytes(self) -> int:
        """Total bytes currently held on disk by the cache."""
        with self._lock:
            return self._used


_CACHES: dict[str, FileBytesCache] = {}
_CACHES_LOCK = threading.Lock()


def get_file_cache() -> FileBytesCache | None:
    """The process-wide file cache for the active config, or `None` when disabled.

    Memoized per cache directory, so `config_context` overriding `file_cache_dir`
    (e.g. in a test) yields a distinct cache without disturbing the default one.
    """
    from batcher.config import active_config

    mem = active_config().memory
    if not mem.file_cache_dir:
        return None
    with _CACHES_LOCK:
        cache = _CACHES.get(mem.file_cache_dir)
        if cache is None:
            cache = FileBytesCache(mem.file_cache_dir, mem.file_cache_max_bytes)
            _CACHES[mem.file_cache_dir] = cache
        return cache
