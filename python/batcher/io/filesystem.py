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
import functools
import io
import os
import posixpath
import uuid
from collections.abc import Iterator
from typing import IO, Any, Protocol, runtime_checkable

import pyarrow as pa
import pyarrow.fs as pafs

from batcher._internal.errors import IOError

# The local-SSD read-through cache lives in a sibling module (its own responsibility);
# re-exported here since the filesystem is its only caller and tests import it by this path.
from batcher.io._file_cache import FileBytesCache, get_file_cache

__all__ = [
    "FileBytesCache",
    "FileSystem",
    "LocalFileSystem",
    "get_file_cache",
    "resolve_filesystem",
]

# Object stores where a single PUT is already atomic (no partial-read visibility),
# so a write goes straight to the destination — a temp-then-rename would only add a
# full-object server-side copy with no atomicity gain. Everything else (local, NFS,
# HDFS) gets temp-write-then-rename so a crash never leaves a truncated file.
_OBJECT_STORE_SCHEMES = frozenset(
    {"s3", "s3a", "gs", "gcs", "abfs", "abfss", "az", "azure", "wasb", "wasbs"}
)
# Cloud scheme aliases → the canonical scheme `from_uri` / fsspec understand.
_SCHEME_ALIASES = {"s3a": "s3", "gcs": "gs", "abfss": "abfs", "wasbs": "wasb"}
# Read-ahead buffer for a remote handle-path read — 1 MiB coalesces the split readers'
# tiny reads into few large GETs instead of the 8 KiB `BufferedReader` default.
_REMOTE_READ_BUFFER = 1 << 20


@functools.cache
def ensure_io_threads() -> None:
    """Lift pyarrow's IO thread pool above its 8-thread default, once per process.

    A wide object-store read is otherwise throttled to ~8 concurrent GETs, so a
    many-small-files scan can't saturate the link. Idempotent/cached; a no-op if the pool
    is already wider. `BATCHER_IO_THREADS` overrides the target (default 32)."""
    target = max(8, int(os.environ.get("BATCHER_IO_THREADS", "32")))
    if pa.io_thread_count() < target:
        pa.set_io_thread_count(target)
    cap_arrow_cpu_threads()


def cap_arrow_cpu_threads() -> None:
    """Cap pyarrow's CPU thread pool to the cores this process may actually use.

    pyarrow sizes its compute/decode pool to `os.cpu_count()` (host cores), so under a
    cgroup CPU quota (a Kubernetes/Ray pod) its kernels over-subscribe cores the container
    never gets, thrashing the scheduler. Only ever *lowers* the count. Result-invariant."""
    from batcher._internal.hardware import available_cpu_count

    usable = available_cpu_count()
    if pa.cpu_count() > usable:
        pa.set_cpu_count(usable)


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

    def expand(self, path: str, *, suffix: str | tuple[str, ...]) -> list[str]:
        """Resolve a file, directory, or glob into a sorted list of file paths.

        ``suffix`` may be a single extension or a tuple of them; a directory listing
        keeps files matching *any* of them in one pass (a source with several accepted
        extensions must not re-list the directory once per extension).
        """
        ...

    def open(self, path: str, mode: str = "rb") -> IO[Any]:
        """Open a single file for reading; the handle is accepted by pyarrow."""
        ...

    def native_read_target(self, path: str) -> tuple[Any, str] | None:
        """The `(pyarrow.fs.FileSystem, in_path)` pair for `path`, or None.

        A reader handed this pair does its own I/O in C++ — pre-buffering, parallel column
        chunks, no GIL. Handed a Python file object it round-trips every read through the
        interpreter, serializing its decode threads: a four-column read of a 16 GB Parquet
        file took 2,831 ms through a handle against 1,653 ms through this pair (one column
        is identical — the cost is per column chunk). None when the backend cannot expose
        one (a read-through byte cache serves reads through `open`).
        """
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

    def native_read_target(self, path: str) -> tuple[pafs.FileSystem, str] | None:
        """This backend *is* a pyarrow filesystem, so hand it over directly.

        Withheld when a read-through byte cache is configured: that cache serves reads
        through `open`, and bypassing it would silently disable it.
        """
        if self._cacheable and get_file_cache() is not None:
            return None
        return self._fs, self._p(path)

    # ---- shared surface ----------------------------------------------------
    def expand(self, path: str, *, suffix: str | tuple[str, ...]) -> list[str]:
        if any(ch in path for ch in "*?["):
            return self._glob(path)
        # `str.endswith` takes a tuple directly, so a multi-extension source lists the
        # directory once and keeps any matching file — not once per extension.
        suffixes = (suffix,) if isinstance(suffix, str) else tuple(suffix)
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
                and fi.path.endswith(suffixes)
                and _is_data_file(fi.path)
            )
            if not files:
                raise IOError(f"no {'/'.join(suffixes)} files found in directory {path!r}")
            return [self._uri(f) for f in files]
        if info.type == pafs.FileType.NotFound:
            raise IOError(f"path {path!r} does not exist")
        return [path]

    def _glob(self, pattern: str) -> list[str]:
        in_pat = self._p(pattern)
        # Fast path: push the glob's literal key-prefix to the store's own prefix-scoped
        # LIST (via fsspec) — globbing `dir/00000*.jpg` in a 200k-object bucket is one LIST
        # of the ~10 matches, not a page of every object. Opt-only: it short-circuits only
        # on a positive hit, so the pyarrow listing below stays the correctness backstop.
        fast = self._glob_prefix_scoped(pattern, in_pat)
        if fast is not None:
            return fast
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

    def _glob_prefix_scoped(self, pattern: str, in_pat: str) -> list[str] | None:
        """A prefix-scoped remote glob via fsspec, or ``None`` to fall back to pyarrow.

        Returns matched URIs only when fsspec is installed for the scheme *and* found files,
        so an empty/errored probe never masks the pyarrow listing (which owns the empty-is-
        error and credential-failure semantics). Local/backendless schemes return ``None``.
        """
        scheme = _scheme(pattern)
        if scheme in ("", "file"):
            return None  # local globbing is already a cheap single-directory listing.
        # Only worth it when the *filename* has a literal prefix before its wildcard
        # (``dir/PREFIX*.ext``) — that prefix scopes the LIST; a bare ``dir/*.ext`` lists
        # the whole directory either way, so skip fsspec and let pyarrow do it.
        last = in_pat.rsplit("/", 1)[-1]
        first_wild = min((last.find(c) for c in "*?[" if c in last), default=len(last))
        if first_wild <= 0:
            return None
        try:
            import fsspec
        except ImportError:
            return None
        try:
            backend = fsspec.filesystem(scheme)
            matches = backend.glob(in_pat)
        except Exception:
            return None  # missing backend (e.g. s3fs), credential, or API issue → pyarrow.
        files = sorted(self._uri(m) for m in matches if _is_data_file(m))
        return files or None

    def open(self, path: str, mode: str = "rb") -> IO[Any]:  # noqa: ARG002 (read-only façade)
        # A buffered wrapper over the pyarrow input file gives the full Python file protocol
        # (read/readline/seek) the byte-range split readers rely on, and pyarrow accepts it.
        in_path = self._p(path)
        local = self._cached_local(in_path)
        if local is not None:
            return io.BufferedReader(open(local, "rb"))
        # A 1 MiB buffer (not the 8 KiB default) coalesces the tiny reads the byte-range
        # split readers issue into far fewer, larger GETs against object storage — the
        # small-request tax on a high-latency remote path. Matches `_download`'s chunk size.
        return io.BufferedReader(
            self._fs.open_input_file(in_path),  # type: ignore[arg-type]
            buffer_size=_REMOTE_READ_BUFFER,
        )

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
    # Cache the resolved object-store filesystem per (scheme, authority, query-options): every
    # `from_uri` re-walks the credential chain (an IMDS round-trip on S3) and opens a fresh
    # connection pool, and the scan path resolves a filesystem *per split / footer / open* —
    # precisely the many-small-files tax. The FS is stateless/thread-safe and identical for
    # every object key in the same bucket + config, so it is safe to memoize (the native store
    # cache does the same). The object *path* is dropped from the key so all keys share one FS.
    base = path.split("?", 1)[0]
    query = path.split("?", 1)[1] if "?" in path else ""
    authority = base.split("://", 1)[1].split("/", 1)[0] if "://" in base else ""
    return _resolve_uri_fs(f"{scheme}://{authority}" + (f"?{query}" if query else ""))


@functools.lru_cache(maxsize=128)
def _resolve_uri_fs(uri: str) -> FileSystem:
    """Build (once, cached) the `pyarrow.fs` façade for an object-store `uri` reduced to
    `scheme://authority?query` (see `resolve_filesystem`)."""
    scheme = _scheme(uri)
    try:
        fs, in_path = pafs.FileSystem.from_uri(uri)
    except (ValueError, OSError, pa.ArrowInvalid, pa.ArrowNotImplementedError):
        # A scheme pyarrow.fs doesn't implement natively → fsspec fallback.
        return _fsspec_backed(scheme, uri)
    path = uri
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
