"""The `pyarrow.fs`-backed filesystem façade every IO source and sink talks to.

This module owns the *interface* — the `FileSystem` protocol and the `_ArrowFileSystem`
adapter implementing it over any `pyarrow.fs` backend, including the URI <-> in-filesystem
path mapping, listing/globbing, atomic writes, and the read-through cache hook. Choosing
*which* backend a path resolves to lives in the sibling `filesystem` module, which
re-exports everything here; splitting the two keeps each within the size limit without
changing a single import path.
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

import pyarrow.fs as pafs

from batcher._internal.errors import IOError
from batcher._internal.logging import note_suppressed
from batcher.io._file_cache import get_file_cache

__all__ = ["FileSystem", "_ArrowFileSystem", "_is_data_file", "_scheme"]


# Read-ahead buffer for a remote handle-path read — 1 MiB coalesces the split readers'
# tiny reads into few large GETs instead of the 8 KiB `BufferedReader` default.
_REMOTE_READ_BUFFER = 1 << 20

# The characters that make a path a glob rather than a literal.
_WILDCARDS = "*?["


def _has_wildcard(segment: str) -> bool:
    """Whether one path component contains a glob metacharacter."""
    return any(ch in segment for ch in _WILDCARDS)


def _data_files(infos: Any, matches: Any) -> list:
    """The data files among `infos` whose path satisfies the `matches` predicate."""
    return [
        fi
        for fi in infos
        if fi.type == pafs.FileType.File and _is_data_file(fi.path) and matches(fi.path)
    ]


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

    def listing_info(self, path: str) -> tuple[int, int] | None:  # noqa: ARG002
        """`(size, mtime_ns)` for `path` if a previous `expand` saw it, else None.

        Lets a caller learn whether a file could have changed without paying a stat, using
        what the directory listing already returned. A backend that keeps no listing
        information returns None and the caller falls back to a stat.
        """
        return None

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

    __slots__ = (
        "_atomic_rename",
        "_cacheable",
        "_fs",
        "_listing_info",
        "_prefix",
        "_root",
        "_strip_query",
    )

    def __init__(
        self,
        fs: pafs.FileSystem,
        prefix: str,
        *,
        atomic_rename: bool,
        strip_query: bool = True,
        cacheable: bool = False,
        root: str = "",
    ) -> None:
        self._fs = fs
        # `(size, mtime_ns)` per URI, harvested from the directory listing `expand`
        # already performs. It is not a data cache — it only answers "could this file have
        # changed", so a metadata memo can be keyed on the file's version rather than its
        # path. Bounded by the listing that produced it.
        self._listing_info: dict[str, tuple[int, int]] = {}
        self._prefix = prefix
        # What the backend prepends to the URL path. Usually "" — the in-filesystem path is
        # a *suffix* of the URI (``s3://bucket/k`` → ``bucket/k``), so stripping `_prefix`
        # is the whole mapping. Azure is not: its container lives in the URI authority
        # (``abfs://c@acct.dfs.core.windows.net/k`` → ``c/k``), so no prefix strip can
        # produce it and the mapping is ``_root + urlpath`` instead.
        self._root = root
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
            return f"{self._root}{p[len(self._prefix) :]}"
        return p

    def _parent_dir(self, in_path: str) -> str:
        """The parent directory of an in-filesystem path, per the backend's separator.

        Object-store keys are always ``/``-separated, so `posixpath` is right for them —
        and splitting those on ``\\`` would be actively wrong, since a backslash is a legal
        character *inside* an S3 key. A local path uses the platform separator, which on
        Windows is ``\\``: `posixpath.dirname` finds no separator there, returns ``""``, and
        the parent directory is silently never created, so the write fails on a path the
        caller had every reason to expect to work. `os.path` is `posixpath` on POSIX, so
        this is only a behavior change on Windows.
        """
        if isinstance(self._fs, pafs.LocalFileSystem):
            return os.path.dirname(in_path)
        return posixpath.dirname(in_path)

    def _uri(self, in_path: str) -> str:
        """An in-filesystem path → the full path/URI callers see."""
        if self._root and in_path.startswith(self._root):
            in_path = in_path[len(self._root) :]
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
    def listing_info(self, path: str) -> tuple[int, int] | None:
        """`(size, mtime_ns)` for `path` when the directory listing already reported it."""
        return self._listing_info.get(path)

    def expand(self, path: str, *, suffix: str | tuple[str, ...]) -> list[str]:
        if _has_wildcard(path):
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

            def _list(recursive: bool) -> list:
                sel = pafs.FileSelector(in_path, recursive=recursive)
                return _data_files(self._fs.get_file_info(sel), lambda p: p.endswith(suffixes))

            # Flat listing first: it is the common layout and stays one cheap LIST. Only when
            # it finds nothing — where this used to raise outright — descend. A Hive tree
            # (`out/p=a/part-0.parquet`) or a nested media corpus (`videos/2024/01/…`) keeps
            # every data file one or more levels down, so a non-recursive read of a directory
            # Batcher itself wrote with `partition_by=` failed with "no .parquet files found".
            matched = _list(recursive=False) or _list(recursive=True)
            if not matched:
                raise IOError(f"no {'/'.join(suffixes)} files found in directory {path!r}")
            matched.sort(key=lambda fi: fi.path)
            # Remember each file's size/mtime so `stats_version` need not stat them all.
            self._record_listing(matched)
            return [self._uri(fi.path) for fi in matched]
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
        # `**` keeps the flat list-the-subtree-then-match-the-path strategy, because that is
        # what `**` means; everything else walks the pattern per component (`_walk_glob`).
        infos = self._recursive_glob(in_pat) if "**" in in_pat else self._walk_glob(in_pat)
        if not infos:
            raise IOError(f"glob {pattern!r} matched no files")
        infos.sort(key=lambda fi: fi.path)
        # Record what the listing already told us, as the directory branch does. Withholding
        # it made `file_identity` stat every matched file — three times per query, each its
        # own pool task; on a 2,000-file read that storm outweighed the Parquet read itself
        # (820ms -> 513ms; vs DuckDB 6.6x -> 3.8x, vs Polars 3.2x -> 1.8x). It was withheld
        # because entries outlive the listing on a cached filesystem, so a file overwritten
        # after the glob kept reporting its old `(size, mtime)`. `atomic_writer`/`remove`
        # now drop a path's entry as they write it, closing the case that produces it: this
        # process overwriting its own deterministically-named output. An overwrite by
        # another process needs a stat, and the directory branch has that same exposure.
        self._record_listing(infos)
        return [self._uri(fi.path) for fi in infos]

    def _recursive_glob(self, in_pat: str) -> list:
        """Every file under the pattern's literal root whose *full path* matches `in_pat`.

        `**` crosses directory boundaries, so the subtree is listed and the whole path
        matched — `fnmatch`'s `*` crosses `/` too, which is exactly right here."""
        base = in_pat
        for i, ch in enumerate(in_pat):
            if ch in _WILDCARDS:
                base = self._parent_dir(in_pat[:i])
                break
        sel = pafs.FileSelector(base or ".", recursive=True, allow_not_found=True)
        return _data_files(self._fs.get_file_info(sel), lambda p: fnmatch.fnmatch(p, in_pat))

    def _walk_glob(self, in_pat: str) -> list:
        """Match `in_pat` component by component, listing only the directories that match.

        Fixes two faults in the previous single-listing-plus-`fnmatch` approach. **A wildcard
        in a directory component matched nothing**: the old code listed the parent of the
        *first* wildcard non-recursively, so ``data/*/part.parquet`` saw only ``data``'s
        direct children — the partition *directories*, never the files — and raised "matched
        no files" on the most common layout there is. And **it over-listed**: per-component
        walking turns ``data/date=*/hour=*/*.parquet`` into one LIST of ``data`` plus one per
        *matching* partition, not a flat listing of everything beneath it filtered in Python.

        Each component matches a path *segment*, so `*` does not cross `/`.
        """
        segments = in_pat.split("/")
        # Leading literal components are the listing root: never listed, just joined.
        first_wild = next(
            (i for i, s in enumerate(segments) if _has_wildcard(s)), len(segments) - 1
        )
        roots = ["/".join(segments[:first_wild])]
        for segment in segments[first_wild:-1]:
            if not _has_wildcard(segment):
                roots = [f"{root}/{segment}" if root else segment for root in roots]
                continue
            roots = [d for root in roots for d in self._match_dirs(root, segment)]
            if not roots:
                return []
        leaf = segments[-1]

        def matches(path: str) -> bool:
            return fnmatch.fnmatch(posixpath.basename(path), leaf)

        return [fi for root in roots for fi in _data_files(self._list_dir(root), matches)]

    def _list_dir(self, in_path: str) -> list:
        """One non-recursive listing of `in_path`, empty rather than raising if absent."""
        sel = pafs.FileSelector(in_path or ".", recursive=False, allow_not_found=True)
        return list(self._fs.get_file_info(sel))

    def _match_dirs(self, root: str, seg: str) -> list[str]:
        """`root`'s immediate subdirectories whose name matches the glob component `seg`."""
        dirs = (fi for fi in self._list_dir(root) if fi.type == pafs.FileType.Directory)
        return [fi.path for fi in dirs if fnmatch.fnmatch(posixpath.basename(fi.path), seg)]

    def _record_listing(self, infos: list) -> None:
        """Remember each listed file's `(size, mtime_ns)` so a caller need not stat it — the
        listing already fetched both, and re-asking is one HEAD per file (`listing_info`)."""
        for fi in infos:
            self._listing_info[self._uri(fi.path)] = (int(fi.size or 0), int(fi.mtime_ns or 0))

    def _forget_listing(self, path: str) -> None:
        """Drop `path`'s listing entry, because this filesystem is about to write it.

        Otherwise a source that listed a directory then overwrote a file in it keeps
        answering `listing_info` with the pre-overwrite `(size, mtime)`, and every cache
        keyed on that identity (`io.stats.file_identity`) serves the *previous* file's
        footer, row count, and zone maps for the new bytes.
        """
        self._listing_info.pop(path, None)

    def _glob_prefix_scoped(self, pattern: str, in_pat: str) -> list[str] | None:
        """A prefix-scoped remote glob via fsspec, or ``None`` to fall back to pyarrow.

        Returns URIs only when fsspec is installed for the scheme *and* found files, so an
        empty/errored probe never masks the pyarrow listing (which owns the empty-is-error
        and credential-failure semantics). Local/backendless schemes return ``None``.
        """
        scheme = _scheme(pattern)
        if scheme in ("", "file"):
            return None  # local globbing is already a cheap single-directory listing.
        # Only worth it when the *filename* has a literal prefix before its wildcard
        # (``dir/PREFIX*.ext``): that prefix narrows the LIST below the directory, which is
        # the one thing `_walk_glob`'s per-component listing cannot do. For a bare
        # ``dir/*.ext`` the two issue the same single LIST, so skip fsspec.
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
        """The local-cache copy of a remote file, fetched on a miss; `None` when caching is off.

        Best-effort: any failure falls back to a direct remote read, so the cache never breaks
        a read. The cache key folds in the file's size and mtime (one cheap HEAD/stat per open),
        so overwriting the same remote path with new content is a miss, not a stale hit —
        correctness over saving a metadata round-trip."""
        if not self._cacheable:
            return None
        try:
            cache = get_file_cache()
            if cache is None:
                return None
            info = self._fs.get_file_info(in_path)
            key = f"{in_path}\0{info.size}\0{info.mtime_ns}"
            return cache.get_or_fetch(key, lambda dst: self._download(in_path, dst))
        except Exception as exc:  # pragma: no cover - a cache failure must not break reads
            note_suppressed("io", "read cached remote file", exc)
            return None

    def _download(self, in_path: str, dst: str) -> None:
        """Stream a remote file to local `dst`, chunked so a large file never fully materializes."""
        with self._fs.open_input_file(in_path) as src, open(dst, "wb") as out:
            while chunk := src.read(1 << 20):
                out.write(chunk)

    @contextlib.contextmanager
    def atomic_writer(self, path: str) -> Iterator[IO[Any]]:
        # This filesystem is about to change `path`, so anything a previous listing recorded
        # about it is now stale — see `_forget_listing`.
        self._forget_listing(path)
        dest = self._p(path)
        # Ensure the parent directory exists (pyarrow's output stream does not create
        # it). Cheap and idempotent; a no-op marker on object stores.
        parent = self._parent_dir(dest)
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
        self._forget_listing(path)
        in_path = self._p(path)
        if self._fs.get_file_info(in_path).type != pafs.FileType.NotFound:
            with contextlib.suppress(FileNotFoundError):
                self._fs.delete_file(in_path)
