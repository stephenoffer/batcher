"""Path normalization shared by every file source and sink.

One place decides what counts as "a path" at the IO boundary, so `pathlib.Path`, an
`os.PathLike`, a ``~`` shorthand, and a list of files behave identically for every
format rather than each reader re-deciding. Everything downstream (`resolve_filesystem`,
glob expansion, split construction) then sees a plain `str` URI and needs no such logic.

The rules, in order:

* `os.PathLike` (which `pathlib.Path` implements) becomes its `os.fspath` string.
* ``~`` / ``~user`` expands, but **only for a local path** — a ``~`` in an object-store
  key is a legal character, and expanding it would rewrite the key to a home directory
  that has nothing to do with the bucket.
* A list/tuple of paths becomes a root plus an explicit file list, which is what
  `FileSource(files=...)` already consumes to pin a source to named files.

`bytes` is deliberately *not* accepted: a bytes path cannot carry a URI scheme
unambiguously, and silently decoding one hides an encoding bug behind a read that
half-works.
"""

from __future__ import annotations

import os
from typing import Any

from batcher._internal.errors import IOError as _IOError

__all__ = ["hive_segment", "normalize_path", "normalize_source_path"]


def hive_segment(name: str) -> tuple[str, str] | None:
    """Parse a ``col=val`` directory basename, or None if it isn't one.

    Lives here rather than beside either reader because both the Hive-aware dataset
    source (which *uses* the partition columns) and the plain file source (which must
    notice it is about to *lose* them) have to agree on what a partition directory is.
    Two copies of this rule would drift into one reader warning about a layout the other
    does not treat as partitioned.

    Args:
        name: A directory name or path whose last segment is examined.

    Returns:
        The ``(column, value)`` pair, or None when the segment is not ``col=val``.

    Examples:
        .. doctest::

            >>> from batcher.io.base._paths import hive_segment
            >>> hive_segment("dt=2024-01-01")
            ('dt', '2024-01-01')
            >>> hive_segment("plain") is None
            True
    """
    base = name.rstrip("/").rsplit("/", 1)[-1]
    if "=" not in base:
        return None
    col, _, val = base.partition("=")
    return (col, val) if col else None


def _scheme_of(path: str) -> str:
    idx = path.find("://")
    return path[:idx].lower() if idx > 0 else ""


def normalize_path(path: Any, *, what: str = "path") -> str:
    """Coerce one path-like value to the plain `str` URI the IO layer works in.

    Accepts a `str`, a `pathlib.Path`, or any `os.PathLike`, and expands a leading ``~``
    for local paths. A value that is not path-like raises rather than failing later
    inside a filesystem call, where the message would name an attribute rather than the
    argument the caller actually got wrong.

    Args:
        path: The path-like value to normalize.
        what: What the value is, used in the error message (e.g. ``"path"``).

    Returns:
        The normalized path as a string.

    Examples:
        .. doctest::

            >>> import pathlib
            >>> from batcher.io.base._paths import normalize_path
            >>> normalize_path(pathlib.Path("/tmp/a.csv"))
            '/tmp/a.csv'
            >>> normalize_path("s3://bucket/~keep/a.csv")
            's3://bucket/~keep/a.csv'
    """
    if isinstance(path, str):
        text = path
    elif isinstance(path, os.PathLike):
        text = os.fspath(path)
        if not isinstance(text, str):  # a bytes-flavored PathLike
            raise _IOError(
                f"{what} must be a string or a path-like object producing a string, got "
                f"{type(path).__name__} yielding bytes. Decode it first."
            )
    elif isinstance(path, bytes):
        raise _IOError(
            f"{what} must be a str or os.PathLike, not bytes. Decode it first "
            "(path.decode()) so the URI scheme is unambiguous."
        )
    else:
        raise _IOError(
            f"{what} must be a str, pathlib.Path, or os.PathLike, got "
            f"{type(path).__name__}. Pass a file, directory, or glob — for several "
            "explicit files, pass a list of them."
        )
    # `~` is a legal character in an object-store key, so expand it only where it is a
    # shell shorthand: a local path with no URI scheme.
    if text.startswith("~") and not _scheme_of(text):
        text = os.path.expanduser(text)
    return text


def normalize_source_path(path: Any) -> tuple[str, list[str] | None]:
    """Normalize a source path that may also be an explicit list of files.

    A list is how a caller names several files that share no useful glob — the shape
    `pandas.concat([...])` and Spark's ``spark.read.parquet(*paths)`` cover. It becomes
    a root path (their longest common directory, which is what statistics and the plan
    display) plus the explicit file list `FileSource` pins itself to, so the read opens
    exactly those files and never lists the directory around them.

    Args:
        path: A path-like value, or a list/tuple of them.

    Returns:
        A ``(root, files)`` pair. `files` is None for a single path, in which case
        `root` is the path itself.

    Examples:
        .. doctest::

            >>> from batcher.io.base._paths import normalize_source_path
            >>> normalize_source_path("/data/a.csv")
            ('/data/a.csv', None)
            >>> normalize_source_path(["/data/a.csv", "/data/b.csv"])
            ('/data', ['/data/a.csv', '/data/b.csv'])
    """
    if not isinstance(path, list | tuple):
        return normalize_path(path), None
    files = [normalize_path(p, what="each path in the list") for p in path]
    if not files:
        raise _IOError(
            "the list of paths is empty, so there is nothing to read. Pass at least one "
            "file, or a directory/glob instead of a list."
        )
    if len(files) == 1:
        return files[0], None
    return _common_root(files), files


def _common_root(files: list[str]) -> str:
    """The directory the file list hangs off, for display and statistics identity.

    Falls back to the first file's directory when the paths share no common prefix (two
    buckets, or a mix of schemes): the root is descriptive only — the pinned `files` list
    is what is actually read — so an imprecise root costs nothing but a nicer plan label.
    """
    dirs = [f.rsplit("/", 1)[0] if "/" in f else "" for f in files]
    if len({_scheme_of(f) for f in files}) > 1:
        return dirs[0]
    common = os.path.commonpath(dirs) if all(not _scheme_of(f) for f in files) else None
    if common:
        return common
    first = dirs[0]
    return first if all(d == first for d in dirs) else os.path.commonprefix(dirs).rstrip("/")
