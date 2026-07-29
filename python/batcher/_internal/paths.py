"""Filesystem locations of the installed package, and how to create things there safely.

Kept here, in a neutral leaf, so that asking *where batcher lives on disk* does not require
importing the root `batcher` package. `import batcher` pulls in `api`, which imports every
subsystem — so a module that only wants a directory path would drag the whole conductor into
the import graph and break the layer-independence contract. This has bitten the Ray worker
bootstrap twice; the answer is a helper that computes the path from its own location.

The same neutrality is why `private_dir`/`open_private` live here rather than in whichever
subsystem writes first. `carbonite` (spill), `dist` (shuffle scratch), `io` (the file cache),
`metadata` (the learned-stats database), and `api` (the event log) all write artifacts, and
they sit on four different layers with a mutual-independence contract between them. Sharing
by copy-paste is the one *wrong* way to share between those, so the helper goes down.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import sys
from typing import BinaryIO

__all__ = ["batcher_home", "open_private", "package_dir", "private_dir"]

#: Directory mode: owner-only. Applied to anything Batcher creates to hold its artifacts.
_DIR_MODE = 0o700
#: File mode: owner-only.
_FILE_MODE = 0o600


def package_dir() -> str:
    """Return the directory of the installed `batcher` package.

    Used to ship the package to Ray workers (`py_modules`), which needs the *directory*, not
    the imported module.

    Returns:
        Absolute path to the `batcher` package directory.
    """
    return str(pathlib.Path(__file__).resolve().parent.parent)


def batcher_home() -> pathlib.Path:
    """The base directory for Batcher's per-user state, `$BATCHER_HOME` or `~/.batcher`.

    One answer to "where is `~/.batcher`", so the event log, the learned-stats database, and
    the pipeline registry resolve the same base the same way. Duplicating the `~/.batcher`
    literal across those callers is how one of them ends up writing where the others do not
    look.

    Not created here — a caller that writes into it makes the specific subdirectory it
    needs, so merely resolving a path never touches the filesystem.

    Returns:
        The resolved base directory as a `Path`.
    """
    base = os.environ.get("BATCHER_HOME") or os.path.join(os.path.expanduser("~"), ".batcher")
    return pathlib.Path(base)


def private_dir(path: str | os.PathLike[str]) -> pathlib.Path:
    """Create `path` (and parents) owner-only, and tighten it if it already exists.

    `os.makedirs(mode=...)` honours the process umask, so it *requests* 0700 and may get
    less. It also silently leaves an existing directory's mode alone, which is the common
    case for a shared scratch root that some earlier run — or the operator — created 0755.
    So the mode is both requested and, on POSIX, asserted afterwards.

    Args:
        path: The directory to create.

    Returns:
        The directory as a `Path`.
    """
    resolved = pathlib.Path(path)
    os.makedirs(resolved, mode=_DIR_MODE, exist_ok=True)
    if sys.platform != "win32":
        # A directory this process does not own (a shared mount an operator set up) cannot
        # be tightened, and failing the query over it would be worse than the exposure —
        # the files written inside are still created 0600 by `open_private`.
        with contextlib.suppress(OSError):
            os.chmod(resolved, _DIR_MODE)
    return resolved


def open_private(path: str | os.PathLike[str], mode: str = "wb") -> BinaryIO:
    """Open `path` for writing, owner-only **from the moment it exists**.

    The mode is set in the `open` call rather than by a following `chmod`, because a chmod
    leaves a window in which the file is world-readable — and these files are the query's
    own data. A reader that wins that race gets everything.

    Args:
        path: The file to create or truncate.
        mode: A binary write mode (``wb`` or ``ab``).

    Returns:
        An open binary file object.

    Raises:
        ValueError: If `mode` is not a binary write mode.
    """
    if mode not in ("wb", "ab"):
        raise ValueError(f"open_private is for binary writes, got mode={mode!r}")
    flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if mode == "ab" else os.O_TRUNC)
    return os.fdopen(os.open(path, flags, _FILE_MODE), mode)
