"""Filesystem locations of the installed package.

Kept here, in a neutral leaf, so that asking *where batcher lives on disk* does not require
importing the root `batcher` package. `import batcher` pulls in `api`, which imports every
subsystem — so a module that only wants a directory path would drag the whole conductor into
the import graph and break the layer-independence contract. This has bitten the Ray worker
bootstrap twice; the answer is a helper that computes the path from its own location.
"""

from __future__ import annotations

import os
import pathlib

__all__ = ["batcher_home", "package_dir"]


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
