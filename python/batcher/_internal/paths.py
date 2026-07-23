"""Filesystem locations of the installed package.

Kept here, in a neutral leaf, so that asking *where batcher lives on disk* does not require
importing the root `batcher` package. `import batcher` pulls in `api`, which imports every
subsystem — so a module that only wants a directory path would drag the whole conductor into
the import graph and break the layer-independence contract. This has bitten the Ray worker
bootstrap twice; the answer is a helper that computes the path from its own location.
"""

from __future__ import annotations

import pathlib

__all__ = ["package_dir"]


def package_dir() -> str:
    """Return the directory of the installed `batcher` package.

    Used to ship the package to Ray workers (`py_modules`), which needs the *directory*, not
    the imported module.

    Returns:
        Absolute path to the `batcher` package directory.
    """
    return str(pathlib.Path(__file__).resolve().parent.parent)
