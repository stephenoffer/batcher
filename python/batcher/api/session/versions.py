"""Version and environment reporting (`engine_version`, `show_versions`).

The first question on any bug report is "which build, which optional backends".
`show_versions` answers it in one call, the way pandas and Polars do.
"""

from __future__ import annotations

import importlib.metadata
import platform
import sys

__all__ = ["engine_version", "show_versions", "versions"]

# The optional framework integrations a `bt.from_*` / `bt.read.*` path can need. Reported
# as "not installed" rather than omitted, because an absent row reads as a missing check.
_OPTIONAL = (
    "pyarrow",
    "numpy",
    "pandas",
    "polars",
    "duckdb",
    "ray",
    "torch",
    "datasets",
    "dask",
    "pyspark",
    "deltalake",
    "pyiceberg",
)


def engine_version() -> str:
    """Return the version string reported by the compiled Rust data plane.

    The version of the native ``bc_py`` extension, distinct from the Python
    package version. Useful for confirming which engine build is loaded.

    Returns:
        The engine version, e.g. ``"0.1.0"``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> isinstance(bt.engine_version(), str)
            True
    """
    from batcher._internal.native import engine

    return str(engine().__engine_version__)


def _engine_profile() -> str:
    """The Cargo profile the loaded engine was built with: ``release`` or ``debug``.

    Not public on its own — it is a row in `versions`, which is where someone diagnosing
    a slow pipeline or filing a bug report will look. It is worth reporting because a
    ``just build`` (dev-profile) engine is unoptimized and dramatically slower, and
    nothing else about a running query says so.
    """
    from batcher._internal.native import engine

    return str(getattr(engine(), "__engine_profile__", "unknown"))


def versions() -> dict[str, str]:
    """Return the Batcher, engine, Python, platform, and optional-backend versions.

    The machine-readable form of `show_versions`: paste it into a bug report, or
    assert on it in a test that needs a particular backend present. Optional
    integrations that are not installed map to ``"not installed"`` rather than
    being omitted, so a missing key always means an unknown package.

    ``engine_profile`` is ``release`` or ``debug``. A ``debug`` engine — what ``just
    build`` installs — is unoptimized, and nothing else about a running query says so, so
    it is the first thing to check when a pipeline is unexpectedly slow.

    Returns:
        A mapping of component name to version string.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.versions()["batcher"]
            '0.1.0'
    """
    import batcher

    out = {
        "batcher": batcher.__version__,
        "engine": engine_version(),
        # `debug` here is the answer to "why is this slow?" more often than any plan.
        "engine_profile": _engine_profile(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in _OPTIONAL:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def show_versions() -> None:
    """Print the Batcher, engine, Python, and optional-backend versions (pandas idiom).

    The report to paste into a bug report. Use `versions` for the same information as
    a dict.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.show_versions()  # doctest: +SKIP
            batcher     : 0.1.0
            engine      : 0.1.0
            ...
    """
    info = versions()
    width = max(len(k) for k in info)
    for key, value in info.items():
        print(f"{key.ljust(width)} : {value}")
