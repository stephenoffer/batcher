"""A query must not require `sqlite3` unless the SQLite metadata backend was chosen.

The executor imports `batcher.metadata.backends` to build its default in-process store, and
that package used to import `SQLiteBackend` — and with it the stdlib `sqlite3` — eagerly. That
made `sqlite3` a precondition of every query, and it is not always loadable: on a common
Anaconda install with a pip-installed `pyarrow`, pyarrow binds the system `libstdc++` first and
Anaconda's `_sqlite3` then fails to load its ICU dependency (`CXXABI_1.3.15 not found`). Every
query and every example failed there, on a deployment that had never asked for SQLite.

The checks run in a subprocess, because "was `sqlite3` imported" is a property of a whole
process and any earlier test in this one may already have imported it.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)


def test_a_default_query_never_imports_sqlite3() -> None:
    result = _run(
        "import sys, batcher as bt\n"
        "assert bt.from_pydict({'a': [1, 2]}).agg(s=bt.col('a').sum()).to_pydict() == {'s': [3]}\n"
        "print('sqlite3' in sys.modules)\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", (
        "a query on the default in_process backend imported sqlite3, so it fails wherever "
        "sqlite3 cannot load"
    )


def test_the_facade_still_exports_the_sqlite_backend() -> None:
    """Lazy, not removed: `from batcher.metadata.backends import SQLiteBackend` keeps working."""
    result = _run(
        "import sqlite3\n"  # loaded first, so this checks the facade rather than the environment
        "from batcher.metadata.backends import SQLiteBackend, __all__\n"
        "assert 'SQLiteBackend' in __all__\n"
        "print(SQLiteBackend.__name__)\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "SQLiteBackend"


def test_choosing_sqlite_where_it_cannot_load_is_a_config_error_not_a_crash(monkeypatch) -> None:
    """The failure moves to the one place that asked for SQLite, and it names the setting."""
    import builtins

    from batcher._internal.errors import ConfigError
    from batcher.metadata.backends import factory

    real_import = builtins.__import__

    def refuse_sqlite(name, *args, **kwargs):
        if name == "batcher.metadata.backends.sqlite" or name == "sqlite3":
            raise ImportError("simulated: sqlite3 cannot load")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "batcher.metadata.backends.sqlite", raising=False)
    monkeypatch.setattr(builtins, "__import__", refuse_sqlite)
    with pytest.raises(ConfigError, match=r"metadata\.backend='sqlite'"):
        factory.make_backend("sqlite", ":memory:")
