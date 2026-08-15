"""Every optional dependency is reached through one guard, and its hint is a real command.

`_internal.optional.require` exists so that an absent driver produces one sentence, one
exception class, and an install command that works. Four more copies of it had grown anyway —
`io.formats.sql._common.require_module`, `io.formats.nosql.base.require_driver`,
`io.formats.sql.dbapi.source`'s importer, and `io.interop`'s `_missing` — and the copies did
not merely phrase things differently. Two things actually broke:

* **The exception class.** `require` raises `MissingDependencyError`, which is both a
  `BackendError` and an `ImportError`. The copies raised a plain `BackendError` or, in one
  case, a bare `ImportError` with no install field. So ``except ImportError`` around
  ``bt.from_pandas(df)`` did not catch what the identical spelling around
  ``bt.read.parquet(...)`` does — a difference no user could predict and none of them
  documented.

* **The install command.** Four hints named an extra that did not exist:
  ``batcher-engine[spark]``, ``[dask]``, ``[jax]``, and — naming the wrong distribution
  as well — ``batcher[oidc]``. The single actionable line in the error was a command that
  fails. The extras are declared now, which is what those messages were always claiming.

So the tests here are about the *contract*, not about any one backend: that the hint names a
declared extra, and that the error is catchable the way a user would try to catch it.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from batcher._internal.errors import BackendError, MissingDependencyError
from batcher._internal.optional import require

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE = _ROOT / "python" / "batcher"

#: A module name no environment will have, so the guard's failure path is the one under test.
_ABSENT = "batcher_no_such_dependency_xyz"


def _declared_extras() -> set[str]:
    """Every extra `pyproject.toml` declares, including the convenience bundles."""
    data = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    return set(data["project"]["optional-dependencies"])


def test_a_missing_dependency_is_catchable_as_an_import_error() -> None:
    """The spelling a user reaches for around an optional import."""
    with pytest.raises(ImportError):
        require(_ABSENT, feature="A feature", provides="Something", extra="pandas")


def test_a_missing_dependency_is_also_catchable_as_a_backend_error() -> None:
    """The spelling the engine's own handlers use, so the copies' callers keep working."""
    with pytest.raises(BackendError):
        require(_ABSENT, feature="A feature", provides="Something", extra="pandas")


def test_the_error_carries_the_install_command_as_a_field() -> None:
    """So a caller can surface the command its own way instead of parsing the message."""
    with pytest.raises(MissingDependencyError) as caught:
        require(_ABSENT, feature="A feature", provides="Something", extra="pandas")
    assert caught.value.install == "pip install 'batcher-engine[pandas]'"


def test_the_two_sql_guards_delegate_rather_than_restate() -> None:
    """`require_module` and `require_driver` are adapters now, not two more copies."""
    from batcher.io.formats.nosql.base import require_driver
    from batcher.io.formats.sql._common import require_module

    for guard in (require_module, require_driver):
        with pytest.raises(MissingDependencyError) as caught:
            guard(_ABSENT, extra="snowflake")
        assert caught.value.install == "pip install 'batcher-engine[snowflake]'"


def test_a_present_module_is_returned_unwrapped() -> None:
    """The happy path, which is every call in a real install."""
    import json as stdlib_json

    from batcher.io.formats.sql._common import require_module

    assert require_module("json", extra="sql") is stdlib_json


def test_every_extra_a_guard_names_is_declared() -> None:
    """The bug class this file exists for: a hint pointing at a nonexistent extra.

    A literal ``extra="..."`` argument is the whole install command the user is told to run, so
    an undeclared one is not a cosmetic slip — it is the error's only actionable line being
    wrong. Read from the source rather than by calling every guard, because most of them need
    the dependency absent to fire.
    """
    declared = _declared_extras()
    pattern = re.compile(r'extra=(?:"([a-z0-9_-]+)"|\'([a-z0-9_-]+)\')')
    undeclared: dict[str, set[str]] = {}
    for path in sorted(_PACKAGE.rglob("*.py")):
        for match in pattern.finditer(path.read_text()):
            extra = match.group(1) or match.group(2)
            if extra not in declared:
                undeclared.setdefault(path.relative_to(_PACKAGE).as_posix(), set()).add(extra)
    assert not undeclared, (
        "these guards tell the user to install an extra that pyproject.toml does not "
        f"declare, so the install command in the error fails: {undeclared}"
    )


def test_no_install_hint_names_the_wrong_distribution() -> None:
    """It is `batcher-engine`; one hint said `batcher`, which installs someone else's package."""
    offenders = [
        path.relative_to(_PACKAGE).as_posix()
        for path in sorted(_PACKAGE.rglob("*.py"))
        if re.search(r"pip install ['\"]?batcher\[", path.read_text())
    ]
    assert not offenders, f"these name a distribution that is not this one: {offenders}"


def test_the_interop_adapters_all_go_through_the_shared_guard() -> None:
    """`io.interop` had its own `_missing`, whose errors were not `ImportError`s."""
    source = (_PACKAGE / "io" / "interop.py").read_text()
    assert "def _missing(" not in source
    assert source.count("require(") >= 8, "one guarded adapter per optional framework"
