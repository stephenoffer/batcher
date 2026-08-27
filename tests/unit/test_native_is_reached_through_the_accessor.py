"""Exactly one module may import `batcher._native` directly, and this is the check.

`CLAUDE.md` opens its silent-failure section with this rule: "**Never `import
batcher._native`.** Always `from batcher._internal.native import engine`." The reason is
mechanical -- the static import graph cannot see into a compiled extension, so grimp
attributes such an import to the **root** `batcher` package, which re-exports `api`, which
imports every subsystem. The result is a phantom `core -> batcher -> api -> kyber` cycle. That
is not hypothetical: it broke all six independence directions at once, when a new IO format
added the import and was not on the allowlist that used to paper over it.

**Nothing enforces the rule.** `pyproject.toml` deleted that allowlist on purpose and says so
-- "If you find yourself adding an entry here, you have almost certainly bypassed that shim --
don't" -- but a deleted allowlist is not a gate. And `lint-layers` structurally *cannot* be
one: the same blindness that causes the phantom edge means grimp sees the import as an edge to
`batcher`, and `_internal` is not a member of the independence contract, so the offending
import produces no violation there at all. The rule has been guidance with no teeth.

One module is exempt, and it has to be. `_internal/errors/hierarchy.py` lifts five error types
out of the extension so they can be `raise`d and caught by type. It cannot use the accessor,
because `_internal/native.py` itself does `from batcher._internal.errors import BackendError`
-- routing the errors module through the accessor would be a genuine cycle inside `_internal`,
not a phantom one. So the exemption is recorded here by path, with that reason, and the
assertion is that it stays a set of exactly one.
"""

from __future__ import annotations

import pathlib
import re

import pytest

pytestmark = pytest.mark.unit

_PACKAGE = pathlib.Path(__file__).resolve().parents[2] / "python" / "batcher"

#: Matches both spellings, at any indentation, including inside a `try:`.
_DIRECT_IMPORT = re.compile(
    r"^\s*(?:from\s+batcher\._native\s+import|import\s+batcher\._native)", re.M
)

#: The one module allowed to bypass the accessor, and why.
_ALLOWED: dict[str, str] = {
    "_internal/errors/hierarchy.py": (
        "lifts the engine's error types so callers can catch them by type; cannot use "
        "`_internal.native`, which imports `_internal.errors` and would cycle"
    ),
}


def _offenders() -> list[str]:
    found = []
    for path in _PACKAGE.rglob("*.py"):
        if _DIRECT_IMPORT.search(path.read_text()):
            found.append(str(path.relative_to(_PACKAGE)))
    return sorted(found)


def test_only_the_recorded_module_imports_the_extension_directly():
    unexpected = sorted(set(_offenders()) - set(_ALLOWED))
    assert not unexpected, (
        f"{unexpected} import `batcher._native` directly. Use "
        "`from batcher._internal.native import engine`. A direct import is attributed to the "
        "root `batcher` package, forging a `core -> batcher -> api -> kyber` cycle that "
        "breaks all six independence contracts -- and `lint-layers` cannot see it, which is "
        "why this test exists"
    )


def test_the_recorded_exemption_is_still_real():
    """An exemption for a module that no longer needs it is an invitation to copy it."""
    stale = sorted(set(_ALLOWED) - set(_offenders()))
    assert not stale, (
        f"{stale} no longer imports `batcher._native` directly, so its entry in `_ALLOWED` "
        "is stale. Delete it -- a recorded exception outlives the reason for it and becomes "
        "precedent"
    )


def test_the_accessor_really_would_cycle():
    """Pin the reason the exemption exists, so it cannot quietly stop being true.

    If `_internal/native.py` ever stops importing `_internal.errors`, the cycle argument
    evaporates and `hierarchy.py` should go through the accessor like everything else.
    """
    accessor = (_PACKAGE / "_internal" / "native.py").read_text()
    assert re.search(r"^from batcher\._internal\.errors import", accessor, re.M), (
        "`_internal/native.py` no longer imports `_internal.errors`, so routing "
        "`hierarchy.py` through the accessor would no longer cycle -- the exemption in "
        "`_ALLOWED` has lost its justification and should be retired"
    )


def test_the_scan_reads_the_package():
    """Guard against a vacuous suite.

    Both assertions above are set differences, and an empty scan satisfies the first one
    trivially. Pin that the walk sees the package and that the pattern matches the real
    spelling -- which is indented, inside a `try:`, and would be missed by an anchored
    `^from` without `\\s*`.
    """
    files = list(_PACKAGE.rglob("*.py"))
    assert len(files) > 500, f"only {len(files)} modules found; the package walk is broken"
    assert _offenders() == list(_ALLOWED), (
        f"expected exactly the recorded exemption, found {_offenders()}"
    )
