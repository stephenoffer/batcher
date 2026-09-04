"""No module may import `batcher._native` directly, and this is the check.

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

**There is now no exemption at all**, and the way the last one went is worth keeping. It was
`_internal/errors/hierarchy.py`, which lifted five error types *out of* the extension so they
could be raised and caught by type, and it could not use the accessor because
`_internal/native.py` itself imports `_internal.errors` -- a genuine cycle, not a phantom one.
The reason it disappeared is that lifting them out was itself the bug: a type built by Rust's
`create_exception!` has `RuntimeError` as its base and cannot be re-parented afterwards, so
all five were `RuntimeError` subclasses and none of them was a `BatcherError` in any built
install. Inverting the ownership -- Python defines the classes, `bc_py::errors` looks them up
by name on the error path -- fixed the hierarchy and removed the import in the same move.

So the assertion is now that the set is **empty**. Keep it that way: an exemption is a
precedent, and the one recorded here outlived its reason without anyone noticing until the
scan said so.
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

#: Modules allowed to bypass the accessor, and why. Empty, and meant to stay empty -- see
#: the module docstring for how the last entry came to be retired.
_ALLOWED: dict[str, str] = {}


def _offenders() -> list[str]:
    found = []
    for path in _PACKAGE.rglob("*.py"):
        if _DIRECT_IMPORT.search(path.read_text()):
            found.append(str(path.relative_to(_PACKAGE)))
    return sorted(found)


def test_no_module_imports_the_extension_directly():
    """The contract, now that nothing is exempt from it."""
    offenders = sorted(set(_offenders()) - set(_ALLOWED))
    assert not offenders, (
        f"{offenders} import `batcher._native` directly. Use "
        "`from batcher._internal.native import engine`. A direct import is attributed to the "
        "root `batcher` package, forging a `core -> batcher -> api -> kyber` cycle that "
        "breaks all six independence contracts -- and `lint-layers` cannot see it, which is "
        "why this test exists"
    )


def test_the_allowlist_is_empty():
    """An exemption is a precedent, and the last one outlived its reason unnoticed.

    Asserted separately from the scan so the two failures read differently: one says a
    module started bypassing the accessor, this one says somebody wrote down permission to.
    A new entry should be argued in review, not appended.
    """
    assert _ALLOWED == {}, (
        f"{sorted(_ALLOWED)} are recorded as allowed to import `batcher._native` directly. "
        "The exemption this file used to carry was itself the symptom of a bug -- see the "
        "module docstring -- so adding one back needs an argument, not a line"
    )


def test_the_scan_reads_the_package():
    """Guard against a vacuous suite.

    The assertions above are set differences, and an empty scan satisfies both trivially.
    Pin that the walk sees the package, and that the pattern matches the spelling it has to
    catch -- indented, inside a `try:`, which an anchored `^from` without `\\s*` would miss.
    """
    files = list(_PACKAGE.rglob("*.py"))
    assert len(files) > 500, f"only {len(files)} modules found; the package walk is broken"
    assert _DIRECT_IMPORT.search("    from batcher._native import Foo\n"), (
        "the pattern no longer matches an indented import, so the scan cannot see the "
        "spelling it exists to catch"
    )
    assert not _DIRECT_IMPORT.search("from batcher._internal.native import engine\n"), (
        "the pattern matches the accessor, so every module would read as an offender"
    )
