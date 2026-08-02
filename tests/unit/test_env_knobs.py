"""Every `BATCHER_*` environment variable the code reads must be declared in one place.

`Config` is the documented configuration contract. Beside it a second one had grown: 38
`BATCHER_*` variables read inline with `os.environ.get(...)`, each carrying its own literal
default, spread across `io`, `dist`, `core` and `_internal`. Being env-only is a legitimate
choice for a last-resort operator knob — being *undiscoverable* is not. A variable read at its
point of use appears in no schema, is validated by nothing, is absent from the docs, and can
only be found by reading the line that reads it. Two of them could disagree about the default
for one concept with nothing to say so.

This pins the surface in both directions: a knob the code reads but nobody declared, and a
knob declared but nothing reads. Neither is a crash, which is why neither would otherwise be
noticed.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from batcher.config.env import ENV_KNOBS

pytestmark = pytest.mark.unit

PACKAGE = pathlib.Path(__file__).resolve().parents[2] / "python" / "batcher"


def _read_env_names() -> dict[str, set[str]]:
    """`BATCHER_*` variable -> the modules that read it."""
    found: dict[str, set[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts or path.name == "env.py":
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - a module mid-edit by another session
            continue
        for node in ast.walk(tree):
            name = _env_name(node)
            if isinstance(name, str) and name.startswith("BATCHER_"):
                found.setdefault(name, set()).add(path.relative_to(PACKAGE).as_posix())
    return found


def _env_name(node: ast.AST) -> str | None:
    """The literal variable name this node reads from the environment, if it does.

    Covers the three spellings in the tree: `os.getenv("X")`, `os.environ.get("X")` and
    `os.environ["X"]`.
    """
    if isinstance(node, ast.Subscript):
        value = node.value
        if isinstance(value, ast.Attribute) and value.attr == "environ":
            return node.slice.value if isinstance(node.slice, ast.Constant) else None
        return None
    if not isinstance(node, ast.Call) or not node.args:
        return None
    first = node.args[0]
    if not isinstance(first, ast.Constant):
        return None
    func = node.func
    if getattr(func, "attr", None) == "getenv":
        return first.value
    if getattr(func, "attr", None) in ("get", "pop"):
        owner = getattr(func, "value", None)
        if isinstance(owner, ast.Attribute) and owner.attr == "environ":
            return first.value
    return None


def test_every_env_knob_the_code_reads_is_declared():
    undeclared = {n: sorted(m) for n, m in _read_env_names().items() if n not in ENV_KNOBS}

    assert not undeclared, (
        f"{len(undeclared)} BATCHER_* variable(s) are read but not declared in "
        f"`config/env.py`: {undeclared}.\nA knob nobody can enumerate is a knob nobody can "
        "document, validate, or find. Declare it there — or, if it deserves a type, a default "
        "and a place in the docs, put it in `Config` instead."
    )


def _mentioned_names() -> set[str]:
    """Every `BATCHER_*` string literal anywhere in the package.

    Deliberately weaker than `_read_env_names`. Some knobs are read indirectly — the deadline
    reader walks a `_DEADLINE_VARS` tuple of candidate names — and an AST scan for a literal
    `getenv("X")` cannot see those. For the "is this declaration stale?" direction a mention is
    enough: a name that appears nowhere in the source is certainly not being read.
    """
    names: set[str] = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts or path.name == "env.py":
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - a module mid-edit by another session
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith("BATCHER_")
            ):
                names.add(node.value)
    return names


def test_every_declared_env_knob_is_actually_read():
    read = _mentioned_names()
    stale = sorted(name for name in ENV_KNOBS if name not in read)

    assert not stale, (
        f"{stale} are declared in `config/env.py` but nothing reads them. Either the code that "
        "read them was deleted and the declaration outlived it, or the name was changed on one "
        "side only — which silently turns the knob into a no-op the operator still sets."
    )


def test_no_knob_is_declared_twice_under_different_spellings():
    """Two names for one concept is the drift this registry exists to prevent."""
    purposes: dict[str, list[str]] = {}
    for name, purpose in ENV_KNOBS.items():
        purposes.setdefault(purpose.strip().lower(), []).append(name)
    duplicated = {p: n for p, n in purposes.items() if len(n) > 1}

    assert not duplicated, f"one purpose, several variables: {duplicated}"
