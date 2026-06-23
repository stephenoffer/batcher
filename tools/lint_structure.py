#!/usr/bin/env python3
"""Structural fitness checker — keeps Batcher from regrowing v1's bloat.

v1 collapsed under 5,236 files, 2,951-line modules, a 61-method god class, a
1,597-line ``__init__.py``, and 8-15-level-deep directories. This script makes those
failure modes mechanical: it fails the commit (pre-commit hook) when a file, directory,
or class crosses a size limit, and warns on the softer smells.

Run it directly to scan the whole repo::

    python tools/lint_structure.py        # or: just lint-structure

Limits live in this file (single source of truth, mirrored by
``.claude/rules/maintainability.md``). Genuine, justified exceptions go in
``STRUCTURE_ALLOW`` with a reason — never a scattered inline marker — and the active
allowlist is printed on every run so exemptions stay visible.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

# --- Limits (mirror .claude/rules/maintainability.md) -----------------------------

PY_HARD = 500  # Python module hard ceiling (lines)
PY_SOFT = 400  # Python module soft target (warn)
RUST_HARD = 800  # Rust file ceiling, EXCLUDING the trailing #[cfg(test)] module
DIR_MAX_FILES = 12  # entries per directory (excl. __pycache__)
DIR_MAX_DEPTH = 5  # directory levels under a package/src root
INIT_MAX = 120  # __init__.py is a re-export shim, not a code dump
FUNC_SOFT = 60  # function length soft guideline (warn)
METHODS_SOFT = 25  # public methods per class (warn) — fluent builders excepted

BANNED_FILENAMES = {"utils.py", "helpers.py", "common.py", "misc.py"}

# Roots whose subtree is governed (and the depth origin for each).
PY_ROOT = Path("python/batcher")
CRATE_SRC_GLOB = "crates/*/src"

# Fluent builders / namespace accessors: breadth is the sanctioned Polars pattern
# (thin builders + .str/.dt/... accessors), so the public-method *warning* is muted
# for them. They are still bound by the hard file-size limit.
FLUENT_BUILDERS = {"Expr", "Dataset", "GroupBy", "CaseBuilder", "Reader", "Writer"}
_ACCESSOR_RE = re.compile(r"Namespace$")

# Justified, visible exemptions from the hard file-size check only: path -> reason.
# Empty: every oversized file from the v1->v2 structural refactor has been split.
# Add an entry only with a one-line reason, and only when an invariant genuinely
# blocks a split (see .claude/rules/maintainability.md).
STRUCTURE_ALLOW: dict[str, str] = {
    # The one Expr hierarchy: the base class plus the result nodes its own methods
    # construct (Cast/MathExpr/AggExpr/Coalesce/…). They are mutually referential, so
    # splitting across modules forces a fragile base<->subclass import cycle — the
    # one-Expr invariant (rust-engine.md) wins over the line limit here.
    "python/batcher/plan/expr_ir/core.py": "one-Expr hierarchy; split forces a base/subclass import cycle",
    # Dataset is the canonical wide fluent builder (rust-engine/maintainability rules
    # name it as legitimately wide); its heavy method bodies are already extracted to
    # dataset/_build.py, leaving thin methods + docstrings that shouldn't be cut.
    "python/batcher/api/dataset/frame.py": "Dataset fluent builder; bodies in _build.py",
}

fails: list[str] = []
warns: list[str] = []


def fail(msg: str) -> None:
    fails.append(msg)


def warn(msg: str) -> None:
    warns.append(msg)


# --- Helpers ----------------------------------------------------------------------


def rust_code_lines(text: str) -> int:
    """Line count excluding the trailing top-level ``#[cfg(test)]`` module.

    The codebase convention is one trailing unit-test module per file; counting it
    would penalize good test density. We cut at the first column-0 ``#[cfg(test)]``.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.rstrip() == "#[cfg(test)]":  # top-level attribute, no indentation
            return i
    return len(lines)


def class_public_methods(node: ast.ClassDef) -> list[str]:
    out = []
    for n in node.body:
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and not n.name.startswith("_"):
            out.append(n.name)
    return out


def func_length(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    return (node.end_lineno or node.lineno) - node.lineno + 1


# --- Per-file checks --------------------------------------------------------------


def check_python_file(path: Path) -> None:
    rel = path.as_posix()
    text = path.read_text()
    n = len(text.splitlines())

    if path.name in BANNED_FILENAMES:
        fail(f"{rel}: banned grab-bag filename '{path.name}' — use a purpose-named module")

    if path.name == "__init__.py":
        if n > INIT_MAX:
            fail(f"{rel}: __init__.py is {n} lines (limit {INIT_MAX}) — re-exports only")
        try:
            tree = ast.parse(text)
        except SyntaxError as e:
            fail(f"{rel}: syntax error: {e}")
            return
        logic = [
            d.name
            for d in tree.body
            if isinstance(d, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        ]
        if logic:
            warn(f"{rel}: __init__.py defines {logic} — prefer re-exports, move logic to a module")
        return

    allow = STRUCTURE_ALLOW.get(rel)
    if n > PY_HARD and allow is None:
        fail(f"{rel}: {n} lines (limit {PY_HARD})")
    elif PY_SOFT < n <= PY_HARD and allow is None:
        warn(f"{rel}: {n} lines (soft target {PY_SOFT})")

    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        fail(f"{rel}: syntax error: {e}")
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.name not in FLUENT_BUILDERS and not _ACCESSOR_RE.search(node.name):
                pub = class_public_methods(node)
                if len(pub) > METHODS_SOFT:
                    warn(
                        f"{rel}: class {node.name} has {len(pub)} public methods "
                        f"(soft limit {METHODS_SOFT}) — push breadth to namespace accessors"
                    )
            if node.name.endswith("Mixin"):
                warn(f"{rel}: class {node.name} is a Mixin — prefer composition/accessors")
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            length = func_length(node)
            if length > FUNC_SOFT:
                warn(f"{rel}: function {node.name} is {length} lines (soft limit {FUNC_SOFT})")


def check_rust_file(path: Path) -> None:
    rel = path.as_posix()
    if path.name in BANNED_FILENAMES:
        fail(f"{rel}: banned grab-bag filename '{path.name}'")
    code = rust_code_lines(path.read_text())
    total = len(path.read_text().splitlines())
    allow = STRUCTURE_ALLOW.get(rel)
    if code > RUST_HARD and allow is None:
        fail(f"{rel}: {code} code lines (limit {RUST_HARD}; {total} incl. tests)")


# --- Directory checks -------------------------------------------------------------


_ARTIFACT_SUFFIXES = {".so", ".pyc", ".pyd", ".dylib"}


def check_dirs(root: Path, depth_origin: Path) -> None:
    for d in [root, *(p for p in root.rglob("*") if p.is_dir())]:
        if d.name in {"__pycache__", "target"}:
            continue
        # Count *files* a reader has to scan — subdirectories (subsystems/packages) are
        # how we tame breadth, not overcrowding. Build artifacts don't count.
        files = [
            e
            for e in d.iterdir()
            if e.is_file() and not e.is_symlink() and e.suffix not in _ARTIFACT_SUFFIXES
        ]
        if len(files) > DIR_MAX_FILES:
            fail(f"{d.as_posix()}/: {len(files)} files (limit {DIR_MAX_FILES})")
        rel_depth = len(d.relative_to(depth_origin).parts)
        if rel_depth > DIR_MAX_DEPTH:
            fail(f"{d.as_posix()}/: nesting depth {rel_depth} (limit {DIR_MAX_DEPTH})")


# --- Main -------------------------------------------------------------------------


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    import os

    os.chdir(repo)

    if PY_ROOT.is_dir():
        for p in PY_ROOT.rglob("*.py"):
            if "__pycache__" not in p.parts:
                check_python_file(p)
        check_dirs(PY_ROOT, PY_ROOT.parent)

    for src in sorted(Path().glob(CRATE_SRC_GLOB)):
        for p in src.rglob("*.rs"):
            if "target" not in p.parts:
                check_rust_file(p)
        check_dirs(src, src)

    if STRUCTURE_ALLOW:
        print(f"structure allowlist ({len(STRUCTURE_ALLOW)} active exemptions):")
        for path, reason in STRUCTURE_ALLOW.items():
            print(f"  - {path}: {reason}")
        print()

    for w in sorted(warns):
        print(f"warn: {w}")
    for f in sorted(fails):
        print(f"FAIL: {f}")

    print()
    if fails:
        print(f"lint-structure: {len(fails)} hard violation(s), {len(warns)} warning(s)")
        return 1
    print(f"lint-structure: OK ({len(warns)} warning(s), {len(STRUCTURE_ALLOW)} allowlisted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
