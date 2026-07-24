"""Dead code: definitions and `pub` items nothing outside their own file reaches."""

from __future__ import annotations

import ast
import re
from collections import Counter
from collections.abc import Iterator

from tools.audit.context import (
    _WORD,
    ACCESSOR_DECORATORS,
    PKG,
    REGISTERED_BY,
    Context,
    Finding,
    _allowed,
    _decorator_names,
    _rel,
)

# --- Dead code --------------------------------------------------------------------


def _module_exports(tree: ast.Module) -> set[str]:
    """The names a module names in its `__all__`, which are exported on purpose."""
    for node in tree.body:
        assigns_all = isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        )
        if assigns_all and isinstance(node.value, ast.List | ast.Tuple):
            return {
                e.value
                for e in node.value.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            }
    return set()


def _root_public_surface() -> set[str]:
    """The names `import batcher as bt` exposes — exported on purpose, never 'dead'."""
    try:
        tree = ast.parse((PKG / "__init__.py").read_text())
    except (SyntaxError, OSError):
        return set()
    return _module_exports(tree)


def detect_dead_python(ctx: Context) -> Iterator[Finding]:
    """Definitions in the control plane that no other file mentions by name.

    Covers module-level functions and classes, plus underscore-prefixed methods (a private
    method nobody calls is unambiguously dead). Registered and accessor definitions are
    skipped: their names legitimately never appear at a call site.
    """
    public = _root_public_surface()
    for path, tree in ctx.modules.items():
        rel = _rel(path)
        exports = _module_exports(tree)
        candidates: list[tuple[ast.AST, str]] = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                candidates.append((node, "definition"))
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if not isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef):
                        continue
                    if sub.name.startswith("_") and not sub.name.startswith("__"):
                        candidates.append((sub, "private method"))

        for node, kind in candidates:
            name, lineno = node.name, node.lineno
            if name.startswith("__") or name in exports:
                continue
            decorators = _decorator_names(node)
            if any(hint in d.lower() for d in decorators for hint in REGISTERED_BY):
                continue
            if any(d in ACCESSOR_DECORATORS for d in decorators):
                continue
            if ctx.used_inside(name, rel):
                continue  # used within its own module — a local helper, not dead
            elsewhere = ctx.used_outside(name, rel)
            if not elsewhere:
                yield Finding(
                    "dead-python",
                    "high" if kind == "private method" else "medium",
                    rel,
                    lineno,
                    f"{kind} `{name}` is referenced nowhere else in the tree",
                )
            elif name not in public and all(f.endswith("__init__.py") for f in elsewhere):
                yield Finding(
                    "dead-python",
                    "low",
                    rel,
                    lineno,
                    f"`{name}` is re-exported by {sorted(elsewhere)[0]} but used nowhere",
                )


_RUST_PUB = re.compile(
    r"^pub(?:\(crate\))?\s+(?:async\s+)?(fn|struct|enum|trait|const|type)\s+(\w+)", re.M
)


def detect_dead_rust(ctx: Context) -> Iterator[Finding]:
    """Crate-level `pub` items in the data plane that only their own tests reach.

    Only column-0 items are considered, so `impl` bodies (whose methods are reached through
    a trait or a receiver) never appear here. Non-test uses are counted against the file
    with its `#[cfg(test)]` module cut off, which is what separates "nobody calls this" from
    "only its own unit test calls this" — the second is the shape a speculative primitive
    takes, and it is the one worth arguing about.
    """
    for path, text in sorted(ctx.rust_text.items()):
        rel = _rel(path)
        if not rel.startswith("crates/") or _allowed(rel):
            continue
        body = text.split("\n#[cfg(test)]")[0]
        body_counts = Counter(_WORD.findall(body))
        for match in _RUST_PUB.finditer(body):
            kind, name = match.group(1), match.group(2)
            if name in {"new", "main", "default"} or ctx.used_outside(name, rel):
                continue
            if body_counts[name] > 1:
                continue  # used elsewhere in its own module's product code
            line = body[: match.start()].count("\n") + 1
            in_tests = ctx.name_files.get(name, {}).get(rel, 0) > 1
            yield Finding(
                "dead-rust",
                "medium" if in_tests else "high",
                rel,
                line,
                f"pub {kind} `{name}` is reached only by its own unit tests"
                if in_tests
                else f"pub {kind} `{name}` is referenced nowhere in the workspace",
            )
