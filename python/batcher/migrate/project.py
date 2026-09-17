"""What the helper functions a script imports from its own project return.

Real scripts rarely build a `Dataset` inline. The example suite opens every table through
`from _common import tpch`, and an application has its `from .loaders import read_orders`.
Without knowing that `tpch(...)` returns a `Dataset`, every `.head(10)` after it is an unknown
receiver and nothing is rewritten. This module looks the imported module up beside the script
(the script's directory and its parents, the places `sys.path` bootstraps and relative imports
reach), infers what each top-level function returns with the same receiver inference the
rewrite uses, and follows one level of re-export through a package `__init__`.

Nothing is imported or executed; a module that cannot be found or parsed contributes nothing.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from batcher._internal.optional import require
from batcher.migrate.receivers import FunctionRef, infer_module

cst = require("libcst", feature="batcher.migrate", provides="libcst", extra="migrate")

__all__ = ["imported_function_receivers"]

_MAX_PARENTS = 3


def _locate(module: str, start: Path) -> Path | None:
    parts = module.split(".")
    for base in [start, *list(start.parents)[:_MAX_PARENTS]]:
        candidate = base.joinpath(*parts)
        for path in (candidate.with_suffix(".py"), candidate / "__init__.py"):
            if path.is_file():
                return path
    return None


@lru_cache(maxsize=512)
def _module_functions(path: Path, returns_key: int, depth: int) -> dict[str, str]:
    try:
        tree = cst.parse_module(path.read_text())
    except Exception:  # an unreadable or unparseable helper contributes nothing
        return {}
    returns = _RETURNS[returns_key]
    scopes = infer_module(tree, returns)
    found = {
        name: value.receiver
        for name, value in scopes[tree].scope.names.items()
        if isinstance(value, FunctionRef)
    }
    if depth > 0:
        for name, receiver in _reexports(tree, path.parent, returns_key, depth - 1).items():
            found.setdefault(name, receiver)
    return found


def _reexports(tree: object, directory: Path, returns_key: int, depth: int) -> dict[str, str]:
    out: dict[str, str] = {}
    for stmt in tree.body:  # type: ignore[attr-defined]
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            if not isinstance(small, cst.ImportFrom) or isinstance(small.names, cst.ImportStar):
                continue
            source = _module_name(small)
            if source is None or source.startswith("batcher"):
                continue
            located = _locate(source, directory)
            if located is None:
                continue
            functions = _module_functions(located, returns_key, depth)
            for alias in small.names:
                if not (isinstance(alias.name, cst.Name) and alias.name.value in functions):
                    continue
                asname = alias.asname.name if alias.asname else None
                bound = asname.value if isinstance(asname, cst.Name) else alias.name.value
                out[bound] = functions[alias.name.value]
    return out


def _module_name(node: object) -> str | None:
    """The dotted module an import names; a relative import resolves from the script's folder."""
    module = node.module  # type: ignore[attr-defined]
    if module is None:
        return None
    parts = []
    while isinstance(module, cst.Attribute):
        parts.append(module.attr.value)
        module = module.value
    if not isinstance(module, cst.Name):
        return None
    parts.append(module.value)
    return ".".join(reversed(parts))


_RETURNS: dict[int, dict[str, dict[str, str]]] = {}


def imported_function_receivers(
    source: str, path: Path, returns: dict[str, dict[str, str]]
) -> dict[str, str]:
    """Receivers returned by functions the script imports from modules in its own project.

    Args:
        source: The script's source.
        path: Where the script lives, used to find the modules it imports.
        returns: The generated returns table.

    Returns:
        `{bound_name: receiver}` for every imported function whose return is a Batcher receiver.
    """
    key = id(returns)
    _RETURNS[key] = returns
    try:
        tree = cst.parse_module(source)
    except Exception:  # the rewrite itself reports the unparseable file
        return {}
    return _reexports(tree, path.parent, key, depth=2)
