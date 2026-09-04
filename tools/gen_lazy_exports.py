"""Generate `batcher/_exports.py`, the routing tables behind the lazy re-export façades.

`batcher`, `batcher.api`, and `batcher.api.session` are pure re-export façades. Binding
their names eagerly meant importing the entire control plane to reach any one of them:
545 ms and 609 modules for a script that wanted `bt.col`, paid once per process, in an
engine whose scaling target is millions of processes. They now resolve each name lazily
(PEP 562, via `batcher._lazy`), which needs a table mapping every public name to the
module that *defines* it — and that table has to be static, because computing it means
importing everything, which is the cost the laziness exists to avoid.

**The input is each façade's own `if TYPE_CHECKING:` block.** That block already has to
list the surface accurately for type checkers and editors to resolve it, so making it
the generator's input keeps one declaration instead of two that can disagree. A star
import contributes the named module's whole `__all__`; an explicit `as` import
contributes that one name; a star import from a *lower* façade contributes whatever
this run just computed for it, so the three are generated bottom-up in one pass.

Run: `just gen-exports`. `tests/unit/test_lazy_exports.py` fails when the committed
file is stale, so adding a public name is a regeneration rather than a silent hole.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "python" / "batcher" / "_exports.py"

#: The façades, generated bottom-up so a star import from one that is already done
#: contributes the names this run computed rather than the committed (possibly stale)
#: table. Each entry is `(package, table constant, shadow constant)`.
FACADES = [
    ("batcher.api.session", "SESSION_EXPORTS", "SESSION_SHADOWED"),
    ("batcher.api", "API_EXPORTS", "API_SHADOWED"),
    ("batcher", "EXPORTS", "ROOT_SHADOWED"),
]

HEADER = '''"""The routing tables behind Batcher's lazy re-export façades — GENERATED, do not edit.

Every name each façade exports maps to the module that *defines* it, so a lazy attribute
lookup imports one leaf module rather than the whole API. Routing to the definition
rather than to the façade above it is the point: `bt.col` costs the expression package,
not the IO registry and the SQL session as well.

A ``"module:attr"`` value is a name renamed on re-export, such as `concat_str` from
`string.building.concat`. A ``*_SHADOWED`` set holds the exported names that are also
submodule names of their package, which the import machinery would otherwise overwrite.

Regenerate with `just gen-exports`; `tests/unit/test_lazy_exports.py` gates the drift.
"""

from __future__ import annotations

__all__ = [
'''


def declared_surface(package: str, computed: dict[str, dict[str, object]]) -> dict[str, object]:
    """The names a façade declares, in order, resolved to the objects they name.

    Args:
        package: The façade package, whose `TYPE_CHECKING` block is the declaration.
        computed: Façade package -> its already-resolved surface, for star imports that
            name a lower façade.

    Returns:
        Public name -> the object the eager façade would have bound.
    """
    source = Path(importlib.import_module(package).__file__ or "").read_text()
    # Every `TYPE_CHECKING` block, concatenated in file order. More than one because
    # **later binds win**, exactly as they did in the eager façade — `session`'s `concat`
    # deliberately shadows `functions`', and `read` is the reader *namespace* shadowing
    # `session`'s plain `read` function. isort sorts within a block but never across
    # blocks, so a trailing block is how a façade expresses "this one binds last".
    block = [
        stmt
        for node in ast.parse(source).body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING"
        for stmt in node.body
    ]
    surface: dict[str, object] = {}
    for stmt in block:
        if not isinstance(stmt, ast.ImportFrom) or stmt.module is None:
            continue
        if any(alias.name == "*" for alias in stmt.names):
            if stmt.module in computed:
                surface.update(computed[stmt.module])
                continue
            module = importlib.import_module(stmt.module)
            for name in module.__all__:
                surface[name] = getattr(module, name)
            continue
        module = importlib.import_module(stmt.module)
        for alias in stmt.names:
            # Assignment, not `setdefault`: a later binding replaces an earlier one while
            # keeping the name's original position, which is what reproduces both the eager
            # façade's values and its `__all__` ordering.
            surface[alias.asname or alias.name] = getattr(module, alias.name)
    return surface


def route(name: str, obj: object) -> str:
    """Where a lazy lookup of `name` should go to find exactly `obj`.

    Args:
        name: The public name.
        obj: The object the eager façade bound to it.

    Returns:
        ``"module"``, or ``"module:attr"`` when the name was renamed on re-export.

    Raises:
        SystemExit: If the object cannot be reached from the module that defines it,
            which would leave the lazy façade unable to serve the name.
    """
    defining = getattr(obj, "__module__", None)
    if isinstance(obj, ModuleType):
        defining = obj.__name__.rpartition(".")[0] or obj.__name__
    if not (isinstance(defining, str) and defining.startswith("batcher")):
        raise SystemExit(f"unroutable public name: {name!r} (__module__={defining!r})")
    module = importlib.import_module(defining)
    if getattr(module, name, None) is obj:
        return defining
    alias = next((k for k, v in vars(module).items() if v is obj and not k.startswith("__")), None)
    if alias is None:
        raise SystemExit(f"{name!r} is not reachable from its defining module {defining!r}")
    return f"{defining}:{alias}"


def submodules(package: str) -> set[str]:
    """The importable submodule names of `package`, which shadow same-named exports."""
    directory = Path(importlib.import_module(package).__file__ or "").parent
    names = {p.stem for p in directory.glob("*.py") if p.name != "__init__.py"}
    return names | {p.name for p in directory.iterdir() if (p / "__init__.py").exists()}


def collect() -> list[tuple[str, str, dict[str, str], list[str]]]:
    """Resolve every façade's surface into a routing table and a shadow list.

    Returns:
        One `(package, table constant, table, shadowed names)` per façade.
    """
    computed: dict[str, dict[str, object]] = {}
    out = []
    for package, table_name, shadow_name in FACADES:
        surface = declared_surface(package, computed)
        computed[package] = surface
        table = {name: route(name, obj) for name, obj in surface.items()}
        shadowed = sorted(submodules(package) & set(table))
        out.append((table_name, shadow_name, table, shadowed))
    return out


def render(tables: list[tuple[str, str, dict[str, str], list[str]]]) -> str:
    """Render the routing tables as a formatted Python module.

    Args:
        tables: What `collect` returned.

    Returns:
        The full source text of `batcher/_exports.py`.
    """
    exported = [n for table_name, shadow_name, _, _ in tables for n in (table_name, shadow_name)]
    lines = [HEADER, *(f'    "{n}",\n' for n in sorted(exported)), "]\n"]
    for table_name, shadow_name, table, shadowed in tables:
        lines.append(f"\n#: Public name -> the module defining it, for `{table_name}`'s façade.\n")
        lines.append(f"{table_name}: dict[str, str] = {{\n")
        # Insertion order is each façade's declaration order, not alphabetical: `__all__`
        # is rebuilt from these keys and the docs-coverage gate reads it as a sequence.
        lines.extend(f'    "{name}": "{module}",\n' for name, module in table.items())
        lines.append("}\n")
        lines.append("\n#: Exports of that package which are also submodule names of it.\n")
        # Double quotes and a `[...]` literal so the output is already what `ruff format`
        # would produce; the drift gate compares text, and a quote style would fail it.
        members = ", ".join(f'"{name}"' for name in shadowed)
        lines.append(f"{shadow_name}: frozenset[str] = frozenset([{members}])\n")
    return "".join(lines)


def main() -> None:
    """Regenerate the committed routing tables."""
    tables = collect()
    TARGET.write_text(render(tables))
    summary = ", ".join(f"{t}={len(tbl)}" for t, _, tbl, _ in tables)
    print(f"wrote {TARGET.relative_to(ROOT)} ({summary})")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "python"))
    main()
