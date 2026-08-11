"""Shared state for every detector: parsed sources, and a global identifier index.

The index is the expensive part and the reason it lives here — one pass over the tree
answers `is this name mentioned anywhere else?` for every detector that asks.
"""

from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "python" / "batcher"
CRATES = ROOT / "crates"

#: Everything that can *reference* a name. Cast wide: a name cited only in a doc or a
#: benchmark is still referenced, and calling it dead would be wrong.
SEARCH_ROOTS = ("python", "tests", "benchmarks", "examples", "tools", "docs", "crates")
SEARCH_SUFFIXES = {".py", ".rs", ".md", ".toml", ".txt", ".pyi"}

#: Generated output to keep out of the reference index, anchored to the directory that
#: produces it rather than matched by component name. Matching a bare ``_build`` component
#: also excluded the real package ``python/batcher/api/dataset/_build/`` — which is still
#: *scanned for definitions*, so every name referenced only from those three modules read
#: as dead. `_all_bounded` was reported as unreferenced while `sessions.py` imports it.
BUILD_OUTPUT_PREFIXES = ("docs/_build",)

#: Decorators that make a definition reachable without its name ever appearing at a call
#: site. Kyber rules, IO formats, and pytest fixtures all work this way.
REGISTERED_BY = ("register", "rule", "fixture", "hookimpl", "overload", "abstractmethod")
#: Accessor decorators — the name is reached through an attribute, not a call.
ACCESSOR_DECORATORS = ("property", "setter", "getter", "deleter", "cached_property")

#: A near-duplicate needs this much body overlap to be worth a look. Below ~0.85 the pairs
#: are genuinely different functions that happen to share a shape.
NEAR_DUP_JACCARD = 0.85
#: ...and this many statements, so two-line guards don't flood the report.
MIN_STATEMENTS = 5
#: Methods that are structurally identical across unrelated classes *by design* — an
#: `__init__` that assigns its arguments, an `__eq__` that compares fields. Same exclusion,
#: for the same reason, as `lint_duplication.py::SKIP_NAMES`; keep the two lists agreeing.
DUP_SKIP_NAMES = {"__init__", "__eq__", "__hash__", "__repr__", "__str__", "__enter__", "__exit__"}

#: AST-node count above which a `try` body is too big to be guarded by a broad `except`.
#: Roughly "more than a couple of calls" — enough room for an unanticipated bug to hide.
BROAD_TRY_NODES = 60

#: Known-dead-by-design, with the reason. Same ledger discipline as the other linters:
#: an entry is visible debt, not an amnesty.
DEAD_ALLOW: dict[str, str] = {
    "crates/bc-udf": (
        "documented in CLAUDE.md as not wired into bc-py — the UDF/inference plane is not "
        "on a live path, so its public surface is unreferenced on purpose"
    ),
    "crates/bc-secrets/src/lib.rs": (
        "`clear_cache` is reached only by this crate's own tests, and its docstring says why "
        "it is still `pub`: the resolution cache is process-global with a TTL, so a host that "
        "rotates a credential needs the same escape hatch the tests use to make the new value "
        "visible before the TTL expires. Cache invalidation is the API, not an unused seam"
    ),
}

#: The handful of handlers that are silent on purpose and cannot be made otherwise, keyed by
#: ``file::enclosing-function``. Keep this list at approximately zero entries: the fix for a
#: silent best-effort path is `note_suppressed`, and an exemption is only right when *calling*
#: it is the thing that would fail.
#:
#: Keyed by the enclosing symbol and **not** by line number, because the line-numbered form
#: silently stopped applying: the one entry here read `scan_read.py:171` while the handler had
#: drifted to line 194, so a waived site came back as a `high` finding and the waiver became
#: invisible debt. A key that no longer resolves is now reported as `stale-waiver` rather than
#: quietly ignored — the same failure mode `DUPLICATION_ALLOW` had at `flight_sort.py:332`.
SILENT_ALLOW: dict[str, str] = {
    "python/batcher/dist/executors/scan_read.py::_record_skipped": (
        "this handler *is* the logging path for a skipped split, so anything it could report "
        "would take the same route that just failed — the one place where staying silent is "
        "the only option, and it is documented in the handler body"
    ),
}

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


@dataclass(frozen=True)
class Finding:
    """One thing worth looking at, at one place in the tree."""

    category: str
    severity: str
    path: str
    line: int
    message: str


@dataclass
class Context:
    """Everything the detectors share: parsed sources and a global name index."""

    modules: dict[Path, ast.Module]
    #: The raw text of each parsed module, for the checks that need what `ast` throws away —
    #: comments, chiefly, which are how this codebase marks a silence as deliberate.
    sources: dict[Path, str]
    #: name -> {file: how many times the token appears}. A definition is dead when no other
    #: file mentions it *and* its own file mentions it exactly once (the definition itself).
    name_files: dict[str, dict[str, int]]
    rust_text: dict[Path, str]

    def used_outside(self, name: str, rel: str) -> set[str]:
        """Files other than `rel` that mention `name`."""
        return set(self.name_files.get(name, {})) - {rel}

    def used_inside(self, name: str, rel: str) -> bool:
        """True when `name` appears in its own file somewhere other than its definition."""
        return self.name_files.get(name, {}).get(rel, 0) > 1


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _allowed(rel: str) -> str | None:
    for prefix, reason in DEAD_ALLOW.items():
        if rel.startswith(prefix):
            return reason
    return None


# --- Index ------------------------------------------------------------------------


_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def build_context() -> Context:
    """Parse the package, and index every identifier token in the searchable tree."""
    modules: dict[Path, ast.Module] = {}
    sources: dict[Path, str] = {}
    for path in sorted(PKG.rglob("*.py")):
        try:
            text = path.read_text()
            modules[path] = ast.parse(text)
        except (SyntaxError, UnicodeDecodeError):
            continue
        sources[path] = text

    name_files: dict[str, dict[str, int]] = defaultdict(dict)
    rust_text: dict[Path, str] = {}
    for root in SEARCH_ROOTS:
        for path in sorted((ROOT / root).rglob("*")):
            if path.suffix not in SEARCH_SUFFIXES or not path.is_file():
                continue
            rel = _rel(path)
            if "__pycache__" in path.parts or rel.startswith(BUILD_OUTPUT_PREFIXES):
                continue
            try:
                text = path.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            if path.suffix == ".rs":
                rust_text[path] = text
            for token, count in Counter(_WORD.findall(text)).items():
                name_files[token][rel] = count
    return Context(modules=modules, sources=sources, name_files=name_files, rust_text=rust_text)


def _decorator_names(node: ast.AST) -> list[str]:
    """Decorator names on `node`, flattened — a definition can be reachable through one
    without its own name ever appearing at a call site."""
    out: list[str] = []
    for dec in getattr(node, "decorator_list", []):
        for sub in ast.walk(dec):
            if isinstance(sub, ast.Name):
                out.append(sub.id)
            elif isinstance(sub, ast.Attribute):
                out.append(sub.attr)
    return out
