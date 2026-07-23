#!/usr/bin/env python3
"""Generate ``MAP.md`` — the one index of what every file in Batcher is for.

Batcher is 543 Python modules and 147 Rust files across 79 packages and 12 crates.
Every one of them already carries a module docstring (Python) or a ``//!`` doc
(Rust) that states its single responsibility — that convention is enforced
elsewhere and it holds. The problem this script solves is that the index those
docstrings *form* only existed if you opened all 690 files.

So ``MAP.md`` is derived, never written by hand. Each entry's one-liner is the
module's own first docstring line, and each crate's dependency list is read from
its ``Cargo.toml``. A map that restates what the code says can drift from it; a
map that *quotes* the code cannot. Renaming a module or rewriting its docstring
updates the map on the next run, and ``tests/docs/test_map_current.py`` fails the
build if someone forgets to run it.

The hand-written half — the disambiguation tables for confusable name clusters
and the "where does new X go" routing rules — is prose that no docstring can
derive, so it lives in ``tools/map_notes.md`` and is inlined verbatim. Keeping it
in a separate file means this script owns structure and that file owns judgment,
and neither has to be edited to change the other.

Run it::

    python tools/gen_map.py           # rewrite MAP.md
    python tools/gen_map.py --check   # exit 1 if MAP.md is stale (CI + pre-commit)
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import tomllib

REPO = Path(__file__).resolve().parent.parent
PY_ROOT = REPO / "python" / "batcher"
CRATES = REPO / "crates"
OUT = REPO / "MAP.md"
NOTES = Path(__file__).resolve().parent / "map_notes.md"

# The architectural layer of every Python package, mirroring the import matrix in
# `.claude/rules/architecture.md`. Keyed by path relative to `python/batcher`; a
# package inherits its nearest ancestor's layer, so only the roots are listed.
# Read as: a package may import anything with a strictly lower number, never
# higher and never sideways (the layer-3 subsystems are mutually independent).
LAYERS: dict[str, str] = {
    ".": "root",
    "ml": "6 · front-end",
    "_sql": "6 · front-end",
    "api": "5 · conductor",
    "dist": "4 · backend",
    "kyber": "3 · subsystem",
    "carbonite": "3 · subsystem",
    "core": "3 · subsystem",
    "governance": "3 · subsystem",
    "io": "2 · neutral IO",
    "observe": "2 · neutral sinks",
    "plan": "1 · contract",
    "metadata": "1 · contract",
    "config": "0 · utility",
    "_internal": "0 · utility",
}

# Reading order for the Python section: layers top-down, so the file reads like the
# architecture does — the user-facing surface first, the contracts it rests on last.
PY_ORDER = [
    "api",
    "ml",
    "_sql",
    "dist",
    "kyber",
    "carbonite",
    "core",
    "governance",
    "io",
    "observe",
    "plan",
    "metadata",
    "config",
    "_internal",
]

# Reading order for the crates: down the dependency DAG, leaves last.
CRATE_ORDER = [
    "bc-py",
    "bc-interp",
    "bc-runtime",
    "bc-codegen",
    "bc-ir",
    "bc-expr",
    "bc-arrow",
    "bc-sketches",
    "bc-transport",
    "bc-resource",
    "bc-io",
    "bc-udf",
]

_TEST_MOD = re.compile(r"^#\[cfg\(test\)\]", re.MULTILINE)


def layer_of(rel: str) -> str:
    """Return the architectural layer for a package path relative to ``python/batcher``."""
    top = rel.split("/")[0] if rel != "." else "."
    return LAYERS.get(top, "?")


def py_summary(path: Path) -> str:
    """Return the first line of a Python module's docstring, or a placeholder."""
    try:
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    except (SyntaxError, UnicodeDecodeError):
        return "_(unparseable)_"
    if not doc:
        return "_(no docstring)_"
    return " ".join(doc.strip().split("\n")[0].split())


def rust_summary(path: Path) -> str:
    """Return the first sentence of a Rust file's ``//!`` module doc, or a placeholder.

    Unlike a Python docstring summary — which the style gate forces onto one line —
    a ``//!`` opening sentence wraps across several comment lines. Taking only the
    first *line* would cut most of these mid-clause, so the leading paragraph is
    rejoined and split at its first sentence boundary instead.
    """
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("//!"):
            body = line[3:].strip()
            if not body and lines:
                break
            if body:
                lines.append(body)
        elif lines or (line and not line.startswith("//")):
            break
    if not lines:
        return "_(no //! doc)_"
    para = " ".join(" ".join(lines).split())
    return _first_sentence(para)


_SENTENCE_END = re.compile(r"\.\s+(?=[A-Z`(])")
_SUMMARY_MAX = 160


def _first_sentence(text: str) -> str:
    """Cut ``text`` at its first sentence boundary, falling back to a word-boundary trim.

    The boundary test requires the next character to open a new sentence (a capital,
    a backtick, or a paren) so that ``e.g.``, ``i.e.``, and a trailing ``mod.rs`` do
    not read as full stops.
    """
    match = _SENTENCE_END.search(text)
    if match:
        return text[: match.start() + 1]
    if len(text) <= _SUMMARY_MAX:
        return text
    return text[:_SUMMARY_MAX].rsplit(" ", 1)[0] + "…"


def rust_code_lines(path: Path) -> int:
    """Count a Rust file's lines excluding the trailing ``#[cfg(test)]`` module.

    Mirrors ``tools/lint_structure.py`` so the two agree on what "800 lines" means:
    Rust co-locates unit tests, and counting them would punish good test density.
    """
    text = path.read_text(encoding="utf-8")
    match = _TEST_MOD.search(text)
    if match:
        text = text[: match.start()]
    return len(text.splitlines())


def crate_deps(crate: Path) -> list[str]:
    """Return the in-workspace ``bc-*`` dependencies declared by a crate's manifest."""
    manifest = tomllib.loads((crate / "Cargo.toml").read_text(encoding="utf-8"))
    deps = set(manifest.get("dependencies", {})) | set(manifest.get("dev-dependencies", {}))
    return sorted(d for d in deps if d.startswith("bc-"))


def py_packages() -> list[tuple[str, Path]]:
    """Return every Python package under ``python/batcher``, in layer reading order."""
    found = {
        str(p.parent.relative_to(PY_ROOT)): p.parent
        for p in PY_ROOT.rglob("__init__.py")
        if "__pycache__" not in p.parts
    }
    ordered: list[tuple[str, Path]] = []
    if "." in found:
        ordered.append((".", found.pop(".")))
    for top in PY_ORDER:
        for rel in sorted(k for k in found if k == top or k.startswith(f"{top}/")):
            ordered.append((rel, found.pop(rel)))
    ordered.extend((rel, found[rel]) for rel in sorted(found))
    return ordered


def render_python() -> list[str]:
    """Render the Python control-plane section: one block per package."""
    out = ["## Python control plane — `python/batcher/`", ""]
    out.append(
        "Every package, its architectural layer, and every module it contains. "
        "The one-liner is the module's own docstring summary."
    )
    out.append("")
    for rel, path in py_packages():
        label = "batcher" if rel == "." else f"batcher/{rel}"
        init = path / "__init__.py"
        mods = sorted(p for p in path.glob("*.py") if p.name != "__init__.py")
        out.append(f"### `{label}/` — {layer_of(rel)}")
        out.append("")
        out.append(f"{py_summary(init)}")
        out.append("")
        if not mods:
            out.append("_(façade only — see the subpackages below)_")
            out.append("")
            continue
        out.append("| module | lines | what it is |")
        out.append("|---|---|---|")
        for m in mods:
            n = len(m.read_text(encoding="utf-8").splitlines())
            out.append(f"| `{m.name}` | {n} | {py_summary(m)} |")
        out.append("")
    return out


def render_rust() -> list[str]:
    """Render the Rust data-plane section: one block per crate, DAG order."""
    out = ["## Rust data plane — `crates/`", ""]
    out.append(
        "Crates in dependency order (dependents first). The `depends on` line is read "
        "from each `Cargo.toml`, so it is the real DAG, not a remembered one. Rust line "
        "counts exclude the trailing `#[cfg(test)]` module, matching `lint-structure`."
    )
    out.append("")
    present = {p.name for p in CRATES.iterdir() if (p / "Cargo.toml").exists()}
    order = [c for c in CRATE_ORDER if c in present] + sorted(present - set(CRATE_ORDER))
    for name in order:
        crate = CRATES / name
        src = crate / "src"
        deps = crate_deps(crate)
        lib = src / "lib.rs"
        out.append(f"### `{name}`")
        out.append("")
        out.append(rust_summary(lib) if lib.exists() else "_(no lib.rs)_")
        out.append("")
        out.append(
            f"**depends on:** {', '.join(f'`{d}`' for d in deps) or '_(nothing — leaf crate)_'}"
        )
        out.append("")
        files = sorted(p for p in src.rglob("*.rs"))
        out.append("| file | lines | what it is |")
        out.append("|---|---|---|")
        for f in files:
            rel = f.relative_to(src).as_posix()
            out.append(f"| `{rel}` | {rust_code_lines(f)} | {rust_summary(f)} |")
        out.append("")
    return out


def render() -> str:
    """Render the whole of ``MAP.md``."""
    head = [
        "<!-- GENERATED by tools/gen_map.py — do not edit by hand.",
        "     Module one-liners are quoted from each module's own docstring; run",
        "     `just map` after adding or re-documenting a module. Curated prose",
        "     (disambiguation, routing) is edited in tools/map_notes.md. -->",
        "",
        "# Batcher — repository map",
        "",
        "**The index of what every file is for.** Grep this file before you search the "
        "tree: it answers *where does X live* and *where does new X go* without opening "
        "690 modules. `CLAUDE.md` holds the invariants (the law); this holds the "
        "territory.",
        "",
        f"Covering {sum(1 for _ in PY_ROOT.rglob('*.py') if '__pycache__' not in _.parts)} "
        f"Python modules across {len(py_packages())} packages and "
        f"{sum(1 for _ in CRATES.rglob('src/**/*.rs'))} Rust files across "
        f"{len([p for p in CRATES.iterdir() if (p / 'Cargo.toml').exists()])} crates.",
        "",
    ]
    notes = NOTES.read_text(encoding="utf-8").strip().splitlines() if NOTES.exists() else []
    return "\n".join([*head, *notes, "", *render_python(), *render_rust()]).rstrip() + "\n"


def main() -> int:
    """Write ``MAP.md``, or verify it is current under ``--check``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if MAP.md is stale instead of rewriting it",
    )
    args = parser.parse_args()
    new = render()
    if args.check:
        old = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if old != new:
            print(
                "MAP.md is stale — a module was added, moved, or re-documented.\n"
                "Run `just map` and commit the result.",
                file=sys.stderr,
            )
            return 1
        print("MAP.md is current.")
        return 0
    OUT.write_text(new, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
