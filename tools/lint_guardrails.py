#!/usr/bin/env python3
"""Fail when the agent-facing guidance points at something that does not exist.

`CLAUDE.md`, `.claude/rules/*.md`, and `.claude/skills/**/SKILL.md` are the instructions an AI
agent reads before touching this repo. They are only useful while they are *true*. A rule that
cites `plan/logical.py` after that module became a package sends the agent to a file that is not
there — and an agent that cannot find the file it was told to edit does not stop; it invents a
new place to put the code. That is how a codebase grows a second home for everything.

This has already happened here: the flagship `add-relational-operator` skill named three paths
that no longer existed (`ops.rs`, `plan/logical.py`, `api/dataset.py` — all now packages), and
`just bench` was documented as the operator-mix benchmark when it actually runs TPC-H.

So the guidance is linted like code:

* every repo-relative path mentioned in a guardrail file must exist; and
* every `just <recipe>` named must be a real recipe in the justfile.

The check is deliberately conservative — it only flags strings that clearly *look* like a repo
path (they contain a `/` and start with a known top-level directory), so prose stays free.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

GUARDRAILS = [
    ROOT / "CLAUDE.md",
    *sorted((ROOT / ".claude" / "rules").glob("*.md")),
    *sorted((ROOT / ".claude" / "skills").rglob("SKILL.md")),
    *sorted(ROOT.glob("*/CLAUDE.md")),
]

#: Only these roots are treated as repo paths; everything else in backticks is prose.
PATH_ROOTS = ("python/", "crates/", "tests/", "tools/", "docs/", "benchmarks/", "examples/", ".claude/")

#: `path/to/thing` inside backticks, optionally with a `::symbol` suffix or a trailing slash.
PATH_RE = re.compile(r"`([A-Za-z0-9_./\-]+/[A-Za-z0-9_./\-]*)`")
RECIPE_RE = re.compile(r"`just ([a-z][a-z0-9-]*)")

#: Paths that are patterns/placeholders, not literal files.
PLACEHOLDER = re.compile(r"[<>*{}]|\.\.\.|\bfmt\b|\bfamily\b|\bname\b")


def _justfile_recipes() -> set[str]:
    """Recipe names from the justfile, including parameterized ones (`bench args="":`)."""
    text = (ROOT / "justfile").read_text()
    return set(re.findall(r"^([a-z][a-z0-9-]*)(?:\s+[^:\n]*)?:", text, flags=re.M))


def main() -> int:
    recipes = _justfile_recipes()
    failures: list[str] = []

    for doc in GUARDRAILS:
        if not doc.exists():
            continue
        rel_doc = doc.relative_to(ROOT)
        text = doc.read_text()

        for lineno, line in enumerate(text.splitlines(), 1):
            for raw in PATH_RE.findall(line):
                path = raw.split("::", 1)[0].rstrip("/")
                if not path.startswith(PATH_ROOTS) or PLACEHOLDER.search(path):
                    continue
                if not (ROOT / path).exists():
                    failures.append(f"{rel_doc}:{lineno}: path does not exist: {path}")

            for recipe in RECIPE_RE.findall(line):
                if recipe not in recipes:
                    failures.append(
                        f"{rel_doc}:{lineno}: `just {recipe}` is not a recipe in the justfile"
                    )

    for failure in failures:
        print(f"FAIL: {failure}")

    if failures:
        print(f"\nlint-guardrails: {len(failures)} stale reference(s) in the agent-facing docs.")
        print("  Guidance that points at a file which is not there is worse than no guidance —")
        print("  an agent that cannot find the file invents a new place to put the code.")
        return 1

    print(f"lint-guardrails: clean ({len(GUARDRAILS)} guardrail files checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
