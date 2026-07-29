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
path (they contain a `/` and start with a known top-level directory), so prose stays free. It
also ignores paths git ignores: a build artifact like `python/batcher/_native.abi3.so` exists
only after `just build`, so its absence says nothing about whether the guidance is true. Without
that carve-out the same unchanged rule file passed or failed depending on whether anyone had
built the extension yet.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

GUARDRAILS = [
    ROOT / "CLAUDE.md",
    *sorted((ROOT / ".claude" / "rules").glob("*.md")),
    *sorted((ROOT / ".claude" / "skills").rglob("SKILL.md")),
    *sorted(ROOT.glob("*/CLAUDE.md")),
    # The contributor cookbook routes a change to the file that should hold it — the same
    # job as a skill, for humans. A stale path here misroutes exactly as badly.
    ROOT / "docs" / "internals" / "extending.md",
]

#: Only these roots are treated as repo paths; everything else in backticks is prose.
PATH_ROOTS = (
    "python/",
    "crates/",
    "tests/",
    "tools/",
    "docs/",
    "benchmarks/",
    "examples/",
    ".claude/",
)

#: The control-plane packages, so a path written the way contributors actually say it
#: ("a rule goes in `kyber/rules/<family>.py`") is checked rather than skipped as prose.
#: Guidance names these relative to `python/batcher/`, which is the natural spelling — but
#: it left them unverified, and that is precisely how `plan/nodes/` survived in the cookbook
#: after the module became `plan/logical/`. Paths under these roots resolve against
#: `python/batcher/` as well as the repo root; see `_resolves`.
PACKAGE_ROOTS = (
    "api/",
    "carbonite/",
    "config/",
    "core/",
    "dist/",
    "governance/",
    "io/",
    "kyber/",
    "metadata/",
    "ml/",
    "plan/",
    "_internal/",
    "_sql/",
)

#: `path/to/thing` inside backticks, optionally with a `::symbol` suffix or a trailing slash.
PATH_RE = re.compile(r"`([A-Za-z0-9_./\-]+/[A-Za-z0-9_./\-]*)`")
RECIPE_RE = re.compile(r"`just ([a-z][a-z0-9-]*)")

#: Paths that are patterns/placeholders, not literal files.
PLACEHOLDER = re.compile(r"[<>*{}]|\.\.\.|\bfmt\b|\bfamily\b|\bname\b")


def _resolves(path: str) -> bool:
    """Return whether a documented path names something that actually exists.

    A path may be written relative to the repo root (`python/batcher/kyber/rules/`) or,
    for the control-plane packages, relative to `python/batcher/` (`kyber/rules/`) — both
    spellings appear in the guidance and both are legitimate. A module that has since become
    a package is still a hit: the import path the prose is teaching survives the split, so
    `plan/logical.py` resolving to `plan/logical/` is correct, not stale.
    """
    for base in (ROOT, ROOT / "python" / "batcher"):
        candidate = base / path
        if candidate.exists():
            return True
        if candidate.suffix == ".py" and candidate.with_suffix("").is_dir():
            return True
    return False


def _git_ignored(paths: list[str]) -> set[str]:
    """Return the subset of `paths` that git ignores, i.e. generated build output.

    A generated path is unverifiable by existence: `python/batcher/_native.abi3.so` is there
    after `just build` and gone after `just clean`, and the guidance that names it is equally
    true either way. Both spellings `_resolves` accepts are tested, so a package-relative path
    is classified the same way as a repo-relative one. If git is unavailable, nothing is
    treated as ignored and the caller reports the path as it would have before.
    """
    if not paths:
        return set()
    candidates = {p for path in paths for p in (path, f"python/batcher/{path}")}
    try:
        proc = subprocess.run(
            ["git", "check-ignore", "--stdin"],
            cwd=ROOT,
            input="\n".join(sorted(candidates)),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return set()
    ignored = set(proc.stdout.split())
    return {p for p in paths if p in ignored or f"python/batcher/{p}" in ignored}


def _justfile_recipes() -> set[str]:
    """Recipe names from the justfile, including parameterized ones (`bench args="":`)."""
    text = (ROOT / "justfile").read_text()
    return set(re.findall(r"^([a-z][a-z0-9-]*)(?:\s+[^:\n]*)?:", text, flags=re.M))


def main() -> int:
    recipes = _justfile_recipes()
    #: (path, message) in document order. `path` is the repo path a message is about, or ""
    #: for a recipe failure. Reporting waits until the whole tree is scanned so the
    #: git-ignore classification is a single batched call.
    pending: list[tuple[str, str]] = []

    for doc in GUARDRAILS:
        if not doc.exists():
            continue
        rel_doc = doc.relative_to(ROOT)
        text = doc.read_text()

        for lineno, line in enumerate(text.splitlines(), 1):
            for raw in PATH_RE.findall(line):
                path = raw.split("::", 1)[0].rstrip("/")
                if not path.startswith(PATH_ROOTS + PACKAGE_ROOTS) or PLACEHOLDER.search(path):
                    continue
                if not _resolves(path):
                    pending.append((path, f"{rel_doc}:{lineno}: path does not exist: {path}"))

            for recipe in RECIPE_RE.findall(line):
                if recipe not in recipes:
                    pending.append(
                        ("", f"{rel_doc}:{lineno}: `just {recipe}` is not a recipe in the justfile")
                    )

    generated = _git_ignored([path for path, _ in pending if path])
    failures = [message for path, message in pending if path not in generated]

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
