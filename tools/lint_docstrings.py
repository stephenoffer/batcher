"""Docstring style gate for the public API (`just lint-docstrings`).

`.claude/rules/python-quality.md` fixes one docstring style for everything a user can
import: a one-line summary inline with the opening quotes, a runnable ``Examples:``
block under a ``.. doctest::`` directive, and ``Args:``/``Returns:`` sections that
carry no types (types live in the signature). This script checks that mechanically,
against the live objects, so the style is a gate rather than a review preference.

It is deliberately structural, not semantic: it cannot tell whether a summary is
*good*, only that one exists and fits on one line. The `.. doctest::` blocks it
insists on are executed for real by ``just docs`` (the Sphinx doctest builder runs
over the autodoc'd docstrings), so an example that lies fails a different gate.

Run standalone (``python tools/lint_docstrings.py``) for the full report, or let
``tests/docs/test_api_coverage.py`` fail the suite on a regression.
"""

from __future__ import annotations

import collections
import inspect
import re
import sys
from pathlib import Path
from typing import Any, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from public_surface import public_callables

# ``Args:``-entry that smuggles a type in parentheses — ``value (int): the value``.
# Types belong in the signature; napoleon renders them from the annotations.
_TYPED_ARG = re.compile(r"^\s{4,}\*{0,2}\w+ \([^)]*\)\s*:", re.M)
_SECTION = re.compile(r"^(?P<indent>[ \t]*)(?P<name>[A-Z][A-Za-z ]*):[ \t]*$", re.M)
_SUMMARY_END = (".", "!", "?", ":")

# Objects whose docstring the style gate cannot sensibly apply to. Keep each with a
# reason; this list is printed on every run so it stays visible and shrinks.
ALLOW: dict[str, str] = {}


class Violation(NamedTuple):
    """A single rule broken by a single public callable."""

    name: str
    rule: str
    file: str
    line: int


def _sections(doc: str) -> set[str]:
    return {m.group("name") for m in _SECTION.finditer(doc)}


def _documented_params(doc: str) -> set[str]:
    """Parameter names listed under ``Args:`` (one entry per ``name:`` line)."""
    match = re.search(r"^([ \t]*)Args:[ \t]*$", doc, re.M)
    if not match:
        return set()
    body = doc[match.end() :]
    base = len(match.group(1))
    names: set[str] = set()
    for line in body.split("\n"):
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= base:  # dedented out of the Args block
            break
        entry = re.match(r"^\s*\*{0,2}(\w+)\s*(\([^)]*\))?\s*:", line)
        if entry and indent == base + 4:
            names.add(entry.group(1))
    return names


def _expected_params(obj: Any) -> list[str]:
    """The parameters a caller can pass, so the ones ``Args:`` must cover.

    Skips the bound receiver and the ``_``-prefixed defaults that the accessor
    factories inject to close over loop variables (``_tag``, ``_build``): those are
    implementation, not surface, and documenting them would mislead.
    """
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return []
    return [n for n in sig.parameters if n not in {"self", "cls"} and not n.startswith("_")]


def _returns_a_value(obj: Any) -> bool:
    if inspect.isclass(obj):
        return False
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return False
    ret = sig.return_annotation
    if ret is inspect.Signature.empty:
        return False
    return ret not in (None, "None", type(None))


def _location(obj: Any) -> tuple[str, int]:
    try:
        file = inspect.getsourcefile(obj) or "?"
        _, line = inspect.getsourcelines(obj)
    except (TypeError, OSError):
        return "?", 0
    return file, line


def check(obj: Any) -> list[str]:
    """Return the style rules `obj` breaks, as a list of short rule ids."""
    doc = getattr(obj, "__doc__", None)
    # A method inheriting a base docstring is documented by that base; only flag the
    # object that actually owns the text.
    if not doc or not doc.strip():
        return ["missing-docstring"]

    broken: list[str] = []
    lines = doc.split("\n")

    if not lines[0].strip():
        broken.append("summary-not-inline")
    else:
        summary = lines[0].strip()
        if len(lines) > 1 and lines[1].strip():
            broken.append("summary-multiline")
        if not summary.endswith(_SUMMARY_END):
            broken.append("summary-unterminated")

    sections = _sections(doc)
    if "Example" in sections and "Examples" not in sections:
        broken.append("examples-heading-singular")
    if "Examples" not in sections:
        broken.append("no-examples")
    elif ".. doctest::" not in doc:
        broken.append("examples-not-doctest")

    params = _expected_params(obj)
    if inspect.isclass(obj):
        params = []  # a class documents its constructor args only if it chooses to
    if params and "Args" not in sections:
        broken.append("no-args-section")
    elif params:
        missing = [p for p in params if p not in _documented_params(doc)]
        if missing:
            broken.append(f"args-undocumented:{','.join(missing)}")
    if _TYPED_ARG.search(doc):
        broken.append("args-have-types")

    if _returns_a_value(obj) and not ({"Returns", "Yields"} & sections):
        broken.append("no-returns-section")
    return broken


def collect() -> list[Violation]:
    """Every style violation across the public surface."""
    out: list[Violation] = []
    for name, obj in public_callables():
        if name in ALLOW:
            continue
        for rule in check(obj):
            file, line = _location(obj)
            out.append(Violation(name, rule, file, line))
    return out


def main() -> int:
    violations = collect()
    total = len(public_callables())

    if ALLOW:
        print(f"docstring-style allowlist ({len(ALLOW)}):")
        for name, reason in sorted(ALLOW.items()):
            print(f"  {name}: {reason}")
        print()

    if not violations:
        print(f"docstring style: OK ({total} public callables)")
        return 0

    by_file: dict[str, list[Violation]] = collections.defaultdict(list)
    for v in violations:
        by_file[v.file].append(v)

    for file in sorted(by_file):
        print(f"\n{file}")
        for v in sorted(by_file[file], key=lambda v: v.line):
            print(f"  {v.line:5d}  {v.name.split('.')[-1]:<28} {v.rule}")

    counts = collections.Counter(v.rule.split(":")[0] for v in violations)
    print(f"\n{len(violations)} violation(s) over {total} public callables:")
    for rule, n in counts.most_common():
        print(f"  {n:5d}  {rule}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
