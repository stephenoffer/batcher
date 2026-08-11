"""Fill the example-library tables in `docs/examples/` from the scripts themselves.

The prose on those pages is written by hand. The tables are not: they list every script
under `examples/`, and a list of 500 entries maintained by hand would be wrong within a
week. This script regenerates each table in place, between the HTML markers that delimit
it, so the library cannot drift from the tree.

    python tools/example_library.py            # rewrite the tables
    python tools/example_library.py --check    # fail if they are out of date

A marker names the example directories the table covers:

    <!-- library-table: relational,joins -->
    ...generated...
    <!-- /library-table -->

`tests/docs/test_example_library.py` runs the check, so a new script that no page claims
fails the suite rather than going undocumented.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"
LIBRARY = REPO / "docs" / "examples"

_MARKER = re.compile(
    r"(<!-- library-table: (?P<dirs>[^>]*?) -->\n)(?P<body>.*?)(<!-- /library-table -->)",
    re.DOTALL,
)

#: The docs forbid Unicode punctuation in source, and the scripts' own docstrings use it
#: freely. Fold it down on the way into a table cell rather than editing 500 docstrings.
#: Written as escapes so the characters themselves never appear in this file either.
_PUNCTUATION = {
    "\N{EM DASH}": " -",
    "\N{EN DASH}": "-",
    "\N{RIGHT SINGLE QUOTATION MARK}": "'",
    "\N{LEFT DOUBLE QUOTATION MARK}": '"',
    "\N{RIGHT DOUBLE QUOTATION MARK}": '"',
}


def _summary(path: Path) -> str:
    """The first line of a script's docstring, as a table cell."""
    doc = ast.get_docstring(ast.parse(path.read_text())) or ""
    line = doc.splitlines()[0] if doc else path.stem.replace("_", " ")
    for bad, good in _PUNCTUATION.items():
        line = line.replace(bad, good)
    line = re.sub(r"\s{2,}", " ", line)
    return line.rstrip(".").replace("|", "\\|")


def _needs_setup(path: Path) -> bool:
    head = path.read_text().splitlines()[:6]
    return any(line.strip().startswith("# examples: skip") for line in head)


def _scripts(directory: str) -> list[Path]:
    root = EXAMPLES if directory == "." else EXAMPLES / directory
    if not root.is_dir():
        raise SystemExit(f"example directory does not exist: {directory}")
    found = sorted(root.glob("*.py"))
    return [p for p in found if not p.name.startswith("_")]


def _table(directories: list[str]) -> str:
    rows = ["| Script | Shows |", "| --- | --- |"]
    for directory in directories:
        for script in _scripts(directory):
            relative = script.relative_to(REPO).as_posix()
            note = " (needs external setup)" if _needs_setup(script) else ""
            rows.append(f"| `{relative}` | {_summary(script)}{note} |")
    return "\n".join(rows) + "\n"


def _rewrite(page: Path) -> tuple[str, int]:
    text = page.read_text()
    tables = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal tables
        tables += 1
        directories = [part.strip() for part in match.group("dirs").split(",") if part.strip()]
        return match.group(1) + _table(directories) + match.group(4)

    return _MARKER.sub(replace, text), tables


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail instead of rewriting")
    arguments = parser.parse_args()

    if not LIBRARY.is_dir():
        raise SystemExit(f"no example library at {LIBRARY}")

    claimed: set[Path] = set()
    stale: list[str] = []
    total_tables = 0

    for page in sorted(LIBRARY.glob("*.md")):
        for match in _MARKER.finditer(page.read_text()):
            for directory in match.group("dirs").split(","):
                if directory.strip():
                    claimed.update(_scripts(directory.strip()))
        updated, tables = _rewrite(page)
        total_tables += tables
        if updated != page.read_text():
            if arguments.check:
                stale.append(page.relative_to(REPO).as_posix())
            else:
                page.write_text(updated)

    everything = {
        p
        for p in EXAMPLES.rglob("*.py")
        if not any(part.startswith("_") for part in p.relative_to(EXAMPLES).parts)
    }
    missing = sorted(p.relative_to(REPO).as_posix() for p in everything - claimed)

    if missing:
        print(f"example-library: {len(missing)} script(s) are in no page table:")
        for name in missing[:20]:
            print(f"  {name}")
        return 1
    if stale:
        print("example-library: out of date, run `python tools/example_library.py`:")
        for name in stale:
            print(f"  {name}")
        return 1

    verb = "checked" if arguments.check else "wrote"
    print(f"example-library: {verb} {total_tables} tables covering {len(everything)} scripts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
