"""Fail the build when the Sphinx doctest builder reported a failure.

`sphinx-build -b doctest` prints its failures and **exits 0 anyway**, so the `just docs`
recipe ran straight past them into the HTML build and reported success. That is not a
theoretical hole: it let two docstring examples that no longer matched the code through two
consecutive `just docs` runs, both of which were read as green.

It matters more here than the two examples do, because `.claude/rules/python-quality.md`
makes the doctests a *contract* -- "the `.. doctest::` examples are then executed for real by
`just docs`, so an example that lies fails the build". They were executed. Nothing failed.

The doctest builder writes a summary to `output.txt` that says what happened, so this reads
that rather than trying to make sphinx exit non-zero. Passing `-W` to that builder is not the
fix: it promotes *warnings*, and a failing example is not one.

Usage:
    python tools/check_doctests.py docs/_build/doctest/output.txt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

#: `NNN failures in tests`, and the same for setup and cleanup code. All three are failures:
#: a broken `testsetup` block silently disables every example that depends on it, which is
#: the quieter of the two ways for this suite to stop testing anything.
_FAILURE_LINE = re.compile(
    r"^\s*(\d+) failures? in (tests|setup code|cleanup code)\s*$", re.MULTILINE
)
_TOTAL_LINE = re.compile(r"^\s*(\d+) tests?\s*$", re.MULTILINE)


def check(path: Path) -> int:
    """Return an exit code: 0 when the summary reports no failures.

    Args:
        path: The `output.txt` the doctest builder wrote.

    Returns:
        0 if every failure count is zero, 1 otherwise.
    """
    if not path.exists():
        print(f"doctest check: {path} does not exist -- did the doctest builder run?")
        return 1
    text = path.read_text(errors="ignore")
    failures = {kind: int(n) for n, kind in _FAILURE_LINE.findall(text)}
    if not failures:
        # No summary at all is a failure, not a pass. A doctest step that produced no
        # summary did not report success -- it reported nothing, which is the state this
        # whole script exists to stop being read as green.
        print(f"doctest check: no doctest summary found in {path}")
        return 1
    totals = _TOTAL_LINE.findall(text)
    total = totals[-1] if totals else "?"
    bad = {k: v for k, v in failures.items() if v}
    if not bad:
        print(f"doctest check: clean ({total} examples)")
        return 0
    print(f"doctest check: FAILED -- {', '.join(f'{v} in {k}' for k, v in bad.items())}")
    print(f"  of {total} examples. The failures are printed above, in the doctest output.")
    print("  A docstring example is a contract: fix the example or fix the code.")
    return 1


if __name__ == "__main__":
    sys.exit(check(Path(sys.argv[1] if len(sys.argv) > 1 else "docs/_build/doctest/output.txt")))
