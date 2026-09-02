"""The doctest step must fail the build when a doctest fails.

`sphinx-build -b doctest` prints its failures and **exits 0**, so `just docs` walked past
them into the HTML build and reported success. Two docstring examples that no longer matched
the code went through two consecutive `just docs` runs that way, both read as green.

That is the failure `CLAUDE.md` calls a green gate that is not a green light, and it is worse
than an ordinary missing check because `.claude/rules/python-quality.md` sells these examples
as a contract -- "executed for real by `just docs`, so an example that lies fails the build".
The examples were executed. The failures were printed. Nothing failed.

`tools/check_doctests.py` reads the summary the builder writes and turns it into an exit
code. This pins that it discriminates, which is the only property that matters: a checker
that returns 0 whatever it reads reproduces the bug it was written to fix, and would look
exactly as green.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_CHECKER = Path(__file__).resolve().parents[2] / "tools" / "check_doctests.py"

_SUMMARY = """Doctest summary
===============
 9598 tests
    {tests} failures in tests
    {setup} failures in setup code
    {cleanup} failures in cleanup code
"""


def _run(text: str | None, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    target = tmp_path / "output.txt"
    if text is not None:
        target.write_text(text)
    return subprocess.run(
        [sys.executable, str(_CHECKER), str(target)], capture_output=True, text=True
    )


def test_the_checker_exists_where_the_justfile_calls_it():
    """`just docs` names this path; a rename that misses the recipe disables the gate."""
    assert _CHECKER.is_file()
    justfile = (_CHECKER.parents[1] / "justfile").read_text()
    assert "tools/check_doctests.py" in justfile, (
        "the docs recipe no longer runs the doctest check, so a failing example passes again"
    )


def test_a_clean_summary_passes(tmp_path):
    result = _run(_SUMMARY.format(tests=0, setup=0, cleanup=0), tmp_path)
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize("kind", ["tests", "setup", "cleanup"])
def test_a_failure_in_any_phase_fails(kind, tmp_path):
    """All three counts, not just `failures in tests`.

    A broken `testsetup` block disables every example depending on it while reporting zero
    test failures, which is the quieter way for this suite to stop testing anything."""
    counts = {"tests": 0, "setup": 0, "cleanup": 0} | {kind: 3}
    result = _run(_SUMMARY.format(**counts), tmp_path)
    assert result.returncode == 1, f"a failure in {kind} did not fail the gate: {result.stdout}"
    assert "FAILED" in result.stdout


def test_a_missing_output_file_fails(tmp_path):
    """Absence is not success. If the doctest builder did not run, the gate has no evidence."""
    assert _run(None, tmp_path).returncode == 1


def test_output_with_no_summary_fails(tmp_path):
    """The same rule one step in: a truncated or empty report says nothing, and saying
    nothing must not read as saying everything passed."""
    result = _run("Document: api/whatever\n1 items passed all tests:\n", tmp_path)
    assert result.returncode == 1, result.stdout


def test_the_real_failing_report_shape_is_rejected(tmp_path):
    """The exact bytes the builder wrote when it let the two stale examples through. A
    synthetic summary proves the parser works on what this test imagines the format to be;
    this proves it works on what sphinx actually emitted."""
    observed = (
        "Document: api/operations/governance\n"
        "-----------------------------------\n"
        "**********************************************************************\n"
        'File "../python/batcher/governance/policy.py", line ?, in default\n'
        "Failed example:\n"
        '    policy.mask(col("ssn"))\n'
        "\n"
        "Doctest summary\n"
        "===============\n"
        " 9598 tests\n"
        "    2 failures in tests\n"
        "    0 failures in setup code\n"
        "    0 failures in cleanup code\n"
    )
    result = _run(observed, tmp_path)
    assert result.returncode == 1
    assert "2 in tests" in result.stdout
