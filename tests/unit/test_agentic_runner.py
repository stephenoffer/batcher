"""The daily loop's keep/reject machinery must be trustworthy before it is trusted.

`tools/agentic/` decides, unattended, whether an agent's change is kept. Two of its
behaviours are load-bearing and neither is obvious from reading it: work is **rejected**
when verification fails, and a **missing baseline aborts the pass** instead of letting a
comparison pass vacuously. If either inverted, the loop would quietly keep bad changes —
and it would look exactly the same from the outside, because a loop that accepts everything
reports the same cheerful summary as one that is working.

These use throwaway `Pass` objects with shell commands (`true`, `false`, `echo`) rather
than real gates, so they test the decision logic rather than any particular linter, and run
in a second. Each uses a uniquely-named worktree and removes it afterwards, so the tests are
safe to run while other agents are working in the repo.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from tools.agentic.bench import improvements, regressions  # noqa: E402
from tools.agentic.runner import Pass, cleanup, run_pass  # noqa: E402


def _spec(name: str, **kw: object) -> Pass:
    """A throwaway pass whose commands are shell built-ins."""
    return Pass(name=f"pytest-{name}", goal="unit test", prompt="unused", **kw)  # type: ignore[arg-type]


@pytest.fixture
def worktree_cleanup() -> object:
    """Remove any worktree a test created, even if it fails."""
    created: list[str] = []
    yield created.append
    for name in created:
        cleanup(name)
        subprocess.run(["git", "branch", "-D", f"agentic/{name}"], cwd=_REPO, capture_output=True)


def test_failing_verification_rejects_the_work(worktree_cleanup) -> None:
    """A pass whose gate fails is rejected, and names the command that failed."""
    spec = _spec("reject", verify=("true", "false"))
    worktree_cleanup(spec.name)

    result = run_pass(spec, dry_run=True)

    assert not result.kept
    assert "false" in result.failures
    assert "true" not in result.failures
    assert result.status == "REJECTED"


def test_passing_verification_keeps_the_work(worktree_cleanup) -> None:
    """A pass whose gate passes is kept."""
    spec = _spec("keep", verify=("true", "true"))
    worktree_cleanup(spec.name)

    result = run_pass(spec, dry_run=True)

    assert result.kept, result.reason
    assert not result.failures


def test_failed_setup_aborts_instead_of_verifying_against_nothing(worktree_cleanup) -> None:
    """A pass whose baseline capture fails is rejected, even though its gate would pass.

    This is the subtle one. If setup failure were ignored, the pass would run its gate
    against a baseline that was never written — and a comparison against a missing
    baseline is the kind of check that passes for the wrong reason.
    """
    spec = _spec("setup-fail", setup=("false",), verify=("true",))
    worktree_cleanup(spec.name)

    result = run_pass(spec, dry_run=True)

    assert not result.kept
    assert "no baseline" in result.reason


def test_setup_and_verify_share_a_baseline_directory(worktree_cleanup) -> None:
    """`$AGENTIC_BASELINE` is set for both, so verify can read what setup wrote."""
    spec = _spec(
        "baseline-env",
        setup=('echo captured > "$AGENTIC_BASELINE/probe.txt"',),
        verify=('test -f "$AGENTIC_BASELINE/probe.txt"',),
    )
    worktree_cleanup(spec.name)

    result = run_pass(spec, dry_run=True)

    assert result.kept, f"verify could not read what setup wrote: {result.reason}"


def test_worktree_isolates_changes_from_the_repo(worktree_cleanup) -> None:
    """A pass writing a file leaves the real working tree untouched."""
    spec = _spec(
        "isolation", setup=("echo scratch > agentic-isolation-probe.txt",), verify=("true",)
    )
    worktree_cleanup(spec.name)

    run_pass(spec, dry_run=True)

    assert not (_REPO / "agentic-isolation-probe.txt").exists(), (
        "a pass wrote into the real repository — worktree isolation is broken, which is the "
        "property that keeps concurrent agents from clobbering each other."
    )


# --- The benchmark regression gate --------------------------------------------------


def _snap(cases: dict[str, tuple[str, float]]) -> dict:
    """Build a benchmark snapshot from {case: (status, batcher_ms)}."""
    return {"cases": {n: {"status": s, "ms": {"batcher": ms}} for n, (s, ms) in cases.items()}}


@pytest.mark.parametrize(
    ("before", "after", "expect_regression"),
    [
        ({"q": ("OK", 100.0)}, {"q": ("OK", 140.0)}, True),  # 40% slower
        ({"q": ("OK", 100.0)}, {"q": ("OK", 105.0)}, False),  # within tolerance
        ({"q": ("OK", 100.0)}, {"q": ("OK", 60.0)}, False),  # faster
        ({"q": ("OK", 100.0)}, {"q": ("FAILED", 60.0)}, True),  # fast but wrong
    ],
)
def test_regression_detection(before: dict, after: dict, expect_regression: bool) -> None:
    """A slowdown beyond tolerance, or a correctness failure, counts as a regression."""
    found = regressions(_snap(before), _snap(after))
    assert bool(found) is expect_regression, found


def test_a_dropped_case_counts_as_a_regression() -> None:
    """Silently not measuring a case is the easiest way to hide a regression."""
    found = regressions(_snap({"q1": ("OK", 100.0)}), _snap({}))
    assert found and "q1" in found[0]


def test_a_new_case_is_not_a_regression() -> None:
    """A case with no baseline has nothing to regress against."""
    assert not regressions(_snap({}), _snap({"new": ("OK", 10.0)}))


def test_improvements_are_reported() -> None:
    """A real speedup is surfaced for the report."""
    found = improvements(_snap({"q": ("OK", 200.0)}), _snap({"q": ("OK", 100.0)}))
    assert found and "-50.0%" in found[0]


def test_bench_snapshot_round_trips_as_json() -> None:
    """Snapshots are compared across processes, so they must survive JSON."""
    snap = _snap({"q1": ("OK", 12.5)})
    assert regressions(snap, json.loads(json.dumps(snap))) == []
