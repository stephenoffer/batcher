#!/usr/bin/env python3
"""The daily self-improvement loop: review the codebase, improve it, verify, report.

Point a cron job or a human at this. It runs a sequence of agent passes — read-only reviews
that find work, then narrow improvement passes that do it — and writes a dated report.

What it will not do, by construction:

* **Touch your working tree.** Writing passes run in `git worktree`s under
  `/tmp/batcher-agentic`. Reviews are read-only.
* **Merge anything.** A verified pass leaves a branch (`agentic/<pass>`) for review. Nothing
  reaches `main` without a human. The loop's job is to prepare reviewable work, not to
  self-commit — an unattended agent merging to a shared branch is how a silent regression
  ships at 3am.
* **Trust an agent's self-report.** A pass is kept only if its gate commands exit zero here,
  after the agent has finished.

Usage::

    python tools/agentic/daily.py                    # reviews + improvements
    python tools/agentic/daily.py --dry-run          # no agent; proves the gates run
    python tools/agentic/daily.py --only perf        # one pass
    python tools/agentic/daily.py --reviews-only     # find work, change nothing
    python tools/agentic/daily.py --clean            # remove leftover worktrees

Set `BATCHER_AGENT_CMD` to your agent CLI (default: `claude -p`).

The baseline check that runs first is deliberate: if the gate is already red before any
agent starts, improvement passes are unverifiable — a green gate afterward would prove
nothing, since it might have been green-ish all along or red for an unrelated reason. So a
red baseline downgrades the run to `fix-gate` only.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.agentic import passes as catalog
from tools.agentic.report import write_report
from tools.agentic.runner import (
    REPO,
    WORK_ROOT,
    PassResult,
    agent_available,
    run_pass,
)

#: The cheap subset used to decide whether the tree is healthy enough to improve. Not the
#: full gate — this must stay fast, since it gates the run rather than a change.
BASELINE = (
    "ruff check python tests benchmarks examples",
    "python tools/lint_structure.py",
    "python tools/lint_guardrails.py",
    "lint-imports --config pyproject.toml",
)


def _baseline() -> list[str]:
    """Return the baseline commands that are currently failing."""
    failing = []
    for cmd in BASELINE:
        if subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True).returncode != 0:
            failing.append(cmd)
    return failing


def _tree_is_dirty() -> bool:
    """Whether the working tree has uncommitted changes."""
    out = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True)
    return bool(out.stdout.strip())


def main() -> int:
    """Run the daily loop and write a report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", metavar="PASS", help="run just this pass (see --list)")
    parser.add_argument("--reviews-only", action="store_true", help="find work, change nothing")
    parser.add_argument(
        "--improvements-only", action="store_true", help="skip the read-only reviews"
    )
    parser.add_argument("--base", default="HEAD", help="git ref to branch worktrees from")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="do not invoke an agent; still creates worktrees and runs verification",
    )
    parser.add_argument("--list", action="store_true", help="list passes and exit")
    parser.add_argument("--clean", action="store_true", help="remove leftover worktrees and exit")
    parser.add_argument(
        "--report-dir",
        default=str(WORK_ROOT / "reports"),
        help="where to write the dated report",
    )
    args = parser.parse_args()

    if args.list:
        for name, spec in catalog.ALL.items():
            kind = "write " if spec.writes else "review"
            print(f"  {kind}  {name:22s} {spec.goal}")
        return 0

    if args.clean:
        for name in catalog.ALL:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(WORK_ROOT / name)],
                cwd=REPO,
                capture_output=True,
            )
        subprocess.run(["git", "worktree", "prune"], cwd=REPO, capture_output=True)
        print("removed agentic worktrees (branches kept)")
        return 0

    if not args.dry_run and not agent_available():
        print(
            "No agent CLI found. Set BATCHER_AGENT_CMD to your agent command, "
            "or pass --dry-run to exercise the loop without one.",
            file=sys.stderr,
        )
        return 2

    started = time.time()
    print(f"Batcher daily loop — {datetime.now(UTC).isoformat(timespec='seconds')}")

    if _tree_is_dirty():
        # Worktrees branch from a committed ref, so uncommitted work is simply not included.
        # Say so rather than letting someone assume their in-progress change was reviewed.
        print(
            "note: working tree is dirty. Passes branch from "
            f"{args.base}, so uncommitted changes are NOT included in this run."
        )

    # Choose the pass list.
    if args.only:
        if args.only not in catalog.ALL:
            print(f"unknown pass {args.only!r}; see --list", file=sys.stderr)
            return 2
        selected = [catalog.ALL[args.only]]
    elif args.reviews_only:
        selected = list(catalog.REVIEWS)
    elif args.improvements_only:
        selected = list(catalog.IMPROVEMENTS)
    else:
        selected = [*catalog.REVIEWS, *catalog.IMPROVEMENTS]

    baseline_failures = _baseline()
    if baseline_failures and not args.only:
        print(
            f"baseline gate is RED ({len(baseline_failures)} failing) — restricting this run to "
            "fix-gate, since improvements cannot be verified against a broken baseline:"
        )
        for cmd in baseline_failures:
            print(f"    FAIL  {cmd}")
        selected = [p for p in selected if not p.writes] + [catalog.FIX_GATE]
    else:
        print("baseline gate: green" if not baseline_failures else "baseline: red (forced run)")

    results: list[PassResult] = []
    for spec in selected:
        print(f"\n=== {spec.name} — {spec.goal}")
        result = run_pass(spec, base=args.base, dry_run=args.dry_run)
        results.append(result)
        print(f"    {result.status}: {result.reason} ({result.seconds}s)")
        if result.diffstat:
            print(f"    {result.diffstat}")
        for cmd in result.failures:
            print(f"    failed: {cmd}")

    report = write_report(results, Path(args.report_dir), baseline_failures, args.dry_run)
    kept = [r for r in results if r.kept and r.diffstat]
    print(f"\nreport: {report}")
    if kept:
        print("branches ready for review:")
        for r in kept:
            print(f"    {r.branch}    ({r.diffstat})")
    else:
        print("no branches produced.")
    print(f"total {round(time.time() - started, 1)}s")

    # Exit non-zero only if a pass was rejected *after* changing something — that is the
    # signal worth waking someone for. A clean no-op run is a success.
    rejected = [r for r in results if not r.kept and r.diffstat]
    return 1 if rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
