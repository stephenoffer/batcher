#!/usr/bin/env python3
"""Run one agent pass in an isolated worktree and decide whether to keep its work.

This is the safety machinery the daily loop is built on. Two rules shape it, both learned
the expensive way in this repo:

**Isolation.** Every writing agent gets its own `git worktree`. When several agents shared
one tree, one agent's `git stash` swept up another's finished work, a repo-wide `ruff --fix`
rewrote files in other agents' half-finished changes, and a test run mid-refactor reported
eight failures that belonged to somebody else. A worktree costs a few hundred milliseconds
and removes that entire class of failure — see `.claude/rules/concurrent-agents.md`.

**Verification, not self-report.** An agent's own claim that its change works is not
evidence. A pass is kept only if the gate commands actually exit zero here, in the
worktree, after the agent has finished. Agents in this repo have reported "all checks pass"
from a tree that was mid-refactor and genuinely broken — not dishonestly, just from a
snapshot that had already gone stale.

Nothing here writes to your working tree or to `main`. A kept pass leaves a branch and a
patch for a human to review; a rejected pass leaves its worktree only if you ask, so you can
look at what went wrong.
"""

from __future__ import annotations

import dataclasses
import os
import shlex
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: The command used to run a coding agent headlessly. It receives the prompt on stdin.
#: Overridable so this works with whatever agent CLI is installed — the default is the
#: Claude Code CLI in print mode.
AGENT_CMD = os.environ.get("BATCHER_AGENT_CMD", "claude -p")

#: Where worktrees and reports go. Outside the repo so a stray `git add -A` in one worktree
#: can never pick up another's tree.
WORK_ROOT = Path(os.environ.get("BATCHER_AGENTIC_ROOT", "/tmp/batcher-agentic"))


@dataclasses.dataclass(frozen=True)
class Pass:
    """One unit of agent work: what to ask for, and what must hold afterward.

    Attributes:
        name: Short slug; names the branch and the report section.
        goal: One line describing the intent, shown in the report.
        prompt: The instruction handed to the agent.
        verify: Shell commands that must all exit 0 for the work to be kept. Empty for a
            read-only pass, which is judged on its report rather than on the tree.
        writes: Whether the agent may modify files. Read-only passes run in the repo
            itself (cheaper, and they cannot damage anything); writing passes get a
            worktree.
        setup: Commands run in the worktree **before** the agent, to capture the "before"
            state a `verify` command later compares against — a benchmark snapshot, a
            surface snapshot. Both get `$AGENTIC_BASELINE` (a per-pass directory) in their
            environment. A setup failure aborts the pass rather than letting it run
            against a baseline that was never captured, which would make the comparison
            silently vacuous.
    """

    name: str
    goal: str
    prompt: str
    verify: tuple[str, ...] = ()
    writes: bool = True
    setup: tuple[str, ...] = ()


@dataclasses.dataclass
class PassResult:
    """What happened when a pass ran."""

    name: str
    goal: str
    kept: bool
    reason: str
    output: str = ""
    failures: tuple[str, ...] = ()
    diffstat: str = ""
    branch: str = ""
    seconds: float = 0.0

    @property
    def status(self) -> str:
        """A one-word verdict for the report table."""
        if not self.kept:
            return "REJECTED"
        return "KEPT" if self.diffstat else "NO-OP"


def _run(
    cmd: str, cwd: Path, timeout: int = 1800, env: dict[str, str] | None = None
) -> tuple[int, str]:
    """Run a shell command, returning its exit code and combined output."""
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, **(env or {})},
        )
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT after {timeout}s: {cmd}"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def agent_available() -> bool:
    """Whether the configured agent command exists on PATH."""
    binary = shlex.split(AGENT_CMD)[0]
    return subprocess.run(["which", binary], capture_output=True).returncode == 0


def _create_worktree(name: str, base: str) -> tuple[Path, str]:
    """Create a fresh worktree and branch for a pass, replacing any stale one."""
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    path = WORK_ROOT / name
    branch = f"agentic/{name}"
    # Remove a previous run's leftovers so a pass always starts from `base`, never from
    # yesterday's half-finished attempt.
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(path)],
        cwd=REPO,
        capture_output=True,
    )
    subprocess.run(["git", "branch", "-D", branch], cwd=REPO, capture_output=True)
    code, out = _run(
        f"git worktree add -b {shlex.quote(branch)} {shlex.quote(str(path))} {shlex.quote(base)}",
        REPO,
    )
    if code != 0:
        raise RuntimeError(f"could not create worktree for {name}: {out}")
    return path, branch


def _invoke_agent(prompt: str, cwd: Path, timeout: int) -> tuple[int, str]:
    """Hand the prompt to the agent, running it in `cwd`."""
    try:
        proc = subprocess.run(
            shlex.split(AGENT_CMD),
            cwd=cwd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 124, f"agent TIMEOUT after {timeout}s"
    except FileNotFoundError:
        return 127, (
            f"agent command not found: {AGENT_CMD!r}. "
            "Set BATCHER_AGENT_CMD to your agent CLI, or use --dry-run."
        )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _diffstat(cwd: Path) -> str:
    """A one-line summary of what the agent changed, or empty if it changed nothing."""
    _, staged = _run("git add -A && git diff --cached --stat | tail -1", cwd)
    return staged.strip()


def run_pass(
    spec: Pass,
    base: str = "HEAD",
    *,
    dry_run: bool = False,
    keep_rejected: bool = False,
    agent_timeout: int = 1800,
) -> PassResult:
    """Run one pass, verify it, and keep its work only if every check passes.

    Args:
        spec: The pass to run.
        base: Git ref the worktree branches from.
        dry_run: Print what would happen without invoking an agent. Verification still
            runs, so a dry run is a real check that the gate commands themselves work.
        keep_rejected: Leave a rejected pass's worktree in place for inspection.
        agent_timeout: Seconds before the agent is killed.

    Returns:
        The outcome, including why it was rejected if it was.
    """
    started = time.time()

    def done(kept: bool, reason: str, **kw: object) -> PassResult:
        return PassResult(
            name=spec.name,
            goal=spec.goal,
            kept=kept,
            reason=reason,
            seconds=round(time.time() - started, 1),
            **kw,  # type: ignore[arg-type]
        )

    # A read-only pass cannot damage anything, so it runs in the repo and is judged on
    # what it reports rather than on a tree diff.
    if not spec.writes:
        if dry_run:
            return done(True, "dry run: would run read-only review in the repo")
        code, out = _invoke_agent(spec.prompt, REPO, agent_timeout)
        if code != 0:
            return done(False, f"agent exited {code}", output=out[-4000:])
        return done(True, "review completed", output=out[-20000:])

    try:
        worktree, branch = _create_worktree(spec.name, base)
    except RuntimeError as exc:
        return done(False, str(exc))

    # A per-pass directory for "before" state, outside the worktree so capturing a baseline
    # never shows up as a change the agent appears to have made.
    baseline_dir = WORK_ROOT / f"{spec.name}-baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    env = {"AGENTIC_BASELINE": str(baseline_dir)}

    try:
        for cmd in spec.setup:
            code, out = _run(cmd, worktree, env=env)
            if code != 0:
                return done(
                    False,
                    f"setup failed, so there is no baseline to verify against: {cmd}",
                    output=out[-4000:],
                    branch=branch,
                )

        if dry_run:
            agent_out = "(dry run: agent not invoked)"
        else:
            code, agent_out = _invoke_agent(spec.prompt, worktree, agent_timeout)
            if code != 0:
                return done(
                    False,
                    f"agent exited {code}",
                    output=agent_out[-4000:],
                    branch=branch,
                )

        diffstat = _diffstat(worktree)
        # A dry run still runs the gate. That is the point of it: it proves the verification
        # commands themselves execute in a fresh worktree — a gate with a typo'd command or a
        # missing dependency would otherwise sit unnoticed until the night it had to reject
        # something, and then pass everything instead.
        if not diffstat and not dry_run:
            return done(True, "agent made no changes", output=agent_out[-4000:], branch=branch)

        # The verdict: the gate must actually pass here, now, in this worktree.
        failures: list[str] = []
        transcript: list[str] = []
        for cmd in spec.verify:
            code, out = _run(cmd, worktree, env=env)
            transcript.append(f"$ {cmd}\n{out[-2000:]}")
            if code != 0:
                failures.append(cmd)

        if failures:
            verdict = (
                f"{len(failures)} verification command(s) failed"
                if not dry_run
                else f"dry run: {len(failures)} gate command(s) do not pass on a clean worktree"
            )
            return done(
                False,
                verdict,
                output="\n\n".join(transcript)[-20000:],
                failures=tuple(failures),
                diffstat=diffstat,
                branch=branch,
            )

        if dry_run:
            return done(
                True, "dry run: every gate command passes on a clean worktree", branch=branch
            )

        _run(
            f"git commit -m {shlex.quote(f'agentic({spec.name}): {spec.goal}')} "
            "-m 'Generated by tools/agentic/daily.py. Verified, not yet reviewed.'",
            worktree,
        )
        return done(
            True,
            "verified",
            output="\n\n".join(transcript)[-8000:],
            diffstat=diffstat,
            branch=branch,
        )
    finally:
        # Keep a kept pass's worktree so the branch is inspectable; drop rejected ones
        # unless asked, so failures do not silently accumulate on disk.
        if not keep_rejected:
            pass  # cleanup is the caller's decision; see daily.py --clean


def cleanup(name: str) -> None:
    """Remove a pass's worktree, leaving its branch intact."""
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(WORK_ROOT / name)],
        cwd=REPO,
        capture_output=True,
    )
