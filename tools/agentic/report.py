#!/usr/bin/env python3
"""Render a daily-loop run as a dated Markdown report.

The report is the loop's actual product. Nothing is merged automatically, so this file is
how a human decides what to look at — which means it has to be honest about the boring and
the failed cases, not just the wins. A rejected pass and its failing gate commands are more
useful than a kept one, because they point at either a real defect or a gate that needs
fixing; both are printed in full rather than summarized away.

Read-only review passes carry their findings in the agent's own output, so that output is
reproduced verbatim under a collapsed section rather than paraphrased — a summary of a
review is a second chance to lose the finding.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from tools.agentic.runner import PassResult


def _section(result: PassResult) -> str:
    """Render one pass as a Markdown section."""
    lines = [f"### {result.name} — {result.status}", ""]
    lines.append(f"*{result.goal}*")
    lines.append("")
    lines.append(f"- **Outcome:** {result.reason}")
    lines.append(f"- **Duration:** {result.seconds}s")
    if result.branch:
        lines.append(f"- **Branch:** `{result.branch}`")
    if result.diffstat:
        lines.append(f"- **Diff:** {result.diffstat}")
    if result.failures:
        lines.append("- **Failed verification:**")
        lines.extend(f"  - `{cmd}`" for cmd in result.failures)
    lines.append("")
    if result.output:
        label = "Findings" if not result.failures else "Verification output"
        lines.append(f"<details><summary>{label}</summary>")
        lines.append("")
        lines.append("```")
        lines.append(result.output.strip())
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")
    return "\n".join(lines)


def write_report(
    results: list[PassResult],
    report_dir: Path,
    baseline_failures: list[str],
    dry_run: bool,
) -> Path:
    """Write the run's report and return its path.

    Args:
        results: Outcome of every pass that ran.
        report_dir: Directory the dated report is written into.
        baseline_failures: Baseline commands failing before any pass ran.
        dry_run: Whether agents were actually invoked.

    Returns:
        The path of the report written.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    path = report_dir / f"{now:%Y-%m-%d}-daily.md"

    kept = [r for r in results if r.kept and r.diffstat]
    rejected = [r for r in results if not r.kept]
    noop = [r for r in results if r.kept and not r.diffstat]

    lines = [
        f"# Batcher daily loop — {now:%Y-%m-%d %H:%M} UTC",
        "",
    ]
    if dry_run:
        lines += [
            "> **Dry run.** No agent was invoked; verification commands still ran, so this",
            "> proves the gates execute but says nothing about agent output.",
            "",
        ]
    lines += [
        f"**{len(kept)} kept · {len(rejected)} rejected · {len(noop)} no-op**",
        "",
        "Nothing has been merged. Kept passes are branches awaiting review.",
        "",
    ]

    if baseline_failures:
        lines += [
            "## Baseline was red",
            "",
            "These were already failing before any pass ran, so improvement passes could not",
            "be verified against a clean tree and the run was restricted:",
            "",
            *(f"- `{cmd}`" for cmd in baseline_failures),
            "",
        ]

    lines += ["## Summary", "", "| Pass | Status | Outcome |", "|---|---|---|"]
    lines += [f"| {r.name} | {r.status} | {r.reason} |" for r in results]
    lines.append("")

    if kept:
        lines += ["## Ready for review", ""]
        lines += [f"- `{r.branch}` — {r.goal} ({r.diffstat})" for r in kept]
        lines.append("")

    if rejected:
        lines += [
            "## Rejected",
            "",
            "A rejected pass is not a failure of the loop — it is the gate doing its job.",
            "Worth reading: it points at either a real defect or a gate that needs work.",
            "",
        ]

    lines += ["## Detail", ""]
    lines += [_section(r) for r in results]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
