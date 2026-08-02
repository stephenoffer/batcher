#!/usr/bin/env python3
"""Fail when the ways of switching a gate off multiply.

Every gate this repo runs can be silenced in place, and each silencer is individually
defensible: `# noqa` for a finding ruff gets wrong, `# pragma: no cover` for a branch the
suite cannot reach, `# type: ignore` for a library without stubs, `except Exception: pass` for
a probe that is allowed to fail. None of them is a defect on its own.

The defect is the *total*. `.claude/rules/python-quality.md` says "never add a `# noqa` to
silence a real finding — fix the cause", and the tree carries hundreds; a rule that is only
prose loses an argument to a deadline every time. Counting them turns "should I suppress this
or fix it?" from a private judgement into a visible one: the number is in the repo, and it can
go down.

Deliberately a **ratchet, not a limit**. The current counts are not a target — they are a
high-water mark, recorded so the next change cannot quietly raise it. A genuinely necessary
new suppression means paying one off somewhere, or arguing for the ceiling to move.

`ARG002` is called out separately because it is not really a suppression: an unused method
argument is dead weight in a signature, and the rule already says to delete it rather than
silence it. It is the largest single category here and the easiest to actually repay.

Usage:
    python tools/lint_suppressions.py            # check against the ratchet
    python tools/lint_suppressions.py --update   # record a *reduced* set (refuses growth)
    python tools/lint_suppressions.py --report   # print the table, exit 0
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGET = ROOT / "python" / "batcher"
RATCHET = ROOT / "tools" / "suppression_budget.json"

#: label -> pattern. Counted per occurrence, over the control plane only; tests and benchmarks
#: legitimately suppress more and are not the public face of the engine.
PATTERNS: dict[str, re.Pattern[str]] = {
    "noqa": re.compile(r"#\s*noqa"),
    "noqa-ARG002": re.compile(r"#\s*noqa:[^\n]*\bARG002\b"),
    "pragma-no-cover": re.compile(r"#\s*pragma:\s*no cover"),
    "type-ignore": re.compile(r"#\s*type:\s*ignore"),
    "bare-untyped-raise": re.compile(r"\braise (ValueError|RuntimeError|TypeError)\b"),
    "production-assert": re.compile(r"^\s*assert\s", re.MULTILINE),
}


def _allowlist_sizes() -> dict[str, int]:
    """How many files and directories are exempted from the structural limits.

    An allowlist entry is a suppression too — it is how a file stays over the size limit — and
    the structure gate deliberately prints its exemptions on every run so they "stay visible
    and shrink over time". Visible is not the same as counted: the list had grown to 43 files
    and 7 directories, several self-labelled "OVER BUDGET AND TRACKED", with nothing to stop
    the 44th.
    """
    text = (ROOT / "tools" / "lint_structure.py").read_text()
    sizes = {}
    for name, label in (("STRUCTURE_ALLOW", "structure-allow"), ("DIR_ALLOW", "dir-allow")):
        block = re.search(rf"^{name}: dict\[str, str\] = \{{.*?^\}}", text, re.S | re.M)
        sizes[label] = len(re.findall(r'^\s+"[^"]+":', block.group(0), re.M)) if block else 0
    return sizes


def survey() -> dict[str, int]:
    """Occurrences of each suppression pattern across the control plane."""
    counts = dict.fromkeys(PATTERNS, 0)
    for path in sorted(TARGET.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text()
        for label, pattern in PATTERNS.items():
            counts[label] += len(pattern.findall(text))
    return counts | _allowlist_sizes()


def _render(counts: dict[str, int], budget: dict[str, int] | None) -> str:
    width = max(len(k) for k in counts)
    lines = [f"{'suppression':{width}}  {'count':>6}  {'budget':>7}"]
    for label, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        allowed = "-" if budget is None else str(budget.get(label, "-"))
        lines.append(f"{label:{width}}  {count:>6}  {allowed:>7}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="record a reduced set")
    parser.add_argument("--report", action="store_true", help="print the table and exit 0")
    args = parser.parse_args()

    counts = survey()
    if args.report:
        budget = json.loads(RATCHET.read_text()) if RATCHET.exists() else None
        print(_render(counts, budget))
        return 0

    if not RATCHET.exists() or args.update:
        if RATCHET.exists():
            budget = json.loads(RATCHET.read_text())
            grew = {k: v for k, v in counts.items() if k in budget and v > budget[k]}
            if grew:
                print("lint-suppressions: refusing to record a LARGER set:")
                for label, count in sorted(grew.items()):
                    print(f"  {label}: {count} > {budget.get(label, 0)}")
                return 1
        RATCHET.write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n")
        print("lint-suppressions: ratchet recorded")
        print(_render(counts, counts))
        return 0

    budget = json.loads(RATCHET.read_text())
    # Only categories already recorded can regress. A category this tool did not previously
    # measure is a new measurement, and refusing to record it would mean the ratchet could
    # never learn to watch anything new.
    grew = {k: v for k, v in counts.items() if k in budget and v > budget[k]}
    unrecorded = sorted(k for k in counts if k not in budget)
    if unrecorded:
        print(f"lint-suppressions: new categor(ies) {unrecorded} — record them:")
        print("  python tools/lint_suppressions.py --update")
        return 1
    if grew:
        print("lint-suppressions: FAIL — a gate was switched off in more places\n")
        for label, count in sorted(grew.items()):
            allowed = budget.get(label, 0)
            print(f"  {label}: {count}, budget {allowed} (+{count - allowed})")
        print(
            "\nEach of these silences a gate. If the new one is genuinely necessary, pay one\n"
            "off elsewhere and re-record:  python tools/lint_suppressions.py --update"
        )
        return 1

    shrank = {k: v for k, v in counts.items() if k in budget and v < budget[k]}
    if shrank:
        print("lint-suppressions: suppressions were removed — tighten the ratchet:")
        for label, count in sorted(shrank.items()):
            print(f"  {label}: {count} (was {budget.get(label)})")
        print("\n  python tools/lint_suppressions.py --update")
        return 1

    print(_render(counts, budget))
    print("\nlint-suppressions: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
