#!/usr/bin/env python3
"""Codebase-health scanner — the mechanical half of the `audit-codebase-health` skill.

The existing linters each guard one thing: `lint_structure` size, `lint_duplication`
copy-paste, `lint_layers` imports, `lint_docstrings` the public surface. Nothing looks for
the failure modes that survive all four, which are exactly the ones machine-written code
produces: a helper nothing calls, a near-copy one line away from its twin, an `except` that
swallows, a function whose docstring promises what its body does not do, a test with no
assertion.

This is a *report*, not a gate (`--gate` opts into a non-zero exit). Every detector is a
heuristic with a real false-positive rate, so the output is a triage list for a human or an
agent to judge, not a to-do list to apply blindly. Judgment is the point: read the site
before you delete it, and **calibrate before you trust a count** — the first run of the
order-blindness detector reported 150 findings whose mechanical fix broke 82 differential
tests, because an `ORDER BY` inside a window clause does not order a result.

    python tools/audit_health.py                 # everything, grouped by category
    python tools/audit_health.py --only dead-python --only near-duplicate
    python tools/audit_health.py --json report.json
    python tools/audit_health.py --gate          # exit 1 if any `high` finding

The detectors live in `tools/audit/`; this file is the command line over them. Detector
reference and the triage procedure: `.claude/skills/audit-codebase-health/SKILL.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.audit import DETECTORS, SEVERITY_ORDER, Finding, build_context


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only", action="append", choices=sorted(DETECTORS), help="run one detector"
    )
    parser.add_argument("--json", type=Path, help="also write the findings as JSON")
    parser.add_argument("--max", type=int, default=25, help="findings printed per category")
    parser.add_argument("--gate", action="store_true", help="exit 1 when a `high` finding exists")
    args = parser.parse_args()

    ctx = build_context()
    findings: list[Finding] = []
    silent: list[str] = []
    for name in args.only or sorted(DETECTORS):
        produced = list(DETECTORS[name](ctx))
        findings.extend(produced)
        if not produced:
            silent.append(name)

    grouped: dict[str, list[Finding]] = defaultdict(list)
    for finding in findings:
        grouped[finding.category].append(finding)

    for category in sorted(grouped):
        items = sorted(
            grouped[category], key=lambda f: (SEVERITY_ORDER[f.severity], f.path, f.line)
        )
        print(f"\n=== {category} ({len(items)}) ===")
        for finding in items[: args.max]:
            print(f"  [{finding.severity:<6}] {finding.path}:{finding.line}  {finding.message}")
        if len(items) > args.max:
            print(f"  ... {len(items) - args.max} more (raise --max to see them)")

    print("\n--- scorecard ---")
    for category in sorted(grouped):
        items = grouped[category]
        counts = {sev: sum(1 for f in items if f.severity == sev) for sev in SEVERITY_ORDER}
        print(
            f"  {category:<17}{len(items):>5}   high={counts['high']:<5} "
            f"med={counts['medium']:<5} low={counts['low']}"
        )
    high = sum(1 for f in findings if f.severity == "high")
    print(f"  {'TOTAL':<17}{len(findings):>5}   high={high}")
    if silent:
        print(f"  clean: {', '.join(silent)}")

    if args.json:
        args.json.write_text(json.dumps([asdict(f) for f in findings], indent=2))
        print(f"\nwrote {args.json}")

    return 1 if args.gate and high else 0


if __name__ == "__main__":
    sys.exit(main())
