#!/usr/bin/env python3
"""Fail the build on a test that cannot fail.

`CLAUDE.md`'s loudest warning is that "a green gate is not a green light": every gate passed
while `sort(descending=True)` returned unsorted data under spill, and while a distributed
`GROUP BY` on a float key split one group in two. Nothing mechanical caught either, because
the tests ran and passed — they just were not *checking* the thing.

The suite is disciplined about this today, and that discipline lives entirely in prose and
reviewer attention. This is the gate, so it survives the next thousand agent-written tests.

The rules and their calibration live in `tools/audit/testing.py`, which
`tools/audit_health.py --only test-quality` also reports from — one implementation, two
consumers, so the gate and the health report can never disagree about what a bad test is:

``order-blind-test``
    A `sort` / `top_k` / `bottom_k` result compared with `assert_same` or a bare
    `assert_tables_equal`, both of which sort before comparing and are therefore blind to a
    sort bug. This is how the spilled-descending-sort bug stayed invisible.

``vacuous-assertion``
    An assertion true by construction: ``assert True``, ``assert len(x) >= 0``,
    ``assert k == k``. It reads as coverage and checks nothing.

``vacuous-test``
    A test that asserts nothing, raises nothing, expects no warning, and never even calls
    anything at statement level — it binds results and discards them.

Every rule is at zero on this tree. Genuine exceptions go in `ALLOW` with a one-line reason,
never an inline marker, and the allowlist prints on every run so exemptions stay visible.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.audit.testing import check_module, test_modules

#: ``"tests/path/test_x.py::test_name"`` -> reason. Keep this empty where possible.
ALLOW: dict[str, str] = {}


def main() -> int:
    findings = []
    modules = 0
    for path, tree in test_modules():
        modules += 1
        for finding in check_module(path, tree):
            name = finding.message.split("`")[1] if "`" in finding.message else ""
            if f"{finding.path}::{name}" in ALLOW:
                continue
            findings.append(finding)

    if ALLOW:
        print(f"lint-tests allowlist ({len(ALLOW)} entries):")
        for key, reason in sorted(ALLOW.items()):
            print(f"  {key} — {reason}")

    if not findings:
        print(f"lint-tests: clean ({modules} test files)")
        return 0

    by_category: dict[str, list] = {}
    for finding in findings:
        by_category.setdefault(finding.category, []).append(finding)
    for category, items in sorted(by_category.items()):
        print(f"\n=== {category}: {len(items)} ===")
        for item in sorted(items, key=lambda f: (f.path, f.line)):
            print(f"  {item.path}:{item.line}: {item.message}")
    print(
        f"\nlint-tests: FAIL ({len(findings)} findings)\n"
        f"A test that cannot fail is worse than no test — it reports coverage it does not "
        f"have. Fix the assertion; do not add an allowlist entry to make this green."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
