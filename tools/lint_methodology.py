#!/usr/bin/env python3
"""Fail the build on a gate that reports a success it did not measure.

`tools/lint_tests.py` gates the test that **cannot fail**. This gates the neighbouring and
harder case: the test, example or benchmark that *can* fail but has been arranged so that it
does not. That distinction is why this is a separate gate rather than more rules in the other
one — every finding here runs real code, takes real time, and turns green, so nothing about
reading the output distinguishes it from a working check.

Each rule is calibrated to a hand-verified finding set on this tree; the rules and the
reasoning behind each live in `tools/audit/methodology.py`, which
`tools/audit_health.py --only methodology` also reports from. One implementation, two
consumers, so the gate and the health report can never disagree about what a false green is.

What it found the first time it ran, all since fixed:

- **18 findings**, 11 of them `high`. Three SQL differential tests compared an
  `ORDER BY` result with `assert_same`, which sorts both sides, so the order those queries
  asked for was the one property never checked. Two `pytest.parametrize`s ran over a
  directory walk with nothing asserting it found anything — `tests/docs/test_examples.py`
  collects **510 executed example scripts** that way, and a moved directory would have
  turned all 510 into zero tests with the run still green. Seven `examples/` scripts ran to
  completion asserting nothing, one of which
  (`examples/graph/graph_ml.py`) printed `None` three times under the heading "each node
  summarizes its neighbours" — its demonstration node had no in-edges, so the section had
  never worked. One test turned an engine failure into `pytest.skip` behind a bare
  `except Exception`, hiding 48 of its own 98 cases. Two asserted a token was *absent* from
  `explain()` output with nothing anywhere proving that token ever appears.

The last one is worth keeping in mind, because it is the shape that decays on its own:
`plan/profile/render/` turned `explain()` into a table with a header and tree glyphs, and a
helper elsewhere that parsed it with `line.strip().split()[0]` silently began returning
`['query', '────', 'OPERATOR', ...]`. One exact-equality assertion failed loudly. The
`assert "sort" not in _ops(ds)` beside it kept passing, and would from then on have passed
whether or not the sort was eliminated. Nothing in that file changed; the meaning of its
assertions did.

Genuine exceptions go in `ALLOW` with a one-line reason, never an inline marker, and the
allowlist prints on every run so exemptions stay visible. Prefer fixing the finding: an
entry here records a false green that is still there.

Usage:
    python tools/lint_methodology.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.audit.methodology import detect_methodology

#: ``"path::detail"`` -> reason. Keep this empty where possible.
ALLOW: dict[str, str] = {}

#: Categories carried as a **ratchet** rather than a hard zero, with the count they may not
#: exceed. A category belongs here when it is real, large, and cannot be cleared in one
#: change — gating on it outright would leave this file permanently red, and a gate that
#: cannot change state is one everybody learns to walk past. (That is not a hypothetical:
#: `tools/skip_budget.json` recorded numbers that were already wrong in the commit that
#: wrote them, so `just lint-skips` returned 1 at every commit afterwards and was ignored
#: until someone measured it.)
#:
#: The number may fall and may never rise. Re-record it *down* when you fix some.
RATCHET: dict[str, int] = {
    # Standalone timing entry points under `benchmarks/` that never call
    # `require_release_build`, so they can publish a number from a debug build — 8-60x
    # slower by the guard's own docstring — without saying so. Was **60 of 64**; 40 have
    # since been guarded, each verified to still load when run as a script (the check that
    # matters, because the guard needs a `sys.path` bootstrap and a test that pre-seeds
    # `sys.path` cannot see a wrong one).
    #
    # The remaining 20 are the cluster and GPU scripts, which cannot even be imported on a
    # box without Ray or a device, so a guard added to one of them could not be verified
    # here at all — and an unverifiable one-line "fix" across 20 files is how a plausible
    # Standalone timing entry points under `benchmarks/` that never call
    # `require_release_build`, so they can publish a number from a debug build — 8-60x slower
    # by the guard's own docstring — without saying so. **Cleared: 64 of 64 now guarded**, so
    # this is 0 and stays listed rather than deleted, for the reason the sibling entry gives.
    #
    # It was 60, then 20, and the last 20 nearly stood: the cluster/GPU scripts were reported
    # three times as unverifiable on a box with no Ray and no device. That was a *harness*
    # bug, twice over. Both attempts (`importlib.import_module`, then `runpy.run_path`) omit
    # the script's own directory from `sys.path`, which `python <file>` supplies — so every
    # one failed on a sibling `from _ray_env import ...` and the failure read as "needs Ray".
    # Two confirming measurements from one broken instrument felt like corroboration and were
    # one error counted twice. With the directory inserted, all 20 load cleanly and were
    # always verifiable to the standard the other 40 were held to.
    #
    # Worth keeping because it is the *conservative* direction of the failure this file is
    # about: every other rule here guards against claiming more than was measured, and a
    # broken instrument is equally capable of hiding work that should have been done.
    "benchmark-unguarded-build": 0,
    # Rust `#[ignore]`s carrying no reason. All were genuinely timing studies, verified by
    # reading each body — but that is the point: from the outside they are indistinguishable
    # from a test quarantined because it broke, and `cargo test`'s "N ignored" reads as
    # success. All seven now name what they measure, so this is **0** and the entry stays
    # only to keep the category gated: a ratchet at zero is a hard gate that says what it is
    # protecting, where deleting the line would let the next one in silently.
    "rust-ignore-without-reason": 0,
}


def main() -> int:
    produced = [f for f in detect_methodology(None) if f"{f.path}::{f.line}" not in ALLOW]

    ratcheted: dict[str, list] = {}
    findings = []
    for finding in produced:
        if finding.category in RATCHET:
            ratcheted.setdefault(finding.category, []).append(finding)
        else:
            findings.append(finding)

    for category, budget in sorted(RATCHET.items()):
        count = len(ratcheted.get(category, []))
        status = "OK" if count <= budget else "OVER BUDGET"
        print(f"lint-methodology ratchet: {category} = {count} (budget {budget}) [{status}]")
        if count > budget:
            findings.extend(ratcheted[category])

    if ALLOW:
        print(f"lint-methodology allowlist ({len(ALLOW)} entries):")
        for key, reason in sorted(ALLOW.items()):
            print(f"  {key} — {reason}")

    if not findings:
        print("lint-methodology: clean")
        return 0

    by_category: dict[str, list] = {}
    for finding in findings:
        by_category.setdefault(finding.category, []).append(finding)
    for category, items in sorted(by_category.items()):
        print(f"\n=== {category}: {len(items)} ===")
        for item in sorted(items, key=lambda f: (f.path, f.line)):
            print(f"  {item.path}:{item.line}: {item.message}")

    print(
        f"\nlint-methodology: FAIL ({len(findings)} findings)\n"
        f"Each of these runs, takes time, and passes — while not checking the thing it is "
        f"named for. Fix the check; an allowlist entry only records that the false green is "
        f"still there."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
