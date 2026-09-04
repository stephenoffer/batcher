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
    # A test hand-listing a few members of a production enumeration it names, where what it
    # leaves out is never exercised. Budget 15 = the measured population, untriaged: roughly a
    # third are real gaps and the rest are deliberate samples, and **nothing mechanical
    # separates them** (see the rule's docstring). The budget stops the shape spreading; it
    # does not claim the 15 are defects.
    #
    # The evidence is one shipped defect, not three. `FORMATS` is hand-listed as
    # `["pyarrow", "numpy", "pandas"]` against six formats and has been since `18b85ead`; the
    # uncovered `polars` silently widened every `string` to `large_string` through
    # `map_batches`, so `Dataset.schema` and `collect()` disagreed and a Parquet file written
    # from that plan was `large_string` on disk. That gap predates the defect's discovery.
    #
    # The `_ROUNDING`/`_IDEMPOTENT_MATH` hits were cited as further evidence when this rule
    # landed and they are **not**: running it against `87d82730^` gives zero findings there.
    # `trunc` was inside the tested list before that fix, and the finding exists only because
    # the fix moved it out. Both are false positives -- `sign` likewise, covered by a sibling
    # test asserting the opposite behaviour (`sign(NaN)` is 0.0).
    #
    # Known-deliberate entries, so nobody re-triages them: `STR_FNS` (4 of 105, a security
    # test sampling), `MATH_FNS` (GPU conformance sampling), `_TRANSIENT_MARKERS`,
    # `_BUILD_ARTIFACT_EXCLUDES`, and `WINDOW_RANKING` -- whose test docstring explains that
    # the rule under test keeps a narrower `_PREFIX_STABLE_RANKING` because the other members
    # divide by a partition total. Re-record **down** as the real ones are derived from their
    # constants.
    # 15 -> 14: `test_resilience_profile` now derives its platform-marker parametrization
    # from `_MANAGED_AUTOSCALE_VARS` instead of retyping five of the six. The omitted
    # `ANYSCALE_CLUSTER_ID` was a real gap and contradicted the test's own claim that no
    # vendor marker is privileged.
    # 14 -> 13: `test_object_store_portability` now derives from `_OBJECT_STORE_SCHEMES` and
    # its alias table, and asserts the *behaviour* (`atomic_rename is False` out of
    # `_wrap_user_filesystem`) rather than membership in the set it drew the scheme from,
    # which was true by construction.
    # 13 -> 11 after three refinements from b9 (see the rule docstring): resolving set
    # arithmetic so a derived constant is visible, suppressing a list that exactly equals some
    # production set, and counting *distinct* values so a `parametrize` row of repeated
    # literals is not read as an enumeration.
    # 11 -> 6: a file that *imports* the constant and derives from it is enumerating it,
    # whatever local classification sets it also defines. Keyed on the import rather than the
    # name, so it does not hide a file that hand-copies the vocabulary instead of importing it
    # -- which is exactly what the surviving `JOIN_TYPES` finding does.
    #
    # Of the 6 that remain: `JOIN_TYPES` and `FORMATS` are the two b9 verified with a
    # per-revision counterfactual (present since each test was written, no narrowing), and
    # `FORMATS`'s uncovered `polars` was a shipped engine defect. `_BUILD_ARTIFACT_EXCLUDES`,
    # `MATH_FNS` and `WINDOW_RANKING` are known deliberate samples. `_FOLDABLE_MATH` is new
    # from set-arithmetic resolution and untriaged.
    # 6 -> 5: `import batcher.x.y as ax` + `ax._CONST` reaches the constant as directly as
    # `from batcher.x.y import _CONST`, and only the second is an `ImportFrom`.
    #
    # 5 -> 1: three were paid off by deriving the parametrization from the production set and
    # *classifying* what it cannot cover, which is the shape the finding asks for and is worth
    # stating because "parametrize over the whole set" is the wrong fix for all three:
    # `FORMATS` splits into the formats that carry a string column and the two that have no
    # representation for one (they drop it, with a warning, and that decline is now its own
    # case); `MATH_FNS` splits into the thirty-one the device tier translates and the four it
    # declines to the CPU engine; `_BUILD_ARTIFACT_EXCLUDES` needed neither -- a containment
    # check over a list built from the tuple is a tautology -- so each pattern now names the
    # build output it must match and is held against paths the upload must carry.
    # The remaining 1 is `JOIN_TYPES`, whose file was another session's at the time.
    "shadowed-production-set": 1,
    # A test comparing two engine runs on a figure describing *how* they ran -- CPU
    # utilization, thread count -- where nothing forced the difference it asserts. Budget 1,
    # and the one is a live failure rather than an accepted shape:
    # `test_hardware_telemetry.py::test_streaming_cpu_utilization_is_measured_not_a_constant`
    # varies the row count and asserts utilization rises, which on a shared box reports the
    # neighbour's load. Its owner has it characterised as load-sensitive and is on it; the
    # budget exists so the shape cannot spread while that is true, not to bless it. **Re-record
    # to 0 when it lands** -- the number may fall and may never rise.
    #
    # Two helpers matched the shape and were correctly exonerated, which is what makes the
    # rule worth having rather than a name-based heuristic: `threads_at`
    # (test_diff_morsel_size_invariance) and `run_with` (test_spilling) both `set_config(...)`
    # before running, so they *force* the difference and then prove it reached the engine.
    # Setting a knob is the discriminator between a control and a hope.
    "uncontrolled-runtime-comparison": 1,
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
