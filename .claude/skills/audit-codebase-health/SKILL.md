---
name: audit-codebase-health
description: Critically audit the whole Batcher tree for the decay that machine-written code produces — dead code, near-duplicate logic, swallowed errors, do-nothing bodies, oversized files, speculative abstractions, tests that cannot fail, and claims the code does not support — then fix what is real and prove it with the gate. Invoke for a periodic health pass, before a release, or when asked whether the codebase is production-ready.
---

# Audit the codebase for health and production readiness

This skill is adversarial by design. Assume the code was written by an agent under time
pressure that wanted the gate to go green, because most of it was. Your job is to find
what the gate cannot see and fix it. Report the number you actually measured, not the
number you hoped for.

Read `.claude/rules/concurrent-agents.md` **first**. This audit sweeps the whole tree, so
it is the single most likely task to clobber another agent's in-flight work.

## Before anything: establish what is yours

```bash
git status                 # your session snapshot is stale; read it yourself
git log --oneline -3
```

Every file that is already dirty belongs to someone else until proven otherwise. Note
them, exclude them from your fix set, and say so in your report. **Never `git stash`,
`git checkout .`, or run a repo-wide `ruff --fix` / `cargo fmt --all`** — scope every fix
command to the paths you are changing.

## Step 1 — measure, and keep the numbers

Run all of it before judging any of it. The scorecard is the artifact a periodic run
compares against, so capture it.

```bash
python tools/audit_health.py --json /tmp/audit-$(date +%F).json   # or: just audit-health
python tools/lint_structure.py          # file/dir/class size, allowlist ledger
python tools/lint_duplication.py        # exact copy-paste across modules
python tools/lint_guardrails.py         # agent guidance citing paths that exist
lint-imports --config pyproject.toml    # layer independence
ruff check python tests benchmarks examples
cargo clippy --workspace --exclude bc-py -- -D warnings
```

`tools/audit_health.py` carries the detectors the other linters do not have. Every one is
a heuristic with a real false-positive rate — **the output is a triage list, not a to-do
list.** Read the site before you touch it.

| Detector | Finds | Read it as |
|---|---|---|
| `dead-python` | a definition no other file mentions, and its own file mentions once | delete it, or explain why it survives |
| `dead-rust` | a column-0 `pub` item reached only by its own unit tests | a primitive built for a caller that never arrived |
| `near-duplicate` | cross-file bodies ≥85% identical | the copy that drifted one line, invisible to `lint_duplication` |
| `stub` | a documented function with a do-nothing body, excluding `-> X \| None` (a documented "unknown" default) | a docstring promising what the body does not do |
| `swallowed-error` | a handler that *declares itself* best-effort and leaves no trace; a comment-less `pass`; a broad `except` over a large `try` | the first is the real one — the learned-stats loop dying with every gate green |
| `vacuous-test` | a `test_*` with no assertion anywhere in its call graph, that never calls anything at statement level either | binds results and discards them |
| `vacuous-assertion` | `assert True`, `assert len(x) >= 0`, `assert k == k` | reads as coverage, checks nothing |
| `order-blind-test` | a `sort`/`top_k`/`bottom_k` result compared with `assert_same` or a bare `assert_tables_equal` | the comparison sorts both sides, so it cannot see a sort bug |
| `production` | `print` outside a display function, `assert` on a public function's argument, mutable defaults | ships to a user as-is |
| `suppression` | `# noqa`, `# type: ignore`, `#[allow(...)]` | a gate that was silenced instead of satisfied; watch the count |

Grep carries the rest, and the counts are the point — a number that grew since the last
run is the finding:

```bash
rg -n --stats 'NotImplementedError' python/batcher      # stubs on live paths
rg -n 'unwrap\(\)|\.expect\(|panic!\(' crates --stats   # panics that can see user data
rg -n '@pytest.mark.(skip|xfail)' tests                 # tests that stopped running
rg -n 'time\.sleep' python/batcher                      # busy waits in the control plane
rg -n '/tmp/|localhost|127\.0\.0\.1' python/batcher     # hardcoded environment

# Directory breadth. `lint_structure.py` caps *files* per directory but not
# subdirectories, so a package can fan out sideways without anything noticing.
for d in $(find python/batcher crates tools -type d ! -path '*__pycache__*'); do
  n=$(find "$d" -maxdepth 1 -mindepth 1 -type d ! -name __pycache__ | wc -l)
  [ "$n" -gt 10 ] && echo "$d: $n subdirs"
done
```

Read a breadth number against what the directory *is*. `python/batcher/` at 14 and
`crates/` at 13 are the layers and the crate DAG — every entry is named in the import
matrix or the DAG, so collapsing one breaks a hard invariant and the invariant wins.
`.claude/skills/` runs wide because a skill is selected by its directory name; nesting
them breaks invocation and the `*/SKILL.md` glob in `tests/docs/test_skill_coverage.py`.
A breadth number that is *not* load-bearing in that way is a finding.

Check the allowlist *reasons* against reality while you are here, not just the entries:
`DIR_ALLOW["python/batcher/kyber"]` says "sits at 13 modules by one" and the directory
holds 17. The exemption still applies; its justification has quietly stopped being true,
which is how "tracked debt" turns into permanent debt.

## Step 1b — calibrate the detector before you trust its count

A large finding count is a claim about the *detector* until you have tested it against the
code. Take three findings from the biggest category, open them, and decide whether each is
real. If two of the three are false positives, **fix the detector, not the code** — and say
so in the report, because "150 order-blind tests" and "31 candidates, one mechanical" are
very different facts about this codebase.

This is not hypothetical. The first run of `order-blind-test` reported 150 findings; the
mechanical fix turned them into **82 failing differential tests**, because an `ORDER BY`
inside an `OVER (...)` window clause, a `WINDOW w AS (...)` definition, or a `string_agg(x
ORDER BY y)` ranks rows *within* a partition or an aggregate and says nothing about the
order of the result. The same run flagged 21 `production` findings, every one of which was
a `print` inside `ds.show()`/`glimpse()`/the console sink or an `assert isinstance(...)`
narrowing for the type checker. After calibration: 31 and 0.

Two later calibrations, for the same reason — record yours the same way:

`swallowed-error` was measuring the wrong thing in **both** directions. All 18 `except ...:
pass` sites in the tree turned out to be legitimate (documented control-flow fall-throughs,
optional imports, browser-disconnect handling), while 47 handlers that *declared themselves*
best-effort — "learning must never break a query" — substituted a fallback and returned with
no trace at all, and the detector saw only three of them. The declaration is the signal, not
the syntax: a broad `except Exception: return None` on an optional-dependency probe is fine,
and the same handler on a path the author promised would never break is the moat going quietly
dead. Keying on the syntax alone reported 190 sites, ~150 of them legitimate. Also: `continue`
is not silence (inside a loop it means "skip this entry" — all 16 sites were legitimate), and
a handler that reads its bound exception is not silent either, because threading it into a
reason string the caller logs carries it just as well.

The three test-quality detectors were then rewritten from regex-over-source to AST, because
the calibrations above had driven the count down without fixing the *kind* of error: the
remaining 23 findings were still all false. The regexes read `.sort(` inside a string literal
as a sort (flagging the detector's own test file), and a `warnings.simplefilter("error")`
block — how every "...stays silent" test here is written, and a real negative assertion — as
a test that asserts nothing. Reading the parse tree removed all 23. That precision is what
promoted these three from a report to a **gate**: `just lint-tests` now fails the build on
them, over the same `tools/audit/testing.py` this report reads, and
`tests/unit/test_lint_tests.py` feeds each rule a violation it must catch and each historical
false positive it must not. A detector calibrated to zero can be a gate; one that cannot be
is still a report.

`order-blind-test` reported 32; 30 were false and 2 were real. The 30 were the detector
treating *inner* ordering as result ordering — an `order_by=` **keyword** picks which row an
aggregate keeps, which duplicate `distinct` survives, or how a window frame ranks rows, and
never orders the result; and a sorted relation feeding `GROUP BY`/`UNION` has no result order
left to check. The 2 real ones both covered a Kyber rule that *rewrites a sort* while comparing
against an oracle query with no `ORDER BY`, so the rule could have destroyed the ordering it
was rewriting and the test would still have passed.

And one that is not a detector but changes every count you read: `lint_structure`'s
function-length check counts the docstring. Public functions here carry mandatory Google-style
docstrings with runnable doctests, so 117 of the 214 functions it flagged were over the line on
docstring alone. It now excludes them, the way the Rust file check excludes `#[cfg(test)]`.
**40% of the Python tree is docstring** — measure code, not lines, before calling anything bloated.

So: **revert a bad batch rather than arguing with the suite.** A conversion that turns a
green suite red is evidence about your rule, not about the tests — unless you can point at
the specific wrong answer, in which case you have found a real bug and should say so loudly.
And when a category calibrates to zero, that is a result worth reporting: it means the
codebase does not have that problem, and the detector now guards against acquiring it.

## Step 2 — judge, with the invariants as the yardstick

Mechanical findings are the cheap half. These are the ones that pass every gate:

**Dead code is not only unreferenced code.** A registry with one entry, a `Protocol` with
one implementation, a config flag nothing reads, an `Executor` strategy that is never
selected — all of it is reachable, and all of it is dead weight. `python-quality.md` bans
speculative generality: no abstraction without a second implementation or an imminent one.
An empty framework is a finding even when every line of it is called.

**Duplication has a direction.** `kyber`, `carbonite`, `core`, and `governance` cannot
import each other, so copy-paste is the *only* wrong way to share between them. A helper
that appears in two subsystems does not get deduplicated sideways — it moves **down** into
`plan` / `metadata` / `config` / `_internal`. Check that the third copy is not already
sitting in a fourth spelling (`_median` reached four).

**An allowlist keyed by line number is a time bomb.** `DUPLICATION_ALLOW` keyed its one
entry at `flight_sort.py:332` while the function had drifted to line 343, so
`just lint-duplication` — a pre-commit hook — failed at `HEAD` for a reason unrelated to
any change that tripped it. Check that every ledger key still resolves, and prefer a key
that survives an edit above it (file + symbol, not file + line).

**A structure violation is a question, not a verdict.** Before splitting an oversized file,
ask whether the split would cross a layer, fork the JSON IR wire contract, or break the
crate DAG. If it would, **the invariant wins**: leave the file oversized, add it to
`STRUCTURE_ALLOW` with a one-line reason, and say so. The checker is the thing pushing you
the wrong way. Equally: an allowlist that only grows is debt with a nice hat on. Compare
today's `STRUCTURE_ALLOW` / `DUPLICATION_ALLOW` / `DIR_ALLOW` against the last run and
justify every addition.

**Best-effort is not free.** `except Exception: pass` under a comment saying "learning must
never break execution" means the learned-stats loop — the moat — can be broken forever with
no signal. Best-effort is a legitimate choice; *silent* best-effort is not. The fix is a
trace, not a re-raise: `batcher._internal.logging.note_suppressed(subsystem, step, exc)`
records it at DEBUG and exists for exactly this. Narrow the handler instead where the
failure is an optional import (`except ImportError`) rather than a degraded capability.

**A test that cannot fail is worse than no test**, because it reports coverage it does not
have. Three shapes to hunt: no assertion at all; an assertion only about a mock the test
itself configured; and an order-independent comparison of an ordered result. `assert_same`
is order-independent *by design* — reaching for it on a sorted result is exactly how a sort
bug stays invisible. `assert_same_ordered` exists in `tests/_harness.py`; use it.

**Coverage of the contract, not of the lines.** The cross-product that matters is
`{collect, spill, iter_batches, distributed} × {nulls, empty, one row, duplicates,
-0.0/NaN, descending}`. A 62% line-coverage gate says nothing about whether any operator
was ever run with a non-default flag on a non-default path.

**Claims are code too.** `docs/architecture/internals/competitive_architecture.md` is the code-checked
scorecard. A performance claim with no committed benchmark behind it, a docstring
describing an argument the signature does not have, a `MAP.md` entry for a module that
moved — each is a defect with the same severity as a wrong result, because it is what the
next agent will build on.

## Step 3 — fix, narrowly and in order

Fix in this order, because each step makes the next one cheaper and lower-risk:

1. **Delete.** Dead code, dead branches, dead flags, commented-out blocks. Git history is
   the archive. Deleting is the only fix with no regression surface — do it first and
   verify, so the later steps run against a smaller tree.
1. **Make silent failures loud.** Swallowed exceptions, do-nothing bodies, `assert` used as
   validation. Prefer the project's typed errors (`batcher._internal.errors`) and the event
   bus over a new logger.
1. **Deduplicate downward.** Move the shared helper into the neutral layer and import it
   from both sides. Then run `just surface-diff` — a dedup that changed observable behavior
   is a bug, not a cleanup.
1. **Repair the tests you just proved were lying.** Fix the assertion before touching the
   code it covers; a vacuous test that starts asserting sometimes fails immediately, and
   that failure is the real finding.
1. **Restructure.** Oversized files and directories last, because a move invalidates
   allowlist keys, registration order, monkeypatch targets, and doc paths.

**One concern per commit.** A commit that deletes dead code *and* splits a file *and*
rewrites a test cannot be bisected, and it is the shape that hides a regression.

**Every fix carries its own proof.** A deletion proves nothing needed it; a dedup proves
`surface-diff` is empty; a test fix proves the assertion fails against the old behavior.
"I only moved code" is not evidence.

## Step 4 — verify, and report honestly

Run the rows the `CLAUDE.md` gate matrix names for what you touched, then re-run the audit
and diff the scorecard:

```bash
python tools/audit_health.py            # compare against the run from Step 1
just lint-py && just lint-layers && just lint-structure && just lint-duplication
just build && just test-py              # the whole suite, at the end, after things settle
just check && just test-rust            # if any Rust changed
just surface-diff /tmp/before.json      # if anything moved
```

A failure count taken mid-refactor is not a fact. **Re-run at the end and report that
number.** If a test was already failing at `HEAD`, prove it (`git status <file>`, check out
a clean copy) before calling it pre-existing — that phrase is the most common false
statement in this repo's agent reports.

The report says, in this order: the scorecard before and after; what you fixed; what you
found and deliberately did **not** fix, with the reason; what you did not touch because it
belonged to another agent; and the single most important remaining risk. A finding you
chose to leave is a legitimate outcome. A finding you quietly dropped is not.

## What good looks like

- Zero `high` findings from `audit_health.py`, or every remaining one has a written reason.
- `lint-structure`, `lint-duplication`, `lint-layers`, `ruff`, and `clippy` clean, with no
  allowlist entry added that a reader would not accept.
- No suppression added to silence a real finding.
- The differential suite is green and *stronger* than it was: no test weakened or deleted to
  make a change pass. If Batcher and DuckDB legitimately differ, that is a decision surfaced
  in the report, never a test quietly relaxed.

## Known limits of the tooling

Say these out loud in the report rather than implying the scan was exhaustive.

- `dead-python` / `dead-rust` match identifiers as text, so a short or common name
  (`execute`, `combine`, `state`) reads as referenced wherever the word appears. The
  detectors under-report by design; absence of a finding is not proof of absence.
- Nothing here detects a *semantic* duplicate — two implementations of the same idea with
  different structure. That needs reading, and it is where the worst duplication lives.
- Detector calibration is empirical and never finished. Every threshold in `tools/audit/`
  was set by running the fix and watching the suite, so a detector that has not been
  re-calibrated against the current tree is a detector whose count you should not quote.
- `tools/audit_health.py` is the CLI; the detectors live in `tools/audit/`, one module per
  family, each under the 500-line bar the rest of the tree holds to. It reached 817 lines as
  one file and was split — with a before/after `--json` diff as the equivalence proof. Keep
  it that way rather than letting a `tools/` file grow because nothing checks it.

## See also

- `run-quality-gate` — the gate sequence and how to triage each failure class.
- `.claude/rules/maintainability.md` — the size limits and the structure conventions.
- `.claude/rules/python-quality.md` — dead code, duplication, and public-API rules.
- `.claude/rules/testing.md` — the oracles and the per-change test requirements.
- `.claude/rules/concurrent-agents.md` — how not to destroy another agent's work.
- `tools/agentic/README.md` — the scheduled loop that runs improvement passes in isolated
  worktrees; this skill is the review it should be pointed at.
