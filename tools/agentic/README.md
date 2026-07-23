# The daily self-improvement loop

Batcher is largely written by agents. This is the scheduled loop that puts that on rails:
agents review the codebase, make narrow improvements, and everything they produce is
verified before a human sees it.

```bash
python tools/agentic/daily.py              # reviews + improvements  (just daily)
python tools/agentic/daily.py --dry-run    # no agent; proves the gates run
python tools/agentic/daily.py --reviews-only   # find work, change nothing
python tools/agentic/daily.py --only perf      # a single pass
python tools/agentic/daily.py --list           # what passes exist
python tools/agentic/daily.py --clean          # remove leftover worktrees
```

Set `BATCHER_AGENT_CMD` to your agent CLI (default `claude -p`). It receives the prompt on
stdin and runs with the worktree as its working directory.

## The safety model

Three properties, each of which exists because its absence caused a real problem here.

**Isolation.** Every writing pass runs in its own `git worktree` under
`/tmp/batcher-agentic`, branched from a committed ref. Your working tree is never touched.
When agents shared one tree, a `git stash` swept up another agent's finished work, a
repo-wide `ruff --fix` rewrote files inside half-finished changes, and test runs reported
failures belonging to somebody else. See `.claude/rules/concurrent-agents.md`.

**Verification, not self-report.** Each pass declares the commands that must exit zero for
its work to be kept, and they are run here, after the agent finishes. An agent's own "all
checks pass" is not evidence — agents have reported that in good faith from a tree that had
gone stale under them.

**No auto-merge.** A verified pass leaves a branch (`agentic/<pass>`) and a report. Nothing
reaches `main` without a human. An unattended agent merging to a shared branch is how a
silent regression ships at 3am; preparing reviewable work is the useful half of the job.

A red baseline downgrades the run to `fix-gate` alone. If the gate is already failing,
"the gate passes after my change" proves nothing.

## What the passes are

Read-only reviews (they report; they never edit):

| Pass | Finds |
|---|---|
| `review-quality` | correctness and quality defects in the recent diff |
| `review-architecture` | layer erosion, duplication across subsystems, docs that have gone stale |
| `review-bench` | where Batcher loses to DuckDB/Polars, and the most promising fix |

Improvements (write in a worktree, behind a gate), cheapest first:

| Pass | Does | Gate |
|---|---|---|
| `fix-gate` | make a red quality gate green, honestly | full Python gate **+ surface unchanged** |
| `fix-index` | repair `MAP.md` and stale agent guidance | map + guardrails + docs tests |
| `fix-docs` | document and *teach* undocumented public names | docstring + docs gates |
| `test-gaps` | cover a missing operator × execution-path combination | ruff + unit |
| `perf` | land a **measured** performance win | Rust gate + differential + **measured benchmark comparison** |

### Before/after gating

A pass can declare `setup` commands that run in the worktree *before* the agent, capturing
the "before" state a `verify` command later compares against. Both get `$AGENTIC_BASELINE`
(a per-pass directory) in their environment. A failed setup **aborts the pass** rather than
letting it verify against a baseline that was never written — a comparison against a missing
file is the kind of check that passes for the wrong reason.

Two uses today:

- **`fix-gate` snapshots the observable surfaces** (`tools/surface_snapshot.py`) and requires
  them identical afterward. Fixing a lint failure has no business moving an optimizer rule's
  position or dropping an IO format — and no test would notice if it did.
- **`perf` snapshots TPC-H timings** (`tools/agentic/bench.py`) and fails if any Batcher
  timing regressed by more than 10%, if a query stopped agreeing with DuckDB, or if a case
  silently stopped being measured. That last one matters: not measuring a query is the
  easiest way to make a regression disappear.

`bench.py` reuses the harness's own `REGISTRY`/`Context`/`compare`, so a benchmark measured
here is measured exactly as `benchmarks/run.py` measures it, correctness check included:

```bash
python tools/agentic/bench.py --save before.json --benchmark tpch
python tools/agentic/bench.py --check before.json --benchmark tpch --tolerance 0.10
```

Passes are narrow and mechanically checkable on purpose. Open-ended "make the code better"
work is deliberately absent: it produces large diffs whose correctness rests on the agent's
judgement rather than on a gate, which is the wrong shape for an unattended loop. That work
belongs in a session with a human in it.

`perf` is the exception — it may touch hot-path code, so it carries the strictest gate here,
and the benchmark harness refuses to time a query whose result disagrees with DuckDB.

## Scheduling it

```cron
# 03:00 daily; report lands in /tmp/batcher-agentic/reports/
0 3 * * *  cd /path/to/batcher && python tools/agentic/daily.py >> /var/log/batcher-agentic.log 2>&1
```

Exit code is non-zero only when a pass **changed something and was then rejected** — the
case worth a notification. A clean or no-op run exits 0.

## Adding a pass

Add a `Pass` to `passes.py` and register it in `REVIEWS` or `IMPROVEMENTS`. Two things
determine whether it is safe to run unattended:

- **`verify` is the contract.** Write the commands that would actually catch this pass going
  wrong, not a generic gate. A weak `verify` is how a regression gets kept.
- **Scope the prompt narrowly.** State what *not* to touch. The shared preamble already
  forbids weakening tests, widening scope, and editing the linters to silence them — the
  three things an unsupervised agent is most likely to talk itself into.

Then run `--dry-run --only <name>` first: it creates the worktree and executes the gate
without an agent, which is how you find out that a verify command has a typo before the
night it matters.

## Limits worth knowing

- The loop branches from a **committed** ref. Uncommitted work is not reviewed; the driver
  warns when the tree is dirty. A pass whose `setup` needs a tool that is not committed yet
  will abort — correctly, but confusingly if you forgot.
- `perf` is **slow**: two full TPC-H runs (baseline + verification) plus a Rust build per
  attempt. Schedule it separately from the cheap passes rather than in the nightly batch.
- Benchmark passes need the benchmark datasets present locally.
- The agent-invocation path is exercised only when an agent CLI is installed. The keep/reject
  logic, worktree isolation, baseline hook, and regression detection are covered by
  `tests/unit/test_agentic_runner.py`, which runs without one.
- `bench.py` exists because `benchmarks/run.py` has no structured output. A `--json` flag
  there would be the tidier home; it was skipped because that file was under concurrent
  edit at the time. Worth folding in later.
