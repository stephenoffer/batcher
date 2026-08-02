#!/usr/bin/env python3
"""The catalog of daily passes: what agents are asked to do, and what must hold after.

Two kinds of pass, and the distinction is the whole safety story.

**Review passes are read-only.** They find and report; they never edit. A reviewer that
can also rewrite tends to "fix" what it found before anyone judged whether the finding was
real, and a wrong finding then becomes a wrong change. Their output is the report.

**Improvement passes write, in an isolated worktree, behind a gate.** Each carries the exact
commands that must exit zero for its work to be kept. Those commands are the contract: an
agent cannot argue its way past them, and a pass whose gate is weak is a pass that can ship
a regression, so `verify` is written to be specific rather than generous.

Every improvement pass here is deliberately **narrow and mechanically checkable** — fix a
lint failure, split an oversized file, document an undocumented public name. Open-ended
"make the code better" work is not in this catalog on purpose: it produces large diffs whose
correctness rests on the agent's judgement rather than on a gate, which is the opposite of
what an unattended loop should do. Broad changes belong in a session with a human in it.

`PERF` is the exception that proves the rule: it may change hot-path code, so its gate is
the strictest here — correctness first (the harness refuses to time a query whose result
disagrees with DuckDB), then the differential suite, then the benchmark itself.
"""

from __future__ import annotations

from tools.agentic.runner import Pass

#: Commands common to any Python change. Kept as one tuple so a pass cannot accidentally
#: omit a gate the project already considers mandatory.
_PY_GATE = (
    "ruff check python tests benchmarks examples",
    "python tools/lint_structure.py",
    "python tools/lint_guardrails.py",
    "lint-imports --config pyproject.toml",
    "python tools/gen_map.py --check",
    "python -m pytest tests/unit tests/docs -q",
)

_RUST_GATE = (
    "cargo check --workspace --exclude bc-py",
    "cargo clippy --workspace --exclude bc-py -- -D warnings",
    "cargo test --workspace --exclude bc-py",
)

#: Prepended to every writing pass. States the constraints that are not negotiable and that
#: an agent working alone, unattended, is most likely to talk itself out of.
_PREAMBLE = """\
You are running unattended in an isolated git worktree as part of Batcher's daily
self-improvement loop. Read CLAUDE.md first — it is the contract, and its invariants
outrank anything in this prompt.

Non-negotiable:
- Your work is KEPT only if every verification command passes. You cannot override this.
- Do NOT weaken, skip, delete, or xfail a test to make a gate pass. If a gate fails and
  you cannot fix it honestly, STOP and leave the tree unchanged — a rejected pass is a
  fine outcome and far better than a weakened oracle.
- Do NOT widen your scope. Fix only what this task names. A large diff will be rejected
  on review even if it passes.
- Do NOT edit tools/lint_*.py to silence a check, and do not add to STRUCTURE_ALLOW
  unless a hard invariant genuinely blocks the split (say which one, in the entry).
- If you move or rename anything, run `just map` and `just lint-guardrails`.

"""


def _p(
    name: str,
    goal: str,
    body: str,
    verify: tuple[str, ...],
    writes: bool = True,
    setup: tuple[str, ...] = (),
) -> Pass:
    """Build a pass, prefixing the shared preamble for writing passes."""
    return Pass(
        name=name,
        goal=goal,
        prompt=(_PREAMBLE + body if writes else body),
        verify=verify,
        writes=writes,
        setup=setup,
    )


#: Capture the observable surfaces before the agent runs, and require them unchanged after.
#: Applied to passes that are supposed to be behaviour-preserving: a structural fix or a
#: docs fix has no business moving an optimizer rule's position or dropping an IO format,
#: and those are exactly the changes that no test would notice.
_SURFACE_SETUP = ('python tools/surface_snapshot.py --save "$AGENTIC_BASELINE/surface.json"',)
_SURFACE_VERIFY = ('python tools/surface_snapshot.py --diff "$AGENTIC_BASELINE/surface.json"',)


# --- Review passes (read-only; they report, they never edit) -------------------------

REVIEW_QUALITY = _p(
    "review-quality",
    "Find real correctness and quality defects in the recent diff",
    """Review this repository's recent changes for defects. Read CLAUDE.md first.

Scope: `git diff main...HEAD` plus anything it touches.

Report only findings you can justify concretely — for each, give the file:line, the
failure scenario (inputs -> wrong output), and why the existing tests miss it. Rank by
severity. An empty report is a valid and useful result; do not manufacture findings to
seem productive.

Pay particular attention to the failure modes this codebase actually has:
- an operator correct under `collect` but wrong under spill / iter_batches / distributed
- a sort asserted with an order-independent comparison (which cannot see a sort bug)
- a stateful operator with no mergeable partial/combine/finalize form
- a JIT path that diverges from the interpreter instead of falling back
- Python `to_ir()` tags out of lockstep with the Rust serde tags
- per-row work that leaked into the Python control plane

Do NOT edit any file. Output findings only.""",
    verify=(),
    writes=False,
)

REVIEW_ARCHITECTURE = _p(
    "review-architecture",
    "Find layering, structure, and documentation drift",
    """Audit this repository for structural and documentation drift. Read CLAUDE.md first.

Report, with file:line evidence:
1. Layer-contract erosion: an import that crosses the matrix in `.claude/rules/architecture.md`,
   or a subsystem doing another's job (Kyber executing, Core optimizing, Carbonite rewriting).
2. Code duplicated across kyber/carbonite/core/governance — they cannot import each other, so
   shared logic must move DOWN into plan/metadata/config/_internal, never be pasted.
3. Modules whose docstring no longer matches what they contain, or that claim a
   responsibility living elsewhere (this has happened: a crate doc advertised sorting that
   lives in a different crate).
4. Guidance in CLAUDE.md / .claude/rules / docs/architecture/internals that is no longer true.
5. Dead code: unreferenced functions, files not declared in their module tree, orphaned
   allowlist entries in tools/lint_structure.py.

Do NOT edit any file. Output findings only, ranked by how likely each is to mislead
someone into putting new code in the wrong place.""",
    verify=(),
    writes=False,
)

REVIEW_BENCH = _p(
    "review-bench",
    "Find the most promising performance work from benchmark evidence",
    """Identify Batcher's best available performance wins, from evidence rather than intuition.

Read `.claude/rules/performance.md`, `docs/architecture/internals/competitive_architecture.md`, and the
benchmark results in `benchmarks/BENCHMARK_RESULTS.md` / `TPCH_FINDINGS.md`.

Report the shapes where Batcher currently loses to DuckDB or Polars, worst ratio first. For
each: name the operator and the code path that dominates, state the hypothesis for why it is
slow, and propose a specific change and how to measure it. Distinguish what the evidence
shows from what you are inferring — say which is which.

Do NOT edit any file, and do NOT report a speedup you have not measured.""",
    verify=(),
    writes=False,
)


# --- Improvement passes (write, in a worktree, behind a gate) ------------------------

FIX_GATE = _p(
    "fix-gate",
    "Make the quality gate green without weakening it",
    """The repository's quality gate is failing. Make it pass honestly.

Run these to see the current state:
    ruff check python tests benchmarks examples
    python tools/lint_structure.py
    python tools/lint_guardrails.py
    lint-imports --config pyproject.toml
    python -m pytest tests/unit tests/docs -q

Fix the underlying cause of each failure. Constraints:
- A `# noqa` or a skipped test is not a fix; fix the cause.
- If a file is over the size limit, split it on a responsibility seam and re-export so
  the public import path is preserved. If splitting would cross a layer or fork the
  wire contract, the invariant wins: allowlist it with the reason instead.
- If a failure is genuinely not fixable in scope, leave it and say so — a partial,
  honest fix is better than a broad one you cannot justify.

Fixing a gate is behaviour-preserving: the observable surfaces (optimizer rule order,
IR tags, public API, IO registry, FFI signatures) are snapshotted before you start and
must be identical afterward. Splitting a file must re-export so nothing moves.""",
    verify=_PY_GATE + _SURFACE_VERIFY,
    setup=_SURFACE_SETUP,
)

FIX_DOCS = _p(
    "fix-docs",
    "Document and teach public API names that are missing coverage",
    """Some public API names fail the documentation gates. Fix that.

    python -m pytest tests/docs -q
    python tools/lint_docstrings.py

For each failing name: write a Google-style docstring (one-line summary inline with the
quotes, `Args:`/`Returns:` without types, a runnable `Examples:` block in a `.. doctest::`),
make sure it is rendered by Sphinx autodoc, and *teach* it somewhere outside the generated
reference — a user guide, tutorial, or `examples/` script.

The doctest examples are executed by the docs build, so an example that lies fails. Use
`# doctest: +SKIP` only for examples genuinely needing a GPU, cloud store, or real model.
Do not delete a name to make the gate pass.""",
    verify=(
        "ruff check python tests benchmarks examples",
        "python tools/lint_docstrings.py",
        "python -m pytest tests/docs -q",
    ),
)

FIX_MAP = _p(
    "fix-index",
    "Repair the index and the agent-facing guidance",
    """`MAP.md` or the agent-facing guidance is stale. Repair it.

    python tools/gen_map.py --check
    python tools/lint_guardrails.py
    python -m pytest tests/docs/test_agent_context.py -q

`MAP.md` is generated — never hand-edit it; run `python tools/gen_map.py`. If a module's
one-liner reads poorly in the map, fix the module's own docstring, since the map quotes it.
For guardrail failures, correct the stale path in the guidance (do not delete the guidance).
Curated prose belongs in `tools/map_notes.md`.""",
    verify=(
        "python tools/gen_map.py --check",
        "python tools/lint_guardrails.py",
        "python -m pytest tests/docs -q",
    ),
)

PERF = _p(
    "perf",
    "Land a measured performance win with no correctness or speed regression",
    """Improve Batcher's performance on a shape where it currently loses, and prove it.

Read `.claude/rules/performance.md` first. Procedure — do not skip the measurement:
1. Measure the current state:  python benchmarks/run.py --benchmark tpch
   The harness verifies correctness against DuckDB before it will time anything.
2. Pick ONE shape with a bad ratio. Form a hypothesis about the dominating code path.
3. Make the smallest change that tests the hypothesis.
4. Re-measure the same command. If it did not improve, revert it and stop — a change
   that does not measurably help is not a win, and keeping it costs complexity for nothing.

Your work is gated on a *measured* comparison: the TPC-H timings were snapshotted before
you started, and the gate re-runs them and fails if any Batcher timing regressed by more
than 10%, or if any query stopped agreeing with DuckDB. You cannot argue past that, so do
not bother claiming a speedup you did not measure.

Constraints:
- Correctness before speed, always. A fast wrong answer is a bug.
- A stateful operator must keep its mergeable partial/combine/finalize form, or it
  silently caps at single-node.
- The JIT must stay bit-for-bit identical to the interpreter or fall back — never diverge.
- Report the before and after numbers in your final message. Do not report a speedup you
  did not measure.""",
    verify=(
        "cargo check --workspace --exclude bc-py",
        "cargo clippy --workspace --exclude bc-py -- -D warnings",
        "cargo test --workspace --exclude bc-py",
        "ruff check python tests benchmarks examples",
        "python tools/lint_structure.py",
        "just build",
        "python -m pytest tests/unit -q",
        "python -m pytest tests/differential -q -x",
        # The measured gate: re-runs TPC-H and fails on a >10% Batcher slowdown or any
        # query that stopped matching DuckDB. This is what makes "it's faster" checkable.
        'python tools/agentic/bench.py --check "$AGENTIC_BASELINE/tpch.json" --benchmark tpch',
    ),
    setup=(
        # Captured before the agent touches anything; `just build` first so the baseline
        # measures the engine as it is now, not a stale artifact.
        "just build",
        'python tools/agentic/bench.py --save "$AGENTIC_BASELINE/tpch.json" --benchmark tpch',
    ),
)

TEST_GAPS = _p(
    "test-gaps",
    "Close a real coverage gap in the operator cross-product",
    """Add differential tests for an operator/execution-path combination that is not covered.

`tests/differential/test_diff_operator_matrix.py` is the cross-product of
{collect, spill, iter_batches, distributed} x {nulls, empty, one row, duplicates,
-0.0/NaN, descending}. Find a genuinely uncovered combination and cover it.

This matters because every gate passed while `sort(descending=True)` returned unsorted data
under spill, and while a distributed GROUP BY on a float key split one group in two — both
because nothing combined an operator with a non-default flag on a non-default path.

Rules:
- Compare against DuckDB via `assert_same`. It is order-INDEPENDENT, so it cannot see a
  sort bug: assert ordering explicitly for any ordered result.
- If a new test FAILS, you have found a real bug. Do not weaken the test. Leave it failing,
  and say clearly in your final message what broke and how to reproduce it.
- Add tests only. Do not change engine code in this pass.""",
    verify=(
        "ruff check python tests benchmarks examples",
        "python -m pytest tests/unit -q",
    ),
)


#: Read-only passes: safe to run any time, at any frequency.
REVIEWS = (REVIEW_QUALITY, REVIEW_ARCHITECTURE, REVIEW_BENCH)

#: Writing passes, ordered cheapest-and-safest first. `fix-gate` runs first deliberately:
#: if the gate is red, every later pass's verification is meaningless.
IMPROVEMENTS = (FIX_GATE, FIX_MAP, FIX_DOCS, TEST_GAPS, PERF)

ALL: dict[str, Pass] = {p.name: p for p in (*REVIEWS, *IMPROVEMENTS)}
