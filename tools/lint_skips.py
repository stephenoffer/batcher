#!/usr/bin/env python3
"""Fail when more of the suite becomes unreachable — the gate that counts what CI cannot run.

CI installs `.[dev]` and nothing heavier: no Ray, no torch, no GPU. Every test that needs one
is written to `pytest.importorskip` its way out, so the suite reports green having never
executed the distributed path, the device tier, or the ML backends. That is a deliberate
trade — the hardware is not available to the PR gate — but it is only survivable if the size
of the hole is *visible*. Today it is not: a run prints "N passed" and says nothing about the
several hundred tests that quietly stood down, so a whole subsystem can stop being exercised
without any signal at all.

This makes the hole a number, and the number a ratchet.

**Conftest gates cascade.** `tests/differential/conftest.py` importorskips `duckdb`, which
gates every test in that directory, not just the ones that name it. A per-file scan
undercounts by an order of magnitude, so a conftest's gate is attributed to its whole subtree.

**Static, not dynamic.** It reads the AST rather than running pytest, so the gate is cheap,
deterministic, and does not itself depend on what happens to be installed on the machine
running it. It therefore sees *module-level* guards — which is where the structural gates
live — and deliberately not a `pytest.skip()` reached halfway through a test body.

**The ratchet is on the SHARE, not the raw count** — and that is the whole reason it works.

An absolute-count budget goes stale on every commit that adds tests, which is every commit.
It did: the budget was written by `5b2508e7` recording ``batcher._native: 3223`` while that
same commit's tree actually gated **4024**. It was wrong on arrival, so `just lint-skips` has
returned 1 at every commit since, and a permanently-red gate is one everybody learns to walk
past. That is the same failure the coverage gate is warned about two recipes down in the
`justfile` — "a ratchet nobody tightens is not a ratchet" — arriving from the other side: a
floor set below reality is as useless as a ceiling set above it, because neither can change
state in response to anything.

The share is the quantity that actually means something. "22.6% of the suite cannot run here"
is the fact worth defending; "4390 tests" is that fact times however many tests exist this
week. Adding a hundred CPU tests and a hundred GPU tests leaves the hole the same size and
must not fail, while a single new gate on a directory conftest cascades to its whole subtree
and moves the share hard — which is exactly the event this file exists to catch.

Measured on this tree, ordinary growth moves a share by about half a point (`batcher._native`
went 22.1% -> 22.6% across ~1,300 new tests between `5b2508e7` and `HEAD`), whereas a cascade
is enormous: `tests/differential` alone is over half the suite. `SHARE_TOLERANCE` therefore
sits at one point — loose enough that proportional growth never fails, tight enough that no
real cascade fits under it.

Raising a budget entry is a normal part of adding a test that needs hardware; what must not
happen silently is the *share* going up because something *stopped* being reachable.

Usage:
    python tools/lint_skips.py            # check against the budget
    python tools/lint_skips.py --update   # rewrite the budget from the current tree
    python tools/lint_skips.py --report   # print the full table, exit 0
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import pathlib
import re
import sys

#: How far a dependency's share of the suite may drift above its recorded value before the
#: gate fails, in share points (0.01 == one percentage point). See the module docstring for
#: why this is a share and not a count, and for the measurement behind the value.
SHARE_TOLERANCE = 0.01

ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
DOCS = ROOT / "docs"

#: A fenced ``python`` block under `docs/` opting out of execution. `tests/docs/
#: test_doc_examples.py` runs every other one, so this marker is the only way a documented
#: example stops being checked — and it is exactly as invisible as an `importorskip` was
#: before this file counted them. The policy is sound (a block needing Kafka, a cloud
#: bucket, a GPU or a real model cannot run in CI); what needs watching is the *share*.
DOCS_SKIP = "# docs: skip"
BUDGET_PATH = ROOT / "tools" / "skip_budget.json"

#: Dependencies whose absence is expected and uninteresting to track individually — they are
#: real third-party backends a contributor may simply not have installed, and each one gating
#: its own format's tests is the design working. The structural gates (the engine itself, the
#: cluster, the device) are what this file exists to watch, so everything else is pooled into
#: `other` and only the total is ratcheted.
TRACKED: frozenset[str] = frozenset(
    {
        "batcher",  # the package itself failed to import — always a defect, never a config
        "batcher._native",  # the compiled engine: nothing below the FFI ran
        "ray",  # the whole distributed path
        "cudf",  # the device tier
        "torch",  # GPU inference and the ML execution path
        "duckdb",  # the differential oracle — without it correctness is unchecked
        "polars",  # the second oracle
    }
)


def _module_gates(tree: ast.Module) -> list[str]:
    """Dependencies this module refuses to run without.

    Only direct children of the module body count: a guard inside a function or a class runs
    per-call and gates that call, not the file.
    """
    gates = []
    for node in tree.body:
        # A bare `pytest.importorskip(...)` and a `mod = pytest.importorskip(...)` are the two
        # spellings; both gate the module, and only their statement wrapper differs.
        if not isinstance(node, ast.Expr | ast.Assign):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        owner = getattr(getattr(func, "value", None), "id", "")
        if f"{owner}.{getattr(func, 'attr', '')}" != "pytest.importorskip":
            continue
        if call.args and isinstance(call.args[0], ast.Constant):
            gates.append(str(call.args[0].value))
    return gates


def _count_tests(tree: ast.Module) -> int:
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name.startswith("test_")
    )


def _unconditional_skips(tree: ast.Module) -> int:
    """Tests marked `@pytest.mark.skip` outright — dead code wearing a test's name."""
    dead = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for deco in node.decorator_list:
            target = deco.func if isinstance(deco, ast.Call) else deco
            if getattr(target, "attr", None) == "skip":
                dead += 1
    return dead


def survey() -> tuple[dict[str, int], int, int]:
    """Tests gated per dependency, the total test count, and the unconditionally-skipped count.

    Returns:
        `(gated, total_tests, dead)` where `gated` maps a tracked dependency (or `other`) to
        the number of test functions that cannot run without it.
    """
    # A conftest's gates apply to every test at or below its directory.
    inherited: dict[pathlib.Path, list[str]] = {}
    for conftest in sorted(TESTS.rglob("conftest.py")):
        try:
            inherited[conftest.parent] = _module_gates(ast.parse(conftest.read_text()))
        except SyntaxError:
            inherited[conftest.parent] = []

    gated: collections.Counter[str] = collections.Counter()
    total = dead = 0
    for path in sorted(TESTS.rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        n_tests = _count_tests(tree)
        total += n_tests
        dead += _unconditional_skips(tree)

        gates = set(_module_gates(tree))
        for directory, deps in inherited.items():
            if directory == path.parent or directory in path.parents:
                gates.update(deps)
        # A test gated on several dependencies is unreachable if *any* is missing, so it is
        # counted against each — these columns overlap by design and must not be summed.
        for dep in gates:
            gated[dep if dep in TRACKED else "other"] += n_tests
    return dict(gated), total, dead


def _budget(gated: dict[str, int], total: int, dead: int) -> dict:
    """The budget document: each dependency's share, with the count that produced it.

    The share is what the gate compares; the count and the total are recorded beside it so a
    reviewer can see what moved without re-running the tool, and so the diff of this file
    reads as a fact about the suite rather than an opaque number.
    """
    doc_skipped, doc_total = doc_blocks()
    return {
        "gated": {
            dep: {"share": round(count / max(1, total), 4), "tests": count}
            for dep, count in sorted(gated.items())
        },
        "total_tests": total,
        "unconditionally_skipped": dead,
        "doc_blocks": {
            "share": round(doc_skipped / max(1, doc_total), 4),
            "skipped": doc_skipped,
            "total": doc_total,
        },
    }


def doc_blocks() -> tuple[int, int]:
    """``(skipped, total)`` fenced python blocks under `docs/`.

    Counted here rather than in a test of its own because it is the same measurement as the
    rest of this file — documentation CI executes, versus documentation it does not — and
    splitting the two would let a reader see half the hole.
    """
    block = re.compile(r"```python\n(.*?)```", re.S)
    skipped = total = 0
    for path in sorted(DOCS.rglob("*.md")):
        if "_build" in path.parts:
            continue
        for match in block.finditer(path.read_text(encoding="utf-8")):
            total += 1
            skipped += match.group(1).lstrip().startswith(DOCS_SKIP)
    return skipped, total


def _render(gated: dict[str, int], total: int, dead: int) -> str:
    width = max((len(k) for k in gated), default=10)
    lines = [f"{'dependency':{width}}  {'tests gated':>11}  {'share':>6}"]
    for dep, count in sorted(gated.items(), key=lambda kv: -kv[1]):
        lines.append(f"{dep:{width}}  {count:>11}  {count / max(1, total):>5.1%}")
    lines.append(f"\n{total} test functions total; {dead} unconditionally skipped")
    doc_skipped, doc_total = doc_blocks()
    lines.append(
        f"{doc_total} fenced python blocks under docs/; {doc_skipped} "
        f"({doc_skipped / max(1, doc_total):.1%}) marked `# docs: skip`"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="rewrite the budget file")
    parser.add_argument("--report", action="store_true", help="print the table and exit 0")
    args = parser.parse_args()

    gated, total, dead = survey()
    if args.report:
        print(_render(gated, total, dead))
        return 0

    current = _budget(gated, total, dead)
    if args.update:
        BUDGET_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print(f"lint-skips: budget written to {BUDGET_PATH.relative_to(ROOT)}")
        print(_render(gated, total, dead))
        return 0

    if not BUDGET_PATH.exists():
        print(f"lint-skips: no budget at {BUDGET_PATH.relative_to(ROOT)}; run --update")
        return 1

    budget = json.loads(BUDGET_PATH.read_text())
    recorded = budget.get("gated", {})
    failures = []
    for dep, count in sorted(gated.items()):
        entry = recorded.get(dep)
        share = count / max(1, total)
        if entry is None:
            failures.append(
                f"  {dep}: {count} tests gated ({share:.1%}), not in the budget (new gate)"
            )
            continue
        allowed = entry["share"]
        if share > allowed + SHARE_TOLERANCE:
            failures.append(
                f"  {dep}: {share:.1%} of the suite gated ({count} of {total}), "
                f"budget {allowed:.1%} +{SHARE_TOLERANCE:.0%} tolerance "
                f"(+{(share - allowed) * 100:.1f} points)"
            )
    recorded_docs = budget.get("doc_blocks")
    if recorded_docs is not None:
        doc_skipped, doc_total = doc_blocks()
        doc_share = doc_skipped / max(1, doc_total)
        if doc_share > recorded_docs["share"] + SHARE_TOLERANCE:
            failures.append(
                f"  docs `# docs: skip`: {doc_share:.1%} of fenced python blocks "
                f"({doc_skipped} of {doc_total}), budget {recorded_docs['share']:.1%} "
                f"+{SHARE_TOLERANCE:.0%} tolerance"
            )

    allowed_dead = budget.get("unconditionally_skipped", 0)
    if dead > allowed_dead:
        failures.append(f"  @pytest.mark.skip: {dead}, budget {allowed_dead}")

    if failures:
        print("lint-skips: FAIL — a larger share of the suite became unreachable\n")
        print("\n".join(failures))
        print(
            "\nCI runs on CPU only, so a gated test is a test nobody runs. A share that grew\n"
            "by more than the tolerance is usually a gate that cascaded: check whether a\n"
            "conftest gained an `importorskip` that now applies to its whole subtree. If the\n"
            "increase is intended (new tests that genuinely need hardware), re-record it:\n"
            "  python tools/lint_skips.py --update"
        )
        return 1

    print(_render(gated, total, dead))
    print("\nlint-skips: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
