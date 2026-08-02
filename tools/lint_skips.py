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

Raising a budget entry is a normal part of adding a test that needs hardware; what must not
happen silently is the count going up because something *stopped* being reachable.

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
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
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


def _render(gated: dict[str, int], total: int, dead: int) -> str:
    width = max((len(k) for k in gated), default=10)
    lines = [f"{'dependency':{width}}  {'tests gated':>11}  {'share':>6}"]
    for dep, count in sorted(gated.items(), key=lambda kv: -kv[1]):
        lines.append(f"{dep:{width}}  {count:>11}  {count / max(1, total):>5.1%}")
    lines.append(f"\n{total} test functions total; {dead} unconditionally skipped")
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

    current = {"gated": gated, "unconditionally_skipped": dead}
    if args.update:
        BUDGET_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print(f"lint-skips: budget written to {BUDGET_PATH.relative_to(ROOT)}")
        print(_render(gated, total, dead))
        return 0

    if not BUDGET_PATH.exists():
        print(f"lint-skips: no budget at {BUDGET_PATH.relative_to(ROOT)}; run --update")
        return 1

    budget = json.loads(BUDGET_PATH.read_text())
    failures = []
    for dep, count in sorted(gated.items()):
        allowed = budget.get("gated", {}).get(dep)
        if allowed is None:
            failures.append(f"  {dep}: {count} tests gated, not in the budget (new gate)")
        elif count > allowed:
            failures.append(f"  {dep}: {count} tests gated, budget {allowed} (+{count - allowed})")
    allowed_dead = budget.get("unconditionally_skipped", 0)
    if dead > allowed_dead:
        failures.append(f"  @pytest.mark.skip: {dead}, budget {allowed_dead}")

    if failures:
        print("lint-skips: FAIL — more of the suite became unreachable\n")
        print("\n".join(failures))
        print(
            "\nCI runs on CPU only, so a gated test is a test nobody runs. If the increase is\n"
            "intended (a new test that genuinely needs hardware), raise the budget:\n"
            "  python tools/lint_skips.py --update"
        )
        return 1

    print(_render(gated, total, dead))
    print("\nlint-skips: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
