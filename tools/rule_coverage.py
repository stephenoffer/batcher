"""Which optimizer rules ever actually fire — the check unit tests structurally cannot make.

Kyber has several hundred rules, and a rule that never runs is invisible to every gate the
project has. `registry.py` says so directly: a rule can be registered, tested, and dead at the
same time, because "its tests still pass (they call the module function directly, never the
registry), and nothing reports it".

That is not a hypothetical. `topn_over_sorted_input_to_limit` was added with seven unit tests,
survived the differential suite and two full-suite sweeps, and fired **zero** times on every
path a user can reach: it matched a `Sort` carrying a `limit`, and `Sort.limit` is only ever
set by rules in a *later* phase, so within one optimize pass the shape it waited for could not
exist yet. Its tests built that shape by hand.

This tool closes that hole the only way it can be closed: run a real corpus through the real
registry and count. It wraps every registered rule, runs a pytest path (the differential suite
is the broadest corpus the project has), and reports the rules nothing ever triggered.

**A never-fired rule is a question, not a verdict.** The corpus may simply not contain its
shape. The output is a list to review, and the useful reading is comparative: a rule in a
family whose siblings all fire, or a rule added alongside tests that pass, is the one worth
opening.

Usage::

    python tools/rule_coverage.py                        # over tests/differential
    python tools/rule_coverage.py tests/unit tests/io    # any pytest paths
    python tools/rule_coverage.py --json coverage.json   # machine-readable
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "python"))


def _instrument() -> collections.Counter:
    """Wrap every registered rule so each firing is counted, and return the counter.

    A rule "fires" when it returns something other than `None` *and* other than the node it
    was handed — the two ways the driver spells "no change". Counting a call rather than a
    change would report every rule in the registry as live, since the driver calls them all.

    **All four entry points are wrapped, and missing one silently breaks the tool.** The
    driver does not call every rule the same way. A whole-plan rule goes through `fn`; a
    node-local rule through `node_fn`; but a rule whose body is a leaf expression rewrite is
    dispatched through `expr_fn` (or `expr_schema_fn`) inside the fused expression traversal,
    and its `node_fn` is *never* called -- `_apply_node_rules` explicitly filters those rules
    out of the per-node loop when it fuses. Wrapping only `node_fn` therefore reported the
    several hundred expression rules as dead, which is the exact false accusation this tool
    exists to avoid making.
    """
    from batcher.kyber.registry import DEFAULT_REGISTRY

    fired: collections.Counter = collections.Counter()

    def wrap_node(name, fn):
        def inner(node, ctx):
            out = fn(node, ctx)
            if out is not None and out is not node:
                fired[name] += 1
            return out

        return inner

    def wrap_plan(name, fn):
        def inner(plan, ctx):
            out = fn(plan, ctx)
            # Identity only, deliberately. Comparing lowered IR here would be exact, and it
            # cost more than the suite it instruments: a whole-plan rule runs once per
            # optimize, and serializing two plans per call made a 25-minute corpus run
            # unfinishable. Identity is the same fast path the driver's own fixpoint uses.
            #
            # The trade is one-sided and in the safe direction: a rule that rebuilds an equal
            # tree unconditionally is counted as fired when it changed nothing, so a
            # whole-plan rule reported *live* is weaker evidence than a node rule reported
            # live. It can never report a live rule as dead, which is what this tool is for.
            if out is not plan:
                fired[name] += 1
            return out

        return inner

    def wrap_expr(name, fn):
        def inner(expr):
            out = fn(expr)
            if out is not None and out is not expr:
                fired[name] += 1
            return out

        return inner

    def wrap_expr_schema(name, fn):
        def inner(expr, schema):
            out = fn(expr, schema)
            if out is not None and out is not expr:
                fired[name] += 1
            return out

        return inner

    wrapped = []
    for rule in DEFAULT_REGISTRY._rules:
        fired.setdefault(rule.name, 0)
        changes = {}
        if rule.node_fn is not None:
            changes["node_fn"] = wrap_node(rule.name, rule.node_fn)
        else:
            changes["fn"] = wrap_plan(rule.name, rule.fn)
        if rule.expr_fn is not None:
            changes["expr_fn"] = wrap_expr(rule.name, rule.expr_fn)
        if rule.expr_schema_fn is not None:
            changes["expr_schema_fn"] = wrap_expr_schema(rule.name, rule.expr_schema_fn)
        wrapped.append(dataclasses.replace(rule, **changes))
    DEFAULT_REGISTRY._rules = wrapped
    DEFAULT_REGISTRY._by_name = {r.name: r for r in wrapped}
    DEFAULT_REGISTRY._phase_cache = None
    return fired


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=None, help="pytest paths (or extra args)")
    parser.add_argument("--json", dest="json_out", default="")
    args = parser.parse_args()
    paths = args.paths or ["tests/differential"]

    fired = _instrument()
    import pytest

    code = pytest.main([*paths, "-q", "--no-header", "-p", "no:randomly"])

    total = len(fired)
    dead = sorted(name for name, count in fired.items() if count == 0)
    live = total - len(dead)
    where = paths[0] if len(paths) == 1 else f"{len(paths)} paths"
    print(f"\nrule coverage over {where}: {live}/{total} fired, {len(dead)} never fired")
    for name in dead:
        print(f"  never fired: {name}")
    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(dict(fired), indent=2, sort_keys=True))
        print(f"wrote {args.json_out}")
    # The exit code is the *test run's*, not a verdict on coverage: a never-fired rule is a
    # question for a human, and failing the build on one would make the tool unrunnable
    # against a narrow path.
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
