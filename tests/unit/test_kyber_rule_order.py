"""The optimizer's rules must run in the order this snapshot records.

Registration order **is** run order: `DEFAULT_REGISTRY._rules` is walked in sequence, and a
rule is registered when its module is imported. So the order is a property of the *import
graph*, which nothing else in the suite looks at — and it changes for reasons that have
nothing to do with the optimizer. Re-exporting a family from a package `__init__`, splitting
an oversized module, or sorting an import block all move rules relative to each other while
every rule still exists, every name still resolves, and every differential test still passes,
because Kyber's rewrites are individually semantics-preserving.

That is not hypothetical here: a naive package split once shifted **283 of 302 rules** and was
found by hand. `tools/surface_snapshot.py` records the order for exactly this reason, but it is
a before/after tool someone has to remember to run around a refactor — which is precisely the
moment a refactor is least likely to think of it.

A reorder is usually harmless and occasionally not: two rules that both match a node can
produce different plans depending on which fires first, and a fixpoint driver can converge
somewhere else. The point of this test is not that reordering is forbidden — it is that it
must be *deliberate*. When the diff is intended, re-record it:

    python tests/unit/test_kyber_rule_order.py --update
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

pytestmark = pytest.mark.unit

SNAPSHOT = pathlib.Path(__file__).with_name("kyber_rule_order.json")


def live_order() -> list[str]:
    """Every registered rule, in the order the optimizer will run it."""
    import batcher.kyber.rules  # noqa: F401  (importing registers every rule family)
    from batcher.kyber.registry import DEFAULT_REGISTRY

    # Position in the list carries the order, so a pure reordering shows up as moved entries
    # rather than as N changed strings.
    return [f"{rule.name}@{rule.phase}" for rule in DEFAULT_REGISTRY._rules]


def _describe(recorded: list[str], live: list[str]) -> str:
    added = [r for r in live if r not in set(recorded)]
    removed = [r for r in recorded if r not in set(live)]
    moved = [
        f"{name}: position {recorded.index(name)} -> {live.index(name)}"
        for name in recorded
        if name in set(live) and recorded.index(name) != live.index(name)
    ]
    parts = []
    if added:
        parts.append(f"{len(added)} added: {added[:5]}")
    if removed:
        parts.append(f"{len(removed)} removed: {removed[:5]}")
    if moved:
        parts.append(f"{len(moved)} moved: {moved[:5]}")
    return "; ".join(parts) or "order differs"


def test_the_rules_run_in_the_recorded_order():
    recorded = json.loads(SNAPSHOT.read_text())["rules"]
    live = live_order()

    assert live == recorded, (
        f"Kyber's rule order changed — {_describe(recorded, live)}.\n"
        "Registration order is run order, and it is decided by the import graph, so this can "
        "move without anyone touching a rule. If the change is intended, re-record it:\n"
        "    python tests/unit/test_kyber_rule_order.py --update"
    )


if __name__ == "__main__":
    if "--update" in sys.argv:
        rules = live_order()
        SNAPSHOT.write_text(json.dumps({"rules": rules}, indent=1) + "\n")
        print(f"recorded {len(rules)} rules in registration order -> {SNAPSHOT.name}")
    else:
        print(__doc__)
