#!/usr/bin/env python3
"""Fail when the layered-architecture contract's exemption list grows.

`pyproject.toml`'s `layers` contract carries an `ignore_imports` list: the upward edges that
already existed the day the contract landed. That list is what let the contract be switched on
at all — without it the gate would have been red from the start and would have been deleted
rather than paid down — but an exemption list nobody watches is how the *previous* attempt
failed. The independence contract used to carry one, it grew by a line per new module, and it
ended up silencing a real breakage in all six directions at once.

So the list is a **ratchet**: it may shrink, never grow. Adding an entry to make a new import
pass is exactly the move this exists to stop; if an edge is genuinely necessary, the layer
assignment in `.claude/rules/architecture.md` is what is wrong, and that is a design change to
argue for rather than a line to append.

Usage:
    python tools/lint_layer_debt.py            # check against the ratchet
    python tools/lint_layer_debt.py --update   # record a *reduced* list (refuses growth)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
RATCHET = ROOT / "tools" / "layer_debt.json"
CONTRACT = "the import matrix is a layered stack"


def current_exemptions() -> list[str]:
    """The `ignore_imports` entries of the layers contract, sorted."""
    data = tomllib.loads(PYPROJECT.read_text())
    for contract in data["tool"]["importlinter"]["contracts"]:
        if contract.get("name") == CONTRACT:
            return sorted(contract.get("ignore_imports", []))
    raise SystemExit(f"lint-layer-debt: no contract named {CONTRACT!r} in pyproject.toml")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="record a reduced list")
    args = parser.parse_args()

    exemptions = current_exemptions()
    if not RATCHET.exists():
        RATCHET.write_text(json.dumps({"exemptions": exemptions}, indent=2) + "\n")
        print(f"lint-layer-debt: ratchet initialized at {len(exemptions)} exemptions")
        return 0

    recorded = set(json.loads(RATCHET.read_text())["exemptions"])
    added = sorted(set(exemptions) - recorded)
    removed = sorted(recorded - set(exemptions))

    if args.update:
        if added:
            print("lint-layer-debt: refusing to record a LARGER list. New entries:")
            print("\n".join(f"  + {entry}" for entry in added))
            return 1
        RATCHET.write_text(json.dumps({"exemptions": exemptions}, indent=2) + "\n")
        print(f"lint-layer-debt: ratchet tightened to {len(exemptions)} ({len(removed)} paid off)")
        return 0

    if added:
        print("lint-layer-debt: FAIL — the layering exemption list grew\n")
        print("\n".join(f"  + {entry}" for entry in added))
        print(
            "\nAn exemption is debt from before the contract existed, not a way to pass a new\n"
            "import. Route the dependency downward instead — or if the edge is genuinely\n"
            "right, the layer assignment in `.claude/rules/architecture.md` is what needs to\n"
            "change, and that is a design decision rather than a line in a list."
        )
        return 1

    if removed:
        print(f"lint-layer-debt: {len(removed)} exemption(s) paid off — tighten the ratchet:")
        print("\n".join(f"  - {entry}" for entry in removed))
        print("\n  python tools/lint_layer_debt.py --update")
        return 1

    print(f"lint-layer-debt: OK ({len(exemptions)} exemptions, unchanged)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
