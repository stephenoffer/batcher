#!/usr/bin/env python3
"""Capture Batcher's observable surfaces so a refactor can *prove* it changed nothing.

A move-and-re-export refactor is the most common shape of agent-written change here —
package-ize an oversized module, group a rule family, split an FFI hub — and it is the
shape whose failure mode is silent. The code still imports, the tests still pass, and
something that is not a test has quietly moved: a rule now runs in a different position,
a format stopped registering, an expression left the public API. "I only moved code" is
an assertion; this makes it a diff.

Use it around any refactor that claims to preserve behavior::

    python tools/surface_snapshot.py --save /tmp/before.json
    ...do the refactor...
    python tools/surface_snapshot.py --diff /tmp/before.json    # exit 1 on any change

The five surfaces are the ones where a silent change is both plausible and costly:

* **kyber_rules** — recorded *in order*, because registration order is run order. A
  naive package split here once shifted 283 of 302 rules while every rule still existed
  and every name still resolved; a set comparison would have called that identical.
* **ir_tags** — the JSON wire contract with Rust. Python's tags and `bc_ir::RelOp`'s
  serde tags must stay in lockstep, and a drift is a silent correctness bug.
* **public_api** / **expressions** — what users import. Losing a name is a breaking
  change; gaining one unintentionally is an unearned commitment.
* **io_formats** — connectors register as an import side effect, so a module that stops
  being imported stops existing, with no error anywhere.
* **native_ffi** — the PyO3 boundary, captured with signatures rather than names alone,
  so a changed default (`credits=32`) is caught and not just a renamed function. Skipped
  with a note when the engine is not built, so this tool stays usable without a compile.

A changed surface is not automatically wrong — adding a rule *should* change it. The
point is that the change becomes visible and deliberate rather than discovered later.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))


def _matches_repr(matches: frozenset[type] | None) -> str:
    """The rule's node-type filter, name-sorted so the rendering is deterministic.

    `matches` is what the driver uses to skip a rule whose node types are absent, so a
    change to it changes which plans the rule sees — a behavior change worth catching.
    """
    if matches is None:
        return "*"
    return ",".join(sorted(t.__name__ for t in matches))


def _kyber_rules() -> list[str]:
    """The optimizer's rules in registration order — which is the order they run in."""
    import batcher.kyber.rules  # noqa: F401  (import registers every rule family)
    from batcher.kyber.registry import DEFAULT_REGISTRY

    # No index in the value: position in the list carries the order, so a pure
    # reordering reports as an order change rather than 302 changed strings.
    return [
        f"{r.name} phase={r.phase} category={r.category} matches={_matches_repr(r.matches)}"
        for r in DEFAULT_REGISTRY._rules
    ]


def _io_formats() -> dict[str, list[str]]:
    """Registered source and sink format names (registration is an import side effect)."""
    from batcher.io.formats.base import SINKS, SOURCES

    return {"sources": SOURCES.names(), "sinks": SINKS.names()}


def _ir_tags() -> dict[str, list[str]]:
    """The IR tag vocabulary — the Python half of the JSON wire contract with Rust."""
    from batcher.plan import ir_tags

    out: dict[str, list[str]] = {}
    for group in ("Op", "ExprTag"):
        enum = getattr(ir_tags, group, None)
        if enum is None:
            continue
        out[group] = sorted(f"{m}={getattr(enum, m)}" for m in dir(enum) if not m.startswith("_"))
    return out


def _public_api() -> dict[str, list[str]]:
    """The user-visible surface, from the same definition the docs gates use."""
    import public_surface as ps

    return {
        "names": sorted(ps.public_names()),
        "expressions": sorted(ps.expression_names()),
    }


def _native_ffi() -> dict[str, Any]:
    """The `_native` PyO3 entry points with signatures, or a note if the engine is unbuilt."""
    # `engine` is the accessor *function*, not the module — calling it is what raises
    # when the extension is unbuilt, so both the import and the call are guarded.
    try:
        from batcher._internal.native import engine

        module = engine()
    except Exception as exc:  # pragma: no cover - depends on build state
        return {"unavailable": f"{type(exc).__name__}: {exc}"}

    surface: dict[str, str] = {}
    for name in sorted(dir(module)):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        try:
            surface[name] = str(inspect.signature(obj))
        except (TypeError, ValueError):
            surface[name] = type(obj).__name__
    return surface


#: Each surface is captured independently so one unavailable surface (an unbuilt engine,
#: a module mid-refactor) degrades to a recorded error instead of losing the whole snapshot.
SURFACES = {
    "kyber_rules": _kyber_rules,
    "ir_tags": _ir_tags,
    "public_api": _public_api,
    "io_formats": _io_formats,
    "native_ffi": _native_ffi,
}


def capture() -> dict[str, Any]:
    """Capture every surface, recording per-surface failures rather than aborting."""
    snapshot: dict[str, Any] = {}
    for name, fn in SURFACES.items():
        try:
            snapshot[name] = fn()
        except Exception as exc:
            snapshot[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return snapshot


def _count(value: Any) -> int:
    """Count the leaf entries in a captured surface, for the save-time summary line."""
    if isinstance(value, dict):
        return sum(_count(v) for v in value.values())
    if isinstance(value, list):
        return len(value)
    return 1


#: Surfaces whose *order* is semantically load-bearing, not just presentation. For these a
#: reordering is reported even when membership is identical — the optimizer's rule list runs
#: in registration order, so "same rules, different sequence" is a behavior change.
ORDERED_SURFACES = {"kyber_rules"}


def _list_diff(path: str, before: list[Any], after: list[Any]) -> list[str]:
    """Diff a list as membership first, then order — so one removal is one line.

    Reporting a list positionally makes a single removal cascade into a "changed" line for
    every subsequent element, which buries the actual finding. Membership is compared as a
    multiset; order is only compared for the surfaces where it carries meaning.
    """
    old, new = [str(x) for x in before], [str(x) for x in after]
    lines = [f"  REMOVED {path}: {item}" for item in sorted(set(old) - set(new))]
    lines += [f"  ADDED   {path}: {item}" for item in sorted(set(new) - set(old))]

    root = path.split(".")[0].split("[")[0]
    if root in ORDERED_SURFACES and not lines:
        moved = [(i, o, n) for i, (o, n) in enumerate(zip(old, new, strict=False)) if o != n]
        if moved:
            lines.append(
                f"  REORDERED {path}: same members, {len(moved)} position(s) differ "
                f"(order is run order — this is a behavior change)"
            )
            for i, o, n in moved[:5]:
                lines.append(f"    [{i}] before: {o}\n         after:  {n}")
            if len(moved) > 5:
                lines.append(f"    … and {len(moved) - 5} more")
    return lines


def diff(before: Any, after: Any, path: str = "") -> list[str]:
    """Return human-readable differences between two snapshots, empty when identical."""
    if isinstance(before, dict) and isinstance(after, dict):
        lines: list[str] = []
        for key in sorted(set(before) | set(after)):
            sub = f"{path}.{key}" if path else str(key)
            if key not in after:
                lines.append(f"  REMOVED {sub}")
            elif key not in before:
                lines.append(f"  ADDED   {sub}")
            else:
                lines.extend(diff(before[key], after[key], sub))
        return lines
    if isinstance(before, list) and isinstance(after, list):
        return _list_diff(path, before, after)
    if before != after:
        return [f"  CHANGED {path}:\n    before: {before}\n    after:  {after}"]
    return []


def _broken_surfaces(snapshot: dict[str, Any]) -> dict[str, str]:
    """The surfaces that failed to capture, mapped to their recorded error.

    `capture()` degrades a failed surface to `{"error": ...}` so one broken surface does
    not lose the whole snapshot. That is right for *saving*, but it is a trap for
    *diffing*: two snapshots both captured while a surface was broken compare equal, so
    the gate prints "no observable change" while verifying nothing at all. This is not
    hypothetical — `_kyber_rules` read a `Rule.idempotent` field that no longer exists,
    and the rule-order gate silently reported one entry instead of three hundred for as
    long as it took someone to look.
    """
    return {
        name: value["error"]
        for name, value in snapshot.items()
        if isinstance(value, dict) and "error" in value
    }


def _report_broken(broken: dict[str, str], verb: str) -> None:
    """Print the failed surfaces and why a diff against them cannot be trusted."""
    print(f"SURFACE CAPTURE FAILED — {len(broken)} surface(s) could not be {verb}:")
    for name, err in sorted(broken.items()):
        print(f"  {name}: {err}")
    print(
        "\nA snapshot missing a surface cannot prove that surface unchanged. Fix the\n"
        "capture (or the build) before trusting this gate."
    )


def main() -> int:
    """Save a snapshot, or diff the current surfaces against a saved one."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--save", metavar="PATH", help="write a snapshot to PATH")
    group.add_argument("--diff", metavar="PATH", help="compare current surfaces against PATH")
    args = parser.parse_args()

    current = capture()
    broken = _broken_surfaces(current)

    if args.save:
        Path(args.save).write_text(json.dumps(current, indent=2, sort_keys=True))
        counts = ", ".join(f"{k}={_count(v)}" for k, v in current.items())
        print(f"wrote {args.save} ({counts})")
        if broken:
            _report_broken(broken, "saved")
            return 1
        return 0

    if broken:
        _report_broken(broken, "captured")
        return 1

    baseline = json.loads(Path(args.diff).read_text())
    if _broken_surfaces(baseline):
        print(
            "SURFACE BASELINE UNUSABLE — the saved snapshot contains a failed surface, so a\n"
            "clean diff against it would prove nothing. Re-save the baseline on a good tree."
        )
        return 1
    differences = diff(baseline, current)
    if not differences:
        print("SURFACE DIFF EMPTY — no observable change.")
        return 0

    print(f"SURFACE CHANGED — {len(differences)} difference(s):")
    for line in differences:
        print(line)
    print(
        "\nIf these changes are intended, say so explicitly in your report.\n"
        "If they are not, the refactor altered behavior it claimed to preserve."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
