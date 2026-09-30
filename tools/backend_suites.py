#!/usr/bin/env python3
"""Select the test files a backend lane runs, and fail when one silently drops out of it.

CI's `gate` job installs no Ray, torch or TensorFlow, so every test that needs one skips
there. The `distributed` job in `ci.yml` and the `ml-frameworks` workflow are the only places
those tests execute, and each runs exactly the files this tool names for its backend. The
Ray selection used to be ``grep -rl 'importorskip("ray"' tests/``, which had two holes:

**It matched a spelling, not a dependency.** A single-quoted ``importorskip('ray')``, a
``pytest.importorskip("ray.serve")``, a Ray gate in a directory ``conftest.py``, a test that
reaches Ray through a shared helper such as ``tests/_ray_cluster.py``, or one that imports
``ray`` inside a test body under a ``skipif`` never matched. Nor did a test that names no
Ray module at all and reaches the cluster through ``collect(distributed=True)``: without Ray
the root ``conftest.py`` turns that failure into a skip. Each such file skipped in the gate
job and was never selected here, so it ran nowhere. This reads the AST instead, so quoting
and helper indirection do not matter.

**Its floor could not see a partial loss.** "At least 40 files" still passes when five of
ninety stop matching. So each selection is also held against a committed inventory,
`tools/backend_suites.json`: a file on the inventory that still exists but is no longer
detected fails the lane, naming the file. That is the event the floor could not see — a suite
whose gate was rewritten into a shape the detector misses. A file that was deleted or
renamed, or a newly written suite, is a normal change and is re-recorded with ``--update``,
so the diff of the inventory shows every addition and removal to a reviewer.

Static on purpose, like `tools/lint_skips.py`: the answer must not depend on what happens to
be installed on the machine asking, which is exactly the property a backend lane and the gate
job disagree on.

Usage:
    python tools/backend_suites.py                    # check every backend's inventory
    python tools/backend_suites.py --dep ray --list   # print one backend's files
    python tools/backend_suites.py --dep ray --shard 1/4   # one round-robin shard
    python tools/backend_suites.py --update           # re-record every inventory
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
INVENTORY = ROOT / "tools" / "backend_suites.json"

#: The backends a lane exists for, by top-level module name.
DEPS = ("ray", "tensorflow", "torch")


def _is_dep(name: str, dep: str) -> bool:
    """Whether a dotted module name is `dep` or one of its submodules (``ray.serve``)."""
    return name == dep or name.startswith(dep + ".")


def _call_name(call: ast.Call) -> str:
    """The trailing attribute or bare name a call targets (``importorskip``, ``find_spec``)."""
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _first_str_arg(call: ast.Call) -> str | None:
    if call.args and isinstance(call.args[0], ast.Constant):
        value = call.args[0].value
        return value if isinstance(value, str) else None
    return None


def _module_gated(tree: ast.Module, dep: str) -> bool:
    """A module-level ``importorskip`` of `dep`: it gates every test in the file (or subtree)."""
    for node in tree.body:
        value = node.value if isinstance(node, ast.Expr | ast.Assign) else None
        if isinstance(value, ast.Call) and _call_name(value) == "importorskip":
            name = _first_str_arg(value)
            if name is not None and _is_dep(name, dep):
                return True
    return False


def _uses(tree: ast.Module, dep: str) -> bool:
    """Any reach for `dep` anywhere in the file, including inside a test body or a helper."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(_is_dep(a.name, dep) for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.level == 0 and _is_dep(node.module or "", dep):
            return True
        if isinstance(node, ast.Call) and _call_name(node) in {"importorskip", "find_spec"}:
            name = _first_str_arg(node)
            if name is not None and _is_dep(name, dep):
                return True
    return False


def _requests_cluster(tree: ast.Module) -> bool:
    """A call passing ``distributed=True``: it reaches Ray at run time, gate or no gate.

    Such a test names no Ray module, so neither a grep nor an import scan sees it. Without
    Ray the root `conftest.py` turns the resulting "backend not installed" error into a
    skip, which is right for the gate job and means the test runs only if this lane picks
    it up.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if (
                    kw.arg == "distributed"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                ):
                    return True
    return False


def _imported_locals(tree: ast.Module) -> set[str]:
    """Top-level names this file imports, so a helper under `tests/` can be followed."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _parse(path: pathlib.Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return None


def discover(tests: pathlib.Path = TESTS, dep: str = "ray") -> list[str]:
    """Every ``test_*.py`` under `tests` that needs `dep`, as paths relative to its parent.

    A file is selected when it gates on `dep` at module level, sits under a ``conftest.py``
    that does, reaches `dep` anywhere in its body, or imports a helper module in `tests/`
    that reaches it. For Ray, a call passing ``distributed=True`` selects the file too.

    Args:
        tests: The test root to scan.
        dep: The backend's top-level module name.

    Returns:
        The selected files, sorted, relative to ``tests.parent`` (``tests/...``).
    """
    base = tests.parent
    gated_dirs = []
    for conftest in sorted(tests.rglob("conftest.py")):
        tree = _parse(conftest)
        if tree is not None and _module_gated(tree, dep):
            gated_dirs.append(conftest.parent)
    helpers = set()
    for helper in tests.glob("_*.py"):
        tree = _parse(helper)
        if tree is not None and _uses(tree, dep):
            helpers.add(helper.stem)

    selected = []
    for path in sorted(tests.rglob("test_*.py")):
        tree = _parse(path)
        if tree is None:
            continue
        cascaded = any(d == path.parent or d in path.parents for d in gated_dirs)
        cluster = dep == "ray" and _requests_cluster(tree)
        needs = _uses(tree, dep) or cluster or _imported_locals(tree) & helpers
        if cascaded or needs:
            selected.append(path.relative_to(base).as_posix())
    return selected


def shard(files: list[str], index: int, count: int) -> list[str]:
    """One round-robin shard of `files` (1-based `index`).

    Round-robin rather than contiguous blocks: the files are sorted by name, so contiguous
    blocks would put every ``test_distributed_*`` file in one shard.

    Args:
        files: The full, sorted selection.
        index: Which shard, from 1 to `count`.
        count: How many shards there are.

    Returns:
        The files in that shard, in selection order.
    """
    if not 1 <= index <= count:
        raise SystemExit(f"backend_suites: shard {index}/{count} is out of range")
    return [f for i, f in enumerate(files, start=1) if i % count == index % count]


def check(
    selected: list[str], inventory: list[str], base: pathlib.Path, dep: str = "ray"
) -> list[str]:
    """The inventory entries the selection lost, each as a one-line problem.

    Args:
        selected: What `discover` found.
        inventory: What `tools/backend_suites.json` records for `dep`.
        base: The repository root the paths are relative to.
        dep: The backend, for the messages.

    Returns:
        One message per problem; empty when the selection covers the inventory.
    """
    found = set(selected)
    problems = []
    for entry in inventory:
        if entry in found:
            continue
        if (base / entry).exists():
            problems.append(
                f"  {entry}: still exists but is no longer detected as a {dep} suite, so "
                "its lane would stop running it"
            )
        else:
            problems.append(f"  {entry}: on the inventory but deleted or renamed")
    problems.extend(
        f"  {entry}: a {dep} suite not on the inventory (new)"
        for entry in sorted(found - set(inventory))
    )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dep", choices=DEPS, help="one backend (default: every backend)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true", help="print the selection")
    group.add_argument("--shard", metavar="I/N", help="print one round-robin shard")
    group.add_argument("--update", action="store_true", help="re-record the inventory")
    args = parser.parse_args(argv)

    deps = (args.dep,) if args.dep else DEPS
    if (args.list or args.shard) and not args.dep:
        parser.error("--list and --shard need --dep")
    selections = {dep: discover(dep=dep) for dep in deps}
    for dep, selected in selections.items():
        if not selected:
            print(f"backend_suites: no {dep} suites found; the detector is broken", file=sys.stderr)
            return 1
    if args.list:
        print("\n".join(selections[args.dep]))
        return 0
    if args.shard:
        index, _, count = args.shard.partition("/")
        files = shard(selections[args.dep], int(index), int(count))
        if not files:
            print(f"backend_suites: shard {args.shard} selected no files", file=sys.stderr)
            return 1
        print(" ".join(files))
        return 0

    recorded = json.loads(INVENTORY.read_text(encoding="utf-8")) if INVENTORY.exists() else {}
    if args.update:
        recorded.update(selections)
        INVENTORY.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n", "utf-8")
        counts = ", ".join(f"{len(v)} {k}" for k, v in selections.items())
        print(f"backend_suites: recorded {counts} suites in {INVENTORY.name}")
        return 0

    problems = [
        line
        for dep, selected in selections.items()
        for line in check(selected, recorded.get(dep, []), ROOT, dep)
    ]
    if problems:
        print("backend_suites: FAIL — a backend lane's selection no longer matches its inventory\n")
        print("\n".join(problems))
        print(
            "\nA suite that still exists but is not detected has a gate this tool cannot read;\n"
            "fix the gate or the detector. Additions, deletions and renames are normal:\n"
            "  python tools/backend_suites.py --update"
        )
        return 1
    counts = ", ".join(f"{len(v)} {k}" for k, v in selections.items())
    print(f"backend_suites: OK — {counts} suites, all on the inventory")
    return 0


if __name__ == "__main__":
    sys.exit(main())
