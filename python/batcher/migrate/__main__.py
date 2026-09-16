"""`python -m batcher.migrate`: rewrite scripts onto Batcher's one spelling per capability.

Today one direction is implemented, `--from batcher --to batcher`: the rename of Batcher's own
removed second spellings (`ds.groupby` to `ds.group_by`). The foreign-engine directions read
the same migration registry and land in later waves; asking for one says so rather than
running a partial rule set.

Without `--write` the command prints a unified diff and changes nothing. `--check` exits 1 when
any file would change, for CI. Every run prints the sites it declined because it could not tell
whether the expression was a Batcher object, so the residue is visible.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

from batcher._internal.errors import ConfigError
from batcher._internal.migration import load_renames, load_returns
from batcher.migrate.canonical import canonicalize

_IMPLEMENTED = {("batcher", "batcher")}
_ENGINES = ("batcher", "pyspark", "polars", "daft", "ray_data")


def _python_files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        path = Path(raw)
        out.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the codemod over files and directories.

    Args:
        argv: Command-line arguments; `sys.argv[1:]` when omitted.

    Returns:
        The process exit code.

    Raises:
        ConfigError: When the requested direction is not implemented.
    """
    parser = argparse.ArgumentParser(prog="python -m batcher.migrate", description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--from", dest="source", choices=_ENGINES, default="batcher")
    parser.add_argument("--to", dest="target", choices=_ENGINES, default="batcher")
    parser.add_argument("--write", action="store_true", help="rewrite files in place")
    parser.add_argument("--check", action="store_true", help="exit 1 if any file would change")
    parser.add_argument("--report", type=Path, help="write a JSON report of every site")
    args = parser.parse_args(argv)
    if (args.source, args.target) not in _IMPLEMENTED:
        raise ConfigError(
            f"--from {args.source} --to {args.target} is not implemented yet; "
            "only --from batcher --to batcher (the canonical-name rewrite) is"
        )
    renames, returns = load_renames(), load_returns()
    changed = 0
    report: dict[str, dict[str, list[dict[str, object]]]] = {}
    for path in _python_files(args.paths):
        before = path.read_text()
        try:
            after, sites = canonicalize(before, renames, returns)
        except Exception as exc:  # an unparseable file is reported, not fatal to the run
            print(f"{path}: skipped, {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if sites.renamed or sites.unresolved:
            report[str(path)] = {
                "renamed": [vars(e) for e in sites.renamed],
                "unresolved": [vars(e) for e in sites.unresolved],
            }
        for edit in sites.unresolved:
            print(f"{path}:{edit.line}: left `{edit.old}` alone: receiver unknown", file=sys.stderr)
        if after == before:
            continue
        changed += 1
        if args.write:
            path.write_text(after)
        else:
            sys.stdout.writelines(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    after.splitlines(keepends=True),
                    fromfile=str(path),
                    tofile=str(path),
                )
            )
    if args.report:
        args.report.write_text(json.dumps(report, indent=1))
    print(f"{changed} file(s) {'rewritten' if args.write else 'would change'}", file=sys.stderr)
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
