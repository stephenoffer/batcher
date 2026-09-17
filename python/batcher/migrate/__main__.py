"""`python -m batcher.migrate`: rewrite scripts onto Batcher's API, or off it.

Three kinds of direction read the same migration registry:

* `--from batcher --to batcher` renames Batcher's own removed second spellings (`ds.groupby` to
  `ds.group_by`), in `.py` files and the Python in `.md` files;
* `--from pyspark|polars|daft|ray_data --to batcher` translates a foreign script onto Batcher
  (`batcher.migrate.translate`);
* `--from batcher --to pyspark|polars|daft|ray_data` translates a Batcher script onto a foreign
  engine where the registry is exact (`batcher.migrate.outbound`).

A foreign direction rewrites `.py` files only. Whatever it cannot translate exactly it leaves as
written with a `# batcher-migrate:` comment saying why, and prints.

Without `--write` the command prints a unified diff and changes nothing. `--check` exits 1 when
any file would change, for CI. `--report` writes every site as JSON, with per-file counts of
what was rewritten and what was left marked.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections.abc import Callable
from pathlib import Path

from batcher._internal.errors import ConfigError
from batcher._internal.migration import ENGINES, load_kwarg_renames, load_renames, load_returns
from batcher.migrate.project import imported_function_receivers
from batcher.migrate.snippets import rewrite_markdown, rewrite_python

_IMPLEMENTED = {
    ("batcher", "batcher"),
    *((engine, "batcher") for engine in ENGINES),
    *(("batcher", engine) for engine in ENGINES),
}
_CHOICES = ("batcher", *ENGINES)

# One file's rewrite: the new text, its JSON report entry, and the lines to print to stderr.
Rewrite = Callable[[Path, str], tuple[str, dict[str, object], list[str]]]


def _source_files(paths: list[str], suffixes: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            out.extend(sorted(p for s in suffixes for p in path.rglob(f"*{s}")))
        else:
            out.append(path)
    return out


def _canonical(assume_accessors: bool) -> Rewrite:
    renames, returns, kwargs = load_renames(), load_returns(), load_kwarg_renames()

    def run(path: Path, before: str) -> tuple[str, dict[str, object], list[str]]:
        rewrite = rewrite_markdown if path.suffix == ".md" else rewrite_python
        imported = (
            imported_function_receivers(before, path, returns) if path.suffix == ".py" else {}
        )
        after, sites = rewrite(
            before, renames, returns, kwargs, assume_accessors=assume_accessors, imported=imported
        )
        entry: dict[str, object] = {}
        if sites.renamed or sites.unresolved:
            entry = {
                "renamed": [vars(e) for e in sites.renamed],
                "unresolved": [vars(e) for e in sites.unresolved],
            }
        lines = [f"{path}:{e.line}: left `{e.old}` alone: {e.new}" for e in sites.unresolved]
        return after, entry, lines

    return run


def _foreign(source: str, target: str) -> Rewrite:
    if source == "batcher":
        from batcher.migrate.outbound import export

        def convert(text: str) -> tuple[str, object]:
            return export(text, target)
    else:
        from batcher.migrate.translate import translate

        def convert(text: str) -> tuple[str, object]:
            return translate(text, source)

    def run(path: Path, before: str) -> tuple[str, dict[str, object], list[str]]:
        after, report = convert(before)
        sites = report.sites  # type: ignore[attr-defined]
        entry: dict[str, object] = {}
        if sites:
            entry = {"counts": report.counts(), "sites": [vars(s) for s in sites]}  # type: ignore[attr-defined]
        lines = [
            f"{path}:{s.line}: left `{s.spelling}` as written: {s.detail}"
            for s in sites
            if s.action == "marked"
        ]
        return after, entry, lines

    return run


def main(argv: list[str] | None = None) -> int:
    """Run the codemod over files and directories.

    Args:
        argv: Command-line arguments; `sys.argv[1:]` when omitted.

    Returns:
        The process exit code.

    Raises:
        ConfigError: When the requested direction is not one of the implemented pairs.
    """
    parser = argparse.ArgumentParser(prog="python -m batcher.migrate", description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--from", dest="source", choices=_CHOICES, default="batcher")
    parser.add_argument("--to", dest="target", choices=_CHOICES, default="batcher")
    parser.add_argument("--write", action="store_true", help="rewrite files in place")
    parser.add_argument("--check", action="store_true", help="exit 1 if any file would change")
    parser.add_argument("--report", type=Path, help="write a JSON report of every site")
    parser.add_argument(
        "--assume-accessors",
        action="store_true",
        help="treat x.str/x.dt/x.list on an unknown x as Batcher (code with no pandas/Polars)",
    )
    args = parser.parse_args(argv)
    if (args.source, args.target) not in _IMPLEMENTED:
        raise ConfigError(
            f"--from {args.source} --to {args.target} is not a supported direction; one side "
            "must be batcher (migrate between two other engines through Batcher)"
        )
    canonical = args.source == args.target == "batcher"
    rewrite = _canonical(args.assume_accessors) if canonical else _foreign(args.source, args.target)
    suffixes = (".py", ".md") if canonical else (".py",)
    changed = 0
    report: dict[str, dict[str, object]] = {}
    for path in _source_files(args.paths, suffixes):
        before = path.read_text()
        try:
            after, entry, lines = rewrite(path, before)
        except Exception as exc:  # an unparseable file is reported, not fatal to the run
            print(f"{path}: skipped, {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if entry:
            report[str(path)] = entry
        for line in lines:
            print(line, file=sys.stderr)
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
