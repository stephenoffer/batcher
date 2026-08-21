"""Which public API callables does the test suite actually *execute*?

``tests/docs/test_api_coverage.py`` proves every public name is **documented**.
Nothing proved any of them is **run**. Those are different failures: a name can be
mentioned in five guides, rendered by autodoc, and taught in a tutorial while no test
ever calls it — which is how a public method ships broken past a green gate.

This module measures the stronger property, and it measures it from execution rather
than from text. Grepping the test corpus for a method name cannot tell
``Dataset.sum`` from ``Expr.sum`` from the builtin, and counts a name in a comment;
line coverage knows exactly which function bodies ran.

The mechanism: read a ``coverage.py`` data file, and for every public callable in
``public_surface.public_callables()`` ask whether any line of *its own code object*
was executed. The ``def`` line is deliberately excluded — it runs at import time for
every function in the package, so counting it would report 100% coverage of a suite
that imports ``batcher`` and does nothing else.

**Measure on a tree you are not editing.** Coverage data records *line numbers*, and this
resolves each callable's lines from the file on disk, so inserting a line into a module
invalidates the recorded hits for everything below it. Adding seven lines to one docstring
in ``api/dataset/frame.py`` between a run and a report moved 96 ``Dataset`` methods from
exercised to unexercised, with no test changed. The number is only meaningful when the
files have not moved since the run: re-run coverage after the last edit, or measure a
copy.

Usage::

    pytest tests --cov=batcher --cov-report=       # writes .coverage
    python tools/api_exercise_coverage.py          # report the gaps
    python tools/api_exercise_coverage.py --json out.json
"""

from __future__ import annotations

import argparse
import ast
import functools
import inspect
import json
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "tools"))


@dataclass(frozen=True)
class Callable_:
    """One public callable, located in the source tree."""

    qualname: str
    filename: str
    body_lines: frozenset[int]


@dataclass
class Report:
    """The exercised/unexercised split over the public callable surface."""

    exercised: list[str] = field(default_factory=list)
    unexercised: list[str] = field(default_factory=list)
    unlocatable: list[str] = field(default_factory=list)

    @property
    def measured(self) -> int:
        return len(self.exercised) + len(self.unexercised)

    @property
    def ratio(self) -> float:
        return len(self.exercised) / self.measured if self.measured else 1.0


def _unwrap(obj: Any) -> Any:
    """Strip decorators so we measure the function that actually holds the body."""
    seen = 0
    while hasattr(obj, "__wrapped__") and seen < 10:
        obj = obj.__wrapped__
        seen += 1
    if isinstance(obj, (staticmethod, classmethod)):
        obj = obj.__func__
    if isinstance(obj, property) and obj.fget is not None:
        obj = obj.fget
    if isinstance(obj, functools.cached_property):
        obj = obj.func
    return obj


def _code_of(obj: Any) -> Any:
    """The code object whose lines constitute this callable's body, or None."""
    obj = _unwrap(obj)
    code = getattr(obj, "__code__", None)
    if code is None:
        code = getattr(getattr(obj, "__func__", None), "__code__", None)
    return code


def _is_declaration_only(obj: Any) -> bool:
    """Whether a callable's whole body is a docstring and/or ``...``.

    A ``typing.Protocol`` method, and any other pure declaration, has nothing to execute:
    its body is an ellipsis, and the interpreter never runs a line of it however thoroughly
    the implementations are tested. Counting those as "unexercised" would put a floor under
    the gap that no test could ever close -- ``batcher.io``'s ``Source`` / ``Split`` /
    ``Sink`` protocols alone contribute more than twenty -- and a target nobody can reach is
    a target nobody aims at.

    Args:
        obj: The callable to inspect.

    Returns:
        True when the body holds no executable statement.
    """
    obj = _unwrap(obj)
    try:
        source = textwrap.dedent(inspect.getsource(obj))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError, IndentationError):
        return False
    definitions = [
        n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not definitions:
        return False
    body = definitions[0].body
    for statement in body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue  # a docstring, or a bare `...`
        if isinstance(statement, ast.Pass):
            continue
        return False
    return True


def _is_excluded_from_coverage(obj: Any) -> bool:
    """Whether the callable carries a ``# pragma: no cover``.

    coverage.py never records a line it has been told to exclude, so a function marked that
    way cannot register as exercised however hard a test tries. Two shapes of that exist on
    this surface and both are legitimate: an abstract base whose every subclass overrides it
    (``Expr.to_ir``, documented "overridden by every subclass; the base raises
    NotImplementedError"), and a branch that only runs on hardware CI does not have. Chasing
    either would mean writing a test that cannot pass.

    Args:
        obj: The callable to inspect.

    Returns:
        True when its definition is excluded from coverage measurement.
    """
    obj = _unwrap(obj)
    try:
        source = textwrap.dedent(inspect.getsource(obj))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError, IndentationError):
        return False
    definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not definitions or not definitions[0].body:
        return False
    # Only the signature counts. A pragma inside the body excludes one *branch* -- a
    # hardware path, an unreachable else -- and says nothing about the function, which the
    # rest of its lines still measure. Searching the whole source removed 96 callables that
    # tests do exercise.
    lines = source.splitlines()
    header = lines[: max(definitions[0].body[0].lineno - 1, 1)]
    return any("pragma: no cover" in line for line in header)


def locate(qualname: str, obj: Any) -> Callable_ | None:
    """Resolve a public callable to its file and the line numbers of its body.

    The first line (the ``def``) is excluded, because a module-level ``def`` executes
    on import. A single-line body (``def f(): return 1``) has no line above the first,
    so it keeps that line and is simply unmeasurable at this resolution.

    Args:
        qualname: The dotted public name, used only for reporting.
        obj: The callable to locate.

    Returns:
        The located callable, or None when it has no Python body (a C function, a
        dataclass-generated method, or a builtin).
    """
    code = _code_of(obj)
    if code is None or not code.co_filename or not Path(code.co_filename).exists():
        return None
    if _is_declaration_only(obj) or _is_excluded_from_coverage(obj):
        return None
    lines = {ln for _, _, ln in code.co_lines() if ln is not None}
    if not lines:
        return None
    body = frozenset(ln for ln in lines if ln > code.co_firstlineno)
    if not body:
        return None
    return Callable_(qualname, str(Path(code.co_filename).resolve()), body)


def collect() -> list[Callable_]:
    """Every public callable that has a locatable Python body."""
    from public_surface import public_callables

    located: dict[str, Callable_] = {}
    for qual, obj in public_callables():
        if inspect.isclass(obj):
            continue
        found = locate(qual, obj)
        if found is not None:
            located.setdefault(qual, found)
    return sorted(located.values(), key=lambda c: c.qualname)


def measure(data_file: str | Path = ".coverage") -> Report:
    """Split the public callable surface into exercised and unexercised.

    Args:
        data_file: Path to the ``coverage.py`` data file produced by the test run.

    Returns:
        The report, with the qualified names in each bucket.
    """
    from coverage import CoverageData

    data = CoverageData(basename=str(data_file))
    data.read()
    hit: dict[str, set[int]] = {}
    for measured_file in data.measured_files():
        hit[str(Path(measured_file).resolve())] = set(data.lines(measured_file) or ())

    report = Report()
    for call in collect():
        lines = hit.get(call.filename)
        if lines is None:
            report.unlocatable.append(call.qualname)
        elif call.body_lines & lines:
            report.exercised.append(call.qualname)
        else:
            report.unexercised.append(call.qualname)
    return report


def main() -> int:
    """CLI entry point: print the gap list and exit non-zero if any gap remains."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default=".coverage")
    parser.add_argument("--json", dest="json_out", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    report = measure(args.data_file)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "exercised": report.exercised,
                    "unexercised": report.unexercised,
                    "unlocatable": report.unlocatable,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
    if not args.quiet:
        for name in report.unexercised:
            print(f"UNEXERCISED {name}")
        print(
            f"\n{len(report.exercised)}/{report.measured} public callables exercised "
            f"({report.ratio:.1%}); {len(report.unlocatable)} in unmeasured files"
        )
    return 1 if report.unexercised else 0


if __name__ == "__main__":
    raise SystemExit(main())
