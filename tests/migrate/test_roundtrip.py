"""X -> Batcher -> X is the identity on every registry row the codemod treats as reversible.

For each `canonical` or `alias` row with one Batcher spelling (and each reversible template),
a one-call program is written in the foreign engine's own terms, translated onto Batcher and
exported back. When the export is exact, the call must come back spelled exactly as it was
written, arguments included. A row whose Batcher spelling several foreign names share comes back
as the preferred one of them, which is still a spelling of the same Batcher call; that case is
checked for meaning rather than text.
"""

from __future__ import annotations

import ast
import textwrap

import pytest

pytest.importorskip("libcst")

from batcher._internal.migration import Status
from batcher.migrate.engines import SPECS, tables
from batcher.migrate.outbound import _inverse, export
from batcher.migrate.templates import Signature
from batcher.migrate.translate import translate

_ENGINES = ("pyspark", "polars", "daft", "ray_data")
_IMPORT = {
    "pyspark": "from pyspark.sql import Column, DataFrame, GroupedData\n"
    "from pyspark.sql import functions as F",
    "polars": "import polars as pl",
    "daft": "import daft\nimport daft.functions",
    "ray_data": "import ray.data\nimport ray.data.expressions",
}
_ANNOTATION = {
    "pyspark": {"DataFrame": "DataFrame", "Column": "Column", "GroupedData": "GroupedData"},
    "polars": {s: f"pl.{s}" for s in ("DataFrame", "LazyFrame", "Expr")},
    "daft": {"DataFrame": "daft.DataFrame", "Expression": "daft.Expression"},
    "ray_data": {"Dataset": "ray.data.Dataset"},
}
_BATCHER_ANNOTATION = {"Dataset": "bt.Dataset", "Expr": "bt.Expr"}


def _call(engine: str, surface: str, name: str) -> tuple[str, str, str] | None:
    """(parameter annotation, the call, the Batcher annotation) for one row, if writable."""
    spec = SPECS[engine]
    tokens = tables(engine).params.get(surface, {}).get(name)
    if tokens is None or tokens == ["@property"]:
        return None
    signature = Signature.parse(tokens)
    required = [n for n, default in signature.positional if not default]
    required_kw = [n for n, default in signature.keyword if not default]
    if required_kw:
        return None
    args = [f"a{i}" for i in range(len(required))]
    if surface in spec.modules:
        target = tables(engine).row(surface, name).batcher[0]  # type: ignore[union-attr]
        if target.startswith("Expr.") and args:
            args[0] = "x"
            annotation = _ANNOTATION[engine].get(sorted(spec.expressions)[0])
            return annotation, f"{spec.modules[surface]}.{name}({', '.join(args)})", "bt.Expr"
        return "object", f"{spec.modules[surface]}.{name}({', '.join(args)})", "object"
    annotation = _ANNOTATION[engine].get(surface)
    receiver = tables(engine).batcher_receiver(surface)
    if annotation is None or receiver not in _BATCHER_ANNOTATION:
        return None
    return annotation, f"x.{name}({', '.join(args)})", _BATCHER_ANNOTATION[receiver]


def _returned(source: str) -> str:
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
    ret = fn.body[-1]
    assert isinstance(ret, ast.Return) and ret.value is not None
    return ast.unparse(ret.value)


def _program(imports: str, annotation: str, call: str) -> str:
    return f"{imports}\n\n\ndef f(x: {annotation}):\n    return {call}\n"


def _reversible(engine: str) -> list[tuple[str, str, bool]]:
    """Every writable reversible row, with whether it is its Batcher spelling's only candidate."""
    index = _inverse(engine)
    owners: dict[tuple[str, str], int] = {}
    for candidates in index.values():
        for c in candidates:
            owners[(c.row.surface, c.row.name)] = len(candidates)
    out = []
    for (surface, name), count in sorted(owners.items()):
        row = tables(engine).row(surface, name)
        if row is None or (row.status is not Status.CANONICAL and not row.template):
            continue
        out.append((surface, name, count == 1))
    return out


@pytest.mark.parametrize("engine", _ENGINES)
def test_reversible_rows_round_trip(engine: str) -> None:
    exact = same_meaning = 0
    for surface, name, unique in _reversible(engine):
        written = _call(engine, surface, name)
        if written is None:
            continue
        annotation, call, batcher_annotation = written
        forward, report = translate(_program(_IMPORT[engine], annotation, call), engine)
        if [s.action for s in report.sites if s.spelling == f"{surface}.{name}"] != ["rewritten"]:
            continue  # the call does not bind 1:1 in this direction, so there is nothing to invert
        batcher_call = _returned(forward)
        back, exported = export(
            _program("import batcher as bt", batcher_annotation, batcher_call), engine
        )
        returned = _returned(back)
        if unique:
            assert returned == ast.unparse(ast.parse(call)), (
                f"{surface}.{name}: {call} -> {returned}"
            )
            exact += 1
        elif any(site.action == "rewritten" for site in exported.sites):
            # Shared by several foreign names: whatever comes back must mean the same Batcher call.
            again, _ = translate(_program(_IMPORT[engine], annotation, returned), engine)
            assert _returned(again) == batcher_call, f"{surface}.{name}: {call} -> {returned}"
            same_meaning += 1
    # Positive control: a broken inverse index or an engine table that no longer types the
    # receivers would make every row `continue` above and the test pass on nothing.
    assert exact >= 10, f"{engine}: only {exact} rows round-tripped exactly"
    assert exact + same_meaning >= 15


def test_the_round_trip_program_is_what_the_rules_see() -> None:
    # The generated one-call programs must type their receiver, or the test above is vacuous.
    source = _program(_IMPORT["polars"], "pl.DataFrame", "x.filter(a0)")
    out, report = translate(source, "polars")
    assert _returned(out) == "x.filter(a0)"
    assert [s.action for s in report.sites] == ["rewritten"]
    back, _ = export(
        textwrap.dedent(_program("import batcher as bt", "bt.Dataset", "x.filter(a0)")), "polars"
    )
    assert _returned(back) == "x.filter(a0)"
