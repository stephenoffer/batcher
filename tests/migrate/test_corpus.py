"""The golden corpus: every engine's scripts translate onto Batcher, and back, exactly as reviewed.

Each case under `corpus/<engine>/<case>/` pins both directions (`_corpus.py` says how the files
relate and how to regenerate them). The goldens are the codemod's behaviour stated as code a
reviewer can read: a rename that stops happening, a marker that stops appearing, or a rewrite of
a row the registry says differs all show up here as a diff.
"""

from __future__ import annotations

import ast

import pytest

pytest.importorskip("libcst")

from _corpus import CORPUS, ENGINES, EXECUTED, cases

from batcher.migrate.outbound import export
from batcher.migrate.translate import translate

_ALL = [(engine, case.name) for engine in ENGINES for case in cases(engine)]
_FOREIGN_ROOTS = {"pyspark": "pyspark", "polars": "polars", "daft": "daft", "ray_data": "ray"}


def test_every_engine_has_at_least_eight_cases() -> None:
    # Positive control for the parametrized tests below: a moved corpus would collect nothing.
    for engine in ENGINES:
        assert len(cases(engine)) >= 8, f"{engine} has {len(cases(engine))} corpus cases"


@pytest.mark.parametrize(("engine", "case"), _ALL)
def test_foreign_script_translates_to_the_golden(engine: str, case: str) -> None:
    folder = CORPUS / engine / case
    out, report = translate((folder / "src.py").read_text(), engine)
    assert out == (folder / "batcher.py").read_text()
    assert report.sites, "a corpus case that touches nothing tests nothing"


@pytest.mark.parametrize(("engine", "case"), _ALL)
def test_batcher_script_exports_to_the_golden(engine: str, case: str) -> None:
    folder = CORPUS / engine / case
    out, _ = export((folder / "batcher.py").read_text(), engine)
    assert out == (folder / "back.py").read_text()


@pytest.mark.parametrize(("engine", "case"), [(e, c) for e in EXECUTED for c in EXECUTED[e]])
def test_an_executed_case_leaves_nothing_foreign_behind(engine: str, case: str) -> None:
    # The executed-equivalence tests compare the two programs' answers; that comparison only
    # proves the translation when the translated program no longer touches the source engine.
    tree = ast.parse((CORPUS / engine / case / "batcher.py").read_text())
    imported = {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    } | {
        a.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for a in node.names
    }
    assert _FOREIGN_ROOTS[engine] not in imported
    assert "batcher" in imported


@pytest.mark.parametrize(
    ("engine", "case", "residue"),
    [
        ("pyspark", "mismatches", "functions.concat` differs in Batcher"),
        ("pyspark", "udfs", "functions.udf` differs in Batcher"),
        ("pyspark", "dates", "needs a manual rewrite"),
        ("polars", "mismatches", "Expr.n_unique` differs in Batcher"),
        ("polars", "udfs", "Expr.map_elements` has no Batcher equivalent yet"),
        ("daft", "udfs", "daft.func` has no Batcher equivalent yet"),
        ("ray_data", "mismatches", "Dataset.random_shuffle` differs in Batcher"),
    ],
)
def test_a_mismatch_is_marked_and_left_as_written(engine: str, case: str, residue: str) -> None:
    folder = CORPUS / engine / case
    golden = (folder / "batcher.py").read_text()
    marker = next(line for line in golden.splitlines() if residue in line)
    assert marker.lstrip().startswith("# batcher-migrate:")
    _, report = translate((folder / "src.py").read_text(), engine)
    assert any(s.action == "marked" for s in report.sites)


def test_the_corpus_rewrites_the_high_value_mappings() -> None:
    # A spot check that the goldens hold the semantics-restoring forms, not just renames.
    spark = {c: (CORPUS / "pyspark" / c / "batcher.py").read_text() for c in ("io", "windows")}
    assert 'events.write.parquet("s3://bucket/copy/", mode="error")' in spark["io"]
    assert 'partition_by=["g"], order_by=["t"], frame=(None, 0)' in spark["windows"]
    polars = (CORPUS / "polars" / "expressions" / "batcher.py").read_text()
    # Polars `contains` is a regex unless told otherwise; Batcher's is literal by default.
    assert '.str.contains("an+", literal=False)' in polars
    assert '.str.contains("rr", literal=True)' in polars
    assert ".str.slice(1, 3)" in polars  # Batcher's `str.slice` is 0-based, as Polars' is
    ray = (CORPUS / "ray_data" / "map_batches" / "batcher.py").read_text()
    assert 'map_batches(add_total, batch_format="numpy")' in ray
