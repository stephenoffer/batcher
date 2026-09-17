"""A translated Polars program returns the rows the Polars program returned.

A golden only says the codemod wrote what a reviewer approved; it cannot say the approved code
means the same thing. So each executed corpus case runs twice over the same in-memory data, as
the original script in Polars and as the translated script in Batcher, and the two answers are
compared, in order where the program sorts.
"""

from __future__ import annotations

import pytest

pytest.importorskip("polars")
pytest.importorskip("libcst")

from _corpus import CORPUS, EXECUTED, equivalent, run

import batcher as bt  # noqa: F401 - the translated programs import it; fail here, not in exec


@pytest.mark.parametrize("case", sorted(EXECUTED["polars"]))
def test_translated_program_returns_the_polars_rows(case: str) -> None:
    source, translated = equivalent("polars", case)
    assert source, "a case whose program returns no rows compares nothing"
    assert translated == source


def test_the_comparison_sees_a_rewrite_that_changes_the_answer(tmp_path) -> None:
    # Positive control: the naive 1:1 rewrite of an ordered Polars window (dropping the
    # `frame=(None, None)` the codemod adds) is a running sum in Batcher, and must not pass.
    golden = (CORPUS / "polars" / "windows" / "batcher.py").read_text()
    naive = golden.replace(", frame=(None, None)", "")
    assert naive != golden
    (tmp_path / "naive.py").write_text(naive)
    source = run(CORPUS / "polars" / "windows" / "src.py")
    assert run(tmp_path / "naive.py") != source
