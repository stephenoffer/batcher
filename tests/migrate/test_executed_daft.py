"""A translated Daft program returns the rows the Daft program returned.

Each executed corpus case runs as the original script in Daft and as the translated script in
Batcher over the same in-memory data; the answers are compared in order where the program sorts.
"""

from __future__ import annotations

import pytest

pytest.importorskip("daft")
pytest.importorskip("libcst")

from _corpus import CORPUS, EXECUTED, equivalent, run


@pytest.mark.parametrize("case", sorted(EXECUTED["daft"]))
def test_translated_program_returns_the_daft_rows(case: str) -> None:
    source, translated = equivalent("daft", case)
    assert source, "a case whose program returns no rows compares nothing"
    assert translated == source


def test_the_comparison_sees_a_rewrite_that_changes_the_answer(tmp_path) -> None:
    # Positive control: Daft's `sort(desc=True)` puts nulls first. Dropping the explicit
    # `nulls_first` the codemod writes reorders the null customer, and must not pass.
    golden = (CORPUS / "daft" / "sort_join" / "batcher.py").read_text()
    naive = golden.replace(", nulls_first=[True, False]", "")
    assert naive != golden
    (tmp_path / "naive.py").write_text(naive)
    assert run(tmp_path / "naive.py") != run(CORPUS / "daft" / "sort_join" / "src.py")
