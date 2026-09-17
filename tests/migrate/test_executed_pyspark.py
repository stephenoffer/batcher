"""A translated PySpark program returns the rows the PySpark program returned.

PySpark needs a JVM, which neither CI nor the development head node has, so on those machines
this module is skipped and the PySpark direction is covered by its goldens alone
(`test_corpus.py`). The skip is a `skipif` on `java` rather than an `importorskip`, so
`tools/lint_skips.py`, which reads module-level `importorskip` calls, counts only the `pyspark`
import here and not the missing JVM: the gap is stated in this docstring instead. It runs
unchanged in a lane with a JDK installed (the plan's `parity-spark` lane).
"""

from __future__ import annotations

import shutil

import pytest

pytest.importorskip("pyspark")
pytest.importorskip("libcst")

from _corpus import EXECUTED, equivalent

pytestmark = pytest.mark.skipif(shutil.which("java") is None, reason="PySpark needs a JVM")


@pytest.mark.parametrize("case", sorted(EXECUTED["pyspark"]))
def test_translated_program_returns_the_pyspark_rows(case: str) -> None:
    source, translated = equivalent("pyspark", case)
    assert source, "a case whose program returns no rows compares nothing"
    assert translated == source
