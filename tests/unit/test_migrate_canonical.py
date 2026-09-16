"""The canonical-name rewrite renames Batcher's second spellings and nothing else.

Both halves matter equally. A missed rename leaves code calling a spelling the alias removal
deletes; a wrong rename changes a pandas or Polars call that merely shares the word, which in a
differential test silently changes what the oracle computes. So every case here pairs the
Batcher expression that must change with a foreign one on the same line that must not.
"""

from __future__ import annotations

import textwrap

import pytest

pytest.importorskip("libcst")

from batcher._internal.migration import load_renames, load_returns
from batcher.migrate import canonicalize


def _run(source: str) -> tuple[str, object]:
    return canonicalize(textwrap.dedent(source), load_renames(), load_returns())


def test_dataset_methods_rename_on_batcher_and_not_on_pandas() -> None:
    out, report = _run(
        """
        import batcher as bt
        import pandas as pd

        ds = bt.from_pydict({"k": [1]})
        pdf = pd.DataFrame({"k": [1]})
        a = ds.groupby("k").size()
        b = pdf.groupby("k").size()
        c = ds.filter(bt.col("k") > 0).fillna(0).head(3)
        d = pdf.fillna(0).head(3)
        """
    )
    assert 'a = ds.group_by("k").len()' in out
    assert 'b = pdf.groupby("k").size()' in out
    assert 'c = ds.filter(bt.col("k") > 0).fill_null(0).limit(3)' in out
    assert "d = pdf.fillna(0).head(3)" in out
    assert {e.old for e in report.renamed} >= {"groupby", "size", "fillna", "head"}


def test_expression_accessors_follow_the_chain() -> None:
    out, _ = _run(
        """
        import batcher as b2
        import polars as pl

        e = b2.col("s").str.to_lowercase().str.len()
        p = pl.col("s").str.to_lowercase().str.len_chars()
        agg = (b2.col("x") + 1).nunique()
        """
    )
    assert 'e = b2.col("s").str.lower().str.len_chars()' in out
    assert 'p = pl.col("s").str.to_lowercase().str.len_chars()' in out
    assert 'agg = (b2.col("x") + 1).count_distinct()' in out


def test_from_imports_and_their_uses_are_renamed_together() -> None:
    out, _ = _run(
        """
        from batcher import from_dict, col

        ds = from_dict({"x": [1]})
        """
    )
    assert "from batcher import from_pydict, col" in out
    assert 'ds = from_pydict({"x": [1]})' in out


def test_a_parameter_typed_by_annotation_or_fixture_is_followed() -> None:
    out, _ = _run(
        """
        import batcher as bt
        import pytest

        @pytest.fixture
        def ds():
            return bt.from_pydict({"x": [1]})

        def test_it(ds):
            return ds.to_dicts()

        def helper(frame: bt.Dataset):
            return frame.vstack(frame)
        """
    )
    assert "return ds.to_pylist()" in out
    assert "return frame.union(frame)" in out


def test_an_unknown_receiver_is_reported_not_rewritten() -> None:
    out, report = _run(
        """
        import batcher as bt

        def f(thing, words):
            words.append("x")
            return thing.groupby("k")
        """
    )
    assert 'thing.groupby("k")' in out
    # `groupby` needs a person; `list.append` is Python's own and is not reported.
    assert [e.old for e in report.unresolved] == ["groupby"]
    assert not report.renamed


def test_a_file_that_never_imports_batcher_reports_nothing() -> None:
    _, report = _run(
        """
        def f(frame):
            return frame.groupby("k")
        """
    )
    assert not report.unresolved


def test_a_name_bound_to_two_receivers_is_unknown() -> None:
    out, _ = _run(
        """
        import batcher as bt
        import pandas as pd

        x = bt.from_pydict({"k": [1]})
        x = pd.DataFrame({"k": [1]})
        y = x.groupby("k")
        """
    )
    assert 'y = x.groupby("k")' in out


def test_calling_the_read_namespace_yields_a_dataset() -> None:
    out, _ = _run(
        """
        import batcher as bt

        a = bt.read("t.parquet").groupby("k")
        b = bt.read.csv("t.csv").unique()
        """
    )
    assert 'a = bt.read("t.parquet").group_by("k")' in out
    assert 'b = bt.read.csv("t.csv").distinct()' in out


def test_a_relative_import_does_not_break_import_seeding() -> None:
    out, _ = _run(
        """
        from . import sibling
        import batcher as bt

        ds = bt.from_pydict({"x": [1]}).unique()
        """
    )
    assert 'ds = bt.from_pydict({"x": [1]}).distinct()' in out


def test_a_foreign_class_annotation_is_not_taken_for_a_batcher_one() -> None:
    out, _ = _run(
        """
        import batcher as bt
        import polars as pl

        def f(e: pl.Expr, d: "pl.DataFrame"):
            return e.str.to_lowercase(), d.head(2)

        def g(e: bt.Expr):
            return e.str.to_lowercase()
        """
    )
    assert "return e.str.to_lowercase(), d.head(2)" in out
    assert "    return e.str.lower()" in out


def test_self_inside_a_batcher_class_and_type_checking_imports() -> None:
    out, _ = _run(
        """
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            from batcher.plan.expr_ir.core import Expr

        class Dataset:
            def preview(self):
                return self.head(3)

        class Other:
            def preview(self):
                return self.head(3)

        def typed(e: Expr):
            return e.isna()
        """
    )
    assert out.count("return self.limit(3)") == 1
    assert out.count("return self.head(3)") == 1
    assert "return e.is_null()" in out
