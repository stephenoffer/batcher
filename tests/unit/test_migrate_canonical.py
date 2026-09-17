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

from batcher._internal.migration import load_kwarg_renames, load_renames, load_returns
from batcher.migrate import canonicalize


def _run(source: str) -> tuple[str, object]:
    return canonicalize(
        textwrap.dedent(source), load_renames(), load_returns(), load_kwarg_renames()
    )


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
    assert "a = ds.group_by(\"k\").len(name='size')" in out
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


def test_paths_calls_and_top_level_reader_imports() -> None:
    out, _ = _run(
        """
        import batcher as bt
        from batcher import read_csv, col

        ds = bt.from_pydict({"x": [1]})
        ds.to_parquet("out")
        n = ds.height
        e = ds.empty
        a = bt.read_parquet("p")
        b = read_csv("c")
        """
    )
    assert 'ds.write.parquet("out")' in out
    assert "n = ds.count()" in out
    assert "e = ds.is_empty()" in out
    assert 'a = bt.read.parquet("p")' in out
    assert "from batcher import read, col" in out
    assert 'b = read.csv("c")' in out


def test_operator_methods_become_operators() -> None:
    out, report = _run(
        """
        import batcher as bt

        x = bt.col("a").add(bt.col("b") * 2)
        y = bt.col("a").ge(3)
        z = bt.col("f").not_()
        w = bt.col("a").add(1, fill=0)
        """
    )
    assert 'x = (bt.col("a") + (bt.col("b") * 2))' in out
    assert 'y = (bt.col("a") >= 3)' in out
    assert 'z = (~bt.col("f"))' in out
    assert 'w = bt.col("a").add(1, fill=0)' in out
    assert [e.old for e in report.unresolved] == ["add"]


def test_argument_reshaping_transforms() -> None:
    out, report = _run(
        """
        import batcher as bt

        ds = bt.from_pydict({"x": [1]})
        a = ds.with_column("y", bt.col("x") + 1)
        b = ds.with_column("has space", bt.col("x"))
        c = ds.slice(10, 5)
        d = ds.slice(10)
        """
    )
    assert 'a = ds.with_columns(y=bt.col("x") + 1)' in out
    assert 'b = ds.with_columns(**{"has space": bt.col("x")})' in out
    assert "c = ds.limit(5, offset=10)" in out
    assert "d = ds.slice(10)" in out
    assert "slice" in {e.old for e in report.unresolved}


def test_pandas_keywords_are_rewritten_or_reported() -> None:
    out, report = _run(
        """
        import batcher as bt

        ds = bt.from_pydict({"x": [1]})
        a = ds.sort(by=["x", "y"], ascending=[True, False], na_position="first")
        b = ds.sort_values("x", ascending=False)
        c = ds.sample(frac=0.5, random_state=7)
        d = ds.melt(id_vars=["k"], value_vars=["v"], var_name="n")
        e = ds.nlargest(3, "x")
        f = ds.sort(by=cols, ascending=flag)
        """
    )
    assert 'a = ds.sort("x", "y", descending=[False, True], nulls_first=True)' in out
    assert 'b = ds.sort("x", descending=True)' in out
    assert "c = ds.sample(fraction=0.5, seed=7)" in out
    assert 'd = ds.unpivot(index=["k"], on=["v"], variable_name="n")' in out
    assert 'e = ds.top_k(3, "x")' in out
    assert "f = ds.sort(by=cols, ascending=flag)" in out
    assert {e.old for e in report.unresolved} == {"sort(by=)", "sort(ascending=)"}


def test_a_multi_line_call_keeps_its_layout() -> None:
    out, _ = _run(
        """
        import batcher as bt

        ds = bt.from_pydict({"x": [1]})
        a = ds.sample(
            frac=0.5,
            random_state=7,
        )
        """
    )
    assert "a = ds.sample(\n    fraction=0.5,\n    seed=7,\n)" in out


def test_doctest_examples_and_markdown_blocks_are_rewritten() -> None:
    from batcher.migrate.snippets import rewrite_markdown, rewrite_python

    tables = (load_renames(), load_returns(), load_kwarg_renames())
    module = textwrap.dedent(
        '''
        def f():
            """Example.

            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"k": [1]})
                >>> ds.groupby("k").size().to_dicts()
                [{'k': 1}]
                >>> pdf.groupby("k")  # doctest: +SKIP
            """
            return 1
        '''
    )
    out, report = rewrite_python(module, *tables)
    assert ">>> ds.group_by(\"k\").len(name='size').to_pylist()" in out
    assert '>>> pdf.groupby("k")  # doctest: +SKIP' in out
    assert "[{'k': 1}]" in out
    assert {e.old for e in report.renamed} >= {"groupby", "size", "to_dicts"}

    page = textwrap.dedent(
        """
        Intro.

        ```python
        import batcher as bt
        ds = bt.from_pydict({"x": [1]})
        ```

        Later, the same session:

        ```python
        print(ds.head(1).to_dict())
        ```
        """
    )
    md, md_report = rewrite_markdown(page, *tables)
    assert "print(ds.limit(1).to_pydict())" in md
    assert "Later, the same session:" in md
    assert all(e.line >= 10 for e in md_report.renamed)


def test_a_differing_default_is_passed_explicitly() -> None:
    out, _ = _run(
        """
        import batcher as bt

        ds = bt.from_pydict({"k": [1], "s": ["a1"]})
        a = ds.head()
        b = ds.head(3)
        c = ds.group_by("k").size()
        d = bt.col("s").str.regexp_extract("([a-z])([0-9])")
        e = bt.col("s").str.regexp_extract("([a-z])([0-9])", 2)
        """
    )
    assert "a = ds.limit(n=5)" in out
    assert "b = ds.limit(3)" in out
    assert "c = ds.group_by(\"k\").len(name='size')" in out
    assert 'd = bt.col("s").str.extract("([a-z])([0-9])", group=0)' in out
    assert 'e = bt.col("s").str.extract("([a-z])([0-9])", 2)' in out


def test_a_keyword_conflict_or_an_unknown_value_is_reported_not_merged() -> None:
    out, report = _run(
        """
        import batcher as bt

        ds = bt.from_pydict({"x": [1]})
        a = ds.sort("x", descending=True, ascending=False)
        b = ds.sort("x", na_position="middle")
        """
    )
    assert 'a = ds.sort("x", descending=True, ascending=False)' in out
    assert 'b = ds.sort("x", na_position="middle")' in out
    assert {e.old for e in report.unresolved} == {"sort(ascending=)", "sort(na_position=)"}


def test_positional_arguments_move_to_the_kept_keywords() -> None:
    out, _ = _run(
        """
        import batcher as bt

        a = bt.col("x").clip_max(2.0)
        b = bt.col("x").clip_min(bt.col("lo"))
        """
    )
    assert 'a = bt.col("x").clip(upper=2.0)' in out
    assert 'b = bt.col("x").clip(lower=bt.col("lo"))' in out


def test_engine_internal_imports_node_classes_and_typed_helpers() -> None:
    out, _ = _run(
        """
        from typing import TYPE_CHECKING

        from batcher.plan.expr_ir.constructors import col
        from batcher.plan.expr_ir.core import Col

        if TYPE_CHECKING:
            from batcher.plan.expr_ir.core import Expr

        def _text(value) -> Expr:
            return value

        def f(column, text):
            a = Col(column).n_unique()
            b = col(column).str.to_lowercase()
            c = _text(text).str.regexp_extract("x", 1)
            return a, b, c
        """
    )
    assert "a = Col(column).count_distinct()" in out
    assert "b = col(column).str.lower()" in out
    assert 'c = _text(text).str.extract("x", 1)' in out


def test_assumed_accessors_only_when_asked() -> None:
    source = textwrap.dedent(
        """
        import batcher as bt

        def f(pred, gold):
            return pred.list.set_intersection(gold)
        """
    )
    tables = (load_renames(), load_returns(), load_kwarg_renames())
    plain, _ = canonicalize(source, *tables)
    assumed, _ = canonicalize(source, *tables, assume_accessors=True)
    assert "pred.list.set_intersection(gold)" in plain
    assert "pred.list.intersect(gold)" in assumed


def test_a_same_named_function_from_a_submodule_is_never_renamed() -> None:
    out, _ = _run(
        """
        from batcher.dist.shuffle_io import read_ipc
        from batcher.ml.preprocessors.persistence import from_dict

        table = read_ipc("path")
        model = from_dict({})
        """
    )
    assert 'table = read_ipc("path")' in out
    assert "model = from_dict({})" in out


def test_a_function_local_import_types_names_in_that_function() -> None:
    out, _ = _run(
        """
        def test_it(ds):
            from batcher.plan.expr_ir import col

            return col("x").skewness()
        """
    )
    assert 'return col("x").skew()' in out


def test_hooks_on_a_listener_subclass_are_renamed() -> None:
    out, _ = _run(
        """
        import batcher as bt

        class Watcher(bt.StreamingQueryListener):
            def onQueryProgress(self, event):
                return event

        class Unrelated:
            def onQueryProgress(self, event):
                return event
        """
    )
    assert out.count("def on_query_progress(self, event):") == 1
    assert out.count("def onQueryProgress(self, event):") == 1


def test_identity_calls_collapse_to_the_dataset() -> None:
    out, _ = _run(
        """
        import batcher as bt

        ds = bt.from_pydict({"x": [1]})
        a = ds.lazy().filter(bt.col("x") > 0)
        b = ds.copy()
        """
    )
    assert 'a = ds.filter(bt.col("x") > 0)' in out
    assert "b = ds" in out


def test_a_helper_imported_from_the_project_is_followed(tmp_path) -> None:
    from batcher.migrate.project import imported_function_receivers

    helpers = tmp_path / "_common"
    helpers.mkdir()
    (helpers / "__init__.py").write_text("from _common.datasets import tpch\n")
    (helpers / "datasets.py").write_text(
        "import batcher as bt\n\n\ndef tpch(table: str) -> bt.Dataset:\n"
        "    return bt.read.parquet(table)\n"
    )
    script = tmp_path / "relational" / "set_ops.py"
    script.parent.mkdir()
    source = "from _common import tpch\n\nrows = tpch('orders').head(10)\n"
    imported = imported_function_receivers(source, script, load_returns())
    assert imported == {"tpch": "Dataset"}
    out, _ = canonicalize(
        source, load_renames(), load_returns(), load_kwarg_renames(), imported=imported
    )
    assert "rows = tpch('orders').limit(10)" in out
