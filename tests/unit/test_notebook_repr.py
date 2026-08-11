"""The notebook repr shows types, and treats column names as the data they are.

A `Dataset` is lazy, so `_repr_html_` must describe the plan rather than run it. Two things
follow that were not true before: it should say what *type* each column is (the question a
reader of a schema is asking), and it must escape the names — they come out of a CSV header, a
JSON key, or a database catalog, and a notebook renders this string as markup, so an unescaped
name is a document written by whoever produced the file.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.unit


def test_it_names_the_columns_and_their_types():
    html = bt.from_pydict({"a": [1], "s": ["x"]})._repr_html_()
    assert "<th>a</th>" in html
    assert "int64" in html
    assert "string" in html


def test_a_column_name_from_a_file_cannot_inject_markup():
    evil = bt.from_arrow(pa.table({"<script>alert(1)</script>": [1]}))
    html = evil._repr_html_()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_it_does_not_execute_the_plan():
    """A repr that runs the query is a repr that hangs on an unbounded source."""
    calls: list[int] = []

    def never(batch):
        calls.append(1)
        return batch

    bt.from_pydict({"a": [1]}).map_batches(never, output_columns=["a"])._repr_html_()
    assert calls == []


def test_an_uninferable_type_shows_a_placeholder_rather_than_failing():
    """A repr must never raise, and a UDF stage's output schema is not knowable."""
    opaque = bt.from_pydict({"a": [1]}).map_batches(lambda b: b, output_columns=["x", "y"])
    html = opaque._repr_html_()
    assert "Dataset" in html
    assert "<th>x</th>" in html


def test_the_plain_repr_is_unchanged():
    assert repr(bt.from_pydict({"a": [1]})) == "Dataset(columns=['a'])"


def test_the_stats_table_escapes_its_cells_too():
    """An operator id or a summary line can carry a column name, and a column name comes
    out of a file."""
    from batcher.api.stats import OpStat, RunStats

    stats = RunStats(
        ops=(
            OpStat(
                op_id=0,
                kind="<img src=x onerror=1>",
                rows_in=1,
                rows_out=1,
                elapsed_ms=1.0,
                result_bytes=1,
                spilled=False,
                backend="native",
            ),
        ),
        total_ms=1.0,
        rows=1,
    )
    html = stats._repr_html_()
    assert "<img src=x" not in html
    assert "&lt;img" in html
