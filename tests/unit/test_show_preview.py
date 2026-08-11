"""`show()` prints rows, and prints them in a form a person can read.

It used to print pyarrow's `Table` repr, which is column-oriented: reading one row means
scanning several lines and counting positions, and it degrades as columns are added. Every
neighbouring tool prints rows. These tests pin the shape of the replacement and, more
importantly, the cases that make a naive table renderer useless — a value long enough to set
the width of the screen, a table wider than the screen, and the column types this engine now
produces routinely whose raw form is a wall of bytes.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher.api.terminal.preview import render

pytestmark = pytest.mark.unit


def _rendered(mapping, *, limit: int = 10) -> str:
    return render(pa.table(mapping), limit=limit)


def test_it_prints_a_row_per_row():
    text = _rendered({"a": [1, 2], "s": ["x", "y"]})
    assert "| 1     | x " in text
    assert "| 2     | y " in text


def test_the_header_carries_the_types():
    """The type is the thing a reader most often opens a preview to check."""
    text = _rendered({"a": [1], "s": ["x"]})
    assert "int64" in text
    assert "string" in text


def test_a_null_reads_as_null_not_as_none():
    assert "null" in _rendered({"a": [None, 1]})
    assert "None" not in _rendered({"a": [None, 1]})


def test_booleans_read_as_they_do_in_sql():
    text = _rendered({"b": [True, False]})
    assert "true" in text and "false" in text


def test_a_long_value_does_not_set_the_width_of_the_screen():
    text = _rendered({"t": ["x" * 500, "short"]})
    assert max(len(line) for line in text.splitlines()) < 60
    assert "..." in text


def test_a_wide_table_is_truncated_and_says_so():
    text = _rendered({f"c{i}": [i] for i in range(40)})
    assert max(len(line) for line in text.splitlines()) <= 130
    assert "not shown" in text


def test_one_column_is_kept_however_wide_it_is():
    """Dropping every column would print an empty frame, which answers nothing."""
    text = _rendered({"t": ["y" * 200]})
    assert "| t" in text


def test_an_empty_result_still_shows_its_schema():
    text = _rendered({"a": pa.array([], type=pa.int64())})
    assert "int64" in text
    assert "0 rows" in text


def test_a_result_with_no_columns_says_so():
    assert "no columns" in render(pa.table({}), limit=10)


def test_the_footer_pluralizes():
    assert "[1 row x 1 column]" in _rendered({"a": [1]})
    assert "[2 rows x 1 column]" in _rendered({"a": [1, 2]})


def test_the_footer_says_first_only_when_the_limit_was_reached():
    """Saying "first 3 rows" about a complete result sends the reader looking for more."""
    assert "first" not in _rendered({"a": [1, 2, 3]}, limit=10)
    assert "first 3 rows" in _rendered({"a": [1, 2, 3]}, limit=3)


def test_a_binary_column_is_summarized_not_dumped():
    text = _rendered({"b": [b"\x00" * 64]})
    assert "<64 bytes>" in text


def test_a_variable_shape_tensor_reads_as_its_shape_and_dtype():
    """The raw form is a buffer, a shape list, and a dtype code — a wall of nothing."""
    from batcher.io.formats.ml.ragged import to_ragged_tensor_column

    column = to_ragged_tensor_column([np.zeros((2, 2), "uint8"), np.ones((3, 4), "float32")])
    text = render(pa.table({"img": column}), limit=10)
    assert "<uint8 2x2>" in text
    assert "<float32 3x4>" in text
    assert "|u1" not in text  # the stored dtype code, which nobody wants to read


def test_a_struct_type_is_named_by_its_family_not_its_fields():
    text = _rendered({"s": [{"a": 1, "b": "long field name here"}]})
    assert "| struct" in text


def test_show_goes_through_it(capsys):
    bt.from_pydict({"a": [1, 2]}).show()
    printed = capsys.readouterr().out
    assert "pyarrow.Table" not in printed
    assert "[2 rows x 1 column]" in printed


def test_show_respects_its_limit(capsys):
    bt.from_pydict({"a": list(range(100))}).show(3)
    printed = capsys.readouterr().out
    assert "first 3 rows" in printed
    assert "| 3 " not in printed
