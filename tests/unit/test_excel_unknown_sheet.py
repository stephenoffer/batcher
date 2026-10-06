"""An unknown Excel sheet is refused with the workbook's sheet names (AP-431).

`python-calamine` is optional and not installed in CI, so a stand-in module with the two
calls the source makes is installed in its place; no skip lands on the ratchet.
"""

from __future__ import annotations

import sys
import types

import pytest

import batcher as bt
from batcher._internal.errors import FormatError

_SHEETS = {"Revenue": [["a"], [1], [2]], "Costs": [["b"], [3]]}


@pytest.fixture
def workbook(tmp_path, monkeypatch):
    class _Sheet:
        def __init__(self, rows):
            self._rows = rows

        def to_python(self):
            return self._rows

    class _Workbook:
        def __init__(self):
            self.sheet_names = list(_SHEETS)

        def get_sheet_by_name(self, name):
            return _Sheet(_SHEETS[name])

    fake = types.ModuleType("python_calamine")
    fake.load_workbook = lambda fh: _Workbook()
    monkeypatch.setitem(sys.modules, "python_calamine", fake)
    path = tmp_path / "book.xlsx"
    path.write_bytes(b"not really a workbook; the stand-in never reads it")
    return str(path)


def test_a_misspelled_sheet_suggests_the_real_one(workbook):
    with pytest.raises(FormatError, match="Did you mean 'Revenue'") as err:
        bt.read.excel(workbook, sheet="Revenu").to_pydict()
    assert "Costs" in str(err.value)


def test_an_out_of_range_index_lists_the_sheets(workbook):
    with pytest.raises(FormatError, match=r"sheet=5 is out of range.*Revenue"):
        bt.read.excel(workbook, sheet=5).to_pydict()


def test_two_sheets_of_one_workbook_keep_their_own_schemas(workbook):
    # Regression: the file-keyed schema cache served the first sheet's columns to the
    # second sheet read from the same workbook.
    assert bt.read.excel(workbook, sheet="Costs").to_pydict() == {"b": [3]}
    assert bt.read.excel(workbook, sheet=0).to_pydict() == {"a": [1, 2]}
