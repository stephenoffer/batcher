"""A file's cached schema must not be served to a read that parses it differently.

`FileSource._file_schema` caches an inferred schema so a repeated query does not re-open the file.
It was keyed on the reader class and the file's identity, which is enough for a self-describing
format and not for CSV: the same file read with `delimiter="\\t"` and then with `delimiter=","` is
two schemas, and the second read was handed the first one — three columns where its own delimiter
yields one, with no error. `examples/io/csv_roundtrip_and_options.py` failed on exactly this, and
the failure read as "the wrong delimiter did not collapse the columns" rather than as a cache bug.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.csv as pacsv
import pytest

import batcher as bt

pytestmark = pytest.mark.unit


@pytest.fixture
def tab_file(tmp_path) -> str:
    """Three tab-separated columns and no comma anywhere, so a comma parse is one column."""
    path = str(tmp_path / "x.tsv")
    table = pa.table({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    pacsv.write_csv(table, path, pacsv.WriteOptions(delimiter="\t"))
    return path


def test_a_second_read_with_another_delimiter_gets_its_own_schema(tab_file: str) -> None:
    assert bt.read.csv(tab_file, delimiter="\t").width == 3  # warms the cache
    assert bt.read.csv(tab_file, delimiter=",").width == 1
    assert bt.read.csv(tab_file, delimiter="\t").width == 3  # and the tab read still hits


def test_the_order_of_the_reads_does_not_change_either_answer(tab_file: str) -> None:
    """The control: each answer is what a *fresh* read gives, so the cache changes no result."""
    assert bt.read.csv(tab_file, delimiter=",").width == 1
    assert bt.read.csv(tab_file, delimiter="\t").width == 3
